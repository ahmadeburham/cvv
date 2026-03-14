
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

try:
    import pytesseract
except Exception as exc:
    pytesseract = None
    _PYTESS_ERROR = exc
else:
    _PYTESS_ERROR = None

try:
    from paddleocr import PaddleOCR
except Exception as exc:
    PaddleOCR = None
    _PADDLE_ERROR = exc
else:
    _PADDLE_ERROR = None

try:
    import face_recognition  # type: ignore
except Exception:
    face_recognition = None

ARABIC_TO_ASCII_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ROI:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TemplateConfig:
    full_name: ROI
    full_address: ROI
    id_number: ROI
    birthday: ROI
    face_photo: ROI
    validation_patches: List[ROI]
    validation_ssim_threshold: float = 0.54
    alignment_min_inliers: int = 24
    vertical_refine_ratio: float = 0.035
    horizontal_refine_ratio: float = 0.02


def get_default_template_config() -> TemplateConfig:
    return TemplateConfig(
        full_name=ROI(0.58, 0.23, 0.95, 0.47),
        full_address=ROI(0.53, 0.46, 0.95, 0.68),
        id_number=ROI(0.39, 0.69, 0.95, 0.83),
        birthday=ROI(0.03, 0.67, 0.31, 0.82),
        face_photo=ROI(0.02, 0.11, 0.35, 0.62),
        validation_patches=[
            ROI(0.40, 0.17, 0.52, 0.29),
            ROI(0.32, 0.53, 0.46, 0.66),
            ROI(0.07, 0.84, 0.22, 0.95),
        ],
        validation_ssim_threshold=0.54,
        alignment_min_inliers=24,
        vertical_refine_ratio=0.035,
        horizontal_refine_ratio=0.02,
    )


def safe_stdio() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_img(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def read_image_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def crop_normalized_roi(img: np.ndarray, roi: ROI) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(round(roi.x1 * w))))
    y1 = max(0, min(h - 1, int(round(roi.y1 * h))))
    x2 = max(1, min(w, int(round(roi.x2 * w))))
    y2 = max(1, min(h, int(round(roi.y2 * h))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: {roi}")
    return img[y1:y2, x1:x2]


def expand_roi(roi: ROI, vx: float = 0.0, vy: float = 0.0) -> ROI:
    return ROI(
        x1=max(0.0, roi.x1 - vx),
        y1=max(0.0, roi.y1 - vy),
        x2=min(1.0, roi.x2 + vx),
        y2=min(1.0, roi.y2 + vy),
    )


def draw_roi(img: np.ndarray, roi: ROI, label: str, color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    x1, y1 = int(round(roi.x1 * w)), int(round(roi.y1 * h))
    x2, y2 = int(round(roi.x2 * w)), int(round(roi.y2 * h))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def draw_polygon(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    pts = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
    return out


def normalize_digits(text: str) -> str:
    return (text or "").translate(ARABIC_TO_ASCII_DIGITS)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or " ").strip()
    return text.strip(" -_.,;:|/")


def clean_id_number(text: str) -> str:
    text = normalize_digits(text)
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) >= 14:
        strong = re.findall(r"[23]\d{13}", digits)
        if strong:
            return strong[0]
        return digits[:14]
    return digits


def infer_birthday_from_id(id_number: str) -> str:
    if len(id_number) != 14 or id_number[0] not in {"2", "3"}:
        return ""
    century = 1900 if id_number[0] == "2" else 2000
    yy = int(id_number[1:3])
    mm = int(id_number[3:5])
    dd = int(id_number[5:7])
    yyyy = century + yy
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return ""
    return f"{yyyy:04d}/{mm:02d}/{dd:02d}"


def clean_birthday(text: str, id_number: str) -> str:
    norm = normalize_digits(text).replace("\\", "/").replace("-", "/").replace(".", "/")
    norm = re.sub(r"[^0-9/]", "", norm)
    norm = re.sub(r"/+", "/", norm).strip("/")
    m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", norm)
    if m:
        y, mo, dd = map(int, m.groups())
        if 1900 <= y <= 2099 and 1 <= mo <= 12 and 1 <= dd <= 31:
            return f"{y:04d}/{mo:02d}/{dd:02d}"
    return infer_birthday_from_id(id_number) or norm


def enhance_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    gray = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    return gray


def generate_ocr_variants(img_bgr: np.ndarray, field_type: str) -> List[Tuple[str, np.ndarray]]:
    gray = enhance_gray(img_bgr)
    resized = cv2.resize(gray, None, fx=2.7, fy=2.7, interpolation=cv2.INTER_CUBIC)
    variants: List[Tuple[str, np.ndarray]] = []
    variants.append(("gray", resized))
    variants.append(("otsu", cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
    variants.append(("adaptive", cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)))
    blur = cv2.GaussianBlur(resized, (0, 0), 2.0)
    variants.append(("unsharp", cv2.addWeighted(resized, 1.8, blur, -0.8, 0)))
    variants.append(("inv_otsu", cv2.bitwise_not(variants[1][1])))
    if field_type in {"id_number", "birthday"}:
        variants.append(("digits_dilate", cv2.dilate(variants[1][1], np.ones((2, 2), np.uint8), iterations=1)))
    return variants


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _compute_homography(detector_name: str, template_bgr: np.ndarray, scene_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    tpl_g = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    scn_g = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    if detector_name == "sift" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=5000)
        kp1, des1 = detector.detectAndCompute(tpl_g, None)
        kp2, des2 = detector.detectAndCompute(scn_g, None)
        if des1 is None or des2 is None:
            return None
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        ratio = 0.74
    else:
        detector = cv2.ORB_create(7000)
        kp1, des1 = detector.detectAndCompute(tpl_g, None)
        kp2, des2 = detector.detectAndCompute(scn_g, None)
        if des1 is None or des2 is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio = 0.78
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    min_matches = 18 if detector_name == "sift" else 24
    if len(good) < min_matches:
        return None
    good = sorted(good, key=lambda m: m.distance)[:350]
    pts_tpl = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_scn = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_tpl2scn, mask = cv2.findHomography(pts_tpl, pts_scn, cv2.RANSAC, 4.0)
    if H_tpl2scn is None:
        return None
    H_scn2tpl = np.linalg.inv(H_tpl2scn)
    inliers = int(mask.ravel().astype(bool).sum()) if mask is not None else 0
    reproj_error = None
    if mask is not None and inliers > 0:
        inmask = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(pts_tpl[inmask], H_tpl2scn)
        errs = np.linalg.norm(pts_scn[inmask] - proj, axis=2)
        reproj_error = float(np.mean(errs))
    return {
        "method": detector_name,
        "H_tpl2scn": H_tpl2scn,
        "H_scn2tpl": H_scn2tpl,
        "inliers": inliers,
        "reproj": reproj_error,
    }


def cast_template_and_crop(template_bgr: np.ndarray, scene_bgr: np.ndarray, cfg: TemplateConfig, debug_dir: Optional[Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    result = None
    for method in ("sift", "orb"):
        result = _compute_homography(method, template_bgr, scene_bgr)
        if result and result["inliers"] >= cfg.alignment_min_inliers:
            break
    if not result:
        raise RuntimeError("Could not cast template into image and locate card outline.")
    h_tpl, w_tpl = template_bgr.shape[:2]
    tpl_corners = np.float32([[0, 0], [w_tpl - 1, 0], [w_tpl - 1, h_tpl - 1], [0, h_tpl - 1]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(tpl_corners, result["H_tpl2scn"])
    ordered_scene = _order_quad(scene_corners)
    ordered_tpl = np.float32([[0, 0], [w_tpl - 1, 0], [w_tpl - 1, h_tpl - 1], [0, h_tpl - 1]])
    perspective = cv2.getPerspectiveTransform(ordered_scene, ordered_tpl)
    rectified = cv2.warpPerspective(scene_bgr, perspective, (w_tpl, h_tpl))
    if debug_dir is not None:
        outline = draw_polygon(scene_bgr, ordered_scene, (0, 255, 0))
        save_img(debug_dir / "scene_with_template_outline.png", outline)
        save_img(debug_dir / "rectified_card.png", rectified)
    return rectified, {
        "method": result["method"],
        "alignment_ok": bool(result["inliers"] >= cfg.alignment_min_inliers),
        "alignment_inliers": result["inliers"],
        "alignment_reprojection_error": result["reproj"],
        "scene_corners": ordered_scene.tolist(),
    }


def stable_validation(template_bgr: np.ndarray, aligned_bgr: np.ndarray, cfg: TemplateConfig, debug_dir: Optional[Path]) -> Dict[str, Any]:
    tpl_g = enhance_gray(template_bgr)
    id_g = enhance_gray(aligned_bgr)
    patch_scores: List[Dict[str, float]] = []
    for idx, roi in enumerate(cfg.validation_patches):
        tp = crop_normalized_roi(tpl_g, roi)
        ip = crop_normalized_roi(id_g, roi)
        tp = cv2.GaussianBlur(tp, (5, 5), 0)
        ip = cv2.GaussianBlur(ip, (5, 5), 0)
        if tp.shape != ip.shape:
            ip = cv2.resize(ip, (tp.shape[1], tp.shape[0]))
        score = float(ssim(tp, ip))
        patch_scores.append({"index": idx, "ssim": score})
        if debug_dir is not None:
            save_img(debug_dir / "validation" / f"template_patch_{idx}.png", tp)
            save_img(debug_dir / "validation" / f"aligned_patch_{idx}.png", ip)
    valid = all(x["ssim"] >= cfg.validation_ssim_threshold for x in patch_scores)
    return {"valid": bool(valid), "patch_scores": patch_scores, "threshold": cfg.validation_ssim_threshold}


class OCRBackend:
    def __init__(self) -> None:
        self.backend = ""
        self.engine = None
        self._setup()

    def _setup(self) -> None:
        forced = os.environ.get("FORCE_OCR_BACKEND", "auto").strip().lower()
        if forced in {"paddle", "auto"} and PaddleOCR is not None:
            try:
                self.engine = PaddleOCR(
                    lang="ar",
                    device="cpu",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
                self.backend = "paddle"
                return
            except Exception:
                self.engine = None
        if pytesseract is None:
            raise RuntimeError(f"No OCR backend available. Paddle error={_PADDLE_ERROR!r}, pytesseract error={_PYTESS_ERROR!r}")
        tesseract_path = _find_tesseract_executable()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.backend = "tesseract"

    def _collect_paddle_texts(self, obj: Any) -> List[str]:
        out: List[str] = []
        if obj is None:
            return out
        if isinstance(obj, str):
            s = obj.strip()
            return [s] if s else []
        if hasattr(obj, "res"):
            return self._collect_paddle_texts(getattr(obj, "res"))
        if isinstance(obj, dict):
            if "res" in obj:
                out.extend(self._collect_paddle_texts(obj["res"]))
            if "prunedResult" in obj:
                out.extend(self._collect_paddle_texts(obj["prunedResult"]))
            if isinstance(obj.get("rec_texts"), (list, tuple)):
                out.extend([str(x).strip() for x in obj["rec_texts"] if str(x).strip()])
            if isinstance(obj.get("texts"), (list, tuple)):
                out.extend([str(x).strip() for x in obj["texts"] if str(x).strip()])
            if isinstance(obj.get("text"), str) and obj["text"].strip():
                out.append(obj["text"].strip())
            return out
        if isinstance(obj, (list, tuple)):
            for item in obj:
                out.extend(self._collect_paddle_texts(item))
        return out

    def _score(self, text: str, field_type: str) -> float:
        text = clean_text(text)
        if not text:
            return 0.0
        if field_type == "id_number":
            digits = clean_id_number(text)
            score = len(digits) * 9.0
            if len(digits) == 14:
                score += 40.0
                if digits[0] in {"2", "3"}:
                    score += 20.0
            return score
        if field_type == "birthday":
            digits = re.sub(r"[^0-9]", "", normalize_digits(text))
            score = len(digits) * 6.0
            if re.search(r"\d{4}/\d{2}/\d{2}", clean_birthday(text, "")):
                score += 30.0
            return score
        arabic_chars = sum(len(x) for x in re.findall(r"[\u0600-\u06FF]+", text))
        latin_penalty = len(re.findall(r"[A-Za-z]", text)) * 3.0
        digit_penalty = len(re.findall(r"[0-9]", normalize_digits(text))) * 2.0
        return arabic_chars * 2.2 + len(text) - latin_penalty - digit_penalty

    def _run_tesseract_variants(self, gray: np.ndarray, field_type: str) -> str:
        outputs: List[str] = []
        if field_type in {"id_number", "birthday"}:
            configs = [
                ("eng", "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/"),
                ("ara", "--oem 1 --psm 7"),
                ("eng", "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/"),
            ]
        else:
            configs = [("ara", "--oem 1 --psm 6"), ("ara", "--oem 1 --psm 7"), ("ara", "--oem 1 --psm 13")]
        for lang, config in configs:
            try:
                txt = pytesseract.image_to_string(gray, lang=lang, config=config, timeout=10)
            except Exception:
                txt = ""
            outputs.append(clean_text(txt))
        return max(outputs, key=lambda t: self._score(t, field_type)) if outputs else ""

    def read_best(self, img_bgr: np.ndarray, field_type: str, debug_dir: Optional[Path] = None, field_name: Optional[str] = None) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        for tag, processed in generate_ocr_variants(img_bgr, field_type):
            text = ""
            if self.backend == "paddle":
                try:
                    texts = self._collect_paddle_texts(self.engine.predict(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)))
                    text = clean_text(" ".join(texts))
                except Exception:
                    text = ""
            else:
                text = self._run_tesseract_variants(processed, field_type)
            score = self._score(text, field_type)
            candidates.append({"tag": tag, "text": text, "score": score, "image": processed})
        best = max(candidates, key=lambda x: x["score"])
        if debug_dir is not None and field_name:
            cand_dir = debug_dir / "ocr_candidates" / field_name
            ensure_dir(cand_dir)
            meta = []
            for idx, cand in enumerate(candidates):
                save_img(cand_dir / f"{idx:02d}_{cand['tag']}.png", cand["image"])
                meta.append({"tag": cand["tag"], "score": cand["score"], "text": cand["text"]})
            (cand_dir / "candidates.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return best


def _find_tesseract_executable() -> Optional[str]:
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return cand
    return None


def refine_roi_by_projection(img_bgr: np.ndarray, roi: ROI, cfg: TemplateConfig, field_type: str) -> Tuple[ROI, np.ndarray]:
    expanded = expand_roi(roi, cfg.horizontal_refine_ratio, cfg.vertical_refine_ratio)
    crop = crop_normalized_roi(img_bgr, expanded)
    gray = enhance_gray(crop)
    big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    thr = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if field_type in {"id_number", "birthday"}:
        thr = cv2.dilate(thr, np.ones((3, 3), np.uint8), iterations=1)
    else:
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8), iterations=1)
    row_score = thr.sum(axis=1).astype(np.float32)
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 31), 0).reshape(-1)
    active = np.where(row_score > max(12.0, row_score.max() * 0.28))[0]
    if active.size == 0:
        return roi, crop_normalized_roi(img_bgr, roi)
    y1p = max(0, int(active[0] / 2) - 3)
    y2p = min(gray.shape[0], int(active[-1] / 2) + 4)
    if y2p - y1p < max(10, gray.shape[0] // 6):
        return roi, crop_normalized_roi(img_bgr, roi)
    base = crop_normalized_roi(img_bgr, expanded)
    refined_crop = base[y1p:y2p, :]
    h, w = img_bgr.shape[:2]
    ex_y1 = int(round(expanded.y1 * h))
    new_y1 = (ex_y1 + y1p) / h
    new_y2 = (ex_y1 + y2p) / h
    refined_roi = ROI(expanded.x1, max(0.0, new_y1), expanded.x2, min(1.0, new_y2))
    return refined_roi, refined_crop


def extract_best_face_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    if face_recognition is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    def area(box: Sequence[int]) -> int:
        top, right, bottom, left = box
        return (right - left) * (bottom - top)
    largest = max(boxes, key=area)
    enc = face_recognition.face_encodings(rgb, known_face_locations=[largest])
    return enc[0] if enc else None


def compare_faces(id_face_bgr: np.ndarray, selfie_bgr: np.ndarray, threshold: float = 0.60) -> Tuple[bool, Optional[float]]:
    id_emb = extract_best_face_embedding(id_face_bgr)
    selfie_emb = extract_best_face_embedding(selfie_bgr)
    if id_emb is None or selfie_emb is None:
        return False, None
    dist = float(np.linalg.norm(id_emb - selfie_emb))
    return dist <= threshold, dist


def extract_fields(aligned_bgr: np.ndarray, ocr: OCRBackend, cfg: TemplateConfig, debug_dir: Optional[Path]) -> Dict[str, Any]:
    field_defs = [
        ("full_name", cfg.full_name, "name"),
        ("full_address", cfg.full_address, "address"),
        ("id_number", cfg.id_number, "id_number"),
        ("birthday", cfg.birthday, "birthday"),
    ]
    results: Dict[str, Any] = {}
    overlay = aligned_bgr.copy()
    for field_name, roi, field_type in field_defs:
        refined_roi, refined_crop = refine_roi_by_projection(aligned_bgr, roi, cfg, field_type)
        best = ocr.read_best(refined_crop, field_type, debug_dir, field_name)
        results[field_name] = best["text"]
        overlay = draw_roi(overlay, roi, f"{field_name}:base", (0, 255, 255))
        overlay = draw_roi(overlay, refined_roi, field_name, (0, 255, 0))
        if debug_dir is not None:
            save_img(debug_dir / "crops_raw" / f"{field_name}.png", crop_normalized_roi(aligned_bgr, roi))
            save_img(debug_dir / "crops_refined" / f"{field_name}.png", refined_crop)
            save_img(debug_dir / "crops_processed" / f"{field_name}_{best['tag']}.png", best["image"])
    results["id_number"] = clean_id_number(results.get("id_number", ""))
    results["birthday"] = clean_birthday(results.get("birthday", ""), results["id_number"])
    if not results["birthday"]:
        results["birthday"] = infer_birthday_from_id(results["id_number"])
    results["full_name"] = clean_text(results.get("full_name", ""))
    results["full_address"] = clean_text(results.get("full_address", ""))
    if debug_dir is not None:
        overlay = draw_roi(overlay, cfg.face_photo, "face_photo", (255, 0, 0))
        save_img(debug_dir / "aligned_with_boxes.png", overlay)
    return results


def process_id_card(template_path: str | Path, id_image_path: str | Path, selfie_path: Optional[str | Path] = None, debug_dir: Optional[str | Path] = None, skip_face: bool = False) -> Dict[str, Any]:
    cfg = get_default_template_config()
    template_bgr = read_image_bgr(template_path)
    id_bgr = read_image_bgr(id_image_path)
    selfie_bgr = read_image_bgr(selfie_path) if selfie_path and Path(selfie_path).exists() else None
    debug_dir_path = Path(debug_dir) if debug_dir else None
    if debug_dir_path is not None:
        ensure_dir(debug_dir_path)
        save_img(debug_dir_path / "template.png", template_bgr)
        save_img(debug_dir_path / "input_scene.png", id_bgr)
        if selfie_bgr is not None:
            save_img(debug_dir_path / "selfie.png", selfie_bgr)
    aligned_bgr, cast_info = cast_template_and_crop(template_bgr, id_bgr, cfg, debug_dir_path)
    if debug_dir_path is not None:
        save_img(debug_dir_path / "aligned_id.png", aligned_bgr)
    validation = stable_validation(template_bgr, aligned_bgr, cfg, debug_dir_path)
    face_crop = crop_normalized_roi(aligned_bgr, cfg.face_photo)
    if debug_dir_path is not None:
        save_img(debug_dir_path / "face_crop.png", face_crop)
    if skip_face or selfie_bgr is None or face_recognition is None:
        faces_match, face_distance = False, None
        face_check_skipped = True
    else:
        faces_match, face_distance = compare_faces(face_crop, selfie_bgr)
        face_check_skipped = False
    ocr = OCRBackend()
    fields = extract_fields(aligned_bgr, ocr, cfg, debug_dir_path)
    id_image_valid = bool(cast_info["alignment_ok"] and validation["valid"] and (True if face_check_skipped else faces_match))
    return {
        "verification": {
            "id_image_valid": id_image_valid,
            "alignment_ok": cast_info["alignment_ok"],
            "alignment_inliers": cast_info["alignment_inliers"],
            "alignment_reprojection_error": cast_info["alignment_reprojection_error"],
            "alignment_method": cast_info["method"],
            "scene_corners": cast_info["scene_corners"],
            "nonchangeable_zones": validation,
            "face_match": faces_match,
            "face_distance": face_distance,
            "face_check_skipped": face_check_skipped,
            "ocr_backend": ocr.backend,
        },
        "fields": fields,
    }


def find_first_image(folder: str | Path) -> Optional[Path]:
    folder = Path(folder)
    if not folder.exists():
        return None
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    return files[0] if files else None


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--id_image", required=True)
    p.add_argument("--selfie", default="")
    p.add_argument("--output_json", required=True)
    p.add_argument("--debug_dir", default="")
    p.add_argument("--skip_face", action="store_true")
    return p.parse_args()


def main() -> None:
    safe_stdio()
    args = parse_args()
    result = process_id_card(
        template_path=args.template,
        id_image_path=args.id_image,
        selfie_path=args.selfie or None,
        debug_dir=args.debug_dir or None,
        skip_face=args.skip_face,
    )
    write_json(args.output_json, result)
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
