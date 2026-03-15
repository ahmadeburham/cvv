
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
    from paddleocr import PaddleOCR  # type: ignore
except Exception as exc:
    PaddleOCR = None
    _PADDLE_ERROR = exc
else:
    _PADDLE_ERROR = None

try:
    import pytesseract  # type: ignore
except Exception as exc:
    pytesseract = None
    _PYTESS_ERROR = exc
else:
    _PYTESS_ERROR = None

try:
    import face_recognition  # type: ignore
except Exception:
    face_recognition = None


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ARABIC_TO_ASCII_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


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
    validation_threshold: float = 0.42
    alignment_min_inliers: int = 24


def get_default_template_config() -> TemplateConfig:
    return TemplateConfig(
        full_name=ROI(0.56, 0.24, 0.96, 0.48),
        full_address=ROI(0.53, 0.46, 0.96, 0.70),
        id_number=ROI(0.38, 0.68, 0.96, 0.84),
        birthday=ROI(0.02, 0.67, 0.31, 0.82),
        face_photo=ROI(0.02, 0.11, 0.35, 0.62),
        validation_patches=[
            ROI(0.37, 0.18, 0.50, 0.31),
            ROI(0.28, 0.49, 0.46, 0.66),
            ROI(0.72, 0.86, 0.95, 0.96),
        ],
        validation_threshold=0.42,
        alignment_min_inliers=24,
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


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return img[y1:y2, x1:x2]


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
    text = text.replace("،", " ")
    return text.strip(" -_.,;:|/")


def clean_id_number(text: str) -> str:
    digits = re.sub(r"[^0-9]", "", normalize_digits(text))
    strong = re.findall(r"[23]\d{13}", digits)
    if strong:
        return strong[0]
    return digits[:14]


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


def egypt_id_plausible(id_number: str) -> bool:
    if len(id_number) != 14 or id_number[0] not in {"2", "3"}:
        return False
    b = infer_birthday_from_id(id_number)
    return bool(b)


def enhance_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


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
    elif detector_name == "akaze":
        detector = cv2.AKAZE_create()
        kp1, des1 = detector.detectAndCompute(tpl_g, None)
        kp2, des2 = detector.detectAndCompute(scn_g, None)
        if des1 is None or des2 is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio = 0.78
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

    if len(good) < 20:
        return None

    good = sorted(good, key=lambda m: m.distance)[:400]
    pts_tpl = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_scn = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_tpl2scn, mask = cv2.findHomography(pts_tpl, pts_scn, cv2.RANSAC, 4.0)
    if H_tpl2scn is None:
        return None
    inliers = int(mask.ravel().astype(bool).sum()) if mask is not None else 0
    reproj_error = None
    if mask is not None and inliers:
        sel = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(pts_tpl[sel], H_tpl2scn)
        reproj_error = float(np.mean(np.linalg.norm(pts_scn[sel] - proj, axis=2)))
    return {"method": detector_name, "H_tpl2scn": H_tpl2scn, "inliers": inliers, "reproj": reproj_error}


def cast_template_and_crop(template_bgr: np.ndarray, scene_bgr: np.ndarray, cfg: TemplateConfig, debug_dir: Optional[Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    best = None
    for method in ("sift", "akaze", "orb"):
        candidate = _compute_homography(method, template_bgr, scene_bgr)
        if not candidate:
            continue
        if best is None or candidate["inliers"] > best["inliers"]:
            best = candidate
        if candidate["inliers"] >= cfg.alignment_min_inliers:
            break
    if best is None:
        raise RuntimeError("Failed to locate the ID outline using template casting.")
    h_tpl, w_tpl = template_bgr.shape[:2]
    tpl_corners = np.float32([[0, 0], [w_tpl - 1, 0], [w_tpl - 1, h_tpl - 1], [0, h_tpl - 1]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(tpl_corners, best["H_tpl2scn"])
    ordered_scene = _order_quad(scene_corners)
    ordered_tpl = np.float32([[0, 0], [w_tpl - 1, 0], [w_tpl - 1, h_tpl - 1], [0, h_tpl - 1]])
    H = cv2.getPerspectiveTransform(ordered_scene, ordered_tpl)
    rectified = cv2.warpPerspective(scene_bgr, H, (w_tpl, h_tpl))
    if debug_dir is not None:
        save_img(debug_dir / "scene_with_template_outline.png", draw_polygon(scene_bgr, ordered_scene, (0, 255, 0)))
        save_img(debug_dir / "rectified_card.png", rectified)
    return rectified, {
        "alignment_ok": bool(best["inliers"] >= cfg.alignment_min_inliers),
        "alignment_method": best["method"],
        "alignment_inliers": best["inliers"],
        "alignment_reprojection_error": best["reproj"],
        "scene_corners": ordered_scene.tolist(),
    }


def patch_edge_score(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(a, 80, 160)
    eb = cv2.Canny(b, 80, 160)
    inter = np.logical_and(ea > 0, eb > 0).sum()
    union = np.logical_or(ea > 0, eb > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def validate_layout(template_bgr: np.ndarray, aligned_bgr: np.ndarray, cfg: TemplateConfig, debug_dir: Optional[Path]) -> Dict[str, Any]:
    tpl_g = enhance_gray(template_bgr)
    aln_g = enhance_gray(aligned_bgr)
    patch_scores = []
    total = []
    for idx, roi in enumerate(cfg.validation_patches):
        tp = crop_normalized_roi(tpl_g, roi)
        ap = crop_normalized_roi(aln_g, roi)
        ap = cv2.resize(ap, (tp.shape[1], tp.shape[0]))
        gray_ssim = float(ssim(tp, ap))
        edge_sim = patch_edge_score(tp, ap)
        score = 0.4 * gray_ssim + 0.6 * edge_sim
        total.append(score)
        patch_scores.append({"index": idx, "ssim": gray_ssim, "edge": edge_sim, "score": score})
        if debug_dir is not None:
            save_img(debug_dir / "validation" / f"template_patch_{idx}.png", tp)
            save_img(debug_dir / "validation" / f"aligned_patch_{idx}.png", ap)
    mean_score = float(np.mean(total)) if total else 0.0
    return {
        "valid": bool(mean_score >= cfg.validation_threshold),
        "mean_score": mean_score,
        "patch_scores": patch_scores,
        "threshold": cfg.validation_threshold,
    }


def _find_tesseract_executable() -> Optional[str]:
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


class OCRBackend:
    def __init__(self) -> None:
        self.backend = ""
        self.engine = None
        self._setup()

    def _setup(self) -> None:
        if PaddleOCR is not None:
            try:
                self.engine = PaddleOCR(
                    lang="ar",
                    device="cpu",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=True,
                    use_textline_orientation=True,
                )
                self.backend = "paddle"
                return
            except Exception:
                self.engine = None
        if pytesseract is None:
            raise RuntimeError(f"No OCR backend available. Paddle={_PADDLE_ERROR!r}, Tesseract={_PYTESS_ERROR!r}")
        cmd = _find_tesseract_executable()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
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
            for k in ("rec_texts", "texts"):
                if isinstance(obj.get(k), (list, tuple)):
                    out.extend([str(x).strip() for x in obj[k] if str(x).strip()])
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
            score = len(digits) * 8
            if egypt_id_plausible(digits):
                score += 50
            return float(score)
        if field_type == "birthday":
            cleaned = clean_birthday(text, "")
            return 40.0 if re.fullmatch(r"\d{4}/\d{2}/\d{2}", cleaned) else float(len(cleaned))
        arabic_chars = sum(1 for ch in text if '\u0600' <= ch <= '\u06ff')
        latin = len(re.findall(r"[A-Za-z]", text))
        digits = len(re.findall(r"\d", normalize_digits(text)))
        return float(arabic_chars * 2 + len(text) - latin * 2 - digits * 1.5)

    def _tesseract_read(self, gray: np.ndarray, field_type: str) -> str:
        outputs = []
        if field_type in {"id_number", "birthday"}:
            configs = [
                ("eng", "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/"),
                ("eng", "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/"),
                ("ara", "--oem 1 --psm 7"),
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

    def read_candidates(self, img_bgr: np.ndarray, field_type: str) -> List[Dict[str, Any]]:
        gray = enhance_gray(img_bgr)
        base = cv2.resize(gray, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
        variants = [
            ("gray", base),
            ("otsu", cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive", cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)),
            ("inv_otsu", cv2.bitwise_not(cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])),
        ]
        if field_type in {"id_number", "birthday"}:
            variants.append(("digits_dilate", cv2.dilate(variants[1][1], np.ones((2, 2), np.uint8), iterations=1)))

        out = []
        for tag, proc in variants:
            if self.backend == "paddle":
                try:
                    texts = self._collect_paddle_texts(self.engine.predict(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)))
                    txt = clean_text(" ".join(texts))
                except Exception:
                    txt = ""
            else:
                txt = self._tesseract_read(proc, field_type)
            out.append({"tag": tag, "image": proc, "text": txt, "score": self._score(txt, field_type)})
        return out

    def read_best(self, img_bgr: np.ndarray, field_type: str, debug_dir: Optional[Path] = None, field_name: Optional[str] = None) -> Dict[str, Any]:
        candidates = self.read_candidates(img_bgr, field_type)
        best = max(candidates, key=lambda c: c["score"]) if candidates else {"text": "", "score": 0.0, "tag": "none", "image": enhance_gray(img_bgr)}
        if debug_dir is not None and field_name:
            cdir = debug_dir / "ocr_candidates" / field_name
            ensure_dir(cdir)
            meta = []
            for idx, cand in enumerate(candidates):
                save_img(cdir / f"{idx:02d}_{cand['tag']}.png", cand["image"])
                meta.append({"tag": cand["tag"], "score": cand["score"], "text": cand["text"]})
            write_json(cdir / "candidates.json", meta)
        return best


def segment_text_lines(block_bgr: np.ndarray, numeric: bool = False) -> List[np.ndarray]:
    gray = enhance_gray(block_bgr)
    big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    thr = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3 if numeric else 25), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    profile = thr.sum(axis=1).astype(np.float32)
    if profile.max() <= 0:
        return [block_bgr]
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 31), 0).reshape(-1)
    active = np.where(smooth > max(10.0, smooth.max() * (0.25 if not numeric else 0.20)))[0]
    if active.size == 0:
        return [block_bgr]
    groups = []
    start = active[0]
    prev = active[0]
    for idx in active[1:]:
        if idx - prev > 12:
            groups.append((start, prev))
            start = idx
        prev = idx
    groups.append((start, prev))
    lines = []
    for a, b in groups:
        y1 = max(0, int(a / 2) - 3)
        y2 = min(block_bgr.shape[0], int(b / 2) + 4)
        if y2 - y1 >= max(12, block_bgr.shape[0] // 10):
            lines.append(block_bgr[y1:y2, :])
    return lines or [block_bgr]


def read_multiline_block(img_bgr: np.ndarray, ocr: OCRBackend, field_name: str, debug_dir: Optional[Path]) -> str:
    lines = segment_text_lines(img_bgr, numeric=False)
    texts = []
    line_dir = debug_dir / "segmented_lines" / field_name if debug_dir is not None else None
    if line_dir is not None:
        ensure_dir(line_dir)
    for idx, line in enumerate(lines):
        best = ocr.read_best(line, "text", debug_dir, f"{field_name}_line_{idx}")
        txt = clean_text(best["text"])
        if txt:
            texts.append(txt)
        if line_dir is not None:
            save_img(line_dir / f"line_{idx}.png", line)
    return clean_text(" ".join(texts))


def refine_numeric_crop(img_bgr: np.ndarray, roi: ROI) -> np.ndarray:
    block = crop_normalized_roi(img_bgr, roi)
    lines = segment_text_lines(block, numeric=True)
    if lines:
        # choose the widest line
        return max(lines, key=lambda x: x.shape[1] * x.shape[0])
    return block


def extract_best_face_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    if face_recognition is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    def area(box: Sequence[int]) -> int:
        t, r, b, l = box
        return max(0, r - l) * max(0, b - t)
    box = max(boxes, key=area)
    enc = face_recognition.face_encodings(rgb, known_face_locations=[box])
    return enc[0] if enc else None


def compare_faces(id_face_bgr: np.ndarray, selfie_bgr: np.ndarray, threshold: float = 0.60) -> Tuple[bool, Optional[float]]:
    emb1 = extract_best_face_embedding(id_face_bgr)
    emb2 = extract_best_face_embedding(selfie_bgr)
    if emb1 is None or emb2 is None:
        return False, None
    dist = float(np.linalg.norm(emb1 - emb2))
    return dist <= threshold, dist


def extract_fields(aligned_bgr: np.ndarray, ocr: OCRBackend, cfg: TemplateConfig, debug_dir: Optional[Path]) -> Dict[str, str]:
    full_name_crop = crop_normalized_roi(aligned_bgr, cfg.full_name)
    full_address_crop = crop_normalized_roi(aligned_bgr, cfg.full_address)
    id_number_crop = refine_numeric_crop(aligned_bgr, cfg.id_number)
    birthday_crop = refine_numeric_crop(aligned_bgr, cfg.birthday)

    if debug_dir is not None:
        save_img(debug_dir / "crops_raw" / "full_name.png", full_name_crop)
        save_img(debug_dir / "crops_raw" / "full_address.png", full_address_crop)
        save_img(debug_dir / "crops_raw" / "id_number.png", id_number_crop)
        save_img(debug_dir / "crops_raw" / "birthday.png", birthday_crop)

    full_name = read_multiline_block(full_name_crop, ocr, "full_name", debug_dir)
    full_address = read_multiline_block(full_address_crop, ocr, "full_address", debug_dir)

    id_best = ocr.read_best(id_number_crop, "id_number", debug_dir, "id_number")
    birthday_best = ocr.read_best(birthday_crop, "birthday", debug_dir, "birthday")
    id_number = clean_id_number(id_best["text"])
    birthday = clean_birthday(birthday_best["text"], id_number)

    if not birthday and egypt_id_plausible(id_number):
        birthday = infer_birthday_from_id(id_number)

    return {
        "full_name": full_name,
        "full_address": full_address,
        "id_number": id_number,
        "birthday": birthday,
    }


def field_plausibility(fields: Dict[str, str]) -> Dict[str, Any]:
    name = fields.get("full_name", "")
    address = fields.get("full_address", "")
    id_number = fields.get("id_number", "")
    birthday = fields.get("birthday", "")
    name_ok = sum(1 for ch in name if '\u0600' <= ch <= '\u06ff') >= 5
    address_ok = sum(1 for ch in address if '\u0600' <= ch <= '\u06ff') >= 8
    id_ok = egypt_id_plausible(id_number)
    bday_ok = bool(re.fullmatch(r"\d{4}/\d{2}/\d{2}", birthday))
    return {
        "name_ok": name_ok,
        "address_ok": address_ok,
        "id_number_ok": id_ok,
        "birthday_ok": bday_ok,
        "all_ok": bool(name_ok and address_ok and id_ok and bday_ok),
    }


def process_id_card(template_path: str | Path, id_image_path: str | Path, selfie_path: Optional[str | Path] = None, debug_dir: Optional[str | Path] = None, skip_face: bool = False) -> Dict[str, Any]:
    cfg = get_default_template_config()
    template_bgr = read_image_bgr(template_path)
    scene_bgr = read_image_bgr(id_image_path)
    selfie_bgr = read_image_bgr(selfie_path) if selfie_path and Path(selfie_path).exists() else None
    dbg = Path(debug_dir) if debug_dir else None
    if dbg is not None:
        ensure_dir(dbg)
        save_img(dbg / "template.png", template_bgr)
        save_img(dbg / "input_scene.png", scene_bgr)
        if selfie_bgr is not None:
            save_img(dbg / "selfie.png", selfie_bgr)

    aligned_bgr, align_info = cast_template_and_crop(template_bgr, scene_bgr, cfg, dbg)
    if dbg is not None:
        save_img(dbg / "aligned_id.png", aligned_bgr)

    validation = validate_layout(template_bgr, aligned_bgr, cfg, dbg)

    face_crop = crop_normalized_roi(aligned_bgr, cfg.face_photo)
    if dbg is not None:
        save_img(dbg / "face_crop.png", face_crop)
        overlay = aligned_bgr.copy()
        overlay = draw_roi(overlay, cfg.full_name, "full_name", (0,255,0))
        overlay = draw_roi(overlay, cfg.full_address, "full_address", (0,255,255))
        overlay = draw_roi(overlay, cfg.id_number, "id_number", (255,0,0))
        overlay = draw_roi(overlay, cfg.birthday, "birthday", (0,0,255))
        overlay = draw_roi(overlay, cfg.face_photo, "face", (255,0,255))
        save_img(dbg / "aligned_with_boxes.png", overlay)

    if skip_face or selfie_bgr is None or face_recognition is None:
        face_match, face_distance, face_check_skipped = False, None, True
    else:
        face_match, face_distance = compare_faces(face_crop, selfie_bgr)
        face_check_skipped = False

    ocr = OCRBackend()
    fields = extract_fields(aligned_bgr, ocr, cfg, dbg)
    plaus = field_plausibility(fields)

    id_image_valid = bool(align_info["alignment_ok"] and plaus["all_ok"] and (True if face_check_skipped else face_match))

    return {
        "verification": {
            **align_info,
            "layout_validation": validation,
            "field_plausibility": plaus,
            "id_image_valid": id_image_valid,
            "face_match": face_match,
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
    imgs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    return imgs[0] if imgs else None


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


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
        args.template,
        args.id_image,
        args.selfie or None,
        args.debug_dir or None,
        args.skip_face,
    )
    write_json(args.output_json, result)
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
