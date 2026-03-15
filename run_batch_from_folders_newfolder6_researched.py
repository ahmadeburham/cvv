
import json
import subprocess
import sys
from pathlib import Path

from id_card_pipeline_researched import find_first_image, list_images, write_json

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "New folder (6)"
TEMPLATE_DIR = DATA_ROOT / "template"
SELFIE_DIR = DATA_ROOT / "selfie"
TESTS_DIR = DATA_ROOT / "tests"
OUTPUTS_DIR = ROOT / "outputs_researched"
DEBUGS_DIR = OUTPUTS_DIR / "debug"

def run_batch(skip_face: bool = False) -> Path:
    template = find_first_image(TEMPLATE_DIR)
    selfie = find_first_image(SELFIE_DIR)
    ids = list_images(TESTS_DIR)
    if template is None:
        raise FileNotFoundError(f"No template image found in: {TEMPLATE_DIR}")
    if not ids:
        raise FileNotFoundError(f"No test ID images found in: {TESTS_DIR}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUGS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, id_path in enumerate(ids, start=1):
        print(f"[{i}/{len(ids)}] {id_path.name}")
        out_json = OUTPUTS_DIR / f"{id_path.stem}.json"
        debug_dir = DEBUGS_DIR / id_path.stem
        cmd = [
            sys.executable,
            str(ROOT / "id_card_pipeline_researched.py"),
            "--template", str(template),
            "--id_image", str(id_path),
            "--output_json", str(out_json),
            "--debug_dir", str(debug_dir),
        ]
        if selfie is not None:
            cmd += ["--selfie", str(selfie)]
        if skip_face or selfie is None:
            cmd.append("--skip_face")
        subprocess.check_call(cmd)
        item = json.loads(out_json.read_text(encoding="utf-8"))
        item["file"] = id_path.name
        summary.append(item)

    summary_path = OUTPUTS_DIR / "summary.json"
    write_json(summary_path, summary)
    print(f"Done: {summary_path}")
    return summary_path

if __name__ == "__main__":
    run_batch()
