
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "New folder (6)"
REQUIREMENTS = ROOT / "requirements_project_casted.txt"

FOLDERS = [
    DATA_ROOT,
    DATA_ROOT / "template",
    DATA_ROOT / "selfie",
    DATA_ROOT / "tests",
    ROOT / "outputs_casted",
    ROOT / "outputs_casted" / "debug",
]

MODULE_CHECKS = {
    "cv2": "opencv-python==4.10.0.84",
    "numpy": "numpy==1.26.4",
    "scikit_image": "scikit-image==0.22.0",
    "PIL": "Pillow==10.4.0",
}

def ensure_folders() -> None:
    for folder in FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)

def ensure_packages() -> None:
    needed = [
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-image==0.22.0",
        "opencv-python==4.10.0.84",
        "Pillow==10.4.0",
        "requests==2.31.0",
        "urllib3==2.2.3",
        "charset-normalizer==3.4.0",
        "chardet==5.2.0",
        "paddlepaddle==3.2.2",
        "paddleocr==3.4.0",
        "paddlex==3.4.2",
        "pytesseract==0.3.13",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *needed])

def main() -> None:
    ensure_folders()
    ensure_packages()
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    subprocess.check_call([sys.executable, str(ROOT / "run_batch_from_folders_newfolder6_casted.py")])

if __name__ == "__main__":
    main()
