
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "New folder (6)"
REQUIREMENTS = ROOT / "requirements_researched.txt"

def ensure_folders() -> None:
    for p in [
        DATA_ROOT,
        DATA_ROOT / "template",
        DATA_ROOT / "selfie",
        DATA_ROOT / "tests",
        ROOT / "outputs_researched",
        ROOT / "outputs_researched" / "debug",
    ]:
        p.mkdir(parents=True, exist_ok=True)

def main() -> None:
    ensure_folders()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(REQUIREMENTS)])
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    subprocess.check_call([sys.executable, str(ROOT / "run_batch_from_folders_newfolder6_researched.py")])

if __name__ == "__main__":
    main()
