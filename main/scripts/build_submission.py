"""
Assemble submission/ for supervisor handoff: copy code, checkpoint, predictions, requirements.
Does not remove submission/README.md or submission/approach.md if present.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = REPO_ROOT / "submission"
CODE_ROOT = SUBMISSION / "code"


def _ignore_main_copy(path: str, names: list[str]) -> set[str]:
    p = Path(path)
    skip: set[str] = set()
    for n in names:
        if n == "__pycache__" or n == ".DS_Store" or n.endswith(".pyc"):
            skip.add(n)
    # Under repo main/, omit bundled model weights and local outputs (copied separately)
    if p.resolve() == (REPO_ROOT / "main").resolve():
        for n in ("models", "outputs"):
            if n in names:
                skip.add(n)
    return skip


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    if CODE_ROOT.exists():
        shutil.rmtree(CODE_ROOT)
    CODE_ROOT.mkdir(parents=True)
    (CODE_ROOT / "models" / "dl").mkdir(parents=True)

    shutil.copytree(REPO_ROOT / "main", CODE_ROOT / "main", ignore=_ignore_main_copy)

    ck_src = REPO_ROOT / "main" / "models" / "dl" / "checkpoint_best.pt"
    ck_dst = CODE_ROOT / "models" / "dl" / "checkpoint_best.pt"
    if ck_src.is_file():
        shutil.copy2(ck_src, ck_dst)

    pred_written = False
    for candidate in (
        REPO_ROOT / "predictions.csv",
        REPO_ROOT / "main" / "outputs" / "predictions_dl.csv",
    ):
        if candidate.is_file():
            shutil.copy2(candidate, SUBMISSION / "predictions.csv")
            pred_written = True
            break

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        (SUBMISSION / "requirements.txt").write_text(proc.stdout)
        req_ok = True
    except (subprocess.CalledProcessError, OSError):
        (SUBMISSION / "requirements.txt").write_text("# pip freeze failed; run manually\n")
        req_ok = False

    readme_ok = (SUBMISSION / "README.md").is_file()
    approach_ok = (SUBMISSION / "approach.md").is_file()
    ck_ok = ck_dst.is_file()

    def mark(ok: bool) -> str:
        return "[✓]" if ok else "[✗]"

    print("\n--- Submission checklist ---")
    print(f"  {mark(pred_written)} predictions.csv present")
    print(f"  {mark(ck_ok)} checkpoint_best.pt present")
    print(f"  {mark(req_ok)} requirements.txt present")
    if approach_ok:
        print("  [✓] approach.md present")
    else:
        print("  [✗] approach.md missing  ← warn if not found")
    if readme_ok:
        print("  [✓] README.md present")
    else:
        print("  [✗] README.md missing")

    print("\nZip and send:")
    print(f"  cd {REPO_ROOT}")
    print("  zip -r submission.zip submission/")


if __name__ == "__main__":
    main()
