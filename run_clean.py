"""data/processed/*.txt 를 정제하여 *_clean.txt 로 저장."""
from pathlib import Path

from app.preprocess.clean_text import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    for path in sorted(DATA_PROCESSED.glob("*.txt")):
        if path.name.endswith("_clean.txt"):
            continue
        text = path.read_text(encoding="utf-8")
        cleaned = clean_text(text)
        out_path = path.parent / (path.stem + "_clean" + path.suffix)
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"{path.name} -> {out_path.name} ({len(text)} -> {len(cleaned)} chars)")


if __name__ == "__main__":
    main()
