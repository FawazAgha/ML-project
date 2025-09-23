import sys
import fitz
from pathlib import Path


def pdf_to_txt(pdf_path: Path, out_txt: Path | None = None) -> Path:
    doc = fitz.open(str(pdf_path))
    parts = [page.get_text("text") for page in doc]
    text = "\n".join(parts)
    target = out_txt or pdf_path.with_suffix(".txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print("wrote", target)
    return target


def _iter_pdf_sources(path: Path):
    if path.is_dir():
        yield from sorted(p for p in path.rglob("*.pdf") if p.is_file())
    elif path.is_file() and path.suffix.lower() == ".pdf":
        yield path
    else:
        print(f"skipping '{path}' (not a PDF)")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python pdf_to_txt.py <pdf-or-directory> [additional paths...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        for pdf in _iter_pdf_sources(Path(arg)):
            try:
                pdf_to_txt(pdf)
            except Exception as exc:  # pragma: no cover
                print(f"failed to convert '{pdf}': {exc}")
