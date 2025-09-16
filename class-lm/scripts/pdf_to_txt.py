import sys, fitz, os

def pdf_to_txt(pdf_path, out_txt=None):
    doc = fitz.open(pdf_path)
    parts = []
    for p in doc:
        parts.append(p.get_text("text"))
    text = "\n".join(parts)
    out_txt = out_txt or os.path.splitext(pdf_path)[0] + ".txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", out_txt)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        pdf_to_txt(p)
