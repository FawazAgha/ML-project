#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, sys, unicodedata, io

def read_text(path):
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path, text):
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text)

def normalize_whitespace(text: str) -> str:
    # unify newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # mark paragraph breaks (2+ newlines)
    text = re.sub(r"\n{2,}", "\n\n", text)
    PARA = "¶¶¶PARA¶¶¶"
    text = text.replace("\n\n", f"\n{PARA}\n")
    # collapse single newlines inside paragraphs -> space
    text = re.sub(r"\n", " ", text)
    # restore paragraph breaks
    text = re.sub(fr"\s*{PARA}\s*", "\n\n", text)
    # collapse spaces
    text = re.sub(r"[ \t]+", " ", text)
    # tidy spaces around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+/","\u0020/\u0020", text)
    return text.strip()

def remove_zero_width(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch) in ("Cf",))

def unwrap_braced(cmd: str, text: str) -> str:
    # \text{...}, \mathrm{...}, \operatorname{...} → inside only
    return re.sub(rf"\\{cmd}\{{([^{{}}]*)\}}", r"\1", text, flags=re.IGNORECASE)

def clean_math(text: str) -> str:
    # 1) Unwrap {\displaystyle ...} -> keep inner
    # (non-greedy to avoid eating the world; good enough for WP dumps)
    text = re.sub(r"\{\s*\\displaystyle\s*.*?\}", "", text, flags=re.DOTALL|re.IGNORECASE)

    # 2) Unwrap common LaTeX commands with single-brace args
    for cmd in ["text", "mathrm", "operatorname", "mathit", "mathbf", "mathsf", "mathtt"]:
        text = unwrap_braced(cmd, text)

    # 3) Remove stray TeX delimiters \( \) \[ \]
    text = re.sub(r"\\[()\[\]]", "", text)

    # 4) Replace common LaTeX symbols to readable unicode/plain
    repl = {
        r"\\epsilon": "ε",
        r"\\phi": "φ",
        r"\\theta": "θ",
        r"\\leq": "≤",
        r"\\geq": "≥",
        r"\\ne": "≠",
        r"\\cdot": "·",
        r"\\times": "×",
        r"\\pm": "±",
        r"\\to": "→",
        r"\\infty": "∞",
        r"\\log": "log",
        r"\\ln": "ln",
        r"\\mathcal": "",
    }
    for k,v in repl.items():
        text = re.sub(k, v, text)

    # 5) Remove the weird function application symbol U+2061 (⁡)
    text = text.replace("⁡", "")

    # 6) Normalize Big-O spacing: "O ( log n )" -> "O(log n)"
    text = re.sub(r"\bO\s*\(\s*", "O(", text)
    text = re.sub(r"\s*\)", ")", text)  # close parens tight
    text = re.sub(r"\(\s+", "(", text) # open parens tight
    text = re.sub(r"\s*,\s*", ", ", text)

    # 7) Tighten inequalities and math spacing a bit
    text = re.sub(r"\s*<\s*", " < ", text)
    text = re.sub(r"\s*>\s*", " > ", text)
    text = re.sub(r"\s*≤\s*", " ≤ ", text)
    text = re.sub(r"\s*≥\s*", " ≥ ", text)
    text = re.sub(r"\s*=\s*", " = ", text)

    return text

def strip_footnotes(text: str) -> str:
    # remove citation-style [12], [1], etc.
    return re.sub(r"\[\d+\]", "", text)

def final_touches(text: str) -> str:
    # collapse multiple spaces again
    text = re.sub(r"[ \t]{2,}", " ", text)
    # ensure spaces around em/en-dash aren't doubled
    text = re.sub(r"\s*–\s*", "–", text)  # keep tight in "2–3 tree"
    return text.strip()

def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_wiki_math.py <input.txt> <output.txt>")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    txt = read_text(src)

    txt = remove_zero_width(txt)
    txt = clean_math(txt)
    txt = strip_footnotes(txt)
    txt = normalize_whitespace(txt)
    txt = final_touches(txt)

    write_text(dst, txt)

if __name__ == "__main__":
    main()
