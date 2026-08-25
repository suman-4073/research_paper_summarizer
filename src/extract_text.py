"""
 Extract raw text from a research paper PDF.

Usage:
    python extract_text.py path/to/paper.pdf
"""

import sys
import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """Extract raw text from every page of a PDF, in reading order."""
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # simple reading-order extraction
        full_text.append(text)
        print(f"--- Page {page_num + 1} extracted ({len(text)} chars) ---")

    doc.close()
    return "\n".join(full_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py path/to/paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    text = extract_text(pdf_path)

    print("\n===== FIRST 1000 CHARACTERS =====\n")
    print(text[:1000])

    # Save full text so you can inspect it in an editor
    out_path = pdf_path.replace(".pdf", "_raw.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nFull text saved to: {out_path}")