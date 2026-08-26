"""
Clean raw extracted text before section detection.

Handles the junk we saw in the real extraction test:
- Broken hyphenation across line breaks ("trans-\nformer" -> "transformer")
- arXiv stamp lines ("arXiv:1706.03762v7 [cs.CL] 2 Aug 2023")
- Excess blank lines / inconsistent whitespace
- Collapsing single line-breaks inside a paragraph (PDF extraction often
  breaks lines at the original page width, not at sentence/paragraph ends)
"""

import re


def fix_hyphenation(text: str) -> str:
    """Rejoin words that were split across a line break with a hyphen."""
    # e.g. "atten-\ntion" -> "attention"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_arxiv_stamp(text: str) -> str:
    """Remove arXiv watermark lines like 'arXiv:1706.03762v7 [cs.CL] 2 Aug 2023'."""
    pattern = r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[\w.]+\]\s*\d{1,2}\s\w+\s\d{4}"
    return re.sub(pattern, "", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse single newlines (mid-paragraph line breaks from PDF layout)
    into spaces, but keep paragraph breaks (double newlines) intact.
    """
    # Temporarily mark real paragraph breaks
    text = re.sub(r"\n\s*\n", "<PARA>", text)
    # Collapse remaining single newlines into spaces
    text = text.replace("\n", " ")
    # Restore paragraph breaks
    text = text.replace("<PARA>", "\n\n")
    # Collapse repeated spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """Run the full cleaning pipeline on raw extracted text."""
    text = fix_hyphenation(raw_text)
    text = remove_arxiv_stamp(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py path/to/raw_text_file.txt")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    print(cleaned[:1000])
    print(f"\n[Raw: {len(raw)} chars -> Cleaned: {len(cleaned)} chars]")