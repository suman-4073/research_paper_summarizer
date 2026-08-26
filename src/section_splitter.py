"""
Stage 3: Split cleaned paper text into named sections
(Abstract, Introduction, Related Work, Methods, Results, Conclusion, etc.)

Key challenge (confirmed from real test output): headings can appear with
NO surrounding whitespace or line breaks once text has been cleaned into
flowing paragraphs. So we can't rely on blank lines to find section starts —
we search for the heading words themselves as patterns in the text.
"""

import re

# Common section names in academic papers, in the order they usually appear.
# Optional leading number (e.g. "1 Introduction", "2. Background") is handled
# by the regex pattern below, not by this list.
SECTION_NAMES = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methods",
    "Methodology",
    "Model Architecture",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "Acknowledgements",
    "Acknowledgments",
    "References",
]


def find_section_positions(text: str) -> list[tuple[str, int]]:
    """
    Find the character position where each known section heading appears.
    Returns a list of (section_name, start_index) sorted by position in text.
    """
    matches = []

    for name in SECTION_NAMES:
        # Pattern: optional number/dot before heading, heading word(s),
        # immediately followed by a capital letter (start of body text)
        # or end of string. Word boundaries prevent partial-word matches
        # (e.g. won't match "Introduction" inside "Reintroduction").
        pattern = rf"(?:\d+\.?\s+)?\b{re.escape(name)}\b(?=\s+[A-Z]|\s*$)"
        match = re.search(pattern, text)
        if match:
            matches.append((name, match.start()))

    # Sort by where they actually appear in the document
    matches.sort(key=lambda m: m[1])
    return matches


def split_sections(clean_text: str) -> dict[str, str]:
    """
    Split cleaned text into a dict of {section_name: section_text}.
    Text before the first detected section is stored under 'Front Matter'
    (title, authors, etc.) — usually not needed for summarization.
    """
    positions = find_section_positions(clean_text)

    if not positions:
        # No known headings found — return everything as one block
        return {"Full Text": clean_text}

    sections = {}

    # Everything before the first detected heading
    first_start = positions[0][1]
    if first_start > 0:
        sections["Front Matter"] = clean_text[:first_start].strip()

    # Slice text between each heading and the next one
    for i, (name, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(clean_text)
        # Skip past the heading word itself so section text doesn't repeat it
        heading_match = re.match(rf"(?:\d+\.?\s+)?{re.escape(name)}\b", clean_text[start:])
        content_start = start + (heading_match.end() if heading_match else len(name))
        sections[name] = clean_text[content_start:end].strip()

    return sections


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python section_splitter.py path/to/cleaned_text_file.txt")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cleaned = f.read()

    sections = split_sections(cleaned)

    for name, content in sections.items():
        preview = content[:150].replace("\n", " ")
        print(f"\n=== {name} ({len(content)} chars) ===")
        print(f"{preview}...")