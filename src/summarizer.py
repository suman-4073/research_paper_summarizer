"""
Stage 4: Summarize each section using a pretrained transformer model.

Uses facebook/bart-large-cnn (a widely-used, freely available summarization
model). Sections longer than the model's input limit are split into chunks,
summarized separately, then the chunk summaries are joined.

Sections we don't need to summarize (References, Acknowledgements, Front
Matter) are skipped.

NOTE: We load the model directly with AutoModelForSeq2SeqLM + .generate()
instead of using pipeline("summarization", ...). Transformers v5 removed
the old SummarizationPipeline shortcut, so pipeline("summarization") no
longer works there -- calling generate() directly works on both v4 and v5.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-large-cnn"

# Sections that are just bibliography/boilerplate -- no value in summarizing
SKIP_SECTIONS = {"Front Matter", "References", "Acknowledgements", "Acknowledgments"}

# BART's real limit is 1024 tokens; stay a bit under that to leave room
MAX_INPUT_TOKENS = 900


def load_summarizer():
    """Load the model and tokenizer once."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_INPUT_TOKENS) -> list[str]:
    """
    Split text into chunks that fit within the model's input token limit.
    Splits on whole sentences so we don't cut a sentence in half.
    """
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_len + sentence_len > max_tokens and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


def summarize_text(text: str, model, tokenizer, device) -> str:
    """Summarize a single piece of text, chunking first if it's too long."""
    chunks = chunk_text(text, tokenizer)
    summaries = []

    for chunk in chunks:
        input_len = len(tokenizer.encode(chunk, add_special_tokens=False))
        # Keep summary length reasonable relative to input
        max_len = min(150, max(30, input_len // 3))

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS + 24,  # small buffer for special tokens
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_length=max_len,
                min_length=15,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)


def summarize_sections(sections: dict) -> dict:
    """Summarize every section in the dict, skipping non-content sections."""
    model, tokenizer, device = load_summarizer()
    summaries = {}

    for name, content in sections.items():
        if name in SKIP_SECTIONS or not content.strip():
            continue
        print(f"Summarizing '{name}' ({len(content)} chars)...")
        summaries[name] = summarize_text(content, model, tokenizer, device)

    return summaries


if __name__ == "__main__":
    import sys
    from section_splitter import split_sections

    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python summarizer.py path/to/cleaned_text_file.txt")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cleaned = f.read()

    sections = split_sections(cleaned)
    summaries = summarize_sections(sections)

    for name, summary in summaries.items():
        print(f"\n=== {name} Summary ===")
        print(summary)