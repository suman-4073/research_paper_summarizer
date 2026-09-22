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

MODEL_OPTIONS = {
    "bart": "facebook/bart-large-cnn",
    "pegasus-arxiv": "google/pegasus-arxiv",
}
DEFAULT_MODEL = "bart"
USE_CHUNKING = {
    "bart": True,
    "pegasus-arxiv": False,
}
# Sections that are just bibliography/boilerplate -- no value in summarizing
SKIP_SECTIONS = {"Front Matter", "References", "Acknowledgements", "Acknowledgments"}

# BART's real limit is 1024 tokens; stay a bit under that to leave room
MAX_INPUT_TOKENS = 900


def load_summarizer(model_key: str = DEFAULT_MODEL):
    """Load the model and tokenizer once."""
    model_name = MODEL_OPTIONS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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


def _generate_summary(text_or_chunk: str, model, tokenizer, device, max_len: int = None) -> str:
    """Run the model on a single piece of text and decode the result."""
    input_len = len(tokenizer.encode(text_or_chunk, add_special_tokens=False))
    if max_len is None:
        max_len = min(200, max(30, input_len // 3))

    inputs = tokenizer(
        text_or_chunk,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS + 24,
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
    summary = summary.replace("<n>", " ").strip()
    return summary


def summarize_text(text: str, model, tokenizer, device, model_key: str = DEFAULT_MODEL) -> str:
    """
    If USE_CHUNKING[model_key] is True (BART), split into chunks and
    summarize each. If False (PEGASUS), summarize the whole text in one
    truncated pass to avoid hallucination on disconnected fragments.
    """
    if USE_CHUNKING.get(model_key, True):
        chunks = chunk_text(text, tokenizer)
        summaries = [_generate_summary(chunk, model, tokenizer, device) for chunk in chunks]
        return " ".join(summaries)
    else:
        return _generate_summary(text, model, tokenizer, device)


def summarize_sections(sections: dict, model_key: str = DEFAULT_MODEL) -> dict:
    """Summarize every section in the dict, skipping non-content sections."""
    model, tokenizer, device = load_summarizer(model_key)
    summaries = {}

    for name, content in sections.items():
        if name in SKIP_SECTIONS or not content.strip():
            continue
        print(f"Summarizing '{name}' ({len(content)} chars)...")
        summaries[name] = summarize_text(content, model, tokenizer, device,model_key)

    return summaries


if __name__ == "__main__":
    import sys
    from section_splitter import split_sections

    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python summarizer.py path/to/cleaned_text_file.txt [bart|pegasus-arxiv]")
        sys.exit(1)

    model_key = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL
    if model_key not in MODEL_OPTIONS:
        print(f"Unknown model '{model_key}'. Choose from: {list(MODEL_OPTIONS)}")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cleaned = f.read()

    sections = split_sections(cleaned)
    summaries = summarize_sections(sections, model_key)

    for name, summary in summaries.items():
        print(f"\n=== {name} Summary ===")
        print(summary)