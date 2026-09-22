"""
Stage 5: Evaluate summary quality using ROUGE score.

We compare our generated Abstract-section summary against the paper's own
real abstract (which we already have, since we extracted it as a section).
This gives an objective quality number instead of just eyeballing output --
this is what turns the project from "a demo" into something with a
measurable result for a resume/README.

ROUGE-1 / ROUGE-2 measure word/phrase overlap; ROUGE-L measures longest
common subsequence (word order matters).
"""

from rouge_score import rouge_scorer


def evaluate_summary(generated_summary: str, reference_text: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores comparing a generated
    summary against a reference (ground-truth) text.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference_text, generated_summary)

    return {
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from section_splitter import split_sections
    from summarizer import load_summarizer, summarize_text, MODEL_OPTIONS, DEFAULT_MODEL

    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py path/to/cleaned_text_file.txt [bart|pegasus-arxiv]")
        sys.exit(1)

    model_key = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL
    if model_key not in MODEL_OPTIONS:
        print(f"Unknown model '{model_key}'. Choose from: {list(MODEL_OPTIONS)}")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cleaned = f.read()

    sections = split_sections(cleaned)

    if "Abstract" not in sections:
        print("No Abstract section found -- can't evaluate.")
        sys.exit(1)

    real_abstract = sections["Abstract"]

    body_sections = [
        content for name, content in sections.items()
        if name not in ("Front Matter", "Abstract", "References",
                         "Acknowledgements", "Acknowledgments")
    ]
    body_text = " ".join(body_sections)

    print(f"Loading model: {MODEL_OPTIONS[model_key]}")
    model, tokenizer, device = load_summarizer(model_key)

    print(f"Summarizing paper body ({len(body_text)} chars)...")
    generated = summarize_text(body_text, model, tokenizer, device,model_key)

    print("\n=== Generated Summary (from paper body) ===")
    print(generated)
    print("\n=== Real Abstract (reference) ===")
    print(real_abstract[:500] + ("..." if len(real_abstract) > 500 else ""))

    scores = evaluate_summary(generated, real_abstract)
    print(f"\n=== ROUGE Scores ({model_key}, generated vs. real abstract) ===")
    for metric, value in scores.items():
        print(f"{metric}: {value}")