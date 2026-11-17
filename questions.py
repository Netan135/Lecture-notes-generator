from transformers import pipeline
import re

print("Loading question-generation model... (~250 MB, one-time download)")
qg = pipeline(
    "text2text-generation",
    model="valhalla/t5-small-e2e-qg",
    tokenizer="valhalla/t5-small-e2e-qg",
    use_fast=False
)
print("Model loaded!\n")
# 1. Read the transcript we made earlier
with open("transcript.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# 2. Split the transcript into sentences (rough split — good enough)
sentences = re.split(r'(?<=[.!?])\s+', text)
sentences = [s.strip() for s in sentences if len(s.split()) >= 5]  # skip too-short lines

print(f"Found {len(sentences)} usable sentence(s).\n")

# 3. Generate questions for each sentence
all_questions = []
for idx, sentence in enumerate(sentences, start=1):
    print(f"--- Sentence {idx}/{len(sentences)} ---")
    print(f"Source: {sentence}")

    # The valhalla/e2e-qg model expects the "generate questions:" prefix
    prompt = "generate questions: " + sentence
    out = qg(prompt, max_length=64, num_beams=4, do_sample=False)
    generated = out[0]["generated_text"]

    # The model separates multiple questions with </s>
    qs = [q.strip() for q in generated.split("</s>") if q.strip()]
    for q in qs:
        print(f"  → {q}")
        all_questions.append(q)
    print()

# 4. Save all questions
with open("questions.txt", "w", encoding="utf-8") as f:
    for i, q in enumerate(all_questions, start=1):
        f.write(f"{i}. {q}\n")

print("--- Done! ---")
print(f"Generated {len(all_questions)} question(s). Saved to questions.txt")