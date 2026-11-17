from transformers import pipeline

print("Loading summarization model... (first time: ~1.6 GB download, please wait)")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model loaded!\n")

# Read the transcript we made in Step 4
with open("transcript.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

print(f"Original transcript length: {len(text)} characters")

# BART works best on ~1024 tokens (~700 words) at a time,
# so if the transcript is long, we cut it into chunks first
chunks = []
max_chars = 2500  # conservative chunk size in characters
for i in range(0, len(text), max_chars):
    chunks.append(text[i:i + max_chars])

print(f"Split into {len(chunks)} chunk(s). Summarizing each...\n")

summaries = []
for idx, chunk in enumerate(chunks, start=1):
    print(f"--- Summarizing chunk {idx}/{len(chunks)} ---")
    out = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
    summaries.append(out[0]["summary_text"])
    print(out[0]["summary_text"], "\n")

final_summary = " ".join(summaries)

with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)

print("--- Done! ---")
print("Summary saved to summary.txt")
print("\nFINAL SUMMARY:\n")
print(final_summary)