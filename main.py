"""
Lecture-to-Notes Generator
--------------------------
Give it an audio file, get back a full study-notes markdown file:
  1. Full transcript (with timestamps)
  2. AI summary
  3. Study questions
"""

import os
import re
from datetime import datetime

# --- Config ---
AUDIO_FILE = "lecture.wav"      # change to your file name (.wav, .mp3, .m4a all work)
OUTPUT_FILE = "notes.md"
WHISPER_MODEL = "base"           # "tiny"=faster, "small"=more accurate
SUMMARY_MODEL = "facebook/bart-large-cnn"
QG_MODEL = "valhalla/t5-small-e2e-qg"
MAX_CHUNK_CHARS = 2500


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============ STEP 1: TRANSCRIBE ============
def transcribe(audio_path):
    from faster_whisper import WhisperModel
    log(f"Loading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    log(f"Transcribing {audio_path}... (this may take a while on CPU)")
    segments, info = model.transcribe(audio_path, beam_size=5)

    lines = []
    full_text = ""
    for seg in segments:
        lines.append(f"[{seg.start:6.1f}s → {seg.end:6.1f}s]  {seg.text.strip()}")
        full_text += seg.text.strip() + " "

    log(f"Transcription done. Detected language: {info.language}")
    return "\n".join(lines), full_text.strip()


# ============ STEP 2: SUMMARIZE ============
def summarize(text):
    from transformers import pipeline
    log(f"Loading summarizer '{SUMMARY_MODEL}'...")
    summarizer = pipeline("summarization", model=SUMMARY_MODEL)

    chunks = [text[i:i + MAX_CHUNK_CHARS] for i in range(0, len(text), MAX_CHUNK_CHARS)]
    log(f"Summarizing {len(chunks)} chunk(s)...")

    summaries = []
    for i, chunk in enumerate(chunks, start=1):
        log(f"  summary chunk {i}/{len(chunks)}")
        out = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summaries.append(out[0]["summary_text"])

    return " ".join(summaries)


# ============ STEP 3: GENERATE QUESTIONS ============
def generate_questions(text):
    from transformers import pipeline
    log(f"Loading question-generator '{QG_MODEL}'...")
    qg = pipeline(
        "text2text-generation",
        model=QG_MODEL,
        tokenizer=QG_MODEL,
        use_fast=False,
    )

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    log(f"Generating questions from {len(sentences)} sentence(s)...")

    questions = []
    for i, sentence in enumerate(sentences, start=1):
        log(f"  question from sentence {i}/{len(sentences)}")
        prompt = "generate questions: " + sentence
        out = qg(prompt, max_length=64, num_beams=4, do_sample=False)
        generated = out[0]["generated_text"]
        for q in generated.split("</s>"):
            q = q.strip()
            if q and q not in questions:
                questions.append(q)

    return questions


# ============ STEP 4: WRITE MARKDOWN ============
def write_notes(audio_path, timed_transcript, summary, questions):
    log(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Lecture Notes\n\n")
        f.write(f"**Source:** `{os.path.basename(audio_path)}`  \n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Whisper model:** {WHISPER_MODEL}\n\n")
        f.write("---\n\n")

        f.write("## 📌 Summary\n\n")
        f.write(summary + "\n\n")
        f.write("---\n\n")

        f.write("## ❓ Study Questions\n\n")
        for i, q in enumerate(questions, start=1):
            f.write(f"{i}. {q}\n")
        f.write("\n---\n\n")

        f.write("## 📝 Full Transcript (with timestamps)\n\n")
        f.write("```\n" + timed_transcript + "\n```\n")


# ============ MAIN ============
def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Audio file '{AUDIO_FILE}' not found in {os.getcwd()}")
        print("   Put your audio file in this folder, or edit AUDIO_FILE at the top of main.py.")
        return

    log(f"Starting pipeline for: {AUDIO_FILE}")

    timed_transcript, full_text = transcribe(AUDIO_FILE)
    summary = summarize(full_text)
    questions = generate_questions(full_text)
    write_notes(AUDIO_FILE, timed_transcript, summary, questions)

    log(f"✅ Done! Open {OUTPUT_FILE} to see your notes.")


if __name__ == "__main__":
    main()