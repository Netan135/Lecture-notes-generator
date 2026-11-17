"""
Lecture-to-Notes Generator — Web UI
-----------------------------------
Run: python app.py
Then open the URL shown in the terminal (usually http://127.0.0.1:7860)
"""

import os
import re
from datetime import datetime
import gradio as gr

# --- Config ---
WHISPER_MODEL = "base"
SUMMARY_MODEL = "facebook/bart-large-cnn"
QG_MODEL = "valhalla/t5-small-e2e-qg"
MAX_CHUNK_CHARS = 2500
OUTPUT_DIR = "outputs"


def log(msg):
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"


# ============ PIPELINE (same logic as main.py) ============
def transcribe(audio_path):
    from faster_whisper import WhisperModel
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)

    lines = []
    full_text = ""
    for seg in segments:
        lines.append(f"[{seg.start:6.1f}s → {seg.end:6.1f}s]  {seg.text.strip()}")
        full_text += seg.text.strip() + " "
    return "\n".join(lines), full_text.strip()


def summarize(text):
    from transformers import pipeline
    summarizer = pipeline("summarization", model=SUMMARY_MODEL)

    chunks = [text[i:i + MAX_CHUNK_CHARS] for i in range(0, len(text), MAX_CHUNK_CHARS)]
    summaries = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=120, min_length=30, do_sample=False)
        summaries.append(out[0]["summary_text"])
    return " ".join(summaries)


def generate_questions(text):
    from transformers import pipeline
    qg = pipeline(
        "text2text-generation",
        model=QG_MODEL,
        tokenizer=QG_MODEL,
        use_fast=False,
    )

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    questions = []
    for sentence in sentences:
        prompt = "generate questions: " + sentence
        out = qg(prompt, max_length=64, num_beams=4, do_sample=False)
        for q in out[0]["generated_text"].split("</s>"):
            q = q.strip()
            if q and q not in questions:
                questions.append(q)
    return questions


# ============ GRADIO CALLBACK ============
def process_audio(audio_file, progress=gr.Progress()):
    if audio_file is None:
        return "❌ Please upload or record an audio file first.", "", "", ""

    try:
        audio_path = audio_file

        progress(0.05, desc="Loading Whisper and transcribing...")
        timed_transcript, full_text = transcribe(audio_path)

        progress(0.4, desc="Summarizing... (this takes a while on CPU)")
        summary = summarize(full_text)

        progress(0.75, desc="Generating study questions...")
        questions = generate_questions(full_text)

        progress(0.95, desc="Saving notes.md...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"notes_{timestamp}.md")

        questions_md = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Lecture Notes\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"## 📌 Summary\n\n{summary}\n\n")
            f.write(f"## ❓ Study Questions\n\n{questions_md}\n\n")
            f.write(f"## 📝 Full Transcript\n\n```\n{timed_transcript}\n```\n")

        progress(1.0, desc="Done!")

        status = f"✅ Done! Notes saved to `{out_path}`"
        return status, summary, questions_md, timed_transcript

    except Exception as e:
        return f"❌ Error: {e}", "", "", ""


# ============ BUILD THE UI ============
with gr.Blocks(title="Lecture-to-Notes Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Lecture-to-Notes Generator")
    gr.Markdown(
        "Drop an audio file (or record directly) and click **Generate Notes**. "
        "Everything runs locally on your machine — no internet, no API keys."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload or record lecture audio",
            )
            generate_btn = gr.Button("🚀 Generate Notes", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📌 Summary"):
                    summary_out = gr.Textbox(label="AI Summary", lines=8, interactive=False)
                with gr.TabItem("❓ Study Questions"):
                    questions_out = gr.Textbox(label="Generated Questions", lines=12, interactive=False)
                with gr.TabItem("📝 Full Transcript"):
                    transcript_out = gr.Textbox(label="Transcript with timestamps", lines=16, interactive=False)

    generate_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[status_box, summary_out, questions_out, transcript_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)