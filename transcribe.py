from faster_whisper import WhisperModel

# 1. Load the model (already downloaded, so this is fast now)
print("Loading model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

# 2. Transcribe the audio file
print("Transcribing... (please wait)")
segments, info = model.transcribe("lecture.wav", beam_size=5)

# 3. Print language and full text
print(f"\nDetected language: {info.language}")
print("\n--- Transcript ---\n")

full_text = ""
for segment in segments:
    print(f"[{segment.start:6.1f}s → {segment.end:6.1f}s]  {segment.text}")
    full_text += segment.text.strip() + " "

# 4. Save the full transcript to a .txt file
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(full_text.strip())

print("\n--- Done! ---")
print("Full transcript saved to transcript.txt")