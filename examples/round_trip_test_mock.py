import time
import wave

# -------------------------
# Mock LLM (token streaming)
# -------------------------
def mock_stream_chat(prompt: str):
    fake_response = "Hello! This is a simulated response from the megakernel LLM."
    
    for word in fake_response.split():
        time.sleep(0.1)  # simulate token delay
        yield word + " "


# -------------------------
# Mock TTS (audio streaming)
# -------------------------
def mock_stream_pcm_chunks(text_stream):
    for chunk in text_stream:
        # simulate audio chunk (fake bytes)
        time.sleep(0.05)
        yield b"\x00\x01" * 800  # fake PCM chunk


# -------------------------
# Save PCM → WAV
# -------------------------
def save_wav(filename, pcm_chunks):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(16000)

        for chunk in pcm_chunks:
            wf.writeframes(chunk)


# -------------------------
# Round Trip Test
# -------------------------
def main():
    prompt = "Hi, how are you?"

    print(f"\n🧠 Input: {prompt}\n")

    # Step 1: LLM streaming
    token_stream = mock_stream_chat(prompt)

    collected_text = ""
    
    def text_generator():
        nonlocal collected_text
        for token in token_stream:
            print(token, end="", flush=True)  # live token output
            collected_text += token
            yield token

    # Step 2: TTS streaming
    audio_stream = mock_stream_pcm_chunks(text_generator())

    print("\n\n🔊 Generating audio...")

    # Step 3: Save output
    save_wav("output.wav", audio_stream)

    print("✅ Saved to output.wav")


if __name__ == "__main__":
    main()