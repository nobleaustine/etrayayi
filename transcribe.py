import assemblyai as aai
import pyaudio
import wave

# Parameters
FORMAT = pyaudio.paInt16  # Format of the audio stream (16-bit PCM)
CHANNELS = 2              # Number of audio channels (stereo)
RATE = 44100              # Sampling rate (44.1 kHz)
CHUNK = 1024              # Size of each audio chunk
DURATION = 10             # Duration of recording in seconds
OUTPUT_FILENAME = "output.wav"  # Name of the output file

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open a new audio stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

# Initialize a list to store the audio frames
frames = []

# Record audio in chunks for the specified duration
for _ in range(0, int(RATE / CHUNK * DURATION)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording finished.")

# Stop and close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded audio as a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio recorded and saved as {OUTPUT_FILENAME}")



aai.settings.api_key = "7a49b75cca8848888025e11348252efd"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("./recorded_audio.wav")
# transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)