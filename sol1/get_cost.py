import editdistance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import assemblyai as aai
import pyaudio
import wave
from google.cloud import speech_v1p1beta1 as speech
import os
import io

from google.cloud import translate_v2 as translate
import os

def translate_it(text):
    # Set the environment variable for the service account key file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

    # Initialize the translation client
    translation_client = translate.Client()

    # Perform translation
    translated = translation_client.translate(
        text,
        source_language='ml',
        target_language='en'
    )

    # Return the translated text
    return translated['translatedText']

def record():
    # Parameters
    FORMAT = pyaudio.paInt16        # Format of the audio stream (16-bit PCM)
    CHANNELS = 1                    # Number of audio channels (stereo)
    RATE = 44100                    # Sampling rate (44.1 kHz)
    CHUNK = 1024                    # Size of each audio chunk
    DURATION = 10                   # Duration of recording in seconds
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

def transcribeG():


    # Set the path to your service account key file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

    # Path to the audio file
    audio_file_path = "output.wav"

    # Create a SpeechClient instance
    client = speech.SpeechClient()

    # Read the audio file
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    # Configure the audio settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ml-in",  # Set language code for Malayalam
    )

    # Perform the speech recognition
    response = client.recognize(config=config, audio=audio)

    # Print the transcribed text
    transcription = ''
    for result in response.results:
        transcription += result.alternatives[0].transcript + " "
    return transcription

def transcribe():
    aai.settings.api_key = "7a49b75cca8848888025e11348252efd"
    config = aai.TranscriptionConfig(language_code="ml")
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe("./output.wav")
    
    return transcript.text

def calculate_distances(word1, word2):
    # Levenshtein distance
    levenshtein_distance = editdistance.eval(word1, word2)
    
    # Jaccard similarity
    jaccard_similarity = len(set(word1).intersection(set(word2))) / len(set(word1).union(set(word2)))
    
    # Cosine similarity
    vectorizer = CountVectorizer().fit_transform([word1, word2])
    vectors = vectorizer.toarray()
    cosine_similarity_value = cosine_similarity(vectors)[0, 1]
    
    return levenshtein_distance+ jaccard_similarity+cosine_similarity_value


if __name__ == "__main__":
    # items = {"chaya":10,"kappi":12,"kaddi":7}
    # count = {"oru":1,"randu":2,"moonu":3,"nalu":4,"anchu":5}
    items = {"tea":10,"coffee":12,"snack":7}
    count = {"one":1,"two":2,"three":3,"four":4,"five":5}
    distances_dict_item = {}
    distances_dict_count = {}

    record()
    # text = transcribe()
    text = transcribeG()
    text = translate_it(text)
    # text = 'oru chaya randu kappi'
    print("Text: ",text)

    l = text.split(" ")
    item_l = l[1::2]
    count_l = l[::2]

    for word1 in item_l:
        distance ={word2:calculate_distances(word1,word2)  for word2 in items}
        min_key = min(distance, key=distance.get)
        distances_dict_item[word1] =items[min_key]
    

    for word1 in count_l:
        distance ={word2:calculate_distances(word1,word2)  for word2 in count}
        min_key = min(distance, key=distance.get)
        distances_dict_count[word1] =count[min_key]

    sum = 0
    for i in range(0,len(l)-1,2):
        sum = sum +  distances_dict_count[l[i]]*distances_dict_item[l[i+1]]
    print("Total amount: ",sum)
    
     
