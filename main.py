import tkinter as tk
from tkinter import filedialog, messagebox
from moviepy.editor import VideoFileClip
import pygame
import torch
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
import librosa
from gtts import gTTS
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
import os

def clone_voice(source_audio_path, target_voice_audio_path):
    # Placeholder for voice cloning
    # TODO: Implement voice cloning logic
    cloned_audio_path = "path_to_cloned_voice_audio.mp3"
    return cloned_audio_path

def select_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if video_path:
        label.config(text=f"File selected: {video_path}")

def extract_audio():
    global video_path, audio_path
    if video_path and video_path.lower().endswith('.mp4'):
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio_path = os.path.join(os.getcwd(), os.path.basename(video_path).rsplit('.', 1)[0] + '.wav')
            audio.write_audiofile(audio_path, codec='pcm_s16le')
            label.config(text=f"Audio extracted: {audio_path}")
        except Exception as e:
            label.config(text=f"Error: {str(e)}")
    else:
        label.config(text="Error: Please select an MP4 file.")

def play_audio():
    global audio_path
    if audio_path:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            label.config(text="Playing audio...")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio: {str(e)}")
    else:
        messagebox.showinfo("Info", "No audio file to play.")

def pause_audio():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
        label.config(text="Audio paused.")

def unpause_audio():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.unpause()
        label.config(text="Audio playing...")

def stop_audio():
    pygame.mixer.music.stop()
    label.config(text="Audio stopped.")

def translate_text(text):
    # Translate text using the Helsinki-NLP model
    translated_tokens = translator_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translated_ids = translator_model.generate(**translated_tokens)
    translated_text = translator_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
    return translated_text

def translate_audio(audio_path):
    if audio_path:
        # Load and process the audio file in chunks
        speech, sr = librosa.load(audio_path, sr=16000)

        # Apply noise reduction
        speech = nr.reduce_noise(y=speech, sr=sr)

        chunk_duration = 10  # seconds
        num_chunks = int(np.ceil(len(speech) / sr / chunk_duration))
        all_translations = []  # List to hold translations of all chunks

        for i in range(num_chunks):
            chunk = speech[i * chunk_duration * sr: (i+1) * chunk_duration * sr]
            input_values = processor(chunk, sampling_rate=sr, return_tensors="pt").input_values
            logits = wav2vec_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            # Translate the transcription
            translated_text = translate_text(transcription)

            all_translations.append(translated_text)

            print(f"Chunk {i+1}/{num_chunks}")
            print(f"English: {transcription}")
            print(f"Spanish: {translated_text}\n")

        # Combine all translations into a single text
        full_translation = " ".join(all_translations)

        # Synthesize the translated text into speech
        tts = gTTS(full_translation, lang='es')
        translated_audio_path = audio_path.rsplit('.', 1)[0] + '_translated.mp3'
        tts.save(translated_audio_path)

        label.config(text=f"Translated audio saved: {translated_audio_path}")

        # Clone and save the translated audio
        cloned_audio_path = clone_voice(audio_path, translated_audio_path)
        label.config(text=f"Cloned and translated audio saved: {cloned_audio_path}")

    else:
        label.config(text="No audio file to translate.")

def play_translated_audio():
    global translated_audio_path
    if translated_audio_path:
        try:
            pygame.mixer.music.load(translated_audio_path)
            pygame.mixer.music.play()
            label.config(text="Playing translated audio...")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play translated audio: {str(e)}")

def stop_translated_audio():
    pygame.mixer.music.stop()
    label.config(text="Translated audio stopped.")

# Create the main window
root = tk.Tk()
root.title("Video to Audio Converter")

# Create a label for instructions or messages
label = tk.Label(root, text="Select a video file to extract and translate its audio")
label.pack()

# Create a button to open the file dialog
button_select = tk.Button(root, text="Select Video File", command=select_file)
button_select.pack()

# Create a submit button to start the extraction process
button_submit = tk.Button(root, text="Extract Audio", command=extract_audio)
button_submit.pack()

# Create buttons for audio control
button_play = tk.Button(root, text="Play Audio", command=play_audio)
button_play.pack()

button_pause = tk.Button(root, text="Pause Audio", command=pause_audio)
button_pause.pack()

button_unpause = tk.Button(root, text="Unpause Audio", command=unpause_audio)
button_unpause.pack()

button_stop = tk.Button(root, text="Stop Audio", command=stop_audio)
button_stop.pack()

# Create a translate button
button_translate = tk.Button(root, text="Translate Audio", command=lambda: translate_audio(audio_path))
button_translate.pack()

button_play_translated = tk.Button(root, text="Play Translated Audio", command=play_translated_audio)
button_play_translated.pack()

button_stop_translated = tk.Button(root, text="Stop Translated Audio", command=stop_translated_audio)
button_stop_translated.pack()

# Initialize the video and audio paths
video_path = ""
audio_path = ""
translated_audio_path = ""

# Initialize Pygame mixer
pygame.mixer.init()

# Initialize models and processors
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
translator_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
translator_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')

# Run the application
root.mainloop()
