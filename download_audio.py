import subprocess
from pytube import YouTube
import librosa
import os
import soundfile

# Convert mp4 to wav
def convert_mp4_to_wav(mp4_path):
    root, ext = os.path.splitext(mp4_path)
    wav_path = root + '.wav'
    # Run ffmpeg command
    command = ['ffmpeg', '-y','-i', mp4_path, wav_path]
    subprocess.run(command, check=True)
    return wav_path

# Save YouTube video from url
def save_youtube_audio(url, filename):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio = True).first()
    filepath = os.path.join('audio', f'{filename}.mp4')
    mp4_path = video.download(filename = filepath)
    wav_path = convert_mp4_to_wav(mp4_path)
    os.remove(mp4_path)
    return wav_path

# Load audio from file
def load_audio_from_file(path, duration = 180):
    audio, sr = librosa.load(path, duration = duration)
    #audio = np.clip(audio, 0, 1)
    return audio, sr

# Save file to disk
def write_audio(path, y, sr):
    soundfile.write(path, y, sr)
