#!/usr/bin/env python3

from sys import argv, exit
import numpy as np
import soundfile
import librosa


def get_inverse_mel(y, sr, n=64, fmin=20, fmax=8000):
    # Calculate power spectogram
    D = np.abs(librosa.stft(y))**2

    # Calculate mel-scaled spectrogram
    mel = librosa.feature.melspectrogram(S=D, 
                                         sr=sr, 
                                         n_mels=n,  # bands
                                         fmin=fmin, # min freq
                                         fmax=fmax) # max freq

    # Now generate audio from the spectogram
    audio_from_mel = librosa.feature.inverse.mel_to_audio(mel)

    return audio_from_mel


if __name__ == '__main__':
    audio_path = 'audio/feel.mp3' # pitch shifted up a bit just in case 
    if len(argv) > 1:
        audio_path = argv[1]

    y, sr = librosa.load(audio_path, duration=10)

    # Process audio in chunks
    audio_from_mel = []
    step = 2048
    for i in range(0, len(y), step):
        audio_from_mel.extend(get_inverse_mel(y[i:i+step], sr, n=64))
    
    soundfile.write(f'{audio_path.split(".")[0]}_restored.wav', 
                       audio_from_mel, 
                       22050, 
                       subtype='PCM_24')
