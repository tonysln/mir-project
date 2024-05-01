#!/usr/bin/env python3

from sys import argv, exit
import numpy as np
import soundfile
import librosa

HLEN = 1024
NFFT = 2048


def get_inverse_mel(y, sr, n=64, fmin=20, fmax=8000):
    # Calculate power spectogram
    D = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HLEN, center=False))**2

    # Calculate mel-scaled spectrogram
    mel = librosa.feature.melspectrogram(S=D, sr=sr, 
                                         n_fft=NFFT, hop_length=HLEN,
                                         center=False,
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

    sr = librosa.get_samplerate(audio_path)

    # Process audio in chunks
    stream = librosa.stream(audio_path,
                        block_length=256,
                        frame_length=NFFT,
                        hop_length=HLEN,
                        duration=None) # None for full

    audio_from_mel = []
    for y_block in stream:
        audio_from_mel.extend(get_inverse_mel(y_block, sr, n=128, fmax=sr/2))
    
    soundfile.write(f'{audio_path.split(".")[0]}_restored.wav', 
                       audio_from_mel, 
                       22050, 
                       subtype='PCM_24')
