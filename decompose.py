#!/usr/bin/env python3

from pysndfx import AudioEffectsChain
from sys import argv, exit
import numpy as np
import soundfile
import librosa

BLEN = 1024
HLEN = 4096
NFFT = 4096

# TODO:

# Probably use tsurumeso/vocal-remover to separate into *good* vocal stem + *excellent* bg
#       -> side: probably just fork it and combine with our code to avoid unnecessary STFT etc

# Extract drums from BG, processing?
# Clean up vocal stem with EQ
# Then apply pitch shift/stretch operations to try and match tempo + key
# Combine samples (watch out for clipping) and run ISTFT

# GUI: PySide? 

# TODO2: other CNN-based clean-up tools for vocals


def lowpass(y, freq):
    flter = AudioEffectsChain().lowpass(frequency=freq).limiter(gain=5.0)
    return flter(y)

def highpass(y, freq):
    flter = AudioEffectsChain().highpass(frequency=freq).limiter(gain=5.0)
    return flter(y)


def decompose_hpcc(y, sr):
    y_for_vocals = highpass(lowpass(y, freq=15000), freq=90)

    S_full, phase = librosa.magphase(librosa.stft(y_for_vocals, center=True))

    # For rhythm, drums
    H, perc = librosa.decompose.hpss(librosa.stft(y, center=True), margin=(2.0,3.0))

    # Ugly, but a temp failsafe for shorter frames
    try: 
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2.5, sr=sr)))
    except:
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(0.1, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    # Extract vocals vs the rest
    mask_i = librosa.util.softmask(S_filter,
                                   4 * (S_full - S_filter),
                                   power=2)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   9 * S_filter,
                                   power=2)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    y_foreground = librosa.istft(S_foreground * phase)
    #y_foreground = lowpass(y_foreground, freq=2000)
    y_bass = librosa.istft(S_background * phase)

    # Try to clean up drums from the vocal part 
    # vocal_H, _ = librosa.decompose.hpss(S_foreground, margin=2.0)

    y_perc = librosa.istft(perc * phase)
    # y_vocal_clean = tg(S_foreground)

    return (y_perc,y_foreground,y_foreground,y_bass)


if __name__ == '__main__':
    audio_path = 'audio/feel.mp3' 
    if len(argv) > 1:
        audio_path = argv[1]

    sr = librosa.get_samplerate(audio_path)

    # Process audio in chunks
    stream = librosa.stream(audio_path,
                        block_length=BLEN,
                        frame_length=NFFT,
                        hop_length=HLEN,
                        duration=None,
                        mono=True)

    perc = []
    vocal = []
    fore = []
    back = []
    for y_block in stream:
        y_perc,y_vocal,y_fore,y_back = decompose_hpcc(y_block, sr)
        perc.extend(y_perc)
        fore.extend(y_fore)
        vocal.extend(y_vocal)
        back.extend(y_back)
    
    soundfile.write(f'{audio_path.split(".")[0]}_drums.wav', 
                       perc, 
                       44100, 
                       subtype='PCM_24')

    soundfile.write(f'{audio_path.split(".")[0]}_vocal.wav', 
                       vocal, 
                       44100, 
                       subtype='PCM_24')

    soundfile.write(f'{audio_path.split(".")[0]}_melody.wav', 
                       fore, 
                       44100, 
                       subtype='PCM_24')

    soundfile.write(f'{audio_path.split(".")[0]}_bass.wav', 
                       back, 
                       44100, 
                       subtype='PCM_24')
