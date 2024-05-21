#!/usr/bin/env python3

from sys import argv, exit
import numpy as np
import soundfile
import librosa

BLEN = 1024
HLEN = 4096
NFFT = 4096


def decompose_hpcc(y, sr):
    D = librosa.stft(y, center=True)

    S_full, phase = librosa.magphase(D)

    # For rhythm.drums
    H, perc = librosa.decompose.hpss(D, margin=(1.0,2.2))

    # Ugly, but a temp failsafe for shorter frames
    try: 
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
    except:
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(0.1, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    # Extract vocals vs the rest
    margin_i, margin_v = 5, 9
    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=2)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=2)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # TODO cut out everything in 300-1200 Hz for Bass and apply lowpass
    # TODO low pass for vocals
    y_foreground = librosa.istft(S_foreground * phase)
    y_bass = librosa.istft(S_background * phase)

    # Try to clean up drums from the vocal part
    vocal_H, _ = librosa.decompose.hpss(S_foreground, margin=2.0)

    # TODO further filtering for drums, clean up with hpss?
    y_perc = librosa.istft(perc * phase)
    y_vocal = librosa.istft(vocal_H * phase)

    return (y_perc,y_vocal,y_foreground,y_bass)


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
                        duration=100,
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
