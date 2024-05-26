#!/usr/bin/env python3

from pysndfx import AudioEffectsChain
from sys import argv, exit
import vocal_remover as vr
import numpy as np
import soundfile
import librosa

BLEN = 1024
HLEN = 4096
NFFT = 4096

# NB! Using vocal-remover code from: https://github.com/tsurumeso/vocal-remover.
# Modified to support mono based on: https://github.com/tsurumeso/vocal-remover/pull/144.
# Further modified to make our own custom calls and essentially merged with our code.



def lowpass(y, freq):
    flter = AudioEffectsChain().lowpass(frequency=freq).limiter(gain=5.0)
    return flter(y)

def highpass(y, freq):
    flter = AudioEffectsChain().highpass(frequency=freq).limiter(gain=5.0)
    return flter(y)


def decompose_hpcc(y, sr):
    y_for_vocals = highpass(lowpass(y, freq=15000), freq=90)

    S_full, phase = librosa.magphase(librosa.stft(y_for_vocals, center=True))
    _, perc = librosa.decompose.hpss(librosa.stft(y, center=True), margin=(1.0,2.0))

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
    y_background = librosa.istft(S_background * phase)
    y_percussion = librosa.istft(perc * phase)

    return (y_foreground,y_background,y_percussion)


def run_decomposer(audio, sr):
    print('[+] Opening file in chunks...')
    stream = [audio[i:i + BLEN] for i in range(0, len(audio), BLEN)]

    perc = []
    fore = []
    back = []
    print('[+] Decomposing...')
    for y_block in stream:
        y_fore,y_back,y_perc = decompose_hpcc(y_block, sr)
        perc.extend(y_perc)
        fore.extend(y_fore)
        back.extend(y_back)

    return (fore,back,perc)


def run_vocal_remover(audio_path, sr):
    print('[+] Calling vocal-remover...')
    y_vocal_w,y_back_w = vr.direct_call(audio_path, sr)

    print('[+] Saving results...')
    soundfile.write(f'{audio_path.split(".")[0]}_vocal_nn.wav', 
                       y_vocal_w.T, 
                       sr)

    soundfile.write(f'{audio_path.split(".")[0]}_backing_nn.wav', 
                       y_back_w.T, 
                       sr)


if __name__ == '__main__':
    audio_path = 'audio/feel.mp3' 
    if len(argv) > 1:
        audio_path = argv[1]

    # TODO keep track of audio names for processed parts 
    sr = librosa.get_samplerate(audio_path)

    run_vocal_remover(audio_path, sr)

    fname = audio_path.split(".")[0]
    run_decomposer(fname + '_backing_nn.wav', sr)

    print('[+] Done.')
