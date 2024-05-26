import librosa
from pymusickit.key_finder import KeyFinder
from collections import defaultdict
import numpy as np


# Extract layers (old)
def extract_fore_and_background(audio, sample_rate, margin_i = 2, margin_v = 10, power = 2):
    audio = np.clip(audio, 0, 1)
    S_full, phase = librosa.magphase(librosa.stft(audio))
    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sample_rate)))
    S_filter = np.minimum(S_full, S_filter)
    mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    y_foreground = librosa.istft(S_foreground * phase)
    y_background = librosa.istft(S_background * phase)

    return(y_foreground, y_background)

# Define key changes
pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
possible_keys = [*[f"{p} major" for p in pitches], *[f"{p} minor" for p in pitches],]
possible_keys.index("C minor")


key_to_key_pitch_change = defaultdict(dict)
for i in range(len(possible_keys)):
    source_key = possible_keys[i]
    source_key_type = source_key.split()[1]
    for j in range(len(possible_keys)):
        target_key = possible_keys[j]
        target_key_type = target_key.split()[1]

        diff = j-i
        if source_key_type == "minor" and target_key_type == "major":
            diff-=3
        if source_key_type == "major" and target_key_type == "minor":
            diff+=3

        diff = diff % 12
        diff = diff if diff<=abs(diff-12) else diff-12
        key_to_key_pitch_change[source_key][target_key] = diff

# Shift audio to match key       
def shift_to_match_target_key(source_song, target_song, source_audio, source_sr):
    # Source - will be changed
    # Target - base for changing
    n_steps = key_to_key_pitch_change[source_song.key_primary][target_song.key_primary]
    shifted_audio = librosa.effects.pitch_shift(y = source_audio, sr = source_sr, n_steps = n_steps)
    return shifted_audio

# Change song tempo
def match_tempo(source_audio, source_sr, target_audio, target_sr):
    # Source audio = source foreground if you want to modify vocals
    source_tempo, _ =  librosa.beat.beat_track(y = source_audio, sr = source_sr)
    target_tempo, _ =  librosa.beat.beat_track(y = target_audio, sr = target_sr)
    rate = np.round(source_tempo[0] / target_tempo[0], 1)
    return librosa.effects.time_stretch(source_audio, rate = rate)


def slider_to_db(slider_value, min_db=-40.0, max_db=0.0):
    db_value = slider_value * (max_db - min_db) + min_db
    return db_value

def db_to_amplitude(db_value):
    amplitude = 10 ** (db_value / 20.0)
    return amplitude

def calculate_rms(y,scale=None):
    rms = np.sqrt(np.mean(np.square(y)))

    if scale:
        db_value = slider_to_db(scale)
        amp_mult = db_to_amplitude(db_value)
        rms = rms * amp_mult
    
    return rms

# Put audio layers together
def combine_audio_layers(audio_list, vol_list):
    lengths = [len(audio) for audio in audio_list]
    length = min(lengths)
    
    if len(audio_list) == len(vol_list):
        combined = audio_list[0][:length] * calculate_rms(audio_list[0], scale=vol_list[0])

        # Set volume level of each audio file from given list
        for audio,vol in zip(audio_list[1:],vol_list[1:]):
            adjusted_audio = audio * calculate_rms(audio, scale=vol)
            combined = combined + adjusted_audio[:length]
    else:
        # Set volume level based on first audio file
        combined = audio_list[0][:length]
        target_rms = calculate_rms(audio_list[0]) # Choose first layer as target
        for audio in audio_list[1:]:
            rate = target_rms / calculate_rms(audio) # Set volume
            adjusted_audio = audio * rate
            combined = combined + adjusted_audio[:length]
    return combined / np.max(np.abs(combined))