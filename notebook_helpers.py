
from IPython.display import Audio

# Display audio in notebook
def display_audio(audio, sr = None):
    return Audio(data = audio, rate = sr)