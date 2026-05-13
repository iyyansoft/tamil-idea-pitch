import numpy as np

def extract_features(text):
    words = text.split()

    features = [
        len(text),              # total characters
        len(words),             # word count
        np.mean([len(w) for w in words]) if words else 0,  # avg word length
    ]

    return np.array(features)