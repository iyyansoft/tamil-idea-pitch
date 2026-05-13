import re
from collections import Counter

import numpy as np
import pandas as pd


# =========================================================
# OPTIONAL: CONNECT YOUR ORIGINAL DATASET BUILDER HERE
# =========================================================
# If you already have a function from training/dataset creation
# that generates the exact 43 features from text, import it here
# and return a dict.
#
# Example:
# from dataset_builder import extract_features_for_row
#
# def real_feature_dict(text: str) -> dict:
#     return extract_features_for_row(text)
#
# If you do not have that ready yet, keep USE_REAL_BUILDER = False
# and use the fallback builder below.
# =========================================================

USE_REAL_BUILDER = False


def real_feature_dict(text: str) -> dict:
    """
    Replace this with your ACTUAL feature builder from training.
    Must return a dictionary: {feature_name: value, ...}
    """
    raise NotImplementedError("Connect your real dataset-builder feature function here.")


# =========================================================
# FALLBACK FEATURE BUILDER
# =========================================================
# This does NOT magically recreate your exact training features.
# It is only a structured fallback so the code can align columns.
# Best result = connect the real training feature builder above.
# =========================================================

TAMIL_CHAR_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
ENGLISH_CHAR_PATTERN = re.compile(r"[A-Za-z]")
DIGIT_PATTERN = re.compile(r"\d")
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def safe_div(a, b):
    return a / b if b else 0.0


def get_words(text):
    return WORD_PATTERN.findall(text)


def count_syllable_like_units_tamil(text):
    tamil_chars = TAMIL_CHAR_PATTERN.findall(text)
    return len(tamil_chars)


def fallback_feature_dict(text: str) -> dict:
    words = get_words(text)
    word_lengths = [len(w) for w in words]
    char_count = len(text)
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    tamil_chars = len(TAMIL_CHAR_PATTERN.findall(text))
    english_chars = len(ENGLISH_CHAR_PATTERN.findall(text))
    digits = len(DIGIT_PATTERN.findall(text))
    spaces = text.count(" ")
    punctuation = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())

    counter = Counter(w.lower() for w in words)

    features = {
        "char_count": char_count,
        "word_count": word_count,
        "unique_word_count": unique_words,
        "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "max_word_length": max(word_lengths) if word_lengths else 0,
        "min_word_length": min(word_lengths) if word_lengths else 0,
        "std_word_length": float(np.std(word_lengths)) if word_lengths else 0.0,
        "tamil_char_count": tamil_chars,
        "english_char_count": english_chars,
        "digit_count": digits,
        "space_count": spaces,
        "punctuation_count": punctuation,
        "lexical_diversity": safe_div(unique_words, word_count),
        "tamil_ratio": safe_div(tamil_chars, char_count),
        "english_ratio": safe_div(english_chars, char_count),
        "digit_ratio": safe_div(digits, char_count),
        "punctuation_ratio": safe_div(punctuation, char_count),
        "long_word_count": sum(1 for w in words if len(w) >= 7),
        "short_word_count": sum(1 for w in words if len(w) <= 3),
        "question_mark_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "newline_count": text.count("\n"),
        "estimated_syllable_units": count_syllable_like_units_tamil(text),
        "repeat_word_count": sum(1 for _, c in counter.items() if c > 1),
        "top_word_freq": max(counter.values()) if counter else 0,
        "hapax_count": sum(1 for _, c in counter.items() if c == 1),
        "avg_repeat_freq": float(np.mean(list(counter.values()))) if counter else 0.0,
        "text_is_empty": int(not text.strip()),
    }

    # Add stable filler-style engineered features so the model input
    # can still be aligned to expected column names when needed.
    vowels_tamil_like = sum(1 for ch in text if ch in "அஆஇஈஉஊஎஏஐஒஓஔ")
    vowels_english = sum(1 for ch in text.lower() if ch in "aeiou")
    numbers_as_words_hint = sum(1 for w in words if any(ch.isdigit() for ch in w))

    extra = {
        "tamil_vowel_count": vowels_tamil_like,
        "english_vowel_count": vowels_english,
        "numbers_as_words_hint": numbers_as_words_hint,
        "uppercase_count": sum(1 for ch in text if ch.isupper()),
        "lowercase_count": sum(1 for ch in text if ch.islower()),
        "avg_sentence_length_words": safe_div(word_count, max(1, text.count(".") + text.count("?") + text.count("!"))),
        "avg_sentence_length_chars": safe_div(char_count, max(1, text.count(".") + text.count("?") + text.count("!"))),
        "symbol_count": sum(1 for ch in text if not ch.isalnum() and not ch.isspace() and ch not in ".,?!"),
        "non_tamil_non_english_count": sum(
            1 for ch in text
            if not TAMIL_CHAR_PATTERN.match(ch)
            and not ENGLISH_CHAR_PATTERN.match(ch)
            and not ch.isdigit()
            and not ch.isspace()
        ),
        "repeated_char_runs": len(re.findall(r"(.)\1{2,}", text)),
        "mixed_script_ratio": safe_div(tamil_chars + english_chars, char_count),
        "ascii_count": sum(1 for ch in text if ord(ch) < 128),
        "non_ascii_count": sum(1 for ch in text if ord(ch) >= 128),
    }

    features.update(extra)
    return features


def get_feature_dict(text: str) -> dict:
    if USE_REAL_BUILDER:
        return real_feature_dict(text)
    return fallback_feature_dict(text)


def align_to_model_features(feature_dict: dict, model) -> pd.DataFrame:
    expected_names = getattr(model, "feature_names_in_", None)

    if expected_names is None:
        # fallback if model has no stored feature names
        df = pd.DataFrame([feature_dict])
        return df

    aligned = {}
    for col in expected_names:
        aligned[col] = feature_dict.get(col, 0)

    return pd.DataFrame([aligned], columns=list(expected_names))


def build_model_input(text: str, model) -> pd.DataFrame:
    feature_dict = get_feature_dict(text)
    feature_df = align_to_model_features(feature_dict, model)
    return feature_df