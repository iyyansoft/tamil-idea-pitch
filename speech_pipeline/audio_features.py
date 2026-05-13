# dataset_builder/audio_features.py

import numpy as np
import librosa


def extract_audio_features(audio_path: str, target_sr: int = 16000, n_mfcc: int = 13) -> dict:
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("Empty audio")

    duration_sec = librosa.get_duration(y=y, sr=sr)

    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
    except Exception:
        tempo = 0.0

    # New features
    silence_ratio = float(np.mean(np.abs(y) < 0.01))
    speech_rate_proxy = float(len(y) / sr / duration_sec) if duration_sec > 0 else 0.0

    features = {
        "duration_sec": float(duration_sec),
        "sample_rate": int(sr),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
        "tempo": tempo,
        "silence_ratio": silence_ratio,
        "speech_rate_proxy": speech_rate_proxy,
        "mfcc_mean": np.mean(mfcc, axis=1).astype(float).tolist(),
        "mfcc_std": np.std(mfcc, axis=1).astype(float).tolist()
    }

    return features