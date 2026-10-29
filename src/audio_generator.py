"""Acoustic Test Signal Generator for Hearing Aid Benchmarking.

Synthesizes realistic speech phonemes (voiced vowels + unvoiced fricatives)
corrupted by ambient noise and acoustic feedback leakage.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy import signal


def generate_benchmark_speech(
    duration_sec: float = 5.0,
    fs: int = 16000,
    snr_db: float = 10.0,
    feedback_attenuation: float = 0.25,
    feedback_delay_ms: float = 12.0
) -> str:
    """Generate and save benchmark speech audio for hearing aid testing."""
    os.makedirs("data", exist_ok=True)
    out_path = "data/benchmark_speech_noisy.wav"

    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Synthesize Formant Speech (simulated vowel /a/ and /i/ syllables)
    f0 = 130.0  # Fundamental frequency (Hz)
    glottal_pulse = signal.sawtooth(2 * np.pi * f0 * t, width=0.1)

    # Resonant formant filters for vowel /a/: F1=700Hz, F2=1200Hz, F3=2600Hz
    nyq = 0.5 * fs
    b1, a1 = signal.butter(2, [600 / nyq, 800 / nyq], btype="band")
    b2, a2 = signal.butter(2, [1100 / nyq, 1300 / nyq], btype="band")
    b3, a3 = signal.butter(2, [2400 / nyq, 2800 / nyq], btype="band")

    formants = (
        signal.lfilter(b1, a1, glottal_pulse) * 1.0 +
        signal.lfilter(b2, a2, glottal_pulse) * 0.7 +
        signal.lfilter(b3, a3, glottal_pulse) * 0.4
    )

    # Syllabic amplitude envelope (speech rhythm ~3 Hz)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t - np.pi / 2))
    envelope = np.clip(envelope, 0.05, 1.0) ** 2
    clean_speech = formants * envelope

    # Normalize clean speech
    clean_speech = clean_speech / (np.max(np.abs(clean_speech)) + 1e-6) * 0.6

    # 2. Add Ambient Noise (Pink noise + 60Hz hum)
    white_noise = np.random.randn(n_samples)
    b_pink, a_pink = signal.butter(1, 0.2, btype="low")
    pink_noise = signal.lfilter(b_pink, a_pink, white_noise)
    hum = 0.15 * np.sin(2 * np.pi * 60.0 * t)
    total_noise = pink_noise + hum

    # Scale noise for desired SNR
    speech_power = np.mean(clean_speech ** 2)
    noise_power = np.mean(total_noise ** 2)
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = total_noise * np.sqrt(target_noise_power / (noise_power + 1e-8))

    noisy_speech = clean_speech + scaled_noise

    # 3. Simulate Acoustic Feedback Leakage (Speaker -> Mic)
    delay_samples = int(feedback_delay_ms * 1e-3 * fs)
    feedback_signal = np.zeros_like(noisy_speech)
    if delay_samples < n_samples:
        feedback_signal[delay_samples:] = clean_speech[:-delay_samples] * feedback_attenuation

    # Final corrupted microphone audio
    mic_audio = np.clip(noisy_speech + feedback_signal, -0.99, 0.99)

    # Save to 16-bit PCM WAV
    audio_int16 = (mic_audio * 32767.0).astype(np.int16)
    wavfile.write(out_path, fs, audio_int16)
    print(f"Generated benchmark audio: {out_path} ({duration_sec}s, fs={fs}Hz)")
    return out_path


if __name__ == "__main__":
    generate_benchmark_speech()
