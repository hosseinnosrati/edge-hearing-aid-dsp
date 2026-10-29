"""Visual & Audio Demo: Hearing Aid Speech Enhancement, Equalization, and AFC.

Generates:
1. Processed audio file: 'data/hearing_aid_processed_output.wav'
2. Publication-quality figure: 'assets/demo_hearing_aid_spectrogram.png'
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

from src.audio_generator import generate_benchmark_speech
from src.pipeline import HearingAidPipeline
from src.dsp.filterbank import STANDARD_FREQUENCIES, AUDIOGRAM_PRESETS


def run_demo():
    print("Running Digital Hearing Aid DSP Demo...")
    raw_audio_path = "data/benchmark_speech_noisy.wav"
    if not os.path.exists(raw_audio_path):
        generate_benchmark_speech(duration_sec=4.0, fs=16000)

    fs, audio_int16 = wavfile.read(raw_audio_path)
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Speaker leakage reference
    speaker_ref = audio_float * 0.25

    # Process through pipeline
    pipeline = HearingAidPipeline(fs=fs)
    res = pipeline.process(
        mic_audio=audio_float,
        speaker_reference=speaker_ref,
        audiogram_preset="Moderate_High_Frequency_Loss"
    )

    clean_out = res["final_output"]

    # Save output WAV file
    os.makedirs("data", exist_ok=True)
    out_wav_path = "data/hearing_aid_processed_output.wav"
    wav_int16 = (clean_out * 32767.0).astype(np.int16)
    wavfile.write(out_wav_path, fs, wav_int16)
    print(f"Enhanced audio saved to: {out_wav_path}")

    # Plot 3-panel spectrogram and waveform comparison
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [1.2, 0.8, 1.2]})
    plt.subplots_adjust(hspace=0.35)

    duration_sec = len(audio_float) / fs
    t = np.linspace(0, duration_sec, len(audio_float), endpoint=False)

    # Panel 1: Input Noisy Spectrogram
    f_in, t_in, Sxx_in = signal.spectrogram(audio_float, fs=fs, nperseg=256, noverlap=128)
    im1 = axes[0].pcolormesh(t_in, f_in, 10 * np.log10(Sxx_in + 1e-8), shading="gouraud", cmap="inferno")
    axes[0].set_title("Stage 1: Input Speech Corrupted by Ambient Noise & Acoustic Feedback Howling", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Frequency (Hz)", fontsize=10)
    axes[0].set_ylim(0, 8000)
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.01)
    cbar1.set_label("Power (dB)", fontsize=9)

    # Panel 2: Audiogram Hearing Loss & Equalization Curve
    freqs = STANDARD_FREQUENCIES
    loss_db = [AUDIOGRAM_PRESETS["Moderate_High_Frequency_Loss"][f] for f in freqs]
    applied_gains = res["applied_gains_db"]

    axes[1].plot(freqs, loss_db, "r--o", lw=2, label="Patient Hearing Loss (dB HL) [Presbycusis]")
    axes[1].plot(freqs, applied_gains, "g-s", lw=2, label="Hearing Aid Prescribed Gain (NAL-R Half-Gain Rule)")
    axes[1].set_title("Stage 2: Clinical Audiogram Profile & Multi-Band Insertion Gains", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Audiometric Frequency (Hz)", fontsize=10)
    axes[1].set_ylabel("Level (dB)", fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xticks(freqs)
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    # Panel 3: Enhanced Output Spectrogram
    f_out, t_out, Sxx_out = signal.spectrogram(clean_out, fs=fs, nperseg=256, noverlap=128)
    im3 = axes[2].pcolormesh(t_out, f_out, 10 * np.log10(Sxx_out + 1e-8), shading="gouraud", cmap="viridis")
    axes[2].set_title("Stage 3: Processed Output (Denoised, Whistling Eliminated, High Frequencies Restored)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].set_ylabel("Frequency (Hz)", fontsize=10)
    axes[2].set_ylim(0, 8000)
    cbar3 = plt.colorbar(im3, ax=axes[2], pad=0.01)
    cbar3.set_label("Power (dB)", fontsize=9)

    os.makedirs("assets", exist_ok=True)
    out_png = "assets/demo_hearing_aid_spectrogram.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Demo plot successfully saved to: {out_png}")


if __name__ == "__main__":
    run_demo()
