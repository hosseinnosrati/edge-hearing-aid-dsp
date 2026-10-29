"""Edge-Hearing-Aid-DSP: Interactive Web Application.

Run locally using:
    streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

from src.audio_generator import generate_benchmark_speech
from src.pipeline import HearingAidPipeline
from src.dsp.filterbank import STANDARD_FREQUENCIES, AUDIOGRAM_PRESETS


st.set_page_config(
    page_title="Edge Hearing Aid DSP",
    page_icon="🦻",
    layout="wide"
)

st.title("🦻 Edge-Hearing-Aid-DSP")
st.markdown(
    "**Real-Time Speech Enhancement, Feedback Cancellation, and Multi-Band Audiogram Equalization** | "
    "*[GitHub Repository](https://github.com/hnosrati0-source/edge-hearing-aid-dsp)*"
)
st.caption("Engineered for ultra-low latency (<10 ms budget) on resource-constrained embedded DSP hardware.")

# Sidebar Controls
st.sidebar.header("🎛️ Clinical Audiogram & Fitting")

preset = st.sidebar.selectbox(
    "Audiogram Prescription Profile:",
    options=list(AUDIOGRAM_PRESETS.keys()),
    index=2
)

st.sidebar.subheader("Frequency Thresholds (dB Hearing Level):")
custom_gains = []
for f in STANDARD_FREQUENCIES:
    default_val = float(AUDIOGRAM_PRESETS[preset][f])
    loss = st.sidebar.slider(f"{f} Hz Loss (dB):", min_value=0.0, max_value=90.0, value=default_val, step=5.0)
    # NAL-R Half-gain rule
    custom_gains.append(0.5 * loss)

st.sidebar.subheader("DSP Processing Modules")
enable_afc = st.sidebar.checkbox("Acoustic Feedback Cancellation (NLMS)", value=True)
enable_denoise = st.sidebar.checkbox("Spectral Subtraction Noise Reduction", value=True)
enable_wdrc = st.sidebar.checkbox("Wide Dynamic Range Compression (WDRC)", value=True)

# Audio preparation
raw_audio_path = "data/benchmark_speech_noisy.wav"
if not os.path.exists(raw_audio_path):
    generate_benchmark_speech(duration_sec=4.0, fs=16000)

fs, audio_int16 = wavfile.read(raw_audio_path)
audio_float = audio_int16.astype(np.float32) / 32768.0
speaker_ref = audio_float * 0.25

# Process
pipeline = HearingAidPipeline(fs=fs)
res = pipeline.process(
    mic_audio=audio_float,
    speaker_reference=speaker_ref,
    custom_gains=custom_gains,
    enable_afc=enable_afc,
    enable_denoiser=enable_denoise,
    enable_wdrc=enable_wdrc
)
output_float = res["final_output"]

# Save processed temp audio for player
out_wav_path = "data/temp_streamlit_output.wav"
wavfile.write(out_wav_path, fs, (output_float * 32767.0).astype(np.int16))

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔊 Audio Playback & Spectrograms",
    "📊 Clinical Audiogram & NAL-R Fitting",
    "💻 Embedded C99 Firmware & Benchmarks"
])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Input Noisy Speech (With Feedback)")
        st.audio(raw_audio_path, format="audio/wav")
        st.caption("Contains speech + ambient pink noise + 60 Hz hum + acoustic whistling.")

    with col_b:
        st.subheader("2. Hearing Aid Enhanced Output")
        st.audio(out_wav_path, format="audio/wav")
        st.caption("Denoised, howling cancelled, high-frequency speech consonants restored.")

    # Spectrogram Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    f1, t1, Sxx1 = signal.spectrogram(audio_float, fs=fs, nperseg=256, noverlap=128)
    ax1.pcolormesh(t1, f1, 10 * np.log10(Sxx1 + 1e-8), shading="gouraud", cmap="inferno")
    ax1.set_title("Input Spectrogram (Noise & Feedback Leakage)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(0, 8000)

    f2, t2, Sxx2 = signal.spectrogram(output_float, fs=fs, nperseg=256, noverlap=128)
    ax2.pcolormesh(t2, f2, 10 * np.log10(Sxx2 + 1e-8), shading="gouraud", cmap="viridis")
    ax2.set_title("Output Spectrogram (Cleaned & Frequency-Boosted)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, 8000)

    st.pyplot(fig)

with tab2:
    st.subheader("Audiogram Curve & Insertion Gains")
    st.markdown("Visualizes prescribed frequency boost based on patient clinical hearing thresholds.")

    fig_audio, ax_audio = plt.subplots(figsize=(9, 4))
    losses = [2.0 * g for g in custom_gains]

    ax_audio.plot(STANDARD_FREQUENCIES, losses, "r--o", lw=2, label="Patient Hearing Loss (dB HL)")
    ax_audio.plot(STANDARD_FREQUENCIES, custom_gains, "g-s", lw=2, label="Prescribed Insertion Gain (NAL-R)")
    ax_audio.set_xscale("log")
    ax_audio.set_xticks(STANDARD_FREQUENCIES)
    ax_audio.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_audio.set_xlabel("Frequency (Hz)")
    ax_audio.set_ylabel("Decibels (dB)")
    ax_audio.set_ylim(-10, 100)
    ax_audio.invert_yaxis()  # Standard audiogram convention (0 dB at top)
    ax_audio.grid(True, linestyle="--", alpha=0.6)
    ax_audio.legend(loc="upper right")
    st.pyplot(fig_audio)

    st.info("💡 **Clinical Principle:** Notice how 2000–6000 Hz frequencies receive the largest amplification to compensate for age-related sensorineural presbycusis.")

with tab3:
    st.subheader("Embedded ANSI C99 Firmware")
    st.markdown("Zero-malloc, sample-by-sample real-time code ready for flashing on ARM Cortex-M, STM32, or TI DSPs.")

    header_path = "models/hearing_aid_dsp.h"
    if os.path.exists(header_path):
        with open(header_path, "r", encoding="utf-8") as f:
            c_code = f.read()

        st.download_button(
            label="📥 Download hearing_aid_dsp.h",
            data=c_code,
            file_name="hearing_aid_dsp.h",
            mime="text/x-c"
        )
        st.code(c_code[:1800] + "\n\n/* ... (truncated preview) ... */", language="c")

    st.subheader("Edge Hardware Latency Benchmark (Dell Inspiron 1525)")
    st.table({
        "Metric": ["Sampling Frequency", "Frame Latency (128 samples)", "Latency Budget", "RAM Footprint", "C Header Size"],
        "Value": ["16 kHz (Wideband Speech)", "0.805 ms / frame", "< 10.0 ms (Echo-free)", "< 95 MB", "4.0 KB"]
    })
