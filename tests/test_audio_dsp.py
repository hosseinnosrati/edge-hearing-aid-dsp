"""Unit tests for Hearing Aid Audio DSP components."""

import numpy as np
from src.dsp.filterbank import AudiogramFilterbank, STANDARD_FREQUENCIES
from src.dsp.wdrc import WDRCCompressor
from src.dsp.adaptive_afc import AdaptiveFeedbackCanceller
from src.dsp.noise_reduction import SpectralNoiseReducer
from src.pipeline import HearingAidPipeline


def test_hearing_aid_dsp_pipeline():
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 1. Test Filterbank
    fb = AudiogramFilterbank(fs=fs)
    sine_1khz = np.sin(2 * np.pi * 1000 * t)
    gains = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # 10dB gain at 1kHz
    fb_out = fb.process(sine_1khz, gains)
    assert len(fb_out) == len(sine_1khz)
    print("Filterbank test passed.")

    # 2. Test WDRC Compressor
    wdrc = WDRCCompressor(fs=fs, compression_ratio=2.0)
    loud_audio = 0.9 * np.sin(2 * np.pi * 500 * t)
    compressed = wdrc.process(loud_audio)
    assert np.max(np.abs(compressed)) <= 1.0
    print("WDRC test passed.")

    # 3. Test Adaptive Feedback Canceller
    afc = AdaptiveFeedbackCanceller(filter_order=32, step_size=0.05)
    ref = np.random.randn(len(t)) * 0.1
    mic = ref * 0.5 + np.sin(2 * np.pi * 400 * t) * 0.2  # speech + 50% feedback
    cleaned = afc.process_block(mic, ref)
    assert len(cleaned) == len(mic)
    print("AFC test passed.")

    # 4. Test Spectral Noise Reducer
    denoiser = SpectralNoiseReducer(fs=fs)
    noisy = np.sin(2 * np.pi * 300 * t) + np.random.randn(len(t)) * 0.3
    denoised, _ = denoiser.enhance(noisy)
    assert len(denoised) == len(noisy)
    print("Noise reduction test passed.")

    # 5. Test Full Pipeline
    pipeline = HearingAidPipeline(fs=fs)
    res = pipeline.process(mic_audio=mic, speaker_reference=ref)
    assert "final_output" in res
    print("Full pipeline test passed successfully!")


if __name__ == "__main__":
    test_hearing_aid_dsp_pipeline()
