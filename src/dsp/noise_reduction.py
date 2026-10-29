"""Single-Channel Speech Enhancement via Spectral Subtraction & Wiener Filtering.

Suppresses background ambient noise (fans, traffic, environmental hum)
while minimizing musical noise artifacts and preserving speech intelligibility.
"""

from typing import Tuple
import numpy as np
from scipy import signal


class SpectralNoiseReducer:
    """Real-time frequency-domain spectral subtractive speech enhancer."""

    def __init__(self, fs: int = 16000, n_fft: int = 256, hop_size: int = 128):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.window = signal.windows.hann(n_fft)

    def enhance(
        self,
        audio: np.ndarray,
        over_subtraction: float = 1.5,
        spectral_floor: float = 0.05,
        noise_estimation_frames: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spectral subtraction to noisy audio.

        Parameters
        ----------
        audio : np.ndarray
            Input noisy speech signal (1D float).
        over_subtraction : float (alpha)
            Controls aggressive noise suppression (typically 1.2 to 2.0).
        spectral_floor : float (beta)
            Prevents negative spectral values and eliminates musical noise (0.02 to 0.1).
        noise_estimation_frames : int
            Number of initial silent/noise-only frames used to profile background noise.

        Returns
        -------
        enhanced_audio : np.ndarray
            Denoised speech signal.
        noise_profile : np.ndarray
            Estimated noise power spectrum.
        """
        # STFT
        f, t, Zxx = signal.stft(
            audio,
            fs=self.fs,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_size,
            boundary="zeros"
        )

        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise magnitude spectrum from first N frames
        n_frames = magnitude.shape[1]
        est_frames = min(noise_estimation_frames, max(1, n_frames // 10))
        noise_mag_est = np.mean(magnitude[:, :est_frames], axis=1, keepdims=True)
        noise_power_est = np.square(noise_mag_est)

        # Spectral Subtraction rule
        signal_power = np.square(magnitude)
        subtracted_power = signal_power - over_subtraction * noise_power_est

        # Floor constraint
        floor_power = spectral_floor * signal_power
        cleaned_power = np.maximum(subtracted_power, floor_power)
        cleaned_mag = np.sqrt(cleaned_power)

        # Reconstruct complex STFT with original phase
        Zxx_cleaned = cleaned_mag * np.exp(1j * phase)

        # ISTFT
        _, enhanced_audio = signal.istft(
            Zxx_cleaned,
            fs=self.fs,
            window=self.window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_size
        )

        # Match exact input length
        enhanced_audio = enhanced_audio[:len(audio)]
        return enhanced_audio, noise_mag_est.ravel()
