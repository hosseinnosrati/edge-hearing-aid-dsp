"""Adaptive Acoustic Feedback Cancellation (AFC) for Digital Hearing Aids.

Implements the Normalized Least Mean Squares (NLMS) adaptive filter to model
and cancel acoustic leakage paths between loudspeaker and microphone,
preventing unstable howling and whistling oscillations.
"""

from typing import Tuple
import numpy as np


class AdaptiveFeedbackCanceller:
    """NLMS Adaptive Filter for real-time Acoustic Feedback Cancellation."""

    def __init__(self, filter_order: int = 64, step_size: float = 0.05, regularization: float = 1e-4):
        """Parameters

        ----------
        filter_order : int
            Number of FIR filter taps modeling the acoustic feedback path.
        step_size : float (mu)
            Adaptation step size (learning rate) in (0, 2).
        regularization : float (epsilon)
            Small constant to prevent division by zero in low-energy regions.
        """
        self.order = filter_order
        self.mu = step_size
        self.eps = regularization
        self.weights = np.zeros(filter_order, dtype=np.float32)
        self.buffer = np.zeros(filter_order, dtype=np.float32)

    def process_sample(self, mic_sample: float, speaker_sample: float) -> Tuple[float, float]:
        """Process a single sample through NLMS cancellation.

        Parameters
        ----------
        mic_sample : float
            Input from hearing aid microphone (contains speech + feedback).
        speaker_sample : float
            Output sent to loudspeaker (reference signal driving feedback).

        Returns
        -------
        error_sample : float
            Cleaned speech signal with feedback component subtracted.
        est_feedback : float
            Estimated acoustic feedback signal.
        """
        # Shift buffer and insert new speaker sample
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = speaker_sample

        # Predict feedback
        est_feedback = float(np.dot(self.weights, self.buffer))

        # Error signal (cancelled output)
        error = mic_sample - est_feedback

        # NLMS Weight update
        norm = float(np.dot(self.buffer, self.buffer)) + self.eps
        self.weights += (self.mu / norm) * error * self.buffer

        return error, est_feedback

    def process_block(self, mic_audio: np.ndarray, speaker_audio: np.ndarray) -> np.ndarray:
        """Process an entire audio buffer."""
        cleaned = np.zeros_like(mic_audio)
        for i in range(len(mic_audio)):
            c_sample, _ = self.process_sample(mic_audio[i], speaker_audio[i])
            cleaned[i] = c_sample
        return cleaned
