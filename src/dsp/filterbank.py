"""Multi-Band Audiogram Filterbank for Digital Hearing Aids.

Splits incoming speech into standard audiometric clinical bands:
[250 Hz, 500 Hz, 1000 Hz, 2000 Hz, 4000 Hz, 6000 Hz]
and applies prescribed frequency-selective gains based on patient audiograms (presbycusis).
"""

from typing import List, Tuple, Dict
import numpy as np
from scipy import signal


# Standard clinical audiogram center frequencies (Hz)
STANDARD_FREQUENCIES = [250, 500, 1000, 2000, 4000, 6000]

# Standard Presbycusis Profiles (Hearing loss in dB HL)
AUDIOGRAM_PRESETS: Dict[str, Dict[int, float]] = {
    "Normal": {250: 0.0, 500: 0.0, 1000: 0.0, 2000: 0.0, 4000: 0.0, 6000: 0.0},
    "Mild_Presbycusis": {250: 10.0, 500: 15.0, 1000: 20.0, 2000: 30.0, 4000: 45.0, 6000: 50.0},
    "Moderate_High_Frequency_Loss": {250: 15.0, 500: 20.0, 1000: 30.0, 2000: 45.0, 4000: 65.0, 6000: 75.0},
    "Severe_Sloping_Loss": {250: 25.0, 500: 35.0, 1000: 50.0, 2000: 70.0, 4000: 85.0, 6000: 90.0},
}


class AudiogramFilterbank:
    """IIR Filterbank with prescribed insertion gains."""

    def __init__(self, fs: int = 16000, order: int = 2):
        self.fs = fs
        self.order = order
        self.bands = [
            (100, 350),      # ~250 Hz
            (350, 750),      # ~500 Hz
            (750, 1500),     # ~1000 Hz
            (1500, 3000),    # ~2000 Hz
            (3000, 5000),    # ~4000 Hz
            (5000, 7800),    # ~6000 Hz
        ]
        self.sos_filters = []
        self._design_filters()

    def _design_filters(self):
        nyq = 0.5 * self.fs
        for low, high in self.bands:
            high_clamped = min(high, nyq - 50.0)
            low_norm = low / nyq
            high_norm = high_clamped / nyq
            sos = signal.butter(self.order, [low_norm, high_norm], btype="bandpass", output="sos")
            self.sos_filters.append(sos)

    def process(self, audio: np.ndarray, gains_db: List[float]) -> np.ndarray:
        """Filter audio into sub-bands, apply band-specific linear gains (NAL-R fitting),

        and recombine.

        Parameters
        ----------
        audio : np.ndarray
            Input audio array (1D float).
        gains_db : List[float]
            Gain in decibels to apply to each sub-band.

        Returns
        -------
        np.ndarray
            Reconstructed speech with frequency-selective equalization.
        """
        output = np.zeros_like(audio)
        for i, sos in enumerate(self.sos_filters):
            band_signal = signal.sosfilt(sos, audio)
            # Half-gain rule (NAL-R principle: compensate ~40-50% of hearing threshold)
            gain_linear = 10.0 ** (gains_db[i] / 20.0)
            output += band_signal * gain_linear
        return output
