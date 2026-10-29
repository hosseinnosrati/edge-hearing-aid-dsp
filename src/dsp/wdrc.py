"""Wide Dynamic Range Compression (WDRC) & Peak Limiter.

Compensates for sensorineural loudness recruitment:
- Amplifies low-intensity speech (soft consonants).
- Compresses moderate levels (normal conversation).
- Limits maximum power output (MPO) to protect the cochlea from acoustic trauma.
"""

import numpy as np


class WDRCCompressor:
    """Multi-channel or single-channel Wide Dynamic Range Compressor."""

    def __init__(
        self,
        fs: int = 16000,
        compression_threshold_db: float = -40.0,
        compression_ratio: float = 2.0,
        max_power_output_db: float = -3.0,
        attack_time_ms: float = 5.0,
        release_time_ms: float = 50.0
    ):
        self.fs = fs
        self.ct_db = compression_threshold_db
        self.cr = max(1.0, compression_ratio)
        self.mpo_db = max_power_output_db

        # Exponential smoothing factors for envelope tracking
        self.alpha_attack = np.exp(-1.0 / (attack_time_ms * 1e-3 * fs))
        self.alpha_release = np.exp(-1.0 / (release_time_ms * 1e-3 * fs))
        self.envelope = 1e-6

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process 1D audio through dynamic range compressor.

        Parameters
        ----------
        audio : np.ndarray
            Input float audio array in [-1.0, 1.0].

        Returns
        -------
        np.ndarray
            Compressed audio signal.
        """
        output = np.zeros_like(audio)
        env = self.envelope

        for i in range(len(audio)):
            sample = audio[i]
            abs_sample = abs(sample)

            # Envelope detector with asymmetric attack/release
            if abs_sample > env:
                env = self.alpha_attack * env + (1.0 - self.alpha_attack) * abs_sample
            else:
                env = self.alpha_release * env + (1.0 - self.alpha_release) * abs_sample

            env = max(env, 1e-6)
            env_db = 20.0 * np.log10(env)

            # Compression gain rule
            if env_db < self.ct_db:
                # Below threshold: linear gain
                gain_db = 0.0
            else:
                # Above threshold: compression
                gain_db = (1.0 / self.cr - 1.0) * (env_db - self.ct_db)

            # Check Maximum Power Output (MPO)
            if env_db + gain_db > self.mpo_db:
                gain_db = self.mpo_db - env_db

            gain_linear = 10.0 ** (gain_db / 20.0)
            output[i] = sample * gain_linear

        self.envelope = env
        return np.clip(output, -1.0, 1.0)
