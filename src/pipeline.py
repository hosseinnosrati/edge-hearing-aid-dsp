"""Unified Real-Time Digital Hearing Aid Processing Pipeline.

Coordinates:
Input Audio -> AFC (Feedback Cancellation) -> Spectral Noise Reduction ->
Multi-band Audiogram Equalization -> WDRC Dynamic Compression -> Audio Output.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.dsp.filterbank import AudiogramFilterbank, AUDIOGRAM_PRESETS
from src.dsp.wdrc import WDRCCompressor
from src.dsp.adaptive_afc import AdaptiveFeedbackCanceller
from src.dsp.noise_reduction import SpectralNoiseReducer


class HearingAidPipeline:
    """Master signal processing engine for digital hearing aid simulation."""

    def __init__(self, fs: int = 16000):
        self.fs = fs
        self.filterbank = AudiogramFilterbank(fs=fs)
        self.wdrc = WDRCCompressor(fs=fs)
        self.afc = AdaptiveFeedbackCanceller(filter_order=64, step_size=0.03)
        self.denoiser = SpectralNoiseReducer(fs=fs)

    def process(
        self,
        mic_audio: np.ndarray,
        speaker_reference: Optional[np.ndarray] = None,
        audiogram_preset: str = "Moderate_High_Frequency_Loss",
        custom_gains: Optional[List[float]] = None,
        enable_afc: bool = True,
        enable_denoiser: bool = True,
        enable_wdrc: bool = True
    ) -> Dict[str, Any]:
        """Execute full hearing aid processing pipeline on an audio segment.

        Returns dictionary containing audio at each intermediate stage.
        """
        # 1. Feedback Cancellation (AFC)
        if enable_afc and speaker_reference is not None:
            stage1_afc = self.afc.process_block(mic_audio, speaker_reference)
        else:
            stage1_afc = mic_audio.copy()

        # 2. Spectral Noise Reduction
        if enable_denoiser:
            stage2_denoised, noise_spec = self.denoiser.enhance(stage1_afc, over_subtraction=1.4)
        else:
            stage2_denoised = stage1_afc.copy()
            noise_spec = np.zeros(128)

        # 3. Audiogram Prescription Gain (Filterbank)
        if custom_gains is not None:
            gains = custom_gains
        else:
            preset_dict = AUDIOGRAM_PRESETS.get(audiogram_preset, AUDIOGRAM_PRESETS["Normal"])
            # Half-gain rule: gain = 0.5 * hearing_loss_db
            gains = [0.5 * preset_dict[f] for f in [250, 500, 1000, 2000, 4000, 6000]]

        stage3_equalized = self.filterbank.process(stage2_denoised, gains_db=gains)

        # 4. Wide Dynamic Range Compression (WDRC) & Output Limiting
        if enable_wdrc:
            stage4_output = self.wdrc.process(stage3_equalized)
        else:
            stage4_output = np.clip(stage3_equalized, -1.0, 1.0)

        return {
            "raw_input": mic_audio,
            "afc_output": stage1_afc,
            "denoised_output": stage2_denoised,
            "equalized_output": stage3_equalized,
            "final_output": stage4_output,
            "applied_gains_db": gains,
            "noise_spectrum": noise_spec
        }
