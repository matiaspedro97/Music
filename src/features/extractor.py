import torch
import numpy as np

from transformers import AutoFeatureExtractor



class HFAudioFeatureExtractor:
    def __init__(
            self,
            model_id: str, 
            do_normalize: bool = True,
            return_attention_mask: bool = True,
            max_dur: float = 10
    ) -> None:
        
        # HF feature extractor
        self.ft_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, 
            do_normalize=do_normalize, 
            return_attention_mask=return_attention_mask
        )

        # sampling rate
        self.sr = self.ft_extractor.sampling_rate

        # maximum duration
        self.max_dur = max_dur

    def transform(self, audios: np.ndarray):
        inputs = self.ft_extractor(
            audios,
            sampling_rate=self.sr,
            max_length=int(self.sr * self.max_dur),
            truncation=True,
            return_attention_mask=True,
        )
        return inputs