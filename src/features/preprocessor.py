import numpy as np
import librosa
import tqdm

from src.utils.util import window_splitter


class AudioPreProcessor:
    def __init__(self, input_dur: float = 10, sample_rate: float = 16_000, check_norm: bool = True) -> None:
        # assign settings
        self.input_dur = input_dur
        self.sr = sample_rate

        # input length (in points)
        self.input_len = int(input_dur * sample_rate)

    def read_audiofile(self, path: str):
        audio, _ = librosa.load(path, sr=self.sr)
        return audio
    
    def norm_audio(self, audio: np.ndarray):
        return librosa.util.normalize(audio)
    
    def process(self, path: str, to_norm: bool = True, **kwargs):
        audio = self.read_audiofile(path)

        # normalize if applicable
        if to_norm:
            audio_norm = self.norm_audio(audio)
        else:
            audio_norm = audio
        return audio_norm
    
    def split_check(self, audio: np.ndarray):
        if len(audio) > self.input_len:
            audios = window_splitter(
                signal=audio, 
                window_size=self.input_len, 
                overlap=0.5
            )
        else:
            audios = [audio]
        return audios

    def transform_onefile(self, path: str, **kwargs):
        audio_pre = self.process(path, **kwargs)
        audios = self.split_check(audio_pre)
        return audios

    def transform_multifile(self, path_list: str, **kwargs):
        ind_list, audio_list = [], []
        for idx, p in tqdm.tqdm(enumerate(path_list), desc='Read audios'):
            # audios (after splitting)
            audios = self.transform_onefile(p, **kwargs)
            
            # audio index
            ind_list += [idx] * len(audios)
            
            # audio list
            audio_list += audios

        return audio_list, ind_list


