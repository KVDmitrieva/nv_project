import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectrogram, audio = [], []
    spectrogram_length, audio_path = [], []

    for item in dataset_items:
        audio.append(item["audio"])
        audio_path.append(item["audio_path"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        spectrogram_length.append(item["spectrogram"].shape[2])

    return {
        "audio": audio,
        "audio_path": audio_path,
        "mel_length": torch.tensor(spectrogram_length),
        "mel":  torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    }