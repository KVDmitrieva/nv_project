import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectrogram, audio = [], []

    for item in dataset_items:
        audio.append(item["audio"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)

    return {
        "audio": audio,
        "mel":  torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    }