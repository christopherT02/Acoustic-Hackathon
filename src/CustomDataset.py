import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, coordinates, preprocessed_data_mfcc=[], preprocessed_data_mel=[], _type="rms", preprocessed_data_rms=[], preprocessed_data_zcr=[]):
        self.preprocessed_data_mfcc = preprocessed_data_mfcc
        self.preprocessed_data_rms = preprocessed_data_rms
        self.preprocessed_data_zcr = preprocessed_data_zcr
        self.preprocessed_data_mel = preprocessed_data_mel
        self.coordinates = coordinates
        self.type = _type
        if self.type == "rms":
            print("RMS: ", preprocessed_data_rms.shape)
        elif self.type == "mel":
            print("MEL: ", preprocessed_data_mel.shape)
        elif self.type == "zcr":
            print("ZCR: ", preprocessed_data_zcr.shape)
        elif self.type == "mfcc":
            print("MFCC: ", preprocessed_data_mfcc.shape)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinates = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        if self.type == "rms":
            rms = self.preprocessed_data_rms[idx]
            rms = torch.tensor(rms, dtype=torch.float32)
            return np.array(rms), coordinates
        elif self.type == "mfcc":
            mfcc = [torch.tensor(self.preprocessed_data_mfcc[idx, mic_index], dtype=torch.float32) for mic_index in range(4)]
            return np.array(mfcc), coordinates
        elif self.type == "zcr":
            zcr = torch.tensor(self.preprocessed_data_zcr[idx], dtype=torch.float32) if self.preprocessed_data_zcr else None
            return np.array(zcr), coordinates
        else:
            print("Error type")
            return None, None
