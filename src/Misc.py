import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tqdm import tqdm
import torch
from EuclideanDistanceLoss import EuclideanDistanceLoss

class Misc:
    def __init__(self, start_time=0, end_time=50000, sr=44100, target_sr=16000):
        self.start_time = start_time
        self.end_time = end_time
        self.sr = sr
        self.target_sr = target_sr

    def euclidean_distance(self, pred_coords, true_coords):
        if pred_coords.requires_grad:
            pred_coords = pred_coords.detach().numpy()
        if true_coords.requires_grad:
            true_coords = true_coords.detach().numpy()
        if not isinstance(pred_coords, np.ndarray):
            pred_coords = pred_coords.numpy()
        if not isinstance(true_coords, np.ndarray):
            pred_coords = true_coords.numpy()
        return np.sqrt(np.sum((pred_coords - true_coords)**2))

    def custom_scoring(self, estimator, X, y):
        EUCLIDEAN_LOSS = EuclideanDistanceLoss()
        pred_coords = estimator.predict(X)
        ed = np.mean([EUCLIDEAN_LOSS(torch.tensor(p), torch.tensor(t)) for p, t in zip(pred_coords, y)])
        return -ed

    def display_loss(self, centroids, predicted_coords):
        all_true_coords = centroids.reshape(-1, 2)
        all_pred_coords = predicted_coords.reshape(-1, 2)

        plt.figure(figsize=(8, 6))
        for true_coord, pred_coord in zip(all_true_coords, all_pred_coords):
            plt.plot([true_coord[0], pred_coord[0]], [true_coord[1], pred_coord[1]], color='red',
                     alpha=0.2, linestyle='-')
            plt.scatter(true_coord[0], true_coord[1], color='green', label='True Coordinates',
                        alpha=0.7, marker='o')
            plt.scatter(pred_coord[0], pred_coord[1], color='black', label='Predicted Coordinates',
                        alpha=0.7, marker='o')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('True (Green) vs Predicted (Black) Coordinates (Connected)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def visualize_and_extract_spectrogram(self, data, microphone_index, sample_index, title=None):
        # Select the specific microphone's data
        microphone_data = data[sample_index, microphone_index, :]

        # Filtering
        # filtered_signal = signal.medfilt(microphone_data, kernel_size=3)  # Uncomment for median filtering

        # Normalization
        normalized_signal = librosa.util.normalize(microphone_data)

        # Resampling
        # resampled_signal = librosa.resample(normalized_signal, orig_sr=48000, target_sr=16000)  # Maintain original fs

        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=normalized_signal, sr=48000)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=normalized_signal, sr=48000, n_mfcc=20)  # Extract 20 MFCCs

        print("Mel spectrogram shape:", mel_spectrogram.shape)

        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=48000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title=title if title is not None else 'Mel-frequency spectrogram')
        plt.show()

        return mel_spectrogram, mfccs

    def preprocess(self, deconvoled_trim):
        preprocessed_data_mfcc = []
        preprocessed_data_mel = []
        preprocessed_data_rms = []
        preprocessed_data_zcr = []

        for instance_index in tqdm(range(deconvoled_trim.shape[0])):
            instance_data_mfcc = []
            instance_data_mel = []
            instance_data_zcr = []
            instance_data_rms = []
            for channel_index in range(deconvoled_trim.shape[1]):
                channel_signal = deconvoled_trim[instance_index, channel_index, :]

                focused_signal = channel_signal[self.start_time:self.end_time + 1]

                filtered_signal = signal.medfilt(focused_signal, kernel_size=3)

                normalized_signal = librosa.util.normalize(filtered_signal)

                mel_features = librosa.feature.melspectrogram(
                    y=normalized_signal,
                    sr=self.target_sr,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128
                )
                mfcc_features = librosa.feature.mfcc(S=librosa.power_to_db(mel_features))

                rms_features = np.sqrt(np.mean(normalized_signal ** 2))

                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=normalized_signal)

                instance_data_mfcc.append(mfcc_features)
                instance_data_mel.append(mel_features)
                instance_data_zcr.append(zero_crossing_rate)
                instance_data_rms.append(rms_features)

            preprocessed_data_mfcc.append(instance_data_mfcc)
            preprocessed_data_mel.append(instance_data_mel)
            preprocessed_data_rms.append(instance_data_rms)
            preprocessed_data_zcr.append(instance_data_zcr)

        preprocessed_data_mfcc = np.array(preprocessed_data_mfcc)
        preprocessed_data_mel = np.array(preprocessed_data_mel)
        preprocessed_data_rms = np.array(preprocessed_data_rms)
        preprocessed_data_zcr = np.array(preprocessed_data_zcr)
        return preprocessed_data_mfcc, preprocessed_data_rms, preprocessed_data_zcr, preprocessed_data_mel

    def preprocess_knn(self, deconvoled_trim):
        preprocessed_data_rms = []
        preprocessed_data_zcr = []
        
        for instance_index in tqdm(range(deconvoled_trim.shape[0])):
            instance_data_zcr = []
            instance_data_rms = []
            for channel_index in range(deconvoled_trim.shape[1]):
                filtered_signal = signal.medfilt(deconvoled_trim[instance_index, channel_index, :], kernel_size=3)
                normalized_signal = librosa.util.normalize(filtered_signal)

                rms_features = np.sqrt(np.mean(deconvoled_trim[instance_index, channel_index, :]**2))
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=deconvoled_trim[instance_index, channel_index, :])
        
                instance_data_zcr.append(zero_crossing_rate)
                instance_data_rms.append(rms_features)

            preprocessed_data_rms.append(instance_data_rms)
            preprocessed_data_zcr.append(instance_data_zcr)
    
        preprocessed_data_rms = np.array(preprocessed_data_rms)
        preprocessed_data_zcr = np.array(preprocessed_data_zcr)

        return preprocessed_data_rms, preprocessed_data_zcr

    def plot_audio_features(self, instance_index, chan_index, mfcc=[], mel=[], rms_features=[], zcr=[]):
        if len(mfcc) > 0:
            # Plot MFCC features
            plt.figure(figsize=(10, 4))
            plt.imshow(mfcc, cmap='viridis', origin='lower', aspect='auto')
            plt.xlabel('Frame')
            plt.ylabel('MFCC Coefficient')
            plt.title('MFCC Features (Instance {}, Channel {})'.format(instance_index+1, chan_index+1))
            plt.colorbar(label='Magnitude')
            plt.tight_layout()
            plt.show()

        if len(rms_features) > 0:
            # Plot RMS Features for each channel
            for i, rms_channel in enumerate(rms_features):
                plt.figure(figsize=(10, 4))
                plt.plot(rms_channel, label=f'RMS Features (Channel {i+1})')
                plt.xlabel('Time (frame)')
                plt.ylabel('RMS Energy')
                plt.title('RMS Features (Instance {}, Channel {})'.format(instance_index+1, chan_index+1))
                plt.legend()
                plt.tight_layout()
                plt.show()

        if len(zcr) > 0:
            # Plot Zero-Crossing Rate for each channel
            for i, zcr_channel in enumerate(zcr):
                plt.figure(figsize=(10, 4))
                plt.plot(zcr_channel, label=f'Zero-Crossing Rate (Channel {i+1})')
                plt.xlabel('Time (frame)')
                plt.ylabel('Zero-Crossing Rate')
                plt.title('Zero-Crossing Rate (Instance {}, Channel {})'.format(instance_index+1, chan_index+1))
                plt.legend()
                plt.tight_layout()
                plt.show()

        if len(mel) > 0:
            plt.figure(figsize=(10, 4))
            plt.imshow(mel, cmap='viridis', origin='lower', aspect='auto')
            plt.xlabel('Time (frame)')
            plt.ylabel('Mel Frequency')
            plt.title('Mel Spectrogram (Instance {}, Channel {})'.format(instance_index+1, chan_index+1))
            plt.colorbar(label='Magnitude (dB)')
            plt.tight_layout()
            plt.show()
