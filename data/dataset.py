import os

import h5py
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import load


class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True) #"hdf"
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5") # hdf/train.hdf5
        # partition train, val, test
        self.random_hops = random_hops #True, False, False
        self.sr = sr #44100
        self.channels = channels #2
        self.shapes = shapes #{'output_start_frame': 4776, 'output_end_frame': 93185, 'output_frames': 88409, 'input_frames': 97961}
        print(shapes)
        self.audio_transform = audio_transform  # augment_func()
        self.in_memory = in_memory #False
        self.instruments = instruments
 
        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir): # hdf/train.hdf5
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr # 44100
                f.attrs["channels"] = channels # 2
                f.attrs["instruments"] = instruments # ['bass', 'drums', 'other', 'vocals']

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])): #idx 노래 곡 수
                    # Load mix
                    # print(example) {'mix': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18-hq/train/d/mixture.wav', 'bass': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18-hq/train/d/bass.wav', 'drums': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18-hq/train/d/drums.wav', 'other': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18-hq/train/d/other.wav', 'vocals': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18-hq/train/d/vocals.wav', 'accompaniment': '/home/bj/data/dnn/cfnet_venv/music_data/musdb18-hq/train/d/accompaniment.wav'}
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))
                    source_audios = []
                    for source in instruments:
                        # In this case, read in audio and convert to target sampling rate
                        source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0) # ch 증가 (8, 8346782길이따라)
                    assert(source_audios.shape[1] == mix_audio.shape[1])
                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]
                    # print('#########',mix_audio.shape ,source_audios.shape )(2, 8346782) (8, 8346782)
        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or f.attrs["channels"] != channels or list(f.attrs["instruments"]) != instruments: # False
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel or instruments are not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]
            # print('lengths',lengths) lengths [8346782, 7213530, 8679219]
            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]
            # print(self.shapes["output_frames"]) 88409
            # self.shapes dict_keys(['output_start_frame', 'output_end_frame', 'output_frames', 'input_frames'])
            # print(lengths) [95, 82, 99]
        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1] # 누적 길이 
        # print(self.start_pos)
        # print(self.length)
        # SortedList([95, 177, 276]) train
        # 276
        # SortedList([86, 175]) test
        # 175
        # SortedList([101, 206, 296]) val
        # 296
    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None: # 처음만
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            # print('core',self.in_memory, driver)  False, None
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver) #

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index) # 인덱스가 몇번째 노래에 해당하는가?
        # print(self.start_pos, index)  ([95, 177, 276]) 187
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1] # 몇번째 노래 처음부터의 인덱스는 얼마인가?

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]
        print('audio_length, target_length', audio_idx, audio_length, target_length)

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))          
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]
        print('start_target_pos', index, start_target_pos, self.shapes["output_frames"])
        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"] # 4776
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = {inst : targets[idx*self.channels:(idx+1)*self.channels] for idx, inst in enumerate(self.instruments)}

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __len__(self):
        return self.length