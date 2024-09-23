import os
import os.path as path
import numpy as np
import torch
import torchaudio
import librosa
import laion_clap
from tqdm import tqdm
from torch_audiomentations import (
    Compose, Gain, PolarityInversion, 
    PeakNormalization, PitchShift
)

clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device='cpu')
clap_model.load_ckpt("./checkpoint/music_audioset_epoch_15_esc_90.14.pt")

fixed_sr = 48000
audio_seg_length = 10
split = 0.9
audio_path = "./audio_segs"
apply_augmentation = Compose(
    transforms=[
        PeakNormalization(),
        Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.5, sample_rate=fixed_sr, target_rate=fixed_sr),
        PolarityInversion(p=0.5, sample_rate=fixed_sr, target_rate=fixed_sr),
        PitchShift(p=0.5, sample_rate=fixed_sr, target_rate=fixed_sr)
    ]
)

## --- training part --- ##
for i in tqdm(range(int(10641 * split))):

    if ((i + 1) < 10):
        dir_num = "0000" + str(i + 1)
    elif ((i + 1) < 100):
        dir_num = "000" + str(i + 1)
    elif ((i + 1) < 1000):
        dir_num = "00" + str(i + 1)
    elif ((i + 1) < 10000):
        dir_num = "0" + str(i + 1)
    else:
        dir_num = str(i + 1)
    
    # check directory exists
    if (not os.path.isdir(os.path.join(audio_path, dir_num))):
        continue

    audio_files = sorted(os.listdir(os.path.join(audio_path, dir_num)))
    for j in range(len(audio_files)):
        data, sr = torchaudio.load(os.path.join(audio_path, dir_num, audio_files[j]))
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=fixed_sr)
        data = torch.mean(data, dim=0, keepdim=True)
        data = data[None, :, 0:(fixed_sr*(audio_seg_length-2))]
        for k in range(1): ## augmentation
            if (k == 0):
                aug_data = data
            else:    
                aug_data = apply_augmentation(data, sample_rate=fixed_sr)
            aug_data = torch.reshape(aug_data, (1, -1)).detach().cpu().numpy()
            if (j == 0 and k == 0):
                audios = aug_data
            else:
                audios = np.concatenate((audios, aug_data), axis=0)

    audio_embed = clap_model.get_audio_embedding_from_data(x=audios, use_tensor=False)
    if (i == 0):
        train_audio_embeds = audio_embed
    else:
        train_audio_embeds = np.concatenate((train_audio_embeds, audio_embed), axis=0)

print(train_audio_embeds.shape)
filename1 = './embeddings/train_audio.npy'
if (os.path.isfile(filename1)):
    os.remove(filename1)
os.system("touch {}".format(filename1))
fp1 = np.memmap(filename1, dtype='float32', mode='w+', shape=(train_audio_embeds.shape[0], train_audio_embeds.shape[1]))
fp1[:] = train_audio_embeds[:]
fp1.filename == path.abspath(filename1)
fp1.flush()

## --- validating part --- ##
for i in tqdm(range(int(10641 * split), 10641)):

    if ((i + 1) < 10):
        dir_num = "0000" + str(i + 1)
    elif ((i + 1) < 100):
        dir_num = "000" + str(i + 1)
    elif ((i + 1) < 1000):
        dir_num = "00" + str(i + 1)
    elif ((i + 1) < 10000):
        dir_num = "0" + str(i + 1)
    else:
        dir_num = str(i + 1)
    
    # check directory exists
    if (not os.path.isdir(os.path.join(audio_path, dir_num))):
        continue
        
    audio_files = sorted(os.listdir(os.path.join(audio_path, dir_num)))
    for j in range(len(audio_files)):
        data, sr = torchaudio.load(os.path.join(audio_path, dir_num, audio_files[j]))
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=fixed_sr)
        data = torch.mean(data, dim=0, keepdim=True)
        data = data[None, :, 0:(fixed_sr*(audio_seg_length-2))]
        for k in range(1): ## augmentation
            if (k == 0):
                aug_data = data
            else:    
                aug_data = apply_augmentation(data, sample_rate=fixed_sr)
            aug_data = torch.reshape(aug_data, (1, -1)).detach().cpu().numpy()
            if (j == 0 and k == 0):
                audios = aug_data
            else:
                audios = np.concatenate((audios, aug_data), axis=0)

    audio_embed = clap_model.get_audio_embedding_from_data(x=audios, use_tensor=False)
    if (i == int(10641 * split)):
        valid_audio_embeds = audio_embed
    else:
        valid_audio_embeds = np.concatenate((valid_audio_embeds, audio_embed), axis=0)

print(valid_audio_embeds.shape)
filename2 = './embeddings/valid_audio.npy'
if (os.path.isfile(filename2)):
    os.remove(filename2)
os.system("touch {}".format(filename2))
fp2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(valid_audio_embeds.shape[0], valid_audio_embeds.shape[1]))
fp2[:] = valid_audio_embeds[:]
fp2.filename == path.abspath(filename2)
fp2.flush()