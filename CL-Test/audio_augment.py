import numpy as np
import os
import os.path as path
import random
from tqdm import tqdm
from torchaudio_augmentations import (
    Compose,
    Delay,
    Gain,
    HighLowPass,
    Noise,
    PitchShift,
    PolarityInversion,
    RandomApply,
    RandomResizedCrop,
    Reverb,
)
import torchaudio

audio_seg_time = 10

def audio_transform(sr):
  transforms = [
    RandomResizedCrop(n_samples=sr*audio_seg_time),
    RandomApply([PolarityInversion()], p=0.5),
    RandomApply([HighLowPass(sample_rate=sr)], p=0.5),
    RandomApply([Delay(sample_rate=sr)], p=0.5),
    RandomApply([PitchShift(sample_rate=sr, n_samples=sr*audio_seg_time)], p=0.5),
    RandomApply([Gain()], p=0.5),
    RandomApply([Noise(max_snr=0.01)], p=0.5),
    RandomApply([Reverb(sample_rate=sr)], p=0.5)
  ]

  audio_transform = Compose(transforms=transforms)
  return audio_transform

audio_aug_path = "./Audio_aug"
audio_path = "./Audios/Data"
os.mkdir(audio_aug_path)
audio_dir = sorted(os.listdir(audio_path))

split = 0.9

## --- Training --- ##
for i in tqdm(range(int(len(audio_dir)*split))):
  os.mkdir(audio_aug_path + "/" + audio_dir[i])
  audio_files = sorted(os.listdir(audio_path + "/" + audio_dir[i]))
  count = 1
  for j in range(len(audio_files)):
    audio, sr = torchaudio.load(audio_path + "/" + audio_dir[i] + "/" + audio_files[j])
    for k in range(3):
        if (k != 0):
            process = audio_transform(sr)
            transformed_audio = process(audio)
        else:
            transformed_audio = audio

        if (count < 10):
            num = "000" + str(count)
        elif (count < 100):
            num = "00" + str(count)
        elif (count < 1000):
            num = "0" + str(count)
        else:
            num = str(count)
        count += 1
        
        output_dir = audio_aug_path + "/" + audio_dir[i] + "/" + audio_dir[i] + "_" + str(num) + ".mp3"
        torchaudio.save(output_dir, transformed_audio, sr, format="mp3")

## --- Validating --- ##
for i in tqdm(range(int(len(audio_dir)*split), len(audio_dir))):
  os.mkdir(audio_aug_path + "/" + audio_dir[i])
  audio_files = sorted(os.listdir(audio_path + "/" + audio_dir[i]))
  count = 1
  for j in range(len(audio_files)):
    audio, sr = torchaudio.load(audio_path + "/" + audio_dir[i] + "/" + audio_files[j])
    for k in range(1):
        if (k != 0):
            process = audio_transform(sr)
            transformed_audio = process(audio)
        else:
            transformed_audio = audio

        if (count < 10):
            num = "000" + str(count)
        elif (count < 100):
            num = "00" + str(count)
        elif (count < 1000):
            num = "0" + str(count)
        else:
            num = str(count)
        count += 1
        
        output_dir = audio_aug_path + "/" + audio_dir[i] + "/" + audio_dir[i] + "_" + str(num) + ".mp3"
        torchaudio.save(output_dir, transformed_audio, sr, format="mp3")