import os
import os.path as path
import numpy as np
import torch
import torchaudio
import random
import laion_clap
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

audio_seg_time = 10

ma = "HTSAT-base"
clap_model = laion_clap.CLAP_Module(enable_fusion = False, amodel = ma)
clap_model.load_ckpt("music_audioset_epoch_15_esc_90.14.pt")

def transform(sr):
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

audio_path = "./Audios/Data"
os.chdir(audio_path)

audios = []
audio_dirs = sorted(os.listdir("./"))

split = 0.9

"""
count = 0
## --- Augmentation --- ##
for i in tqdm(range(int(len(audio_dirs)*split))):
  os.chdir("./" + audio_dirs[i])

  audio_files = sorted(os.listdir("./"))
  for j in range(len(audio_files)):

    new_dir = audio_files[j].split(".")[0]
    os.mkdir(new_dir)

    audio, sr = torchaudio.load(audio_files[j])
    for k in range(3):
      if (k != 0):
        process = transform(sr)
        transformed_audio = process(audio)
      else:
        transformed_audio = audio

      output_dir = "./" + new_dir + "/" + new_dir + "_" + str(k) + ".mp3"
      torchaudio.save(output_dir, transformed_audio, sr, format="mp3")

    # os.remove(audio_files[j])
    audio_files2 = sorted(os.listdir(new_dir))
    for k in range(len(audio_files2)):
      audios.append(new_dir + "/" + audio_files2[k])

    audio_embed = clap_model.get_audio_embedding_from_filelist(x=audios, use_tensor=False)
    # audio_embed = np.reshape(np.mean(audio_embed, axis=0), (1, -1))
    if (count == 0):
      train_audio_embeds = audio_embed
    else:
      train_audio_embeds = np.concatenate((train_audio_embeds, audio_embed), axis=0)

    count += 1
    audios = []
    os.system("rm -r '{}'".format(new_dir))

  os.chdir("../")

count = 0
## --- Augmentation --- ##
for i in tqdm(range(int(len(audio_dirs)*split), len(audio_dirs))):
  os.chdir("./" + audio_dirs[i])

  audio_files = sorted(os.listdir("./"))
  for j in range(len(audio_files)):

    new_dir = audio_files[j].split(".")[0]
    os.mkdir(new_dir)

    audio, sr = torchaudio.load(audio_files[j])
    for k in range(1):
      if (k != 0):
        process = transform(sr)
        transformed_audio = process(audio)
      else:
        transformed_audio = audio

      output_dir = "./" + new_dir + "/" + new_dir + "_" + str(k) + ".mp3"
      torchaudio.save(output_dir, transformed_audio, sr, format="mp3")

    # os.remove(audio_files[j])
    audio_files2 = sorted(os.listdir(new_dir))
    for k in range(len(audio_files2)):
      audios.append(new_dir + "/" + audio_files2[k])

    audio_embed = clap_model.get_audio_embedding_from_filelist(x=audios, use_tensor=False)
    # audio_embed = np.reshape(np.mean(audio_embed, axis=0), (1, -1))
    if (count == 0):
      valid_audio_embeds = audio_embed
    else:
      valid_audio_embeds = np.concatenate((valid_audio_embeds, audio_embed), axis=0)

    count += 1
    audios = []
    os.system("rm -r '{}'".format(new_dir))

  os.chdir("../")
"""

## --- Training Part --- ##
for i in tqdm(range(int(len(audio_dirs)*split))):
  audio_files = sorted(os.listdir(audio_dirs[i]))
  for j in range(len(audio_files)):
    audios.append(audio_dirs[i] + "/" + audio_files[j])

    audio_embed = clap_model.get_audio_embedding_from_filelist(x=audios, use_tensor=False)
    # audio_embed = np.reshape(np.mean(audio_embed, axis=0), (1, -1))
    if (i == 0 and j == 0):
      train_audio_embeds = audio_embed
    else:
      train_audio_embeds = np.concatenate((train_audio_embeds, audio_embed), axis=0)

    audios = []

## --- Validating Part --- ##
for i in tqdm(range(int(len(audio_dirs)*split), len(audio_dirs))):
  audio_files = sorted(os.listdir(audio_dirs[i]))
  for j in range(len(audio_files)):
    audios.append(audio_dirs[i] + "/" + audio_files[j])

    audio_embed = clap_model.get_audio_embedding_from_filelist(x=audios, use_tensor=False)
    # audio_embed = np.reshape(np.mean(audio_embed, axis=0), (1, -1))
    if (i == int(len(audio_dirs)*split) and j == 0):
      valid_audio_embeds = audio_embed
    else:
      valid_audio_embeds = np.concatenate((valid_audio_embeds, audio_embed), axis=0)

    audios = []

print(train_audio_embeds.shape)
print(valid_audio_embeds.shape)

## --- Write Embeddings Data --- ##
filename1 = '../../Embeddings/train_audio.npy'
fp1 = np.memmap(filename1, dtype='float32', mode='w+', shape=(train_audio_embeds.shape[0], train_audio_embeds.shape[1]))
fp1[:] = train_audio_embeds[:]
fp1.filename == path.abspath(filename1)
fp1.flush()

filename2 = '../../Embeddings/valid_audio.npy'
fp2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(valid_audio_embeds.shape[0], valid_audio_embeds.shape[1]))
fp2[:] = valid_audio_embeds[:]
fp2.filename == path.abspath(filename2)
fp2.flush()