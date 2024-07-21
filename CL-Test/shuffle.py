import os
import torch
import numpy as np
import random
import os.path as path
from tqdm import tqdm

split = 0.8
aug_num_train = 5
aug_num_valid = 1

filepath = "Frames/"
dirs = sorted(os.listdir(filepath))

pos_train = [0]
for i in range(int(split*len(dirs))):
    pos_train.append(pos_train[-1] + aug_num_train * len(os.listdir(filepath + dirs[i])))

pos_valid = [0]
for i in range(int(split*len(dirs)), len(dirs)):
    pos_valid.append(pos_valid[-1] + aug_num_valid * len(os.listdir(filepath + dirs[i])))

count_train = len(pos_train) - 1
count_valid = len(pos_valid) - 1

image_train_file = './Embeddings/train_video.npy'
train_video_embeds = torch.from_numpy(np.asarray(np.memmap(image_train_file, dtype='float32', mode='r+', shape=(45480, 1024))))

image_valid_file = './Embeddings/valid_video.npy'
valid_video_embeds = torch.from_numpy(np.asarray(np.memmap(image_valid_file, dtype='float32', mode='r+', shape=(2178, 1024))))

audio_train_file = './Embeddings/train_audio.npy'
train_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_train_file, dtype='float32', mode='r+', shape=(45480, 1024))))

audio_valid_file = './Embeddings/valid_audio.npy'
valid_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_valid_file, dtype='float32', mode='r+', shape=(2178, 1024))))

batch_size = 128

## Training Part ##
for i in tqdm(range(500)):
    li = []
    for k in range(count_train):
        li.append(k+1)
    for j in range(batch_size):
        id = random.randint(0, len(li)-1)
        idx = li[id]
        video = random.randint(pos_train[idx-1], pos_train[idx]-1)
        audio = random.randint(video - video % aug_num_train, video - video % aug_num_train + (aug_num_train - 1))
        video_embed = train_video_embeds[video, :].reshape((1, -1))
        audio_embed = train_audio_embeds[audio, :].reshape((1, -1))
        if (j == 0):
            tmp_train_video_embeds = video_embed
            tmp_train_audio_embeds = audio_embed
        else:
            tmp_train_video_embeds = np.concatenate((tmp_train_video_embeds, video_embed), axis=0)
            tmp_train_audio_embeds = np.concatenate((tmp_train_audio_embeds, audio_embed), axis=0)
        del li[id]
    if (i == 0):
        new_train_video_embeds = tmp_train_video_embeds
        new_train_audio_embeds = tmp_train_audio_embeds
    else:
        new_train_video_embeds = np.concatenate((new_train_video_embeds, tmp_train_video_embeds), axis=0)
        new_train_audio_embeds = np.concatenate((new_train_audio_embeds, tmp_train_audio_embeds), axis=0)

print(new_train_video_embeds.shape)
print(new_train_audio_embeds.shape)


## Validating Part ##
for i in tqdm(range(20)):
    li = []
    for k in range(count_valid):
        li.append(k+1)
    for j in range(batch_size):
        id = random.randint(0, len(li)-1)
        idx = li[id]
        video = random.randint(pos_valid[idx-1], pos_valid[idx]-1)
        audio = random.randint(video - video % aug_num_valid, video - video % aug_num_valid + (aug_num_valid - 1))
        video_embed = valid_video_embeds[video, :].reshape((1, -1))
        audio_embed = valid_audio_embeds[audio, :].reshape((1, -1))
        if (j == 0):
            tmp_valid_video_embeds = video_embed
            tmp_valid_audio_embeds = audio_embed
        else:
            tmp_valid_video_embeds = np.concatenate((tmp_valid_video_embeds, video_embed), axis=0)
            tmp_valid_audio_embeds = np.concatenate((tmp_valid_audio_embeds, audio_embed), axis=0)
        del li[id]
    if (i == 0):
        new_valid_video_embeds = tmp_valid_video_embeds
        new_valid_audio_embeds = tmp_valid_audio_embeds
    else:
        new_valid_video_embeds = np.concatenate((new_valid_video_embeds, tmp_valid_video_embeds), axis=0)
        new_valid_audio_embeds = np.concatenate((new_valid_audio_embeds, tmp_valid_audio_embeds), axis=0)

print(new_valid_video_embeds.shape)
print(new_valid_audio_embeds.shape)

## --- Write New Embeddings Data --- ##
filename1 = './Embeddings/new_train_video.npy'
fp1 = np.memmap(filename1, dtype='float32', mode='w+', shape=(new_train_video_embeds.shape[0], new_train_video_embeds.shape[1]))
fp1[:] = new_train_video_embeds[:]
fp1.filename == path.abspath(filename1)
fp1.flush()

filename2 = './Embeddings/new_valid_video.npy'
fp2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(new_valid_video_embeds.shape[0], new_valid_video_embeds.shape[1]))
fp2[:] = new_valid_video_embeds[:]
fp2.filename == path.abspath(filename2)
fp2.flush()

filename3 = './Embeddings/new_train_audio.npy'
fp3 = np.memmap(filename3, dtype='float32', mode='w+', shape=(new_train_audio_embeds.shape[0], new_train_audio_embeds.shape[1]))
fp3[:] = new_train_audio_embeds[:]
fp3.filename == path.abspath(filename3)
fp3.flush()

filename4 = './Embeddings/new_valid_audio.npy'
fp4 = np.memmap(filename4, dtype='float32', mode='w+', shape=(new_valid_audio_embeds.shape[0], new_valid_audio_embeds.shape[1]))
fp4[:] = new_valid_audio_embeds[:]
fp4.filename == path.abspath(filename4)
fp4.flush()