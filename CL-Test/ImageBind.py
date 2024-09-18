import numpy as np
import os
import os.path as path
import torch
import torchaudio
import random
from pkg_resources import packaging
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

## audio augmentation
audio_seg_time = 30

split = 0.8
audio_path = "./Audio_aug"
image_path = "./Frames_aug"
audios = []
images = []

audio_dirs = sorted(os.listdir(audio_path))
image_dirs = sorted(os.listdir(image_path))

## training set
#for i in tqdm(range(1)):
for i in tqdm(range(int(len(audio_dirs)*split))):

    ## audio
    audio_files = sorted(os.listdir(audio_path + "/" + audio_dirs[i]))
    for j in range(len(audio_files)):
        audios.append(audio_path + "/" + audio_dirs[i] + "/" + audio_files[j])

    audio_input = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
    }
    with torch.no_grad():
        audio_embeddings = model(audio_input)
    
        audio_embed = audio_embeddings[ModalityType.AUDIO]
        if (i == 0):
            train_audio_embeds = audio_embed
        else:
            train_audio_embeds = torch.cat((train_audio_embeds, audio_embed), 0)
    
    audios = []

    ## image
    image_files = sorted(os.listdir(image_path + "/" + image_dirs[i]))
    image_group = len(image_files) // len(audio_files)
    j = 0
    while (j < len(image_files)):
        l = j
        group = 0
        for m in range(1):
            for k in range(l, l + image_group * 5):
                images.append(image_path + "/" + image_dirs[i] + "/" + image_files[k])
            l += image_group * 5
            group += 5
            if (l >= len(image_files)):
                break
        
        image_input = {
            ModalityType.VISION: data.load_and_transform_vision_data(images, device),
        }
        with torch.no_grad():
            image_embeddings = model(image_input)
        
            video_embed = image_embeddings[ModalityType.VISION]
            video_embed = torch.reshape(video_embed, (group, image_group, -1))
            video_embed = torch.mean(video_embed, dim=1)

            if (i == 0 and j == 0):
                train_video_embeds = video_embed
            else:
                train_video_embeds = torch.cat((train_video_embeds, video_embed), 0)

        images = []
        j = l

train_video = train_video_embeds.cpu().numpy()
train_audio = train_audio_embeds.cpu().numpy()

print(train_video.shape)
print(train_audio.shape)

filename1 = './Embeddings/train_video.npy'
fp1 = np.memmap(filename1, dtype='float32', mode='w+', shape=(train_video.shape[0], train_video.shape[1]))
fp1[:] = train_video[:]
fp1.filename == path.abspath(filename1)
fp1.flush()

filename2 = './Embeddings/train_audio.npy'
fp2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(train_audio.shape[0], train_audio.shape[1]))
fp2[:] = train_audio[:]
fp2.filename == path.abspath(filename2)
fp2.flush()


## validation set
#for i in tqdm(range(int(len(audio_dirs)*split), int(len(audio_dirs)*split)+1)):
for i in tqdm(range(int(len(audio_dirs)*split), len(audio_dirs))):

    ## audio
    audio_files = sorted(os.listdir(audio_path + "/" + audio_dirs[i]))
    for j in range(len(audio_files)):
        audios.append(audio_path + "/" + audio_dirs[i] + "/" + audio_files[j])

    audio_input = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
    }
    with torch.no_grad():
        audio_embeddings = model(audio_input)
    
        audio_embed = audio_embeddings[ModalityType.AUDIO]
        if (i == int(len(audio_dirs)*split)):
            valid_audio_embeds = audio_embed
        else:
            valid_audio_embeds = torch.cat((valid_audio_embeds, audio_embed), 0)
    
    audios = []

    ## image
    image_files = sorted(os.listdir(image_path + "/" + image_dirs[i]))
    image_group = len(image_files) // len(audio_files)
    j = 0
    while (j < len(image_files)):
        l = j
        group = 0
        for m in range(5):
            for k in range(l, l + image_group):
                images.append(image_path + "/" + image_dirs[i] + "/" + image_files[k])
            l += image_group
            group += 1
            if (l >= len(image_files)):
                break
        
        image_input = {
            ModalityType.VISION: data.load_and_transform_vision_data(images, device),
        }
        with torch.no_grad():
            image_embeddings = model(image_input)
        
            video_embed = image_embeddings[ModalityType.VISION]
            video_embed = torch.reshape(video_embed, (group, image_group, -1))
            video_embed = torch.mean(video_embed, dim=1)

            if (i == int(len(audio_dirs)*split) and j == 0):
                valid_video_embeds = video_embed
            else:
                valid_video_embeds = torch.cat((valid_video_embeds, video_embed), 0)

        images = []
        j = l

valid_video = valid_video_embeds.cpu().numpy()
valid_audio = valid_audio_embeds.cpu().numpy()

print(valid_video.shape)
print(valid_audio.shape)

filename3 = './Embeddings/valid_video.npy'
fp3 = np.memmap(filename3, dtype='float32', mode='w+', shape=(valid_video.shape[0], valid_video.shape[1]))
fp3[:] = valid_video[:]
fp3.filename == path.abspath(filename3)
fp3.flush()

filename4 = './Embeddings/valid_audio.npy'
fp4 = np.memmap(filename4, dtype='float32', mode='w+', shape=(valid_audio.shape[0], valid_audio.shape[1]))
fp4[:] = valid_audio[:]
fp4.filename == path.abspath(filename4)
fp4.flush()