import numpy as np
import os
import os.path as path
import torch
import torchaudio
import random
from pkg_resources import packaging
from PIL import Image
from PIL import ImageFilter, ImageEnhance
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

## image augmentation
def image_transform(img):

  # blur
  blur_r = random.random()*2
  img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

  # color
  color_r = random.random()*0.3 + 0.7
  enhancer = ImageEnhance.Color(img)
  img = enhancer.enhance(color_r)

  # contrast
  contrast_r = random.random()*0.5 + 0.75
  enhancer = ImageEnhance.Contrast(img)
  img = enhancer.enhance(contrast_r)

  # brightness
  brightness_r = random.random()*0.5 + 0.75
  enhancer = ImageEnhance.Brightness(img)
  img = enhancer.enhance(brightness_r)

  return img


split = 0.8
audio_path = "./Audio/Data"
image_path = "./Frames"
audios = []
images = []

audio_dirs = sorted(os.listdir(audio_path))
image_dirs = sorted(os.listdir(image_path))

## training set
for i in tqdm(range(int(len(audio_dirs)*split))):

    new_dir = "temp"

    ## audio
    audio_files = sorted(os.listdir(audio_path + "/" + audio_dirs[i]))
    for j in range(len(audio_files)):
        count = 0
        os.mkdir(audio_path + "/" + audio_dirs[i] + "/" + new_dir)
        audio, sr = torchaudio.load(audio_path + "/" + audio_dirs[i] + "/" + audio_files[j])
        for k in range(5):
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

            output_dir = audio_path + "/" + audio_dirs[i] + "/" + new_dir + "/" + num + ".mp3"
            torchaudio.save(output_dir, transformed_audio, sr, format="mp3")

        # os.remove(audio_files[j])
        audio_files2 = sorted(os.listdir(audio_path + "/" + audio_dirs[i] + "/" + new_dir))
        for k in range(len(audio_files2)):
            audios.append(audio_path + "/" + audio_dirs[i] + "/" + new_dir + "/" + audio_files2[k])
        
        audio_input = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
        }
        with torch.no_grad():
            audio_embeddings = model(audio_input)
        
            audio_embed = audio_embeddings[ModalityType.AUDIO]
            if (i == 0 and j == 0):
                train_audio_embeds = audio_embed
            else:
                train_audio_embeds = torch.cat((train_audio_embeds, audio_embed), 0)
        
        audios = []
        os.system("rm -r {}".format(audio_path + "/" + audio_dirs[i] + "/" + new_dir))

    
    ## image
    image_dirs2 = sorted(os.listdir(image_path + "/" + image_dirs[i]))
    for j in range(len(image_dirs2)):
        count = 0
        os.mkdir(image_path + "/" + image_dirs[i] + "/" + new_dir)
        image_files = sorted(os.listdir(image_path + "/" + image_dirs[i] + "/" + image_dirs2[j]))
        for k in range(5):
            for m in range(len(image_files)):
                img = Image.open(image_path + "/" + image_dirs[i] + "/" + image_dirs2[j] + "/" + image_files[m])
                if (k != 0):
                    augmented_img = image_transform(img)
                else:
                    augmented_img = img
                
                if (count < 10):
                    num = "000" + str(count)
                elif (count < 100):
                    num = "00" + str(count)
                elif (count < 1000):
                    num = "0" + str(count)
                else:
                    num = str(count)
                count += 1

                output_dir = image_path + "/" + image_dirs[i] + "/" + new_dir + "/" + num + ".jpg"
                augmented_img.save(output_dir)
    
        image_files2 = sorted(os.listdir(image_path + "/" + image_dirs[i] + "/" + new_dir))
        for k in range(len(image_files2)):
            images.append(image_path + "/" + image_dirs[i] + "/" + new_dir + "/" + image_files2[k])
        
        image_input = {
            ModalityType.VISION: data.load_and_transform_vision_data(images, device),
        }
        with torch.no_grad():
            image_embeddings = model(image_input)
        
            video_embed = image_embeddings[ModalityType.VISION]
            video_embed = torch.reshape(video_embed, (5, len(image_files), -1))
            video_embed = torch.mean(video_embed, dim=1)

            if (i == 0 and j == 0):
                train_video_embeds = video_embed
            else:
                train_video_embeds = torch.cat((train_video_embeds, video_embed), 0)

        images = []
        os.system("rm -r {}".format(image_path + "/" + image_dirs[i] + "/" + new_dir))

print(train_video_embeds.shape)
print(train_audio_embeds.shape)

train_video = train_video_embeds.cpu().numpy()
train_audio = train_audio_embeds.cpu().numpy()

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
for i in tqdm(range(int(len(audio_dirs)*split), len(audio_dirs))):

    new_dir = "temp"

    ## audio
    audio_files = sorted(os.listdir(audio_path + "/" + audio_dirs[i]))
    for j in range(len(audio_files)):
        count = 0
        os.mkdir(audio_path + "/" + audio_dirs[i] + "/" + new_dir)
        audio, sr = torchaudio.load(audio_path + "/" + audio_dirs[i] + "/" + audio_files[j])
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

            output_dir = audio_path + "/" + audio_dirs[i] + "/" + new_dir + "/" + num + ".mp3"
            torchaudio.save(output_dir, transformed_audio, sr, format="mp3")

        # os.remove(audio_files[j])
        audio_files2 = sorted(os.listdir(audio_path + "/" + audio_dirs[i] + "/" + new_dir))
        for k in range(len(audio_files2)):
            audios.append(audio_path + "/" + audio_dirs[i] + "/" + new_dir + "/" + audio_files2[k])
        
        audio_input = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
        }
        with torch.no_grad():
            audio_embeddings = model(audio_input)
        
            audio_embed = audio_embeddings[ModalityType.AUDIO]
            if (i == int(len(audio_dirs)*split) and j == 0):
                valid_audio_embeds = audio_embed
            else:
                valid_audio_embeds = torch.cat((valid_audio_embeds, audio_embed), 0)
        
        audios = []
        os.system("rm -r {}".format(audio_path + "/" + audio_dirs[i] + "/" + new_dir))

    
    ## image
    image_dirs2 = sorted(os.listdir(image_path + "/" + image_dirs[i]))
    for j in range(len(image_dirs2)):
        count = 0
        os.mkdir(image_path + "/" + image_dirs[i] + "/" + new_dir)
        image_files = sorted(os.listdir(image_path + "/" + image_dirs[i] + "/" + image_dirs2[j]))
        for k in range(1):
            for m in range(len(image_files)):
                img = Image.open(image_path + "/" + image_dirs[i] + "/" + image_dirs2[j] + "/" + image_files[m])
                if (k != 0):
                    augmented_img = image_transform(img)
                else:
                    augmented_img = img
                
                if (count < 10):
                    num = "000" + str(count)
                elif (count < 100):
                    num = "00" + str(count)
                elif (count < 1000):
                    num = "0" + str(count)
                else:
                    num = str(count)
                count += 1

                output_dir = image_path + "/" + image_dirs[i] + "/" + new_dir + "/" + num + ".jpg"
                augmented_img.save(output_dir)
    
        image_files2 = sorted(os.listdir(image_path + "/" + image_dirs[i] + "/" + new_dir))
        for k in range(len(image_files2)):
            images.append(image_path + "/" + image_dirs[i] + "/" + new_dir + "/" + image_files2[k])
        
        image_input = {
            ModalityType.VISION: data.load_and_transform_vision_data(images, device),
        }
        with torch.no_grad():
            image_embeddings = model(image_input)
        
            video_embed = image_embeddings[ModalityType.VISION]
            video_embed = torch.reshape(video_embed, (1, len(image_files), -1))
            video_embed = torch.mean(video_embed, dim=1)

            if (i == int(len(audio_dirs)*split) and j == 0):
                valid_video_embeds = video_embed
            else:
                valid_video_embeds = torch.cat((valid_video_embeds, video_embed), 0)

        images = []
        os.system("rm -r {}".format(image_path + "/" + image_dirs[i] + "/" + new_dir))


print(valid_video_embeds.shape)
print(valid_audio_embeds.shape)

valid_video = valid_video_embeds.cpu().numpy()
valid_audio = valid_audio_embeds.cpu().numpy()

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