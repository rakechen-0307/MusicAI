import os
import os.path as path
import numpy as np
import torch
import clip
import random
from pkg_resources import packaging
from PIL import Image
from PIL import ImageFilter, ImageEnhance
from tqdm import tqdm

clip_model, preprocess = clip.load("ViT-L/14")
input_resolution = clip_model.visual.input_resolution
context_length = clip_model.context_length
vocab_size = clip_model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

def transform(img):

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

video_path = "./Frames"
os.chdir(video_path)

"""
dirs = sorted(os.listdir("./"))

## --- Augmentation --- ##
for i in range(len(dirs)):
  dirs2 = sorted(os.listdir(dirs[i]))
  for j in range(len(dirs2)):
    files = sorted(os.listdir(dirs[i] + "/" + dirs2[j]))
    for k in range(5):
      os.mkdir(dirs[i] + "/" + dirs2[j] + "/" + str(k+1))
    for m in range(len(files)):
      img = Image.open(dirs[i] + "/" + dirs2[j] + "/" + files[m])
      for k in range(5):
        if (k != 0):
          augmented_img = transform(img)
        else:
          augmented_img = img
        augmented_img.save(dirs[i] + "/" + dirs2[j] + "/" + str(k+1) + "/" + files[m])
      os.remove(dirs[i] + "/" + dirs2[j] + "/" + files[m])
  print(str(i) + "finished")
"""

train_images = []
valid_images = []
dirs = sorted(os.listdir("./"))

split = 0.9

## --- Training Part --- ##
for i in tqdm(range(int(len(dirs)*split))):
  dirs2 = sorted(os.listdir(dirs[i]))
  for j in range(len(dirs2)):
    files = sorted(os.listdir(dirs[i] + "/" + dirs2[j]))
    for k in range(1):
      for m in range(len(files)):
        img = Image.open(dirs[i] + "/" + dirs2[j] + "/" + files[m])
        if (k != 0):
          augmented_img = transform(img)
        else:
          augmented_img = img
        train_images.append(preprocess(augmented_img))
      
    train_video_input = torch.tensor(np.stack(train_images)).cuda()
    with torch.no_grad():
      train_video_embed = clip_model.encode_image(train_video_input).float()
      train_video_embed = torch.reshape(train_video_embed, (1, len(files), -1))
      train_video_embed = torch.mean(train_video_embed, dim=1)
      if (i == 0 and j == 0):
        train_video_embeds = train_video_embed
      else:
        train_video_embeds = torch.cat((train_video_embeds, train_video_embed), axis=0)

    train_images = []

## --- Validating Part --- ##
for i in tqdm(range(int(len(dirs)*split), len(dirs))):
  dirs2 = sorted(os.listdir(dirs[i]))
  for j in range(len(dirs2)):
    files = sorted(os.listdir(dirs[i] + "/" + dirs2[j]))
    for k in range(1):
      for m in range(len(files)):
        img = Image.open(dirs[i] + "/" + dirs2[j] + "/" + files[m])
        if (k != 0):
          augmented_img = transform(img)
        else:
          augmented_img = img
        valid_images.append(preprocess(augmented_img))
      
    valid_video_input = torch.tensor(np.stack(valid_images)).cuda()
    with torch.no_grad():
      valid_video_embed = clip_model.encode_image(valid_video_input).float()
      valid_video_embed = torch.reshape(valid_video_embed, (1, len(files), -1))
      valid_video_embed = torch.mean(valid_video_embed, dim=1)
      if (i == int(len(dirs)*split) and j == 0):
        valid_video_embeds = valid_video_embed
      else:
        valid_video_embeds = torch.cat((valid_video_embeds, valid_video_embed), 0)

    valid_images = []

train_image = train_video_embeds.cpu().numpy()
valid_image = valid_video_embeds.cpu().numpy()
os.chdir("../")

print(train_image.shape)
print(valid_image.shape)

## --- Write Embeddings Data --- ##
filename1 = './Embeddings/train_video.npy'
fp1 = np.memmap(filename1, dtype='float32', mode='w+', shape=(train_image.shape[0], train_image.shape[1]))
fp1[:] = train_image[:]
fp1.filename == path.abspath(filename1)
fp1.flush()

filename2 = './Embeddings/valid_video.npy'
fp2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(valid_image.shape[0], valid_image.shape[1]))
fp2[:] = valid_image[:]
fp2.filename == path.abspath(filename2)
fp2.flush()