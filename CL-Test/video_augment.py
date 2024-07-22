import numpy as np
import os
import os.path as path
import torch
import random
from pkg_resources import packaging
from PIL import Image
from PIL import ImageFilter, ImageEnhance
from tqdm import tqdm

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

video_path = "./Frames"
video_aug_path = "./Frames_aug"
os.mkdir(video_aug_path)
video_dir = sorted(os.listdir(video_path))

split = 0.8

## --- Training --- ##
for i in tqdm(range(int(len(video_dir)*split))):
  os.mkdir(video_aug_path + "/" + video_dir[i])
  video_dir2 = sorted(os.listdir(video_path + "/" + video_dir[i]))
  count = 1
  for j in range(len(video_dir2)):
    video_files = sorted(os.listdir(video_path + "/" + video_dir[i] + "/" + video_dir2[j]))
    for k in range(5):
        for m in range(len(video_files)):
            img = Image.open(video_path + "/" + video_dir[i] + "/" + video_dir2[j] + "/" + video_files[m])
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

            output_dir = video_aug_path + "/" + video_dir[i] + "/" + video_dir[i] + "_" + str(num) + ".jpg"
            augmented_img.save(output_dir)

## --- Validating --- ##
for i in tqdm(range(int(len(video_dir)*split), len(video_dir))):
  os.mkdir(video_aug_path + "/" + video_dir[i])
  video_dir2 = sorted(os.listdir(video_path + "/" + video_dir[i]))
  count = 1
  for j in range(len(video_dir2)):
    video_files = sorted(os.listdir(video_path + "/" + video_dir[i] + "/" + video_dir2[j]))
    for k in range(1):
        for m in range(len(video_files)):
            img = Image.open(video_path + "/" + video_dir[i] + "/" + video_dir2[j] + "/" + video_files[m])
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

            output_dir = video_aug_path + "/" + video_dir[i] + "/" + video_dir[i] + "_" + str(num) + ".jpg"
            augmented_img.save(output_dir)