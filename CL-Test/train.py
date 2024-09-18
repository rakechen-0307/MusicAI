import torch
import math
import os.path as path
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from tqdm import tqdm
from info_nce import InfoNCE, info_nce

class VideoAudioDataset(Dataset):
  def __init__(self, images, audios, transform=None, target_transform=None):
    self.images = images
    self.audios = audios
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    assert self.audios.shape[0] == self.images.shape[0]
    return self.audios.shape[0]

  def __getitem__(self, idx):
    image = self.images[idx]
    audio = self.audios[idx]
    if (self.transform):
      image = self.transform(image)
    if (self.target_transform):
      audio = self.target_transform(audio)

    return image, audio

class Model(nn.Module):
  def __init__(self, input_dim):
    super(Model, self).__init__()
    self.fc = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 512),
        nn.Dropout(p=0.25)
    )

  def forward(self, x):
    x = self.fc(x)
    return x

def trainer(train_loader, valid_loader, model, config, device):

  torch.autograd.set_detect_anomaly(True)

  n_epochs, best_loss, step, early_stop_cnt = config['n_epoch'], math.inf, 0, 0
  optimizer = torch.optim.AdamW(model.fc.parameters(), lr=config['learning_rate'], weight_decay=0.001)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

  criterion = InfoNCE(temperature=0.01)

  for epoch in range(n_epochs):

    ## training
    model.train()
    loss_record = []

    train_pbar = tqdm(train_loader, position=0, leave=True)

    for x, y in train_pbar:
      optimizer.zero_grad()
      x, y = x.to(device), y.to(device)
      pred = model(x)
      loss = criterion(pred, y)
      loss.backward()
      optimizer.step()
      step += 1
      loss_record.append(loss.detach().item())

      # Display current epoch number and loss on tqdm progress bar.
      train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
      train_pbar.set_postfix({'loss': loss.detach().item()})

    mean_train_loss = sum(loss_record)/len(loss_record)
    # scheduler.step(mean_train_loss)

    ## validating
    model.eval()
    loss_record = []

    for x, y in valid_loader:
      x, y = x.to(device), y.to(device)
      with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)

      loss_record.append(loss.item())

    mean_valid_loss = sum(loss_record) / len(loss_record)

    print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

    if mean_valid_loss < best_loss:
      best_loss = mean_valid_loss
      torch.save(model.state_dict(), config['save_path']) # Save your best model
      print('Saving model with loss {:.3f}...'.format(best_loss))
      early_stop_count = 0
    else:
      early_stop_count += 1

    if early_stop_count >= config['early_stop']:
      print('\nModel is not improving, so we halt the training session.')
      return

## --- Read Embeddings Data --- ##
image_train_file = './Embeddings/new_train_video.npy'
train_video_embeds = torch.from_numpy(np.asarray(np.memmap(image_train_file, dtype='float32', mode='r+', shape=(256000, 768))))
image_valid_file = './Embeddings/new_valid_video.npy'
valid_video_embeds = torch.from_numpy(np.asarray(np.memmap(image_valid_file, dtype='float32', mode='r+', shape=(12800, 768))))
audio_train_file = './Embeddings/new_train_audio.npy'
train_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_train_file, dtype='float32', mode='r+', shape=(256000, 512))))
audio_valid_file = './Embeddings/new_valid_audio.npy'
valid_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_valid_file, dtype='float32', mode='r+', shape=(12800, 512))))

## --- Training --- ##
train_data = VideoAudioDataset(train_video_embeds, train_audio_embeds)
valid_data = VideoAudioDataset(valid_video_embeds, valid_audio_embeds)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'n_epoch': 200,
    'batch_size': 128,
    'learning_rate': 1e-6,
    'early_stop': 50,
    'save_path': './model.ckpt'
}

train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False)
valid_dataloader = DataLoader(valid_data, batch_size=config["batch_size"], shuffle=False)

train_model = Model(768).to(device)
trainer(train_dataloader, valid_dataloader, train_model, config, device)