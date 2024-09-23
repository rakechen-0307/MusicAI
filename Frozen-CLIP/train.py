import os
import av
import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from info_nce import InfoNCE

from model import EVLTransformer

mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
spatial_size = 224
num_frames = 8
sampling_rate = 32
num_temporal_views = 1
num_spatial_views = 1
decoder_num_layers = 4
decoder_qkv_dim = 1024
decoder_num_heads = 16

class VideoAudioDataset(Dataset):
    def __init__(
        self, video_data, audio_data, video_dir, audio_embed,
        num_spatial_views, num_temporal_views, num_frames, sampling_rate, spatial_size,
        mean, std
    ):
        self.video_data = video_data
        self.audio_data = audio_data
        self.video_dir = video_dir
        self.audio_embed = audio_embed

        self.spatial_size = spatial_size
        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        self.num_temporal_views = num_temporal_views
        self.num_spatial_views = num_spatial_views

    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        video_idx = self.video_data[idx]
        audio_idx = self.audio_data[idx]

        audio = self.audio_embed[audio_idx]

        dir = sorted(os.listdir(self.video_dir))[video_idx[0]]
        file = sorted(os.listdir(os.path.join(self.video_dir, dir)))[video_idx[1]]
        video_file = os.path.join(self.video_dir, dir, file)

        container = av.open(video_file)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        frame_idx = []
        for i in range(self.num_frames):
            frame_idx.append(i * self.sampling_rate if i * self.sampling_rate < len(frames) else frame_idx[-1])

        cropped_frames = []
        for x in frame_idx:
            img = frames[x].to_image()  # PIL image
            width, height = img.size    # Get dimensions

            new_size = min(width, height)
            left = (width - new_size) // 2
            top = (height - new_size) // 2
            right = left + new_size
            bottom = top + new_size
            img = img.crop((left, top, right, bottom))  # Crop the center of the image

            cropped_frame = av.video.frame.VideoFrame.from_image(img).reformat(width=self.spatial_size, height=self.spatial_size).to_rgb().to_ndarray()
            cropped_frames.append(cropped_frame)

        frames = cropped_frames
        frames = torch.as_tensor(np.stack(frames)).float() / 255.
        frames = (frames - self.mean) / self.std
        frames = frames.permute(3, 0, 1, 2) # C, T, H, W

        return frames, audio
    
def trainer(train_dataloader, valid_dataloader, model, config, device):

    n_epochs, best_loss, step, early_stop_count = config['n_epoch'], math.inf, 0, 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    criterion = InfoNCE(temperature=0.01)

    for epoch in range(n_epochs):

        ## training
        model.train()
        loss_record = []

        train_pbar = tqdm(train_dataloader, position=0, leave=True)

        for frames, audio in train_pbar:

            optimizer.zero_grad()
            frames, audio = frames.to(device), audio.to(device)

            output = model(frames)
            print(output.shape)
            loss = criterion(output, audio)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        scheduler.step(mean_train_loss)

        ## validate
        model.eval()
        loss_record = []
        for frames, audio in valid_dataloader:
            frames, audio = frames.to(device), audio.to(device)
            with torch.no_grad():
                output = model(frames)
                loss = criterion(output, audio)

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

pos_train = [0]
pos_valid = [0]
split = 0.9
video_dir = 'video_segs'
dirs = os.listdir(video_dir)
for i in range(int(10641 * split)):
    pos_train.append(pos_train[-1] + len(os.listdir(os.path.join(video_dir, dirs[i]))))
for i in range(int(10641 * split), 10641):
    pos_valid.append(pos_valid[-1] + len(os.listdir(os.path.join(video_dir, dirs[i]))))

count_train = len(pos_train) - 1
count_valid = len(pos_valid) - 1
total_train = pos_train[-1]
total_valid = pos_valid[-1]

## shuffle
train_video_data = []
valid_video_data = []
train_audio_data = []
valid_audio_data = []
batch_size = 16
# training part 
for i in range(500):
    li = []
    for k in range(count_train):
        li.append(k+1)
    for j in range(batch_size):
        id = random.randint(0, len(li)-1)
        idx = li[id]
        audio = random.randint(pos_train[idx-1], pos_train[idx]-1)
        video = (idx-1, random.randint(pos_train[idx-1], pos_train[idx]-1) - pos_train[idx-1])
        train_audio_data.append(audio)
        train_video_data.append(video)

# validate part
for i in range(20):
    li = []
    for k in range(count_valid):
        li.append(k+1)
    for j in range(batch_size):
        id = random.randint(0, len(li)-1)
        idx = li[id]
        audio = random.randint(pos_train[idx-1], pos_train[idx]-1)
        video = (idx-1+count_train, audio - pos_train[idx-1])
        valid_audio_data.append(audio)
        valid_video_data.append(video)      

audio_train_file = './embeddings/train_audio.npy'
train_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_train_file, dtype='float32', mode='r+', shape=(total_train, 512))))
audio_valid_file = './embeddings/valid_audio.npy'
valid_audio_embeds = torch.from_numpy(np.asarray(np.memmap(audio_valid_file, dtype='float32', mode='r+', shape=(total_valid, 512))))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'n_epoch': 200,
    'batch_size': batch_size,
    'learning_rate': 5e-4,
    'early_stop': 50,
    'save_path': './model.pt'
}

train_dataset = VideoAudioDataset(video_data=train_video_data, audio_data=train_audio_data,
                               video_dir=video_dir, audio_embed=train_audio_embeds,
                               num_spatial_views=num_spatial_views, num_temporal_views=num_temporal_views, 
                               num_frames=num_frames, sampling_rate=sampling_rate, 
                               spatial_size=spatial_size, mean=mean, std=std)
valid_dataset = VideoAudioDataset(video_data=valid_video_data, audio_data=valid_audio_data,
                               video_dir=video_dir, audio_embed=valid_audio_embeds,
                               num_spatial_views=num_spatial_views, num_temporal_views=num_temporal_views, 
                               num_frames=num_frames, sampling_rate=sampling_rate, spatial_size=spatial_size,
                               mean=mean, std=std)

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

model = EVLTransformer(
    num_frames=num_frames,
    backbone_name="ViT-L/14-lnpre",
    backbone_type="clip",
    backbone_path="./checkpoint/ViT-L-14.pt",
    backbone_mode="freeze_fp16",
    decoder_num_layers=decoder_num_layers,
    decoder_qkv_dim=decoder_qkv_dim,
    decoder_num_heads=decoder_num_heads,
    num_classes=512
).to(device)
trainer(train_dataloader, valid_dataloader, model, config, device)