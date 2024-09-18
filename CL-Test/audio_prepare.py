import os
from torchaudio_augmentations import *
from pydub import AudioSegment
from tqdm import tqdm

audio_path = "./Audios"
os.chdir(audio_path)
os.system("rm -r './Data'")

audio_files = sorted(os.listdir("./"))

audio_data_path = "./Data"
os.mkdir(audio_data_path)

split_sec = 10
step = 7

for i in tqdm(range(len(audio_files))):

  dir = audio_data_path + "/" + audio_files[i].split('.')[0]
  os.mkdir(dir)
  song = AudioSegment.from_file(audio_files[i], format="mp3")
  low = len(song)*0.2
  high = len(song)*0.8
  count = 1
  while (low+(split_sec*1000)/2 < high):
    clip_song = song[low : low + split_sec*1000]

    if (count < 10):
      j = "0" + str(count)
    else:
      j = str(count)
    export_dir = dir + "/" + audio_files[i].split('.')[0] + "_" + str(j) + ".mp3"
    clip_song.export(export_dir, format="mp3")

    low += (step*1000)
    count += 1

os.chdir(audio_data_path)