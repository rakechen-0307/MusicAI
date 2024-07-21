import os

os.system("git clone https://github.com/facebookresearch/ImageBind.git")
os.system("cd ImageBind && pip install .")
os.system("pip install pydub ffmpeg torchaudio_augmentations soundfile")
os.system("conda install -c conda-forge sox")
os.system("pip install tqdm")