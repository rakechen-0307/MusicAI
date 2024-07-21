import os

os.system("git clone https://github.com/vaibhavs10/CLAP")
os.system("cd CLAP && pip install .")
os.system("pip install pydub ffmpeg torchaudio_augmentations soundfile")
os.system("conda install -c conda-forge sox")
os.system("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121")
os.system("pip install --upgrade transformers==4.30.0")
os.system("wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt")