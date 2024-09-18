import os
from tqdm import tqdm

data_dir = "./fma_small"
files = sorted(os.listdir(data_dir))
for i in tqdm(range(len(files))):
    os.system("demucs --two-stems=vocals --mp3 --mp3-preset 2 {filename}".format(
        filename = data_dir + "/" + files[i]
    ))