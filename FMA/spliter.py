import os
import pandas as pd

data_dir = "./fma"
files = sorted(os.listdir(data_dir))
files_idx = []
for i in range(len(files)):
    if (files[i].split('.')[-1] == "mp3"):
        files_idx.append(files[i].split('.')[0])
print(len(files_idx))

genre_dict = {}
os.mkdir(data_dir + "/" + "train")
os.mkdir(data_dir + "/" + "valid")

csv_file = "./tracks.csv"
df = pd.read_csv(csv_file, encoding='unicode_escape')
for i in range(df.shape[0]):
    if (df['subset'][i] == "small"):
        j = 0
        while (j < len(files_idx) and int(files_idx[j]) != int(df['track_id'][i])):
            j += 1
        if (j == len(files_idx)):
            continue
        genre = df['genre_top'][i]
        if (genre_dict.get(genre) == None):
            genre_dict[genre] = 1
        else:
            genre_dict[genre] += 1
        if (genre_dict[genre] > 970):
            os.system("cp {target} {output}".format(
                target = data_dir + "/" + files_idx[j] + ".mp3",
                output = data_dir + "/" + "valid"
            ))
            os.system("cp {target} {output}".format(
                target = data_dir + "/" + files_idx[j] + ".json",
                output = data_dir + "/" + "valid"
            ))
        else:
            os.system("cp {target} {output}".format(
                target = data_dir + "/" + files_idx[j] + ".mp3",
                output = data_dir + "/" + "train"
            ))
            os.system("cp {target} {output}".format(
                target = data_dir + "/" + files_idx[j] + ".json",
                output = data_dir + "/" + "train"
            ))