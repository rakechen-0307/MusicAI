import os
import json
from mutagen.mp3 import MP3

data_dir = "./fma"
json_temp = "./template.json"
files = sorted(os.listdir(data_dir))
for i in range(len(files)):
    json_file = data_dir + "/" + files[i].split('.')[0] + ".json"
    audio = MP3(data_dir + "/" + files[i])
    audio_length = audio.info.length

    with open(json_temp, 'r') as f:
        data = json.load(f)
        data["duration"] = audio_length
        data["name"] = files[i].split('.')[0]
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)