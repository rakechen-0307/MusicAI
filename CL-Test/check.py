import os

check = []

audio_path = "./Audio/Data"
frame_path = "./Frames"

audio_dir = sorted(os.listdir(audio_path))
frame_dir = sorted(os.listdir(frame_path))

for i in range(len(audio_dir)):
    audios = sorted(os.listdir(audio_path + "/" + audio_dir[i]))
    frames = sorted(os.listdir(frame_path + "/" + frame_dir[i]))

    if (len(audios) != len(frames)):
        print(str(len(audios)), " ", str(len(frames)))
        check.append(audio_dir[i])

print(check)