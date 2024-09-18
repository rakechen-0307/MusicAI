import os

check = []

audio_path = "./Audios/Data"
frame_path = "./Frames"

audio_dir = sorted(os.listdir(audio_path))
frame_dir = sorted(os.listdir(frame_path))

for i in range(len(audio_dir)):
    audios = sorted(os.listdir(audio_path + "/" + audio_dir[i]))
    frames = sorted(os.listdir(frame_path + "/" + frame_dir[i]))

    if (len(audios) != len(frames)):
        check.append(audio_dir[i])

print(check)

for i in range(len(check)):
    audios = sorted(os.listdir(audio_path + "/" + check[i]))
    frames = sorted(os.listdir(frame_path + "/" + check[i]))
    
    if (len(audios) > len(frames)):
        os.system("rm -r {}".format(
            audio_path + "/" + check[i] + "/" + audios[len(audios)-1]
        ))
    else:
        os.system("rm -r {}".format(
            frame_path + "/" + check[i] + "/" + frames[len(frames)-1]
        ))
