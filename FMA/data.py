import os

input_dir = "./separated/htdemucs"
output_dir = "./fma"
files = sorted(os.listdir(input_dir))
for i in range(len(files)):
    os.system("mv {target} {output}".format(
        target = input_dir + "/" + files[i] + "/" + "no_vocals.mp3",
        output = output_dir + "/" + files[i] + ".mp3"
    )) 