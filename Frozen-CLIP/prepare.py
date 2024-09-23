import os
from tqdm import tqdm

dir1 = "./video_segs"
dir2 = sorted(os.listdir(dir1))
for i in tqdm(range(len(dir2))):
    files = sorted(os.listdir(os.path.join(dir1, dir2[i])))
    for j in range(len(files)):
        fileid = int(files[j].split('.')[0])
        if (fileid >= 10 and "0" + str(fileid) != files[j].split('.')[0]):
            newfilename = "0" + str(fileid) + ".mp3"
            os.system("mv {target} {output}".format(
                target = os.path.join(dir1, dir2[i], files[j]),
                output = os.path.join(dir1, dir2[i], newfilename)
            ))
