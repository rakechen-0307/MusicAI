import os

data_dir = "./fma_small"
dirs = sorted(os.listdir(data_dir))
for i in range(len(dirs)):
    files = sorted(os.listdir(data_dir + "/" + dirs[i]))
    for j in range(len(files)):
        os.system("mv {target} {output}".format(
            target = data_dir + "/" + dirs[i] + "/" + files[j],
            output = data_dir
        ))