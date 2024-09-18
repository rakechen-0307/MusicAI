import os
import cv2
from tqdm import tqdm

os.system("rm -r './Frames'")
os.mkdir("./Frames")
video_files = sorted(os.listdir("./Videos"))

for i in tqdm(range(len(video_files))):

  num = video_files[i].split('.')[0]
  os.mkdir("./Frames_Temp")
  video = cv2.VideoCapture("./Videos/" + video_files[i])
  fps = video.get(cv2.CAP_PROP_FPS)

  success, image = video.read()
  count = 1
  while success:
    if (count < 10):
        output_num = "000" + str(count)
    elif (count < 100):
        output_num = "00" + str(count)
    elif (count < 1000):
        output_num = "0" + str(count)
    else:
        output_num = str(count)
    filename = "./Frames_Temp/" + output_num + ".jpg"
    cv2.imwrite(filename, image)
    success, image = video.read()
    count += 1

  os.mkdir("./Frames/" + num)
  frames = sorted(os.listdir("./Frames_Temp"))
  low = round(len(frames)*0.2)
  high = round(len(frames)*0.8)
  step = round(fps * 7)
  part = round(fps * 10)
  count = 1
  while (low + part/2 < high and low + part < len(frames)):
    if (count < 10):
      output_num = "00" + str(count)
    elif (count < 100):
      output_num = "0" + str(count)
    else:
      output_num = str(count)
    output_dir = "./Frames/" + num + "/" + output_num
    os.mkdir(output_dir)

    for j in range(low, low+part, 15):
      os.system("cp ./Frames_Temp/{target} {output}".format(
          target=frames[j],
          output=output_dir
      ))
    count += 1
    low += step
  os.system("rm -r './Frames_Temp'")