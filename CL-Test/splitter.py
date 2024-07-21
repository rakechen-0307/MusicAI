import os
import cv2
from moviepy.editor import *

files = sorted(os.listdir("./Original"))

for i in range(len(files)):

    origin = "./Original/" + files[i]
    cap = cv2.VideoCapture(origin)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Checking Video {i+1} Frames {frames} fps: {fps}')

    if (frames != 0):
        video = VideoFileClip(origin)

        if ((i + 1) < 10):
            num = "0000" + str(i + 1)
        elif ((i + 1) < 100):
            num = "000" + str(i + 1)
        elif ((i + 1) < 1000):
            num = "00" + str(i + 1)
        elif ((i + 1) < 10000):
            num = "0" + str(i + 1)
        else:
            num = str(i + 1)

        ## extract audio
        audio_output = "./Audios/" + num + ".mp3"
        print(audio_output)
        video.audio.write_audiofile(audio_output)

        ## extract video
        video_output = "./Videos/" + num + ".mp4"
        video = video.without_audio()
        print(video_output)
        video.write_videofile(video_output)