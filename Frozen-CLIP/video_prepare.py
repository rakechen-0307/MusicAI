import os
import cv2
from tqdm import tqdm
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_extract_audio

input_dir = "./originals"
input_files = sorted(os.listdir(input_dir))
output_video_dir = "./video_segs"
output_audio_dir = "./audio_segs"
if (os.path.isdir(output_video_dir)):
    os.rmdir(output_video_dir)
os.mkdir(output_video_dir)
if (os.path.isdir(output_audio_dir)):
    os.rmdir(output_audio_dir)
os.mkdir(output_audio_dir)

seg_length = 10
step = 7
start = 0.2
end = 0.8

for i in tqdm(range(len(input_files))):
    video = os.path.join(input_dir, input_files[i])
    cap = cv2.VideoCapture(video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Checking Video {i+1} Frames {frames} fps: {fps}')

    if (frames != 0):
        clip = VideoFileClip(video)
        duration = clip.duration  # in seconds

        if ((i + 1) < 10):
            dir_num = "0000" + str(i + 1)
        elif ((i + 1) < 100):
            dir_num = "000" + str(i + 1)
        elif ((i + 1) < 1000):
            dir_num = "00" + str(i + 1)
        elif ((i + 1) < 10000):
            dir_num = "0" + str(i + 1)
        else:
            dir_num = str(i + 1)

        video_dir = os.path.join(output_video_dir, dir_num)
        audio_dir = os.path.join(output_audio_dir, dir_num)
        os.mkdir(video_dir)
        os.mkdir(audio_dir)
        clip_start = duration * start
        clip_end = clip_start + seg_length
        count = 0
        while (clip_end < duration * end):
            if ((count + 1) < 10):
                file_num = "00" + str(count + 1)
            elif ((count + 1) < 100):
                file_num = "0" + str(count + 1)
            else:
                file_num = str(count + 1)
            
            video_file = os.path.join(video_dir, "{}.mp4".format(file_num))
            audio_file = os.path.join(audio_dir, "{}.mp3".format(file_num))
            ffmpeg_extract_subclip(video, clip_start, clip_end, targetname=video_file)
            ffmpeg_extract_audio(video_file, audio_file)

            clip_start += step
            clip_end = clip_start + seg_length
            count += 1