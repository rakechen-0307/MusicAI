# Youtube-8M Downloader

This repository provides a moudle to download the original videos in [Youtube-8m dataset](https://research.google.com/youtube8m/index.html)

Since the official `youtube-8m dataset` website contains only videos and frame level features in the format of tensorflow protocol buffers. Hence, in this repository I write a tool to download the original videos.

## Conda Environment

```
conda create --name {env-name} python=3.11
```

## Installation

Dependencies for downloading youtube video ids for categories

```
pip install requests progressbar2
```

We use `yt-dlp` to download the original youtube video
https://ivonblog.com/posts/yt-dlp-installation/

## Preparation

1. Open `categories.txt`.
2. Select the categories and paste them into `downloadlist.txt`. Note, there is only one category for each line and the first letter of each category is Capitalized.
3. Save `downloadlist.txt`.

## Download category videos and ids

```
python downloader.py
```

The IDs of each category are saved at the folder 'ID'. The file of ID are named as the categories.

The Videos of each category are saved at the folder 'videos\YOUR CATEGORY NAME'. By default a video is downloaded in the best possible resolution.
