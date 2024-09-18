from mutagen.mp3 import MP3

audio = MP3("./fma/000005.mp3")
print(audio.info.length)