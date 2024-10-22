import os
import torch
import soundfile as sf
from diffuser_pipeline import MusicTransferPipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
repo_id = "stabilityai/stable-audio-open-1.0"
pipeline_trained = MusicTransferPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipeline_trained = pipeline_trained.to(device)

audio_file = "test/mix47s.mp3"
positive_text_prompt = "high quality music"
negative_text_prompt = "noisy, distortion"

generator = torch.Generator(device=device).manual_seed(0)
audio = pipeline_trained(
    audio_path=audio_file,
    noise_scale=0.2,
    prompt=positive_text_prompt,
    negative_prompt=negative_text_prompt,
    num_inference_steps=50,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
sf.write("output.wav", output, pipeline_trained.vae.sampling_rate)