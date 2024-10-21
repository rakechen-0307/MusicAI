import gc
import torch
import torchaudio
from einops import rearrange
from vae import DACEncoderWrapper, DACDecoderWrapper, VAEBottleneck, AudioAutoencoder

encoder_config = {
    "in_channels": 2,
    "latent_dim": 128,
    "d_model": 128,
    "strides": [4, 4, 8, 8]
}
decoder_config = {
    "out_channels": 2,
    "latent_dim": 64,
    "channels": 1536,
    "rates": [8, 8, 4, 4]
}
model_config = {
    "sample_rate": 44100,
    "latent_dim": 64,
    "downsampling_ratio": 1024,
    "io_channels": 2
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

audio_file = "./test/mix47s.mp3"
enocder = DACEncoderWrapper(**encoder_config).to(device)
decoder = DACDecoderWrapper(**decoder_config).to(device)
bottleneck = VAEBottleneck().to(device)
audioVAE = AudioAutoencoder(encoder=enocder, decoder=decoder, latent_dim=model_config["latent_dim"],
                            downsampling_ratio=model_config["downsampling_ratio"], sample_rate=model_config["sample_rate"],
                            io_channels=model_config["io_channels"], bottleneck=bottleneck).to(device)

audio, sr = torchaudio.load(audio_file, format='mp3')
audio = audioVAE.preprocess_audio_for_encoder(audio.to(device), sr)
latents = audioVAE.encode_audio(audio.to(device), chunked=False)

audio = audioVAE.decode_audio(latents.to(device), chunked=False)
audio = rearrange(audio.to(device), "b d n -> d (b n)")
audio = audio.to(torch.float32).clamp(-1, 1).cpu()
print(audio)
torchaudio.save("output.wav", audio, model_config["sample_rate"])