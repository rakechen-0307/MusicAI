import numpy as np
import torch
import torchaudio
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

audio_file = "./test/orig1.mp3"
enocder = DACEncoderWrapper(**encoder_config)
decoder = DACDecoderWrapper(**decoder_config)
bottleneck = VAEBottleneck()
audioVAE = AudioAutoencoder(encoder=enocder, decoder=decoder, latent_dim=model_config["latent_dim"],
                            downsampling_ratio=model_config["downsampling_ratio"], sample_rate=model_config["sample_rate"],
                            io_channels=model_config["io_channels"], bottleneck=bottleneck)

audio, sr = torchaudio.load(audio_file, format='mp3')
audio = audioVAE.preprocess_audio_for_encoder(audio, sr)
latents = audioVAE.encode_audio(audio, chunked=False)

print(latents)