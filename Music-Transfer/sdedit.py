import gc
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from vae import DACEncoderWrapper, DACDecoderWrapper, VAEBottleneck, AudioAutoencoder

encoder_config = {
    "in_channels": 2,
    "latent_dim": 64,  # Changed from 128 to 64
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

audio_file = "./test/mix.mp3"
encoder = DACEncoderWrapper(**encoder_config).to(device)
decoder = DACDecoderWrapper(**decoder_config).to(device)
bottleneck = VAEBottleneck().to(device)
audioVAE = AudioAutoencoder(encoder=encoder, decoder=decoder, bottleneck=bottleneck,
                            latent_dim=model_config["latent_dim"],
                            downsampling_ratio=model_config["downsampling_ratio"],
                            sample_rate=model_config["sample_rate"],
                            io_channels=model_config["io_channels"]).to(device)

# Load pre-trained weights if available
# checkpoint = torch.load('path_to_checkpoint.pth')
# audioVAE.load_state_dict(checkpoint['model_state_dict'])
audioVAE.eval()

# Load and preprocess audio
audio, sr = torchaudio.load(audio_file, format='mp3')
audio = audioVAE.preprocess_audio_for_encoder(audio, sr)

# Encode and decode
with torch.no_grad():
    latents = audioVAE.encode(audio.to(device))
    reconstructed_audio = audioVAE.decode(latents)

# Postprocess and save the output
reconstructed_audio = reconstructed_audio.squeeze().cpu().detach().numpy()
reconstructed_audio = np.clip(reconstructed_audio, -1, 1)
# reconstructed_audio = (reconstructed_audio * 32767).astype(np.int16)
wavfile.write("output.wav", model_config["sample_rate"], reconstructed_audio.T)