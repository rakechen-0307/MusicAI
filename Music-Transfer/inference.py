import torch
import torchaudio
import soundfile as sf

from music_transfer_pipeline import MusicTransferPipeline

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = MusicTransferPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipeline = pipeline.to(device)

    input_file = "./test/mix15s.mp3"
    audio_input, audio_sr = torchaudio.load(input_file)
    audio_input = audio_input.to(device=device, dtype=torch.float16).unsqueeze(0)

    prompt = ""

    generator = torch.Generator("cuda").manual_seed(0)
    audio_output = pipeline(
        prompt=prompt,
        audio_end_in_s=15.0,
        guidance_scale=7.0,
        num_inference_steps=200,
        generator=generator,
        initial_audio_waveforms=audio_input,
        initial_audio_sampling_rate=audio_sr
    ).audios

    output = audio_output[0].T.float().cpu().numpy()
    sf.write("output.wav", output, pipeline.vae.sampling_rate)

if __name__ == "__main__":
    main()