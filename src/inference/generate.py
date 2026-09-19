import torch
import torchaudio

def generate_audio(model, pitch, loudness, output_path, sample_rate):
    model.eval()
    with torch.no_grad():
        audio = model(pitch, loudness)
        
    torchaudio.save(output_path, audio.cpu(), sample_rate)
