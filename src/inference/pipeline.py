import torch
import warnings
from pathlib import Path

from src.audio.preprocessing import preprocess_audio
from src.features.extraction import extract_features
from src.utils.audio import save_audio
from src.models.trumpet_model import TrumpetModel
from src.training.checkpoint import load_checkpoint

# Ignore librosa warnings during inference
warnings.filterwarnings("ignore")

def convert_audio(input_path: str, output_path: str, model_checkpoint: str, base_cfg: dict, instrument_cfg: dict, device: torch.device):
    """
    End-to-end inference pipeline: Audio -> Features -> Model -> Output Audio
    """
    print(f"Loading model from {model_checkpoint}...")
    
    # 1. Initialize model
    model = TrumpetModel(
        hidden_size=base_cfg["model"]["hidden_size"],
        sample_rate=base_cfg["audio"]["sample_rate"],
        harmonics=instrument_cfg["instrument"]["harmonic_count"],
        noise_bands=instrument_cfg["instrument"]["noise_bands"],
        hop_length=base_cfg["audio"]["hop_length"]
    ).to(device)
    
    # 2. Load weights
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Processing input audio: {input_path}")
    
    # 3. Audio preprocessing
    waveform = preprocess_audio(input_path, target_sr=base_cfg["audio"]["sample_rate"])
    waveform = waveform.to(device)
    
    # 4. Feature Extraction
    features = extract_features(waveform, sample_rate=base_cfg["audio"]["sample_rate"])
    pitch = features["pitch"].to(device)
    loudness = features["loudness"].to(device)
    
    # Ensure batched shapes
    if pitch.dim() == 1:
        pitch = pitch.unsqueeze(0)
    if loudness.dim() == 1:
        loudness = loudness.unsqueeze(0)
        
    # Align lengths
    min_len = min(pitch.shape[-1], loudness.shape[-1])
    pitch = pitch[..., :min_len]
    loudness = loudness[..., :min_len]
    
    print("Generating trumpet audio...")
    
    # 5. Inference
    with torch.no_grad():
        generated_audio = model(pitch, loudness)
        
    # 6. Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_audio(generated_audio.cpu(), str(output_path), base_cfg["audio"]["sample_rate"])
    print(f"Successfully saved converted audio to: {output_path}")
