import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.audio.preprocessing import preprocess_audio
from src.features.extraction import extract_features
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Extract DDSP features from audio")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--input_dir", type=str, default="data/processed/keyboard", help="Directory with processed wavs")
    parser.add_argument("--output_dir", type=str, default="data/processed/features", help="Directory to save npz features")
    args = parser.parse_args()

    config = load_config(args.config)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(input_dir.glob("*.wav"))
    if not audio_files:
        print(f"No .wav files found in {input_dir}")
        return
        
    print(f"Extracting features for {len(audio_files)} files...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for audio_path in tqdm(audio_files):
        try:
            # 1. Preprocess
            waveform = preprocess_audio(str(audio_path), target_sr=config["audio"]["sample_rate"])
            waveform = waveform.to(device)
            
            # 2. Extract Features
            features = extract_features(waveform, sample_rate=config["audio"]["sample_rate"])
            
            # 3. Move back to CPU and convert to numpy
            pitch = features["pitch"].cpu().numpy()
            confidence = features["confidence"].cpu().numpy()
            loudness = features["loudness"].cpu().numpy()
            audio_np = waveform.squeeze(0).cpu().numpy()
            
            # Align lengths in case of hop_length mismatches
            min_len = min(pitch.shape[-1], confidence.shape[-1], loudness.shape[-1])
            pitch = pitch[..., :min_len]
            confidence = confidence[..., :min_len]
            loudness = loudness[..., :min_len]
            
            # 4. Save
            output_path = output_dir / f"{audio_path.stem}.npz"
            np.savez(
                output_path,
                audio=audio_np,
                pitch=pitch,
                confidence=confidence,
                loudness=loudness
            )
        except Exception as e:
            print(f"Failed processing {audio_path.name}: {e}")

if __name__ == "__main__":
    main()
