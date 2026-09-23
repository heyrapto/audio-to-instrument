import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from src.inference.pipeline import convert_audio
from src.utils.config import load_config
from src.losses.spectral import spectral_loss
from src.audio.preprocessing import preprocess_audio

def evaluate_model(eval_dir, output_dir, checkpoint, base_cfg, instrument_cfg, device):
    input_files = list(Path(eval_dir).glob("*.wav"))
    if not input_files:
        print(f"No .wav files found in {eval_dir}")
        return
        
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_loss = 0.0
    
    print(f"Evaluating model on {len(input_files)} files...")
    for f in tqdm(input_files):
        # We assume for evaluation that input audio and target audio are the same
        # for an auto-encoder evaluation, or you can supply paired data.
        # This script runs the conversion and calculates the spectral difference
        out_path = out_dir / f"eval_{f.name}"
        
        # 1. Convert
        convert_audio(str(f), str(out_path), checkpoint, base_cfg, instrument_cfg, device)
        
        # 2. Compare Target vs Generated
        target_audio = preprocess_audio(str(f), target_sr=base_cfg["audio"]["sample_rate"]).to(device)
        generated_audio = preprocess_audio(str(out_path), target_sr=base_cfg["audio"]["sample_rate"]).to(device)
        
        min_len = min(target_audio.shape[-1], generated_audio.shape[-1])
        loss = spectral_loss(generated_audio[..., :min_len], target_audio[..., :min_len])
        total_loss += loss.item()
        
    print(f"Mean Spectral Evaluation Loss: {total_loss / len(input_files):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DDSP Model")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing target evaluation WAVs")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluations", help="Directory to save generated eval WAVs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--instrument_config", type=str, default="configs/keyboard.yaml", help="Path to instrument config")
    
    args = parser.parse_args()
    
    base_cfg = load_config(args.config)
    instrument_cfg = load_config(args.instrument_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_model(
        args.eval_dir, 
        args.output_dir, 
        args.checkpoint, 
        base_cfg, 
        instrument_cfg, 
        device
    )

if __name__ == "__main__":
    main()
