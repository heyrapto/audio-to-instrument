import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from src.inference.pipeline import convert_audio
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Convert an audio file to instrument using a trained DDSP model")
    parser.add_argument("--input", type=str, required=True, help="Input audio file (e.g., singing.wav)")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--instrument_config", type=str, default="configs/keyboard.yaml", help="Path to instrument config")
    
    args = parser.parse_args()
    
    base_cfg = load_config(args.config)
    instrument_cfg = load_config(args.instrument_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    convert_audio(
        input_path=args.input,
        output_path=args.output,
        model_checkpoint=args.checkpoint,
        base_cfg=base_cfg,
        instrument_cfg=instrument_cfg,
        device=device
    )

if __name__ == "__main__":
    main()
