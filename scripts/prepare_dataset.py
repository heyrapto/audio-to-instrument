import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from src.audio.preprocessing import preprocess_audio
from src.utils.audio import save_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, default="keyboard")
    args = parser.parse_args()

    raw_dir = Path(f"data/raw/{args.instrument}")
    output_dir = Path(f"data/processed/{args.instrument}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print(f"Directory {raw_dir} does not exist.")
        return
        
    print(f"Preparing {args.instrument} dataset...")
    for path in raw_dir.glob("*.wav"):
        waveform = preprocess_audio(str(path))
        save_audio(waveform, output_dir / path.name, 16000)

if __name__ == "__main__":
    main()
