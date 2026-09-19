# Audio to Instrument
A PyTorch-based DDSP (Differentiable Digital Signal Processing) system for converting audio performances into realistic instrument audio.

## Goal
Input: `singing.wav` or `humming.wav`
Output: `keyboard.wav`

## Architecture
Audio -> preprocessing -> pitch/loudness extraction -> neural expression model -> differentiable DDSP synthesizer (Harmonic + Filtered Noise) -> output audio

## Current status
- [x] Audio loading
- [x] Resampling
- [x] Pitch extraction
- [x] Loudness extraction
- [x] Onset detection
- [x] Keyboard dataset (NSynth integration)
- [x] DDSP Harmonic Synthesizer
- [x] DDSP Filtered Noise Synthesizer
- [x] Multi-Scale Spectral Loss (STFT Loss)
- [x] End-to-end Trumpet Model
- [x] Training Loop (gradient clipping, checkpointing)
- [x] Evaluation Script
- [x] Inference CLI

## Usage

### 1. Data Prep
The scripts now automatically download and parse keyboard sounds from the NSynth dataset!
```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py --instrument keyboard
python scripts/extract_features.py --input_dir data/processed/keyboard
```

### 2. Training
```bash
python scripts/train.py --instrument_config configs/keyboard.yaml
```

### 3. Inference
```bash
python scripts/convert.py --input data/raw/input/song.wav --output outputs/generated/keyboard.wav --checkpoint checkpoints/trumpet_epoch_100.pth --instrument_config configs/keyboard.yaml
```
# audio-to-instrument
# audio-to-instrument
