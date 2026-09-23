import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader

from src.models.trumpet_model import TrumpetModel
from src.losses.total import total_loss
from src.training.dataset import InstrumentDataset
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="Train the DDSP Model")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--instrument_config", type=str, default="configs/keyboard.yaml", help="Path to instrument config")
    parser.add_argument("--training_config", type=str, default="configs/training.yaml", help="Path to training config")
    args = parser.parse_args()

    # Load configs
    base_cfg = load_config(args.config)
    instrument_cfg = load_config(args.instrument_config)
    train_cfg = load_config(args.training_config)
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    feature_dir = base_cfg["paths"]["features"]
    dataset = InstrumentDataset(
        data_dir=feature_dir, 
        segment_frames=250, # 2.5 seconds at 100Hz (hop=160, sr=16k)
        hop_length=base_cfg["audio"]["hop_length"]
    )
    
    if len(dataset) == 0:
        print(f"No features found in {feature_dir}. Please run scripts/extract_features.py first.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg["training"]["batch_size"], 
        shuffle=True, 
        num_workers=train_cfg["training"].get("num_workers", 0),
        drop_last=True
    )

    # Model
    model = TrumpetModel(
        hidden_size=base_cfg["model"]["hidden_size"],
        sample_rate=base_cfg["audio"]["sample_rate"],
        harmonics=instrument_cfg["instrument"]["harmonic_count"],
        noise_bands=instrument_cfg["instrument"]["noise_bands"],
        hop_length=base_cfg["audio"]["hop_length"]
    ).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg["training"]["learning_rate"],
        weight_decay=train_cfg["training"]["weight_decay"]
    )
    loss_fn = total_loss

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=base_cfg["paths"]["checkpoints"]
    )

    # Train
    trainer.train(
        dataloader=dataloader, 
        epochs=train_cfg["training"]["epochs"],
        save_every=train_cfg["training"]["save_every"]
    )

if __name__ == "__main__":
    main()
