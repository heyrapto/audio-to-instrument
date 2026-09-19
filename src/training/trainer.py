import torch
from pathlib import Path
from tqdm import tqdm
from .checkpoint import save_checkpoint

class Trainer:
    """
    Handles the training loop, loss computation, and checkpointing.
    """
    def __init__(self, model, optimizer, loss_fn, device, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        audio = batch["audio"].to(self.device)
        pitch = batch["pitch"].to(self.device)
        loudness = batch["loudness"].to(self.device)
        
        predicted_audio = self.model(pitch, loudness)
        
        # Match lengths in case of off-by-one errors from upsampling
        min_len = min(predicted_audio.shape[-1], audio.shape[-1])
        predicted_audio = predicted_audio[..., :min_len]
        audio = audio[..., :min_len]
        
        loss = self.loss_fn(predicted_audio, audio)
        
        loss.backward()
        
        # Gradient clipping for stability in audio models
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def train(self, dataloader, epochs, save_every=5):
        print(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
            
            if epoch % save_every == 0:
                ckpt_path = self.checkpoint_dir / f"trumpet_epoch_{epoch}.pth"
                save_checkpoint(self.model, self.optimizer, epoch, avg_loss, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
