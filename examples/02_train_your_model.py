"""
Train-your-own-model example for NN4Soot.

This example uses synthetic data so it can run without the full soot dataset.
"""

from pathlib import Path

import numpy as np
import torch

from nn4soot import SootMLP, SootTrainer, TrainingConfig


np.random.seed(42)
n_samples = 500
input_dim = 10
output_dim = 20

inputs = np.random.rand(n_samples, input_dim).astype(np.float32)
outputs = (np.random.randn(n_samples, output_dim) * 0.5 + 8.0).astype(np.float32)

print(f"Training data: {n_samples} samples")
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {outputs.shape}")

model = SootMLP(input_dim=input_dim, output_dim=output_dim)
print(f"\nModel parameters: {model.get_num_parameters():,}")

output_dir = Path(__file__).resolve().parents[1] / "results" / "example_training"
output_dir.mkdir(parents=True, exist_ok=True)

config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    early_stop_patience=20,
    save_dir=str(output_dir),
    checkpoint_name="example_best_MLP_Hp={hp:.2f}.pth",
)

trainer = SootTrainer(model, config)
history = trainer.train(inputs, outputs, hp=0.70, verbose=True)
history_path = trainer.save_training_history(hp=0.70, n_samples=n_samples)
plot_path = trainer.plot_training_curves(hp=0.70, n_samples=n_samples)

model_path = output_dir / "my_model.pth"
torch.save(model.state_dict(), model_path)

print("\nTraining completed.")
print(f"Final train loss: {history['train_loss'][-1]:.6f}")
print(f"Final test loss: {history['test_loss'][-1]:.6f}")
print(f"Model saved to: {model_path}")
print(f"Loss history saved to: {history_path}")
print(f"Training plot saved to: {plot_path}")