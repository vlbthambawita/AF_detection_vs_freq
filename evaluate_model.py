import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the model
class ECG1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(256, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
           
            nn.Conv1d(64, 1, 7, padding=3)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ECG1DCNN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Create some dummy data for testing
x = torch.randn(1, 12, 500).to(device)  # Batch size 1, 12 leads, 500 time points
with torch.no_grad():
    pred = model(x)

print(f"Model loaded successfully!")
print(f"Input shape: {x.shape}")
print(f"Output shape: {pred.shape}")
print(f"Sample prediction values: {pred[0, :10].cpu().numpy()}")

# Calculate SNR-like metric
def calculate_snr(original, reconstructed):
    noise = original - reconstructed
    snr = 10 * np.log10(np.mean(original**2) / np.mean(noise**2))
    return snr

# Test on dummy data
original = x[0, 0, :].cpu().numpy()  # Lead I
reconstructed = pred[0, :].cpu().numpy()

mae = mean_absolute_error(original, reconstructed)
mse = mean_squared_error(original, reconstructed)
snr = calculate_snr(original, reconstructed)

print("\nEvaluation Metrics:")
print(f"MAE: {mae:.6f}")
print(f"MSE: {mse:.6f}")
print(f"SNR: {snr:.2f} dB")

# Plot comparison
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(original, label='Original Lead I')
plt.plot(reconstructed, label='Reconstructed Lead I')
plt.legend()
plt.title('ECG Lead Reconstruction')
plt.xlabel('Time points')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(original - reconstructed, label='Error')
plt.legend()
plt.title('Reconstruction Error')
plt.xlabel('Time points')
plt.ylabel('Error')

plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=150)
plt.show()

print("Plot saved as 'reconstruction_comparison.png'")