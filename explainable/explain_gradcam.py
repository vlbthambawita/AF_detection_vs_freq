from __future__ import annotations
import os
import sys
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.cnn1d import CNN1D
from models.cnn_lstm import CNN_LSTM_ECG

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble Grad-CAM on ECG Data")
    parser.add_argument("--frequency", type=str, required=True, choices=["62", "100", "250", "500"])
    parser.add_argument("--model_type", type=str, required=True, choices=["cnn1d", "cnn_lstm"])
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--representative_case", type=str, default=None,
                        choices=["correct_afib", "correct_normal", "borderline"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="outputs/gradcam")
    return parser.parse_args()

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def load_ensemble_models(checkpoint_dir: Path, model_type: str, device: torch.device):
    models = []
    fold_dirs = sorted(checkpoint_dir.glob("fold_*"))
    for f_dir in fold_dirs:
        best_pt = f_dir / "best.pt"
        if best_pt.exists():
            if model_type == "cnn1d":
                model = CNN1D(in_channels=12, num_classes=2)
            else:
                model = CNN_LSTM_ECG(in_channels=12, num_classes=2)
            
            checkpoint = torch.load(best_pt, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            models.append(model)
    return models

def main():
    args = parse_args()
    device = get_device(args.device)
    
    prepared_root = PROJECT_ROOT / "prepared_data"
    matching_folders = list(prepared_root.glob("ptb*"))
    
    if not matching_folders:
        print(f"Error: No dataset folder (ptb-xl/ptbl-xl) found in {prepared_root}")
        sys.exit(1)
        
    dataset_folder_name = matching_folders[0].name
    
    data_path = prepared_root / dataset_folder_name / f"{args.frequency}hz" / "test" / "test.pt"
    checkpoint_root = PROJECT_ROOT / "checkpoints"
    checkpoint_dir = checkpoint_root / dataset_folder_name / f"{args.frequency}hz" / args.model_type
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: Test data file not found at {data_path}")
        sys.exit(1)

    test_data = torch.load(data_path, map_location=device)
    X = test_data["X"]
    y = test_data["y"]
    
    models = load_ensemble_models(checkpoint_dir, args.model_type, device)
    if not models:
        print(f"Error: No trained folds found at {checkpoint_dir}")
        sys.exit(1)

    idx = args.sample_idx
    if idx is None and args.representative_case:
        print(f"Finding a {args.representative_case} representative case sample...")
        all_probs = []
        with torch.no_grad():
            for i in range(len(X)):
                inputs = X[i:i+1].to(device)
                outputs = [torch.softmax(m(inputs), dim=1) for m in models]
                avg_prob = torch.stack(outputs).mean(dim=0)
                all_probs.append(avg_prob.cpu().numpy()[0])
        all_probs = np.array(all_probs)
        preds = np.argmax(all_probs, axis=1)
        true_labels = y.cpu().numpy()

        if args.representative_case == "correct_afib":
            candidates = np.where((preds == 1) & (true_labels == 1))[0]
        elif args.representative_case == "correct_normal":
            candidates = np.where((preds == 0) & (true_labels == 0))[0]
        else:  
            candidates = np.where((all_probs[:, 1] > 0.4) & (all_probs[:, 1] < 0.6))[0]

        if len(candidates) > 0:
            idx = int(candidates[0])
            print(f"Selected representative sample_idx: {idx}")
        else:
            idx = 0
            print("Fallback: No matches found. Selecting sample_idx: 0")
    elif idx is None:
        idx = 0

    print(f"Running Grad-CAM extraction on sample index: {idx}")
    input_tensor = X[idx:idx+1].to(device)
    input_tensor.requires_grad = True

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    if args.model_type == "cnn1d":
        target_layer = models[0].features[-2]
    else:
        target_layer = models[0].cnn[-1].conv

    def hook_fn(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook = target_layer.register_forward_hook(hook_fn)

    scores = []
    for model in models:
        out = model(input_tensor)
        scores.append(out)
    
    avg_scores = torch.stack(scores).mean(dim=0)
    target_class = torch.argmax(avg_scores, dim=1).item()
    
    models[0].zero_grad()
    avg_scores[0, target_class].backward()
    hook.remove()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=1, keepdims=True)
    
    cam = np.sum(weights * acts, axis=0)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    input_len = input_tensor.shape[-1]
    cam_upsampled = np.interp(
        np.linspace(0, len(cam) - 1, input_len),
        np.arange(len(cam)),
        cam
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ecg_signal = input_tensor[0, 0].detach().cpu().numpy()
    time_axis = np.arange(input_len) / float(args.frequency)
    
    ax.plot(time_axis, ecg_signal, color='black', alpha=0.5, label='ECG Lead 0')
    scatter = ax.scatter(time_axis, ecg_signal, c=cam_upsampled, cmap='jet', s=3, label='Grad-CAM Heat')
    
    plt.colorbar(scatter, ax=ax, label="Relevance Score")
    plt.title(f"Grad-CAM (Freq: {args.frequency}Hz | Case: {args.model_type.upper()} | Sample: {idx})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Voltage")
    
    save_filename = output_path / f"gradcam_sample_{idx}.png"
    plt.savefig(save_filename, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved Grad-CAM plot to: {save_filename}")

if __name__ == "__main__":
    main()