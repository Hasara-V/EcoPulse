"""
EcoPulse - Bioacoustic module

"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16000
DURATION = 5.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# Audio formats librosa/soundfile can read. wav/flac/ogg work with no
# extra setup. mp3/m4a/aac/wma go through the "audioread" fallback, which
# needs ffmpeg installed and on PATH (https://ffmpeg.org/download.html --
# on Windows, download a build, unzip, and add the bin/ folder to PATH).
SUPPORTED_AUDIO_EXTS = [".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma"]


def find_audio_files(data_dir):
    """Case-insensitive glob across every supported audio extension."""
    files = []
    for ext in SUPPORTED_AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(data_dir, f"*{ext.upper()}")))
    return sorted(set(files))


# ---------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------
def extract_mel(path, sr=SAMPLE_RATE, n_mels=N_MELS, duration=DURATION):
    """Load an audio file (any supported format) and turn it into a
    normalised, fixed-size log-mel spectrogram."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    except Exception as e:
        raise RuntimeError(
            f"Could not read '{path}'. If this is an mp3/m4a/aac/wma file, "
            f"make sure ffmpeg is installed and on your PATH. Original error: {e}"
        )
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return S_db.astype(np.float32)


def build_dataset(data_dir, background_dir=None, cache_path=None):
    """Scan data_dir for elephant audio files (any supported format) and,
    if background_dir is given, also scan it for background/negative
    sound files (e.g. the ESC-50 set from download_esc50.py). Returns a
    feature matrix + integer labels covering classes:
        ["background", "elephant_1", "elephant_2", ..., "elephant_8"]
    (or just the elephant_* classes if background_dir is not given, same
    as before -- background_dir is optional so this script still works
    with only the original dataset).
    """
    elephant_files = find_audio_files(data_dir)
    if not elephant_files:
        raise FileNotFoundError(
            f"No audio files found in {data_dir} (looked for {SUPPORTED_AUDIO_EXTS})"
        )

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True).item()
        return cache["X"], cache["y"], cache["classes"], cache["files"]

    files = list(elephant_files)
    labels = []
    for f in elephant_files:
        m = re.match(r".*?(elephant_\d+)_part_\d+\.\w+", os.path.basename(f), re.IGNORECASE)
        if not m:
            raise ValueError(f"Unexpected filename pattern: {f}")
        labels.append(m.group(1))

    if background_dir:
        bg_files = find_audio_files(background_dir)
        if not bg_files:
            raise FileNotFoundError(f"No background audio files found in {background_dir}")
        print(f"Adding {len(bg_files)} background/negative clips from {background_dir}")
        files.extend(bg_files)
        labels.extend(["background"] * len(bg_files))

    classes = sorted(set(labels))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2idx[l] for l in labels])

    print(f"Extracting mel-spectrograms for {len(files)} files ...")
    X = np.stack([extract_mel(f) for f in files])

    if cache_path:
        np.save(cache_path, {"X": X, "y": y, "classes": classes, "files": files})

    return X, y, classes, files


# ---------------------------------------------------------------------
# 2. Dataset / Model
# ---------------------------------------------------------------------
class SpecDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spec = self.X[idx].copy()
        if self.augment:
            spec = self._spec_augment(spec)
        return torch.tensor(spec).unsqueeze(0), torch.tensor(self.y[idx])

    @staticmethod
    def _spec_augment(spec, freq_mask=8, time_mask=20):
        spec = spec.copy()
        n_mels, n_frames = spec.shape
        f0 = np.random.randint(0, max(1, n_mels - freq_mask))
        spec[f0:f0 + freq_mask, :] = spec.mean()
        t0 = np.random.randint(0, max(1, n_frames - time_mask))
        spec[:, t0:t0 + time_mask] = spec.mean()
        return spec


class SmallCNN(nn.Module):
    """A compact CNN, deliberately small so it trains fast on CPU/laptop
    GPUs. Mirrors the "lightweight edge-deployable model" idea in the
    proposal."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------
# 3. Train / evaluate
# ---------------------------------------------------------------------
def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, classes, files = build_dataset(
        args.data_dir,
        background_dir=args.background_dir,
        cache_path=str(out_dir / "audio_features_cache.npy"),
    )
    print("Feature matrix:", X.shape, "| classes:", classes)
    if "background" not in classes:
        print("\nNOTE: no --background_dir given, so this model can only tell elephant\n"
              "call signatures apart from EACH OTHER, not from non-elephant sounds.\n"
              "Run download_esc50.py and pass --background_dir to train a model that can\n"
              "genuinely say 'this is/isn't an elephant call'.\n")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_dl = DataLoader(SpecDataset(Xtr, ytr, augment=True), batch_size=16, shuffle=True)
    test_dl = DataLoader(SpecDataset(Xte, yte, augment=False), batch_size=16)

    model = SmallCNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                out = model(xb)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        acc = np.mean(np.array(preds) == np.array(trues))
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss {total_loss/len(Xtr):.4f} | val_acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / "audio_model_best.pt")

    print(f"\nBest validation accuracy: {best_acc:.3f}")

    # Final report using best checkpoint
    model.load_state_dict(torch.load(out_dir / "audio_model_best.pt"))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())

    print("\nClassification report:")
    print(classification_report(trues, preds, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(trues, preds))

    with open(out_dir / "audio_classes.json", "w") as f:
        json.dump(classes, f)

    print(f"\nSaved model to {out_dir / 'audio_model_best.pt'}")
    print(f"Saved class mapping to {out_dir / 'audio_classes.json'}")


def load_inference_model(model_path, classes_path):
    """Helper used by fusion_alert.py to load the trained model."""
    with open(classes_path) as f:
        classes = json.load(f)
    model = SmallCNN(len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, classes


def predict_clip(model, classes, wav_path):
    spec = extract_mel(wav_path)
    x = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()[0]
    top = int(np.argmax(probs))
    return classes[top], float(probs[top]), probs


def elephant_confidence(classes, probs):
    """Given the class list and probability vector from predict_clip,
    return P(this clip is ANY elephant signature) -- i.e. 1 - P(background).
    Returns the top prob unchanged if this model has no background class
    (was trained without --background_dir)."""
    if "background" not in classes:
        return float(np.max(probs))
    bg_idx = classes.index("background")
    return float(1.0 - probs[bg_idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=r"D:\Researches\Data Odyssey 2026\unique\Data Odyssey 2026 Project\data\audio",
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument(
        "--background_dir",
        default=None,
        help="Folder of background/negative sound clips (e.g. output of download_esc50.py). "
             "Strongly recommended -- see README for why.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train(args)
