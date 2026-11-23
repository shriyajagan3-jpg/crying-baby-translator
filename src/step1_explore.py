"""Step 1: explore the Donate-a-Cry dataset and make example spectrograms.

Usage:
    python src/step1_explore.py --data data/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data

Outputs (in outputs/):
    metadata.csv          one row per clip: path, label, baby_id, gender, age, duration
    class_counts.png      bar chart of clips per label
    example_spectrograms.png  one mel-spectrogram per label
"""
import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SR = 16000  # sample rate we resample everything to
N_MELS = 64

AGE_BANDS = {
    "04": "0-4 weeks",
    "48": "4-8 weeks",
    "26": "2-6 months",
    "72": "7 months-2 years",
    "22": "over 2 years",
}


def parse_filename(path: Path) -> dict:
    """Filenames look like <app-id>-<timestamp>-<version>-<gender>-<age>-<reason>.wav.
    The app id is the closest thing we have to a baby id (one phone = usually one baby)."""
    parts = path.stem.split("-")
    gender, age, _reason = parts[-3], parts[-2], parts[-1]
    baby_id = "-".join(parts[:5])  # the app id is a UUID (5 dash-separated chunks)
    return {"baby_id": baby_id, "gender": gender, "age": AGE_BANDS.get(age, age)}


def build_metadata(data_dir: Path) -> pd.DataFrame:
    rows = []
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for wav in sorted(label_dir.glob("*.wav")):
            y, sr = librosa.load(wav, sr=SR)
            rows.append(
                {
                    "path": str(wav),
                    "label": label_dir.name,
                    **parse_filename(wav),
                    "duration_s": round(len(y) / sr, 2),
                }
            )
    return pd.DataFrame(rows)


def mel_spectrogram(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)


def plot_class_counts(df: pd.DataFrame, out: Path) -> None:
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_title("Clips per label (Donate-a-Cry, cleaned)")
    ax.set_ylabel("Number of clips")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_examples(df: pd.DataFrame, out: Path) -> None:
    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.5))
    for ax, label in zip(axes, labels):
        path = df[df["label"] == label].iloc[0]["path"]
        S = mel_spectrogram(path)
        librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(label)
    fig.suptitle("Example mel-spectrogram for each label")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", default=Path("outputs"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = build_metadata(args.data)
    df.to_csv(args.out / "metadata.csv", index=False)

    print(f"Total clips: {len(df)}")
    print(f"Different babies (phones): {df['baby_id'].nunique()}")
    print(f"Clip length: {df['duration_s'].min()}s to {df['duration_s'].max()}s "
          f"(average {df['duration_s'].mean():.1f}s)")
    print("\nClips per label:")
    print(df["label"].value_counts().to_string())
    print("\nBabies per label:")
    print(df.groupby("label")["baby_id"].nunique().to_string())

    plot_class_counts(df, args.out / "class_counts.png")
    plot_examples(df, args.out / "example_spectrograms.png")
    print(f"\nSaved charts and metadata.csv to {args.out}/")


if __name__ == "__main__":
    main()
