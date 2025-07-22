from __future__ import annotations

import glob
import os
import re
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

TARGET_SR: int = 16_000       # Hz
SEG_SAMPLES: int = 32_000     # 2 s at 16 kHz

# Regex to parse official file names
NAME_RE = re.compile(
    r"^(train|eval|test)_(?P<drone>[ABC])_"
    r"(?P<dir>Front|Back|Right|Left|Clockwise|CounterClockwise)"
    r"(?:_(?P<flag>normal|anomaly))?_\d+\.wav$"
)


def _fix_length(x: np.ndarray, lgth: int) -> np.ndarray:
    """Pad or trim waveform to *exactly* `lgth` samples (wrap‑pad)."""
    if x.shape[0] == lgth:
        return x
    if x.shape[0] > lgth:
        return x[:lgth]
    pad = lgth - x.shape[0]
    return np.pad(x, (0, pad), mode="wrap")


def load_split(root: str, split: str, encoder: LabelEncoder | None = None,
               return_ids: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str] | None, LabelEncoder]:
    """Load one of the data splits.

    Parameters
    ----------
    root       : root folder (e.g. "data_icsv")
    split      : "train", "eval" or "test"
    encoder    : optional *shared* `LabelEncoder`; when `None` a new one is
                 fitted (use same encoder for all splits to keep label space
                 consistent).
    return_ids : whether to also return the string pseudo‑class ids.
    """
    assert split in {"train", "eval", "test"}
    files = sorted(glob.glob(os.path.join(root, split, "*.wav")))
    if not files:
        raise RuntimeError(f"No wavs found in {root}/{split}")

    waves: List[np.ndarray] = []
    ids:   List[str] = []

    for fp in files:
        fname = os.path.basename(fp)
        m = NAME_RE.match(fname)
        if m is None:
            raise ValueError(f"Unexpected file name pattern: {fname}")
        drone = m.group("drone")
        direction = m.group("dir")
        flag = m.group("flag")  # None for test
        if split == "train" and flag != "normal":
            # train split should contain only normal clips
            continue

        # read mono waveform
        wav, sr = sf.read(fp)
        if wav.ndim > 1:
            wav = librosa.to_mono(wav.T)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        wav = _fix_length(wav, SEG_SAMPLES)
        waves.append(wav)
        ids.append(f"{drone}_{direction}")

    X = np.expand_dims(np.asarray(waves, dtype=np.float32), -1)  # (N, 32000, 1)
    if encoder is None:
        encoder = LabelEncoder().fit(ids)
    y = encoder.transform(ids)

    if return_ids:
        return X, y, ids, encoder
    return X, y, None, encoder


def load_all(root: str = "data_icsv"):
    """Convenience wrapper that loads *all* splits with a shared encoder."""
    X_train, y_train, _, enc = load_split(root, "train", None, False)
    X_eval, y_eval, _, _ = load_split(root, "eval", enc, False)
    X_test, y_test, ids_test, _ = load_split(root, "test", enc, True)
    return (X_train, y_train), (X_eval, y_eval), (X_test, y_test, ids_test), enc


if __name__ == "__main__":
    (Xtr, ytr), (Xev, yev), (Xte, yte, ids), le = load_all()
    print("Loaded", Xtr.shape, Xev.shape, Xte.shape)
    print("Pseudo‑classes:", le.classes_)
