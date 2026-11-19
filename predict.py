#!/usr/bin/env python
"""
Apply a trained model and map the [N, J] output back to the jagged jet axis.

Returns / writes one score per jet, aligned 1:1 with ``jets_pt`` of the input
tree.  Jets failing isSel, or beyond ``max_jets`` (truncated at input), get NaN.

Example:
    python predict.py runs/test/best.pt --max-files 2 --out scores.root
"""

from __future__ import annotations

import argparse

import awkward as ak
import numpy as np
import torch

from data import (DEFAULT_INPUT, EventBatcher, InputConfig, Standardizer,
                  list_files, load_arrays, to_tensors)
from models import build_model


def load_model(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], *ck["dims"], **ck["model_kwargs"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), InputConfig(**ck["input_config"]), Standardizer(ck["scaler"]), ck


@torch.no_grad()
def predict_padded(model, arrays, scaler, batch_size=4096, device="cpu") -> np.ndarray:
    batcher = EventBatcher(to_tensors(scaler.transform(arrays)), batch_size, shuffle=False, device=device)
    return torch.cat([torch.sigmoid(model(b)).cpu() for b in batcher]).numpy()


def to_jagged(scores: np.ndarray, jet_index: np.ndarray, njets_all: np.ndarray) -> ak.Array:
    """[N, J] padded scores (selected jets) -> jagged [N, njets_all] over the
    original jets_* collection; NaN where no score exists."""
    full = np.full((len(njets_all), int(njets_all.max())), np.nan, dtype=np.float32)
    ev, slot = np.nonzero(jet_index >= 0)
    full[ev, jet_index[ev, slot]] = scores[ev, slot]
    keep = np.arange(full.shape[1])[None, :] < njets_all[:, None]
    return ak.unflatten(full[keep], njets_all)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="ROOT file with tree 'Events' (event_number, jets_nn_score), usable as friend tree")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, cfg, scaler, ck = load_model(args.checkpoint, device)
    arrays = load_arrays(list_files(args.input_dir, args.max_files), cfg)

    scores = to_jagged(predict_padded(model, arrays, scaler, device=device),
                       arrays["jet_index"], arrays["njets_all"])

    # sanity: model trained on split s should be applied to the other parity
    other = arrays["event"] % 2 != ck["split"]
    print(f"{len(scores):,} events, {other.mean():.1%} from the held-out parity")
    print("first event scores:", scores[0].tolist())

    if args.out:
        import uproot
        with uproot.recreate(args.out) as f:
            f.mktree("Events", {"event_number": arrays["event"].dtype, "jets_nn_score": scores.type})
            f["Events"].extend({"event_number": arrays["event"], "jets_nn_score": scores})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
