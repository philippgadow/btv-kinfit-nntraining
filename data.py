"""
Input pipeline: event-based ROOT tree -> padded per-event tensors.

The Events tree has one row per event with jagged jet / lepton collections.
The network sees the whole event (jets + leptons + MET) but predicts one
label per jet.  We therefore keep the event as the unit of a sample and pad
the jet axis to a fixed length ``max_jets``:

    jets      [N, J, F_jet]   float32   (zero-padded)
    jet_mask  [N, J]          bool      True for real jets
    leptons   [N, L, F_lep]   float32
    lep_mask  [N, L]          bool
    met       [N, F_met]      float32
    label     [N, J]          float32   1 if |hadronFlavour| == 5, padded = 0
    weight    [N]             float32
    event     [N]             uint64    event_number (used for the even/odd split)
    jet_index [N, J]          int32     position of the jet in the original jets_* collection (-1 = pad)
    njets_all [N]             int32     length of the original jets_* collection

Only jets passing the standard selection (``jets_isSel``) are used, as input
and as classification targets.

The loss and metrics use ``jet_mask`` so padded slots never contribute.
``predict.py`` maps the [N, J] output back onto the jagged jet collection.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, asdict

import awkward as ak
import numpy as np
import torch
import uproot

DEFAULT_INPUT = (
    "/data/dust/user/pgadow/btv/phys_btag/sfb-ttkinfit/arrays_transformer/"
    "arrays_hists_transformer_MC/nominal/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8"
)


@dataclass
class InputConfig:
    max_jets: int = 8           # isSel jets beyond this (pT-ordered) are dropped (max seen: 7)
    max_leptons: int = 2

    # Input features: raw lab-frame kinematics, read from the branches
    # jets_<name>, leptons_<name>, met_<name>.  To add a feature, add its name
    # here and the branch to BASE_BRANCHES.  Never add b-tagging scores.
    # Scaling to mean 0 / width 1 is done later by the Standardizer.
    @property
    def jet_feature_names(self):
        return ["pt", "eta", "phi", "mass"]

    @property
    def lep_feature_names(self):
        return ["pt", "eta", "phi", "mass"]

    @property
    def met_feature_names(self):
        return ["pt", "phi"]

    def to_dict(self):
        return asdict(self)


BASE_BRANCHES = [
    "jets_pt", "jets_eta", "jets_phi", "jets_mass", "jets_flavour", "jets_isSel",
    "leptons_pt", "leptons_eta", "leptons_phi", "leptons_mass",
    "met_pt", "met_phi", "weight", "event_number",
]


def list_files(input_dir: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(input_dir, "*.root")))
    if not files:
        raise FileNotFoundError(f"No ROOT files in {input_dir}")
    return files[:max_files] if max_files else files


# ── awkward -> numpy helpers ────────────────────────────────────────────────
def _pad(x, n: int, fill=0.0, dtype=np.float32) -> np.ndarray:
    """Jagged [N, var] -> dense [N, n], clipping and filling."""
    return ak.to_numpy(ak.fill_none(ak.pad_none(x, n, clip=True), fill)).astype(dtype)


def build_arrays(ev: ak.Array, cfg: InputConfig) -> dict[str, np.ndarray]:
    """Turn one chunk of the Events tree into the padded numpy dict."""
    J, L = cfg.max_jets, cfg.max_leptons

    # keep only jets passing the standard selection; remember where they sat
    sel = ev.jets_isSel
    njets_all = ak.to_numpy(ak.num(ev.jets_pt))
    jet_index = ak.local_index(ev.jets_pt)[sel]
    for b in [k for k in ev.fields if k.startswith("jets_")]:
        ev[b] = ev[b][sel]

    jets = np.stack([_pad(ev[f"jets_{v}"], J) for v in cfg.jet_feature_names], axis=-1)
    leptons = np.stack([_pad(ev[f"leptons_{v}"], L) for v in cfg.lep_feature_names], axis=-1)
    met = np.stack([ak.to_numpy(ev[f"met_{v}"]) for v in cfg.met_feature_names],
                   axis=-1).astype(np.float32)
    njets = ak.to_numpy(ak.num(ev.jets_pt))
    nleps = ak.to_numpy(ak.num(ev.leptons_pt))
    jet_mask = np.arange(J)[None, :] < njets[:, None]
    lep_mask = np.arange(L)[None, :] < nleps[:, None]

    label = _pad(abs(ev.jets_flavour) == 5, J)

    return {
        "jets": jets,
        "jet_mask": jet_mask,
        "leptons": leptons,
        "lep_mask": lep_mask,
        "met": met,
        "label": label,
        "weight": ak.to_numpy(ev.weight).astype(np.float32),
        "event": ak.to_numpy(ev.event_number).astype(np.uint64),
        "njets": njets.astype(np.int32),   # selected jets, untruncated
        # only needed by predict.py to write scores back onto the full jet list
        "jet_index": _pad(jet_index, J, fill=-1, dtype=np.int32),
        "njets_all": njets_all.astype(np.int32),
    }


def load_arrays(files: list[str], cfg: InputConfig, step_size="200 MB",
                cache: str | None = None, verbose=True) -> dict[str, np.ndarray]:
    """Read all files into one padded numpy dict (optionally cached as .npz).

    Reading the ROOT files is the slow part, so pass ``cache`` to store the
    result.  The cache remembers the input configuration and refuses to load
    if the features changed since it was written: then delete it.
    """
    signature = np.array(repr((cfg.to_dict(), cfg.jet_feature_names,
                               cfg.lep_feature_names, cfg.met_feature_names, len(files))))
    if cache and os.path.exists(cache):
        if verbose:
            print(f"Loading cached arrays from {cache}")
        with np.load(cache) as f:
            if "signature" not in f.files or str(f["signature"]) != str(signature):
                raise RuntimeError(f"{cache} was made with different inputs or files; delete it")
            return {k: f[k] for k in f.files if k != "signature"}

    chunks = []
    it = uproot.iterate([f"{p}:Events" for p in files], expressions=BASE_BRANCHES,
                        step_size=step_size, library="ak")
    if verbose:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="reading", unit="chunk")
        except ImportError:
            pass
    for ev in it:
        chunks.append(build_arrays(ev, cfg))
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    if verbose:
        n_trunc = int((out["njets"] > cfg.max_jets).sum())
        print(f"Loaded {len(out['weight']):,} events from {len(files)} files "
              f"({n_trunc} events with > {cfg.max_jets} isSel jets truncated)")
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez(cache, signature=signature, **out)
    return out


# ── Feature standardisation (fitted on training events, stored in checkpoint) ─
class Standardizer:
    """Shift and scale every feature to mean 0, standard deviation 1.
    Only real objects enter the mean/std; padded slots stay exactly 0."""

    KEYS = (("jets", "jet_mask"), ("leptons", "lep_mask"), ("met", None))

    def __init__(self, stats: dict | None = None):
        self.stats = stats or {}

    def fit(self, arrays: dict[str, np.ndarray]):
        for key, mkey in self.KEYS:
            x = arrays[key]
            x = x[arrays[mkey]] if mkey else x          # only real objects
            std = x.std(axis=0)
            self.stats[key] = (x.mean(axis=0), np.where(std > 0, std, 1.0))
        return self

    def transform(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(arrays)
        for key, mkey in self.KEYS:
            mean, std = self.stats[key]
            x = ((arrays[key] - mean) / std).astype(np.float32)
            if mkey:
                x = x * arrays[mkey][..., None]          # keep padding at exactly 0
            out[key] = x
        return out


# ── Batching ────────────────────────────────────────────────────────────────
TENSOR_KEYS = ("jets", "jet_mask", "leptons", "lep_mask", "met", "label", "weight")


def to_tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(arrays[k])) for k in TENSOR_KEYS}


class EventBatcher:
    """Minimal in-memory batch iterator over events (faster than a per-item DataLoader)."""

    def __init__(self, tensors: dict[str, torch.Tensor], batch_size: int,
                 shuffle: bool = True, device="cpu"):
        self.t = tensors
        self.n = len(tensors["weight"])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.batch_size):
            b = idx[i:i + self.batch_size]
            yield {k: v[b].to(self.device, non_blocking=True) for k, v in self.t.items()}


def select(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in arrays.items()}
