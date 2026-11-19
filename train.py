#!/usr/bin/env python
"""
Train a per-jet b vs non-b classifier on full-event inputs.

Split convention (same as the XGBoost kinematic BDT): a model trained with
``--split s`` uses events with event_number % 2 == s, and is meant to be
applied to the other parity.  A random ``--val-frac`` of the training-parity
events is held out for validation / early stopping.

Example:
    python train.py --model transformer --max-files 50 --epochs 5 --outdir runs/test
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import (DEFAULT_INPUT, EventBatcher, InputConfig, Standardizer,
                  list_files, load_arrays, select, to_tensors)
from models import MODELS, build_model


def masked_bce(logits, batch):
    """Binary cross-entropy per jet slot [B, J], weighted by the event weight.
    Padded slots get weight 0, so the result is the average over real jets."""
    w = batch["weight"][:, None] * batch["jet_mask"].float()     # [B, J]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch["label"], reduction="none")               # [B, J]
    return (loss * w).sum() / w.sum().clamp(min=1e-12)


@torch.no_grad()
def evaluate(model, batcher):
    model.eval()
    tot_loss, tot_w, scores, labels, weights, masks = 0.0, 0.0, [], [], [], []
    for batch in batcher:
        logits = model(batch)
        w = (batch["weight"][:, None] * batch["jet_mask"]).sum().item()
        tot_loss += masked_bce(logits, batch).item() * w
        tot_w += w
        scores.append(torch.sigmoid(logits).cpu())
        labels.append(batch["label"].cpu())
        weights.append(batch["weight"].cpu())
        masks.append(batch["jet_mask"].cpu())
    s, y, w, m = (torch.cat(x).numpy() for x in (scores, labels, weights, masks))

    # jet-level metrics: flatten over real jets only
    w_jet = np.broadcast_to(w[:, None], m.shape)[m]
    auc = roc_auc_score(y[m], s[m], sample_weight=w_jet)

    # event-level metric relevant for the kinfit: in events with exactly two
    # true b jets in the input, are the two highest-scoring jets those b jets?
    s_masked = np.where(m, s, -1.0)
    top2 = np.argsort(-s_masked, axis=1)[:, :2]
    two_b = y.sum(1) == 2
    hit = np.take_along_axis(y, top2, axis=1).sum(1) == 2
    top2_eff = float(np.average(hit[two_b], weights=w[two_b]))

    return {"loss": tot_loss / tot_w, "auc": float(auc), "top2_eff": top2_eff}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--cache", default=None, help="npz cache of the padded arrays (read if it exists)")
    p.add_argument("--outdir", default="runs/default")
    p.add_argument("--model", default="transformer", choices=list(MODELS))
    p.add_argument("--model-kwargs", default="{}", help='JSON, e.g. \'{"dim": 128, "layers": 4}\'')
    p.add_argument("--max-jets", type=int, default=8)
    p.add_argument("--split", type=int, default=0, choices=[0, 1])
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))
    os.makedirs(args.outdir, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────────
    cfg = InputConfig(max_jets=args.max_jets)
    arrays = load_arrays(list_files(args.input_dir, args.max_files), cfg, cache=args.cache)

    keep = (arrays["event"] % 2 == args.split) & (arrays["weight"] > 0) & (arrays["njets"] > 0)
    arrays = select(arrays, keep)
    arrays["weight"] = arrays["weight"] / arrays["weight"].mean()   # O(1) weights

    is_val = rng.random(len(arrays["weight"])) < args.val_frac
    train, val = select(arrays, ~is_val), select(arrays, is_val)

    scaler = Standardizer().fit(train)
    train, val = scaler.transform(train), scaler.transform(val)
    tr = EventBatcher(to_tensors(train), args.batch_size, shuffle=True, device=device)
    va = EventBatcher(to_tensors(val), 4 * args.batch_size, shuffle=False, device=device)

    n_b = train["label"][train["jet_mask"]].sum()
    n_jets = train["jet_mask"].sum()
    print(f"train: {len(train['weight']):,} events / {n_jets:,} jets ({n_b / n_jets:.1%} b)   "
          f"val: {len(val['weight']):,} events")

    # ── model ───────────────────────────────────────────────────────────────
    model_kwargs = json.loads(args.model_kwargs)
    dims = (len(cfg.jet_feature_names), len(cfg.lep_feature_names), len(cfg.met_feature_names))
    model = build_model(args.model, *dims, n_leptons=cfg.max_leptons, **model_kwargs).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def checkpoint():
        return {
            "model_name": args.model,
            "model_kwargs": {**model_kwargs, "n_leptons": cfg.max_leptons},
            "dims": dims,
            "input_config": cfg.to_dict(),
            "scaler": scaler.stats,
            "state_dict": model.state_dict(),
            "split": args.split,
        }

    # ── training loop ───────────────────────────────────────────────────────
    best, bad_epochs, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for batch in tr:
            loss = masked_bce(model(batch), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item()
        sched.step()

        metrics = evaluate(model, va)
        metrics.update(epoch=epoch, train_loss=run_loss / len(tr), time=time.time() - t0)
        history.append(metrics)
        print(f"epoch {epoch:3d}  train {metrics['train_loss']:.4f}  val {metrics['loss']:.4f}  "
              f"AUC {metrics['auc']:.4f}  top2 {metrics['top2_eff']:.3f}  ({metrics['time']:.0f}s)")

        if metrics["loss"] < best:
            best, bad_epochs = metrics["loss"], 0
            torch.save(checkpoint(), os.path.join(args.outdir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    with open(os.path.join(args.outdir, "history.json"), "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"best val loss {best:.4f} -> {args.outdir}/best.pt")


if __name__ == "__main__":
    main()
