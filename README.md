# kinfit_nn_training

Neural networks that classify each jet in a dileptonic tt̄ event as **b jet or not**, using only
kinematics: the jets, the two leptons, and the missing transverse momentum (MET) of the event.

The task is to explore different network architectures for this. The code gives you a working
data pipeline, training loop, and three example models. You will mostly work in `models.py`.

## 1. Setup

```bash
./setup_env.sh            # creates the conda env in /data/dust/user/$USER/conda/envs/kinfit-nn
./setup_env.sh --gpu      # use this instead if you will train on a GPU
conda activate /data/dust/user/$USER/conda/envs/kinfit-nn
```

Without conda, see the Miniforge instructions at the top of `setup_env.sh`. The environment goes
on dust because pytorch is too large for the AFS home quota.

## 2. First run

```bash
python train.py --model mlp --max-files 40 --epochs 5 --cache cache/40files.npz --outdir runs/mlp_test
```

The first run reads the ROOT files (about a minute). `--cache` stores the result, so later runs
start in seconds. If you change the input features, delete the cache; the code refuses to use a
stale one. The device is detected automatically: a GPU if there is one, otherwise the CPU.

Each epoch prints the training and validation loss and two metrics:

- **AUC:** jet-level ROC AUC for b vs non-b. 0.5 is random, 1 is perfect.
- **top2:** in events with exactly two true b jets, the fraction where the two highest-scoring
  jets are exactly those two.

The best model is saved as `runs/<name>/best.pt`, and the loss and metrics per epoch go to
`history.json`.

## 3. The data

Input: `TTto2L2Nu` simulation, one ROOT file entry per event
(`/data/dust/user/pgadow/btv/phys_btag/sfb-ttkinfit/arrays_transformer/...`), with
variable-length lists of jets and leptons.

| object | features | notes |
|---|---|---|
| jets | pT, η, φ, mass | only jets passing the standard selection (`jets_isSel`), 2 to 7 per event |
| leptons | pT, η, φ, mass | exactly 2 (one electron, one muon) |
| MET | pT, φ | one per event |

The target is 1 if the jet's simulated flavour is b (`|jets_flavour| == 5`), otherwise 0.

## 4. Event-level inputs, jet-level labels

The network sees a whole event but gives an answer for each jet. Events have different numbers of
jets, so each event is padded to a fixed number of jet slots `J` (default 8). A mask records which
slots hold real jets. One batch of `B` events is a dict:

```
batch["jets"]      [B, J, 4]   pT, eta, phi, mass per jet slot (0 for padding)
batch["jet_mask"]  [B, J]      True = real jet
batch["leptons"]   [B, 2, 4]
batch["lep_mask"]  [B, 2]
batch["met"]       [B, 2]      pT, phi
batch["label"]     [B, J]      1 = b jet
batch["weight"]    [B]         event weight
```

All features are standardised to mean 0 and width 1 before training.

Every model returns `logits [B, J]`, one number per jet slot. The loss is computed per jet and
multiplied by the mask, so padded slots never count.

## 5. The example models (`models.py`)

| name | idea | jets see each other? |
|---|---|---|
| `mlp` | each jet + both leptons + MET → MLP | no |
| `deepsets` | embed every object, average them into an event summary, classify each jet from (own embedding, summary) | yes, through the summary |
| `transformer` | every object is a token; self-attention between all tokens; classify the jet tokens | yes, pairwise |

Change hyperparameters from the command line:

```bash
python train.py --model transformer --model-kwargs '{"dim": 128, "layers": 4}' --outdir runs/tf_big
```

## 6. Adding your own model

1. In `models.py`, write a subclass of `JetClassifier`:

   ```python
   class MyModel(JetClassifier):
       def __init__(self, n_jet_feat, n_lep_feat, n_met_feat, n_leptons=2, dim=64):
           super().__init__(n_jet_feat, n_lep_feat, n_met_feat, n_leptons)
           ...  # define layers

       def forward(self, batch):
           ...  # use batch["jets"], batch["jet_mask"], ...
           return logits  # shape [B, J]
   ```

   If jets in your model see each other, use `jet_mask` so padded slots don't influence real jets.
2. Add it to the `MODELS` dict at the bottom of `models.py`.
3. Run `python train.py --model mymodel ...`.

## 7. Applying a trained model

```bash
python predict.py runs/mlp_test/best.pt --max-files 2 --out scores.root
```

This writes a ROOT tree with one score per jet (`jets_nn_score`) next to `event_number`. Its jets
are in the same order as `jets_pt` in the input; jets outside the selection get NaN.

**Train/test split:** `--split 0` trains on events with even `event_number`, `--split 1` on odd
ones. A model must only be evaluated on the other half. 10% of the training half is used for
validation during training.

## Files

| file | content |
|---|---|
| `data.py` | reading ROOT files, padding, feature list (`InputConfig`), standardisation, batching |
| `models.py` | the models |
| `train.py` | loss, training loop, metrics, checkpoint |
| `predict.py` | apply a model, write per-jet scores |
| `environment.yml`, `setup_env.sh` | conda environment |
