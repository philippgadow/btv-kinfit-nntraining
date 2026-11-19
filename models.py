"""
Model zoo: every model takes one batch of events and returns one logit per jet.

Input (``batch`` dict from ``data.EventBatcher``; B = events, J = jet slots,
L = 2 leptons, F_* = number of features per object):

    batch["jets"]      [B, J, F_jet]   zero-padded
    batch["jet_mask"]  [B, J]          True for real jets, False for padding
    batch["leptons"]   [B, L, F_lep]
    batch["lep_mask"]  [B, L]
    batch["met"]       [B, F_met]

Output: logits [B, J].  Padded jet slots are ignored by the loss, so their
value does not matter.  But a model in which jets see each other (DeepSets,
Transformer) must use the masks so padded slots don't leak into real jets.

To add a model: subclass ``JetClassifier``, implement ``forward``, and add it
to ``MODELS`` at the bottom.  Hyperparameters are keyword arguments of
``__init__`` and can be set from the command line with ``--model-kwargs``.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(n_in: int, hidden: list[int], n_out: int, dropout=0.0) -> nn.Sequential:
    """Fully connected network: n_in -> hidden[0] -> ... -> n_out.
    Acts on the last axis, so it works on [B, F] and on [B, J, F] alike."""
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class JetClassifier(nn.Module):
    """Base class. Stores the input sizes, which every model needs."""

    def __init__(self, n_jet_feat: int, n_lep_feat: int, n_met_feat: int, n_leptons: int = 2):
        super().__init__()
        self.n_jet_feat = n_jet_feat
        self.n_lep_feat = n_lep_feat
        self.n_met_feat = n_met_feat
        self.n_leptons = n_leptons

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # -> [B, J]
        raise NotImplementedError


class PerJetMLP(JetClassifier):
    """Baseline without jet-jet interaction: each jet is classified from its own
    features plus the leptons and MET.  The other jets are not seen."""

    def __init__(self, n_jet_feat, n_lep_feat, n_met_feat, n_leptons=2,
                 hidden=(128, 128, 64), dropout=0.1):
        super().__init__(n_jet_feat, n_lep_feat, n_met_feat, n_leptons)
        n_event = n_leptons * n_lep_feat + n_met_feat
        self.net = mlp(n_jet_feat + n_event, list(hidden), 1, dropout)

    def forward(self, batch):
        jets = batch["jets"]                                        # [B, J, F_jet]
        n_slots = jets.shape[1]
        # event features: both leptons and MET in one vector
        event = torch.cat([batch["leptons"].flatten(1), batch["met"]], dim=-1)  # [B, F_event]
        # give every jet slot its own copy of the event features
        event = event.unsqueeze(1).expand(-1, n_slots, -1)          # [B, J, F_event]
        x = torch.cat([jets, event], dim=-1)                        # [B, J, F_jet + F_event]
        return self.net(x).squeeze(-1)                              # [B, J]


class DeepSets(JetClassifier):
    """Each object is embedded independently, the embeddings of all real objects
    are averaged into one event summary, and each jet is classified from
    [its own embedding, event summary].  The average does not depend on the
    object order, so the model is permutation invariant."""

    def __init__(self, n_jet_feat, n_lep_feat, n_met_feat, n_leptons=2,
                 dim=64, hidden=(128, 128), dropout=0.1):
        super().__init__(n_jet_feat, n_lep_feat, n_met_feat, n_leptons)
        self.embed_jet = mlp(n_jet_feat, [dim], dim)
        self.embed_lep = mlp(n_lep_feat, [dim], dim)
        self.embed_met = mlp(n_met_feat, [dim], dim)
        self.head = mlp(2 * dim, list(hidden), 1, dropout)

    def forward(self, batch):
        jet_mask = batch["jet_mask"].unsqueeze(-1).float()          # [B, J, 1]
        lep_mask = batch["lep_mask"].unsqueeze(-1).float()          # [B, L, 1]

        h_jet = self.embed_jet(batch["jets"]) * jet_mask            # [B, J, D], padding -> 0
        h_lep = self.embed_lep(batch["leptons"]) * lep_mask         # [B, L, D]
        h_met = self.embed_met(batch["met"])                        # [B, D]

        # mean over all real objects (jets + leptons + MET)
        n_objects = jet_mask.sum(1) + lep_mask.sum(1) + 1           # [B, 1]
        summary = (h_jet.sum(1) + h_lep.sum(1) + h_met) / n_objects  # [B, D]

        summary = summary.unsqueeze(1).expand_as(h_jet)             # [B, J, D]
        return self.head(torch.cat([h_jet, summary], dim=-1)).squeeze(-1)  # [B, J]


class Transformer(JetClassifier):
    """Every object (each jet, each lepton, MET) becomes one token, a vector of
    size ``dim``.  Self-attention lets every token look at all other tokens.
    The jet tokens are then classified individually."""

    def __init__(self, n_jet_feat, n_lep_feat, n_met_feat, n_leptons=2,
                 dim=64, heads=4, layers=3, ff=128, dropout=0.1):
        super().__init__(n_jet_feat, n_lep_feat, n_met_feat, n_leptons)
        # one linear map per object type; this also tells the network the type
        self.embed_jet = nn.Linear(n_jet_feat, dim)
        self.embed_lep = nn.Linear(n_lep_feat, dim)
        self.embed_met = nn.Linear(n_met_feat, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=ff, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = mlp(dim, [dim], 1)

    def forward(self, batch):
        n_slots = batch["jets"].shape[1]

        # tokens: J jet slots, then L leptons, then 1 MET -> sequence of length J+L+1
        tokens = torch.cat([
            self.embed_jet(batch["jets"]),                          # [B, J, D]
            self.embed_lep(batch["leptons"]),                       # [B, L, D]
            self.embed_met(batch["met"]).unsqueeze(1),              # [B, 1, D]
        ], dim=1)                                                   # [B, J+L+1, D]

        # which tokens are real (MET always is); attention ignores padded tokens
        is_real = torch.cat([
            batch["jet_mask"],
            batch["lep_mask"],
            torch.ones_like(batch["jet_mask"][:, :1]),
        ], dim=1)                                                   # [B, J+L+1]

        h = self.encoder(tokens, src_key_padding_mask=~is_real)     # [B, J+L+1, D]
        h_jets = h[:, :n_slots]                                     # keep only the jet tokens
        return self.head(h_jets).squeeze(-1)                        # [B, J]


MODELS = {
    "mlp": PerJetMLP,
    "deepsets": DeepSets,
    "transformer": Transformer,
}


def build_model(name: str, n_jet_feat: int, n_lep_feat: int, n_met_feat: int,
                **kwargs) -> JetClassifier:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', choose from {list(MODELS)}")
    return MODELS[name](n_jet_feat, n_lep_feat, n_met_feat, **kwargs)
