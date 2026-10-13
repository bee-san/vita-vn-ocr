"""Training-only attention guidance for the CTC recognizer (GTC, Hu et al. AAAI 2020; PaddleOCR's
GTC-NRTR multi-head in PP-OCRv4-v6). A small Transformer decoder reads the frame-row features of
train/model.py's TinyVNOCR (the output of `features`, before the bottleneck) and predicts the label
autoregressively with teacher forcing; its cross-entropy is added to the CTC loss, so its gradients
shape the shared encoder. It is never exported: checkpoints keep it apart from "model" (as
"guide"), and the deployed model, the blob and the C runtime are the plain TinyVNOCR.

Decoder classes: 0 = padding (ignored), 1..V = the CTC classes 1..V (data/vocab.txt order),
V + 1 = BOS, V + 2 = EOS.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid(n, d, device=None):
    pos = torch.arange(n, dtype=torch.float32, device=device)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32, device=device)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class NRTRGuide(nn.Module):
    """Transformer decoder over the 1 x T feature row; n_classes counts the CTC blank."""

    def __init__(self, in_ch, n_classes, d=256, layers=2, heads=4, ff=2, max_len=64, dropout=0.1):
        super().__init__()
        self.V = n_classes - 1
        self.bos, self.eos, self.max_len, self.d = self.V + 1, self.V + 2, max_len, d
        self.proj = nn.Linear(in_ch, d)
        self.embed = nn.Embedding(self.V + 3, d, padding_idx=0)
        layer = nn.TransformerDecoderLayer(d, heads, ff * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.V + 3)

    def targets(self, ids_list):
        """(decoder input, target) token tensors, N x L, from per-line CTC class id arrays."""
        seqs = [list(map(int, ids))[:self.max_len - 1] for ids in ids_list]
        L = max(len(s) for s in seqs) + 1
        inp = torch.zeros(len(seqs), L, dtype=torch.long)
        tgt = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, s in enumerate(seqs):
            inp[j, :len(s) + 1] = torch.tensor([self.bos] + s)
            tgt[j, :len(s) + 1] = torch.tensor(s + [self.eos])
        return inp, tgt

    def forward(self, feats, frames, inp):
        """feats: N x C x 1 x T; frames: valid frames per line (N); inp: N x L tokens. Returns
        N x L x (V + 3) logits."""
        mem = feats.squeeze(2).transpose(1, 2)                            # N x T x C
        T = mem.shape[1]
        mem = self.proj(mem) + sinusoid(T, self.d, mem.device)
        mem_pad = torch.arange(T, device=mem.device)[None, :] >= frames[:, None]
        L = inp.shape[1]
        x = self.embed(inp) * math.sqrt(self.d) + sinusoid(L, self.d, inp.device)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=inp.device), 1)   # True = masked
        y = self.dec(x, mem, tgt_mask=causal, tgt_key_padding_mask=inp == 0, memory_key_padding_mask=mem_pad)
        return self.out(self.norm(y))

    def loss(self, feats, frames, ids_list, smoothing=0.1):
        """Label-smoothed cross-entropy of the teacher-forced decoder, mean over target tokens."""
        inp, tgt = self.targets(ids_list)
        logits = self(feats, frames, inp)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
                               ignore_index=0, label_smoothing=smoothing)

    @torch.no_grad()
    def greedy(self, feats, frames):
        """Greedy autoregressive decode (for tests and diagnostics only): lists of class ids."""
        N = feats.shape[0]
        inp = torch.full((N, 1), self.bos, dtype=torch.long)
        done = torch.zeros(N, dtype=torch.bool)
        for _ in range(self.max_len):
            nxt = self(feats, frames, inp)[:, -1].argmax(-1)
            nxt[done] = 0
            inp = torch.cat([inp, nxt[:, None]], 1)
            done |= nxt == self.eos
            if done.all():
                break
        return [[t for t in row[1:].tolist() if 0 < t <= self.V] for row in inp]
