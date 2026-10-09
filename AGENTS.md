# AGENTS.md

## Goal

The smallest Japanese OCR that reads visual-novel dialogue reliably on PS Vita (PCH-1000: Cortex-A9 at 444 MHz, one spare core, no fp16) and on PSP (Allegrex MIPS at 333 MHz, 32/64 MB RAM). It runs on the console inside vita-jp-overlay.

Memory comes first, then accuracy on real VN screenshots, then speed.

## Working agreement

- Every change goes through a pull request from its own branch. Never push to `main`.
- One coherent outcome per PR, in small commits.
- Each PR body states the problem, the change, how it was measured (command, data, environment, repeated samples where noise matters), and the results.
- Never commit copyrighted or bulky data: screenshots, crops, rendered shards, corpora, fonts or checkpoints. Commit code, manifests (URLs plus hashes), small derived tables and results. Weights go on GitHub releases.
- Report real-screenshot CER from the shared benchmark (`eval/`, once it exists), never from synthetic validation alone. Keep a hand-checked test set separate from anything used for training or calibration.
- Don't claim numbers you didn't measure. Mark host estimates of console speed or memory as estimates.
- Pin dependency versions. Prefer established libraries, and keep the console runtime dependency-free C.
