# AGENTS.md

## Training data policy: read this first (owner directive, 2026-10-25)

**This overrides every other rule, doc, manifest, script or old result in this repo. If anything disagrees with it, this section wins. Fix or delete the conflicting rule in your PR instead of following it.**

The owner wants **the best OCR possible**. The real screenshots we collect exist to **train** the model. They are not there to fill evaluation sets.

- **Train on the data.** This includes every real screenshot set and the PSP/Vita screenshot farm, which has exact game-memory hook text, renderer line boxes and per-line labels. Refusing to train on gathered data, or moving most of it into test, eval or "reserved" sets, is wrong. VOCR5 did exactly this and put most of the real pages into evaluation instead of training. **Never do that again.** The owner has had to say this many times.
- **CLANNAD is the test set.** It is the held-out game the owner cares about. Keep CLANNAD's own releases out of training and calibration, and report real-screenshot CER on it.
- **Keep any other held-out data small.** If you need an extra check (for example on PSP), hold out at most a couple of **whole games**, never random lines from games you train on. **All held-out data together, CLANNAD included, stays at or under 10% of the real data.** 20% is the owner's absolute ceiling and is never a target.
- **The old reservations are revoked as reasons to withhold training data.** That covers `eval/manifests/test_games.json` (`all`, about 172 VNs), `is_test_vn()`, VNDB family/relation reservations, "test session" rules and the background-exclusion lists built from them. Those games, including the old vita-test and psp-test games, are **training data** now. You may keep a small frozen benchmark for comparison with old releases, but it must stay within the 10% budget and must never block training on the rest.
- Rendering synthetic backgrounds or text from those games is allowed. CLANNAD is the only exception.
- Treat data that gets excluded or quarantined for **label quality** (wrong text, misaligned boxes) as a cleaning step, and report it separately. It is not a test split.

## Goal

The smallest Japanese OCR that reads visual-novel dialogue reliably on PS Vita (PCH-1000: Cortex-A9 at 444 MHz, one spare core, no fp16) and on PSP (Allegrex MIPS at 333 MHz, 32/64 MB RAM). It runs on the console inside vita-jp-overlay.

Memory comes first, then accuracy on real VN screenshots, then speed.

## Working agreement

- Every change goes through a pull request from its own branch. Never push to `main`.
- One coherent outcome per PR, in small commits.
- Each PR body states the problem, the change, how it was measured (command, data, environment, repeated samples where noise matters), and the results.
- Never commit copyrighted or bulky data: screenshots, crops, rendered shards, corpora, fonts or checkpoints. Commit code, manifests (URLs plus hashes), small derived tables and results. Weights go on GitHub releases.
- Report real-screenshot CER from the shared benchmark (`eval/`), never from synthetic validation alone. Keep the hand-checked test set (CLANNAD; see the training data policy above) separate from anything used for training or calibration.
- Don't claim numbers you didn't measure. Mark host estimates of console speed or memory as estimates.
- Pin dependency versions. Prefer established libraries, and keep the console runtime dependency-free C.
