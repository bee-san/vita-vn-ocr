# VOCR4: the console-sized recognizers retrained with the new real data

**Status: protocol frozen, runs not started.** Results, selections and the locked report are added here
once they exist. Nothing below is a measurement of a VOCR4 model yet.

VOCR4 trains VOCR3's three recognizers again (FL10, H15, W20: `docs/vocr3.md`) with the real data that
arrived after VOCR3, and with synth3 version 3 in place of version 2. It selects on development data only,
then scores the protected sets once through the plain-C W8I runtime, against VOCR1, VOCR2, VOCR_H15, F20
and VOCR3. The machine-readable protocol is `eval/vocr4_protocol.json`; this page restates it.

## What is new since VOCR3

| data | what | state at the freeze |
|---|---|---|
| realplus v3 interim (`rp3i`) | realplus v2 plus the lines realplus-v3 has decided since (Claude and Codex Luna agree) | built 21:55Z: 48,479 lines of 9,499 VNs (v2 + 5,310), 79,644 ink and 58,735 GRAD crops; pinned by sha256 in the protocol |
| native pool v1 (`nat`, PR #66) | reviewed native Vita/PSP lines from VNDB screenshots | 557 ink + 411 GRAD crops, 186 VNs |
| the owner's frames (`own`) | the owner's own console captures of training titles: planetarian (v34), Sharin no Kuni (v57), CHAOS;CHILD (v14018), labelled by native-frames (Claude plus engine agreement; private, never sent to Codex) | 577 frames, 946 lines read; review running, expected about 02:30Z |
| synth3 version 3 | the VN-style synthetic data of study 1b (#56), never used by VOCR3 | pinned 1,000,000 crops (`data/manifests/synth3_shards_v3.json`) |

synth4 (#69) measured no gain over synth3 version 3 (teacher_val +0.28 points, CI +0.08 to +0.48), so
there is no synth4 arm; synth3 version 3 is the synthetic arm.

## Arms

Each size has three arms, two seeds each (12 runs per size, 36 in all):

- **C, the control.** VOCR3's realplus fine-tune (`eval/vocr3_rp_run.sh`) on VOCR4's cleaned pools, from the
  pretrain of VOCR3's selection of that size (FL10 128,000 steps, H15 40,000, W20 32,000; reused, pinned
  in `eval/vocr4_inputs.sha256`). Fine-tune mix: nvsynth 0.25, synth2 0.125, synth3 v2 0.125,
  teacher_train 0.2, real_train 0.1, realplus v2 0.2.
- **A, the new real pools.** The same, with the real half re-split: teacher_train 0.175, real_train 0.1,
  realplus v3i 0.175, native pool v1 0.025, owner frames 0.025.
- **M, synth3 version 3.** A new pretrain from scratch with synth3 v3 in place of v2 (nvsynth 0.5,
  synth2 0.25, synth3 v3 0.25; VOCR3's steps and schedule), then A's fine-tune with synth3 v3 in place of v2.

Every fine-tune is VOCR3's: 15,000 steps, batch 96, lr 1.2e-3, EMA 0.999, hard labels (kd 0); the final EMA
checkpoint is the run. C and A start from one pretrain, so their seed varies the fine-tune only; M's seed
seeds its pretrain too. Runs: `eval/vocr4_run.sh SIZE ARM SEED` through `heavy.sh` (5 threads), at most 8
at once on the shared box.

## Pools and hygiene

`eval/vocr4_pools.py` builds every training pool from one exclusion list (2026-10-13 VNDB dump):
everything `realplus/shots.reserved_vns()` reserves for a protected or development reason (CLANNAD with its
family and relations, the owner's test series with their relations and release partners, eval's test games
and `is_test_vn`, the fresh native titles, the development VNs), every VN that shares a VNDB release or a
family relation with a CLANNAD, owner-test, test-set or fresh-native VN, the owner's 50 test VNs and every
VN of teacher_val and devnative. planetarian (v34) is the one reserved VN allowed, by the owner's 20:47Z rule.
VNs linked only to development or hash-held-out VNs stay, as in the pools every released model trained on.
The list has 10,487 VNs.

That rule is stricter than the pools VOCR3 used, which predate the owner's reservations and the
release-partner rule:

| pool | rows before | after | dropped |
|---|---:|---:|---|
| teacher_train | 12,100 | 11,153 | 695 linked to a test game by a shared release or family relation, 146 owner test set, 68 development VNs, 38 sharing a release with CLANNAD |
| real_train | 9,795 | 9,005 | 570 test-game links, 137 owner test set, 51 development, 32 CLANNAD release partners |
| realplus v2, ink / GRAD | 71,153 / 53,566 | 71,142 / 53,558 | owner test set |

The new pools (rp3i, nat, own) are cleaned the same way and also deduplicated by screenshot against
teacher_train, real_train and each other (native pool first, then the owner frames, then realplus).
`eval/vocr4_pools.py check` must pass on every pool before it is pinned: no excluded VN, no development
VN or screenshot, no test-set screenshot or capture session. The control's pools are pinned with the protocol
(`eval/vocr4_pools_c.sha256`); the new pools are pinned in `eval/vocr4_pools_a.sha256` before the first A or
M fine-tune starts, at 2026-10-14 06:00Z at the latest. If realplus v3i or the owner pool is not published by
then, A uses realplus v2 in its place, or gives the owner frames' weight to the native pool.

## The planetarian rule

The owner (2026-10-13 20:47Z) made planetarian training data and asked that its lines leave teacher_val,
devnative, vita-test and psp-test, with old and new scores side by side. There are none:
`eval/vocr4_planetarian_check_20261013.json` counts 0 rows of v34 or its VNDB relations (v27747, v5698),
by VN id and by any of their 84 releases, in teacher_val, devnative v3 (the originals and the copies VOCR3
selected on), vita-test and psp-test. The planetarian-free versions of those sets are the current ones byte for
byte, so every score in this document holds for both; the tables say so instead of repeating the column.

## Selection (development data only)

- Cohorts: VOCR3's copies of teacher_val (1,643 lines) and devnative v3 (986 crops: strict and tv_native,
  ink and GRAD), pinned by sha256 in the protocol.
- Score: `eval/synth3_ablation.py score` (C runtime W8I, both polarity views); a run's development score is
  the mean of its teacher_val CER and its devnative CER.
- Real decision, per size, C to A over the two seed pairs: A is adopted iff the stratified VN-cluster 95%
  interval of the development-score difference lies entirely below 0 (10,000 resamples, teacher_val's and
  devnative's VNs resampled independently), both pairs improve, and neither cohort's own interval lies
  entirely above 0 (`eval/vocr4_select.py decide`).
- Synth decision, per size: the arm the real decision keeps (A, or C) to M, by the same rule.
- Selection: the two runs of the arm the decisions end on; the lowest development score wins (ties: the lower
  seed); every blob must pass `vita/budget_sweep.py check` (`eval/vocr4_select.py select`). Every run is
  scored and listed.

## Locked report (once, after the three selections are committed)

`eval/vocr3_report.py` on vita-test and psp-test: fixed lines (console polarity primary, native too; per set,
style and VN, pooled; full text and hiragana) and whole dialogue (each region at base, +4 and +8 px, with the
arm's own C finder), paired bootstrap intervals over lines and VN clusters, sizes, arenas, A9 estimates and the
overlay host cross-check. Arms: VOCR1, VOCR2, VOCR_H15, F20, VOCR3 (three sizes) and VOCR4 (three sizes).
Pairs: each VOCR4 size against VOCR3's same size and against the released model it would replace, VOCR4-FL10
against VOCR_H15, VOCR4-W20 against F20 and VOCR4-H15. Gates: VOCR2's fixed-line gate and whole-dialogue guard
against VOCR1 (FL10) and VOCR_H15 (H15, W20).

CLANNAD: once owner-drive-ingest freezes its CLANNAD long-run references (Claude references read twice,
`eval/clannad_refs_protocol.json` on its branch), the three VOCR4 selections are read once through its
extra-arms path and scored once against those references, beside VOCR1, VOCR2, VOCR_H15, F20, VOCR3 and the
cached Google Lens readings. If that happens after the rest of the report, it is an addendum that cannot change
anything. Disclosure: arms A and M train on planetarian's PSP frames, a Key / Prototype port like CLANNAD's
Vita release. The fresh native textbox test needs private inputs that are not on this machine.

The three VOCR4 blobs go on the pre-release `v0.2.5-vocr4.1` whatever the result. A released default changes
only if a candidate passes every gate and the owner agrees.

## Disclosure carried from VOCR3

The reused pretrains and synth3's pinned shards predate the owner's reservations of 10:51Z and 13:10Z and
hold some material of those titles (`docs/owner-games.md`, `docs/vocr3.md`). Any test on the owner's
captures of Memories Off, Aiyoku no Eustia or 428 must carry that caveat for every VOCR4 model.
