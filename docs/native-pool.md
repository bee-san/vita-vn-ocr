# The native pool: more native Vita / PSP training lines

Native console frames are scarce in the real training data. realplus v2's 43k lines are all PC
screenshots scaled down, and teacher_val is two thirds scaled PC shots. The owner asked for every
native PS Vita and PSP frame VNDB has, labelled with realplus v3's pipeline (Claude + Codex gpt-6-luna
agreement). `realplus/native.py` does the harvest. The labelling is realplus v3's code, run on a
separate pool directory (`$HERD/shared/artifacts/native-frames/pool`, `REALPLUS` pointed there).

## VNDB harvest (`native.py enumerate`, dump 2026-10-13)

| | |
|---|---:|
| releases for PSP or PS Vita | 3,474 (PSP 1,695, Vita 1,751, both 28; 3,350 for these consoles only) |
| titles with such a release | 1,243: 157 PSP-only, 62 Vita-only, 19 PSP + Vita only, 1,005 multi-platform |
| screenshots on those releases | 4,089 from 498 VNs |
| of them native frames (psv 960x544, psp 480x272) | 3,959: 856 Vita, 3,103 PSP |
| **new** native frames (training side, in no pool or set) | **0** |
| training-side native frames already in realdata's / teacher's pools | 1,671 from 235 VNs (+6 only on synth2's scene list) |

Every screenshot's first exclusion reason is listed below, under main's rules after #63 (owner,
13:10Z). Two earlier runs used older rules:

- The 09:5xZ run used the 09:25Z rule. It differed in 10 Memories Off frames, which it selected, and in
  15 frames of STEINS;GATE's relations, which it excluded.
- The 10:51Z run differed in 街's 10 screenshots: 6 pooled, 3 linefind and 1 rating.

The counts:

| reason | screenshots |
|---|---:|
| training-side, already pooled (realdata / teacher) | 1,671 |
| VN of eval's test games (`test_games.json` "all") | 642 |
| development VN (teacher_val, general-VN development, devnative) | 492 |
| frame of an eval/linefind set | 304 |
| VN shares a release with a reserved VN | 278 |
| family of CLANNAD, a fresh-native title or a development VN | 196 |
| not a native size (960x540, 480x270, PSP games captured on a Vita at 960x544, ...) | 130 |
| rating (sexual >= 0.3 or violence >= 0.5) | 125 |
| frame of eval's candidate sets | 64 |
| release language other than ja / ja+en | 54 |
| validation-bucket VN | 35 |
| the owner's test series (Memories Off; `docs/owner-games.md`) | 67 |
| a relation of the owner's test titles (街, a 428 relation, #63) | 10 |
| `is_test_vn` | 10 |
| only on synth2's scene list | 6 |
| CLANNAD | 5 |

VNDB holds no native frame the training side has not already used. realplus v2 found the same on the
2026-10-12 dump. The native supply that is still reachable is the lines realdata and teacher left
unlabelled on their own native frames: their engines disagreed (teacher_distill: 1,426 PSP and 501 Vita
lines on these frames), or their rules never considered the lines.

## Frames (`native.py select`, `prep`)

Under main's rules the selection holds 1,677 frames (1,348 PSP, 329 Vita; 236 VNs). They are ordered
console exclusives first: PSP-only titles, then Vita-only, then PSP + Vita, then multi-platform.

Before reading:

- 7 frames were pixel copies of another frame, and 2 were linefind frames under another VNDB id
  (v3429_58100 and v3429_58101, `eval/tools/leakcheck.py`, correlation 1.000). All 9 were dropped.
- The leak checks were re-run at the end over all 1,693 frame files, the 1,684 read and the 9 dropped.
  They found no pair at 0.95 or more against vita-test and psp-test (157 screenshots), devnative's 346
  cohort frames or teacher_val's 646 frames. The only linefind pairs are the 2 dropped frames.

1,684 frames were read with devnative's `read.py`: MeikiOCR plus PP-OCRv6 small, four readings per
line, offline, in one heavy.sh slot. 1,669 were read at 10:0xZ and 15 more after the 10:51Z rule.

- 1,668 of them are in the selection now.
- 16 are of VNs reserved after they were read: 10 of v1964 (Memories Off, 10:51Z) and 6 of 街 (v1360,
  13:10Z). Their lines are in no batch after the rule and in no build.

## Lines

`pool.py plan` (realplus v3) finds 4,089 candidate lines on the 1,684 frames. 1,455 of them are auto:
all four readings agree. The other 2,634 go to review.

Lines teacher or realdata already labelled on the same frame are left out of review and of the build
(same row and overlapping ink; `native.py known`, `filter-plan`). Most auto lines are among them.

| review lines | lines |
|---|---:|
| in the plan | 2,634 (t1 1,574, t2 106, t3 954) |
| already labelled by teacher or realdata | -1,337 |
| of VNs reserved after the read (v1964, 街) | -46 |
| **new, reviewed** | **1,278** (t1 389, t2 80, t3 809) |

Pass 1 (`review/pass1/b0001`-`b0008`, 130 sheets) covered all of them. It held 1,300 lines before the
13:10Z reservation took 街's lines out. These are lines the old pools' engines found hard.

## Labelling (realplus v3's rule)

Every stage ran realplus v3's code (`feat/realplus-v3` at 01a7f84, run unchanged from a read-only worktree
with `REALPLUS=$NATIVE_POOL`, `REALPLUS_V3_PASS1=1`), under realplus v3's agreement rule
(`AGREEMENT.md`, copied verbatim before the first native batch):

- `pool.py plan` and `batches` (pass 1);
- `luna.py` (Codex gpt-6-luna, at most 2 calls at a time);
- `pool.py final`;
- `confirm-batches`, `edge-batches` and `edge-apply`.

`native.py` adds what native frames need on top:

- `edge-boxes`: realplus's edge boxes assume a PC shot read at Vita scale; these frames are read at their
  own console's size;
- `known` and `filter-plan`: lines realdata or teacher already labelled are left out;
- `filter-labels`: VNs reserved after the frames were read are dropped;
- `build`: native-target crops only;
- `check` and `check-score`: the precision sample.

Thirteen Claude helpers (review-nf-1 to 13, at most 3 at a time) decided the batches, one image per tool
call:

| stage | lines | result |
|---|---:|---|
| pass 1 (b0001-b0008, 130 sheets) | 1,300 | Claude: 527 picks, 59 typed, 16 unsure; skipped 584 ui, 80 cut, 34 bad |
| Luna, same sheets | 1,300 | of the 586 lines Claude read, Luna read the same key on **450 (76.8%, Wilson 73.2-80.0%)**; 68 disagreed (13 only at an edge 「」。), 68 skipped or unsure |
| confirm looks (blind, final crop) | 321 | single-family labels and lines Luna skipped: 314 confirmed, 7 dropped |
| edge checks (final crop) | 370 | 354 passed, 16 cut |

Under the rule, 437 lines are kept by Claude + Luna agreement and 51 by a confirm look after Luna skipped.
35 Luna disagreements and the confirm drops are out.

## Pool (`native.py build --tag v1`)

The build starts from the 1,924 kept labels: 1,446 auto, 188 with both engine families, 290 with one
family plus a confirm look. It leaves out reserved VNs and leak-dropped frames, and drops these lines:

| dropped | lines |
|---|---:|
| teacher or realdata already labelled | 1,333 |
| the precision check disagreed (below) | 4 |
| a character outside `data/vocab.txt` | 3 |
| cut at an edge (edge check) | 16 |
| no usable crop | 6 |
| over realplus's repeat cap | 5 |

| | |
|---|---:|
| **new native lines** | **557** from 186 VNs (6,406 characters, median 11 per line) |
| PSP / Vita | 435 / 122 |
| ink crops (eval's crop, widened to the labelled characters) | 557 |
| GRAD crops (the console finder's box, holding the line and nothing else) | 411 (334 PSP, 77 Vita) |
| how | 105 auto (all four readings), 182 both engine families, 270 one family plus a Claude confirm look |
| tier | auto 105, t1 184, t2 46, t3 222 |
| files | `native_train_ink_v1.npz` sha256 ca202a1c..., `native_train_grad_v1.npz` 2cf40190... |

Every line has an ink crop; 411 also have a GRAD crop. A GRAD crop is dropped, while its line is kept,
when the finder's box misses the line, mixes it with another, cuts its label, or needs a check nobody
made.

The pool is for the next training round, not VOCR3 round 1 (conductor, 09:50Z). The files are under
`$HERD/shared/artifacts/native-frames/pool/` (`native_train_{ink,grad}_<tag>.npz`, teacher_train's shard
format, console polarity). `realplus/manifests/native/` holds:

- the frames: URL, VNDB id, file and pixel sha256, kind, release, session, title group, drops;
- the lines: ids, crop boxes, provenance, no text;
- every stage's counts and every file's sha256.

### Precision

The check is independent and blind (`native.py check`, `check-score`). Before the final build, 160 lines
were drawn from the pre-build (561 lines; seed 20261013): 40 auto and 120 reviewed. A Claude helper
(review-nf-13) that had not labelled them typed the framed text of each crop. No candidate was shown.

| | agree | rate (Wilson 95%) |
|---|---:|---|
| **all** | **156 / 160** | **97.5% (93.8-99.0%)** |
| auto | 38 / 40 | 95.0% (83.5-98.6%) |
| reviewed (Claude + Luna or confirm) | 118 / 120 | 98.3% (94.1-99.5%) |
| ink crops / GRAD crops | 105 / 108, 51 / 52 | 97.2% / 98.1% |

The 4 disagreements:

- 1 GRAD crop cut its line (auto);
- 1 auto label with 一 where the line has ー;
- 1 small ゥ read as ウ;
- 1 プ read as ブ.

All 4 lines were dropped from v1 (`review/resolutions.jsonl`). The rate is measured on the pre-build;
v1 is the same pool without those 4.

## Owner rules and Codex

The owner's rules changed during the run, and each change was applied to the pool:

- **10:51Z** reserved the Memories Off series. The pool held 10 frames of Your Memories Off ~Girl's Style~
  (v1964). Their lines were dropped from the plan and the labels.
- **11:32:02Z, after that rule:** 3 public VNDB crops of v1964 were sent to gpt-6-luna (sheet
  pass1/b0008/S001: v1964_52004_1, v1964_52005_0, v1964_52006_1). Re-prepping the frame list had dropped
  those frames from the known-line bookkeeping, so their lines came back into the plan. audit-native-frames
  found this (O1), and the conductor was told.
  - The fix: known lines now come from every frame read, and `filter-plan` and `filter-labels` drop every
    VN `reserved_vns` reserves today.
  - The Luna loop was stopped after pass 1. Pass-1 Luna votes on v1964 lines from the other batches were
    sent 10:25-10:41Z, before the rule.
- **13:10Z** reserved 428 and 街 (v1360, a 428 relation). The pool held 6 frames of 街. Its 19 kept labels
  were dropped before the build; their Luna votes date from 10:25-10:41Z.
- The owner's own frames (`shared/private/owner-games/`) never enter this pool and never go to Codex.

## Owner frames (`realplus/owner_frames.py`)

The owner captures live screenshots into `shared/private/owner-games/<console>-<title>-v<id>[-TEST]/`.

- A `-TEST` folder belongs to an owner test title (`shots.OWNER_TEST_SERIES`: Memories Off, Aiyoku no
  Eustia, 428). Its frames go to a fresh native test set only.
- Any other folder must name a training-side VN: `reserved_vns` is checked again at ingest. A reserved title
  is refused, whatever the owner's list says.
- Frames from either kind of folder stay under `shared/private/` and never go to Codex. They are labelled
  by Claude only, with a blind second look in place of Luna.

`owner_frames.py inventory` records per file:

- the file sha256, devnative's decoded-pixel sha256 and the fresh-native harness's RGB sha256;
- the size and duplicates;
- the source resolution.

A 960x544 frame that is a PSP picture scaled 2x by Adrenaline is recorded as source 480x272, method
`nearest` or `smooth`, with the statistics that decided it (`upscale_2x`):

- nearest: the frame survives a 2x down / 2x up round trip almost unchanged;
- smooth: it holds almost no power above 0.3 cycles per pixel.

On held-out training-side frames this called 261 of 262 native Vita frames native, 200 of 200 nearest-2x
PSP frames nearest, and 190 of 200 bilinear-2x frames smooth (after a Q90 re-encode). `prep` writes
`devnative/read.py`'s input for one folder's unique frames.

No owner frame had arrived by 15:15Z, so none has been read or labelled yet.
