# Reservations: what is held out of training, and why

Owner, 2026-10-14 13:25Z (final; it supersedes every earlier reservation rule): the only test sets are
vita-test, psp-test and the CLANNAD family, with those exact VNs plus their shared releases. Everything else is
training data. Model selection may use a small development sample of about 100 VNs taken from the training pool.
Old reports may stop reproducing; the goal is the best OCR, not the best test set.

Before this change the repo held back 18,805 VNs (`reserved_vns.json` on the 2026-10-12 dump). 16,439 of them came
from two random md5 splits that no scorer reads outside the sets below. Now 184 VNs are held back: 71 test VNs and
the 113-VN development sample. The change frees 21,789 VNDB screenshots of 5,077 VNs, 1,075 labelled development
lines, 637 owner frames and 22,189 VNDB character names for training.

## The policy

| part | VNs | rule |
|---|---:|---|
| `test-set` | 28 | the games of vita-test (16) and psp-test (14); two games are in both sets |
| `test-shared-release` | 19 | on one VNDB release with a set game: compilations (r23905, 10 VNs, with v4147), bundles of a series (Utawarerumono, Detective Club), an original on one disc with its fan disc |
| `clannad`, `clannad-family`, `clannad-relation` | 21 | the CLANNAD family as the repo protects it: CLANNAD (v4) and every VN titled CLANNAD, the conductor's family list (`realplus/shots.CLANNAD_FAMILY`), CLANNAD's direct VNDB relations of every kind, and the FAMILY relations of the family list |
| `clannad-shared-release` | 3 | on one VNDB release with a CLANNAD-family VN: Kanon (v33), AIR (v36), Little Busters! (v5), through the Key boxes. planetarian (v34) is excepted by the owner's decision of 2026-10-13 |
| **test, total** | **71** | never train, tune, calibrate, select or label-tune on them, and never use them as synthetic scenes or names |
| `dev-sample` | 100 | the development sample (below) |
| `dev-sample-shared-release` | 13 | on a two-VN release with a sample VN: one game filed under two VN entries, or a game and its fan disc |
| **held out, total** | **184** | |

The 31 capture sessions of vita-test and psp-test (VNDB release ids, `test_games.json` "sessions") stay protected
too: no screenshot of those releases is used, and neither is any vita-test or psp-test screenshot by image id, URL,
file sha256 or decoded pixels.

Everything else trains:

- the old hash splits: `synth/common.is_test_vn` (md5 % 100 < 15, 9,816 VNs) and the val bucket (15 ≤ md5 % 100 <
  25, 6,623 VNs);
- the set games' other VNDB relations (sequels, fan discs, series), unless they share a release with a set game;
- teacher_val, real_val and devnative outside the development sample;
- the fresh native textbox titles and their relations;
- the owner's former test series (Memories Off, Aiyoku no Eustia, 428);
- every owner screenshot, apart from the CLANNAD frames. Four former `-TEST` folders become training data: Amagami
  (113 frames), Narcissu (200), Danganronpa V3 (84) and Iwaihime (240).

`eval/manifests/reserved.json` holds the list. `realplus/shots.py reserve` builds it from a VNDB dump; this file
used the 2026-10-13 dump. It contains `test` and `dev_sample` (VN → reason), `sessions`, the release ids behind every
`test-shared-release`, the counts, and the sample's per-VN rows. `test_games.json` "all" is now the vita-test /
psp-test part: the 28 games and their 19 release partners. The old 172-VN list stays there as `all_20261012`. That
list was the set games, every `is_test_vn` game with Vita or PSP screenshots, and the set games' family. Code reads
the reservation through `synth/common.py`:

```python
from common import is_reserved_vn, test_set_vns, dev_sample_vns, reserved_sessions
is_reserved_vn(vn)              # True: keep it out of training (a test VN or a development-sample VN)
is_reserved_vn(vn, dev=False)   # test VNs only (e.g. to sort owner frames into test and training)
```

`is_test_vn` stays as it was, for rebuilding the manifests of earlier reports (eval's candidate rule,
`train/sweep_eval.py`, `data/label_real.py`, `synth/ab_eval.py`). It reserves nothing.

### "Shared releases"

A VN is reserved when any VNDB release lists it together with a test VN. This is the repo's existing
`shared-release` rule for protected VNs (`realplus/shots.py` before this change). It is broader than eval's own
session rule. The set screenshots' 31 releases list no other VN, so that rule alone would add nothing. Under the
broader rule, 19 partners of 8 set games and 3 Key-box partners of CLANNAD are reserved. The narrower reading would
reserve only the shared releases' screenshots and would free those 22 VNs. That costs little either way, so the
conservative reading is used.

### The development sample

There are 100 VNs, chosen from VNs that already have labelled development lines. They are filled by tier, and inside
a tier by a seeded hash of the VN id (`stable_rank("dev-sample-20261014", vn)`). One VN's choice never depends on
another's.

| tier | VNs | what they bring |
|---|---:|---|
| 1 | 38 | devnative's reviewed native console lines (full Claude review plus a second pass) |
| 2 | 8 | native Vita/PSP teacher_val lines |
| 3 | 54 | scaled PC teacher_val lines |

A candidate is passed over in five cases:

- it is in the test reservation;
- it is an owner game (1 VN), because every owner screenshot trains;
- it shares a release with an owner game (4 VNs: Memories Off After Rain v1331, Corpse Party v10231 and v12794,
  Owaru Sekai to Birthday v9196);
- it is already in a training pool (none were);
- it shares a two-VN release with a VN of a training pool (21 VNs, such as v1762, whose twin entry v1763 is in
  teacher_train).

The sample holds 486 devnative lines (378 PSP, 108 Vita), 847 teacher_val lines (160 Vita, 366 PSP, 281 scaled PC)
and 659 real_val lines.

The sample is kept out of the training pools of any round that selects on it. Every gate below reads
`is_reserved_vn`, which counts it by default. A final refit that trains on the sample on purpose, after selection,
may run the leak check with `--dev-sample info`. That choice belongs to the training round's protocol.

## What the hash splits were used for (audit)

Line numbers are from main at f022ab8, before this change. The full tables are in the PR evidence. Three read-only
audits covered eval/ teacher/ realdata/ data/ train/, synth/ nvsynth/, and realplus/ devnative/ plus the owner
tools.

**Scored or reported sets built on the buckets:**

| set | bucket | reported where | now |
|---|---|---|---|
| vita-test, psp-test | `eval/tools/collect.py:45` drew candidates from `is_test_vn` games: 26 of the 28 set VNs are in that bucket, and the other two are hand-added (`collect.py:31-35`) | README, `docs/results.md`, `docs/train2.md:217-223`, every VOCR report | test sets, unchanged |
| teacher_val | the val bucket (`teacher/build.py:63-64`), 181 VNs | the selection metric of train2, VOCR3 and VOCR4 (`docs/train2.md:217-235`, `docs/vocr3.md:121,147-175`, `docs/vocr4.md:160-168`, `docs/synth3*.md`, `docs/synth4.md:14`) | 101 VNs in the sample (957 crops), 80 VNs train (694 crops) |
| devnative | its 47 VNs are all val-bucket teacher_val VNs (`devnative/select.py:73,89-94`; `docs/devnative.md:99-101`) | `docs/vocr3.md`, `docs/vocr4.md`, `docs/synth3-native.md:200` | 38 VNs in the sample; 65 lines train |
| synth2's provisional A/B | `is_test_vn` lines of `data/real` (`synth/ab_eval.py:58-62`, `data/label_real.py:233`) | `docs/synth2.md:269-276`, labelled provisional | history |
| teacher's 2-fold acceptance rule | `teacher/measure.py:231-232`, by game over the set games | `docs/teacher.md:76` | history |

**Never scored:**

- real_val (`realdata/build.py:62-63`; "not used": `docs/train2.md:96-97`, `docs/teacher.md:151`);
- the arch sweep's `real-pseudo` set (`train/sweep_eval.py:229-241`; no doc reports it);
- teacher_distill_val;
- the other 9,800 `is_test_vn` VNs and 6,400 val-bucket VNs, which were only ever held out.

linefind's holdout uses its own hash (`eval/linefind/build_gt.py:231-232`) and is reported for the line finder only
(`docs/linefind.md`).

**Training gates, switched to `is_reserved_vn` / `reserved.json` in this change:**

- realplus: `realplus/shots.py` `reserved_vns` (`:317-420`), which drives the VNDB pool selection (`:486`), the
  build-time exclusions (`realplus/pool.py:844-849`), the native pool (`realplus/native.py:72-78`, 129-134; its
  forbidden screenshot sets are now only `eval-test`) and owner-frame sorting (`realplus/owner_frames.py:62-78`;
  the folder's `-TEST` suffix no longer decides);
- owner drop roles: `synth/owner_drop.py` `decide` (`:160-196`), where only vita-test / psp-test membership or a
  test reservation makes a game test-only;
- synthetic scenes, names and statistics: `synth/render.py:171-177`, `calibrate.py:47-53`, `fetch_synth2.py:123-126`
  (`--refresh` only; the pinned pick is unchanged), `fresh_dataset.py:151`, `provenance_producer.py:297`,
  `synth3.py:463-489`, `synth3_text.py:376-392`, `synth3_mine.py:92`, `synth3_real_labels.py:43`,
  `synth3_stats.py:141-173`, `synth4_gap.py:98-113` and, through them, `native_calib.py` and `synth4.py`. The
  default exclusion manifests shrink to the CLANNAD one; fresh native and the owner series are `RETIRED_MANIFESTS`.
  `synth/bg_exclude.py` rewrote `data/manifests/synth2_bg_exclude.txt`: 71 test VNs, 121 sample VNs (with partners)
  and the 106 test-session screenshots. It had 172 old-list VNs plus 28 research screenshots of synth2's
  provisional A/B;
- other gates: `data/fresh_backgrounds.py:76`, `train/vocab.py:91`, `teacher/build.py:90-93` (and `teacher/label.py`
  through it), `realdata/vndb.py:90-99`, `realdata/build.py:167`, `devnative/select.py:333` (test VNs only),
  `eval/tools/leakcheck.py:80`;
- the list generator `eval/tools/manifest.py:145,166`, which now writes "all" as the set games plus their release
  partners.

**Left as history:** `eval/tools/collect.py`, `eval/linefind/build_gt.py`, `train/sweep_eval.py`,
`data/label_real.py`, `synth/ab_eval.py`, `teacher/measure.py`, the `split_of` of teacher/ and realdata/ (they define
the pinned teacher_val and real_val), and `eval/vocr4_pools.py` (VOCR4's frozen pool rule; its `STRICT` set still
names `test-vn`, so a VOCR5 pool must not reuse it unchanged).

## What the change frees

`realplus/reclaim.py` compares the new rules with what the old ones dropped, using the files those rules wrote.
`realplus/manifests/reclaim_20261014.json` holds the counts; `_splits.json` holds the labelled-set cuts.

| source | freed | notes |
|---|---:|---|
| realplus (VNDB, 2026-10-13 dump) | 20,867 screenshots of 4,883 VNs | 20,572 scaled PC (4,847 VNs) and 295 PSP (41 VNs). The old rules had dropped them as `test-vn` 9,468, `val-bucket` 6,126, `shared-release` 4,228, `test-set` 752, `family` 242, fresh native 36; 15 are new in the later dump. All pass the rating, language and kind filters |
| native pool (PSP/Vita frames) | 1,217 frames of 245 VNs | 1,026 PSP, 191 Vita. 886 are already on disk and read (realdata/teacher, lines unlabelled), 295 are new (the same 295 PSP shots as above), 36 are in other pools. They had been dropped as `test-set` 295, linefind 299, `shared-release` 245, `family` 142, owner series 74, `development` 53, `val-bucket` 24, eval candidates 15 |
| both lists, distinct | 21,789 screenshots of 5,077 VNs | |
| synthetic names | 22,189 VNDB characters | synth3's name gate dropped 24,331 of 127,809; it now drops 2,361 |
| synthetic scenes on disk | 388 screenshots | 28 Vita, 80 PSP, 280 scaled PC of synth3's current candidates. Every freed VNDB screenshot can become a scene once fetched and text-detected |
| labelled lines (teacher_val, real_val, devnative) | 1,075 lines of 87 VNs | teacher_val 551, real_val 459, devnative 65 |
| owner frames | 637 | Amagami, Narcissu, Danganronpa V3, Iwaihime; CLANNAD's 2,206 stay test |

The labelled sets are cut by VN under `$HERD/shared/artifacts/reserve-policy/reclaim/pools/`, in teacher_val's npz
format (`reclaim.py --split`). For each set there is a `*_train.npz` (training data now) and a `*_dev.npz` (the
sample).

| set | train crops (VNs) | sample crops (VNs) |
|---|---:|---:|
| teacher_val | 694 (80) | 957 (101) |
| teacher_distill_val (unlabelled) | 511 (76) | 750 (96) |
| real_val | 681 (86) | 814 (98) |
| devnative v3, tv_native ink / GRAD | 60 / 50 (6) | 453 / 373 (38) |
| devnative v3, strict ink / GRAD | 5 / 3 | 33 / 23 |

## Leak check

`eval/tools/reserve_check.py NAME=PATH ...` checks npz pools, line, shot and label manifests, and screenshot
directories. It extends the checks of realplus-v3's `realplus/hygiene.py` (branch feat/realplus-v3) to this policy:

- it blocks a row whose VN is a test VN, both from the committed list and recomputed on a given dump with `--dump`
  (titles, relations and release partners of a newer dump count);
- it blocks a row whose release is a test session or lists a test VN;
- it blocks a vita-test or psp-test screenshot by id, URL or sha256, and by decoded pixels with `--pixels`. The
  pixel check compares against the 2,363 protected images on disk: the set screenshots and the owner's CLANNAD frames;
- it blocks a development-sample VN (`--dev-sample info` reports instead);
- it reports the old hash splits, the set games' family and the former reservations as information.

Run on the 2026-10-13 dump with `--pixels`, the check gave **RESERVE_OK** on 25 training pools and lists:

- realplus v2 (ink, GRAD, lines);
- realplus v3i3 (ink, GRAD, lines);
- native pool v1 (ink, GRAD, lines);
- owner pool v2 (ink, GRAD) and the 1,215 owner training frames;
- teacher_train, teacher_distill_train, real_train and real_unlabelled, cleaned as below;
- the train parts of teacher_val, teacher_distill_val, real_val and devnative;
- both reclaim lists.

As negative controls, vita-test, psp-test, the sample parts and the original teacher_train fail.

The check also found an older exposure. It comes from the old rules, not from this change. The train2-era files hold
rows of 8 VNs that the test reservation now covers:

- CLANNAD (v4) and CLANNAD Side Stories (v4060);
- Tomoyo After (v12);
- Kanon (v33), AIR (v36) and Little Busters! (v5);
- two release partners of set games: v11706, on a bundle with psp-test's v11708, and v1267, on the r23905
  compilation with psp-test's v4147.

| file | rows of those 8 VNs |
|---|---:|
| teacher_train | 86 |
| real_train | 78 |
| real_unlabelled | 61 |
| teacher_distill_train | 50 |

linefind's frozen `psp_dev` holds v1267. VOCR4's pool builder had already left all 8 out (`docs/vocr4.md:103-119`;
its `exclusions.json` lists them as `clannad`, `clannad-family`, `shared-release:clannad` and
`shared-release:testset`).
Recipes that read the files directly saw them; that was not re-checked per recipe here. The cleaned copies are
`pools/{teacher_train,teacher_distill_train,real_train,real_unlabelled}_train.npz` (12,029, 8,994, 9,733 and
11,113 rows). A new round should use those copies or run `reserve_check.py` on its own pools.

## Past reports

Every number already published stands as measured, on pinned files:

- vita-test, psp-test and the CLANNAD long run have not changed, so the VOCR1-VOCR4 test results are unaffected;
- teacher_val, devnative and real_val keep their pinned npz files, but future selection uses the development sample,
  so old development scores and new ones are not comparable;
- the fresh native textbox and owner-series reports, and any report that relied on the hash splits being held out,
  no longer describe held-out data.

Old reports not reproducing under the new rules is accepted (owner, 13:25Z).

## Reproduce

```sh
D=$HERD/shared/artifacts/native-frames/vndb/dump/db        # VNDB dump 2026-10-13 08:00:26+00
.venv-ci/bin/python realplus/shots.py reserve --dump $D --test-games    # eval/manifests/reserved.json, test_games.json
.venv-ci/bin/python synth/bg_exclude.py                                 # needs data/vndb (and data/vndb_psp) locally
$HERD/heavy.sh .venv-ci/bin/python realplus/reclaim.py --dump $D --manifest realplus/manifests/reclaim_20261014.json
$HERD/heavy.sh .venv-ci/bin/python realplus/reclaim.py --split teacher_val=$A/teacher/teacher_val.npz ...
$HERD/heavy.sh .venv-ci/bin/python eval/tools/reserve_check.py --dump $D --pixels NAME=POOL ...
```

A rebuild on a newer dump can only grow the reservation, because `reserved_vns()` unions the committed file. The
sample changes only when `reserve` is run again. Run it again when an owner drop names a sample VN; `owner_vns()`
then leaves that VN out of the sample.
