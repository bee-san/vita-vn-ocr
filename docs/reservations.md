# Reservations: what is held out of training, and why

Owner, 2026-10-14 13:25Z (final; it supersedes every earlier reservation rule): the only test sets are
vita-test, psp-test and the CLANNAD family, with those exact VNs plus their shared releases. Everything else is
training data. Model selection may use a small development sample of about 100 VNs taken from the training pool.
Old reports may stop reproducing; the goal is the best OCR, not the best test set.

Before this change the repo held back 18,805 VNs (`reserved_vns.json` on the 2026-10-12 dump). 16,439 of them came
from two random md5 splits that no scorer reads outside the sets below. Now 187 VNs are held back: 71 test VNs and
the 116-VN development sample, re-carved on 2026-10-14 after the owner's second Drive drop (below; the first carve held
113). The change frees 21,809 VNDB screenshots of 5,079 VNs, 1,184 labelled development lines, 637 owner frames and
22,163 VNDB character names for training.

## The policy

| part | VNs | rule |
|---|---:|---|
| `test-set` | 28 | the games of vita-test (16) and psp-test (14); two games are in both sets |
| `test-shared-release` | 19 | on one VNDB release with a set game: compilations (r23905, 10 VNs, with v4147), bundles of a series (Utawarerumono, Detective Club), an original on one disc with its fan disc |
| `clannad`, `clannad-family`, `clannad-relation` | 21 | the CLANNAD family as the repo protects it: CLANNAD (v4) and every VN titled CLANNAD, the conductor's family list (`realplus/shots.CLANNAD_FAMILY`), CLANNAD's direct VNDB relations of every kind, and the FAMILY relations of the family list |
| `clannad-shared-release` | 3 | on one VNDB release with a CLANNAD-family VN: Kanon (v33), AIR (v36), Little Busters! (v5), through the Key boxes. planetarian (v34) is excepted by the owner's decision of 2026-10-13 |
| **test, total** | **71** | never train, tune, calibrate, select or label-tune on them, and never use them as synthetic scenes or names |
| `dev-sample` | 100 | the development sample (below) |
| `dev-sample-shared-release` | 16 | on a two-VN release with a sample VN: one game filed under two VN entries, or a game and its fan disc. Counting every release partner would hold out 170 more VNs, mostly compilations of unrelated games (176 for the first carve); counting same-series VNs on compilations would have held out 30 more (first carve) |
| **held out, total** | **187** | |

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
- every owner screenshot except those of a test game. CLANNAD's 2,206 frames stay test, and so do the second drop's
  408 frames of four test games (Time Travelers v10856, Utawarerumono v3, Collar x Malice v18160, Sweet Clown v16460).
  Four former `-TEST` folders of the first drop become training data: Amagami (113 frames), Narcissu (200),
  Danganronpa V3 (84) and Iwaihime (240).

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
| 1 | 34 | devnative's reviewed native console lines (full Claude review plus a second pass) |
| 2 | 7 | native Vita/PSP teacher_val lines |
| 3 | 59 | scaled PC teacher_val lines |

A candidate is passed over in five cases:

- it is in the test reservation;
- it is an owner game (8 VNs: v3278, v3333, v7833, v10271, v11677, v15773, v16386, v17863), because every owner
  screenshot trains;
- it shares a release with an owner game (7 VNs: Memories Off After Rain v1331, Corpse Party v10231 and v12794,
  Owaru Sekai to Birthday v9196, Natsuzora Kanata v701, World Election v18141, Ripkiss v18458);
- it is already in a training pool (none were);
- it shares a two-VN release with a VN of a training pool (21 VNs, such as v1762, whose twin entry v1763 is in
  teacher_train).

The 100 VNs hold 424 devnative lines (344 PSP, 80 Vita), 824 teacher_val lines (138 Vita, 336 PSP, 350 scaled PC)
and 625 real_val lines; with their release partners, 830 teacher_val and 629 real_val lines. (The first carve held 486
devnative lines: four of its devnative games are owner games now, and their replacements are scaled-PC teacher_val
VNs.)

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
| teacher_val | the val bucket (`teacher/build.py:63-64`), 181 VNs | the selection metric of train2, VOCR3 and VOCR4 (`docs/train2.md:217-235`, `docs/vocr3.md:121,147-175`, `docs/vocr4.md:160-168`, `docs/synth3*.md`, `docs/synth4.md:14`) | 101 VNs in the sample (959 crops), 80 VNs train (692 crops) |
| devnative | its 47 VNs are all val-bucket teacher_val VNs (`devnative/select.py:73,89-94`; `docs/devnative.md:99-101`) | `docs/vocr3.md`, `docs/vocr4.md`, `docs/synth3-native.md:200` | 34 VNs in the sample; 127 lines train |
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
  `synth/bg_exclude.py` rewrote `data/manifests/synth2_bg_exclude.txt`: 71 test VNs, 116 sample VNs (with partners)
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
| realplus (VNDB, 2026-10-13 dump) | 20,849 screenshots of 4,880 VNs (the first carve's 20,867 were fetched on 2026-10-14, with hashes; the re-carve takes out the 18 of its 3 new sample partners. 5 pixel copies within the tier and 2 of realplus v2 screenshots, so 20,842 new) | 20,554 scaled PC (4,844 VNs) and 295 PSP (41 VNs). The old rules had dropped them as `test-vn` 9,461, `val-bucket` 6,126, `shared-release` 4,223, `test-set` 752, `family` 236, fresh native 36; 15 are new in the later dump. All pass the rating, language and kind filters |
| native pool (PSP/Vita frames) | 1,255 frames of 250 VNs | 1,054 PSP, 201 Vita. 924 are already on disk and read (realdata/teacher, lines unlabelled), 295 are new (the same 295 PSP shots as above), 36 are in other pools. They had been dropped as linefind 299, `test-set` 295, `shared-release` 245, `family` 176, `development` 127, owner series 74, `val-bucket` 24, eval candidates 15 |
| both lists, distinct | 21,809 screenshots of 5,079 VNs | |
| synthetic names | 22,163 VNDB characters | synth3's name gate dropped 24,331 of 127,809; it now drops 2,388 |
| synthetic scenes on disk | 416 screenshots | 48 Vita, 108 PSP, 260 scaled PC of synth3's current candidates. Every freed VNDB screenshot can become a scene once fetched and text-detected |
| labelled lines (teacher_val, real_val, devnative) | 1,184 lines of 86 VNs | teacher_val 568, real_val 489, devnative 127 (the sample and its partners keep the rest) |
| owner frames | 637 | Amagami, Narcissu, Danganronpa V3, Iwaihime; CLANNAD's 2,206 stay test. The second drop (2026-10-14) is training data too, apart from its 408 frames of 4 test games (Time Travelers, Utawarerumono, Collar x Malice, Sweet Clown) |

The labelled sets are cut by VN under `$HERD/shared/artifacts/owner-drive-ingest-2/reclaim/pools/` (the re-carve's
cut; the first carve's, under `reserve-policy/reclaim/pools/`, are superseded), in teacher_val's npz format
(`reclaim.py --split`). For each set there is a `*_train.npz` (training data now) and a `*_dev.npz` (the sample).

| set | train crops (VNs) | sample crops (VNs) |
|---|---:|---:|
| teacher_val | 692 (80) | 959 (101) |
| teacher_distill_val (unlabelled) | 522 (77) | 739 (95) |
| real_val | 688 (85) | 807 (99) |
| devnative v3, tv_native ink / GRAD | 121 / 104 (10) | 392 / 319 (34) |
| devnative v3, strict ink / GRAD | 6 / 4 | 32 / 22 |

## Leak check

`eval/tools/reserve_check.py NAME=PATH ...` checks npz pools, line, shot and label manifests, and screenshot
directories. It extends the checks of realplus-v3's `realplus/hygiene.py` (branch feat/realplus-v3) to this policy:

- it blocks a row whose VN is a test VN, both from the committed list and recomputed on a given dump with `--dump`
  (titles, relations and release partners of a newer dump count);
- it blocks a row whose release is a test session or lists a test VN;
- it blocks a vita-test or psp-test screenshot by id, URL or sha256, and by decoded pixels with `--pixels`. The
  pixel and file-hash checks compare against the 2,363 protected images on disk: the set screenshots and the owner's
  CLANNAD frames;
- it blocks a development-sample VN (`--dev-sample info` reports instead);
- it reports the old hash splits, the set games' family and the former reservations as information.

Pool rows (npz meta, line manifests) carry no screenshot hashes. The check joins each row's shot key
(`<vn>_<image>`, the line id less its index) to the screenshot manifests of the pools: realplus, the reclaim tier,
the native and owner pools, devnative, realdata and teacher. With `--pixels` it decodes a local copy when no pixel
hash is known. Each pool reports how many rows were hash-checked (`hash_coverage`) and warns when none were;
`--require-hashes` makes an unhashed row a blocker. An audit of this PR found the first version checked hashes only
on rows that carried them, which was 1 pool of 25. The join fixed that; the results are below.

Run on the 2026-10-13 dump with `--pixels`, the check gave **RESERVE_OK** on 25 training pools and lists, every row
hash-checked (435,783 rows, all with a pixel hash):

- realplus v2 (ink, GRAD, lines);
- realplus v3i3 (ink, GRAD, lines);
- native pool v1 (ink, GRAD, lines);
- owner pool v2 (ink, GRAD) and the 1,215 owner training frames;
- teacher_train, teacher_distill_train, real_train and real_unlabelled, cleaned as below;
- the train parts of teacher_val, teacher_distill_val, real_val and devnative;
- both reclaim lists (the realplus tier after its fetch).

As negative controls, vita-test, psp-test, the sample parts and the original teacher_train fail.

Re-run with the re-carved sample (below) on the same 25 pools, with the re-cut train parts, the reclaim lists of the
re-carve and all 4,994 owner training frames of both drops in place of the first 1,215: **RESERVE_OK**, 439,715 rows,
every one pixel-checked. The first carve's cuts now fail as they should: its teacher_val train part on 65 rows of the 7
VNs that entered the sample, its realplus reclaim list on the 18 screenshots of the 3 new partners.

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

## Limits and follow-ups

- The sample's VNs have no lines in any labelled training pool. 17 of them, however, have screenshots that the old
  gate allowed as synth2 backgrounds (10 in `data/vndb/shots`, 8 in the pinned `synth2_psp_shots.jsonl`; the first
  carve had 19 = 11 + 9, before v15773 and v11677 left for the owner games), so pinned synth2 shards may hold their
  scenes. teacher_val had the same exposure. New renders leave them out.
- `data/manifests/synth3_exclude_clannad.json` is now synth3's only default exclusion manifest. It predates this
  policy: it lists planetarian (v34, training data), so v34's names and scenes stay out of synth3, and it lacks
  v16902, a CLANNAD relation. v16902 is still kept out by VN id wherever the code asks `is_reserved_vn`, but its
  character names are not hashed into the manifest. The next synth render should rebuild the manifest from
  `reserved.json`; the file is pinned by the v3 and synth4 shard recipes.
- `realplus/shots.py dedupe` no longer drops copies of fresh-native test images; they are training data now.
- realplus-v3's branch has its own narrower `reserved_vns_final`. It lacks the r23905 compilation partners, the
  Key-box partners and the development sample. Its build must switch to this reservation before it merges (see the
  hand-off).

## The re-carve after the owner's second drop (2026-10-14)

The owner's second Google Drive drop (2026-10-14, 5,050 frames of 87 games) was sorted into
`shared/private/owner-games/` at 18:14:43Z, adding 79 VN folders. Five of its games were sample VNs, and two sample
VNs share a release with one of its games. Every owner screenshot trains, so `shots.py reserve` was run again on the
same dump with the folders as they are now:

- out of the sample: the owner games v3278 (No Fate!), v3333 (Disgaea Infinite), v11677 (Hanasaku Manimani), v15773
  (Hyakka Yakou) and v16386 (MISSING PARTS), and v701 and v18458, which share a release with owner games;
- into it, by the same tiers and seeded order: v566, v933, v5000, v5349, v7290, v13999, v20062 (all tier 3, scaled-PC
  teacher_val VNs), with three new two-VN partners, v1372, v4827 and v21894;
- the test reservation, the sessions and `test_games.json` are unchanged. 187 VNs are held back (116 in the sample).

What moved with it:

- the labelled-set cuts (table above, re-cut under `owner-drive-ingest-2/reclaim/pools/`): the 7 entering VNs' rows
  go to the `*_dev` side (65 teacher_val, 63 real_val, 38 teacher_distill_val in the first cut's train parts) and the
  7 leaving VNs' to `*_train` (the 4 devnative games among them give devnative's train side 127 lines instead of 65);
- the realplus reclaim list loses the 18 screenshots of the 3 new partners (20,867 to 20,849). realplus-v3's working
  labels (`$HERD/shared/artifacts/realplus-v3/labels.jsonl`) hold 70 rows from 16 of those 18 shots, 16 of them
  auto-kept (v1372 3, v21894 4, v4827 9); realplus-v3 already filters them out of its reads and build (it unions the
  new partners into its exclusions), and none of the checked pools holds them. The native reclaim list gains 38
  pooled frames of 5 leaving VNs (1,217 to 1,255);
- `data/manifests/synth2_bg_exclude.txt` lists the new sample (its test and session lines are unchanged).

Before this change the test of the sample scanned the live owner-games folders (`owner_vns()`), so it failed on any
box where a later drop had been sorted. `reserve` now pins the owner list it carved with (`reserved.json`
`owner_vns`, 102 VNs), and the test checks the sample against that pin and the committed owner drop tables.

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
R=$HERD/shared/artifacts/owner-drive-ingest-2/reclaim                    # the re-carve's lists and cuts
$HERD/heavy.sh .venv-ci/bin/python realplus/reclaim.py --dump $D --out $R --manifest realplus/manifests/reclaim_20261014.json
$HERD/heavy.sh .venv-ci/bin/python realplus/reclaim.py --out $R --split teacher_val=$A/teacher/teacher_val.npz ...
cp $R/pools/split.json realplus/manifests/reclaim_20261014_splits.json
$HERD/heavy.sh .venv-ci/bin/python eval/tools/reserve_check.py --dump $D --pixels NAME=POOL ...
```

A rebuild on a newer dump can only grow the reservation, because `reserved_vns()` unions the committed file. The
sample changes only when `reserve` is run again. Run it again when an owner drop names a sample VN or a release
partner of one; `owner_vns()` then leaves that VN out of the sample. `reserve` pins the owner list it carved with in
`reserved.json` (`owner_vns`), and the tests check the sample against that pin (and the committed drop tables), never
against the live owner-games folders, which grow with every drop.
