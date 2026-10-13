# The owner's games: Memories Off (test only) and CHAOS;HEAD NOAH (training)

The owner captures live screenshots of two games on a PS Vita (owner, 2026-10-13 10:51Z). This
supersedes the 09:25Z and 09:37Z notes; STEINS;GATE, planetarian and Ao no Kanata are not coming.

- **Memories Off** (PSP; VNDB v1152) is test only. The owner's frames go to a new fresh native test set
  (`shared/private/owner-games/memories-off-TEST/`) and to on-console overlay tests. The whole series is
  reserved: nothing may be trained, tuned, calibrated, selected or rendered from it, no font or render
  source may come from it, and its frames never go to Codex.
- **CHAOS;HEAD NOAH** (v22505) is training-eligible. Its owner frames (`shared/private/owner-games/
  chaos-head-noah-v22505/`) may be labelled for training. They are private frames, not public VNDB
  images, so they never go to Codex either.

Some frames may be Adrenaline captures: a PSP game upscaled to 960x544 on the Vita. The source
resolution is recorded per file.

## What is reserved

`realplus/shots.owner_test_vns()` reserves 45 VNs in the 2026-10-13 VNDB dump:

- **The series** (38, reason `owner-test-series`): every VN reachable from v1152 through VNDB's family
  relations (alternative versions, originals, fan discs, parent and side stories, sequels, prequels,
  same series), or titled like the series (Memories Off, メモリーズオフ, 秋之回忆). Several were on the
  training side until now: Your Memories Off ~Girl's Style~ (v1964), Memories Off Festa (v1966),
  Memories Off -Innocent Fille- (v21964) and its for Dearest (v23725), and most of the Chinese fan
  titles.
- **Direct relations of any kind** of a series VN, in either direction (6, `owner-test-relation`).
  These are crossovers and same-setting titles: Never 7 (v248), Chaos;Head (v382, a character
  crossover with Memories Off 6), 夢のつばさ (v5850), KID Mix Section (v26011), MemOff Mix (v27682)
  and SINce Memories (v28683).
- **Release partners** (1, `owner-test-shared-release`): every VN on one VNDB release with a series VN.
  That is てんたま (v11517).

A relation's own relations and release partners are not reserved, so Never 7's compilation partners
(Ever17, Remember11, 12RIVEN) stay on the training side. CHAOS;HEAD NOAH is an alternative version of
Chaos;Head, not a relation of the series, and stays on the training side as the owner asked.

`reserved_vns()` gives these reasons, so realplus, the native pool (`realplus/native.py`) and every
selection built on it leave the 45 VNs out. Any VN sharing a release with one of them is reserved as
`shared-release`, as for the other protected titles.

`data/manifests/synth3_exclude_owner_games.json` (built by `synth/owner_games.py manifest`) is a synth3
exclusion manifest. synth3 renders and `synth3_stats.py` calibrations now use it by default, next to
the fresh-native and CLANNAD manifests. It holds:

- the 45 VNs, the series' relations with their kinds, and the release partners;
- 269 name hashes of the 359 VNDB characters of those VNs. A name row that hashes to one of them is
  never drawn;
- 485 text markers. A pool line or focus context that holds one is never drawn. The markers are every
  title, every two-part full name, and one-part names that hit at most 50 text-pool lines; 49 one-part
  names are common words and are listed but not used;
- the file and decoded-RGB sha256 of every local copy of a VNDB screenshot of those VNs (113 files),
  plus every file in an owner `*-TEST` folder when the manifest was built (none yet).

## What existing data already holds

The rule came after these pools were built. `synth/owner_games.py exposure` counts what they hold of
the 45 VNs, without text (`data/manifests/owner_games_exposure_20261013.json`). The exact row ids per
pool are kept locally in `$HERD/shared/artifacts/native-frames/owner_games_rows.json`.

| pool | rows | series rows |
|---|---:|---:|
| teacher_train | 12,115 | 66 |
| teacher_distill_train (unlabelled) | 9,044 | 97 |
| real_train | 9,811 | 70 |
| real_unlabelled | 11,174 | 86 |
| realplus v1 and v2, ink / GRAD | 71,153 / 53,566 (v2) | 8 / 6 (Innocent Fille for Dearest) |
| **teacher_val** (development) | 1,651 | **8** (v1331, Memories Off After Rain) |
| **devnative v3** (development): tv_native ink / GRAD, strict ink / GRAD | 513 / 423, 38 / 26 | **4 / 3, 4 / 3** (v1331) |
| real_val, teacher_distill_val | 1,495, 1,261 | 12, 10 (v1331) |

- teacher_val and devnative hold Memories Off After Rain, a development VN: any selection on them so
  far has touched a series title.
- synth3's scene candidates (realdata's shots) hold 56 PSP and 20 PC screenshots of the set. synth2's
  scene list holds 13 PSP ones.
- synth3's VNDB name pool has 202 train and 9 dev rows of the set's characters.
- 670 of synth3's ~3.8M text-pool lines hold a marker.

synth3 version 3's renders excluded v1152 and the series' test and development VNs from scenes, but not
the rest of the series.

## Reproduce

```sh
DB=$HERD/shared/artifacts/native-frames/vndb/dump/db          # vndb-db-2026-10-13 (realplus/shots.py dump)
.venv-ci/bin/python synth/owner_games.py manifest --dump $DB --out data/manifests/synth3_exclude_owner_games.json
.venv-ci/bin/python synth/owner_games.py exposure --manifest data/manifests/synth3_exclude_owner_games.json \
    --out data/manifests/owner_games_exposure_20261013.json
```

Rebuild the manifest when the owner's test frames land, so their file and pixel hashes are listed too.
