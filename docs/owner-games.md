# The owner's games: test-only series and training-eligible titles

The owner captures live screenshots of games on a PS Vita (owner, 2026-10-13 10:51Z, 13:10Z and 13:12Z).
This supersedes the 09:25Z and 09:37Z notes; STEINS;GATE, planetarian and Ao no Kanata are not coming.
The owner's folders are under `shared/private/owner-games/`. A `-TEST` suffix means test only.

**Test only.** The owner's frames of these titles go to new fresh native test sets and to on-console
overlay tests. Each series is reserved: nothing may be trained, tuned, calibrated, selected or rendered
from it, no font or render source may come from it, and its frames never go to Codex.

- **Memories Off** (PSP; v1152), with its whole series.
- **Aiyoku no Eustia** (Vita; v3770).
- **428** (PSP; v1299).

The last two were reserved at 13:10Z, before any harvest touched them. The owner described them as new
and never seen anywhere. Existing pools do hold some of their lines, scaled from PC and other screenshots
(table below).

**Training-eligible** (`OWNER_TRAINING`). These may be labelled for training, re-checked against
`reserved_vns` at ingest. They are private frames, so they never go to Codex.

- Vita: CHAOS;HEAD NOAH (v22505), CHAOS;CHILD (v14018), Rewrite (v751), ATRI (v27448).
- PSP: Sharin no Kuni (v57), Toradora! Portable (v4137), Kikokugai (v183), Fate/EXTRA CCC (v12369),
  Togainu no Chi (v434), Otome wa Boku ni Koishiteru (v195), Mashiro-iro Symphony (v1552),
  Koi to Senkyo to Chocolate (v4028).

None of them is reserved under the rules below (checked on the 2026-10-13 dump). Owner frames of any
reserved title (the popular PSP titles the owner listed are) are never accepted as training.

Some frames may be Adrenaline captures: a PSP game upscaled to 960x544 on the Vita. The source
resolution is recorded per file.

## What is reserved

`realplus/shots.owner_test_vns()` reserves 50 VNs in the 2026-10-13 VNDB dump:

- **The series** (40, reason `owner-test-series`): every VN reachable from a root (v1152, v3770, v1299)
  through VNDB's family relations (alternative versions, originals, fan discs, parent and side stories,
  sequels, prequels, same series), or titled like Memories Off (Memories Off, メモリーズオフ, 秋之回忆).
  Memories Off's series is 38 VNs. Several of them were on the training side until now: Your Memories Off
  ~Girl's Style~ (v1964), Memories Off Festa (v1966), Memories Off -Innocent Fille- (v21964) and its for
  Dearest (v23725), and most of the Chinese fan titles. Aiyoku no Eustia and 428 have no family relations.
- **Direct relations of any kind** of a series VN, in either direction (9, `owner-test-relation`).
  These are crossovers and same-setting titles:
  - for Memories Off: Never 7 (v248); Chaos;Head (v382), a character crossover with Memories Off 6;
    夢のつばさ (v5850); KID Mix Section (v26011); MemOff Mix (v27682); SINce Memories (v28683);
  - for 428: 街 (v1360), タイムトラベラーズ (v10856) and 3年B組金八先生 (v45398).
- **Release partners** (1, `owner-test-shared-release`): every VN on one VNDB release with a series VN.
  That is てんたま (v11517).

`owner_test_vns()` does not reserve a relation's own relations or release partners. CHAOS;HEAD NOAH is an
alternative version of Chaos;Head, not a relation of the series, and stays on the training side as the
owner asked.

`reserved_vns()` gives these reasons, so realplus, the native pool and every selection built on it leave
the 50 VNs out. Its general rule then reserves any VN on a release with one of them as `shared-release`,
as for the other protected titles. Against main's rules, this PR newly reserves 20 VNs:

- 19 of the set; the other 26 were already reserved for other reasons;
- Ever 17 - The Out of Infinity (v19373), which shares the Switch Ever17 / Never7 double pack with Never 7.
  Its rows in existing pools: teacher_train 11, teacher_distill_train 5, real_train 10, real_unlabelled 8,
  realplus v2 ink / GRAD 3 / 2.

Never 7's other compilation partners (Ever17, Remember11, 12RIVEN) were already reserved. synth3's
manifest holds the 50 VNs only, so v19373 stays usable for synth3's names and scenes.

`data/manifests/synth3_exclude_owner_games.json` (built by `synth/owner_games.py manifest`) is a synth3
exclusion manifest. synth3 renders and `synth3_stats.py` calibrations now use it by default, next to
the fresh-native and CLANNAD manifests. It holds:

- the 50 VNs, the series' relations with their kinds, and the release partners;
- 503 name hashes of the 593 VNDB characters of those VNs. A name row that hashes to one of them is
  never drawn;
- 730 text markers. A pool line or focus context that holds one is never drawn. The markers are every
  title, every two-part full name, and one-part names that hit at most 50 text-pool lines; 84 one-part
  names are common words and are listed but not used;
- the file and decoded-RGB sha256 of every local copy of a VNDB screenshot of those VNs (144 files),
  plus every file in an owner `*-TEST` folder when the manifest was built (none yet).

## What existing data already holds

The rules came after these pools were built. `synth/owner_games.py exposure` counts what they hold of
the 50 VNs, without text (`data/manifests/owner_games_exposure_20261013.json`). The exact row ids per
pool are kept locally in `$HERD/shared/artifacts/native-frames/owner_games_rows.json`.

| pool | rows | set rows | of which 428 / Eustia / 街 (v1299 / v3770 / v1360) |
|---|---:|---:|---|
| teacher_train | 12,115 | 123 | 14 / 2 / 41 |
| teacher_distill_train (unlabelled) | 9,044 | 122 | 2 / 2 / 21 |
| real_train | 9,811 | 111 | 14 / 4 / 23 |
| real_unlabelled | 11,174 | 127 | 3 / 2 / 36 |
| realplus v1 and v2, ink / GRAD | 71,153 / 53,566 (v2) | 8 / 6 (Innocent Fille for Dearest) | 0 |
| **teacher_val** (development) | 1,651 | **8** (v1331, Memories Off After Rain) | 0 |
| **devnative v3** (development): tv_native ink / GRAD, strict ink / GRAD | 513 / 423, 38 / 26 | **4 / 3, 4 / 3** (v1331) | 0 |
| real_val, teacher_distill_val | 1,495, 1,261 | 12, 10 (v1331) | 0 |

- teacher_val and devnative hold Memories Off After Rain, a development VN: any selection on them so
  far has touched a series title.
- synth3's scene candidates (realdata's shots) hold 65 PSP and 28 PC screenshots of the set (9 PSP shots of
  街, 4 PC shots each of 428 and Eustia). synth2's scene list holds 13 PSP ones.
- synth3's VNDB name pool has 437 train and 15 dev rows of the set's characters.
- 1,217 of synth3's ~3.8M text-pool lines hold a marker.

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
