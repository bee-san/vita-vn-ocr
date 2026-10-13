# The owner's Vita games: STEINS;GATE and planetarian (test only)

The owner supplies two PS Vita games, STEINS;GATE (VNDB v2002) and planetarian (v34). Their frames are a
fresh native console test set and material for on-console overlay tests only (owner, 2026-10-13 09:25Z):
nothing may be trained, tuned, calibrated or selected on them, no font or render source may come from
them, and their frames never go to Codex. The owner's frames stay outside Git, under
`shared/private/owner-games/`.

## What is reserved

`realplus/shots.owner_games_vns()` reserves 28 VNs in the 2026-10-13 VNDB dump:

- the two games (reason `owner-game`);
- every direct VNDB relation of any kind, in either direction (18, `owner-game-relation`). For
  STEINS;GATE these are its fan discs, side stories, the Steins;Gate 0 side story and the
  Science Adventure titles VNDB links as the same setting. For planetarian they are Snow Globe and
  CLANNAD: Lost "Winter";
- every VN on a VNDB release with one of these (8, `owner-game-shared-release`). These include the Key
  compilation boxes, which put CLANNAD, Tomoyo After, Kanon, AIR and Little Busters! on a release with
  planetarian.

`realplus/shots.reserved_vns()` now gives these reasons, so realplus, the native pool
(`realplus/native.py`) and every selection built on `reserved_vns` leave the 28 VNs out. A VN sharing a
release with any of them is reserved as `shared-release`, as for the other protected titles.

`data/manifests/synth3_exclude_owner_games.json` (built by `synth/owner_games.py manifest`) is a synth3
exclusion manifest. synth3 renders and `synth3_stats.py` calibrations now use it by default, next to
the fresh-native and CLANNAD manifests. It holds:

- the 28 VNs, the relations with their kinds, and the release partners;
- 172 name hashes of the 186 VNDB characters of those VNs. A name row that hashes to one of them is
  never drawn;
- 249 text markers. A text-pool line or focus context that holds one is never drawn. The markers are
  every title, every two-part full name, and one-part names that hit at most 50 pool lines. Seven
  one-part names are common words and are listed but not used: ボタン, ポテト, みちる, るか, エリン,
  グリム, レノン;
- the file and decoded-RGB sha256 of every local copy of a VNDB screenshot of those VNs (74 files),
  plus any file under `shared/private/owner-games/` when it was built (none yet).

## What existing data already holds

The rule came after these pools were built. `synth/owner_games.py exposure` counts, without text, what
they hold of the 28 VNs (`data/manifests/owner_games_exposure_20261013.json`). The exact row ids per pool
are kept locally in `$HERD/shared/artifacts/native-frames/owner_games_rows.json`.

| pool | rows | owner-set rows | of which STEINS;GATE (v2002) | planetarian or Snow Globe (v34, v27747) |
|---|---:|---:|---:|---:|
| teacher_train | 12,115 | 72 | 12 | 4 |
| teacher_distill_train (unlabelled) | 9,044 | 65 | 8 | 0 |
| real_train | 9,811 | 52 | 0 | 3 |
| real_unlabelled | 11,174 | 69 | 12 | 4 |
| realplus v1 and v2, ink / GRAD | 71,153 / 53,566 (v2) | 6 / 6 | 0 | 0 |
| teacher_val, real_val, devnative (every version) | | 0 | 0 | 0 |

The rest of the owner-set rows belong to the Key titles and the Science Adventure titles. realplus's six
rows are CHAOS;HEAD NOAH (v22505), a v2002 relation.

synth2's scene list holds 3 Vita and 7 PSP screenshots of STEINS;GATE: My Darling's Embrace (v6618) and
8 PSP screenshots of v3091. synth3's scene candidates (realdata's shots) include 4 STEINS;GATE PC
screenshots and 3 of Snow Globe. synth3's VNDB name pool has 158 rows (train) and 23 (dev) whose name is
a character of the set. 172 of synth3's ~3.8M text-pool lines hold a marker. synth3 version 3's renders
already excluded planetarian and the Key boxes (CLANNAD manifest) but not STEINS;GATE. Renders from now on
exclude all of them.

The conductor's decision (09:50Z): VOCR3 round 1 keeps its pinned pools and discloses these counts. Every
pool built from now on (realplus v3, the native pool, round-2 study pools) excludes the 28-VN set.

## Reproduce

```sh
DB=$HERD/shared/artifacts/native-frames/vndb/dump/db          # vndb-db-2026-10-13 (realplus/shots.py dump)
.venv-ci/bin/python synth/owner_games.py manifest --dump $DB --out data/manifests/synth3_exclude_owner_games.json
.venv-ci/bin/python synth/owner_games.py exposure --manifest data/manifests/synth3_exclude_owner_games.json \
    --out data/manifests/owner_games_exposure_20261013.json
```

Rebuild the manifest when the owner's frames land, so their file and pixel hashes are listed too.
