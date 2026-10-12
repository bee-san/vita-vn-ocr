# Provenance synthetic snapshot, 2026-10-11

This snapshot contains 20,000 synthetic training crops and 1,000 synthetic
development crops for PSP/Vita textbox OCR. It records source ownership, split,
ordered vocabulary and Jiten policy, deliberate correction tags, crop geometry,
recipes and immutable sample IDs. No OCR model was trained or scored. This is a
data preparation and audit result, with no measured accuracy improvement.

[The compact manifest](../data/manifests/provenance_synthetic_20261011.json)
pins every shard and records source URLs, raw source/font/license hashes,
role-specific pool and book-cohort identities, code, dependency/environment
identities, seeds and audit artifacts. Pixels, labels, corpora, fonts, checkpoints
and the large per-chunk registries remain in the ignored local directory
`build/general-vn/provenance-synthetic-20261011`. No dataset release was made.

| Pool | Role | Crops | Shards | Seed | Render workers |
| --- | --- | ---: | ---: | ---: | ---: |
| `broad_ofl` | train | 16,000 | 32 | 20261031 | 2 |
| `broad_dev` | dev | 1,000 | 2 | 20262031 | 1 |
| `kana_corrective` | train | 2,000 | 4 | 20261032 | 1 |
| `kanji_corrective` | train | 2,000 | 4 | 20261033 | 1 |

Each of the 42 shards contains 500 crops. Worker counts come from the snapshot
operator's execution record; the producer recipe does not store that field.
Existing producer fixture tests compare one- and two-worker output identities;
this snapshot was not rerendered to test determinism.

The producer is commit
[`1c55d8d`](https://github.com/bee-san/vita-vn-ocr/tree/1c55d8dac471eda5b98437dda8368b462f6b364c).
The metadata-only source-policy validator is commit
[`01641d1`](https://github.com/bee-san/vita-vn-ocr/tree/01641d124df0d395d14cc3fc1a57a29a89dccfe8).
The rendering environment recorded Python 3.11.15, NumPy 2.4.6, Pillow 12.3.0,
OpenCV 5.0.0, fontTools 4.66.1 and FreeType 2.14.3. The manifest also pins the
dependency lock and OpenCV build hash. No console runtime was run or measured.

## Sources and separation

All nine fonts have separately pinned OFL-1.1 license text: Noto Sans JP, Noto
Serif JP, M PLUS 1p, Zen Maru Gothic, Klee One, BIZ UD Gothic, DotGothic16, and
Fusion Pixel proportional Japanese at 10 and 12 pixels. Google Fonts files use
revision `bd8f81ddb5c74d5c8897b36ad88b440266245103`; Fusion Pixel archives use
release `2026.09.25`. The preflight found full ordered-vocabulary coverage in five
fonts; remaining fonts use the producer's coverage/fallback checks. Actual
selected font references are recorded per crop. There are no game fonts,
screenshot backgrounds or game-script text in this snapshot.

Text comes from the pinned `globis-university/aozorabunko-clean` archive at
revision `42a9c9c0f1d67e6a5554d9bea4201973dc9b049c`, whose source metadata declares
CC-BY-4.0. Modern-kana books were assigned by
`sha256(seed:book_id) uint64 modulo10: dev0,train1..9`, seed 20261011. The frozen
cohorts contain 9,225 train books and 1,021 development books, with no common book
IDs. Training pools have 50,000 dialogue and 75,000 narration chunks. Preparation
removed 178 exact dialogue matches and 54 exact narration matches from the
development pools, leaving 9,822 and 14,946 chunks. The resulting exact source
chunk intersection is zero. No additional unsafe chunks were found at this
preparation stage; the earlier pool builder had already excluded out-of-vocabulary
chunks. Unsafe source chunks are rejected whole, without deleting characters.

The earlier pool builder did not retain each chunk's original book ID. The new
source indexes preserve the original pool SHA, zero-based source line, raw chunk
SHA and cohort SHA, but cannot reconstruct the missing per-chunk book identity.
The whole-book cohort split and exact chunk exclusion remain independently
pinned. This limitation must remain visible in future snapshot comparisons.

The vocabulary is the released ordered 3,500-class file with SHA256
`7f1cb2eb073ddae5f6c39bdd49562cd3bd3ac757f5ac8c0eca4bd98beda4fed8`.
The original Jiten policy SHA256 is
`8fc9ea20804cc45876e25b60ed669446dc75fc7f0b6b383d65cc4688e050f763`.
All 3,112 supported kanji belong to the pinned Jiten list. The manifest retains
Jiten and similar-kanji source URLs, revisions, license metadata and hashes.
The ordered vocabulary was not expanded for this snapshot.

Backgrounds and name/dressing text are procedural identities bound to each
recipe and renderer hash. All runs use gradient text finding, dark-text
inversion, imperfect textbox edge jitter and zero native-font probability.
PSP, PSP 2× and Vita rendering scales are sampled; these are synthetic styles,
not evidence from console captures. Ruby, cursors and partial/typewriter ghosts
are unlabelled distractors. Both original and final crop boxes must preserve
labelled glyph cores.

The input manifest reserves 184 VN IDs, including the development-only Corpse
Party `v2632`. That inventory is pinned here; later training still requires its
separate generic real-development readiness and complete isolation checks.
The `broad_dev` shards cannot enter the composed training policy. This snapshot
does not replace the original CLANNAD or original synthetic selection gates.

## Audit result and training exposure

The immutable audit SHA256 is
`58074b6deb62363b454e7c0803ffa0f1a14df5765699c05d3be7038106126686`.
The audit checked all 21,000 stored crops and reported zero:

- out-of-vocabulary or Jiten label violations;
- CTC-infeasible labels at height 32;
- crop boxes excluding the recorded labelled glyph core;
- game/unknown asset references or duplicate sample IDs;
- train/dev pixel matches under raw, inverted, height-32-resized and inverted
  height-32-resized identities, each including image shape.

The separate source-policy byte/metadata validation found 16,000 verified broad
fallback rows and zero unknown rows across the 20,000 training crops. This checks
provenance and input identity, and does not authorize bypassing readiness gates.
The snapshot operator viewed all seven contact sheets, covering 112 selected
crops, and observed no missing glyph, clipped label or label mismatch. Some
crops deliberately have blur, low contrast, finder-edge clutter, cursors or
ruby. Visual sampling and recorded core bounds do not hand-check every crop.

The 2,000-row kana pool deliberately requests at least one of `ず` or `っ` and
uses the canonical `kana:ずっ` pattern. Ordinary occurrences in the broad pools
have no correction tag. The 2,000-row kanji pool uses Jiten-approved confusable
pairs. Its observed rank-band counts are 1,704 common, 200 uncommon, 80 rare and
16 very rare; pool frequencies are not training exposure frequencies.

Actual forwarded exposure has not been measured: there were zero OCR model
forwards. The separate consumer must cap every forwarded prefix at 0.5% for each
deliberate correction pattern and 2% for all corrective-kanji rows, with verified
`broad_ofl` fallback. Its game font/text cap is 5% per VN and all-game union cap
is 20%; no game assets occur here. Concatenating the corrective pools as ordinary
training data would exceed these intended limits. Counts, queue weights and
available pool sizes do not prove that the caps were met.

## Reproduction and byte verification

The following are the recorded recipe-equivalent rendering commands. Use the
producer commit and the exact privately prepared assets, input manifest and
vocabulary. Run from the producer checkout, with the absolute private input paths shown below.
Each command requires a new, absent output directory. Original
directories, source manifests, policies and audit artifacts remain immutable.
Change the output root for a rerun, then compare recipes and shard hashes; no
rerender comparison is reported here.

```sh
snapshot_root=/home/bee/Documents/src/vita-vn-ocr/build/general-vn/provenance-synthetic-20261011
snapshot_output=build/general-vn/provenance-synthetic-reproduction
snapshot_python=/home/bee/Documents/src/vita-vn-ocr/.venv-ci/bin/python
snapshot_vocabulary=/home/bee/Documents/src/vita-vn-ocr/build/hiragana/vocab.txt
"$snapshot_python" synth/provenance_producer.py --manifest "$snapshot_root/producer-inputs.json" --asset-root "$snapshot_root/assets" --vocab "$snapshot_vocabulary" --kanji-policy data/manifests/fl10_kanji_policy_20261011.json --split train --source broad_ofl --broad --seed 20261031 --shards 32 --per-shard 500 --workers 2 --finder grad --crop-edge-jitter --out "$snapshot_output/broad-train"
"$snapshot_python" synth/provenance_producer.py --manifest "$snapshot_root/producer-inputs.json" --asset-root "$snapshot_root/assets" --vocab "$snapshot_vocabulary" --kanji-policy data/manifests/fl10_kanji_policy_20261011.json --split dev --source broad_dev --broad --seed 20262031 --shards 2 --per-shard 500 --workers 1 --finder grad --crop-edge-jitter --out "$snapshot_output/broad-dev"
"$snapshot_python" synth/provenance_producer.py --manifest "$snapshot_root/producer-inputs.json" --asset-root "$snapshot_root/assets" --vocab "$snapshot_vocabulary" --kanji-policy data/manifests/fl10_kanji_policy_20261011.json --split train --source kana_corrective --target-kana ずっ --seed 20261032 --shards 4 --per-shard 500 --workers 1 --finder grad --crop-edge-jitter --out "$snapshot_output/kana-train"
"$snapshot_python" synth/provenance_producer.py --manifest "$snapshot_root/producer-inputs.json" --asset-root "$snapshot_root/assets" --vocab "$snapshot_vocabulary" --kanji-policy data/manifests/fl10_kanji_policy_20261011.json --split train --source kanji_corrective --kanji-pairs --seed 20261033 --shards 4 --per-shard 500 --workers 1 --finder grad --crop-edge-jitter --out "$snapshot_output/kanji-train"
```

The private preparation and audit scripts are pinned by code hash and retained
locally. Those scripts and prepared pools are not distributed here, so downloading
the upstream archives alone does not reproduce the exact snapshot. The manifest's
`recipe_common` plus each set's `recipe_overrides` reconstructs its original
canonical recipe; full used registries remain pinned through source-manifest
hashes.

The audit artifact comes from the privately pinned `audit_snapshot.py`, which
asserts the checks above and writes the immutable audit/contact artifacts. The
stored `source-validation.json` is pinned separately. During preparation of this
metadata record, source-policy validation was independently repeated with this
read-only command, without writing outputs or decoding crop arrays:

```sh
/home/bee/Documents/src/vita-vn-ocr/.venv-ci/bin/python /tmp/vita-vn-ocr-general-vn-source-balancing/train/source_policy.py --policy /home/bee/Documents/src/vita-vn-ocr/build/general-vn/provenance-synthetic-20261011/training-policy.json
```

For review, the following read-only command verifies compact metadata against
the actual source bytes without loading crop arrays or model tensors. Run from
this metadata checkout with the ignored snapshot accessible at the supplied path:

```sh
python - /home/bee/Documents/src/vita-vn-ocr/build/general-vn/provenance-synthetic-20261011 <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
manifest = json.loads(Path('data/manifests/provenance_synthetic_20261011.json').read_text())
def check(pin):
    path = (root / pin['file']).resolve()
    assert path.is_relative_to(root) and path.is_file()
    assert path.stat().st_size == pin['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin['sha256']
for pin in manifest['artifacts'].values(): check(pin)
for font in manifest['sources']['fonts'].values():
    check(font); check(font['license_file'])
for pool in manifest['preparation']['text_pools'].values():
    check(pool); check(pool['source_index']); check(pool['source_book_cohort'])
for cohort in manifest['preparation']['cohorts'].values(): check(cohort)
for group in manifest['sets'].values():
    check(group['source_manifest'])
    recipe = {**manifest['recipe_common'], **group['recipe_overrides']}
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert identity == group['recipe_sha256']
    for shard in group['shards']: check(shard)
    assert sum(shard['samples'] for shard in group['shards']) == group['counts']['samples']
    if 'standalone_training_policy' in group: check(group['standalone_training_policy'])
assert sum(group['counts']['samples'] for group in manifest['sets'].values()) == 21000
print('verified 21,000 crops in 42 shards by metadata and file hashes; zero OCR forwards')
PY
```
