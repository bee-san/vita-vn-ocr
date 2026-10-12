# Provenance-aware synthetic snapshots

`synth/provenance_producer.py` is an opt-in wrapper around the existing renderer. It reads only explicit, hash-pinned local inputs and writes a new snapshot directory. `synth/render.py` and existing shards/manifests keep their current behavior. This is a producer for later training; generating crops does not measure OCR accuracy or enforce training exposure limits.

Broad fallback runs use OFL fonts, explicitly reviewed non-game text and procedural backgrounds. The constructor excludes native fonts, game/unknown text and screenshot backgrounds before reading their files. It rejects deliberate kana targets, pair sampling and nonzero native probability. Ordinary occurrences of `ず`, `っ` or similar kanji have no correction tag. A mixed/targeted snapshot needs a separately generated broad fallback source when composing a training policy.

Every eligible text file must contain one nonempty source chunk per UTF-8 line. Chunks must already fit the ordered model vocabulary and pinned Jiten policy. Any unsafe chunk rejects the input; characters are never deleted from source sentences to make them fit. A policy is required for any Han text, including Unicode extension blocks, and for deliberate pair sampling. Reuse the exact released vocabulary file whose byte hash the policy pins; files with the same class count can have a different class order.

## Input manifest

Keep assets and their manifests under an ignored private directory. All file paths are relative to `--asset-root`; absolute paths, `..` and escaping symlinks fail. Each file pin is `{file, bytes, sha256}` with an exact positive byte count and lowercase SHA256. No files are fetched. The following is a template: replace angle-bracket values with measured local pins and real identities.

```json
{
  "schema": 1,
  "kind": "synthetic-producer-inputs",
  "forbidden_vns": ["<development-or-protected-vN>"],
  "assets": {
    "font:broad-gothic": {
      "kind": "font", "origin": "ofl", "license": "OFL-1.1",
      "file": "fonts/face.ttf", "bytes": 123, "sha256": "<font SHA256>",
      "license_file": {"file": "fonts/OFL.txt", "bytes": 123, "sha256": "<license SHA256>"},
      "format": "truetype", "category": "gothic", "family": "<renderer family name>"
    },
    "text:broad-dialogue": {
      "kind": "text", "origin": "non_game", "game_free_reviewed": true,
      "file": "text/dialogue-train.txt", "bytes": 123, "sha256": "<raw UTF-8 file SHA256>",
      "split": "train", "pool": "dialogue"
    },
    "text:broad-narration": {
      "kind": "text", "origin": "non_game", "game_free_reviewed": true,
      "file": "text/narration-train.txt", "bytes": 123, "sha256": "<raw UTF-8 file SHA256>",
      "split": "train", "pool": "narration"
    }
  }
}
```

The renderer can sample choice lines from dialogue and narration pools, so provide both. Supported pools are `dialogue`, `narration` and `wiki`. Fonts require a nonempty family, category from `render.CATEGORY`, and a full-width space glyph. OFL fonts require the separately pinned OFL-1.1 licence text. Name generation uses only reviewed non-game frequency inputs or common Jiten/vocabulary priors; game and unknown text cannot seed name frequencies.

Text/background inputs require explicit `split: train|dev`. Fonts can omit split for shared typography, but their declared game ownership remains counted in every role. All asset roles and train/dev byte aliases are checked before reading any asset file. Inputs belonging to the other split are never decoded. `--split dev` cannot deliberately target corrections and never emits a training policy. Reserve development/protected VN IDs in `forbidden_vns`; existing renderer test-game exclusions also apply. Keep held-out books separate when preparing text files.

Owned game fonts/text/backgrounds must declare `origin: game`, canonical `vn: vN` and `release: rN`. Missing ownership cannot be called game-free; an explicitly `unknown` asset remains unknown and cannot qualify as fallback. The complete declared registry, including excluded resources and native metrics, is checked for conflicting raw-byte ownership before reading assets. A native font adds `format: native_bitmap` and a `metrics` file pin for its `.pro` alongside the `.fnt` pin. Both components enter its per-crop font references. Existing glyph ambiguity checks exclude indistinguishable labelled characters from native coverage and allow OFL fallback.

Screenshot backgrounds add `platform: psp|psv`, `scenery_reviewed: true` and a reviewed `scenery_box: [left, top, right, bottom]`. Require exact native dimensions, 480×272 or 960×544. Register only approved scenery without text; unsuccessful patches produce a procedural background reference. This wrapper does not discover screenshots or infer ownership from pixels or filenames. Use screenshot inputs only in a deliberately mixed run.

## Produce a new snapshot

```sh
.venv-ci/bin/python synth/provenance_producer.py \
  --manifest build/general-vn/producer-inputs.json \
  --asset-root build/general-vn/producer-assets \
  --vocab build/hiragana/vocab.txt \
  --kanji-policy data/manifests/fl10_kanji_policy_20261011.json \
  --split train --source broad_ofl --broad --seed 20261011 \
  --shards 2 --per-shard 500 --workers 2 --finder grad \
  --crop-edge-jitter --out build/general-vn/broad-ofl-new
```

`--crop-edge-jitter` retains the renderer's native PSP, PSP 2× and Vita scale choices, with imperfect textbox margins and procedural edge clutter. The wrapper additionally measures alpha≥128 glyph bounds at the actual fractional drawing origins for all fonts. Both the original finder crop and final jittered crop must preserve the labelled ink. Rejected crops are not widened. Ruby, cursors and typewriter ghosts remain unlabelled distractors.

A deliberate kana run omits `--broad` and adds, for example, `--target-kana ずっ`. The target selector means “at least one requested kana”, matching the existing renderer. Its pattern ID is derived as `kana:<sorted unique targets>`; an optional `--correction-pattern-id` must equal that value. Reordering or repeating targets cannot create new exposure allowances. `--kanji-pairs` is a separate deliberate stream: both endpoint orders use the same sorted `kanji-pair:<pair>` ID. `--native-probability` is a requested region probability; actual font use is recorded after coverage/fallback checks. None of these settings bypasses the later exposure caps.

## Output identity and policy composition

Each NPZ row has explicit `split` and `source_provenance`:

```json
{
  "schema": 1,
  "sample_id": "synth:<recipe SHA256>:<shard index>:<crop index>",
  "asset_ids": ["<selected font>", "<source chunk>", "<selected background>"],
  "correction_pattern_ids": [],
  "kanji_corrective": false,
  "broad_sampling": true
}
```

Source chunk references survive cleaning of procedural dressing, slicing, wrapping, partial text and inline-name concatenation. A line retains every contributing chunk reference conservatively when sliced. Text asset identity is SHA256 of canonical compact JSON `{source_sha256, line, chunk_sha256}` with sorted keys, a zero-based line index and a raw UTF-8 chunk hash. Registry entries retain `source_asset_id`, `source_sha256`, `chunk_sha256` and `chunk_index`. Thus a common phrase from separate corpora does not acquire conflicting ownership; the raw corpus bytes are still pinned. Identical asset identities cannot be relabelled with conflicting kind/origin/VN/release.

The immutable recipe pins the full input-manifest hash, all raw corpus/asset hashes, exact ordered vocabulary bytes, Jiten-policy bytes, seed/options, renderer/helper code, exclusion manifests, dependency lock and relevant runtime library versions. Procedural name/dressing/background identities bind that recipe and the exact renderer code. Input hashes are rechecked before and after rendering; a changed lazy font/background input prevents publishing the snapshot manifest. Sample IDs remain identical for the same recipe and seed across worker counts. Changing inputs, code or options creates a new recipe and new IDs. An existing output directory is refused.

Every run writes `source-manifest.json` with the used registry, exact shard paths/hashes/bytes/sample counts, roles and recipe. Only a fully broad training run also writes self-contained `policy.json` with `kind: general-vn-source-policy`, `fallback_source` equal to its source name and a complete training shard inventory. Composition must keep files under the policy directory, merge registry IDs without changing their ownership, include every declared training file exactly once, and designate a separate entirely verified broad fallback source. Old shards are not rewritten or inferred from their appearance.

The consumer owns the forwarded-exposure caps: 5% per game's font/text union, 20% union of all game assets, 0.5% per deliberate correction pattern and 2% corrective-kanji rows. Snapshot counts and queued samples are not proof of those limits. The producer therefore reports `exposure_caps_enforced: false`; measure consumed rows through the consumer ledger and real-screenshot CER through the separate frozen evaluation protocol.

## Fixture validation

```sh
.venv-ci/bin/python -m pytest -q tests/test_provenance_producer.py
.venv-ci/bin/ruff check synth/provenance_producer.py tests/test_provenance_producer.py
```

Tests use the locally cached pinned OFL CI font, generated native bitmaps/backgrounds and the repository's small fixture corpus. Generated policy fixtures do not rewrite or weaken frozen policy/vocabulary files. No game archive, protected image, model training or OCR scoring is required.
