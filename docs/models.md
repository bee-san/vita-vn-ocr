# Models: every asset of release v0.5.0

Release [v0.5.0](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.5.0) collects every model this repository
has published since VOCR2 in one place: the VOCR6 recognizers, the dialogue-line spotter, the character LMs, and every
earlier model of those rounds, each byte for byte as its own release published it. Nothing was retrained, re-exported
or changed. 86 files come from 20 releases; with `SHA256SUMS` and `v0.5.0-assets.json` (this document's
`docs/models.json`) the release has 88.

- **Checked.** Every file was downloaded from its source release and checked against GitHub's sha256 digest of that
  asset and, for the 78 files its release's `SHA256SUMS` lists, against that file too (`docs/models.py manifest`
  repeats the second check and checks Rubi's pins). The release's own `SHA256SUMS` lists the other 87 files; the
  published release is downloaded again and every file checked against it and against `docs/models.json`
  (`docs/models.py verify`).
- **Names.** A file keeps its release name, except where two releases hold different bytes under one name, or the
  name says nothing on its own:
  - each release's `ATTRIBUTION.md` is `ATTRIBUTION_<tag>.md`, or the name Rubi ships it under (`VOCR6-`, `MODEL-`,
    `LM-`, `SPOTTER-`, `GATE-` and `VERTICAL-ATTRIBUTION.md`);
  - v0.4.0's and v0.4.1's model cards and asset lists carry the set's number (`MODEL_CARD_VOCR6.1_N.md`,
    `vocr6.2-assets.json`, ...);
  - v0.3.7-lm6.2's `settings.json` is `LM-settings.json`.

  The tables below give each file's source name.
- **Measured** means measured on the host: the int8 weights through the plain-C runtime, dialogue CER on the sets
  named, from the records cited. **Console times and memory are estimates** (Cortex-A9 cycle models calibrated to
  VOCR3-H15's measured console rate). Nothing of VOCR6 or the spotter has run on a console.

## What Rubi uses

Rubi's pins ([bee-san/Rubi_for_vita](https://github.com/bee-san/Rubi_for_vita) main `dc272d1`, 2026-10-28, what
the next Rubi release, 0.10.0, ships: `tools/prepare_ocr.py` and `include/vjo_vocr_sizes.h`, each file by size and
SHA-256) name these files of this release, byte for byte:

| role in Rubi | file | bytes |
|---|---|---:|
| Small model size | `VOCR6.2_S_w8.vocr` | 819,792 |
| **Normal model size (the default)** | `VOCR6.2_N_w8g2.vocr` | 1,189,408 |
| Large model size | `VOCR6.2_L_w8_2l.vocr` | 2,475,680 |
| the line finder (`ocr_finder = spotter`, the default; GRAD is the fallback) | `textspot8_t64n8c16_w8.vtsp` | 52,176 |
| vertical text (`ocr_vertical`) | `VOCR5_V_w8.vocr` | 819,792 |
| the beam + LM decoder's four LM tiers (1 MB by default) | `vocr_lm_o3_500k.vlm`, `vocr_lm6_o3_1000k.vlm`, `vocr_lm_o3_2000k.vlm`, `vocr_lm6_o3_4000k.vlm` | 499,824 / 982,320 / 1,971,568 / 3,603,552 |
| the text-presence gate (`ocr_text_gate`, off by default) | `textgate_p8_c8_w8.vtd` | 1,540 |
| the attribution files beside them | `VOCR6-ATTRIBUTION.md`, `MODEL-ATTRIBUTION.md`, `LM-ATTRIBUTION.md`, `SPOTTER-ATTRIBUTION.md`, `GATE-ATTRIBUTION.md`, `VERTICAL-ATTRIBUTION.md` | |

- **Rubi's host tools and tests only** (pinned, never shipped): `VOCR6.2_N_w8.vocr` and `VOCR6.2_L_w8.vocr`, the
  plain Normal and Large weights its tests read against the shipped blobs with the head sections off;
  `VOCR4_W20_w8.vocr`, the cascade model of a request by name; `VOCR3_H15_w8.vocr` and `VOCR3_FL10_w8.vocr`.
- **Rubi 0.9.8**, the latest Rubi release, pins VOCR5's v0.3.6-head2level.1 blobs as its sizes (`VOCR5_S_w8g.vocr`,
  `VOCR5_N_w8g2.vocr`, `VOCR5_L_w8g2.vocr`) and no spotter; its LMs, vertical model and gate are the ones above.

## The recognizers

Dialogue CER, measured on the host. "Rubi's path" is the final reads' model of Rubi's line path (the 1 MB LM at Rubi's
settings, the cascade S → N → L, Rubi 892cfb9's GRAD finder with its COLGAP fix); "greedy" is the recognizer alone.
Sources: `docs/vocr6.md` and `eval/vocr6_final_test_followup_20261027.json` (VOCR6), `docs/vocr5.md` (VOCR5),
`docs/vocr4.md`, `docs/vocr3.md`, `docs/vocr2-qualification.md`, `docs/vertical.md`, `docs/vocr6-psp.md`.

For VOCR6, two kinds of number are not the shipped blobs' own read:
- **The held-out pages** are the mean of three seeds of each tier's recipe, held out from hval: stage 4d's for Small,
  stage 4a's (v0.4.0-vocr6.1's 15,000-step recipe) for Normal and Large. The shipped blobs then trained on hval too,
  and Normal and Large three times as long, so their own hval reads (0.98 / 0.88 / 0.87%) are not held out.
- **The dev score** (VOCR5's development score on other games) is for the set as it ships, Rubi's cascade S → N → L,
  in point. Small's drop comes from the better Normal above it: VOCR6-S against VOCR5-S in the same cascade reads
  +0.041 [−0.016, +0.099], worse in point (not resolved).

| file | model, tier | W8I arena | est. s per 3-line box, 333 / 444 MHz | measured |
|---|---|---:|---|---|
| `VOCR6.2_S_w8.vocr` | VOCR6-S, Small | 124,832 | 0.63 / 0.50 | hooked games' held-out pages, the recipe's 3 seeds: 1.049% (VOCR5-S 1.177%); other games' dev score, the set as it ships: 1.549 (VOCR5-S's set 1.674; like for like +0.041, above); read once: CLANNAD 0.47% on Rubi's path, 0.72% greedy; vita-test + psp-test 0.58%; Dance with Devils 0.21% |
| `VOCR6.2_N_w8g2.vocr` | VOCR6-N, Normal (default) | 184,448 | 0.77 / 0.61 | held-out pages, stage 4a's recipe, 3 seeds: 0.936% (VOCR5-N 1.008%); dev score, the set as it ships: 1.389 (1.482); read once: CLANNAD 0.33% (0.40% greedy), vita-test + psp-test 0.36%, Dance with Devils 0.19% |
| `VOCR6.2_L_w8_2l.vocr` | VOCR6-L, Large | 262,480 | 1.88 / 1.47 | held-out pages, stage 4a's recipe, 3 seeds: 0.917% (VOCR5-L 0.990%); dev score, the set as it ships: 1.396 (1.439); read once: CLANNAD 0.31% (0.26% greedy), vita-test + psp-test 0.40%, Dance with Devils 0.19% |
| `VOCR6.2_N_w8.vocr`, `VOCR6.2_L_w8.vocr` | the same Normal and Large, plain | 182,944 / 261,024 | 0.92 / 0.71, 2.07 / 1.62 | the shipped blobs' weights without their speed sections, each of which moved the development score by 0.000 to +0.008 points (`docs/vocr6.md`, the follow-up's gate fits) |
| `VOCR6_N_w8_2l.vocr`, `VOCR6_L_w8_2l.vocr` (and their plain `_w8`) | VOCR6-N and -L of v0.4.0-vocr6.1 | 184,400 / 262,480 | 0.79 / 0.62, 1.87 / 1.46 | dev score 1.451 / 1.442; read once: CLANNAD 0.44% / 0.36%, vita-test + psp-test 0.49% / 0.43% |
| `VOCR5_S_w8g.vocr`, `VOCR5_N_w8g2.vocr`, `VOCR5_L_w8g2.vocr` | VOCR5 with its head sections (Rubi 0.9.8) | 124,880 / 184,448 / 262,528 | 0.52, about 0.76, about 1.79 (333 MHz) | VOCR5's numbers (below); read once on the protected sets, the gate changes dialogue CER by −0.043 to +0.006 points and the two-level head by at most +0.009 more |
| `VOCR5_N_w8g.vocr`, `VOCR5_L_w8g.vocr` | VOCR5-N and -L with the head gate alone | 182,992 / 261,072 | est. 0.14 / 0.16 s less than plain (333 MHz) | as above |
| `VOCR5_S_w8.vocr`, `VOCR5_N_w8.vocr`, `VOCR5_L_w8.vocr` | VOCR5 (v0.3.0), plain | 124,832 / 182,944 / 261,024 | 0.63 / 0.50, 0.92 / 0.72, 2.08 / 1.63 | CLANNAD 1.38 / 0.96 / 0.73% greedy; vita-test + psp-test fixed lines 1.60 / 1.13 / 0.79% greedy, 1.15 / 0.72 / 0.53% with beam + 1 MB LM |
| `VOCR5_V_w8.vocr` | VOCR5-V, vertical text | 124,832 | 0.14 / 0.11 per median vertical region | test vertical VN 7.17% fixed columns, 11.29% end to end; manga 0.86% / 2.45% |
| `VOCR4_FL10_w8.vocr`, `VOCR4_H15_w8.vocr`, `VOCR4_W20_w8.vocr` | VOCR4 Small / Normal / Large | 124,832 / 182,944 / 241,056 | 125-239 / 221-403 / 314-554 ms per 640-px line (333 MHz) | vita-test + psp-test fixed lines (all lines) 2.15 / 1.83 / 1.40%; CLANNAD (all lines) 3.05 / 2.79 / 1.81% |
| `VOCR3_FL10_w8.vocr`, `VOCR3_H15_w8.vocr` | VOCR3 Small / Normal | 124,832 / 182,944 | 125-239 / 221-403 ms per 640-px line (333 MHz) | vita-test + psp-test fixed lines (all lines) 2.13 / 1.73% |
| `VOCR2_FL10_w8.vocr` | VOCR2-FL10, the overlay's opt-in trial | 124,832 | VOCR3-FL10's shape: 125-239 ms per 640-px line (333 MHz) | fixed lines (all lines): vita-test 59 / 2,564 errors (FL10 61), psp-test 119 / 2,510 (FL10 133); promotion rejected on the whole-textbox guard |
| `MODELB_B24_w8g2.vocr`, `MODELB_S32_w8g2.vocr` | model B, PSP candidates | memory in all 1,042,498 / 1,291,938 B | 1.21 / 2.05 per box on the PSP's scalar build at 333 MHz (PPSSPP-based estimates) | development PSP whole dialogue against VOCR5-S: +3.93 points [+2.63, +5.59] (B24); plain S32 +0.04 [−0.24, +0.31] |

How VOCR6 compares with the other OCR engines on the same crops is in the README ("VOCR6 against other OCRs",
`eval/other_readers_20261028.json`).

## The line finders, the gate and the detector

Dialogue CER with VOCR5-N reading the lines: through `eval/pagepath.py`'s page path with the 1 MB LM for the line
spotters, on whole frames for textspot3 and textspot, and on whole dialogue regions for the gate and the detector.
Sources: `docs/textspot8.md`, `docs/textspot7.md`, `docs/textspot6.md`, `docs/textspot3.md`, `docs/textspot.md`,
`docs/textgate.md`, `docs/textdet.md`. The arenas are at a whole 960×544 frame.

| file | model | arena | est. ms at 333 MHz | measured |
|---|---|---:|---|---|
| `textspot8_t64n8c16_w8.vtsp` | textspot8, Rubi's line finder | 244,607 | 82 per 900×150 region, at most 264 per frame | unseen dev 1.03% (0.99% with VOCR6-N → L; Rubi's finder 2.38%); read once: Dance with Devils 0.97%, CLANNAD regions 0.97%, vita-test + psp-test whole frames 1.91% |
| `textspot7_t64_w8.vtsp` | textspot7 (decode D) | 244,607 + 1,924 | 264 + at most 32 per frame | unseen dev 1.18% (screening model); read once: Dance with Devils 0.67%, CLANNAD regions 0.82% |
| `textspot6_qS4_w8.vtsp` | textspot6 qS4 | 256,247 | 199 per frame | unseen dev 1.39%; Dance with Devils 1.39% |
| `textspot3_t64n8c16_kd_w8.vtsp` | textspot3 (whole frames) | 240,991 | 268 per frame | vita-test + psp-test whole frames: dialogue 2.29%, box F1 0.776; CLANNAD 0.99% |
| `textspot_s4_c24n4_w8.vtsp` | textspot s4_c24n4 (whole frames) | 80,799 | 139 per frame | vita-test + psp-test whole frames: dialogue 3.00%, box F1 0.694; CLANNAD 1.14% |
| `textspot_s4p_c8_w8.vtsp` | textspot s4p_c8 | 72,391 | 80 per frame | met textspot's bar on development; never read on a test set |
| `textgate_p8_c8_w8.vtd` | text-presence gate | 35,529 | 67-135 per 900×150 box | development: no line on 59 of 66 empty regions (GRAD + refine: 5) at +0.06 to +0.07 points of whole-dialogue CER |
| `textdet_s8_c24_w8.vtd` | text detector | 92,639 (det + refine) | 53 per 900×150 box | development whole dialogue 3.39% (VOCR5-N; GRAD 6.87%, GRAD + refine 4.78%); vita-test + psp-test 5.99% (GRAD 10.23%) |

## The character LMs

For the CTC beam decoder (`runtime/vocr_lm.c`); sources `docs/vocr6-lm.md`, `docs/ctc-lm-decode.md`. Rubi's settings
per tier are in `LM-settings.json`.

| file | tier | measured |
|---|---|---|
| `vocr_lm6_o3_1000k.vlm` | 1 MB, the default (game text) | hooked games' held-out page path 4.12% → 3.39% against ctclm.2's 1 MB LM (VOCR5-N with its cascade); unseen-game dev level (−0.03 [−0.10, +0.04]) |
| `vocr_lm6_o3_4000k.vlm` | 4 MB (game text) | held-out page path 4.03% → 3.57%; unseen-game dev level (+0.01 [−0.05, +0.07]) |
| `vocr_lm_o3_500k.vlm`, `vocr_lm_o3_2000k.vlm` | 0.5 MB and 2 MB (ctclm.2's) | development dialogue errors, six models pooled: 1,656 greedy → 1,153 and 1,000 |
| `vocr_lm_o3_1000k.vlm`, `vocr_lm_o3_4000k.vlm` | ctclm.2's 1 MB and 4 MB, replaced by lm6 | the same: 1,048 and 969 |

## Every file

<!-- models.py:files -->
| file | what | tier | status | why | attribution | licence |
|---|---|---|---|---|---|---|
| `VOCR6.2_S_w8.vocr` | recognizer: VOCR6-S | Small | used by Rubi | Rubi's Small size (no head sections: the gate rule kept none) | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6.2_N_w8g2.vocr` | recognizer: VOCR6-N | Normal (default) | used by Rubi | Rubi's Normal size, the default (head gate at 1e-4 and two-level head) | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6.2_L_w8_2l.vocr` | recognizer: VOCR6-L | Large | used by Rubi | Rubi's Large size (two-level head) | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6.2_N_w8.vocr` | recognizer: VOCR6-N | Normal (default) | used by Rubi (host tools and tests only, not shipped) | the Normal weights without speed sections: Rubi's host tests' off-path reference | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6.2_L_w8.vocr` | recognizer: VOCR6-L | Large | used by Rubi (host tools and tests only, not shipped) | the Large weights without speed sections: Rubi's host tests' off-path reference | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_VOCR6.2_S.md` | model card: VOCR6-S | Small | documentation | VOCR6-S's model card | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_VOCR6.2_N.md` | model card: VOCR6-N | Normal (default) | documentation | VOCR6-N's model card | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_VOCR6.2_L.md` | model card: VOCR6-L | Large | documentation | VOCR6-L's model card | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `vocr6.2-assets.json` | asset list: VOCR6 | - | documentation | v0.4.1-vocr6.2's list of its five blobs | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6-ATTRIBUTION.md` | attribution: VOCR6 | - | used by Rubi | v0.4.1-vocr6.2's ATTRIBUTION.md; Rubi ships it as ocr/VOCR6-ATTRIBUTION.md | `VOCR6-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6_N_w8_2l.vocr` | recognizer: VOCR6-N (v0.4.0) | Normal (default) | previous | replaced by VOCR6.2_N_w8g2.vocr (3x the training, stage 4c); Rubi's test builds v0.9.8-test.4 and test.5 carried it | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6_L_w8_2l.vocr` | recognizer: VOCR6-L (v0.4.0) | Large | previous | replaced by VOCR6.2_L_w8_2l.vocr (stage 4c); Rubi's test builds v0.9.8-test.4 and test.5 carried it | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6_N_w8.vocr` | recognizer: VOCR6-N (v0.4.0) | Normal (default) | previous | VOCR6_N_w8_2l.vocr without its two-level head | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR6_L_w8.vocr` | recognizer: VOCR6-L (v0.4.0) | Large | previous | VOCR6_L_w8_2l.vocr without its two-level head | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_VOCR6.1_N.md` | model card: VOCR6-N (v0.4.0) | Normal (default) | documentation | v0.4.0-vocr6.1's VOCR6-N model card | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_VOCR6.1_L.md` | model card: VOCR6-L (v0.4.0) | Large | documentation | v0.4.0-vocr6.1's VOCR6-L model card | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `vocr6.1-assets.json` | asset list: VOCR6 (v0.4.0) | - | documentation | v0.4.0-vocr6.1's list of its four blobs | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `ATTRIBUTION_v0.4.0-vocr6.1.md` | attribution: VOCR6 (v0.4.0) | - | documentation | v0.4.0-vocr6.1's ATTRIBUTION.md | `ATTRIBUTION_v0.4.0-vocr6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `textspot8_t64n8c16_w8.vtsp` | line spotter: textspot8 | - | used by Rubi | Rubi's default line finder (ocr_finder = spotter; GRAD the fallback) | `SPOTTER-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_TEXTSPOT8.md` | model card: textspot8 | - | documentation | textspot8's model card | `SPOTTER-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `SPOTTER-ATTRIBUTION.md` | attribution: textspot8 | - | used by Rubi | v0.3.10-textspot8.1's ATTRIBUTION.md; Rubi ships it as ocr/SPOTTER-ATTRIBUTION.md | `SPOTTER-ATTRIBUTION.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `textspot7_t64_w8.vtsp` | line spotter: textspot7 | - | research / not selected | selected by textspot7's rule but not resolved better than Rubi's finder, so not Rubi's default; Rubi took textspot8 | `ATTRIBUTION_v0.3.9-textspot7.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_TEXTSPOT7.md` | model card: textspot7 | - | documentation | textspot7's model card | `ATTRIBUTION_v0.3.9-textspot7.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `ATTRIBUTION_v0.3.9-textspot7.1.md` | attribution: textspot7 | - | documentation | v0.3.9-textspot7.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.9-textspot7.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `textspot6_qS4_w8.vtsp` | line spotter: textspot6 qS4 | - | research / not selected | textspot6's rule selected none (no unseen-dev interval below 0); its best eligible blob, as trained | `ATTRIBUTION_v0.3.8-textspot6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODEL_CARD_TEXTSPOT6.md` | model card: textspot6 qS4 | - | documentation | textspot6's model card | `ATTRIBUTION_v0.3.8-textspot6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `ATTRIBUTION_v0.3.8-textspot6.1.md` | attribution: textspot6 qS4 | - | documentation | v0.3.8-textspot6.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.8-textspot6.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `textspot3_t64n8c16_kd_w8.vtsp` | text spotter (whole frame): textspot3 | - | previous | the pick of textspot3's rule (#108); Rubi's spotter is textspot8 | `ATTRIBUTION_v0.3.4-textspot3.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_TEXTSPOT3.md` | model card: textspot3 | - | documentation | textspot3's model card | `ATTRIBUTION_v0.3.4-textspot3.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `ATTRIBUTION_v0.3.4-textspot3.1.md` | attribution: textspot3 | - | documentation | v0.3.4-textspot3.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.4-textspot3.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `textspot_s4_c24n4_w8.vtsp` | text spotter (whole frame): textspot s4_c24n4 | - | previous | the winner of textspot's rule (#104); superseded by textspot3's pick, then by textspot8 | `ATTRIBUTION_v0.3.3-textspot.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `textspot_s4p_c8_w8.vtsp` | text spotter (whole frame): textspot s4p_c8 | - | research / not selected | the smallest candidate that met textspot's bar, not its selection; never read on a test set | `ATTRIBUTION_v0.3.3-textspot.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_TEXTSPOT.md` | model card: textspot s4_c24n4 | - | documentation | textspot's model card | `ATTRIBUTION_v0.3.3-textspot.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `ATTRIBUTION_v0.3.3-textspot.1.md` | attribution: textspot | - | documentation | v0.3.3-textspot.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.3-textspot.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `vocr_lm_o3_500k.vlm` | character LM: ctclm.2 0.5 MB | LM tier 1 | used by Rubi | Rubi's 0.5 MB LM tier (ctclm.2's blob, byte for byte) | `LM-ATTRIBUTION.md` | CC BY-SA 4.0 |
| `vocr_lm6_o3_1000k.vlm` | character LM: lm6 1 MB | LM tier 2 (default) | used by Rubi | Rubi's default 1 MB LM tier (the game-text build) | `LM-ATTRIBUTION.md` | CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures |
| `vocr_lm_o3_2000k.vlm` | character LM: ctclm.2 2 MB | LM tier 3 | used by Rubi | Rubi's 2 MB LM tier (ctclm.2's blob, byte for byte) | `LM-ATTRIBUTION.md` | CC BY-SA 4.0 |
| `vocr_lm6_o3_4000k.vlm` | character LM: lm6 4 MB | LM tier 4 | used by Rubi | Rubi's 4 MB LM tier (the game-text build) | `LM-ATTRIBUTION.md` | CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures |
| `vocr_lm_o3_500k.json` | LM build manifest: ctclm.2 0.5 MB | LM tier 1 | documentation | the 0.5 MB LM's build manifest | `LM-ATTRIBUTION.md` | CC BY-SA 4.0 |
| `vocr_lm6_o3_1000k.json` | LM build manifest: lm6 1 MB | LM tier 2 (default) | documentation | the 1 MB LM's build manifest | `LM-ATTRIBUTION.md` | CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures |
| `vocr_lm_o3_2000k.json` | LM build manifest: ctclm.2 2 MB | LM tier 3 | documentation | the 2 MB LM's build manifest | `LM-ATTRIBUTION.md` | CC BY-SA 4.0 |
| `vocr_lm6_o3_4000k.json` | LM build manifest: lm6 4 MB | LM tier 4 | documentation | the 4 MB LM's build manifest | `LM-ATTRIBUTION.md` | CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures |
| `LM-settings.json` | LM decoder settings: LMs | - | documentation | each LM tier's decoder settings (v0.3.7-lm6.2's settings.json) | `LM-ATTRIBUTION.md` | CC BY-SA 4.0 |
| `LM-ATTRIBUTION.md` | attribution: LMs | - | used by Rubi | v0.3.7-lm6.2's ATTRIBUTION.md; Rubi ships it beside the LMs | `LM-ATTRIBUTION.md` | CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures |
| `ATTRIBUTION_v0.3.7-lm6.1.md` | attribution: LMs | - | documentation | v0.3.7-lm6.1's ATTRIBUTION.md, superseded by LM-ATTRIBUTION.md (lm6.2's), which corrects it: it missed a seen title (Little Busters!) and called Collar x Malice a vita-test game | `LM-ATTRIBUTION.md` | CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures |
| `vocr_lm_o3_1000k.vlm` | character LM: ctclm.2 1 MB | LM tier 2 (default) | previous | Rubi's default 1 MB LM before lm6 (also in v0.3.0); replaced by vocr_lm6_o3_1000k.vlm | `ATTRIBUTION_v0.2.6-ctclm.2.md` | CC BY-SA 4.0 |
| `vocr_lm_o3_1000k.json` | LM build manifest: ctclm.2 1 MB | LM tier 2 (default) | documentation | ctclm.2's 1 MB LM's build manifest | `ATTRIBUTION_v0.2.6-ctclm.2.md` | CC BY-SA 4.0 |
| `vocr_lm_o3_4000k.vlm` | character LM: ctclm.2 4 MB | LM tier 4 | previous | Rubi's 4 MB LM tier before lm6; replaced by vocr_lm6_o3_4000k.vlm | `ATTRIBUTION_v0.2.6-ctclm.2.md` | CC BY-SA 4.0 |
| `vocr_lm_o3_4000k.json` | LM build manifest: ctclm.2 4 MB | LM tier 4 | documentation | ctclm.2's 4 MB LM's build manifest | `ATTRIBUTION_v0.2.6-ctclm.2.md` | CC BY-SA 4.0 |
| `ATTRIBUTION_v0.2.6-ctclm.2.md` | attribution: ctclm.2 | - | documentation | v0.2.6-ctclm.2's ATTRIBUTION.md | `ATTRIBUTION_v0.2.6-ctclm.2.md` | CC BY-SA 4.0 |
| `VOCR5_V_w8.vocr` | recognizer (vertical): VOCR5-V | - | used by Rubi | Rubi's vertical-text model (ocr_vertical) | `VERTICAL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_VOCR5_V.md` | model card: VOCR5-V | - | documentation | VOCR5-V's model card | `VERTICAL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VERTICAL-ATTRIBUTION.md` | attribution: VOCR5-V | - | used by Rubi | v0.3.2-vertical.1's ATTRIBUTION.md; Rubi ships it as ocr/VERTICAL-ATTRIBUTION.md | `VERTICAL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `textgate_p8_c8_w8.vtd` | text-presence gate: textgate p8_c8 | - | used by Rubi | Rubi's text-presence gate (ocr_text_gate, off by default) | `GATE-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_TEXTGATE_P8C8.md` | model card: textgate p8_c8 | - | documentation | the gate's model card | `GATE-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `GATE-ATTRIBUTION.md` | attribution: textgate p8_c8 | - | used by Rubi | v0.3.2-textgate.1's ATTRIBUTION.md; Rubi ships it as ocr/GATE-ATTRIBUTION.md | `GATE-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `textdet_s8_c24_w8.vtd` | text detector: textdet s8_c24 | - | research / not selected | not promoted: not resolved better than GRAD + refine on development, worse on CLANNAD as scored | `ATTRIBUTION_v0.3.1-textdet.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_TEXTDET_S8C24.md` | model card: textdet s8_c24 | - | documentation | the detector's model card | `ATTRIBUTION_v0.3.1-textdet.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `ATTRIBUTION_v0.3.1-textdet.1.md` | attribution: textdet s8_c24 | - | documentation | v0.3.1-textdet.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.1-textdet.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODELB_B24_w8g2.vocr` | recognizer (PSP candidate): model B B24 | - | research / not selected | a console-test candidate only: +3.93 points worse than VOCR5-S on development PSP games (#126) | `ATTRIBUTION_v0.3.7-modelb-test.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `MODELB_S32_w8g2.vocr` | recognizer (PSP candidate): model B S32 | - | research / not selected | a console-test candidate only: ties VOCR5-S but est. 2.05 s per box on the PSP, over the 1 s budget (#126) | `ATTRIBUTION_v0.3.7-modelb-test.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `ATTRIBUTION_v0.3.7-modelb-test.1.md` | attribution: model B | - | documentation | v0.3.7-modelb-test.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.7-modelb-test.1.md` | non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts) |
| `VOCR5_S_w8g.vocr` | recognizer: VOCR5-S | Small | previous | Rubi 0.9.8's Small (head gate); replaced by VOCR6.2_S_w8.vocr | `MODEL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_N_w8g2.vocr` | recognizer: VOCR5-N | Normal (default) | previous | Rubi 0.9.8's Normal (head gate and two-level head); replaced by VOCR6.2_N_w8g2.vocr | `MODEL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_L_w8g2.vocr` | recognizer: VOCR5-L | Large | previous | Rubi 0.9.8's Large (head gate and two-level head); replaced by VOCR6.2_L_w8_2l.vocr | `MODEL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL-ATTRIBUTION.md` | attribution: VOCR5 | - | used by Rubi | v0.3.6-head2level.1's ATTRIBUTION.md (VOCR5's training data, which every VOCR6 tier starts from); Rubi ships it as ocr/MODEL-ATTRIBUTION.md | `MODEL-ATTRIBUTION.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_N_w8g.vocr` | recognizer: VOCR5-N | Normal (default) | previous | VOCR5-N with the head gate alone; superseded by VOCR5_N_w8g2.vocr | `ATTRIBUTION_v0.3.5-headgate.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_L_w8g.vocr` | recognizer: VOCR5-L | Large | previous | VOCR5-L with the head gate alone; superseded by VOCR5_L_w8g2.vocr | `ATTRIBUTION_v0.3.5-headgate.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `ATTRIBUTION_v0.3.5-headgate.1.md` | attribution: VOCR5 (head gate) | - | documentation | v0.3.5-headgate.1's ATTRIBUTION.md | `ATTRIBUTION_v0.3.5-headgate.1.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_S_w8.vocr` | recognizer: VOCR5-S | Small | previous | v0.3.0's Small, plain; replaced by VOCR6-S | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_N_w8.vocr` | recognizer: VOCR5-N | Normal (default) | previous | v0.3.0's Normal, plain; replaced by VOCR6-N | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR5_L_w8.vocr` | recognizer: VOCR5-L | Large | previous | v0.3.0's Large, plain; replaced by VOCR6-L | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_VOCR5_S.md` | model card: VOCR5-S | Small | documentation | VOCR5-S's model card | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_VOCR5_N.md` | model card: VOCR5-N | Normal (default) | documentation | VOCR5-N's model card | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `MODEL_CARD_VOCR5_L.md` | model card: VOCR5-L | Large | documentation | VOCR5-L's model card | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `vocr5-assets.json` | asset list: VOCR5 | - | documentation | v0.3.0's list of its assets | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `ATTRIBUTION_v0.3.0.md` | attribution: VOCR5 | - | documentation | v0.3.0's ATTRIBUTION.md | `ATTRIBUTION_v0.3.0.md` | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR4_FL10_w8.vocr` | recognizer: VOCR4-FL10 | Small | research / not selected | not qualified: fails VOCR2's whole-dialogue guard (docs/vocr4.md) | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR4_H15_w8.vocr` | recognizer: VOCR4-H15 | Normal (default) | research / not selected | not qualified: fails VOCR2's whole-dialogue guard and the fixed-line hiragana condition (docs/vocr4.md) | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR4_W20_w8.vocr` | recognizer: VOCR4-W20 | Large | used by Rubi (host tools and tests only, not shipped) | Rubi's Large before VOCR5; now Rubi's host tools' cascade model of a request by name | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `vocr4-assets.json` | asset list: VOCR4 | - | documentation | v0.2.5-vocr4.1's list of its blobs | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR3_FL10_w8.vocr` | recognizer: VOCR3-FL10 | Small | used by Rubi (host tools and tests only, not shipped) | Rubi's Small before VOCR5; now pinned by Rubi's host tools | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR3_H15_w8.vocr` | recognizer: VOCR3-H15 | Normal (default) | used by Rubi (host tools and tests only, not shipped) | Rubi's Normal before VOCR5 (the model the console timings are calibrated to); now pinned by Rubi's host tools | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `vocr3-assets.json` | asset list: VOCR3 | - | documentation | v0.2.5-vocr3.1's list of its blobs | none in its release (see Licences) | non-commercial use only (non-commercial fonts and texts in its synthetic training data) |
| `VOCR2_FL10_w8.vocr` | recognizer: VOCR2-FL10 | - | research / not selected | promotion rejected: it failed the strict whole-textbox guard (19 of 372 title checks regressed); the overlay's opt-in VOCR2 trial (v0.7.12-vocr2.1, removed after v0.8.0-test.1) | none in its release (see Licences) | no attribution file: its release notes credit NVIDIA's OCR-Synthetic-Multilingual-v1 (CC BY 4.0, through v0.2.0's FL10); the README lists the rest of its training data |
| `vocr2-qualification.md` | qualification notes: VOCR2-FL10 | - | documentation | VOCR2's qualification notes | none in its release (see Licences) | no attribution file: its release notes credit NVIDIA's OCR-Synthetic-Multilingual-v1 (CC BY 4.0, through v0.2.0's FL10); the README lists the rest of its training data |
<!-- /models.py:files -->

## Licences

- **Non-commercial use only:** every recognizer since VOCR3, every spotter, the gate and the detector.
  - VOCR6, textspot6/7/8 and model B were trained on the owner's hooked PS Vita and PSP captures, copyrighted game
    data used for training only and never redistributed.
  - Every model since VOCR3 was trained on synthetic renders whose `personal` profile holds fonts and texts free for
    non-commercial use only (README, "What the released weights were trained on").
  - Each file's attribution file is in the table above. v0.2.5-vocr3.1 and v0.2.5-vocr4.1 published no attribution
    file: their release notes credit NVIDIA's OCR-Synthetic-Multilingual-v1 (CC BY 4.0), and the README lists the
    rest of their training data.
- **VOCR2** (v0.2.3-vocr2.1, before the non-commercial synthetic profile) published no attribution file either: its
  release notes credit NVIDIA's OCR-Synthetic-Multilingual-v1 (CC BY 4.0, through v0.2.0's FL10), and the README
  lists the fl10 research's renders.
- **The LMs: CC BY-SA 4.0** (`LM-ATTRIBUTION.md`, `ATTRIBUTION_v0.2.6-ctclm.2.md`). The 1 MB and 4 MB game-text tiers
  also carry character 3-gram counts of the owner's hooked captures; the games' text belongs to its publishers.
- **The test-set disclosures** (seen titles, CLANNAD's glyph raster in VOCR6's training pages, the LMs' word lists)
  are in VOCR6's and the LMs' attribution files and in `docs/vocr6.md`; the spotters' attribution files name the
  training titles that share a release with CLANNAD or a vita-test game.

## Left out

| release | why |
|---|---|
| v0.3.7-lm6.1 | superseded by v0.3.7-lm6.2: the same four LM blobs and build manifests byte for byte (here from lm6.2); its `settings.json` differs only in its release name and note. Its `ATTRIBUTION.md` is here as `ATTRIBUTION_v0.3.7-lm6.1.md`. |
| v0.2.6-ctclm.1 | superseded by v0.2.6-ctclm.2 the same day: built before the protected-title name filter |
| v0.3.7-modelb-test.1's `vocr6_psp_console_test.zip` | the PSP console test apps (9 EBOOTs with generated test lines), not a model: its two blobs are here, and the zip stays on its release for issue #145's console test |
| every source release's `SHA256SUMS` | replaced by this release's `SHA256SUMS` |
| v0.2.2-fl10-training.1, v0.2.2-fl10-bitmap.1, v0.2.2-fl10-general.1, v0.2.4-fl10-head.1 | the FL10 research snapshots before VOCR3, each rejected or research only: archives of training checkpoints, code and results of 54,417,095 to 363,004,078 B each, 639,150,581 B together, so not trivially small; they stay on their own pre-releases |
| v0.2.3-vocr2.1's checkpoint and result files | VOCR2's training checkpoint (2,426,644 B) and its result and identity files stay on its pre-release; its int8 blob and qualification notes are here |
| v0.2.2-meiki.1 | MeikiOCR's own recognizer converted to MNN for the overlay's Meiki trial: a third-party model (LGPL-3.0 weights), not one of this repository's, which keeps its licence files on its release |
| v0.1.0-baseline, v0.2.0, v0.2.1 | the releases before VOCR2 (v1; FL10, H15 and F20): normal releases already, which stay where they are; Rubi's host tools pin v0.2.0's int8 blobs from there |

## Checksums

The same as the release's `SHA256SUMS`, with each file's source.

<!-- models.py:checksums -->
| file | from | bytes | sha256 |
|---|---|---:|---|
| `VOCR6.2_S_w8.vocr` | v0.4.1-vocr6.2 | 819,792 | `864bbd196d5bc5d3a7bd7918cf7274b3503dfe65d440017ef0781ea7e7ba37d2` |
| `VOCR6.2_N_w8g2.vocr` | v0.4.1-vocr6.2 | 1,189,408 | `107cf084cc0c1a3a61ce4c3f9b9cf6b31d6dff80f0a1bdb94afd3f7ee9ae3bc9` |
| `VOCR6.2_L_w8_2l.vocr` | v0.4.1-vocr6.2 | 2,475,680 | `65310013e8f990886e4e8097b2fe55eb98cdf082d7193eea510601fc0ac26c82` |
| `VOCR6.2_N_w8.vocr` | v0.4.1-vocr6.2 | 1,185,632 | `c8ce844ed0a637eb5d263dff7837d846bd2a520278da3b74c49c5ee33eaa24c9` |
| `VOCR6.2_L_w8.vocr` | v0.4.1-vocr6.2 | 2,473,696 | `be5a8e4f3280b58edc8398616ce5068f9803551407fbc9a843c0a2963584b7b2` |
| `MODEL_CARD_VOCR6.2_S.md` | v0.4.1-vocr6.2 (`MODEL_CARD_VOCR6_S.md`) | 6,330 | `270eb261a08636b2f18fbb2d63734860174055558bbe9727cd4cd4614ccd62bd` |
| `MODEL_CARD_VOCR6.2_N.md` | v0.4.1-vocr6.2 (`MODEL_CARD_VOCR6_N.md`) | 5,806 | `8c1b11f43ba7ec2353543d5018afffb8d3c4bcbaeb2fc281d0881819794c157e` |
| `MODEL_CARD_VOCR6.2_L.md` | v0.4.1-vocr6.2 (`MODEL_CARD_VOCR6_L.md`) | 5,792 | `66d13ac294a3127e5d033a125beeb3ad7f29299ad92792d5c97eb7fc22ebae54` |
| `vocr6.2-assets.json` | v0.4.1-vocr6.2 (`vocr6-assets.json`) | 4,819 | `76b73fe11baf1467e8f128c64bb4b76aa2fc9e3a0ff969f2ea08fbd84f020180` |
| `VOCR6-ATTRIBUTION.md` | v0.4.1-vocr6.2 (`ATTRIBUTION.md`) | 4,442 | `9a5ff2aee4cb8194f41e54f3a2dac88ce3de6ff4aa7b8ddd123dd5b792c9843b` |
| `VOCR6_N_w8_2l.vocr` | v0.4.0-vocr6.1 | 1,187,616 | `929652e33da10fee3b7bd3991a557e8bbf1a16fba83eb5775bbd0298ab65c4d6` |
| `VOCR6_L_w8_2l.vocr` | v0.4.0-vocr6.1 | 2,475,680 | `b051fd4f3abe6451927d74b1d1209f44a3072ab755ee9e1b9363368f00feb45b` |
| `VOCR6_N_w8.vocr` | v0.4.0-vocr6.1 | 1,185,632 | `11a306c3f66020ef2ea3aa20cac67f4eb8f5e45a2f59ca7811eda50a17c80dcd` |
| `VOCR6_L_w8.vocr` | v0.4.0-vocr6.1 | 2,473,696 | `d32744ca9de9413cce6b5728a84319aa7b04040022bab3a4cf9e451544c031e2` |
| `MODEL_CARD_VOCR6.1_N.md` | v0.4.0-vocr6.1 (`MODEL_CARD_VOCR6_N.md`) | 6,113 | `b20d085ec4033b3e96bfb8c1272d71954e7639b155786c4b68f1a845da2bd86b` |
| `MODEL_CARD_VOCR6.1_L.md` | v0.4.0-vocr6.1 (`MODEL_CARD_VOCR6_L.md`) | 5,994 | `7d44e1c7f2a0a5fd1f79c7167a3862ab63a97b9412bed900fe3dc13bab854f4e` |
| `vocr6.1-assets.json` | v0.4.0-vocr6.1 (`vocr6-assets.json`) | 4,049 | `a981d5640d61eaa438ecfce1fe655221c957e09acad6a6ef49d86c1b5360d6ec` |
| `ATTRIBUTION_v0.4.0-vocr6.1.md` | v0.4.0-vocr6.1 (`ATTRIBUTION.md`) | 4,222 | `44459773ecb1ec7da2e8325c6885221b122f771ccc34ca89d3a578e2b4c02a96` |
| `textspot8_t64n8c16_w8.vtsp` | v0.3.10-textspot8.1 | 52,176 | `e129d41adfa02c658a7d13d6f98064e6ab6ce9a2bc8c7a56da700a21357b5e1c` |
| `MODEL_CARD_TEXTSPOT8.md` | v0.3.10-textspot8.1 | 5,100 | `bcbdebf0437b86a476d9f2b0f8a7f5986274c790969497f1b2950b73ef72354a` |
| `SPOTTER-ATTRIBUTION.md` | v0.3.10-textspot8.1 (`ATTRIBUTION.md`) | 4,183 | `fd44f500114ca2c8373e02bda1134753619e76de869ddf8ec86c14e008be5b84` |
| `textspot7_t64_w8.vtsp` | v0.3.9-textspot7.1 | 52,176 | `78330f96c9e02646b2108b896cc035544e70fbb7d9fd896981bcf2f9ab365a3e` |
| `MODEL_CARD_TEXTSPOT7.md` | v0.3.9-textspot7.1 | 5,970 | `aaf3a102c78952d3bcff93e8e78404d30923129c469c1b67c0967530a94a871a` |
| `ATTRIBUTION_v0.3.9-textspot7.1.md` | v0.3.9-textspot7.1 (`ATTRIBUTION.md`) | 4,079 | `de89a125a6e69863a9a578b3c56b0610db4f88ad22c11c4cf03e38128944ec00` |
| `textspot6_qS4_w8.vtsp` | v0.3.8-textspot6.1 | 118,096 | `d4626b49dfb526c7595c1754a35b93f53a9d73833dfcdaf7176a8f8a6faf0aba` |
| `MODEL_CARD_TEXTSPOT6.md` | v0.3.8-textspot6.1 | 4,691 | `61f6e97909d6c364355328cee49218d560b1b8639269602e4fc9d92cf5865667` |
| `ATTRIBUTION_v0.3.8-textspot6.1.md` | v0.3.8-textspot6.1 (`ATTRIBUTION.md`) | 4,075 | `8fda8dc95f62a12060d3af4514123675f532f4a29795eb666dd7c69e2ef4fad0` |
| `textspot3_t64n8c16_kd_w8.vtsp` | v0.3.4-textspot3.1 | 51,008 | `20c3f7cc4f07b443741bbb45fe4270f6fa8c74ce29045a796a5ed8887a3fb26a` |
| `MODEL_CARD_TEXTSPOT3.md` | v0.3.4-textspot3.1 | 3,061 | `6461af7401d6ef2f8b03600477ddc5d11a4010dbe0da0ce75bf852bf3d40571f` |
| `ATTRIBUTION_v0.3.4-textspot3.1.md` | v0.3.4-textspot3.1 (`ATTRIBUTION.md`) | 2,952 | `cd5970887179ffbf1521f3657c5872e1b7e026a2dc35eba9701822734961d74a` |
| `textspot_s4_c24n4_w8.vtsp` | v0.3.3-textspot.1 | 5,732 | `852506919ba94e54a4efc55e3695f365626d9d4ce6347f420cbf2afe3e8360cf` |
| `textspot_s4p_c8_w8.vtsp` | v0.3.3-textspot.1 | 2,732 | `d4dd598f6ddb1575ebc1a957cb1dae29f03aa41f7533bbb7d5fe25210161b185` |
| `MODEL_CARD_TEXTSPOT.md` | v0.3.3-textspot.1 | 3,120 | `82efbfd806115f0a564215dcc7c73114feb756348118e85698dc23baedff9167` |
| `ATTRIBUTION_v0.3.3-textspot.1.md` | v0.3.3-textspot.1 (`ATTRIBUTION.md`) | 3,345 | `0226b181342d2ff93b836b8e0a8a01f7d5736b75cb93bda11a3a0bbe59fae5f7` |
| `vocr_lm_o3_500k.vlm` | v0.3.7-lm6.2 | 499,824 | `38ea94e0c4ec93863454387e7c2c469d209aaf21f2f6ee5d040e75a09d00d286` |
| `vocr_lm6_o3_1000k.vlm` | v0.3.7-lm6.2 | 982,320 | `41591accf8499f1c34557e585cca382de2c4b5ca15f270c985612c32487e251f` |
| `vocr_lm_o3_2000k.vlm` | v0.3.7-lm6.2 | 1,971,568 | `ac33e461bd346617aae27168aee47230a758f4b301d78b59eb55ca25a21bcb34` |
| `vocr_lm6_o3_4000k.vlm` | v0.3.7-lm6.2 | 3,603,552 | `a65841b5bdce3e3123b2d1e712deb857fb577cbd2fd8667e8fb06ab35d368335` |
| `vocr_lm_o3_500k.json` | v0.3.7-lm6.2 | 5,419 | `3273c93b31735f50b2222be8e1e410277d28117a5d170395d92715004ec15f48` |
| `vocr_lm6_o3_1000k.json` | v0.3.7-lm6.2 | 6,264 | `5b508ce67f32fdb7bd1ca653f900cd421f412109f0b2272b0b55d1ae39c46dc3` |
| `vocr_lm_o3_2000k.json` | v0.3.7-lm6.2 | 5,427 | `0b2bfc08a23d6d3236b506f2c98ccbd73ecd098bf3dd704ea09d0ab35ea8b4e3` |
| `vocr_lm6_o3_4000k.json` | v0.3.7-lm6.2 | 6,444 | `9cccf26ab5f0507f64d9f85a2591e4572402c7e1028db51f31c8e7ee1e8d859e` |
| `LM-settings.json` | v0.3.7-lm6.2 (`settings.json`) | 2,318 | `1fd365b16850c3c134b613bc7e9000b97c5c6706a2380736fccc46ad0b628faf` |
| `LM-ATTRIBUTION.md` | v0.3.7-lm6.2 (`ATTRIBUTION.md`) | 6,514 | `9418ed87e6d9fa196aba2315e05a0cf0dca7056482cce03577ced423e5feb1d3` |
| `ATTRIBUTION_v0.3.7-lm6.1.md` | v0.3.7-lm6.1 (`ATTRIBUTION.md`) | 3,641 | `c612199c55374b04cadfb43efa31f8b286f0391aa0d0cd69d5545b663551dc4d` |
| `vocr_lm_o3_1000k.vlm` | v0.2.6-ctclm.2 | 987,600 | `993000611555171daf6361e29a503b0871a5f3dd240a19555d55f16aaa3ab3d9` |
| `vocr_lm_o3_1000k.json` | v0.2.6-ctclm.2 | 5,417 | `eaf17a80c9987417d0a6b170a6ebae0fdaf6bd951591c120f2101b47f697b8fd` |
| `vocr_lm_o3_4000k.vlm` | v0.2.6-ctclm.2 | 3,781,072 | `01f49be1691a1aab5a6d33414ce7d17c3e5b2daf97dfccb0d341a380a68e5898` |
| `vocr_lm_o3_4000k.json` | v0.2.6-ctclm.2 | 5,420 | `97ff19daae9e4b88c988bf17207ec5d43b865ad0f63d23c28cc772dec49d6633` |
| `ATTRIBUTION_v0.2.6-ctclm.2.md` | v0.2.6-ctclm.2 (`ATTRIBUTION.md`) | 2,538 | `1de91ab9c181e204c53ae7c2796af47366026901fc8b2c268cbcef820ebaf19c` |
| `VOCR5_V_w8.vocr` | v0.3.2-vertical.1 | 819,792 | `ccd20c63bb2620f054538b456904bdf19b74d483d3da8de865009e282bb04172` |
| `MODEL_CARD_VOCR5_V.md` | v0.3.2-vertical.1 | 4,596 | `341a9c2bdb01acda1868ebffae272ab89d83c5f52f756a43f5cc2d5f128f6972` |
| `VERTICAL-ATTRIBUTION.md` | v0.3.2-vertical.1 (`ATTRIBUTION.md`) | 3,323 | `0f644c6dce8c24abb9ac56ecbb684ab7cd1f1950cad0aae15eafadba04bcbcd9` |
| `textgate_p8_c8_w8.vtd` | v0.3.2-textgate.1 | 1,540 | `a4e73ad78451c75a19f88a479fc0108daac85beaa88d875b73212ed3c09d85d1` |
| `MODEL_CARD_TEXTGATE_P8C8.md` | v0.3.2-textgate.1 | 2,799 | `ade0651cd30307f0fd90c47f684edfb249d26e556fa5b979612307a640547438` |
| `GATE-ATTRIBUTION.md` | v0.3.2-textgate.1 (`ATTRIBUTION.md`) | 3,706 | `44a353c76e19c6bdabcb9dbd06d14867696dcb59ce0460db22d09749af687f6e` |
| `textdet_s8_c24_w8.vtd` | v0.3.1-textdet.1 | 5,700 | `8a13dbddbc7d6305ccfaf8164ad1c4d9030437fde76eeedca68f24ac5cc47179` |
| `MODEL_CARD_TEXTDET_S8C24.md` | v0.3.1-textdet.1 | 2,499 | `728b52da87e72871158e05df184d3fa7f1cbc839e4bc578b6e82d2ae3365929a` |
| `ATTRIBUTION_v0.3.1-textdet.1.md` | v0.3.1-textdet.1 (`ATTRIBUTION.md`) | 2,713 | `f317b185df1539b0aa0135cbf8fd787d7d324197ba329189ff04f7aae8d58aac` |
| `MODELB_B24_w8g2.vocr` | v0.3.7-modelb-test.1 | 760,208 | `84a6c0908ef1ca448626dde1a80ce71c47e38a80326cd4d644d6d4034b0047bb` |
| `MODELB_S32_w8g2.vocr` | v0.3.7-modelb-test.1 | 823,568 | `975d6b90178c9789f3e5286345714c1a9d8ee71f801e4b48a1206f74aee96198` |
| `ATTRIBUTION_v0.3.7-modelb-test.1.md` | v0.3.7-modelb-test.1 (`ATTRIBUTION.md`) | 5,125 | `7f68933917aa81720498020efc343311aacf4dfc8721cdc5b49ff3a54228bee5` |
| `VOCR5_S_w8g.vocr` | v0.3.6-head2level.1 | 821,584 | `76a1bc5de5cd0419a7edad31bc198af8016bda5592a7e595de75697671d3e928` |
| `VOCR5_N_w8g2.vocr` | v0.3.6-head2level.1 | 1,189,408 | `f0c8e52c9c1d9f2a714d5ca3ecfba9c0b031ee7ac51a522a2c29fd558d51761b` |
| `VOCR5_L_w8g2.vocr` | v0.3.6-head2level.1 | 2,477,472 | `41c3543ff8dddd45dc953a6a5f5d579ef9ca779f0e43196406be59c1d985c5b5` |
| `MODEL-ATTRIBUTION.md` | v0.3.6-head2level.1 (`ATTRIBUTION.md`) | 5,536 | `74b15e8a8ffb01a997ff15bff3ebc137c48a51439732fd838300a9ddff4953c3` |
| `VOCR5_N_w8g.vocr` | v0.3.5-headgate.1 | 1,187,424 | `18ddf93f7065b1cb9d00bd46afe389a644be15586c7942fb146b9baf139ad510` |
| `VOCR5_L_w8g.vocr` | v0.3.5-headgate.1 | 2,475,488 | `ee76e93512d02de3a4f9afda82ddf8d50b9ffdeef4bc15691e21164f0c7a9700` |
| `ATTRIBUTION_v0.3.5-headgate.1.md` | v0.3.5-headgate.1 (`ATTRIBUTION.md`) | 3,648 | `971f0422be26ecead1a7e672e15e516e5315c97a707a96db46a38ecb28f0e05f` |
| `VOCR5_S_w8.vocr` | v0.3.0 | 819,792 | `f175bfcb8cec65d69e85dd5f2cf707c9a711ada7203c0d1099cd2d0142978b8d` |
| `VOCR5_N_w8.vocr` | v0.3.0 | 1,185,632 | `20be179def8033171435ba62388657b3d3c932b365093737948c636e025e3bdc` |
| `VOCR5_L_w8.vocr` | v0.3.0 | 2,473,696 | `50413b37dbd85b2acb0908967bdeaf7ca67803850e0627a7b7cc3d274edc0ba3` |
| `MODEL_CARD_VOCR5_S.md` | v0.3.0 | 11,413 | `07765286b60a431ddac8e4ee20e74a236de1646ab24645393d86f8541fddb463` |
| `MODEL_CARD_VOCR5_N.md` | v0.3.0 | 11,372 | `f054a28d27d1617b06b26b933b9669af7c966201bc5910effc2d15a65c62683f` |
| `MODEL_CARD_VOCR5_L.md` | v0.3.0 | 11,453 | `efdd51dcfbfd2e6ce53977f4bb53309c08b935235da8aa86a7423e7f07b68925` |
| `vocr5-assets.json` | v0.3.0 | 4,064 | `3643476af50cd3a66c37019b807e9f143dbb481bc43ffc4ecce78e349a010dbe` |
| `ATTRIBUTION_v0.3.0.md` | v0.3.0 (`ATTRIBUTION.md`) | 3,158 | `db307ca08f4282a235173a77d0ef0f9c4c918dc43ac17fa289d27b831bf1c194` |
| `VOCR4_FL10_w8.vocr` | v0.2.5-vocr4.1 | 657,664 | `d7c4c08a34ab5349cab30a81ac87c3dd436dd2ea7b69d022537bd78337c181a2` |
| `VOCR4_H15_w8.vocr` | v0.2.5-vocr4.1 | 1,155,968 | `d380d1a4cf4c3c76e4e4b0f9a73fd1619cc1068c4f070d0da07e5e4d05254b9b` |
| `VOCR4_W20_w8.vocr` | v0.2.5-vocr4.1 | 1,558,016 | `2745e69699ad7b2aba3092f83e5ddc1912f405f54b9df8a2e55be82c385b716e` |
| `vocr4-assets.json` | v0.2.5-vocr4.1 | 1,439 | `9a385e3a9ddba0c8e3900744ffbf596eaed5140e84eb98501dcebf171373d2b0` |
| `VOCR3_FL10_w8.vocr` | v0.2.5-vocr3.1 | 657,664 | `87fb4d0c18416f5f3771cecd274c8db3cf69b8b4a4ee9f204edebb4b82079154` |
| `VOCR3_H15_w8.vocr` | v0.2.5-vocr3.1 | 1,155,968 | `cb3c61a8f91f9aeeb3efcc0e79313f664112f25c4bd55865bae9df52522b360c` |
| `vocr3-assets.json` | v0.2.5-vocr3.1 | 1,205 | `e5b829d8bc56237280117398b145bdf00eb50e93868a544073d5b5059bb9008d` |
| `VOCR2_FL10_w8.vocr` | v0.2.3-vocr2.1 | 657,664 | `d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766` |
| `vocr2-qualification.md` | v0.2.3-vocr2.1 | 9,877 | `f7dda3316dfa969c4096da4605b7092864dc1e848672aff1cdfd3e706cd04e32` |
<!-- /models.py:checksums -->

## Reproduce

```sh
gh release download TAG -R bee-san/vita-vn-ocr -D SRC/TAG     # each source release (docs/models.json "from")
python docs/models.py manifest --src SRC                       # docs/models.json: bytes, sha256, Rubi's pins checked
python docs/models.py md                                       # this file's two generated tables
python docs/models.py stage --out DIR --src SRC                # the release directory and its SHA256SUMS
gh release download v0.5.0 -R bee-san/vita-vn-ocr -D CHECK && python docs/models.py verify --dir CHECK
```
