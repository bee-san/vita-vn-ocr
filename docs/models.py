#!/usr/bin/env python3
"""Release v0.5.0's assets: every model from VOCR3 on and VOCR2's trial blob, collected from the releases that published
them (docs/models.md).

  python docs/models.py manifest [--src DIR]          # docs/models.json from the source releases' files
  python docs/models.py md                            # docs/models.md's generated tables from docs/models.json
  python docs/models.py stage --out DIR [--src DIR]   # the release directory: every asset + SHA256SUMS
  python docs/models.py verify --dir DIR              # every file of DIR against SHA256SUMS and docs/models.json

ASSETS below is the release's table: each file's name in v0.5.0, the release and name it comes from, what it is, its
status and its attribution file. `manifest` reads each source file (SRC/<tag>/<name>, default the downloads under
$HERD/shared/artifacts/vvo-release/assets, fetched with `gh release download`), checks it against its source
release's SHA256SUMS when that lists it, and writes docs/models.json with the bytes and sha256 of every asset and
Rubi's pins (bee-san/Rubi_for_vita main dc272d1: tools/prepare_ocr.py, include/vjo_vocr_sizes.h; byte for byte the
pins of Rubi 0.10.0, tag efbee95). A file keeps its release name unless two source releases hold different bytes under
one name or the name says nothing on its own (ATTRIBUTION.md, settings.json, the model cards and asset lists of v0.4.0
and v0.4.1): then it is renamed, and docs/models.json records the original. The source releases' own SHA256SUMS are
not copied: SHA256SUMS lists every other file of v0.5.0. Nothing here retrains, re-exports or changes a byte of any
model.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "models.json"
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
SRC = HERD / "shared" / "artifacts" / "vvo-release" / "assets"
RELEASE = "v0.5.0"
LIST_NAME = "v0.5.0-assets.json"            # docs/models.json, published beside the assets

NC = "non-commercial use only"
NC_GAME = "non-commercial use only (trained on the owner's hooked game captures and non-commercial fonts and texts)"
NC_SYNTH = "non-commercial use only (non-commercial fonts and texts in its synthetic training data)"
BYSA = "CC BY-SA 4.0"
BYSA_GAME = "CC BY-SA 4.0; carries character 3-gram counts of the owner's hooked game captures"

VOCR2_LIC = ("no attribution file: its release notes credit NVIDIA's OCR-Synthetic-Multilingual-v1 (CC BY 4.0, "
             "through v0.2.0's FL10); the README lists the rest of its training data")

RUBI = "used by Rubi"
RUBI_TOOLS = "used by Rubi (host tools and tests only, not shipped)"
PREVIOUS = "previous"
RESEARCH = "research / not selected"
DOC = "documentation"


def a(name, tag, kind, model, tier, status, why, attribution, licence, src=None):
    return {"name": name, "from": {"tag": tag, "name": src or name}, "kind": kind, "model": model, "tier": tier,
            "status": status, "why": why, "attribution": attribution, "licence": licence}


T62, T61, TS8 = "v0.4.1-vocr6.2", "v0.4.0-vocr6.1", "v0.3.10-textspot8.1"
ASSETS = [
    # ------------------------------------------------------------------ VOCR6, the follow-up (Rubi's three sizes)
    a("VOCR6.2_S_w8.vocr", T62, "recognizer", "VOCR6-S", "Small", RUBI,
      "Rubi's Small size (no head sections: the gate rule kept none)", "VOCR6-ATTRIBUTION.md", NC_GAME),
    a("VOCR6.2_N_w8g2.vocr", T62, "recognizer", "VOCR6-N", "Normal (default)", RUBI,
      "Rubi's Normal size, the default (head gate at 1e-4 and two-level head)", "VOCR6-ATTRIBUTION.md", NC_GAME),
    a("VOCR6.2_L_w8_2l.vocr", T62, "recognizer", "VOCR6-L", "Large", RUBI,
      "Rubi's Large size (two-level head)", "VOCR6-ATTRIBUTION.md", NC_GAME),
    a("VOCR6.2_N_w8.vocr", T62, "recognizer", "VOCR6-N", "Normal (default)", RUBI_TOOLS,
      "the Normal weights without speed sections: Rubi's host tests' off-path reference", "VOCR6-ATTRIBUTION.md",
      NC_GAME),
    a("VOCR6.2_L_w8.vocr", T62, "recognizer", "VOCR6-L", "Large", RUBI_TOOLS,
      "the Large weights without speed sections: Rubi's host tests' off-path reference", "VOCR6-ATTRIBUTION.md",
      NC_GAME),
    a("MODEL_CARD_VOCR6.2_S.md", T62, "model card", "VOCR6-S", "Small", DOC, "VOCR6-S's model card",
      "VOCR6-ATTRIBUTION.md", NC_GAME, src="MODEL_CARD_VOCR6_S.md"),
    a("MODEL_CARD_VOCR6.2_N.md", T62, "model card", "VOCR6-N", "Normal (default)", DOC, "VOCR6-N's model card",
      "VOCR6-ATTRIBUTION.md", NC_GAME, src="MODEL_CARD_VOCR6_N.md"),
    a("MODEL_CARD_VOCR6.2_L.md", T62, "model card", "VOCR6-L", "Large", DOC, "VOCR6-L's model card",
      "VOCR6-ATTRIBUTION.md", NC_GAME, src="MODEL_CARD_VOCR6_L.md"),
    a("vocr6.2-assets.json", T62, "asset list", "VOCR6", "-", DOC, "v0.4.1-vocr6.2's list of its five blobs",
      "VOCR6-ATTRIBUTION.md", NC_GAME, src="vocr6-assets.json"),
    a("VOCR6-ATTRIBUTION.md", T62, "attribution", "VOCR6", "-", RUBI,
      "v0.4.1-vocr6.2's ATTRIBUTION.md; Rubi ships it as ocr/VOCR6-ATTRIBUTION.md", "VOCR6-ATTRIBUTION.md", NC_GAME,
      src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ VOCR6, the first set
    a("VOCR6_N_w8_2l.vocr", T61, "recognizer", "VOCR6-N (v0.4.0)", "Normal (default)", PREVIOUS,
      "replaced by VOCR6.2_N_w8g2.vocr (3x the training, stage 4c); Rubi's test builds v0.9.8-test.4 and test.5 "
      "carried it",
      "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME),
    a("VOCR6_L_w8_2l.vocr", T61, "recognizer", "VOCR6-L (v0.4.0)", "Large", PREVIOUS,
      "replaced by VOCR6.2_L_w8_2l.vocr (stage 4c); Rubi's test builds v0.9.8-test.4 and test.5 carried it",
      "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME),
    a("VOCR6_N_w8.vocr", T61, "recognizer", "VOCR6-N (v0.4.0)", "Normal (default)", PREVIOUS,
      "VOCR6_N_w8_2l.vocr without its two-level head", "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME),
    a("VOCR6_L_w8.vocr", T61, "recognizer", "VOCR6-L (v0.4.0)", "Large", PREVIOUS,
      "VOCR6_L_w8_2l.vocr without its two-level head", "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME),
    a("MODEL_CARD_VOCR6.1_N.md", T61, "model card", "VOCR6-N (v0.4.0)", "Normal (default)", DOC,
      "v0.4.0-vocr6.1's VOCR6-N model card", "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME, src="MODEL_CARD_VOCR6_N.md"),
    a("MODEL_CARD_VOCR6.1_L.md", T61, "model card", "VOCR6-L (v0.4.0)", "Large", DOC,
      "v0.4.0-vocr6.1's VOCR6-L model card", "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME, src="MODEL_CARD_VOCR6_L.md"),
    a("vocr6.1-assets.json", T61, "asset list", "VOCR6 (v0.4.0)", "-", DOC, "v0.4.0-vocr6.1's list of its four blobs",
      "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME, src="vocr6-assets.json"),
    a("ATTRIBUTION_v0.4.0-vocr6.1.md", T61, "attribution", "VOCR6 (v0.4.0)", "-", DOC,
      "v0.4.0-vocr6.1's ATTRIBUTION.md", "ATTRIBUTION_v0.4.0-vocr6.1.md", NC_GAME, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ line spotters
    a("textspot8_t64n8c16_w8.vtsp", TS8, "line spotter", "textspot8", "-", RUBI,
      "Rubi's default line finder (ocr_finder = spotter; GRAD the fallback)", "SPOTTER-ATTRIBUTION.md", NC_GAME),
    a("MODEL_CARD_TEXTSPOT8.md", TS8, "model card", "textspot8", "-", DOC, "textspot8's model card",
      "SPOTTER-ATTRIBUTION.md", NC_GAME),
    a("SPOTTER-ATTRIBUTION.md", TS8, "attribution", "textspot8", "-", RUBI,
      "v0.3.10-textspot8.1's ATTRIBUTION.md; Rubi ships it as ocr/SPOTTER-ATTRIBUTION.md", "SPOTTER-ATTRIBUTION.md",
      NC_GAME, src="ATTRIBUTION.md"),
    a("textspot7_t64_w8.vtsp", "v0.3.9-textspot7.1", "line spotter", "textspot7", "-", RESEARCH,
      "selected by textspot7's rule but not resolved better than Rubi's finder, so not Rubi's default; Rubi took "
      "textspot8", "ATTRIBUTION_v0.3.9-textspot7.1.md", NC_GAME),
    a("MODEL_CARD_TEXTSPOT7.md", "v0.3.9-textspot7.1", "model card", "textspot7", "-", DOC, "textspot7's model card",
      "ATTRIBUTION_v0.3.9-textspot7.1.md", NC_GAME),
    a("ATTRIBUTION_v0.3.9-textspot7.1.md", "v0.3.9-textspot7.1", "attribution", "textspot7", "-", DOC,
      "v0.3.9-textspot7.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.9-textspot7.1.md", NC_GAME, src="ATTRIBUTION.md"),
    a("textspot6_qS4_w8.vtsp", "v0.3.8-textspot6.1", "line spotter", "textspot6 qS4", "-", RESEARCH,
      "textspot6's rule selected none (no unseen-dev interval below 0); its best eligible blob, as trained",
      "ATTRIBUTION_v0.3.8-textspot6.1.md", NC_GAME),
    a("MODEL_CARD_TEXTSPOT6.md", "v0.3.8-textspot6.1", "model card", "textspot6 qS4", "-", DOC,
      "textspot6's model card", "ATTRIBUTION_v0.3.8-textspot6.1.md", NC_GAME),
    a("ATTRIBUTION_v0.3.8-textspot6.1.md", "v0.3.8-textspot6.1", "attribution", "textspot6 qS4", "-", DOC,
      "v0.3.8-textspot6.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.8-textspot6.1.md", NC_GAME, src="ATTRIBUTION.md"),
    a("textspot3_t64n8c16_kd_w8.vtsp", "v0.3.4-textspot3.1", "text spotter (whole frame)", "textspot3", "-",
      PREVIOUS, "the pick of textspot3's rule (#108); Rubi's spotter is textspot8",
      "ATTRIBUTION_v0.3.4-textspot3.1.md", NC_SYNTH),
    a("MODEL_CARD_TEXTSPOT3.md", "v0.3.4-textspot3.1", "model card", "textspot3", "-", DOC,
      "textspot3's model card", "ATTRIBUTION_v0.3.4-textspot3.1.md", NC_SYNTH),
    a("ATTRIBUTION_v0.3.4-textspot3.1.md", "v0.3.4-textspot3.1", "attribution", "textspot3", "-", DOC,
      "v0.3.4-textspot3.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.4-textspot3.1.md", NC_SYNTH, src="ATTRIBUTION.md"),
    a("textspot_s4_c24n4_w8.vtsp", "v0.3.3-textspot.1", "text spotter (whole frame)", "textspot s4_c24n4", "-",
      PREVIOUS, "the winner of textspot's rule (#104); superseded by textspot3's pick, then by textspot8",
      "ATTRIBUTION_v0.3.3-textspot.1.md", NC_SYNTH),
    a("textspot_s4p_c8_w8.vtsp", "v0.3.3-textspot.1", "text spotter (whole frame)", "textspot s4p_c8", "-",
      RESEARCH, "the smallest candidate that met textspot's bar, not its selection; never read on a test set",
      "ATTRIBUTION_v0.3.3-textspot.1.md", NC_SYNTH),
    a("MODEL_CARD_TEXTSPOT.md", "v0.3.3-textspot.1", "model card", "textspot s4_c24n4", "-", DOC,
      "textspot's model card", "ATTRIBUTION_v0.3.3-textspot.1.md", NC_SYNTH),
    a("ATTRIBUTION_v0.3.3-textspot.1.md", "v0.3.3-textspot.1", "attribution", "textspot", "-", DOC,
      "v0.3.3-textspot.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.3-textspot.1.md", NC_SYNTH, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ the character LMs
    a("vocr_lm_o3_500k.vlm", "v0.3.7-lm6.2", "character LM", "ctclm.2 0.5 MB", "LM tier 1", RUBI,
      "Rubi's 0.5 MB LM tier (ctclm.2's blob, byte for byte)", "LM-ATTRIBUTION.md", BYSA),
    a("vocr_lm6_o3_1000k.vlm", "v0.3.7-lm6.2", "character LM", "lm6 1 MB", "LM tier 2 (default)", RUBI,
      "Rubi's default 1 MB LM tier (the game-text build)", "LM-ATTRIBUTION.md", BYSA_GAME),
    a("vocr_lm_o3_2000k.vlm", "v0.3.7-lm6.2", "character LM", "ctclm.2 2 MB", "LM tier 3", RUBI,
      "Rubi's 2 MB LM tier (ctclm.2's blob, byte for byte)", "LM-ATTRIBUTION.md", BYSA),
    a("vocr_lm6_o3_4000k.vlm", "v0.3.7-lm6.2", "character LM", "lm6 4 MB", "LM tier 4", RUBI,
      "Rubi's 4 MB LM tier (the game-text build)", "LM-ATTRIBUTION.md", BYSA_GAME),
    a("vocr_lm_o3_500k.json", "v0.3.7-lm6.2", "LM build manifest", "ctclm.2 0.5 MB", "LM tier 1", DOC,
      "the 0.5 MB LM's build manifest", "LM-ATTRIBUTION.md", BYSA),
    a("vocr_lm6_o3_1000k.json", "v0.3.7-lm6.2", "LM build manifest", "lm6 1 MB", "LM tier 2 (default)", DOC,
      "the 1 MB LM's build manifest", "LM-ATTRIBUTION.md", BYSA_GAME),
    a("vocr_lm_o3_2000k.json", "v0.3.7-lm6.2", "LM build manifest", "ctclm.2 2 MB", "LM tier 3", DOC,
      "the 2 MB LM's build manifest", "LM-ATTRIBUTION.md", BYSA),
    a("vocr_lm6_o3_4000k.json", "v0.3.7-lm6.2", "LM build manifest", "lm6 4 MB", "LM tier 4", DOC,
      "the 4 MB LM's build manifest", "LM-ATTRIBUTION.md", BYSA_GAME),
    a("LM-settings.json", "v0.3.7-lm6.2", "LM decoder settings", "LMs", "-", DOC,
      "each LM tier's decoder settings (v0.3.7-lm6.2's settings.json)", "LM-ATTRIBUTION.md", BYSA,
      src="settings.json"),
    a("LM-ATTRIBUTION.md", "v0.3.7-lm6.2", "attribution", "LMs", "-", RUBI,
      "v0.3.7-lm6.2's ATTRIBUTION.md; Rubi ships it beside the LMs", "LM-ATTRIBUTION.md", BYSA_GAME,
      src="ATTRIBUTION.md"),
    a("ATTRIBUTION_v0.3.7-lm6.1.md", "v0.3.7-lm6.1", "attribution", "LMs", "-", DOC,
      "v0.3.7-lm6.1's ATTRIBUTION.md, superseded by LM-ATTRIBUTION.md (lm6.2's), which corrects it: it missed a "
      "seen title (Little Busters!) and called Collar x Malice a vita-test game", "LM-ATTRIBUTION.md", BYSA_GAME,
      src="ATTRIBUTION.md"),
    a("vocr_lm_o3_1000k.vlm", "v0.2.6-ctclm.2", "character LM", "ctclm.2 1 MB", "LM tier 2 (default)", PREVIOUS,
      "Rubi's default 1 MB LM before lm6 (also in v0.3.0); replaced by vocr_lm6_o3_1000k.vlm",
      "ATTRIBUTION_v0.2.6-ctclm.2.md", BYSA),
    a("vocr_lm_o3_1000k.json", "v0.2.6-ctclm.2", "LM build manifest", "ctclm.2 1 MB", "LM tier 2 (default)", DOC,
      "ctclm.2's 1 MB LM's build manifest", "ATTRIBUTION_v0.2.6-ctclm.2.md", BYSA),
    a("vocr_lm_o3_4000k.vlm", "v0.2.6-ctclm.2", "character LM", "ctclm.2 4 MB", "LM tier 4", PREVIOUS,
      "Rubi's 4 MB LM tier before lm6; replaced by vocr_lm6_o3_4000k.vlm", "ATTRIBUTION_v0.2.6-ctclm.2.md", BYSA),
    a("vocr_lm_o3_4000k.json", "v0.2.6-ctclm.2", "LM build manifest", "ctclm.2 4 MB", "LM tier 4", DOC,
      "ctclm.2's 4 MB LM's build manifest", "ATTRIBUTION_v0.2.6-ctclm.2.md", BYSA),
    a("ATTRIBUTION_v0.2.6-ctclm.2.md", "v0.2.6-ctclm.2", "attribution", "ctclm.2", "-", DOC,
      "v0.2.6-ctclm.2's ATTRIBUTION.md", "ATTRIBUTION_v0.2.6-ctclm.2.md", BYSA, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ VOCR5-V, the gate, the detector
    a("VOCR5_V_w8.vocr", "v0.3.2-vertical.1", "recognizer (vertical)", "VOCR5-V", "-", RUBI,
      "Rubi's vertical-text model (ocr_vertical)", "VERTICAL-ATTRIBUTION.md", NC_SYNTH),
    a("MODEL_CARD_VOCR5_V.md", "v0.3.2-vertical.1", "model card", "VOCR5-V", "-", DOC, "VOCR5-V's model card",
      "VERTICAL-ATTRIBUTION.md", NC_SYNTH),
    a("VERTICAL-ATTRIBUTION.md", "v0.3.2-vertical.1", "attribution", "VOCR5-V", "-", RUBI,
      "v0.3.2-vertical.1's ATTRIBUTION.md; Rubi ships it as ocr/VERTICAL-ATTRIBUTION.md", "VERTICAL-ATTRIBUTION.md",
      NC_SYNTH, src="ATTRIBUTION.md"),
    a("textgate_p8_c8_w8.vtd", "v0.3.2-textgate.1", "text-presence gate", "textgate p8_c8", "-", RUBI,
      "Rubi's text-presence gate (ocr_text_gate, off by default)", "GATE-ATTRIBUTION.md", NC_SYNTH),
    a("MODEL_CARD_TEXTGATE_P8C8.md", "v0.3.2-textgate.1", "model card", "textgate p8_c8", "-", DOC,
      "the gate's model card", "GATE-ATTRIBUTION.md", NC_SYNTH),
    a("GATE-ATTRIBUTION.md", "v0.3.2-textgate.1", "attribution", "textgate p8_c8", "-", RUBI,
      "v0.3.2-textgate.1's ATTRIBUTION.md; Rubi ships it as ocr/GATE-ATTRIBUTION.md", "GATE-ATTRIBUTION.md",
      NC_SYNTH, src="ATTRIBUTION.md"),
    a("textdet_s8_c24_w8.vtd", "v0.3.1-textdet.1", "text detector", "textdet s8_c24", "-", RESEARCH,
      "not promoted: not resolved better than GRAD + refine on development, worse on CLANNAD as scored",
      "ATTRIBUTION_v0.3.1-textdet.1.md", NC_SYNTH),
    a("MODEL_CARD_TEXTDET_S8C24.md", "v0.3.1-textdet.1", "model card", "textdet s8_c24", "-", DOC,
      "the detector's model card", "ATTRIBUTION_v0.3.1-textdet.1.md", NC_SYNTH),
    a("ATTRIBUTION_v0.3.1-textdet.1.md", "v0.3.1-textdet.1", "attribution", "textdet s8_c24", "-", DOC,
      "v0.3.1-textdet.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.1-textdet.1.md", NC_SYNTH, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ model B (PSP)
    a("MODELB_B24_w8g2.vocr", "v0.3.7-modelb-test.1", "recognizer (PSP candidate)", "model B B24", "-", RESEARCH,
      "a console-test candidate only: +3.93 points worse than VOCR5-S on development PSP games (#126)",
      "ATTRIBUTION_v0.3.7-modelb-test.1.md", NC_GAME),
    a("MODELB_S32_w8g2.vocr", "v0.3.7-modelb-test.1", "recognizer (PSP candidate)", "model B S32", "-", RESEARCH,
      "a console-test candidate only: ties VOCR5-S but est. 2.05 s per box on the PSP, over the 1 s budget (#126)",
      "ATTRIBUTION_v0.3.7-modelb-test.1.md", NC_GAME),
    a("ATTRIBUTION_v0.3.7-modelb-test.1.md", "v0.3.7-modelb-test.1", "attribution", "model B", "-", DOC,
      "v0.3.7-modelb-test.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.7-modelb-test.1.md", NC_GAME, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ VOCR5 with its speed sections
    a("VOCR5_S_w8g.vocr", "v0.3.6-head2level.1", "recognizer", "VOCR5-S", "Small", PREVIOUS,
      "Rubi 0.9.8's Small (head gate); replaced by VOCR6.2_S_w8.vocr", "MODEL-ATTRIBUTION.md", NC_SYNTH),
    a("VOCR5_N_w8g2.vocr", "v0.3.6-head2level.1", "recognizer", "VOCR5-N", "Normal (default)", PREVIOUS,
      "Rubi 0.9.8's Normal (head gate and two-level head); replaced by VOCR6.2_N_w8g2.vocr", "MODEL-ATTRIBUTION.md",
      NC_SYNTH),
    a("VOCR5_L_w8g2.vocr", "v0.3.6-head2level.1", "recognizer", "VOCR5-L", "Large", PREVIOUS,
      "Rubi 0.9.8's Large (head gate and two-level head); replaced by VOCR6.2_L_w8_2l.vocr", "MODEL-ATTRIBUTION.md",
      NC_SYNTH),
    a("MODEL-ATTRIBUTION.md", "v0.3.6-head2level.1", "attribution", "VOCR5", "-", RUBI,
      "v0.3.6-head2level.1's ATTRIBUTION.md (VOCR5's training data, which every VOCR6 tier starts from); Rubi ships "
      "it as ocr/MODEL-ATTRIBUTION.md", "MODEL-ATTRIBUTION.md", NC_SYNTH, src="ATTRIBUTION.md"),
    a("VOCR5_N_w8g.vocr", "v0.3.5-headgate.1", "recognizer", "VOCR5-N", "Normal (default)", PREVIOUS,
      "VOCR5-N with the head gate alone; superseded by VOCR5_N_w8g2.vocr", "ATTRIBUTION_v0.3.5-headgate.1.md",
      NC_SYNTH),
    a("VOCR5_L_w8g.vocr", "v0.3.5-headgate.1", "recognizer", "VOCR5-L", "Large", PREVIOUS,
      "VOCR5-L with the head gate alone; superseded by VOCR5_L_w8g2.vocr", "ATTRIBUTION_v0.3.5-headgate.1.md",
      NC_SYNTH),
    a("ATTRIBUTION_v0.3.5-headgate.1.md", "v0.3.5-headgate.1", "attribution", "VOCR5 (head gate)", "-", DOC,
      "v0.3.5-headgate.1's ATTRIBUTION.md", "ATTRIBUTION_v0.3.5-headgate.1.md", NC_SYNTH, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ VOCR5 (v0.3.0)
    a("VOCR5_S_w8.vocr", "v0.3.0", "recognizer", "VOCR5-S", "Small", PREVIOUS,
      "v0.3.0's Small, plain; replaced by VOCR6-S", "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("VOCR5_N_w8.vocr", "v0.3.0", "recognizer", "VOCR5-N", "Normal (default)", PREVIOUS,
      "v0.3.0's Normal, plain; replaced by VOCR6-N", "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("VOCR5_L_w8.vocr", "v0.3.0", "recognizer", "VOCR5-L", "Large", PREVIOUS,
      "v0.3.0's Large, plain; replaced by VOCR6-L", "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("MODEL_CARD_VOCR5_S.md", "v0.3.0", "model card", "VOCR5-S", "Small", DOC, "VOCR5-S's model card",
      "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("MODEL_CARD_VOCR5_N.md", "v0.3.0", "model card", "VOCR5-N", "Normal (default)", DOC, "VOCR5-N's model card",
      "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("MODEL_CARD_VOCR5_L.md", "v0.3.0", "model card", "VOCR5-L", "Large", DOC, "VOCR5-L's model card",
      "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("vocr5-assets.json", "v0.3.0", "asset list", "VOCR5", "-", DOC, "v0.3.0's list of its assets",
      "ATTRIBUTION_v0.3.0.md", NC_SYNTH),
    a("ATTRIBUTION_v0.3.0.md", "v0.3.0", "attribution", "VOCR5", "-", DOC, "v0.3.0's ATTRIBUTION.md",
      "ATTRIBUTION_v0.3.0.md", NC_SYNTH, src="ATTRIBUTION.md"),
    # ------------------------------------------------------------------ VOCR4 and VOCR3
    a("VOCR4_FL10_w8.vocr", "v0.2.5-vocr4.1", "recognizer", "VOCR4-FL10", "Small", RESEARCH,
      "not qualified: fails VOCR2's whole-dialogue guard (docs/vocr4.md)", None, NC_SYNTH),
    a("VOCR4_H15_w8.vocr", "v0.2.5-vocr4.1", "recognizer", "VOCR4-H15", "Normal (default)", RESEARCH,
      "not qualified: fails VOCR2's whole-dialogue guard and the fixed-line hiragana condition (docs/vocr4.md)",
      None, NC_SYNTH),
    a("VOCR4_W20_w8.vocr", "v0.2.5-vocr4.1", "recognizer", "VOCR4-W20", "Large", RUBI_TOOLS,
      "Rubi's Large before VOCR5; now Rubi's host tools' cascade model of a request by name",
      None, NC_SYNTH),
    a("vocr4-assets.json", "v0.2.5-vocr4.1", "asset list", "VOCR4", "-", DOC, "v0.2.5-vocr4.1's list of its blobs",
      None, NC_SYNTH),
    a("VOCR3_FL10_w8.vocr", "v0.2.5-vocr3.1", "recognizer", "VOCR3-FL10", "Small", RUBI_TOOLS,
      "Rubi's Small before VOCR5; now pinned by Rubi's host tools", None, NC_SYNTH),
    a("VOCR3_H15_w8.vocr", "v0.2.5-vocr3.1", "recognizer", "VOCR3-H15", "Normal (default)", RUBI_TOOLS,
      "Rubi's Normal before VOCR5 (the model the console timings are calibrated to); now pinned by Rubi's host "
      "tools", None, NC_SYNTH),
    a("vocr3-assets.json", "v0.2.5-vocr3.1", "asset list", "VOCR3", "-", DOC, "v0.2.5-vocr3.1's list of its blobs",
      None, NC_SYNTH),
    # ------------------------------------------------------------------ VOCR2 (the trial before VOCR3)
    a("VOCR2_FL10_w8.vocr", "v0.2.3-vocr2.1", "recognizer", "VOCR2-FL10", "-", RESEARCH,
      "promotion rejected: it failed the strict whole-textbox guard (19 of 372 title checks regressed); the overlay's "
      "opt-in VOCR2 trial (v0.7.12-vocr2.1, removed after v0.8.0-test.1)", None, VOCR2_LIC),
    a("vocr2-qualification.md", "v0.2.3-vocr2.1", "qualification notes", "VOCR2-FL10", "-", DOC,
      "VOCR2's qualification notes", None, VOCR2_LIC),
]

LEFT_OUT = [
    {"tag": "v0.3.7-lm6.1", "why": "superseded by v0.3.7-lm6.2: the same four LM blobs and build manifests byte for "
     "byte (here from v0.3.7-lm6.2); its settings.json differs only in its release name and note. Its ATTRIBUTION.md "
     "is here as ATTRIBUTION_v0.3.7-lm6.1.md"},
    {"tag": "v0.2.6-ctclm.1", "why": "superseded by v0.2.6-ctclm.2 the same day: built before the protected-title "
     "name filter"},
    {"tag": "v0.3.7-modelb-test.1", "asset": "vocr6_psp_console_test.zip",
     "why": "the PSP console test apps (9 EBOOTs with generated test lines), not a model: its two blobs are here, "
     "and the zip stays on v0.3.7-modelb-test.1 for issue #145's console test"},
    {"tag": "every source release", "asset": "SHA256SUMS",
     "why": "replaced by this release's SHA256SUMS, which lists every other file of v0.5.0 under its v0.5.0 name"},
    {"tag": "v0.2.2-fl10-training.1, v0.2.2-fl10-bitmap.1, v0.2.2-fl10-general.1, v0.2.4-fl10-head.1",
     "why": "the FL10 research snapshots before VOCR3, each rejected or research only: archives of training "
     "checkpoints, code and results of 54,417,095 to 363,004,078 B each, 639,150,581 B together, so not trivially "
     "small; they stay on their own pre-releases"},
    {"tag": "v0.2.3-vocr2.1", "asset": "vocr2-quarter-checkpoint.pt, snapshot.json, export-verification.json, "
     "vocr2-candidate-20261012.json",
     "why": "VOCR2's training checkpoint (2,426,644 B) and its result and identity files stay on its pre-release; "
     "its int8 blob and qualification notes are here"},
    {"tag": "v0.2.2-meiki.1", "why": "MeikiOCR's own recognizer converted to MNN for the overlay's Meiki trial: a "
     "third-party model (LGPL-3.0 weights), not one of this repository's, which keeps its licence files on its "
     "release"},
    {"tag": "v0.1.0-baseline, v0.2.0, v0.2.1", "why": "the releases before VOCR2 (v1; FL10, H15 and F20): normal "
     "releases already, and they stay where they are; Rubi's host tools pin v0.2.0's int8 blobs from there"},
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def release_sums(d):
    out = {}
    p = Path(d) / "SHA256SUMS"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([0-9a-f]{64})\s+\*?(.+)$", line.strip())
            if m:
                out[m.group(2)] = m.group(1)
    return out


RUBI_PINS = {
    "repo": "bee-san/Rubi_for_vita",
    "commit": "dc272d107ee910661e555f70ca5ef1db67425666",
    "files": ["tools/prepare_ocr.py", "include/vjo_vocr_sizes.h", "tools/model_tiers.json"],
    "release": "v0.10.0",
    "release_commit": "efbee95e28eab11a4f960702acc0bc6d183df812",
    "note": "read at Rubi main dc272d1 (2026-10-28); Rubi 0.10.0 (released 2026-10-28T09:00:56Z, tag efbee95) has "
            "the same tools/prepare_ocr.py and include/vjo_vocr_sizes.h, byte for byte, so these are the pins it "
            "ships; the release before, 0.9.8, pins VOCR5's v0.3.6-head2level.1 blobs as its sizes and no spotter",
    "shipped": {
        "VOCR6.2_S_w8.vocr": "864bbd196d5bc5d3a7bd7918cf7274b3503dfe65d440017ef0781ea7e7ba37d2",
        "VOCR6.2_N_w8g2.vocr": "107cf084cc0c1a3a61ce4c3f9b9cf6b31d6dff80f0a1bdb94afd3f7ee9ae3bc9",
        "VOCR6.2_L_w8_2l.vocr": "65310013e8f990886e4e8097b2fe55eb98cdf082d7193eea510601fc0ac26c82",
        "VOCR5_V_w8.vocr": "ccd20c63bb2620f054538b456904bdf19b74d483d3da8de865009e282bb04172",
        "vocr_lm_o3_500k.vlm": "38ea94e0c4ec93863454387e7c2c469d209aaf21f2f6ee5d040e75a09d00d286",
        "vocr_lm6_o3_1000k.vlm": "41591accf8499f1c34557e585cca382de2c4b5ca15f270c985612c32487e251f",
        "vocr_lm_o3_2000k.vlm": "ac33e461bd346617aae27168aee47230a758f4b301d78b59eb55ca25a21bcb34",
        "vocr_lm6_o3_4000k.vlm": "a65841b5bdce3e3123b2d1e712deb857fb577cbd2fd8667e8fb06ab35d368335",
        "textgate_p8_c8_w8.vtd": "a4e73ad78451c75a19f88a479fc0108daac85beaa88d875b73212ed3c09d85d1",
        "textspot8_t64n8c16_w8.vtsp": "e129d41adfa02c658a7d13d6f98064e6ab6ce9a2bc8c7a56da700a21357b5e1c",
        "VOCR6-ATTRIBUTION.md": "9a5ff2aee4cb8194f41e54f3a2dac88ce3de6ff4aa7b8ddd123dd5b792c9843b",
        "MODEL-ATTRIBUTION.md": "74b15e8a8ffb01a997ff15bff3ebc137c48a51439732fd838300a9ddff4953c3",
        "LM-ATTRIBUTION.md": "9418ed87e6d9fa196aba2315e05a0cf0dca7056482cce03577ced423e5feb1d3",
        "GATE-ATTRIBUTION.md": "44a353c76e19c6bdabcb9dbd06d14867696dcb59ce0460db22d09749af687f6e",
        "VERTICAL-ATTRIBUTION.md": "0f644c6dce8c24abb9ac56ecbb684ab7cd1f1950cad0aae15eafadba04bcbcd9",
        "SPOTTER-ATTRIBUTION.md": "fd44f500114ca2c8373e02bda1134753619e76de869ddf8ec86c14e008be5b84",
    },
    "host_tools_and_tests": {
        "VOCR6.2_N_w8.vocr": "c8ce844ed0a637eb5d263dff7837d846bd2a520278da3b74c49c5ee33eaa24c9",
        "VOCR6.2_L_w8.vocr": "be5a8e4f3280b58edc8398616ce5068f9803551407fbc9a843c0a2963584b7b2",
        "VOCR4_W20_w8.vocr": "2745e69699ad7b2aba3092f83e5ddc1912f405f54b9df8a2e55be82c385b716e",
        "VOCR3_H15_w8.vocr": "cb3c61a8f91f9aeeb3efcc0e79313f664112f25c4bd55865bae9df52522b360c",
        "VOCR3_FL10_w8.vocr": "87fb4d0c18416f5f3771cecd274c8db3cf69b8b4a4ee9f204edebb4b82079154",
    },
}


def source_path(src, x):
    return Path(src) / x["from"]["tag"] / x["from"]["name"]


def cmd_manifest(args):
    names = [x["name"] for x in ASSETS]
    if len(set(names)) != len(names) or LIST_NAME in names or "SHA256SUMS" in names:
        raise SystemExit("asset names must be unique and leave the list's and SHA256SUMS's free")
    out = []
    for x in ASSETS:
        p = source_path(args.src, x)
        h, n = sha256(p), p.stat().st_size
        listed = release_sums(p.parent).get(x["from"]["name"])
        if listed is not None and listed != h:
            raise SystemExit(f"{p}: sha256 {h} differs from its release's SHA256SUMS ({listed})")
        out.append({**x, "bytes": n, "sha256": h, "from": {**x["from"], "in_release_sha256sums": listed is not None}})
    by = {x["name"]: x for x in out}
    for group in ("shipped", "host_tools_and_tests"):
        for name, h in RUBI_PINS[group].items():
            if by[name]["sha256"] != h:
                raise SystemExit(f"{name}: not the blob Rubi pins ({h})")
    for x in out:
        shipped, tools = x["name"] in RUBI_PINS["shipped"], x["name"] in RUBI_PINS["host_tools_and_tests"]
        if shipped != (x["status"] == RUBI) or tools != (x["status"] == RUBI_TOOLS):
            raise SystemExit(f"{x['name']}: status {x['status']!r} disagrees with Rubi's pins")
    seen = {}
    for x in out:                                     # one file per set of bytes, except the renamed attributions
        seen.setdefault(x["sha256"], []).append(x["name"])
    dup = {h: v for h, v in seen.items() if len(v) > 1}
    if dup:
        raise SystemExit(f"the same bytes under two names: {dup}")
    doc = {"kind": "vita-vn-ocr-release-assets", "schema": 1, "release": RELEASE,
           "files": len(out) + 2, "assets": out, "rubi": RUBI_PINS, "left_out": LEFT_OUT,
           "statuses": [RUBI, RUBI_TOOLS, PREVIOUS, RESEARCH, DOC],
           "note": "SHA256SUMS and this list (published as " + LIST_NAME + ") are the release's other two files"}
    MANIFEST.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}: {len(out)} assets, {sum(x['bytes'] for x in out):,} B")


def sums_text(d):
    files = sorted(p.name for p in Path(d).iterdir() if p.is_file() and p.name != "SHA256SUMS")
    return "".join(f"{sha256(Path(d) / f)}  {f}\n" for f in files)


def cmd_stage(args):
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for x in doc["assets"]:
        p = source_path(args.src, x)
        if sha256(p) != x["sha256"] or p.stat().st_size != x["bytes"]:
            raise SystemExit(f"{p}: not the bytes docs/models.json lists")
        shutil.copyfile(p, out / x["name"])
    shutil.copyfile(MANIFEST, out / LIST_NAME)
    (out / "SHA256SUMS").write_text(sums_text(out), encoding="utf-8")
    print(f"staged {len(doc['assets']) + 2} files in {out}")


def cmd_verify(args):
    d = Path(args.dir)
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = release_sums(d)
    want = {x["name"]: x["sha256"] for x in doc["assets"]}
    want[LIST_NAME] = sha256(MANIFEST)
    got = {p.name for p in d.iterdir() if p.is_file()} - {"SHA256SUMS"}
    bad = []
    if set(listed) != set(want) or got != set(want):
        bad.append(f"files differ: missing {sorted(set(want) - got)}, extra {sorted(got - set(want))}, "
                   f"SHA256SUMS lists {len(listed)}")
    for name, h in want.items():
        if name in got:
            g = sha256(d / name)
            if g != h or listed.get(name) != h:
                bad.append(f"{name}: {g} (SHA256SUMS {listed.get(name)}, docs/models.json {h})")
    print(f"verified {len(want) - len(bad)} of {len(want)} files against SHA256SUMS and docs/models.json")
    if bad:
        raise SystemExit("\n".join(bad))


def files_table(doc):
    out = ["| file | what | tier | status | why | attribution | licence |", "|---|---|---|---|---|---|---|"]
    for x in doc["assets"]:
        att = f"`{x['attribution']}`" if x["attribution"] else "none in its release (see Licences)"
        out.append(f"| `{x['name']}` | {x['kind']}: {x['model']} | {x['tier']} | {x['status']} | {x['why']} | {att} | "
                   f"{x['licence']} |")
    return "\n".join(out)


def checksums_table(doc):
    out = ["| file | from | bytes | sha256 |", "|---|---|---:|---|"]
    for x in doc["assets"]:
        src = x["from"]["tag"] + ("" if x["from"]["name"] == x["name"] else f" (`{x['from']['name']}`)")
        out.append(f"| `{x['name']}` | {src} | {x['bytes']:,} | `{x['sha256']}` |")
    return "\n".join(out)


BLOCKS = {"files": files_table, "checksums": checksums_table}


def render_md(text, doc):
    """docs/models.md with its generated blocks (<!-- models.py:NAME --> ... <!-- /models.py:NAME -->) redrawn."""
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- models\.py:{name} -->\n)(?:.*?\n)?(<!-- /models\.py:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"docs/models.md has no {name} block")
        text = pat.sub(lambda m: m.group(1) + fn(doc) + "\n" + m.group(2), text)
    return text


def cmd_md(args):
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    md = HERE / "models.md"
    md.write_text(render_md(md.read_text(encoding="utf-8"), doc), encoding="utf-8")
    print(f"redrew {md}'s generated blocks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--src", default=str(SRC))
    sub.add_parser("md")
    s = sub.add_parser("stage")
    s.add_argument("--out", required=True)
    s.add_argument("--src", default=str(SRC))
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    args = ap.parse_args()
    {"manifest": cmd_manifest, "md": cmd_md, "stage": cmd_stage, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
