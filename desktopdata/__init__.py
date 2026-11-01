"""Desktop VN data (issue #169's data side, docs/desktop-data.md): resolution-diverse training pages for a desktop
text detector and a frozen real PC test set.

  desktopdata/capture.py   the desktop capture model: screen and window sizes, letterbox / pillarbox / stretch
                           geometry, resamplers, DPI scaling, downscaled and JPEG captures; exact box maps
  desktopdata/hooked.py    1a: the hooked pools v2.1 train pages re-rendered as desktop captures
  desktopdata/realplus.py  1b: realplus's PC screenshots at their own size, its line boxes, ignore areas for the
                           text its labels do not cover
  desktopdata/synth.py     1c: synthetic desktop frames at desktop resolutions (PC-style fonts, 24-64 px lines)
  desktopdata/shards.py    the shard formats the trainers read (textspot6-pages, #97's frames) and the manifests
  desktopdata/stats.py     per-resolution counts and text-height distributions against the Vita / PSP pools
  desktopdata/pctest.py    2: the real PC test set (VNDB PC screenshots of VNs in no pool, engine readings,
                           Claude adjudication, the frozen manifest, eval/desktop.py's frames)
  desktopdata/owner.py     the owner's desktop captures in the hooked bucket's format: import and native pages
  desktopdata/qa.py        QA sheets of the shards (local, never committed)

Pixels never enter git: frames, shards, crops and sheets stay under $HERD/shared/artifacts/desktop-data and
$HERD/shared/private/desktop-data. The repository holds code, manifests (ids, URLs, sha256, boxes, labels of public
VNDB screenshots), counts and docs.
"""
