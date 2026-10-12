# Fresh native textbox test: C4 rejected

The optional C4 row-recovery finder reduced pooled errors on a fresh 157-frame
console story-textbox test, but **failed its fixed acceptance gate**. It is not
promoted into the overlay release. VitaJPOverlay 0.7.12 continues to offer the
previous experimental VOCR2 implementation.

The same original FL10 weights were used with GRAD, C3 and C4. This isolates
finder behavior; it is **not** a retest of the adapted weights shipped with
VOCR2, and its results must not be combined with the separate
[CLANNAD Google-proxy comparison](clannad-vocr2-comparison-20261012.md).

## Cohort and reference review

The cohort was fixed before any candidate, baseline, finder or teacher output.
All images are original native 480×272 PSP or 960×544 Vita sources. Dialogue
window ROIs include names where present, with fixed symmetric 4/8-pixel
expansions to represent imprecise user crops. Incidental decorations, background
edges and any detected ruby/UI remain unmasked and can incur insertion errors.

| Console | Title | Unique body pages | Distinct body characters | Hiragana |
| --- | --- | ---: | ---: | ---: |
| PSP | Juusei to Diamond | 21 | 515 | 308 |
| PSP | Infinite Loop | 43 | 952 | 588 |
| PSP | Enkaku Sousa | 25 | 601 | 359 |
| Vita | Trillion | 17 | 522 | 290 |
| Vita | Raging Loop | 26 | 797 | 487 |
| Vita | Shiin | 25 | 526 | 296 |
| Total | Six titles | 157 | 3,913 | 2,328 |

All titles meet the unchanged requirement of five unique frames, 500 distinct
normalized body characters and 200 hiragana; each console has three titles.
Sixteen repeated whole-body pages were excluded deterministically before
scoring. Partially recurring body lines count once toward admission quotas,
but every admitted reference is scored: each crop variant has 4,247 all-text
characters, 3,933 body characters and 2,343 hiragana characters.

References and native ink coordinates received separate source-only assistant
reviews. Misassigned lines, shifted boxes and distinct small-kana/kanji readings
were corrected before inference, retaining the drafts and review hashes.
Uncertain frames were excluded. These are **assistant-reviewed references, not
independent human ground truth**. No Google calls were made.

Trillion is an RPG with ordinary VN-style story textboxes. The source plan
allowed that domain before its pixels were acquired; combat, menu, status and
reward screens were excluded. This is a console story-dialogue cohort, not six
pure visual novels. Oumagatoki and Genji remained below the fixed character
minimums after bounded native-source searches; they were not scored, and their
replacements were chosen from source availability before model output.

Known recorded training, calibration, development, prior evaluation and related
families were excluded using frozen manifests. **The original initializer's
training snapshot is missing**, so historical training independence cannot be
proven. Release IDs identify Japanese console products; exact retail binary
editions are not asserted for press/preview screenshots.

## Measured results

All-text error counts use the same 4,247-character denominator per variant.
Hiragana uses the same 2,343-character denominator. Lower is better.

| Crop | GRAD full errors | C3 full errors | C4 full errors | C3 → C4 reduction | C3 → C4 hiragana errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 995 | 965 | 876 | 89 fewer; 9.22% | 430 → 389; 9.53% fewer |
| +4 pixels | 1,101 | 1,039 | 996 | 43 fewer; 4.14% | 498 → 479; 3.82% fewer |
| +8 pixels | 1,208 | 1,146 | 1,024 | 122 fewer; 10.65% | 506 → 443; 12.45% fewer |

For the base crop, all-text CER against these references is 23.43% for GRAD,
22.72% for C3 and 20.63% for C4. C4's pooled gains do not establish non-regression:

- Infinite Loop hiragana errors increase from GRAD's 44 to C4's 47 at base,
  135 to 140 at +4, and 109 to 112 at +8. At +8 its all-text errors also increase
  from 209 to 213.
- Raging Loop base all-text errors increase from 100 to 101; body errors increase
  from 71 to 72.
- Base improves body hiragana on only one Vita title; the fixed gate requires
  two per console in every variant.
- Base and +4 miss the required 10% pooled full-error reduction versus C3.

All twelve pooled C4 metrics are non-regressing versus C3, but the per-title
checks and material-gain requirements still fail. The full per-title all/body
and full/hiragana counters and every failed condition are in the result JSON.
There was no retry, post-test tuning, checkpoint selection or fallback candidate.
These references remain outside training and calibration.

## Execution and artifacts

One completed run processed all **1,413 frame/variant/mode cases** and performed
3,776 recognition calls: GRAD 1,225, C3 1,242 and C4 1,309. Every crop and blank
prediction was retained; there were zero finder/reader failures, zero truncations
and no cases with zero detections. More C4 calls reflect additional detected rows.
This run does not establish a speed improvement.

A separate read-only audit independently recomputed all 1,413 case scores,
all aggregate counters and the exact failed acceptance decision from the saved
predictions, without importing the harness scorer or running inference. It also
verified case coverage, blank-output accounting, source/ROI pixel hashes,
deduplication, quotas and frozen input/code/model/environment identities.
An audit pass confirms the recorded measurements; the candidate still fails.

The W8I blob remains 657,664 bytes, with a 124,832-byte recognizer arena at maximum
width 1,024 and 3,501 classes. Those are allocation contracts, not total physical
console memory measurements. Execution was scalar C on a Linux x86_64 host,
Python 3.11.15, NumPy 2.4.6, Pillow 12.3.0 and Torch 2.5.1+cpu; dependencies match
the committed lock. No physical Vita or PSP test is claimed.

The run used the [fixed harness](fresh-native-textbox.md), its unchanged source
snapshot `5ae8a95746718e06c70fcd61f25f5ff6edef08cd`, and candidate finder source
`cfa7ee6ce5c381ab57dcbca37bbde931d3282e54`:

- Model SHA256: `500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.
- Scalar library SHA256: `4a5635ba9a0721373f35e8957c0788b74e1725ea8cd96b8200cc5e427536b11b`.
- Finder C source SHA256: `90c30a1c1e1707b818c2bd919243ee589700ff44ceac8b8d7a842b9d55d8d9ca`.
- Prepared-run SHA256: `eeae8e7ee40fa9af44a1bdb8101f41ea09879f2665d89cdbdd7b48b362b5f8a9`.
- [Unchanged aggregate result](../eval/fresh_native_textbox_20261012.json):
  `32669a0d4faf9a4647754cf169ad0bf9d5fe45e20188f79e68092c3022e2a85f`.
- [Source URLs, hashes, crop geometry, release provenance and review pins](../eval/fresh_native_source_provenance_20261012.json):
  `bf30e3c69a337efe8024fc405a9dbcadd923dc4339140042f174b490817dad01`.
- [Frozen reproduction helpers](../eval/reproducibility/fresh-native-20261012/README.md).
- [Independent stored-output audit](../eval/fresh_native_textbox_independent_audit_20261012.json):
  `7070cf320dd0007249e2c4557b12732f2bf25d298ad36ea7bd2bad78a3aec8c8`.

Images, transcripts, raw predictions and source HTML remain private. Public
manifests preserve their hashes; reproducing scores requires those exact private
references and pixels. The existing harness's generated tests validate its
contract separately from this real-screenshot result.
