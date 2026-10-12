# Frozen released-VOCR2 comparison source

These three `.snapshot` files are byte-for-byte copies of the private helpers used
for the completed once-only comparison and independent read-only audit. The
suffix archives source without automatic Python import, lint or test collection.
Restore the original `.py` filenames in a private working directory to inspect or
verify them. This is an exact historical source snapshot, **not a general portable
CI helper**. Its tests require the frozen baseline harness and C3-only library.

| Source snapshot | Original filename | SHA256 |
| --- | --- | --- |
| [fixed_compare.py.snapshot](fixed_compare.py.snapshot) | `fixed_compare.py` | `70ba2b7956dcef89d4f22c87d7f280d0189577264a20a16b21a8c672e995de8b` |
| [test_fixed_compare.py.snapshot](test_fixed_compare.py.snapshot) | `test_fixed_compare.py` | `8c2c5e60a5a43dc47ef4003f7d4e5a36bc3cd37c1be1f97c3e47d10909ad4a2a` |
| [audit_result.py.snapshot](audit_result.py.snapshot) | `audit_result.py` | `1d7e0d796d46063e667774941c9c5071adf205633ab468dbea44d4b1b176da50` |

The [results report](../../../docs/clannad-vocr2-comparison-20261012.md),
[unchanged aggregate](../../clannad_vocr2_comparison_20261012.json) and
[independent audit](../../clannad_vocr2_comparison_audit_20261012.json) record
the measured before/after counts, all four rate/relative decreases and artifact
bindings. The JSON bytes remain exactly those produced and independently verified
by this run. No OCR inference is needed to calculate the decreases from them.

## Exact source and environment

The original capture/cache harness is
[`e80690eb6c2f29ee84f5dddf55a1d21d8007c90b`](https://github.com/bee-san/vita-vn-ocr/commit/e80690eb6c2f29ee84f5dddf55a1d21d8007c90b).
The released C3 runtime source is
[`49ff6f4d1c55804c600d098810574a49d382a009`](https://github.com/bee-san/vita-vn-ocr/commit/49ff6f4d1c55804c600d098810574a49d382a009).
The recognizer C/H bytes are identical between these commits. C4 is not used.

The host run used x86-64 Linux 7.2.6-1-cachyos with glibc 2.44, Python 3.11.15,
NumPy 2.4.6, Pillow 12.3.0 and GCC 16.2.1 20260810. Generated verification uses
pytest 9.1.1; lint used Ruff 0.16.10. Scalar compilation uses
`-std=c99 -O2 -ffp-contract=off -DVOCR_NO_NEON -fPIC -shared`.
The original recognizer library also links `-lm`; the C3-only finder build adds
`-Wall -Wextra -Werror` and compiles only released `runtime/vocr_lines.c`.
Both explicit library byte hashes must match. Rebuilding with a different
compiler/toolchain may produce different library hashes and a different protocol.

The generated-test fixture expects the baseline checkout at the historical
`/tmp/vita-vn-ocr-clannad-longrun` path. The helper resolves input/library paths
from the frozen private protocol. Changing those paths, engine behavior or the
environment requires a newly reviewed preparation; it cannot silently be called
the original run. Byte-for-byte copies of the source preserve the listed hashes.

## Required private inputs

The public artifacts do not contain the private dataset. Reproducing the numerical
comparison requires the owner's retained exact inputs, cached outputs and frozen
receipts. The relevant hashes include:

| Input or receipt | SHA256 |
| --- | --- |
| Original session lock | `14ba9297269b1cc741a9993425b3f901a1086bed74be4eaad274381a83ac6425` |
| Original capture journal | `b6cd582c582d22af883710f202f60f9c44848be23c4a8fe2f726799c7f8f81ed` |
| Original Google/FL10 comparison journal | `f2754897c8b69f555706b93a3cd4f709b8066c4a7578c03baf828458db584cd9` |
| Candidate prediction journal | `664688518e01e065e6f8943d031bae8ced7588bc66b241d4b8e0d30cf25ebc6c` |
| Fixed private protocol | `105660119aa7d0151500d396b7071715e4e784f3ae904ef7419a13fd656847e8` |
| Prepared receipt | `5170914a9c693c8cb5549b48d7e05e58e932b8166b323180e024c66306b329d1` |
| Input/reference plan | `8f3553ae4c9b09cb587631a6fcfe98e38073356576c9cbc870069516b6887c45` |
| Finder build provenance | `08796873e3cd957ab4eec335d4821ee2c45b264bb34ba4ee4a50597730ede03b` |
| Original scalar recognizer library | `4f9cef809b88025ccaf3b5ee10f0f22eabc07505c6cd118f8a6af814656bfb59` |
| Released C3-only finder library | `e6495076c58f491e0801e68221bc6a4c78afc0a64f12c2b9be68b2722f537f4e` |
| Released VOCR2 model | `d3adc52e5cf62b548af04c5ab07a2f57cf0419386bbdf49ba19c9703be6e1766` |

The original journals bind all retained screenshot/ROI/line PNG file and decoded
pixel hashes and cached Google/FL10 text. The candidate journal binds the 4,271
original processed-line inputs and 4,345 reconstructed C3 crop inputs. The original
27 failed/empty-reference pages remain retained and excluded without retry.
Detailed journals, screenshots, crops, transcripts, game-license material, model
bytes and compiled libraries are not committed. Weights are available separately
on the [VOCR2 trial model release](https://github.com/bee-san/vita-vn-ocr/releases/tag/v0.2.3-vocr2.1).

The private `audit/` directory also retains the exact root authorization, once-only
attempt and completion receipts. Their hashes are in the independent audit.
`audit_result.py` expects that directory beside the restored helpers and makes no
model, native finder or Google calls. It reconstructs crop pixels from the original
ROIs and recorded geometry/polarity, reproduces aggregates and verifies original
and candidate artifacts did not change during the audit. It writes its result
with exclusive creation and must not overwrite the historical audit receipt.

## Generated verification

Copy each source snapshot to its original filename outside the repository, put
the exact C3-only library beside the tests as `libvocr-lines-c3.so`, and make the
frozen baseline checkout available at the fixture path. Then:

```sh
"$HOST_PYTHON" -m pytest -q test_fixed_compare.py
```

`HOST_PYTHON` is the pinned Python 3.11 environment above. The 37 tests use generated
images/fake readers and the actual frozen C3 library on generated block rows.
They issue no Google request or real-model inference. They cover cache/reference
binding, exact pixels/polarity, empty/reference/error accounting, normalization,
native mode 2 and row ordering/capacity, frozen hashes, authorization before data
or model access, and once-only/nonoverwrite guards. This command needs the
frozen harness/library fixtures, but no private screenshots or cached reference
text; it is separate from ordinary repository CI.

The real comparison is already complete. Its preserved `attempt.json` prevents
repeating it. A new diagnostic run needs a new nonoverwriting preparation and
exact review authorization. Google proxy output alone cannot promote the model;
the source snapshot does not permit selecting/tuning weights from this session.
