---
title: Full-feed picker ranking failure
created: 2026-10-10
updated: 2026-10-10
status: fixed-full-feed-two-page-verified
tags: [vita, clannad, ranking, validation]
---

# Full-feed picker ranking failure

Capturing a matching dialogue string is necessary but insufficient for the user's workflow. The shell displays only five of 24 retained sources. A full-feed real-game experiment showed that the production comparator can hide the initial dialogue behind noisy sources, even after the import mismatch is repaired.

## Why the first bridge missed this

The first test helper filtered short/non-Japanese events before offering them to the production ranker. It proved that the production text plugin could intercept real Japanese bytes and decode two visible narration pages from the same immediate stream. It did not reproduce the complete candidate population competing for the picker.

The next helper fed observations into the source table without that prefilter and saved complete source/top-five snapshots. That stronger check exposed the failure. Keep both results: the extraction result is valid within its scope, and the picker regression is a separate production defect.

## Failing baseline

| Snapshot | Observation | Meaning |
|---|---|---|
| First page, sequence 425 | Actual six-normalized-character dialogue retained in three streams; each display rank reported as 0 | Rank 0 is this diagnostic's “outside displayed top five,” not best rank. A user cannot pick these sources from the initial five. |
| Competing sources | 395-update three-character noise; nine-update mixed data with 25% Japanese characters; static font table, date and single kanji | Mere activity or the presence of any Japanese character is not sufficient evidence of useful dialogue. |
| Later page 4, sequence 1684 | Immediate source `0x8102CA7F` has three updates and display rank 5; other matching streams still outside the five | Later activity can lift a useful source, but initial usability is not assured. |

The earlier filtered helper reported only two updates by page 4. The full feed includes a one-kanji page that the prefilter excluded, so the count difference has a known cause. Do not merge those helper measurements into one unlabeled series.

## Root cause inspected in production source

At the captured baseline, `core/text_source.c::better` compares:

1. Whether each candidate has any Japanese codepoint.
2. OCR similarity score.
3. Update count.
4. Japanese fraction, using cross multiplication.
5. Normalized length.
6. Stable source ID.

The update-count comparison before fraction/length allows rapidly changing short or mixed data to dominate. This comparator governs both retained-source replacement and the final five displayed choices, so a correction must consider both admission and presentation. A selected source has separate retention protection; do not break selected-source continuity while improving initial discovery.

## Correction and required evidence

The comparator change and synthetic regression fixture are now committed as `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e` in PR #12. Ordering is Japanese presence, OCR similarity, Japanese fraction, normalized length, update count, stable ID. No new allocation or source-table storage is added.

The invented fixture presents five active noise/menu streams alongside three quiet complete Japanese streams. It also preserves OCR-match priority and selected-source updates to short lines. Replacing the corrected comparator with the old one makes four fixture checks fail, providing a discriminating regression test. No game strings were added to the tests.

| Validation at `6eef6c0` | Result |
|---|---|
| Full host CTest | 10 suites passed, 5.44 s recorded run |
| Vita release build and packaged ABI | Passed |
| Kernel instruction audit | 5,921 ARM instructions; zero FPU/NEON |
| Isolated Vita3K | 264 checks passed: 52 portable + 192 observer + 20 plugin |
| Source table | 26,480 B, unchanged |
| Old comparator with new regression fixture | Four failures |
| Corrected full-feed real-game rerun | Passed for the two checked visible pages; three matching sources displayed on each |

The new 38,620 B kernel SHA-256 is `6cc2dc868dc4403bcab252d689acd2f0779a55f9bd8069654b335eaffa571933`. Shell `45f86b1e...` and text `84259631...` are unchanged. The corrected emulator-only full-feed bridge hash is `106be9d495ce17b622f67913d0f5cd8c09ce7dd767c5e647c59b85095364c316`; it is separate from the earlier helper and must never be installed on hardware.

## Corrected full-feed result

Private proof status: `generic_capture_and_top5_ranking_verified_for_two_visible_pages`. The same three streams contain the matching current page in both complete snapshots:

| Stream | First-page rank / updates | Later page-4 rank / updates |
|---|---|---|
| `0x8102CA7F` | 2 / 1 | 3 / 3 |
| `0x810A2393` | 3 / 1 | 4 / 3 |
| `0x810A34EF` | 4 / 1 | 5 / 3 |

First-page sequence 424 recorded 173,669 bridge calls; later-page sequence 1680 recorded 228,453. These are cumulative diagnostic counters, not an overhead or throughput benchmark. The first matching text has six normalized Japanese codepoints (24 UTF-8 B); the later text has four (18 UTF-8 B). No text is reproduced here.

Both snapshots contain five choices and `selected=0`. Thus generic capture, page changes and top-five **data** ranking are established for these two pages; manual selection, actual shell UI behavior, selected-source lifetime and dictionary operation are not exercised by this experiment. The emulator bridge still does not establish physical kernel mapping, pinning, syscall cost or boot.

The failed proof remains `offline-capture-proof-before-ranking.json` (5,962 B, SHA-256 `5e707f06762778b18a3e7c1d2eb1913ec00abc8b0463407dce41816d56426579`). The successful proof records separate screenshot, capture-log and source-snapshot hashes. See [[11-Sources-and-artifacts]]. The earlier 255-check run predates the ranking change; its synthetic regression result is the later 264-check run.

Physical deployment remains pending. The PR's coherent outcome is usable generic capture, which includes source selection as well as interception. See [[12-Open-questions-and-next-steps]].

Related: [[03-Native-text-architecture]], [[07-CLANNAD-investigation]], [[14-Retail-libc-hook-coverage]], [[15-Vita3K-runtime-investigation]], [[2026-10-10-Investigation-log]].
