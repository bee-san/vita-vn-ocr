# realplus: a large real training pool from training-side VNDB screenshots

realplus is a pool of real Japanese visual-novel text lines with clean labels, for fine-tuning the
console recognizer. It uses every usable training-side VNDB screenshot and labels the lines with
devnative's pipeline (PP-OCRv6 small and MeikiOCR read every line; Claude adjudicates on contact
sheets; devnative's training rule decides). The screenshots, crops, labels and npz files stay local
under `$HERD/shared/artifacts/realplus` (v1, v2) and `$HERD/shared/artifacts/realplus-v3` (v3).
The repository holds the code (`realplus/`), a text-free
screenshot manifest (`realplus/manifests/shots_NN.jsonl`: URL, file and pixel sha256, VN, release,
capture session, line and crop counts), the counts at every stage (`realplus/manifests/stats.json`),
this report and the tests.

## Result

The pool is version 3 (`--tag v3`, built 2026-10-16 under `$HERD/shared/artifacts/realplus-v3`). v1 and v2 are
still on disk, unchanged, because VOCR3 pinned them: v2 drops three CLANNAD fan works from v1, and v3 adds the
Stage A review and the reclaim tier to v2 (see "Versions" and "Version 3").

| | |
|---|---|
| distinct lines | **101,350**, from **14,461 VNs** and 49,285 screenshots (v2: 43,228 / 9,137 / 25,314) |
| characters | 1,273,030 (eval's scoring key); median line 11 characters, mean 12.6 |
| ink crops | 170,387: 100,915 Vita, 69,472 PSP (`realplus_train_ink_v3.npz`, 782 MB) |
| GRAD crops | 116,994: 71,897 Vita, 45,097 PSP (`realplus_train_grad_v3.npz`, 645 MB) |
| labels | 61,992 lines (61.2%) agreed by all four readings (auto); 39,358 reviewed by Claude: 37,329 in v3's pass 1 under the agreement rule (30,837 with Codex gpt-6-luna reading the same text, 6,492 confirmed by a second, blind Claude look after Luna skipped or was unsure) and 2,029 from v2's review |
| precision (v3) | independent transcription of 800 stratified lines: **742 agree** as measured (92.75%, Wilson 95% 90.7-94.4%); after third looks at the 58 disagreements, **786 labels right** (98.25%, 97.1-99.0%), 4 wrong, 10 unsettled (see "Version 3") |

v3 reaches the 100k-line goal v2 fell short of: 101,350 distinct lines and 287,381 crops in all, against train2's
real fine-tune data of about 32k crops (tt 12,115, rd 9,811, ps 10,706).

The files are under `$HERD/shared/artifacts/realplus-v3` (v1 and v2: `$HERD/shared/artifacts/realplus`).
`pool.py build` never overwrites a built pool: a new build needs `--tag`, which gives every output a new file
name. `realplus/manifests/stats.json` pins every file's sha256:

- the two v3 npz files, and v2's (`supersedes`; v2's own stats.json is in git history at 75cae68);
- the local line manifests: `manifests/lines_v3.jsonl` (crop boxes and provenance, no text) and
  `lines_full_v3.jsonl` (the same rows with labels);
- `labels.jsonl` and the review files the precision measurement rests on.

## Versions

| | v1 (untagged) | v2 | v3 |
|---|---|---|---|
| ink npz sha256 | `eed888a6f06034ddd99b4b610c9867ff2366b19f77b548cbdebe3c78c5a6e9d8` | `e11467b2f81d5437b966fbe9faa9cf5d8e6c9912ccf76f97ee1747e815f7163d` | `8ae031e3fee872febe2ebf600ebc07971cf1abbebeef8496662aec608f8a8970` |
| GRAD npz sha256 | `f2a3b7d33b2a8678f047493cb962d48c5f923c4809296c0edd319d162363351f` | `1b5419dcd22091bc63d82c19a225283a0fcf7dc4b0bb45df448b8454e9c7135f` | `88ba60a439262ae588bb8de4380535adc0d4967fa67fb6d2c8ff9c2fbd7489fa` |
| lines / VNs | 43,264 / 9,140 | 43,228 / 9,137 | 101,350 / 14,461 |
| ink / GRAD crops | 71,219 / 53,626 | 71,153 / 53,566 | 170,387 / 116,994 |

The audit of PR #57 found three CLANNAD fan works in v1:

- v11057 *NETANNAD* and v42392 *YAMANNAD*, which VNDB links to CLANNAD (v4) by a `char` relation;
- v16902 *杏アフターアフター*, a sequel of v16901, which is on the CLANNAD family list.

Between them, 36 lines on 11 screenshots, whose labels name CLANNAD characters. The 11-VN family
list did not cover them. `shots.reserved_vns` now also reserves:

- CLANNAD's VNDB relations of every kind;
- the FAMILY relations of the family list (`clannad-relation`, 9 VNs in the 2026-10-12 dump).

v2 is v1 minus every crop whose meta `vn` is one of the three. The two files were compared crop for
crop: every other crop is byte-identical and in the same order, with the same label and meta
(`logs/v2_vs_v1.json`). No line of these VNs was in the precision sample, so the measurement holds
for both versions.

## Version 3

v3 is Stage A of the v3 plan (`$R/v3_plan.md`, where `R=$HERD/shared/artifacts/realplus`). It reviews the
candidate lines v2 had read but left undecided, in the plan's tiers t1 and t2 (t1 round-robin over VNs, then t2).
It completes the 2,093 edge checks v2 withdrew at its freeze (`review/edge/b0020`-`b0030`) and adds the reclaim
tier (below). v2's screenshots got no new read. v2's files, labels and finished batches were reused read-only:
pass 1 b0001-b0017, pass 2, edge b0001-b0019 and the blind re-check. Everything v3 made is under
`R3=$HERD/shared/artifacts/realplus-v3`.

| v3 pool | lines |
|---|---:|
| v2's lines (of 43,228) | 42,067 |
| new auto lines of the reclaim tier | 20,245 |
| new auto lines whose withdrawn v2 edge check v3 completed | 1,709 |
| new reviewed lines (v3 pass 1, the agreement rule) | 37,329 |
| **total** | **101,350** |

Of the 37,329 new reviewed lines, Luna read the same text as Claude on 30,837. On the other 6,492 Luna skipped or
was unsure (cut 5,185, ui 1,296, unsure 8, bad 3), and a second, blind Claude look read Claude's text (rule 3
below).

v3 is not a strict superset of v2: 1,161 of v2's lines are not in it. Most of them, 1,150, are cut by the repeat
caps, because new lines now share their labels: 1,052 labels already have 8 lines in v3 (the overall cap) and 98
already have 2 in their VN (the per-VN cap). 7 more leave through other build rules. The last 4 were dropped by
v3's third looks: 3 unsure, 1 label wrong.

### The reclaim tier

The owner's FINAL reservation policy (2026-10-14 13:25Z, `docs/reservations.md`) made everything except the
CLANNAD family, vita-test and psp-test (with their shared releases) training data. reserve-policy then fetched the
20,867 VNDB screenshots of VNs that the old rules had reserved (`$HERD/shared/artifacts/reserve-policy/realplus/
manifests/shots.jsonl`).

- Before the read, `shots.dedupe` removed 287 of them: 283 pixel copies of eval's candidate screenshots and
  4 copies within the tier.
- devnative/read.py read the other 20,580 with v2's frozen pipeline (`$R3/reclaim/reads/reads.jsonl`).
- After the read, three groups were taken out: 18 screenshots of three VNs that became development-sample
  partners, 12 PSP-resolution images that the Vita-scale cutter does not handle, and 2 copies on reserve-policy's
  lists (`$R3/reclaim/manifests/dups.json`, 301 entries in all).

At the read the reclaim tier had 66,044 candidate lines (auto 23,665, t1 21,781, t2 2,461, t3 18,137). After the
filters above, the plan holds t1 21,751, t2 2,458 and t3 18,098 of them, and the yields table uses those. They were
planned together with v2's and went through the same batches, rule and build. 32,687 of v3's lines come from the
tier: 20,245 auto and 12,442 reviewed. Its yields are listed separately below.

### The agreement rule

Every new training label needs two independent reviewers to read the same text (the owner's rule of 2026-10-13
05:40Z). The exact rule was fixed in `$R3/AGREEMENT.md` before the first v3 batch was reviewed. In short:

- A Claude reviewer decides each pass-1 sheet as in v2 (`reviewer_kind: "assistant"`). The model differed by batch.
  The table below is from the helpers' kiro session metadata. `$R3/review/helpers.jsonl` agrees only in part: it
  names claude-fable-5.1 for 74 pass-1 and 9 confirm batches, but 4 of those 74 ran Opus after spawn.sh began
  enforcing it, and 16 Fable pass-1 batches and 2 Fable confirm batches have no model recorded.

  | reviewer | pass 1 (351) | confirm looks (55) | edge checks (92) |
  |---|---|---|---|
  | claude-fable-5.1 review helpers | 86: b0018-b0033, b0071-b0140 (17,200 lines; 9,919 of the pool's new reviewed lines) | 11: b0001-b0002, b0007-b0015 | none |
  | claude-opus-5.5 review helpers | 59: b0034-b0070, b0141-b0158, b0166, b0272-b0273, b0294 | 8 | 24: b0031-b0053, b0064 |
  | claude-opus-5.5, the agent's own sub-sessions | 206: b0159-b0165, b0167-b0271, b0274-b0293, b0295-b0368 | 36 | 68 |

  The helpers of b0018-b0033 ran Fable because they were spawned on 2026-10-13 from 07:04Z, before the owner's 09:15Z
  update returned new agents to Opus. b0071-b0140 ran Fable because the helpers were pinned to claude-fable-5.1 at
  15:30Z that day (`REALPLUS_HELPER_MODEL`), after and against that update. The pin lasted until spawn.sh enforced
  Opus for every agent on 2026-10-14 at 07:48Z.
  No quality difference is visible: in the precision check, new t1/t2 lines from Fable batches agree 75 / 83 as
  measured and 82 / 83 are right after the third looks, against 289 / 317 and 310 / 317 for the other batches. The
  own sub-sessions follow `$R3/prompts/self_*.md`, calibrated on copies of helper-decided batches. The precision
  check (7 batches) and the third looks were the agent's own claude-opus-5.5 sessions.
- Codex gpt-6-luna votes on the same sheet in a fresh session (`tools/codex_sheet.sh`, one call per sheet, public
  VNDB crops only; `reviewer_kind: "codex:gpt-6-luna"`). It ran at most 6 calls at a time, the owner's limit, and at
  most 4 from 2026-10-13 10:35Z. It sees the README, the sheet and the candidate texts, never the Claude decision.
- The two readings are compared by `teacher/vote.key`, eval's scoring normalisation. A difference only in an edge
  「」『』。、！？ is a disagreement.
- The outcomes:
  - agree: the line goes on to devnative's training rule;
  - Luna reads another text: the line is dropped (`luna-disagree`);
  - Luna skips (ui / cut / bad) or is unsure: a second, blind Claude look on the final crop decides (`review/
    confirm`, candidates shuffled and unattributed, `decide.py --confirm`);
  - Claude skips or is unsure: the line is dropped as in v2, whatever Luna says.
- A label only one engine family read also needs the confirm look (v2's pass 2).
- A line still without a Luna vote at the freeze is dropped. Sheets whose vote missed entries were run up to 3
  times (`luna.py run --retries 3` for the last round). 9 lines stayed without a vote.
- Amendment 1 (after the first decided batch): a Luna line with both a reading and a skip counts as the skip.

Luna's agreement with Claude (`luna.py stats`; `$R3/luna/agreement_first10.json`, `agreement_all.json`):

| | first 10 batches (b0018-b0027) | all 351 v3 batches (b0018-b0368) |
|---|---:|---:|
| lines | 2,000 | 70,155 |
| Claude read (pick or text) | 1,327 | 42,276 |
| Luna read the same key | 969 (73.0%, Wilson 95% 70.6-75.3%) | 33,151 (78.4%, 78.0-78.8%) |
| Luna read another key (dropped); of which only an edge character differs | 53; 23 | 1,872; 672 |
| Luna skipped or unsure (confirm look) | 304 (cut 222, ui 81, unsure 1) | 7,244 (cut 5,651, ui 1,581, unsure 9, bad 3) |
| no Luna vote (dropped) | 1 | 9 |
| **agreement where both read** | **969 / 1,022 (94.8%)** | **33,151 / 35,023 (94.65%)** |

Most of Luna's skips were false alarms against the pipeline's conventions: it called wrapped screen lines "cut"
and name plates "ui". So `pool.WRAP_NOTE`, which restates those conventions, was appended to the READMEs of
b0022-b0029 on 2026-10-13 at 07:30Z, and every batch from b0038 on carries it. b0018-b0021 and b0030-b0037 do not.
Luna's skip-or-unsure share of the lines Claude read is 24.6% on the 12 batches without the note (395 / 1,608) and
16.8% on those with it (6,849 / 40,668). Luna also read a text on 10,242 lines Claude skipped; under the rule these
never enter the pool.

The table counts the vote each line was decided by, which is not always Luna's first vote:

- b0022-b0029 were voted on again under the new README; their first votes are kept in `$R3/luna/pass1_prewrap/`.
- A sheet whose vote missed entries was run again in full (304 sheets; with b0022-b0029's 166 re-voted sheets, 465
  have more than one vote file), and the newest run wins for every entry, not only the missing ones
  (`luna.luna_votes`).

With each line's first complete vote, the first 10 batches give 950 / 1,327 (71.6%) of Claude's readings and
950 / 1,000 (95.0%) where both read. In the pool, 18 lines had a first vote that disagreed with Claude: 11 later
agreed, and 7 later skipped and were confirmed by a confirm look. Under rule 2, with no re-vote, they would have been
dropped. 146 more lines had a first-vote skip that later became agree, so they did not need a confirm look, and 100
went the other way, from agree to a skip and a confirm look. A future rule should keep each entry's first complete
vote.

Two Luna records, `{"pick": 1, "cut": "cut"}` with the key `cut` instead of `skip`, parse as plain picks and count
as agreements. By amendment 1's intent they are skips. Both lines are in the pool without a confirm look:
v26337_127493_1 (b0328) and v17793_138107_1 (b0321).

The confirm looks covered 10,388 lines and agreed with the first reading on 10,026 (96.5%). By the reason a line
was sent:

| reason sent | agreed |
|---|---:|
| Luna skip "cut" | 5,472 / 5,567 |
| Luna skip "ui" | 1,437 / 1,555 |
| Luna unsure | 9 / 9 |
| Luna skip "bad" | 3 / 3 |
| single-family labels | 3,105 / 3,254 |

The other 362 were dropped (another reading 87, unsure 129, skip 146). These counts come from the `confirm` field of
`labels_v3.jsonl`. The confirm batches themselves hold 10,392 decisions, 10,030 agreeing. The 4 extra lines agreed,
but the precision check's third looks dropped them later. The single-family row includes 5 lines that were first
sent for a Luna skip that a later re-vote turned into agree.

### Yields

v3 pass 1 (b0018-b0368, 70,155 lines; "kept" = kept after both reviewers and the confirm looks; "in the pool" =
after the edge check, the crop rules and the repeat caps):

| source | tier | lines reviewed | Claude read | kept | in the pool |
|---|---|---:|---:|---:|---:|
| v2's candidates | t1 | 40,921 | 25,568 | 24,423 | 22,857 |
| v2's candidates | t2 | 5,012 | 2,642 | 2,140 | 2,030 |
| reclaim tier | t1 | 21,751 | 12,739 | 12,204 | 11,421 |
| reclaim tier | t2 | 2,458 | 1,316 | 1,075 | 1,021 |
| lines the plan dropped later (reserved VNs, PSP-scale reads) | | 13 | 11 | 10 | 0 |
| **all** | | **70,155** | **42,276** | **39,852** | **37,329** |

The pool yield per reviewed line is 54.7% in t1 (34,278 / 62,672) and 40.8% in t2 (3,051 / 7,470).

Edge checks (Claude only, `$R3/prompts/self_edge.md` for the agent's own sessions):

| | lines | pick | text | unsure | cut | in the pool |
|---|---:|---:|---:|---:|---:|---:|
| v2's withdrawn b0020-b0030 (all auto lines) | 2,093 | 1,964 | 9 | 27 | 93 | 1,709 |
| v3's own b0031-b0111 (new labels and flagged reclaim auto lines) | 15,457 | 14,399 | 21 | 144 | 893 | 13,739 |

### Reservations and hygiene

v3 is built under the FINAL policy (`pool.POLICY = "final-2026-10-14"`, recorded in build_stats and stats.json).
One reservation, `shots.reserved_vns` over `eval/manifests/reserved.json`, serves the plan, the build and
`realplus/hygiene.py`. On the 2026-10-12 dump it reserves 187 VNs. 71 are test VNs: CLANNAD 7, its family 6, its
VNDB relations 8 and their shared releases 3, and the 28 VNs of the vita-test and psp-test games with 19 shared
releases. The other 116 are the development sample (100 VNs and 16 shared-release partners), which training must
also skip.

- `realplus/hygiene.py` reran from the raw inputs before the build (labels) and after it (lines): HYGIENE_OK both
  times, no blockers, 0 pixel-hash hits against the 629 protected screenshots (`$R3/logs/hygiene.json`). The
  build dropped 60 lines of reserved VNs (`excluded-late`).
- Main's `eval/tools/reserve_check.py --require-hashes --pixels` (by VN, release, screenshot id, file and pixel
  sha256) gives RESERVE_OK. Every row was hash-checked: ink 170,387 / 170,387, GRAD 116,994 / 116,994, lines
  101,350 / 101,350 (`$R3/logs/reserve_check_v3.json`).
- Thumbnail leak checks (`eval/tools/leakcheck.py`, correlation >= 0.95, run before the first v3 build): v2's
  screenshots against vita-test / psp-test, 2 cross-VN pairs (dropped through `leak_drop.json`); devnative, 0;
  teacher_val, 1 pair with unrelated text. The reclaim tier came later. Its leak check was missed when the tier was
  added, and it ran only after the v3 build (2026-10-16 03:50Z): its 20,867 screenshots against the 157 vita-test
  and psp-test screenshots gave 0 pairs, with the closest test screenshot at 0.90-0.95
  (`$R3/logs/leakcheck_reclaim.log`). So no build changes. The exact checks (screenshot ids and file and pixel
  hashes) covered the tier in every build that held it.
- Codex exposure: 9 lines of three CLANNAD-relation fan works (v11057 *NETANNAD*, v42392 *YAMANNAD*, v16902
  *杏アフターアフター*; public VNDB screenshots) sat on 8 pass-1 batches (b0039, b0050, b0059, b0084, b0093,
  b0101, b0132, b0139). Those batches were made between 2026-10-13 09:05Z and 2026-10-14 06:52Z, and Luna voted on
  each within minutes. These VNs were already reserved, by `clannad-relation`, which came in with PR #57 (merged
  2026-10-13 06:32Z, before the first v3 batch), and every build excluded them. The gap was in v3's plan: until the
  hand filter of 2026-10-14 19:55Z it applied no reservation, so lines of reserved VNs could reach a sheet.
  `pool.py plan` has applied the build's reservation (`build_exclusions`) since 2026-10-15. The build drops these 9
  lines and they are never trained on, but CLANNAD-family crops did reach Codex. A scan of every v3 pass-1 sheet
  finds no other test-reserved line. One development-sample line was there (b0159), and the build drops it too.

### Precision (independent transcription)

The check was pre-registered in `pool.py` (`CHECK_STRATA`, seed `realplus-v3-check-20261013`) on 2026-10-13 at
07:20Z. It was amended once, on 2026-10-15 at 07:25Z, still about 19 hours before the first check sheet (2026-10-16
02:12Z); the comment above `CHECK_STRATA` records the amendment. It added the stratum `new:auto` (200) for the
reclaim tier's auto lines, which would otherwise have filled `new:edge`. The sample was drawn once from v3pre
(`review/check_spec_v3pre.json`). v3pre is the build with all of Stage A decided, before the check's third looks.

- Each entry is one of the line's crops in the npz, chosen at random (vita_ink 313, vita_grad 201, psp_ink 180,
  psp_grad 106). It is framed on its own screen at its own scale, with no candidate shown.
- Seven fresh Claude sessions, one per check batch (120 entries; the last 80), transcribed the frames. They saw no
  label, candidate or other batch.
- A skip or unsure counts as a disagreement.
- The stratum `new:edge` (100 planned) was empty: every line of v2's withdrawn edge checks is an auto line, so those
  lines fall in `new:auto`. That stratum's 200 entries are 182 reclaim auto lines and 18 such lines, so the withdrawn
  edge checks got 18 entries rather than the 100 the first registration meant for them, and the sample is 800 lines,
  not 900.

Third looks: one fresh Claude session looked at the 58 disagreements (`pool.py look3`, `review/look3/b0001`). Each
entry showed the label and the check's reading, shuffled and unattributed, or only the label when the check typed
nothing. A line stays only if this look reads the label: 44 did; 4 read another text, 9 were unsure and 1 was cut,
and those 14 are dropped (`review/resolutions_v3.jsonl`). v3 is v3pre minus those 14 lines. In crops the difference
is not quite a subtraction: dropping a line frees its neighbours' GRAD regions, so 5 lines on the same screenshots
changed (6 new GRAD crops on 4 lines, and 1 GRAD crop re-cut 1 px off). Ink is v3pre's minus 23 crops, with no new
or changed pixels; GRAD is minus 20, plus 6 new, with 1 changed. The `looks` field rose by 1 on the 34 reviewed
lines a third look kept, but not on the 10 auto lines it kept (`apply_auto_resolutions` does not count the look);
this is metadata only and changes no label. Training note, from the audit of PR #89: train2's
`load_shard` drops 2 of v3's 287,381 crops on its CTC length check, both PSP crops of long repeated-kana runs
(v7762_21992_1 psp_ink, v2593_17776_7 psp_grad).

| | lines | agree as measured | right after third looks | wrong | unsettled |
|---|---:|---:|---:|---:|---:|
| **all** | 800 | **742 (92.75%, 90.7-94.4%)** | **786 (98.25%, 97.1-99.0%)** | 4 (0.2-1.3%) | 10 |
| new: t1 | 300 | 277 | 295 | 1 | 4 |
| new: t2 | 100 | 87 | 97 | 1 | 2 |
| new: auto | 200 | 194 | 198 | 1 | 1 |
| v2: auto | 120 | 111 | 117 | 1 | 2 |
| v2: reviewed | 80 | 73 | 79 | 0 | 1 |
| Vita ink | 313 | 302 | 309 | 3 | 1 |
| Vita GRAD | 201 | 192 | 197 | 1 | 3 |
| PSP ink | 180 | 159 | 179 | 0 | 1 |
| PSP GRAD | 106 | 89 | 101 | 0 | 5 |

(Percentages in brackets: Wilson 95% intervals; `review/check_report_v3pre.json`, after the looks in
stats.json `check.after_third_looks`.)

**What it measures.**

- The first read is an independent transcription: no candidate is shown, unlike v2's confirmation re-check.
- The third looks show the label, so they are a confirmation and inherit its anchoring. Without them, 742 / 800
  is the conservative number.
- Most disagreements were the transcriber's caution on the small PSP crops. 34 of the 38 PSP disagreements were
  unsure or skip (half-scale text; a PSP crop needs only 10 px of ink). After the looks PSP stands at 280 / 286
  and Vita at 506 / 514.
- Every reader is Claude, recorded as reviewer_kind "assistant": the same model family as pass 1, in separate
  sessions with no shared context.
- The 14 dropped lines leave only the sample. The rest of the pool carries the measured rate.

### Decisions

- Stage B (t3, 53,841 lines with three or more candidates, none read by both families) was not started. The task
  makes it conditional on time, and Stage A was fully decided only on 2026-10-16 at 01:56Z, about four hours before
  VOCR5 pins its fine-tune pools (about 06:00Z). t2's pool yield (40.8%) was already well below t1's (54.7%).
- Interim snapshots (v3i to v3i9, then v3pre) were built along the way from the lines decided by then, for the
  VOCR4 and VOCR5 fine-tunes. Their files are kept. From v3i5 on, each one passed hygiene.py and main's
  reserve_check with `--require-hashes --pixels` before it was announced.

## Sources and selection

The listing comes from VNDB's daily database dump (`vndb-db-2026-10-12.tar.zst`, sha256
`c6ada5e33d3163d1de97fc7e6e24b91048506a2a0979172c5205f439376ef4aa`, 190,763,803 bytes, dump
timestamp 2026-10-12 08:00:26+00), not thousands of API pages. Screenshots come from VNDB's image
server over HTTPS (`https://t.vndb.org/sf/NN/ID.jpg`), at most two requests at a time with a 0.5 s
delay, under a descriptive User-Agent.

`realplus/shots.py select` keeps a screenshot when:

- its VN is a Japanese original, and its release is Japanese-only or Japanese plus English, so no
  Chinese or Korean text is on screen;
- its VNDB flag averages are sexual < 0.3 and violence < 0.5, the repository's filters (realdata,
  data/fetch_assets.py);
- it is one of realdata's three kinds:
  - vita: a psv release at 960x544;
  - psp: a psp release at 480x272;
  - hires: any other release's shot, aspect 1.25 to 1.85, at least 640x400. It is later scaled to
    fit 960x544 and 480x272.
- its VN is on the training side (next section);
- the screenshot is in no existing pool or evaluation set, by VNDB image id at selection and by
  decoded pixels after the download.

Every training-side native Vita and PSP screenshot was already used. Each one sits in realdata's or
teacher's pools, in eval's sets, in devnative's cohort, or in a release shared with a reserved VN. So
the pool is all hires: PC and home-console screenshots, scaled the way teacher scales its hires kind
(`cv2.INTER_AREA` to fit 960x544, then 480x272 for the PSP crops).

Selected: 43,198 screenshots from 10,436 VNs. The drops at selection were:

| reason | screenshots |
|---|---:|
| VN not a Japanese original | 59,199 |
| rating (sexual >= 0.3 or violence >= 0.5) | 39,558 |
| VN is a test VN (`synth/common.is_test_vn`) | 18,758 |
| VN in the validation bucket (15 <= md5(vn) % 100 < 25) | 12,587 |
| VN shares a release with a reserved VN | 11,017 |
| release language other than ja / ja+en | 4,112 |
| already in a pool or set (realdata, teacher, synth2/3, data/vndb, linefind, eval candidates, fresh-synthetic) | 4,458 |
| size or aspect outside every kind | 1,789 |
| development VN | 1,732 |
| VN of `eval/manifests/test_games.json` "all" | 1,426 |
| family of CLANNAD, a fresh native title or a development VN | 1,081 |
| fresh native textbox title | 34 |
| relation of a fresh native textbox title | 27 |
| CLANNAD | 23 |

## Exclusions

These sets are never used for anything: vita-test, psp-test, the fresh native textbox test,
CLANNAD and the CLANNAD long-run references. A VN is off limits (`shots.reserved_vns`, first match
wins) when it is:

- CLANNAD (`v4`, or "clannad" / "クラナド" in a title), and CLANNAD's direct family as the conductor
  listed it (`shots.CLANNAD_FAMILY`): v12, v4060, v28752, v27548, v10058, v29951, v23354, v16901,
  v5698, v7983, v54298. All 11 were already reserved, by title or by relation; none was selected,
  downloaded or read;
- related to CLANNAD by any VNDB relation kind, or by a FAMILY relation to one of the 11. In the
  2026-10-12 dump that is 9 VNs:
  - v869, v8324, v11057, v16902, v21002, v25142, v42392 and v64301 get the reason
    `clannad-relation`;
  - v64642 *QUIZ de CLANNAD* is reserved first as `clannad`, by its title.

  This rule was added after the audit (see "Versions"); v1 still holds three of these VNs;
- in `eval/manifests/test_games.json` "all";
- a test VN (`synth/common.is_test_vn`);
- a fresh native textbox title:
  - PSP v14909, v25459, v29454; Vita v21289, v21976, v20519; reserves v16509, v19140;
  - the explicit list in `shots.FRESH_NATIVE`, plus every VN id in eval's fresh-native files;
- a development VN:
  - synth3's `exclude_dev_vns.txt`;
  - the general-VN and CLANNAD development manifests;
  - devnative's cohort.
- in the validation bucket;
- related to a fresh native title by any VNDB relation kind (v15053, v52836, v25385, v28146,
  v57035);
- family of CLANNAD or of a development VN (alternative version, original, fan disc, parent, side
  story, sequel, prequel, same series);
- sharing a release with any of the above. For test and validation VNs, this applies only when the
  release holds two VNs or the partner is family.

At selection, 9,816 test VNs, 6,623 validation VNs, 172 test-set VNs, 189 development VNs, 8 fresh
native titles, 4 more fresh-native relations, 7 CLANNAD entries, 146 family VNs and 1,840
shared-release VNs were reserved. With the later CLANNAD rules, the same dump gives 6 clannad-family,
140 family and 8 clannad-relation VNs. Separately, 8,459 screenshot ids of existing pools and sets
were reserved, along with eval's 31 test capture sessions.

The fresh native textbox titles were added to the exclusions on 2026-10-12, before any labelling.
None of the 8 titles or their 5 relations was ever selected, and no screenshot of them was
downloaded. Nothing had to be removed. `dedupe` also drops any screenshot whose file sha256 or
decoded RGB pixels match an image recorded by the fresh native test. At build time, the reserved
VNs are recomputed from the current inputs, and screenshots that dedupe marked as copies are left
out.

After the download, `shots.py dedupe` compares decoded pixels (devnative's `pixel_sha256`). It
checks against every local copy of an existing pool or set (`screenshot_dirs`, hashed by
`pool_pixel_hashes`) and against earlier realplus screenshots.
No realplus screenshot matched a pool or set image (7,845 pixel hashes) or an image the fresh
native test records. Two pairs of realplus screenshots were pixel-identical copies under different
image ids; for each pair, the copy not yet read was dropped.

`eval/tools/leakcheck.py` compares 32x18 grey thumbnails and flags pairs that correlate at 0.95 or
more. It ran in three passes:

- Against vita-test and psp-test (157 screenshots): it found two cross-VN pairs at 0.955, both with
  the same vita-test screenshot (v20326_144968). Both pool screenshots were dropped
  (`manifests/leak_drop.json`), which removed one kept line.
- Against devnative's 346 cohort screenshots: it found no pair.
- Against teacher_val's 646 screenshots (after the audit): it found one cross-VN pair at 0.954
  (v17012_144281 ~ v56110_242656). The audit compared the two by eye: their text is unrelated, so
  they are not the same frame and nothing was removed.

The vita-test / psp-test and devnative passes ran after the download and before the first build
(20:57-21:13Z), and every build applies their `leak_drop.json`. The teacher_val pass ran later
(02:10Z, after the v2 build). It removed nothing, so no build depends on it.

For the vita-test / psp-test pass, eval's test screenshots were fetched into the artifacts directory
with `eval/fetch.py` (sha256-checked against the manifests). Nothing was trained, tuned or scored on
them.

The fresh native test's frames are not stored locally. They are covered by VN id, by relation and by
the hashes the test records.

## Pipeline

The labelling pipeline is devnative's (`devnative/README.md`), reused unchanged:

- read and sheets as frozen at 35a86ca;
- the GRAD containment check from 0cb084e;
- final crops, the edge check and looks from f523e30 (branch head f5a1dbd).

`realplus/pool.py` drives it over the pool:

1. `prep` writes read.py's input (43,196 screenshots; the two pixel copies are not read).
   `devnative/read.py` runs MeikiOCR's detector and four readings per line: MeikiOCR on its own
   box, MeikiOCR on a wider view, PP-OCRv6 small on that view, and PP-OCRv6 small on the crop. It
   runs in a network namespace with no network, through `heavy.sh`.
2. `plan` sorts the candidate lines (devnative's `is_candidate`: Japanese, 10-72 px of ink, not at
   the screen edge) into tiers:
   - auto: all four readings agree;
   - t1: the most supported candidate is read by both engine families;
   - t2: two candidates;
   - t3: three or more candidates.
   A review line is skipped when every candidate is a label its VN already has twice among the auto
   lines; the repeat cap would drop it anyway.
3. `batches` makes disjoint pass-1 batches of 200 lines with devnative's contact sheets. The order
   is by tier, then round-robin over VNs.
4. Claude reviews each batch: one review-rpN helper agent per batch, at most 6 at a time (8 after
   the owner's 19:05Z update).
   - The helper picks a candidate, types the text, marks the line unsure, or skips it as ui, cut or
     bad.
   - It saves its decisions after every sheet.
   - Every decision has `reviewer_kind: "assistant"`.
5. `final` applies `decide.py final --mode train --auto-consensus`:
   - a label needs at least one engine family after review;
   - a changed label is dropped;
   - pass 2 is optional and must agree when given;
   - lines that all four readings agree on are kept without review.
   Pass 2 (`decide.py queue`, then `sheets.py make --mode pass2`) re-read, blind, the 66 labels
   that only one engine family supported when the queue was made (20:59Z):
   - 60 agreed and were kept;
   - 5 disagreed and were dropped;
   - 1 (v2219_37672_5) was in the end skipped as cut by pass 1 itself: its batch's final
     decisions were written at 21:24Z, after the queue. Pass 2 also called it cut.
6. `edge-boxes`, `edge-batches` and `edge-apply` run devnative's edge check. See "Crops".
7. `build` cuts the crops and writes the npz files. `blind` draws the re-check sample, and
   `blind-score` scores it. `publish` writes the committed manifests.

## Crops

Each kept line yields up to four grey crops, in console polarity:

- a Vita crop from the screenshot scaled to fit 960x544;
- a PSP crop from it scaled to fit 480x272, when the line has at least 10 px of ink at that scale;
- each of these as an ink crop and, where the GRAD check passes, as a GRAD crop.

A dark-on-light crop is inverted, and meta `dark` / `inverted` record it.

- The ink crop is eval's and teacher's crop: the ink box with furigana plus 4 px (Vita) or 2 px
  (PSP). It is then widened left and right to the boxes MeikiOCR gives the labelled characters
  (`lines.final_crop`), so every labelled glyph is inside. The reads store the padded box under
  `crop` and `crop_box` pads it again, so devnative's own crops and contact-sheet frames are
  ink + 8 / 4 px. realplus cuts the single-padded crop (`pool.read_line`); this was reported on
  PR #52.
- An ink crop is dropped when a quarter or more of another detected text box lies inside it. This
  is common with the tight line spacing of 640x480 PC games.
- Edge check: devnative's `edgecheck.flags` runs on the frame the reviewers saw. It flags a line
  when ink continues past that frame, or when its labelled characters lie beyond it. A label that
  no MeikiOCR reading spells cannot be widened, so it is flagged too. A flagged line is kept only
  when an edge-check batch (contact sheets in mode edge, framed on the final crop) picked its
  label. Otherwise it is dropped as `edge-unchecked`.
- The GRAD crop is the box the console's GRAD finder returns for the line's dialogue region
  (`lines.grad_assign`). It is used only when:
  - it holds the whole line and no other detected text (status "ok");
  - it does not run beyond the frame the reviewers saw (devnative's `grad_default`), at either scale;
  - it covers the final crop's labelled extent (`build.covers`).
- Labels with characters outside `data/vocab.txt` are dropped. Identical labels are kept at most
  twice per VN (devnative) and eight times overall (teacher).

The npz files use teacher_train's shard format (`teacher/render.py`: data, offsets, shapes, labels,
meta), one file per crop kind, with Vita and PSP crops together (meta `target`):

- `realplus_train_ink_v2.npz` (v1: `realplus_train_ink.npz`)
- `realplus_train_grad_v2.npz` (v1: `realplus_train_grad.npz`)

Meta per crop:

- provenance: real, pool, id, vn, session, release, kind "hires", target, crop, box;
- polarity: dark, inverted, pol;
- review: how, support, readers, decision, looks, reviewer_kind, name;
- characters: chars, chars_match;
- geometry: src_dims, screen.

Build counts (v2). The edge check covered 3,800 flagged lines: 3,643 passed and 157 failed (111
cut, 36 bad, 9 other text, 1 unsure). The build counts 110 edge-cut lines, not 111: one of them had
already been dropped by pass 2. Another 2,091 flagged lines were not checked and were left out
(2,093 in v1; two belong to the three dropped VNs).

Other lines left out at the build:

| reason | lines |
|---|---:|
| identical label already kept twice in the VN, or 8 times overall | 4,959 |
| no crop left after the per-crop rules | 375 |
| VN or screenshot excluded at build: the three CLANNAD fan works (39), the leak check (1) | 40 |
| label character outside `data/vocab.txt` | 15 |

Per-crop drops. These are counted while cutting, before the repeat cap removes 4,959 lines, so they
include lines that are not in the pool:

- 18,026 lines are below 10 px of ink at PSP scale, so they have no PSP crop; in the final pool,
  15,174 lines have no PSP crop.
- 507 Vita and 472 PSP ink crops take in a quarter of a neighbouring line.
- GRAD crops are dropped when the box:
  - is missing because the finder returned no box for the line (5,542 Vita, 3,780 PSP);
  - holds a quarter or more of other detected text (2,371, 3,956);
  - runs beyond the reviewed frame (2,644, 1,713);
  - holds only part of the line (1,418, 197);
  - cuts the labelled extent (774, 20).

Crop polarity:

- 14.6% of the ink crops and 13.6% of the GRAD crops were dark-on-light, and are stored inverted.
- In the Vita ink crops: 36,621 bright-on-dark and 6,482 dark-on-light.

All 71,153 ink crops load through `train/train2.py`'s `load_shard` with the repository vocabulary
(none dropped). With FL10's 3,500-character vocabulary, the audit found 455 ink and 384 GRAD crops
in v1 that hold a character outside it.

The audit also noted, for information:
- GRAD crops can hold slivers of the neighbouring line, below the 25% "mixed" rule, as console finder
  crops do.
- The four crops of one line can differ in inversion, because `line_dark` runs per crop, as on the
  console.

## Precision

This section measures v2, with the blind confirmation re-check. v3's independent transcription check is under
"Version 3".

**What it measures.** This is a confirmation re-check by fresh agents, not an independent
transcription:

- 481 of the 500 scored lines had exactly one candidate on the sheet: all scored auto lines, whose
  four readings agree, so the only candidate is the label. For these lines the reviewer accepted or
  rejected the label instead of reading the line from scratch, and anchoring on a shown answer
  makes such checks optimistic.
- Only the Vita ink crops were checked. The PSP ink crops and all GRAD crops (81,616 of 124,719
  crops in v2, 65%) take their labels from the same line through the geometry rules above.
- The audit's own independent read gave 40 of 40 lines agreeing: 40 random pool lines read blind
  with no candidate shown, and all 117 of their stored crops show the label. That is consistent
  with the re-check, but 40 lines are too few to narrow its interval (Wilson 91.2-100%).

The re-check sampled kept lines at random (seed 20261013) from the 50,874 kept at the freeze, in a
pre-registered order. Each line was framed on the final Vita ink crop the npz holds. It was shown
as devnative's blind sheets show lines: candidates shuffled and unattributed, engine support
hidden. Separate review-rpN helper agents decided the lines (rp32-35, rp43). None of them had
reviewed these lines before, although 93 of the drawn lines had been on earlier sheets for other
helpers.

Lines drawn and scored:

- 540 lines were drawn first. Some kept lines leave the pool at the build (an edge check, a
  per-crop rule, the repeat cap), so only 446 of them were in a preliminary build. That build lacked
  the last edge decisions; its agreement was 441 of 446. Both numbers come from the session log
  only: the preliminary build's files were overwritten by the final build (before the overwrite
  guard existed), so they cannot be re-derived from stored files.
- 70 more lines were then drawn from the same frozen set (seed `20261013:1`), continuing the order.
  The extension was drawn after the first 540 were decided. Its size was fixed by the shortfall
  before any of its lines was decided, and every one of its lines counts up to the 500th.
- The first 500 lines of the order that are in the final pool were scored with devnative's
  `decide.blind_agreement`:
  - a line agrees when the blind reading's scoring key equals the label's;
  - a skip or "unsure" counts as a disagreement.
- The first draw supplied 461 of the 500 lines and agreed on 456 (98.9%, Wilson 97.5-99.5%). The
  extension supplied 39 lines and agreed on all 39.
- The five lines dropped by the third looks below still count as in the pool, with their frozen
  labels (`pool.measured_pool`). So `blind-score` reproduces the measurement after the drops.

| lines | agree | rate | Wilson 95% |
|---|---:|---:|---|
| all | 495 / 500 | 99.0% | 97.7-99.6% |
| auto (all four readings; one candidate shown) | 476 / 481 | 99.0% | |
| reviewed, both engine families | 18 / 18 | 100% | |
| reviewed, one engine family | 1 / 1 | 100% | |

The audit re-derived the order, the decisions and the 495/500 from the raw files.

A third look (assistant, `review/resolutions.jsonl`) at the five disagreements found:

- two wrong labels, both on auto lines: an ellipsis run of three cells read as two, and 。 read
  where the screen has 、;
- one line whose first kana could be either of a hiragana / katakana pair, so the image does not
  settle it;
- two cut lines, where the crop holds exactly the label but the line goes on just outside it: a
  second ！ after a gap, and a thin 一 that ends at the crop's edge.

All five were dropped from the pool. The 99.0% above was measured before these drops.

The point estimate meets the 99% target. The 95% interval does not exclude a precision as low as
97.7%: 500 lines cannot show more than that.

devnative's `decide_line` returns before its resolution block when a line has no pass-1 decision.
So realplus applies third looks to auto lines itself (`pool.apply_auto_resolutions`), with the same
semantics.

## Limits

These limits were written for v2. v3 changes four of them:

- Review coverage: v3 decided all of t1 and t2, 70,155 more pass-1 lines, so 38.8% of v3's lines are reviewed
  (v2: 4.7%).
- The edge check: v2's 2,093 withdrawn lines are now checked.
- Precision: v3's check is an independent transcription, stratified over tiers, new and v2 lines, Vita and PSP,
  ink and GRAD.
- Reviewers: every new v3 label also had a Codex gpt-6-luna vote. Of the 37,329 in the pool, 30,837 had Luna read
  the same text; on the other 6,492 Luna skipped or was unsure, and a second, blind Claude look confirmed them.

The rest hold for v3 as well: hires-only crops (the reclaim tier's 12 PSP-resolution reads are excluded), text of
every kind and not only dialogue, devnative's edge rule, and reviews by AI models, never a human.

- All crops are hires. Every crop comes from a PC or home-console screenshot scaled down to the
  console screen; no training-side native Vita or PSP screenshot was left. Downscaled text is
  smoother than native console rendering, so the pool complements the native pools and synth3's
  console rendering rather than replacing them.
- Most lines were kept by engine agreement. 95.3% of the lines are lines all four readings
  agree on, kept without review. These are text the engines already read well.
  - A misreading both engines share survives. The confirmation re-check found two such labels in
    481 auto lines (see Precision).
  - Hard styles are under-represented: stylised fonts, coloured or outlined text, low contrast.
    Their lines mostly go to review, and only 17 review batches were reviewed.
- Review coverage is small. The four readings did not all agree on 85,109 candidate lines; pass 1
  reviewed 3,400 of them (17 batches). The other 81,709 are undecided and not in the pool. Of the
  3,400 pass-1 lines, 2,117 kept a label (62.3%) and 2,046 are in the pool (60.2%). More pass-1
  review would add lines at about that yield, plus an edge check for about one line in nine
  (5,893 of 50,874 kept labels were flagged).
  In this run, the herd's 12-agent cap left 3-5 review helpers running at a time.
- The pool holds every kind of text the engines read on screen, not only dialogue:
  - speaker names;
  - menus, buttons, status and location text, because auto lines are not reviewed and so UI is
    not skipped there. In the audit's random 40-line sample, 9 were such strings (Wilson
    12-38%). The most repeated labels are menu items at the 8x cap (戻る, はじめから, つづきから,
    オプション, セーブ, ゲーム終了);
  - 35% of the lines have at most four characters; most are names, some are UI strings or the ends
    of wrapped lines.
  Repeats are capped at 2 per VN and 8 overall, so no string appears more than 8 times. Meta `name`
  is set only on reviewed lines (374), never on auto lines.
- The edge check was not finished in v2. 2,091 flagged lines (v2) had no edge check at the freeze and
  were dropped; the 11 remaining edge batches were withdrawn. v3 decided them (b0020-b0030, 2,093 lines).
- devnative's edge rule misses some cut lines. `lines.ink_beyond` looks for ink only inside the
  line's read view, which ends a few pixels past the detected ink. It also needs more than
  max(2 px, 0.2 ink heights) of it. So it misses:
  - a second ！ about 10 px past the view;
  - a thin 一 that enters the frame by 1 px.
  The re-check found these two lines. A stricter rule (any ink within one ink height, outside the
  view too) flags both of them, but it would also flag 1,213 of 3,000 random unflagged lines
  (40.4%; `logs/strict_edge.log` and `logs/strict_edge.py`). That is too many to check, so the rule
  was left as devnative froze it.
- All reviews are Claude's, recorded as reviewer_kind "assistant", not a human's. This covers
  pass 1, pass 2, the edge check, the re-check and the third looks.
- The re-check uses helper agents of the same model as pass 1. They run in separate sessions with
  no shared context, and see shuffled, unattributed candidates. For 96% of the scored lines the
  only candidate was the label (see Precision).
- Precision was measured on 500 lines of the Vita ink crops. The Wilson 95% interval is
  97.7-99.6%.

## Reproduce

The commands below run at this commit, in the order the pool was made, and produce v2. The run
itself differed in three ways:

- **Builds.** It built four times without a tag, before the overwrite guard existed:

  | time | build | lines |
  |---|---|---:|
  | 22:22Z | preliminary | 41,757 |
  | 00:09Z | final | 43,269 |
  | 00:13Z | final, rebuilt | 43,269 |
  | 00:18Z | after the third looks: v1, which VOCR3 pinned | 43,264 |

  v2 was then built with `--tag v2` at 02:06Z. Below, the build used for scoring is tagged `pre`.
- **CLANNAD rule.** v1 was cut by the code before d8776fb, which added the `clannad-relation` rule;
  the cutting code itself has not changed since. At this commit `select` and every build apply that
  rule, so any build gives v2's content. To rebuild v1, check out 27b7a8d (d8776fb's parent) and
  build there without a tag.
- **Pass 2.** Its queue and sheets were made at 20:58-20:59Z from pass-1 batches b0001-b0017, as
  step 4 does. b0017's decisions were revised at 21:24Z (v2219_37672_5 became a skip "cut"), so
  re-running step 4 now queues 65 single-family lines, not 66.

```sh
HERD=/local/home/skerraut/herd-vita-ocr; R=$HERD/shared/artifacts/realplus; PY=.venv/bin/python
H="HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200"
# 1. screenshots
$PY realplus/shots.py dump --date 2026-10-12
$PY realplus/shots.py select
$PY realplus/shots.py fetch --workers 2 --delay 0.5
$PY realplus/shots.py dedupe
# 2. reads, pass-1 batches and review helpers (the read itself goes through heavy.sh)
$PY realplus/pool.py feed --loop --max 8 --kinds edge,pass1
# 3. leak checks against vita-test / psp-test and devnative, before any build
$PY eval/fetch.py --set all --data $R/evaldata                   # sha256-checked
env $H $PY eval/tools/leakcheck.py $R/shots --data $R/evaldata
env $H $PY eval/tools/leakcheck.py $R/shots --test $HERD/shared/artifacts/devnative/shots --keep-reserved
#   every pool screenshot of a pair at >= 0.95 -> $R/manifests/leak_drop.json (a list of shot keys)
# 4. pass 2 of the single-family labels (pass-1 batches b0001-b0017)
$PY devnative/decide.py --reads $R/reads/reads.jsonl queue --pass1 $R/review/pass1/b00{01..17} \
    --out $R/review/pass2_queue_all.json
#   keep the ids whose pass-1 label one engine family supports (decide.classify == "single"),
#   with their "extra" texts -> $R/review/pass2_queue.json, then
$PY devnative/sheets.py make --mode pass2 --reads $R/reads/reads.jsonl \
    --lines $R/review/pass2_queue.json --out $R/review/pass2 --chunk 150
# 5. the freeze
$PY realplus/pool.py final && cp $R/labels.jsonl $R/labels_frozen_2128.jsonl
# 6. edge check and the re-check sample
$PY realplus/pool.py edge-boxes && $PY realplus/pool.py edge-batches --n 99 --size 200 --min 1
$PY realplus/pool.py blind                                       # 540 lines
$PY realplus/pool.py feed --loop --ready 0 --max 8 --kinds blind,pass2,edge
$PY realplus/pool.py blind --extend 70 --size 70                 # after the shortfall showed
$PY realplus/pool.py edge-apply && $PY realplus/pool.py final
# 7. score, third looks, the pool
env $H $PY realplus/pool.py build --tag pre --procs 5
$PY realplus/pool.py blind-score --tag pre && cp $R/review/blind_report_pre.json $R/review/blind_report_measured.json
#   third looks at the disagreements -> $R/review/resolutions.jsonl, then
$PY realplus/pool.py final && env $H $PY realplus/pool.py build --tag v2 --procs 5
$PY realplus/pool.py blind-score --tag v2 && $PY realplus/pool.py publish --tag v2
# 8. leak check against teacher_val (run after the v2 build; it removed nothing):
#   teacher_val's 646 screenshots (named in teacher_val.npz's meta) linked into $R/tvdata, then
env $H $PY eval/tools/leakcheck.py $R/shots --test $R/tvdata --keep-reserved
```

v3 (from this branch; every command with `REALPLUS=$R3`, v2's `$R` files are read-only inputs). The run itself
differed in three ways:

- **Loops.** Detached loops drove it: `feed`, `luna.py run --loop`, and a two-hourly round of `final`,
  `edge-boxes`, `confirm-batches` and `edge-batches`. They also made the interim snapshots. Every round is in
  `$R3/logs` and `$R3/notes.md`.
- **The plan.** It was made on 2026-10-13, extended by the reclaim tier on 2026-10-14, and filtered by hand on
  2026-10-14 at 19:55Z (reserved VNs, PSP-scale reads, copies). `pool.py plan` has applied the same filters since
  2026-10-15. Re-run then, it gave the filtered plan minus 13 lines, 12 of them already batched.
- **Third looks and the last Luna round.** Local scripts made them (`$R3/look3.py`, `$R3/luna_retry3.py`), which
  `pool.py look3` and `luna.py run --retries 3` replace. On the same inputs, `look3` gives the same 58 candidate
  lists and the same 58 resolutions (checked).

```sh
HERD=/local/home/skerraut/herd-vita-ocr; R3=$HERD/shared/artifacts/realplus-v3; PY=.venv-ci/bin/python
export REALPLUS=$R3 HERD_DIR=/local/home/skerraut/herd-synth3 REALPLUS_HELPER=review-v3-
H="HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200"
# 1. the reclaim tier (fetched by reserve-policy): shots.dedupe over it (-> $R3/reclaim/manifests/dups.json) and
#    the read input $R3/reclaim/reads/shots.jsonl (hires shots read as kind vita, as v2's), then
env $H unshare -rn .venv/bin/python devnative/read.py run --shots $R3/reclaim/reads/shots.jsonl \
    --out $R3/reclaim/reads --procs 5
# 2. the plan, pass 1 (Claude), the Luna votes, edge checks and confirm looks, until nothing is pending
$PY realplus/pool.py plan
$PY realplus/pool.py feed --loop --every 30 --ready 4 --max 6 --min-new 1000000000 \
    --kinds confirm,edge-new,pass1,edge --tiers t1,t2
$PY realplus/luna.py run --loop --every 120 --max 4
env $H $PY realplus/pool.py final && env $H $PY realplus/pool.py edge-boxes
env $H $PY realplus/pool.py confirm-batches --n 20 --size 200 --min 1
env $H $PY realplus/pool.py edge-batches --n 20 --size 200 --min 1
$PY realplus/luna.py run --retries 3                                # the last votes before the freeze
# 3. v3pre (all of Stage A), its gates, the precision check and its third looks
cp $R3/labels.jsonl $R3/labels_v3pre.jsonl && env $H $PY realplus/pool.py edge-apply
env $H $PY realplus/hygiene.py --labels $R3/labels_v3pre.jsonl --out $R3/logs/hygiene_labels_v3pre.json
env $H $PY realplus/pool.py build --tag v3pre --procs 5
env $H $PY realplus/pool.py check --tag v3pre                       # 800 entries: review/check/b0001-b0007
$PY realplus/pool.py check-score --tag v3pre
env $H $PY realplus/pool.py look3 --tag v3pre                       # review/look3/b0001, then a fresh look
$PY realplus/pool.py look3 --tag v3pre --resolve                    # -> review/resolutions_v3.jsonl
# 4. v3: the same gates, Luna's agreement, the committed manifests
env $H $PY realplus/pool.py final && cp $R3/labels.jsonl $R3/labels_v3.jsonl
env $H $PY realplus/pool.py edge-apply
env $H $PY realplus/hygiene.py --labels $R3/labels_v3.jsonl --out $R3/logs/hygiene_labels_v3.json
env $H $PY realplus/pool.py build --tag v3 --procs 5
env $H $PY realplus/hygiene.py --lines $R3/manifests/lines_v3.jsonl --out $R3/logs/hygiene_v3.json
env $H $PY eval/tools/reserve_check.py ink=$R3/realplus_train_ink_v3.npz grad=$R3/realplus_train_grad_v3.npz \
    lines=$R3/manifests/lines_v3.jsonl --require-hashes --pixels --out $R3/logs/reserve_check_v3.json
env $H $PY eval/tools/leakcheck.py $HERD/shared/artifacts/reserve-policy/realplus/shots \
    --data $HERD/shared/artifacts/eval --keep-reserved                # the reclaim tier vs vita-test / psp-test
$PY realplus/luna.py stats --batches 10 --out $R3/luna/agreement_first10.json
$PY realplus/luna.py stats --batches 1000 --out $R3/luna/agreement_all.json
env $H $PY realplus/pool.py publish --tag v3 --supersedes v2 --check v3pre
```
