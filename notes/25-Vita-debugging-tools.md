---
title: Vita debugging tools and evidence workflow
created: 2026-10-11
updated: 2026-10-11
status: measured-tools-reference
tags: [vita, debugging, errors, coredump, tools]
---

# Vita debugging tools and evidence workflow

Use `psp2_err` for error names, `psp2-core-parse` for completed CPU crash dumps, and `vita-agent-use` for live native observations and collection. They answer different questions: an error name suggests a failing subsystem, a dump records a crash-time state, and a live log connects that state to the operation being tested. Follow [[09-Device-operations-and-recovery]] before device access and [[08-Validation-and-measurements]] when reporting a result.

## Tool revisions inspected

| Evidence ID | Tool | Revision and scope |
|---|---|---|
| DBG01 | [psp2_err 0.1.0](https://github.com/shoui520/psp2_err/tree/5fef5fb55cebc82ce90185acb0b55d49cc0fdf18) | Offline lookup; local README, `--help` and `--version` inspected on 2026-10-11. |
| DBG02 | [psp2-core-parse 0.2.0](https://github.com/shoui520/psp2_core_parse/tree/3b1219cf403c3a4a9ff7e8b1085ac0ead90ff52c) | Offline CPU dump analysis; local README and subcommand help inspected. |
| DBG03 | Local `vita-agent-use` checkout, base `87e9bc6feea3401d16844ee7d9b5d89c74ab3316` | CLI help and operating guide inspected. Runtime/skill files have local modifications: this is command-syntax provenance, not proof of a clean deployed build. |

The first two tools are installed beneath `agent/tools/` in the agent repository. `vita-agent-use` is a repository with an operating `AGENTS.md`, not a root `SKILL.md`. Its companion guides are `docs/vita-err/SKILL.md` and `docs/vita-coredump/SKILL.md`. Private configuration, tools and artifacts stay outside this OCR repository.

## Decode the exact error first

Run from the agent repository; these commands are offline:

```sh
agent/tools/psp2_err/psp2_err --version
agent/tools/psp2_err/psp2_err --json 0x80010013 0x8010113D 0xBFFFFFFD
agent/tools/psp2_err/psp2_err --search "access" --limit 10
```

Keep the raw hexadecimal value, signed decimal value, operation/stage, and tool revision. A database miss is useful evidence; it does not make the code success, and it does not justify assigning a nearby known error's meaning. Custom overlay errors must also be checked against the exact installed `include/vjo_api.h`.

**Measured, DBG04:** the pinned offline lookup was run on 2026-10-11 with `0xBFFFFFFD -1073741827 0x80010013 0x8010113D`, and separately with `-3`. The custom-code lookup exits 1 because no exact database match exists; its structured decomposition is still available.

| Input | Tool result | Interpretation limit |
|---|---|---|
| `0x80010013` | `SCE_ERROR_ERRNO_ENODEV`, display `C1-2755-9` | Storage device/namespace unavailable; it is not EACCES (`0x8001000D`). |
| `0x8010113D` | VSH facility, display `C2-12691-9`, no exact symbolic name/remark | The lookup alone cannot identify the VPK install cause. |
| `0xBFFFFFFD` / `-1073741827` | Noncritical error, reserved facility `0xfff`, code `0xfffd`, no exact native match | Compare against the overlay's custom API. |
| `-3` / `0xFFFFFFFD` | Critical error, same reserved facility/code | The tool [labels bit 30 as the critical flag](https://github.com/shoui520/psp2_err/blob/5fef5fb55cebc82ce90185acb0b55d49cc0fdf18/src/psp2_err/database.py#L137); this is not proof of the firmware's return mechanism. |

**Observed, DBG05:** physical overlay v0.7.2 logs show captures succeeding before the first text-bearing OCR job reports `io=0xBFFFFFFD cleanup=-117`; later capture requests return the same code. API 12 defined BUSY as `-3` (`0xFFFFFFFD`). **Inferred:** the measured return is that value with bit 30 cleared; this note does not claim a universal firmware dispatcher rule. [Overlay PR #27](https://github.com/bee-san/vita-jp-overlay/pull/27), exact head `b08969b2cf924b6198c52e29b6640b0b4ecc68dd`, changes all nine shared errors to signed stable encodings and requires API 13 in all three plugins. A host harness using the actual game worker/kernel broker reproduces retained ownership with the old header; the fix completes FL10/H15 recognition, drains cleanup, and permits another real capture request. This was a host regression result at the initial tool checkpoint. Later native API 13 model load/inference and clean repeated captures are recorded in [[26-Game-process-OCR-hardware-checkpoint]]; accuracy remains unmeasured.

## Collect a completed dump and analyze locally

For a live agent, retain the existing paired configuration and check capabilities before collecting evidence. These are verified command forms; run them one at a time when device access is authorized:

```sh
export VITA_AGENT_CONFIG_DIR="/absolute/path/to/existing-private-agent-config"
./vita-agent server status
./vita-agent system capabilities
./vita-agent plugins list
./vita-agent server events --after 0 --wait 30
```

Use the returned event cursor as the next `--after`. `server status` is PC-side cached state; a socket or cached connection flag is not a successful native query. Plugin configuration, loaded module state, accepted requests and completed operations are separate observations.

After a matching `coredump.complete`, create a private local output directory and download its completed path. Substitute the actual path from the event:

```sh
mkdir -p agent/private-debug
./vita-agent fs download ux0:data/COMPLETED-DUMP.psp2dmp agent/private-debug/dump.psp2dmp
agent/tools/psp2_core_parse/psp2-core-parse info agent/private-debug/dump.psp2dmp --strict --json
agent/tools/psp2_core_parse/psp2-core-parse analyze agent/private-debug/dump.psp2dmp --strict --json
agent/tools/psp2_core_parse/psp2-core-parse modules agent/private-debug/dump.psp2dmp
```

The current `fs download` takes two positional arguments, `PATH OUTPUT`; screen capture uses `--output`. Help works without a paired config. Do not analyze a still-saving `.psp2dmp.tmp` as a finalized experiment, close the app, or reboot while its dump is being saved. The parser can inspect interrupted input, but record it as incomplete.

For symbols/disassembly, supply the matching decrypted ELF under its **actual runtime module name** reported by `modules`, with VitaSDK on `PATH` or `VITASDK`:

```sh
agent/tools/psp2_core_parse/psp2-core-parse analyze agent/private-debug/dump.psp2dmp \
  --image RUNTIME_MODULE=agent/private-debug/matching-module.elf --json
agent/tools/psp2_core_parse/psp2-core-parse triage agent/private-debug/*.psp2dmp --json
```

Without matching images, separate the captured PC/LR, memory coverage and module mapping from stack-scan candidates or inferred cause. A historical Shell dump cannot prove the current game-process worker failed, and an empty current dump inventory cannot prove successful OCR.

## Historical dump measurements

**Measured, DBG06:** `triage` and `analyze --json` were run on three completed **2026-10-10 Shell dumps**, retrieved into the current private evidence folder. The commands analyzed 3/3 dumps and grouped them into two failures. All were structurally complete with complete compression and no decoder errors; header counts were 306/306, 308/308 and 302/302.

| Dumps | Crashing thread and module association | What is established |
|---|---|---|
| 2 | `VauService`; `UNDEFINED_INSTRUCTION`, `pngshot@1+0x8000` | Same module-offset failure group. No matching pngshot ELF was supplied. |
| 1 | `SceLiveAreaDailyCronJob`; `PREFETCH_ABORT`, `ScePaf@1+0x13d8` | Execute fault with ScePaf/SceShell return candidates. No matching ScePaf ELF was supplied. |

A bounded `tty --tail 20` query on the third dump records Shell physical-memory allocation failures: requests for 131,072 bytes and 8,192 bytes both failed. The adjacent diagnostic values are not treated as measured available memory because the log does not label them that way. This supports the historical pressure concern behind moving OCR away from Shell. It does not measure current OCR peak memory or prove the source-level cause of the prefetch abort. No October 11/API 13 dump or successful physical OCR is established by these old files. The later supplied Shell/LibKernel image compatibility and stripped-disassembly limits are recorded in [[26-Game-process-OCR-hardware-checkpoint]].

Private evidence locator: `/home/bee/Downloads/vita-release-2026-10-11/capture-error-20261011T091332Z/core-dumps/`. Keep the dumps, analysis JSON and TTY content private; only the derived counts/groupings above belong in the notebook.

## Repeatable test sequence

1. Record the tested source revision, installed plugin hashes/API, game/title, UTC time and the exact action. Keep device identity and raw dialogue private.
2. Capture current native capabilities/module state and fetch the bounded overlay logs. Record native errors verbatim and decode them offline.
3. If a crash occurs, wait for dump completion, preserve its hash and analyze the exact dump with matching build symbols. Compare repeated dumps rather than treating one trace as a full diagnosis.
4. Connect evidence across stages: capture, line finding, model read/load, recognition, cleanup and the next capture. A generic UI capture error can be a consequence of an earlier retained OCR claim.
5. After a repair, distinguish built/host-tested, uploaded/read-back, booted and physical in-game recognition. Require recognized text plus `rc=0 cleanup=0` in fresh runtime logs; host tests and FTP hashes do not establish hardware OCR.

Use the supported agent client when healthy, preserve pairing and unfinished operation IDs, and follow its documented recovery path when unavailable. This reference does not activate the agent or change the console configuration. Do not publish full snapshots, dump JSON, raw dumps, screenshots, game images, credentials, actual device endpoints or identity fields. Publish only derived metadata needed for the claim and narrow private artifact links; see [[11-Sources-and-artifacts]] and [[13-Note-maintenance]].
