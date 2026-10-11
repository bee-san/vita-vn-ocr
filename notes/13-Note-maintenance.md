---
title: Maintaining this investigation notebook
created: 2026-10-10
updated: 2026-10-10
status: conventions
tags: [notes, conventions, evidence]
---

# Maintaining this investigation notebook

This is a working evidence system, not a retrospective success story. Keep enough detail to reproduce a finding and avoid repeating failed assumptions.

## Note types

- The index and topic notes explain the current model of the system.
- [[01-Evidence-and-claims]] labels observations, hypotheses and unsupported claims.
- [[02-Current-device-state]] is a replaceable dated snapshot; historical states remain in the experiment log.
- `experiments/` holds chronological evidence, commands and outcomes.
- `templates/` provides repeatable experiment and decision records.
- [[11-Sources-and-artifacts]] points to exact public revisions and private evidence without copying private content.

Use ordinary Markdown, YAML frontmatter and Obsidian links such as [[00-Start-Here]]. Titles/stems should be unique. No external Obsidian plugins or synced workspace settings are required. Add both an index link and a related-topic link for a new note so it is discoverable.

## Evidence labels

**Observed** means a specific log, binary, tool result or user report supports it; state which. **Measured** adds an explicit method, data and environment. **Inferred** explains the reasoning and its assumptions. **Hypothesis** needs a discriminating next test. **Unknown** is a valid result. **Superseded** preserves a previous explanation and links its correction.

Separate these statuses for every deployed change:

```text
implemented -> built -> locally validated -> pushed -> PR checked -> merged
                                                      |
                                              installed -> read back
                                                            |
                                               booted -> game tested
```

The paths are not interchangeable. A merged commit is not proof its bytes are installed; a successful upload is not a functional test; a module marked STARTED is not proof it attached a useful hook.

## After each experiment

1. Record source commit or clearly label a dirty experimental tree, build configuration, command and environment.
2. Record exact artifact hashes for anything installed; use private logs for transient IDs.
3. State observed result and confidence boundary. Retain failed attempts and distinguish accepted from completed.
4. Update the topic note and stable claim ID if the finding changes the model. Mark old hypotheses superseded rather than deleting their rationale.
5. Update the device snapshot only when new evidence establishes it; include UTC time. Do not infer state from elapsed time.
6. Update next steps so they answer the new uncertainty rather than repeating the previous plan.

## Publication and scope

This notebook lives in the OCR repository because that is the user's requested folder, while the native runtime changes live in `bee-san/vita-jp-overlay`. Use the separate notes branch/worktree for publication and mirror `notes/` into the requested active folder without switching that folder's branch or staging its unrelated work. See [[10-Repositories-and-revisions]].

Never commit game screenshots, dialogue, executables, extracted fonts, full RAM/coredumps, license files, token/identity material or actual device network addresses. Link narrow local evidence paths and retain small derived numeric tables or hashes. Check the staged diff before every commit. Never sweep `hooker/`, `build/`, `agent/` or another session's dirty files into the notes PR.

No console tests are required for prose edits. Check internal links, frontmatter, factual consistency, stale current-state claims and publication scope. If a command/example is changed, verify its syntax against local source/help without running an unintended device action.
