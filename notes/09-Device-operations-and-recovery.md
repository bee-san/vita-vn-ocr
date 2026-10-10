---
title: Device operations and recovery
created: 2026-10-10
updated: 2026-10-10
status: operating-reference
tags: [vita, operations, recovery]
---

# Device operations and recovery

Read the current `vita-agent-use/AGENTS.md` and command references before operating. This note records the workflow used here; it does not replace current tool capabilities or broaden authorization. The user already authorized this investigation's installs and use of the paired agent. Preserve that context rather than repeatedly asking to do the same approved action.

## Access and concurrency

Use `./vita-agent` or `python3 client/vita_agent.py` from the agent repository. Keep its existing paired configuration directory; do not delete credentials/state or create a new identity to repair a transport error. Public notes use placeholders and contain no device endpoint or private identity.

```sh
export VITA_AGENT_CONFIG_DIR="/absolute/path/to/existing-private-agent-config"
python3 client/vita_agent.py server status
python3 client/vita_agent.py plugins list
python3 client/vita_agent.py fs download   ux0:data/VitaJPOverlay/status.txt agent/NEW-EXPERIMENT/status.txt
```

Create the local output parent first. Verify exact syntax with `--help` if the client has changed. The normal configuration and artifact directories are listed as local paths in [[11-Sources-and-artifacts]], without opening their contents.

Use one device client/transfer at a time and one long-lived `serve` per configuration. Check for an existing server before starting another. Wait for each transfer to finish, inspect its final result and verify readback before the next mutation or reboot. A timed-out operation may have succeeded: keep its operation ID and transfer-state file and query recovery/status. Do not mint a second operation to hide uncertainty.

`accepted` means the request was admitted. It does not prove a launch, installation or reboot completed. Query native state afterward. A socket existing is not proof the server is connected.

## Installing a plugin

1. Identify the target path and preserve a local binary backup of the currently installed file.
2. Build the intended source; record SHA-256 and size. Run the relevant packaged ABI checks.
3. Use the tool's staged `fs upload` with its transfer-state file and expected old hash. `ur0:tai/` requires the existing scoped ACL and `--yes`; no bypass is implied.
4. Wait for completion, download the installed file and compare the whole file/hash.
5. Reboot only after readback and all transfers are complete.
6. Verify startup markers/module states, then the intended game behavior. Record each result separately in [[02-Current-device-state]].

For uncertain upload results, resume the same source/target/state operation according to the agent guide. Never reuse a transfer-state file for different bytes. Configuration changes must use `config plan`/`config apply`, with the reviewed diff and original-hash precondition; never upload `tai/config.txt` as an ordinary file.

## Boot recovery

The tested recovery approach for this setup is to hold **L during power-on** to bypass taiHEN plugins, then open VitaShell and its FTP address screen for recovery access. Holding L is a recovery action, not the normal boot validation test. Normal boot validation omits it.

Do not conflate a boot loop with proof of a permanently bricked device. The original incident recovered through this path and was traced to [[04-Firmware-365-boot-ABI]]. If a later shutdown occurs, first record the current installed hashes and last stage; it may have a different cause.

With the agent healthy, use its supported CLI; its workspace explicitly disallows ad hoc FTP, curl/raw HTTP and legacy client scripts for normal operations. Recovery access is a separate contingency. Read the repository's recovery guide for the exact supported route rather than inventing flags.

## Human-control boundaries

Pairing, new ACL and Content Manager approvals require the human's physical approval. Existing authorized ACLs do not need to be requested again without a new scope. Do not send controller input to approve a pending security dialog. PS+SELECT means “Agent stopped”; honor it and wait for explicit authorization before resuming control.

If a coredump is being saved, wait for its matching completion event. Do not close the game or reboot while a `.psp2dmp.tmp` is still being written. USB attachment alone does not establish a supported USB control transport; the inspected client did not provide one.

## Useful local references

- `vita-agent-use/AGENTS.md`
- `docs/agent-reference/commands.md`
- `docs/agent-reference/recovery.md`
- `docs/agent-reference/events-runs.md`
- `docs/vita-err/SKILL.md` for exact error decoding

These paths are relative to the agent repository, not this notebook. See [[10-Repositories-and-revisions]].
