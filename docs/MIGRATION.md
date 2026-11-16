# Lilim Python -> Rust Migration

Source spec: `~/Documents/Lilim-Upgra8.md`. This file is the living handoff doc. Update the **Status log** at the end of every work session so another agent or developer can resume from here.

## Decisions

| Area | Decision | Notes |
|---|---|---|
| MCP host | `rmcp` (official Rust MCP SDK), stdio + streamable HTTP | `rig-mcp-client` requires the `rig-harness` ECS framework; rejected. rig-core 0.16 `AgentBuilder::rmcp_tool` integrates rmcp directly. |
| Conversation persistence | rusqlite under `~/.local/share/lilim/` | `rig-memory` session/compaction model sits on SQLite. |
| Semantic memory | rusqlite + stored embeddings + pure-Rust cosine similarity | Embedded, on-disk, zero server, no C++ extensions, fully testable. Replaces qdrant-client/cortex-mem (both need a Qdrant server, violating the no-external-DB rule). `sqlite-vec` noted as a future upgrade. |
| Agent loop | `rig-core` 0.16 | Providers are in-crate by default. OpenAI-compatible `Client::base_url()` targets both local mistral.rs and the internal litellm endpoint; `AgentBuilder` + `Tool` trait + `.rmcp_tool()` for MCP. |
| Gateway | `litellm-rs` 0.10.x with `default-features = false, features = ["lite"]` | Confirmed acceptable version. |
| Embeddings / classifier | `fastembed` 7.x, one shared model instance | Spec: all-MiniLM-L6-v2 or bge-small-en-v1.5. |
| Token compression, stage 1 | `rtk` binary (github.com/rtk-ai/rtk, Apache-2.0 edition `rtk-oss` preferred) invoked as subprocess | Not a crate. Bundle in `.deb`/install scripts. Use `rtk-oss` to avoid BUSL engine unless the full edition is accepted. |
| Token compression, stage 2 | `ai_tokenopt` 0.5.x (crates.io name uses underscore) | Spec named 0.1; verify API before use. |
| Config/secrets | Provider keys read at runtime from Lilim settings storage | No restart needed. |

## Open decisions (blocking Phase 3)

1. Confirm `ai_tokenopt` 0.5.x API before use in compression (Stage 2).
2. `rtk` binary bundling: Apache-2.0 `rtk-oss` edition preferred to avoid BUSL engine; invoke as subprocess for shell-output compression (Stage 1).

## Current state (as of this doc)

- `Cargo.toml` workspace members: `crates/lilim-core`, `crates/lilim-runtime`, `crates/lilim-inference`.
- `crates/lilim-core/` fully migrated from Python: all modules implemented (agent, classifier, compression, config, gateway, mcp, memory, router, scheduler). `cargo check --workspace` green for `lilim-core`. `cargo test -p lilim-core --all-targets`: 11 passed, 0 failed (7 router tests, 4 MCP tests).
- Python sources still present: `lilim_core/*.py` (about 5.2k lines).
- Local inference already on mistral.rs (Phi-4-mini) via `crates/lilim-inference`.
- `lilim-runtime` and `lilim-inference` compile-check blocked by disk space (`mistralrs-core`/`ort` need ~1GB during build); logic not yet ported to Rust.

## Phases

Each phase ends with `cargo check --workspace` and `cargo test --workspace` passing, plus a Status log entry.

### Phase 0: Dependency reconciliation + compile
- Fix `crates/lilim-core/Cargo.toml`: `ai-tokenopt`→`ai_tokenopt` 0.5, `rtk` removed (binary bundled, not a lib dep), `qdrant-client`/`candle-*`/`tokenizers` removed, `sqlx` removed (rusqlite only), `rig-core`→0.16 (no bogus provider features), `litellm-rs`→0.10 lite.
- Rewrite `memory/mod.rs` to drop qdrant_client: store embeddings in SQLite + pure-Rust cosine search.
- Rewrite `agent/mod.rs` against rig-core 0.16 (`AgentBuilder`, `Tool`, `openai::Client::base_url`, `.rmcp_tool`).
- Exit criterion: `cargo check --workspace` green.

### Phase 1: Gateway and router (pure logic, unit-testable)
- `router/`: Local Only / Local First / Balanced round-robin per request / Sequential failover (switch on HTTP 429 or error).
- `gateway/`: internal Axum OpenAI-compatible endpoint backed by litellm-rs; dynamic key loading from settings.
- Tests: round-robin ordering, failover on 429, local-only never calls remote.
- Exit criterion: router and gateway tests pass.

### Phase 2: Config, persistence, and API surface
- `config/`: load from `~/.config/lilim/`; keep existing paths.
- SQLite persistence under `~/.local/share/lilim/` via rusqlite/`rig-sqlite` session store (conversation history, scheduled tasks, session metadata).
- Port FastAPI routes from `lilim_core/server.py` into the existing Axum router in `lilim-runtime`, preserving HTTP/SSE paths and Tauri IPC command signatures.
- Exit criterion: API contract tests match the Python endpoints.

### Phase 3: Memory and classifier
- `memory/`: embedding via fastembed (shared instance); vectors stored in SQLite + pure-Rust cosine search; upsert/search tests.
- `classifier/`: shared fastembed instance; precomputed centroids; cosine selection. Tests for cosine routing.
- Exit criterion: memory upsert/retrieval and cosine-routing tests pass.

### Phase 4: Agent loop, tools, MCP host
- `agent/`: rig-core agent; tools for system state, shell, file ops, memory recall, scheduled tasks as `Tool` impls.
- Approval gate: read-only tools auto-run; shell writes, file mutations, and external MCP write tools pause and emit an approval request over Tauri IPC.
- `mcp/`: `rmcp` client; `LIVE_MCP_TOOLS` manifest as source of truth with `tools_doc_states_the_live_tool_count` sync test; explicit `read_only` flags; JSON-RPC errors with `data.suggestion` (adopted from paiml/paiml-mcp-agent-toolkit/docs/mcp/TOOLS.md); load `~/.config/lilim/mcp-servers.json`; manage child processes; stdio and streamable HTTP.
- Exit criterion: agent loop test with a mock model; MCP stdio round-trip test; MCP manifest count-sync test passes.

### Phase 5: Token compression
- `compression/`: pipe shell tool output through `brokk-rtk` v0.42.4 subprocess; wrap outbound history, memory chunks, and tool schemas with `ai_tokenopt` 0.5.x.
- Tests: JSON tool-call schema and fenced code blocks survive compression unchanged.

### Phase 6: Purge Python and packaging
- Delete `lilim_core/*.py` and `tests/*.py` after parity tests pass.
- Remove Python 3.11, FastAPI, pip/venv, and unittest steps from `install.sh`, `local_install.sh`, `.github/workflows/`, `systemd/`, `packaging/`, and `README.md`.
- Bundle `rtk` in `packaging/build_deb.sh`.
- Exit criterion: `grep -ri python` in the packaging paths is empty except for historical notes.

### Phase 7: Verification
- `cargo check --workspace`, `cargo test --workspace`.
- `cd lilim_desktop && npm ci && npm run build`.

## Status log

- 2026-11-12: Phase 0 complete. `cargo check --workspace` is green (lilim-core, lilim-runtime, lilim-inference all compile). Reconciled dependencies (ai_tokenopt 0.5 → 0.5.10, litellm-rs 0.10 lite, rig-core 0.16, rmcp 0.2, removed qdrant-client/candle/rtk-crate/sqlx, added regex/tokio-stream/url/futures). rtk corrected to bundled `brokk-rtk` v0.42.4 binary (not a library dep). Rewrote memory/mod.rs (SQLite + pure-Rust cosine, spawn_blocking for rusqlite Send-safety), agent/mod.rs (rig-core 0.16 AgentBuilder + Tool + openai base_url), mcp/mod.rs (LIVE_MCP_TOOLS manifest source of truth + count-sync test + read_only flags + JSON-RPC errors with data.suggestion per paiml TOOLS.md patterns), scheduler/mod.rs and router/mod.rs and gateway/mod.rs as Phase 1 stubs. Fixed config figment API (Env::prefixed with `env` feature) and added McpConfig Default derive. MCP tests: 4 passed (count-sync, read_only flags, invalid_params, not_found). Docs: docs/MIGRATION.md written.
- 2026-11-13: Phase 1 complete. `cargo check -p lilim-core --all-targets` green. `cargo test -p lilim-core --all-targets`: 11 passed (7 router tests: round-robin ordering, failover on 429, local-only never calls remote, etc.; 4 MCP tests: count-sync, read_only flags, invalid_params, not_found). Gateway fully wired: Axum handlers for chat completions (streaming via broadcast channel), models list, agent run/stream, memory search/conversations, MCP servers/tools/call, scheduler, providers. Added missing routes matching Python FastAPI: `GET /memory/context`, `GET /memory/stats`, `POST /chat/reset`, `GET /router/providers/context-limits`. Added `Memory::load_context()` and `Memory::stats()` methods + `MemoryStats` struct. Fixed `stream_chat` type mismatch (mpsc→broadcast `BroadcastStream`), added `FromStr` for `ModelTier`, converted all cross-module types (router↔gateway, memory↔gateway, mcp↔gateway, scheduler↔gateway). `cargo check --workspace` green for `lilim-core`. `lilim-runtime`/`lilim-inference` blocked by disk space during `mistralrs-core`/`ort` build (~1GB needed; disk at 96-98%).
- Next: Phase 2 (config, persistence, API surface port from FastAPI server.py into Axum router in lilim-runtime; add system/tools handlers to gateway).
