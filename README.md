# Lilim 1.0.0

Lilim is a local-first Linux assistant for Lilith Linux and Debian/Ubuntu desktops. It combines Rust and Candle inference, a Python agent service, and a Tauri desktop application.

<p align="center">
  <img src="assets/lilim-col.svg" alt="Lilim launcher icon" width="160" />
</p>

## 1.0.0 upgrades

- **Phi-3.5-mini replaces Phi-2** as the local Candle model, using a Q4_K_M GGUF build. The runtime remains Rust-native and does not require Ollama or a Python inference server.
- **Autonomous agent loop** can inspect system state and use shell, file, web, memory, and scheduled-task tools, with confirmation for consequential actions.
- **MCP integrations** support local stdio and Streamable HTTP servers. Read-only tools can run in a task; other tools request user approval in chat.
- **BrowserOS integration** connects through the MCP endpoint shown in BrowserOS settings and can launch the installed browser.
- **Persistent memory and prompt enhancement** support context-aware conversations across sessions.
- **Desktop and terminal workflows** include a Tauri launcher, global hotkey, command-line client, and one-command Debian/Ubuntu installer.
- **Installation reliability** now installs the Lilim desktop entry and its icon in the standard Linux icon theme locations.

## Model decision

Phi-4 was considered for this release, but its Candle support was not stable enough for the production path. I was hesitant to replace the existing model without confidence in the result. I’m dedicated to quality and speed, memory safety, and building something unique for Lilith Linux. Phi-3.5-mini is the choice for 1.0.0 because it is supported by Candle’s quantized Phi-3 implementation and lets Lilim keep its Rust inference engine.

## Planned features

- [ ] OAuth-based authorization for supported integrations
- [ ] Request proxies for remote model and tool traffic
- [ ] Deeper integration with Lilith Linux desktop and system services
- [ ] More capable memory retrieval and management
- [ ] Further prompt enhancement controls and tuning

## Install

On Debian/Ubuntu x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/BlancoBAM/Lilim/main/install.sh | bash
```

The installer fetches the latest release package and uses APT to install it. The package configures the desktop launcher, icon, system service, and Python environment. Launch Lilim from the Applications menu or run `lilim`; `lilim-cli` provides a terminal interface.

For a source build, use Rust, Node.js 22, Python 3.11, and the Tauri Linux system dependencies (`libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `librsvg2-dev`, `libappindicator3-dev`, and `patchelf`):

```bash
git clone https://github.com/BlancoBAM/Lilim.git
cd Lilim
./local_install.sh
```

## Features and settings

- Configure remote providers in the desktop Settings panel; keys remain on the local machine.
- Add MCP servers under **Settings → MCP & BrowserOS**. Use an HTTPS endpoint for remote servers and copy BrowserOS’s endpoint from its own MCP settings. Local stdio servers use an executable and argument list.
- MCP server settings are stored at `~/.config/lilim/mcp-servers.json` with owner-only permissions. Read-only tools may execute during a task; other calls wait for approval.
- The 2.4 GB Q4_K_M local model is downloaded from Hugging Face on first service start; the Debian release package does not bundle model weights. First launch needs an internet connection and enough free disk space. Offline deployments can provide the model in `/usr/lib/lilim/models/phi-3.5-mini-q4/` before starting the service.
- Memory is stored in SQLite under `~/.local/share/lilim/`.

## Architecture

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Local inference | Rust, Candle, Phi-3.5-mini GGUF | Private on-device generation |
| Runtime gateway | Rust, Axum, Tokio | Desktop API, process supervision, and tool routing |
| Agent service | Python, FastAPI | Provider routing, MCP, memory, and agent loop |
| Desktop | Tauri 2, React, TypeScript | Chat, settings, and approvals |
| Packaging | Debian package and systemd | Linux installation and service lifecycle |

## Development checks

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig cargo check --workspace
cd lilim_desktop && npm ci && npm run build
```

## License

Lilim is distributed under the GNU Affero General Public License, version 3.0. See [LICENSE](LICENSE).
