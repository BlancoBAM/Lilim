#!/usr/bin/env bash
# Install Lilim from the latest published Debian release without cloning the repo.
set -euo pipefail

RELEASE_DEB_URL="https://github.com/BlancoBAM/Lilim/releases/latest/download/lilim_0.1.0_amd64.deb"

fail() { printf 'Lilim installer: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Linux ]] || fail "This installer supports Debian/Ubuntu Linux only."
[[ "$(dpkg --print-architecture 2>/dev/null || true)" == amd64 ]] || \
  fail "Prebuilt releases currently support amd64. Build from source for other architectures."

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    case " ${ID:-} ${ID_LIKE:-} " in
        *debian*|*ubuntu*) ;;
        *) fail "Use this installer on Debian, Ubuntu, or a compatible distribution." ;;
    esac
else
    fail "Cannot identify this Linux distribution (/etc/os-release is missing)."
fi

if (( EUID != 0 )) && ! command -v sudo >/dev/null 2>&1; then
    fail "Install sudo or run this script as root."
fi

run_root() {
    if (( EUID == 0 )); then "$@"; else sudo "$@"; fi
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
deb_file="$tmp_dir/lilim.deb"

if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error "$RELEASE_DEB_URL" -o "$deb_file" || \
        fail "Could not download the latest release package: $RELEASE_DEB_URL"
elif command -v wget >/dev/null 2>&1; then
    wget --quiet "$RELEASE_DEB_URL" -O "$deb_file" || \
        fail "Could not download the latest release package: $RELEASE_DEB_URL"
else
    fail "Install curl or wget and run this command again."
fi

[[ -s "$deb_file" ]] || fail "The downloaded package is empty."
printf 'Installing the latest Lilim release...\n'
run_root apt-get update
(cd "$tmp_dir" && run_root apt-get install -y ./lilim.deb)
printf 'Lilim installation completed. Launch with: lilim\n'
