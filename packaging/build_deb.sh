#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
DEB_ROOT="$ROOT_DIR/packaging/deb_root"
DEB_OUTPUT="$ROOT_DIR/dist"
DEB_NAME="lilim-linux-component.deb"

echo "Building Debian package into $DEB_OUTPUT/$DEB_NAME"
mkdir -p "$DEB_ROOT" "$DEB_OUTPUT"

set -x
# Ensure a clean Debian layout exists
rm -rf "$DEB_ROOT"/ || true
mkdir -p "$DEB_ROOT/usr/local/bin" "$DEB_ROOT/DEBIAN" "$DEB_ROOT/etc/lilith" "$DEB_ROOT/usr/lib/lilim" || true

## Copy a minimal launcher script (example)
cat > "$DEB_ROOT/usr/local/bin/lilim-run" <<'EOS'
#!/usr/bin/env bash
echo "Lilim runtime entrypoint (deb package placeholder)"
EOS
chmod +x "$DEB_ROOT/usr/local/bin/lilim-run"

# Debian control file
cat > "$DEB_ROOT/DEBIAN/control" <<'CTRL'
Package: lilim
Version: 0.1.0
Section: base
Priority: optional
Architecture: amd64
Maintainer: Lilim Maintainers <maintainer@example.com>
Description: Lilim component for production-ready Open Interpreter-based AI assistant
 This package includes the production-ready Lilim runtime components and their launcher.
CTRL

dpkg-deb --root-owner-group --build "$DEB_ROOT" "$DEB_OUTPUT/$DEB_NAME"
echo "DEB built at $DEB_OUTPUT/$DEB_NAME"
