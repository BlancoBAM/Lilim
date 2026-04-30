#!/usr/bin/env bash
set -euo pipefail

# Releases a Deb packaging artifact to GitHub Releases using the REST API.
# Prereqs: a valid GitHub token with release scope in GITHUB_TOKEN (or GITHUB_TOKEN env).
# Usage: ./scripts/release_to_github.sh [tag] [path-to-deb]
# - tag: semantic tag to release (defaults to v0.2.0-prod)
# - path-to-deb: path to the .deb file to attach (defaults to dist/lilim-linux-component.deb)

TAG=${1:-v0.2.0-prod}
DEB_PATH=${2:-dist/lilim-linux-component.deb}
REPO="BlancoBAM/Lilim"

if [ ! -f "$DEB_PATH" ]; then
  echo "Deb package not found at $DEB_PATH" >&2
  exit 1
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN is not set. Set GITHUB_TOKEN to a GitHub token with repo scope to upload release assets." >&2
  exit 1
fi

set +e
RELEASE_JSON=$(curl -sS -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"tag_name\": \"$TAG\", \"name\": \"Lilim Production Release $TAG\", \"body\": \"Production release: idempotent host script, packaging artifacts, tests and health checks.\", \"draft\": false, \"prerelease\": false}" \
  https://api.github.com/repos/$REPO/releases)
set -e

RELEASE_ID=$(echo "$RELEASE_JSON" | jq -r '.id')
UPLOAD_URL=$(echo "$RELEASE_JSON" | jq -r '.upload_url')
if [ "$RELEASE_ID" = "null" ] || [ -z "$UPLOAD_URL" ]; then
  echo "Failed to create release on GitHub. Response was:" >&2
  echo "$RELEASE_JSON" >&2
  exit 1
fi

# Replace the {...} placeholder in the upload URL and append the filename as query param
UPLOAD_URL="${UPLOAD_URL//\{?name,label\}/}"
UPLOAD_URL="${UPLOAD_URL}?name=$(basename "$DEB_PATH")"

echo "Uploading $DEB_PATH to release $TAG (ID $RELEASE_ID)"
cURL_OUT=$(curl -sS -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @"$DEB_PATH" \
  "$UPLOAD_URL")
if echo "$cURL_OUT" | jq -e . >/dev/null 2>&1; then
  echo "Upload response:"; echo "$cURL_OUT" | jq
else
  echo "$cURL_OUT"
fi

# Print release URL if available
RELEASE_HTML_URL=$(echo "$RELEASE_JSON" | jq -r '.html_url')
echo "Release page: $RELEASE_HTML_URL"
