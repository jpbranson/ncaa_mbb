#!/bin/bash
# Install the live poller and the nightly ratings refresh as macOS LaunchAgents.
#
#   bash deploy/install_macos.sh          # install and start
#   bash deploy/install_macos.sh --remove # stop and uninstall
#
# Paths are resolved at install time and written into the plists, because a
# LaunchAgent has no shell and inherits almost no environment. That is the usual
# reason "it works in my terminal" and "it works under launchd" disagree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
AGENTS="$HOME/Library/LaunchAgents"
LIVE="$AGENTS/com.cbbwp.live.plist"
RATINGS="$AGENTS/com.cbbwp.ratings.plist"
LOGS="$ROOT/data/logs"

if [[ "${1:-}" == "--remove" ]]; then
  launchctl unload "$LIVE" 2>/dev/null || true
  launchctl unload "$RATINGS" 2>/dev/null || true
  rm -f "$LIVE" "$RATINGS"
  echo "removed. logs left in $LOGS"
  exit 0
fi

mkdir -p "$AGENTS" "$LOGS"
echo "root:   $ROOT"
echo "python: $PY"

cat > "$LIVE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.cbbwp.live</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$ROOT/scripts/serve_live.py</string><string>--quiet</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key><dict>
    <key>CBBWP_ROOT</key><string>$ROOT</string>
    <key>CBBWP_MODEL_VERSION</key><string>v2</string>
    <key>CBBWP_API_PORT</key><string>8808</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOGS/live.log</string>
  <key>StandardErrorPath</key><string>$LOGS/live.err</string>
</dict></plist>
PLIST

# The ratings snapshot decays; rebuild it every morning, well before tip-off.
cat > "$RATINGS" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.cbbwp.ratings</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$ROOT/scripts/build_live_context.py</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key><dict><key>CBBWP_ROOT</key><string>$ROOT</string></dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>30</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>$LOGS/ratings.log</string>
  <key>StandardErrorPath</key><string>$LOGS/ratings.err</string>
</dict></plist>
PLIST

launchctl unload "$LIVE" 2>/dev/null || true
launchctl unload "$RATINGS" 2>/dev/null || true
launchctl load "$LIVE"
launchctl load "$RATINGS"

echo
echo "installed:"
echo "  com.cbbwp.live     always on, polls the slate, serves the API"
echo "  com.cbbwp.ratings  daily at 09:30, refreshes the ratings snapshot"
echo
echo "check:   curl -s http://127.0.0.1:8808/health"
echo "logs:    tail -f $LOGS/live.log"
echo "stop:    bash deploy/install_macos.sh --remove"
