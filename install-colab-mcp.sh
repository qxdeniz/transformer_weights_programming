#!/bin/bash
set -euo pipefail

# Official Google Colab MCP: https://github.com/googlecolab/colab-mcp
# Run in macOS Terminal, outside the restricted Codex session.
colab_source='git+https://github.com/googlecolab/colab-mcp'
colab_uvx="$(command -v uvx || true)"
colab_codex="$(command -v codex || true)"

if [[ -z "$colab_uvx" && -x /opt/homebrew/bin/uvx ]]; then
  colab_uvx=/opt/homebrew/bin/uvx
fi
if [[ -z "$colab_codex" && -x "$HOME/.local/bin/codex" ]]; then
  colab_codex="$HOME/.local/bin/codex"
fi
if [[ -z "$colab_uvx" || -z "$colab_codex" ]]; then
  printf '%s\n' 'Required commands: uvx (from uv) and codex.' >&2
  exit 1
fi

# Download dependencies and check the entry point before registering the server.
"$colab_uvx" --python 3.13 --from "$colab_source" colab-mcp --help
"$colab_codex" mcp add colab -- "$colab_uvx" --python 3.13 --from "$colab_source" colab-mcp
"$colab_codex" mcp get colab
printf '%s\n' 'Registered Colab MCP. Restart the MCP connection in the app, open a Colab notebook in your browser, and verify the connection before running research.'
