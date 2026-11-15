# Connect your own assistant

The public ORBITA dashboard is readable without an account. Chat requires **your own** Codex, Claude Code or OpenCode account running on your computer. The published site has no shared provider key, no account belonging to the author, and no hosted agent backend.

## 1. Sign in on your computer

Install a recent version of your preferred CLI using its official instructions, then sign in:

| Agent | Sign in | Official documentation |
|---|---|---|
| Codex | `codex login` | https://developers.openai.com/codex/cli/ |
| Claude Code | `claude auth login` | https://code.claude.com/docs/en/cli-reference |
| OpenCode | `opencode auth login` | https://opencode.ai/docs/cli/commands |

The account and any usage charges belong to the person running the CLI. ORBITA does not ask for your provider API key or password.

## 2. Start your local connector

Clone the repository and, from its root, run one of:

```sh
python scripts/local_agent_bridge.py --provider codex
python scripts/local_agent_bridge.py --provider claude
python scripts/local_agent_bridge.py --provider opencode
```

Only Python's standard library is required for the connector. It binds exclusively to `127.0.0.1:8766`. Keep that terminal open. For OpenCode you may select your configured provider/model explicitly with `--model provider/model`.

## 3. Pair the website

Open **Ask ORBITA → Connect your account**, choose the same agent and paste the pairing code printed by your local connector. The connector checks local sign-in before enabling chat. Your browser may request permission to access the local network; allow it only for the connector you started.

The pairing code is held only in page memory. Reloading or disconnecting clears it. It is not a provider API key, is not saved in localStorage, and is never sent to GitHub. Requests go directly to your loopback connector, which sends the selected forecast context and recent conversation through your own CLI.

## Behaviour and boundaries

- Before connection, chat inputs and question shortcuts are disabled. There is no silent shared-account or canned-response fallback.
- Provider credentials remain in the visitor's normal CLI authentication store. They are never copied into this repository or the website.
- The connector accepts only the paired bearer code and explicit allowed origins. It logs neither chat contents nor authentication output.
- Agents run in a temporary directory. Codex uses a read-only sandbox; Claude Code disables tools and external MCP configurations; OpenCode uses a dedicated no-tools agent. ORBITA does not enable bypass-permission flags.
- The assistant explains research forecasts and data provenance. It must not present clinical validation or invent causal treatment/meal predictions.
- No account access or billable agent request is performed by the automated tests; they use fake local agents.

If authentication fails, sign in again in your terminal and reconnect. If a CLI has changed its options, update it or consult its official documentation. The adapters are included in `scripts/local_agent_bridge.py`; successful provider authentication does not constitute clinical validation.
