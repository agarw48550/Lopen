# SECURITY.md — Lopen Security Considerations

## 1. WhatsApp Integration (Mode A) Caveats

Lopen uses **WhatsApp Web automation** via Playwright — this is an unofficial integration.

### Risks
- **Ban risk**: WhatsApp's Terms of Service prohibit automation of the WhatsApp Web interface. Use at your own risk. Accounts engaging in automated messaging may be temporarily or permanently banned.
- **Session exposure**: The WhatsApp session file (`storage/whatsapp_session/session.json`) contains authentication tokens. **Never commit this file to version control** (it is already in `.gitignore`).
- **Local only**: The WhatsApp bridge is designed to run exclusively on your local machine. Do **not** expose the Lopen API to the public internet while WhatsApp is enabled.

### Mitigations
- `whatsapp.enabled` defaults to `false` — you must opt in explicitly.
- Session files are stored in `storage/whatsapp_session/` which is gitignored.
- Use a secondary/dedicated WhatsApp account for automation.

---

## 2. Permission Model for Desktop and File Operations

Lopen implements a tiered permission model to prevent accidental or malicious file operations.

### PermissionLevel
| Level | Value | Allowed Operations |
|-------|-------|--------------------|
| LOW | 0 | Read-only, no side effects |
| MEDIUM | 1 | File reads, web browsing |
| HIGH | 2 | File writes, system queries (default) |
| CRITICAL | 3 | Shell execution, WhatsApp, desktop control |

### Safe Directories
The `FileOps` and `DesktopOrganizer` tools restrict all operations to:
- `~/Documents`
- `~/Desktop`
- `~/Downloads`

Any attempt to access paths outside these directories is rejected. System directories (`/etc`, `/System`, `/usr`, etc.) are never accessible.

### Decorator Usage
Sensitive functions are decorated with `@permission_required(PermissionLevel.HIGH)`. The global threshold can be raised at runtime via `set_permission_threshold()` — e.g., set to `MEDIUM` to prevent any file writes.

---

## 3. Network Exposure Considerations

### Default binding
- Orchestrator: `0.0.0.0:8000`
- Dashboard: `0.0.0.0:8080`

Both bind to all interfaces by default to support local network access from other devices on the same Wi-Fi (e.g., iPhone → Mac). 

### Recommendations
- **Firewall**: Restrict inbound connections to ports 8000 and 8080 using macOS Application Firewall or `pf`.
- **LAN only**: Never expose these ports to the internet (no public port forwarding).
- **Secret key**: Set `LOPEN_DASHBOARD_SECRET_KEY` in `.env` to a long random string.
- **No authentication by default**: If you need access control, add an API key middleware or run behind a reverse proxy (e.g., Caddy with local TLS).

---

## 4. Safe Defaults

| Setting | Default | Security Rationale |
|---------|---------|-------------------|
| `whatsapp.enabled` | `false` | Prevent accidental WhatsApp automation |
| `whatsapp.headless` | `true` | No UI exposure |
| `llm.memory_conservative` | `true` | Prevents keeping sensitive prompts in memory |
| File ops allowed dirs | `~/Documents`, `~/Desktop`, `~/Downloads` | No system dir access |
| CORS origins | `["*"]` | LAN access — restrict to specific origins in production |
| Permission threshold | `HIGH` | File writes allowed; shell execution requires override |

### Secrets
- **Never** commit `.env` to version control (gitignored by default)
- **Never** commit model files or WhatsApp session files
- Rotate `LOPEN_DASHBOARD_SECRET_KEY` if you believe it has been exposed

---

## 5. Reporting Vulnerabilities

If you discover a security vulnerability in Lopen, please open a GitHub issue marked **[SECURITY]** or contact the maintainers directly. Do not disclose security issues publicly before they have been assessed and patched.
