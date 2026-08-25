# Security Audit

This document summarizes the security posture and implemented controls of the Mini Search Engine.

## Configuration & Secrets
- **No `.env` Committed:** The repository only tracks `.env.example`. Live environment variables are kept locally and out of source control.
- **No Secrets in Codebase:** All sensitive parameters, such as salts or potential keys, are managed via configuration or environment variables.

## Network & Access
- **CORS Configuration:** Configurable `ALLOWED_ORIGINS` ensures that the API only accepts cross-origin requests from trusted domains.
- **Rate Limiting:** Enforced at 120 requests per minute per IP to mitigate DoS and brute-force scraping attempts.
- **Request Size Limits:** Hard limits are placed on incoming request payloads to prevent memory exhaustion attacks.

## Input Validation & Protection
- **Strict Parameter Bounding:** Inputs such as `q` (MAX_QUERY_LENGTH=500), `top_k`, `page`, and `limit` are strictly typed and bounded.
- **SQL Injection Protection:** Database queries (e.g., analytics SQLite database) heavily utilize parameterized queries instead of string concatenation.
- **XSS Prevention:** The web layer leverages Jinja2 auto-escaping to neutralize Cross-Site Scripting vulnerabilities on user-facing endpoints.

## Operational Security
- **Error Exposure:** User-friendly messages are returned to the client, while detailed tracebacks are restricted to server-side logs.
- **Debug Mode Disabled:** Flask debug mode is explicitly disabled in the production configuration.
- **Security Headers:** Essential HTTP security headers are enforced (e.g., `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`).

## Deployment Security
- **Docker Non-root User:** The Dockerfile specifies an `appuser` so the container process does not run as root.
- **Systemd Sandboxing:** The systemd service configuration utilizes `ProtectSystem`, `ProtectHome`, and `NoNewPrivileges` to contain the application surface.
- **Dependency Minimal:** By relying almost exclusively on the Python standard library (with the exception of Flask), the project inherently minimizes its supply chain vulnerability footprint.
