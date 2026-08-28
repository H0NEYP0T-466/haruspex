# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in HARUSPEX, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **Email:** Send a detailed report to the maintainer via GitHub
2. **GitHub Security Advisories:** Use the [Security Advisories](https://github.com/H0NEYP0T-466/haruspex/security/advisories) feature
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Include

- Type of vulnerability (e.g., code injection, path traversal, model poisoning)
- Affected component (backend API, model loading, PE parsing, YARA rules)
- Attack vector and prerequisites
- Severity assessment (Critical / High / Medium / Low)
- Any proof-of-concept code

---

## Response Timeline

| Action | Timeline |
|--------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 1 week |
| Fix or mitigation | Within 2 weeks for critical, 30 days for others |
| Public disclosure | After fix is released |

---

## Scope

### In Scope

- Backend API vulnerabilities (FastAPI endpoints)
- Model loading/prediction pipeline security
- PE file parsing vulnerabilities (LIEF integration)
- YARA rule execution safety
- Path traversal in file upload/download
- Authentication/authorization issues (if implemented)
- Dependency vulnerabilities

### Out of Scope

- Social engineering attacks
- Issues in third-party dependencies (report upstream)
- Theoretical model weaknesses (e.g., adversarial ML attacks on the classifier itself)
- Frontend-only XSS that requires user interaction with malicious content

---

## Security Best Practices

When contributing to HARUSPEX, follow these security guidelines:

- **Never commit secrets** — API keys, credentials, tokens, `.env` files
- **Validate inputs** — Sanitize all user inputs at API boundaries
- **Use parameterized queries** — When interacting with databases
- **Limit file uploads** — Validate file types and sizes
- **Keep dependencies updated** — Regularly audit and update packages
- **Follow least privilege** — Grant minimal necessary permissions
- **Use HTTPS** — In production deployments

---

## Dependency Security

```bash
# Python
pip audit                    # Check for known vulnerabilities
safety check                 # Alternative checker

# Node.js
npm audit                    # Check for known vulnerabilities
```

---

## Model Security

- Models are loaded from trusted local paths only
- No remote model loading without verification
- Model files should be integrity-checked before loading
- Training data should be validated for poisoning attacks

---

## Contact

For security-related inquiries, use [GitHub Security Advisories](https://github.com/H0NEYP0T-466/haruspex/security/advisories).

We appreciate your help in keeping HARUSPEX secure!
