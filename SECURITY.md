# Security Policy

## Supported versions

kvasir is pre-1.0. Only `main` receives fixes.

## Reporting a vulnerability

Do not open a public issue. Report privately through
[GitHub security advisories](https://github.com/zvdy/kvasir/security/advisories/new).

You will get an acknowledgement within 7 days. Fixes are released as soon as practical and disclosed in an advisory
after a patch is available.

Relevant areas include malformed model files (safetensors/config parsing), out-of-bounds device memory access in
kernels, and, once it exists, the HTTP server.
