# jev_compat — Jev-hosted-API compatibility shim

Translates the hosted Jev API (POST /v1/systemone) to DeciServ's POST /decide,
so unmodified Jev-ecosystem skills (typesafe-mcp, winnow, semdecide, canny,
fast-jev-compaction, prism, jev-review, blink, jev-codex-router, jev-curate)
run against the local Laya checkpoint by pointing their base-URL override at
this shim instead of api.typesafe.ai.

Run: DECISERV_SHIM_UPSTREAM=http://192.168.2.185:8710/decide python3 shim.py
Listens on 127.0.0.1:8711, accepts any Bearer key. See SKILL jev-skill-deciserv-wiring
for the per-skill env override table and normalization contract.
