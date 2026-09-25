# Harvest Backup Manifest — arcadia games JSONL

**Date:** 2026-09-24 (MDT)
**Source:** Mac `/tmp/arcadia*.jsonl` (VOLATILE — cleared on reboot)
**Destination:** Spark `jaita@192.168.2.185:~/deciserv-data/harvest/`
**Method:** `rsync -av --checksum` over SSH, then SHA-256 (`shasum -a 256`) on BOTH ends + row count (`wc -l`).

**Result: 6/6 files transferred, 6/6 SHA-256 MATCH Mac↔Spark, 6/6 row counts match. Backup verified.**

| # | File | Bytes (Mac) | Rows | SHA-256 (Mac == Spark) |
|---|------|-------------|------|------------------------|
| 1 | arcadia_moves.jsonl | 1,006,123 | 11,325 | `0c4b2aa1dbf6e36725ff6a1e8ccc1dc436af0e3a2adb576d81f696acc93fcdd9` ✅ MATCH |
| 2 | arcadia_rows_snake.jsonl | 23,422,181 | 14,814 | `67985c2d95b99292887fcd749c5f0764a0c25462814d01e7b6eaa3e176a5664f` ✅ MATCH |
| 3 | arcadia_rows_rest.jsonl | 814,269 | 472 | `41715747503cea6e5a23e9a04cce391a7e88fe882094b982985b6a0eb41e0136` ✅ MATCH |
| 4 | arcadia_snake_expl.jsonl | 22,836,683 | 14,818 | `afa8d3a95de6662dd01fdf16d496dcb3a74da6f37fd0cd0bea2ba15dd44122a6` ✅ MATCH |
| 5 | arcadia_expl_archive.jsonl | 4,792,800 | 2,883 | `b57ea9262369d864d222215b8fdde105ad61a05bfcf2420cdfdbac09f1e85af8` ✅ MATCH |
| 6 | arcadia_harvest.jsonl | 1,026,911 | 11,786 | `b098c3515c0cd96465cc40df7dc0b714cc84e5c2e4d59f855d22a6571d1124d2` ✅ MATCH |

**Total:** 53,898,967 bytes (~54.5 MB), 56,098 rows across 6 files.

## Notes
- Mac-side `/tmp/arcadia_expl2.jsonl` from the shortlist does NOT exist (checked `ls /tmp/arcadia*` — 6 files only). If the games harvested an expl2 file later, re-run the backup.
- Spark listing at transfer time: 6 files, sizes byte-identical to Mac.
- These are the only copies besides volatile Mac /tmp. Phase 4 fine-tuning should read from the Spark copy.