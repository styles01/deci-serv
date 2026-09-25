# Third-party components

## arbiter showcase — MIT — © 0xBakeer (github.com/0xBakeer)

`vendor/arbiter/` is a clone of https://github.com/0xBakeer/arbiter (MIT license,
LICENSE file included in that directory).

Used by DeciServ:

- `games/serve_showcase.py` — serves the unmodified `vendor/arbiter/showcase/`
  pages over an adapter that forwards `POST /v1/systemone` to DeciServ's
  `/decide`. All page code, canvas rendering, panel UI, theme and the
  recorded-run format are arbiter's.
- `games/shoot_sixup.py` — composites six of his showcase pages into the
  six-up grid video format.
- `games/arcadia_extra.py` — the paddle and dungeon games are ports of
  arbiter's `showcase/paddle/logic.mjs` and `showcase/dungeon/logic.mjs`
  to DeciServ's Python harness (same mechanics, generator tables and
  shield semantics; MIT).
- `games/grid_replay.py` — re-simulation fallbacks are ports of his game
  rules; the replay-file schema follows his `recorder.mjs`.

The pages are served with an on-the-wire rebrand ("deciserv plays",
DeciServ engine naming) applied by `serve_showcase.py`; the files in
`vendor/arbiter/` stay byte-identical to upstream. His credit lives in
the served footer link, the compositor's credit chip, README.md Credits,
and this NOTICE. Nothing in `vendor/arbiter/` is modified.

Laya itself is © Convai Innovations / Nandha Kishor M, Apache-2.0 —
see the arbiter CREDITS.md for the full model attribution chain.