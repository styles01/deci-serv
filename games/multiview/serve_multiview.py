#!/usr/bin/env python3
"""Static HTTP server for the DeciServ multiview grid (games/multiview/index.html).

Serves games/multiview/ on port 8030. The six showcase iframes inside the page
point at http://127.0.0.1:8010/showcase/<game>/, which is served by
games/serve_showcase.py (leave that one running — do not restart it).

Run:    python3 games/multiview/serve_multiview.py
Then:   open http://127.0.0.1:8030/   (LAN: http://192.168.2.173:8030/)
"""
import argparse
import http.server
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler, silent (keeps the nohup log empty)."""

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the multiview grid page.")
    ap.add_argument("--port", type=int, default=8030)
    args = ap.parse_args()

    # Serve this file's directory regardless of where the script is invoked from.
    os.chdir(HERE)

    with http.server.ThreadingHTTPServer(("0.0.0.0", args.port), QuietHandler) as httpd:
        print(f"multiview grid: http://0.0.0.0:{args.port}/ (dir={HERE})", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())