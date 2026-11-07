"""포트 80 요청을 https://minerva-ai.duckdns.org 같은 경로로 301 리다이렉트한다.

표준 라이브러리만 쓴다. certbot(standalone)이 갱신할 때는 80번을 써야 하므로
deploy/certbot-pre-hook.sh 가 이 서비스를 멈추고 post-hook 이 다시 켠다.
"""

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("MINERVA_HTTPS_HOST", "minerva-ai.duckdns.org")


class Redirect(BaseHTTPRequestHandler):
    def _go(self):
        self.send_response(301)
        self.send_header("Location", f"https://{HOST}{self.path}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_OPTIONS = _go

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 80), Redirect).serve_forever()
