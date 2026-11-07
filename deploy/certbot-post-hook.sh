#!/bin/sh
# certbot 갱신 시도 뒤(성공/실패 무관) 리다이렉트를 다시 켠다.
# 설치: sudo install -m 755 deploy/certbot-post-hook.sh /etc/letsencrypt/renewal-hooks/post/minerva-redirect.sh
systemctl start minerva-redirect || true
