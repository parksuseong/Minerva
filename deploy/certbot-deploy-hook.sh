#!/bin/sh
# certbot 갱신 후 실행: Let's Encrypt 인증서를 minerva 서비스(ec2-user)가 읽는 곳에 복사하고 재시작한다.
# 설치: sudo install -m 755 deploy/certbot-deploy-hook.sh /etc/letsencrypt/renewal-hooks/deploy/minerva.sh
set -e
SRC=/etc/letsencrypt/live/minerva-ai.duckdns.org
DST=/home/ec2-user/.minerva/tls
install -o ec2-user -g ec2-user -m 644 "$SRC/fullchain.pem" "$DST/cert.pem"
install -o ec2-user -g ec2-user -m 600 "$SRC/privkey.pem" "$DST/key.pem"
systemctl restart minerva
