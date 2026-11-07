#!/bin/sh
# certbot standalone 갱신 전에 80번을 비운다.
# 설치: sudo install -m 755 deploy/certbot-pre-hook.sh /etc/letsencrypt/renewal-hooks/pre/minerva-redirect.sh
systemctl stop minerva-redirect || true
