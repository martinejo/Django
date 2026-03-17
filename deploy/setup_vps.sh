#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/anomaly-web
SERVICE_NAME=anomaly-web

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Este script solo soporta Debian/Ubuntu (apt-get)." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y python3 python3-venv python3-pip nginx git

mkdir -p "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
  echo "ERROR: falta repositorio en $APP_DIR. Copia tu proyecto ahí antes de continuar." >&2
  exit 1
fi

cd "$APP_DIR"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp deploy/app.service /etc/systemd/system/${SERVICE_NAME}.service
cp deploy/nginx-anomaly.conf /etc/nginx/sites-available/${SERVICE_NAME}
ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/${SERVICE_NAME}
rm -f /etc/nginx/sites-enabled/default

systemctl daemon-reload
systemctl enable --now ${SERVICE_NAME}
nginx -t
systemctl reload nginx

# Seguridad de conectividad:
# - NO modificamos reglas de firewall (ufw/iptables)
# - NO reiniciamos interfaces ni servicios de red
# De esta forma, el script no debería cortar acceso remoto a Internet/SSH.

echo "Despliegue completado sin cambios de red/firewall. Verifica:"
echo "  systemctl status ${SERVICE_NAME} --no-pager"
echo "  journalctl -u ${SERVICE_NAME} -n 100 --no-pager"
