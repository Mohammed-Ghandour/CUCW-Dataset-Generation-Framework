#!/bin/bash

#
# Cause the script to exit on failure.
set -eo pipefail

## Install Docker
apt update && apt install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "Types: deb" > /etc/apt/sources.list.d/docker.sources
echo "URIs: https://download.docker.com/linux/ubuntu" >> /etc/apt/sources.list.d/docker.sources
echo "Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")" >> /etc/apt/sources.list.d/docker.sources
echo "Components: stable" >> /etc/apt/sources.list.d/docker.sources
echo "Signed-By: /etc/apt/keyrings/docker.asc"  >> /etc/apt/sources.list.d/docker.sources

apt update && apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Set up any additional services
echo "[program:docker]" > /etc/supervisor/conf.d/docker.conf
echo "command=/usr/bin/dockerd -H unix:///var/run/docker.sock" >> /etc/supervisor/conf.d/docker.conf
echo "autostart=true" >> /etc/supervisor/conf.d/docker.conf
echo "autorestart=true" >> /etc/supervisor/conf.d/docker.conf
echo "stderr_logfile=/var/log/docker.err.log" >> /etc/supervisor/conf.d/docker.conf
echo "stdout_logfile=/var/log/docker.out.log" >> /etc/supervisor/conf.d/docker.conf

# Reload Supervisor
supervisorctl reload
sleep 2
supervisorctl update

# Start Docker
supervisorctl start docker
