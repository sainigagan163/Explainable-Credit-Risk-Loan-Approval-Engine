#!/usr/bin/env bash
set -e

echo "Starting Credit Risk Engine..."
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
