#!/bin/bash
while :; do
  echo "=== $(date -u +%FT%TZ)"
  pmset -g therm
  pmset -g batt
  uptime
  ps -Ao %cpu,comm -r | head -8
  sleep 10
done
