#!/bin/bash
while :; do
  echo "=== sample $(date -u +%FT%TZ)"
  echo "-- thermal throttling / CPU speed limits --"
  pmset -g therm
  echo "-- power source & battery --"
  pmset -g batt
  echo "-- load averages --"
  uptime
  echo "-- top processes by CPU --"
  ps -Ao %cpu,comm -r | head -8
  sleep 10
done
