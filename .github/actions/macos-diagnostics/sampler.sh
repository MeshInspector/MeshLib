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
  top -l 2 -o cpu -n 8 -stats pid,cpu,command | awk '/^Processes:/ { n++ } n == 2'
  sleep 10
done
