#!/bin/bash
echo "--- pmset -g: active power settings (see lowpowermode) ---"
pmset -g
echo "--- pmset -g batt: power source & battery ---"
pmset -g batt
echo "--- pmset -g therm: thermal throttling / CPU speed limits ---"
pmset -g therm
echo "--- uptime & load averages ---"
uptime
echo "--- top processes by CPU ---"
ps -Ao %cpu,%mem,etime,comm -r | head -16
echo "--- vm_stat: memory pages ---"
vm_stat | head -8
echo "--- swap usage ---"
sysctl vm.swapusage
