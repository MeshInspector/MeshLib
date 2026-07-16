#!/bin/bash
echo "--- launchctl managername: Aqua = GUI login session, else user-domain XPC services are unavailable ---"
launchctl managername
launchctl print "gui/$(id -u)" >/dev/null 2>&1 && echo "gui/$(id -u) launchd domain: present" || echo "gui/$(id -u) launchd domain: MISSING (no GUI login session)"
echo "--- runner IP addresses: local IPv4 & public egress ---"
ifconfig -a | awk '/inet /{print $2}' | grep -v '^127\.' || true
curl -fsS --max-time 5 https://checkip.amazonaws.com || echo "public IP lookup failed"
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
