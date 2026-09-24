#!/bin/bash
echo "--- launchctl managername: Aqua = GUI login session, else user-domain XPC services are unavailable ---"
launchctl managername
launchctl print "gui/$(id -u)" >/dev/null 2>&1 && echo "gui/$(id -u) launchd domain: present" || echo "gui/$(id -u) launchd domain: MISSING (no GUI login session)"
echo "--- console owner / autoLoginUser / FileVault (does auto-login restore the GUI session after reboot) ---"
stat -f 'console owner: %Su' /dev/console || true
echo "autoLoginUser: $(defaults read /Library/Preferences/com.apple.loginwindow autoLoginUser 2>/dev/null || echo '<not set>')"
fdesetup status || true
echo "--- runner service plists: a LaunchDaemon starts the runner outside the GUI session, a LaunchAgent inside ---"
ls -l /Library/LaunchDaemons/actions.runner.*.plist "$HOME"/Library/LaunchAgents/actions.runner.*.plist 2>/dev/null || echo "no actions.runner service plists"
echo "--- process ancestry up to launchd: which service actually runs this job ---"
pid=$$
while [ -n "$pid" ] && [ "$pid" -gt 1 ]; do ps -o pid=,ppid=,user=,comm= -p "$pid"; pid=$(ps -o ppid= -p "$pid" | tr -d ' '); done
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
