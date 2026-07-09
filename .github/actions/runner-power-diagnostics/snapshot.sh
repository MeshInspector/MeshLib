#!/bin/bash
pmset -g
pmset -g batt
pmset -g therm
uptime
ps -Ao %cpu,%mem,etime,comm -r | head -16
vm_stat | head -8
sysctl vm.swapusage
