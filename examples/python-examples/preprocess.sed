#!/bin/sed -f
# remove the hashbang line
/^#!/ s/^.*$//
# comment out code lines
/^##/! s/^/## /
