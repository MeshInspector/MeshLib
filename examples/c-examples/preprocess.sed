#!/bin/sed -f
# comment out code lines
\:^///:! s:^:/// :
