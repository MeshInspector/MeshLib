#!/usr/bin/env python3
"""
Helper script for retrieving non-system Python distribution info (libpython3 path, pkg-config path, etc.)
"""
import sysconfig

if __name__ == '__main__':
    import sys
    print(sysconfig.get_config_var(sys.argv[1]))
