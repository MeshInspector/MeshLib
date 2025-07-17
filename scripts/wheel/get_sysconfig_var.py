#!/usr/bin/env python3
import sysconfig

if __name__ == '__main__':
    import sys
    print(sysconfig.get_config_var(sys.argv[1]))
