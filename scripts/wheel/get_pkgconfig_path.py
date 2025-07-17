#!/usr/bin/env python3
import sysconfig

if __name__ == '__main__':
    print(sysconfig.get_config_var('LIBPC'))
