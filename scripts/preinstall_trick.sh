#!/bin/bash

# This dirty trick allows to fix default boost signals2 header for C++20 campatability
# `distribution.sh` uses this script as preinstall

# fix boost signal2 C++20 error in default version 1.71.0 from `apt`
# NOTE: 1.75+ version already has this fix
# https://github.com/boostorg/signals2/commit/15fcf213563718d2378b6b83a1614680a4fa8cec
FILENAME=/usr/include/boost/signals2/detail/auto_buffer.hpp
cat $FILENAME | tr '\n' '\r' | \
sed -e 's/\r        typedef typename Allocator::pointer              allocator_pointer;\r/\
#ifdef BOOST_NO_CXX11_ALLOCATOR\
        typedef typename Allocator::pointer allocator_pointer;\
#else\
        typedef typename std::allocator_traits<Allocator>::pointer allocator_pointer;\
#endif\
/g' | tr '\r' '\n' | sudo tee $FILENAME
