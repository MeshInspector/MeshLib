#!/bin/bash
set -eo pipefail

LIB_PATH="$1"

FILE_IS_MODIFIED=0
for OLD_DEP_RPATH in $(otool -l $LIB_PATH | egrep -o "@loader_path/\.\./dummy.dylibs/.*\.dylib") ; do
  NEW_DEP_RPATH=$(echo $OLD_DEP_RPATH | sed -e 's @loader_path/../dummy.dylibs/ @loader_path/ ')
  echo "$LIB_PATH: patching $OLD_DEP_RPATH to $NEW_DEP_RPATH..."
  install_name_tool -change $OLD_DEP_RPATH $NEW_DEP_RPATH $LIB_PATH
  FILE_IS_MODIFIED=1
done
if [ $FILE_IS_MODIFIED -eq 1 ] ; then
  codesign --force --deep --preserve-metadata=entitlements,requirements,flags,runtime --sign - $LIB_PATH
fi
