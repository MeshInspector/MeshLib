#!/bin/bash
# pack ubuntu python libs to wheel
# input args
# ${1} - manylinux version
# ${2} - packet version
# strongly recommended to run in python virtualenv

# clear directory
rm -rf ./scripts/wheel/meshlib
mkdir -p ./scripts/wheel/meshlib/meshlib/

# copy python libs
cp ./build/Release/bin/meshlib/mr*.so ./scripts/wheel/meshlib/meshlib/

# copy readme and license
cp readme.md ./scripts/wheel/meshlib/
cp LICENSE ./scripts/wheel/meshlib/

cd ./scripts/wheel/meshlib || return
cp ../setup.py .
touch ./meshlib/__init__.py

# update version in setup.py
VERSION_LINE_FIND="version=''"
VERSION_LINE="version=\"${2:1}\""
sed -i "s/$VERSION_LINE_FIND/$VERSION_LINE/" ./setup.py

PY_VERSION="$(python3 -c 'import sys;print(str(sys.version_info[0])+"."+str(sys.version_info[1]))')"
# update python version in setup.py
VERSION_LINE_FIND="python_requires=''"
VERSION_LINE="python_requires=\"==${PY_VERSION}.*\""
sed -i "s/$VERSION_LINE_FIND/$VERSION_LINE/" ./setup.py

pwd
ls -al .
ls -al ./meshlib
python3 -m pip install auditwheel wheel setuptools
python3 setup.py bdist_wheel
ls -al ./dist
python3 -m auditwheel repair --plat "${1}" ./dist/*.whl