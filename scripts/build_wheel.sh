#!/bin/bash
set -euo pipefail

MANYLINUX_VERSION="2_31"
# TODO: detect platform automatically
PLATFORM="${PLATFORM:-x86_64}"

PYTHON_TAG="cp${PYTHON_VERSION/./}"
PYTHON_EXECUTABLE="python${PYTHON_VERSION}"

MESHLIB_VERSION="${MESHLIB_VERSION}"

MESHLIB_PYTHON_MODULES=(mrmeshpy mrmeshnumpy mrviewerpy)

set +e
which $PYTHON_EXECUTABLE >/dev/null
if [ $? -ne 0 ] ; then 
    echo "No Python ${PYTHON_VERSION} executable found"
    exit 1
fi
set -e

$PYTHON_EXECUTABLE -m pip install --upgrade --requirement ./requirements/distribution_python.txt
$PYTHON_EXECUTABLE -m pip install auditwheel wheel setuptools pybind11-stubgen
$PYTHON_EXECUTABLE ./scripts/wheel/setup_workspace.py
pushd ./scripts/wheel/meshlib/
    for MODULE in ${MESHLIB_PYTHON_MODULES[*]} ; do
        PYTHONPATH=. pybind11-stubgen --output-dir . meshlib.$MODULE
    done
    $PYTHON_EXECUTABLE setup.py bdist_wheel --python-tag=$PYTHON_TAG --version ${MESHLIB_VERSION}
    $PYTHON_EXECUTABLE -m auditwheel repair --plat "manylinux_${MANYLINUX_VERSION}_${PLATFORM}" ./dist/*.whl
popd

echo "Wheel files are ready:"
ls -1 ./scripts/wheel/meshlib/wheelhouse/meshlib-*.whl
