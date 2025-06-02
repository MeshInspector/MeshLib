#!/bin/bash

set -euxo pipefail

if [ -d "./Library" ];
  then rm -rf "./Library";
fi

cd ./build/Release
cmake --install . --prefix=../..
cd -
MR_VERSION=$(ls ./Library/Frameworks/MeshLib.framework/Versions/)
echo "version: ${MR_VERSION}"
MR_PREFIX="./Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}"
echo "prefix: ${MR_PREFIX}"

cp -rL ./lib "${MR_PREFIX}/lib/"
cp -rL ./include "${MR_PREFIX}/include/"

cp ./LICENSE ./macos/Resources
mkdir "${MR_PREFIX}"/requirements/
cp ./requirements/macos.txt "${MR_PREFIX}"/requirements/

ln -s "/Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}" "./Library/Frameworks/MeshLib.framework/Versions/Current"
ln -s "/Library/Frameworks/MeshLib.framework/Resources" "./Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}/Resources"

# be carefull with pkg names! The pkg can fail to build
pkgbuild \
            --root Library \
            --identifier com.MeshInspector.MeshLib \
            --install-location  /Library \
            MeshLib.pkg


productbuild \
          --distribution ./macos/Distribution.xml \
          --package-path ./MeshLib.pkg \
          --resources ./macos/Resources \
          MeshLib_.pkg