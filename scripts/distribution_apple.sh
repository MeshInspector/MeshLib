#!/bin/bash


bundle_dylib() {
    local fix_file="$1"
    local dest_dir="$2"
    shift 2  # Shift the first two arguments to capture the rest as search paths
    local search_paths=("$@") # can be multiple

    if [[ -z "$fix_file" ]]; then
        echo "Error: fix_file is required."
        return 1
    fi
    if [[ -z "$dest_dir" ]]; then
        echo "Error: dest_dir is required."
        return 1
    fi

    local search_paths_args=""
    for path in "${search_paths[@]}"; do
        search_paths_args+=" --search-path \"$path\""
    done

    echo "Fixing MeshLib executable @rpath"

    echo "dylibbundler \
                  --bundle-deps \
                  --create-dir \
                  --overwrite-files \
                  --fix-file ${fix_file} \
                  --dest-dir ${dest_dir} \
                  ${MISSED_LIBS} \
                  ${search_paths_args} "

    dylibbundler \
        --bundle-deps \
        --create-dir \
        --overwrite-files \
        --fix-file "${fix_file}" \
        --dest-dir "${dest_dir}" \
        ${MISSED_LIBS} \
        ${search_paths_args}
}

echo "Installing required brew pkgs"
brew install dylibbundler jq

if [ -d "./Library" ];
  then rm -rf "./Library";
fi

cd ./build/Release
cmake --install . --prefix=../..
cd -

MR_VERSION=$(ls ./Library/Frameworks/MeshLib.framework/Versions/)
MR_PREFIX="./Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}"

echo "version: ${MR_VERSION}"
echo "prefix: ${MR_PREFIX}"

cp -rL ./lib/ "${MR_PREFIX}/libs/"
cp -rL ./include "${MR_PREFIX}/"

cp ./LICENSE ./macos/Resources

mkdir "${MR_PREFIX}"/requirements/
cp ./requirements/macos.txt "${MR_PREFIX}"/requirements/
cp ./requirements/distribution_python.txt "${MR_PREFIX}"/requirements/python.txt

mkdir "${MR_PREFIX}"/share/
cp -r "$(brew --prefix)"/share/glib-2.0 "${MR_PREFIX}"/share

ln -s "./Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}" "./Library/Frameworks/MeshLib.framework/Versions/Current"
ln -s "./Library/Frameworks/MeshLib.framework/Resources" "./Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}/Resources"

echo "Embedded python: processing requirements"
if [ -f "python.py" ];
  then rm "python.py";
fi
touch "python.py"
for PYTHON_LIB_NAME in $(cat "${MR_PREFIX}"/requirements/python.txt)
do
  PYTHON_LIB_NAME=${PYTHON_LIB_NAME%%>*}
  PYTHON_LIB_NAME=${PYTHON_LIB_NAME%%=*}
  echo "$PYTHON_LIB_NAME"
  echo "import $PYTHON_LIB_NAME" >> python.py
done

python3.10 -m pip install --upgrade pip
python3.10 -m pip install virtualenv
python3.10 -m venv ./venv
source ./venv/bin/activate
python3.10 -m pip install pyinstaller
if [ -d "./dist" ];
  then rm -rf "./dist";
fi
pyinstaller ./python.py --distpath ./dist

echo "Embedded python: changing libMRMesh.dylib embedded python path"
PYTHON_PATH=$(otool -L ./build/Release/bin/libMRMesh.dylib | grep 'Python' | cut -d ' ' -f 1 | sed 's/[[:blank:]]//g')
NEW_PYTHON_PATH="@rpath/../python/_internal/Python.framework/Versions/3.10/Python"
echo "old: $PYTHON_PATH, new: $NEW_PYTHON_PATH"
install_name_tool -change "$PYTHON_PATH" "$NEW_PYTHON_PATH" ./build/Release/bin/libMRMesh.dylib
echo "Done"
#otool -L ./build/Release/bin/libMRMesh.dylib

echo "Fixing main libs @rpath"
MISSED_LIBS=""
# no MRCommonPlugins because MRPlugins will handle it
# no MRCUDAPlugins because it is not built for mac
for LIB_NAME in ./build/Release/bin/libMRCommonPlugins.dylib
do
  LIB_NAME="${LIB_NAME##*/}"
  echo "$LIB_NAME"
  # cp ./build/Release/bin/"${LIB_NAME}" ${APPNAME}/Contents/libs/
  MISSED_LIBS="${MISSED_LIBS} -x ${MR_PREFIX}/libs/${LIB_NAME}"
done
# python modules
for LIB_NAME in ${MR_PREFIX}/libs/meshlib/*.so
do
  LIB_NAME="${LIB_NAME##*/}"
  echo "$LIB_NAME"
  MISSED_LIBS="${MISSED_LIBS} -x ${MR_PREFIX}/libs/meshlib/${LIB_NAME}"
done

echo "$MISSED_LIBS"

echo "Fixing MeshLib executable @rpath" -x ${MR_PREFIX}/bin/meshconv
# dylibbundler -b -cd -of -x ${MR_PREFIX}/bin/MeshViewer  -s ./build/Release/bin/ -s ./lib -s ./dist/python -d ${MR_PREFIX}/lib/ ${MISSED_LIBS}

bin_dir="${MR_PREFIX}/bin/"
dest_dir="${MR_PREFIX}/libs/"
search_paths=("./build/Release/bin/" "./lib/" "./dist/python/" "$dest_dir")

for binary in "$bin_dir"*; do
    if [[ -x "$binary" && -f "$binary" ]]; then
        bundle_dylib "$binary" "$dest_dir" "${search_paths[@]}"
    fi
done

deactivate


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