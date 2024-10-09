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
        search_paths_args+=" --search-path $path"
    done

    local install_path_arg=""
    if [[ -n "$use_install_path" ]]; then
        install_path_arg="--install-path @executable_path/../thirdparty-libs"
    fi

    echo "Fixing MeshLib executable @rpath"

    echo "dylibbundler \
                  --bundle-deps \
                  --create-dir \
                  --overwrite-files \
                  --fix-file ${fix_file} \
                  --dest-dir ${dest_dir} \
                  ${MISSED_LIBS} \
                  ${install_path_arg} "

    dylibbundler \
        --bundle-deps \
        --create-dir \
        --overwrite-files \
        --fix-file "${fix_file}" \
        --dest-dir "${dest_dir}" \
        ${search_paths_args} \
        --install-path @executable_path/../libs
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


mkdir -p "${MR_PREFIX}/thirdparty-libs/"

cp -L ./lib/*.dylib "${MR_PREFIX}/thirdparty-libs/"
cp -rL ./include "${MR_PREFIX}/"
cp ./LICENSE ./macos/Resources

mkdir "${MR_PREFIX}"/requirements/
cp ./requirements/macos.txt "${MR_PREFIX}"/requirements/
cp ./requirements/distribution_python.txt "${MR_PREFIX}"/requirements/python.txt

mkdir "${MR_PREFIX}"/share/
cp -r "$(brew --prefix)"/share/glib-2.0 "${MR_PREFIX}"/share

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

cd ./Library/Frameworks/MeshLib.framework/Versions/
ln -s ${MR_VERSION} Current
cd -

echo "Fixing MeshLib executable @rpath" -x ${MR_PREFIX}/bin/meshconv

bin_dir="/Users/maxraiskii/CLionProjects/MeshLib/Library/Frameworks/MeshLib.framework/Versions/0.0.0/bin/"
lib_dir="./Library/Frameworks/MeshLib.framework/Versions/0.0.0/libs"
dest_dir="./Library/Frameworks/MeshLib.framework/Versions/0.0.0/libs"
search_paths=("${dest_dir}" "/Users/maxraiskii/CLionProjects/MeshLib/lib/" "./dist/python")


for binary in "$bin_dir"*; do
    if [[ -x "$binary" && -f "$lib" ]]; then
        bundle_dylib "$binary" "$dest_dir" "${search_paths[@]}"
    fi
done

for lib in "$lib_dir"*; do
    if [[ -x "$lib" && -f "$lib" ]]; then
        bundle_dylib "$lib" "$dest_dir" "${search_paths[@]}"
    fi
done

for lib in "$lib_dir"/meshlib/*; do
    if [[ -x "$lib" && -f "$lib" ]]; then
        bundle_dylib "$lib" "$dest_dir" "${search_paths[@]}"
    fi
done

deactivate


pkgbuild \
            --root Library \
            --identifier com.MeshInspector.MeshLib \
            --install-location /Library \
            MeshLib.pkg


productbuild \
          --distribution ./macos/Distribution.xml \
          --package-path ./MeshLib.pkg \
          --resources ./macos/Resources \
          MeshLib_.pkg