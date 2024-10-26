#!/bin/bash

# exit if any command failed
set -eo pipefail

create_framework_dir() {
  if [ -d "./Library" ];
    then rm -rf "./Library";
  fi

  cmake --install ./build/Release  --prefix "."

  MR_VERSION=$(ls ./Library/Frameworks/MeshLib.framework/Versions/)
  MR_PREFIX="./Library/Frameworks/MeshLib.framework/Versions/${MR_VERSION}"

  echo "version: ${MR_VERSION}"
  echo "prefix: ${MR_PREFIX}"

  mkdir -p "${MR_PREFIX}/libs/"

  cp -rL ./include "${MR_PREFIX}/"

  cp ./macos/Info.plist ./macos/Resources
  cp ./LICENSE ./macos/Resources

  mkdir -p "${MR_PREFIX}"/requirements/
  cp ./requirements/macos.txt "${MR_PREFIX}"/requirements/
  cp ./requirements/distribution_python.txt "${MR_PREFIX}"/requirements/python.txt

  mkdir -p "${MR_PREFIX}"/share/
  cp -r "$(brew --prefix)"/share/glib-2.0 "${MR_PREFIX}"/share

  cd ./Library/Frameworks/MeshLib.framework/Versions/
  ln -s "${MR_VERSION}" ./Current
  cd -
}

embed_python() {
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

  echo "Embedded python: changing libMRPython.dylib embedded python path"
  PYTHON_PATH=$(otool -L ./build/Release/bin/libMRPython.dylib | grep 'Python' | cut -d ' ' -f 1 | sed 's/[[:blank:]]//g')
  NEW_PYTHON_PATH="@rpath/../python/_internal/Python.framework/Versions/3.10/Python"
  echo "old: $PYTHON_PATH, new: $NEW_PYTHON_PATH"
  install_name_tool -change "$PYTHON_PATH" "$NEW_PYTHON_PATH" ./build/Release/bin/libMRMesh.dylib
  deactivate
  echo "Done"
}

pack_dylibs() {
  echo "Fixing MeshLib executable @rpath" -x ${MR_PREFIX}/bin/meshconv

  bin_dir="${MR_PREFIX}/bin/"
  lib_dir="${MR_PREFIX}/libs/"
  dest_dir="${lib_dir}"
  search_paths=("${dest_dir}" "./lib/" "./dist/python")

  for binary in "$bin_dir"*; do
      if [[ -x "$binary" && -f "$binary" ]]; then
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
}

bundle_dylib() {
  local fix_file="$1"
  local dest_dir="$2"
  shift 2  # Shift the first two arguments to capture the rest as search paths
  local search_paths=("$@") # can be multiple

  if [[ -z "$fix_file" ]] || [[ -z "$dest_dir" ]]; then
    echo "Error: fix_file and dest_dir are required for bundle_dylib."
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

  echo "Fixing MeshLib executable @rpath"

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
brew install dylibbundler

create_framework_dir
embed_python
pack_dylibs

echo "Framework creation: done"
