# How to update Clang for the bindings

Update the number in `scripts/mrbind/clang_version.txt` for Linux and in `scripts/mrbind/clang_version_macos.txt` for macOS.

The Linux images that set `LLVM_PREFIX` (`docker/rockylinux8-vcpkgDockerfile`,
`docker/emscripten-generate-c-bindingsDockerfile`) build the bindings with our own PGO-optimized
Clang instead, unpacked into `/opt` from a [MeshInspector/toolchains](https://github.com/MeshInspector/toolchains)
release and pinned in those Dockerfiles - bump it there. `clang_version.txt` still drives the apt
packages everywhere else.

On Windows (MSYS2), you can only lock the latest Clang version they offer.

Perform a fresh MSYS2 installation, open the MSYS2 terminal (run `msys2.exe` in that installation), and in that terminal run:

* `pacman -Syu` to update. If it closes itself during update, restart the MSYS2 shell and run `pacman -Syu` again to finish updating.

* `pacman -S --noconfirm --needed gawk make procps-ng mingw-w64-clang-x86_64-{clang,clang-tools-extra,cmake}`.

* `scripts/mrbind/msys2_remember_current_packages.sh`.

Commit the modified files to Git: `scripts/mrbind/msys2_package_{urls,hashes}.txt`.

Running that script downloads some files to `scripts/mrbind/msys2_packages`. You can either delete them to save space, or archive them somewhere in case they stop being available for download.

Also zip and upload this MSYS2 installation to S3, for our CI to download:

* Run `pacman -Scc` to clean the package manager cache, reducing the size of the installation.

* Rename the installation directory to `msys64_meshlib_mrbind`.

* Zip it into a `.zip`. The archive must contain the installation directory, so e.g. it should contain `msys64_meshlib_mrbind/msys2.exe`.

* Upload the zip to S3 in place of the old one: https://vcpkg-export.s3.us-east-1.amazonaws.com/msys64_meshlib_mrbind.zip
