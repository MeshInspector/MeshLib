# How to update Clang for the bindings

Update the number in `scripts/mrbind/clang_version.txt`. This affects Mac and Linux.

On Windows (MSYS2), you can only lock the latest Clang version they offer.

Perform a fresh MSYS2 installation, and in there run:

* `pacman -Syu` to update. If it closes itself during update, restart the MSYS2 shell and run `pacman -Syu` again to finish updating.

* `pacman -S --noconfirm --needed gawk make procps-ng mingw-w64-clang-x86_64-{clang,clang-tools-extra,cmake}`.

* Run `scripts/mrbind/msys2_store_current_packages.sh`.

* Commit the modified files to Git: `scripts/mrbind/msys2_package_{urls,hashes}.txt`
