Name:           meshlib-dev
Version:
Release:        1
Summary:        Advanced mesh modeling tools

License:        Proprietary
BuildRoot:      ./rpmbuild/
BuildArch:	    x86_64

Requires:

%description
Advanced mesh modeling tools

%define _rpmfilename %%{NAME}.rpm

%prep
printf "start prep\n"
echo "BUILDROOT = $RPM_BUILD_ROOT"
cd ../..
echo $PWD

MR_INSTALL_LIB_DIR="$RPM_BUILD_ROOT/usr/local/lib/MeshLib/"
MR_INSTALL_RES_DIR="$RPM_BUILD_ROOT/usr/local/etc/MeshLib/"
MR_INSTALL_INCLUDE_DIR="$RPM_BUILD_ROOT/usr/local/include/MeshLib/"

cd ./build/Release
cmake --install . --prefix "$RPM_BUILD_ROOT/usr/local/"
cd -

#mkdirs
mkdir -p "${MR_INSTALL_LIB_DIR}"
mkdir -p "${MR_INSTALL_INCLUDE_DIR}"
mkdir -p "${MR_INSTALL_RES_DIR}"

#copy lib dir
CURRENT_DIR="`pwd`"
cp -rL ./lib "${MR_INSTALL_LIB_DIR}"
cp -rL ./include "${MR_INSTALL_INCLUDE_DIR}"
printf "lib copy done\n"

#copy verison file
cp build/Release/bin/mr.version "${MR_INSTALL_RES_DIR}"
printf "MR version copy done\n"

cd "${RPM_BUILD_ROOT}"
exit

%files
/usr/local/bin/
/usr/local/include/MeshLib/
/usr/local/lib/MeshLib/
/usr/local/etc/MeshLib/
/usr/local/share/fonts/

%post
# This script adds MR libs symbolic links to python3
# Expand ld search paths, if `/usr/local/lib` is not added to default

# exit if any command failed
set -eo pipefail

#TODO: handle 'home' python installations (conda, ...)
if [ -d /usr/lib/python3.9 ]; then
 printf "\rPython3 was found                       \n"
 if [ "$EUID" -ne 0 ]; then
  printf "Root access required!\n"
  RUN_AS_ROOT="NO"
 fi
 sudo mkdir -p /usr/lib/python3.9/site-packages/meshlib/
 sudo ln -sf /usr/local/lib/MeshLib/meshlib/mrmeshpy.so /usr/lib/python3.9/site-packages/meshlib/mrmeshpy.so
 sudo ln -sf /usr/local/lib/MeshLib/meshlib/mrmeshnumpy.so /usr/lib/python3.9/site-packages/meshlib/mrmeshnumpy.so
 sudo ln -sf /usr/local/lib/MeshLib/meshlib/mrviewerpy.so /usr/lib/python3.9/site-packages/meshlib/mrviewerpy.so
 printf "Python3 has symlink to MR libs. Run 'sudo ln -sf /usr/local/lib/MeshLib/mr<lib_name>py.so /<pathToPython>/site-packages/meshlib/mr<lib_name>py.so' for custom python installations\n"
fi

printf "Updating ldconfig for '/usr/local/lib/MeshLib'\n"
echo "/usr/local/lib/MeshLib" | sudo tee /etc/ld.so.conf.d/local_libs.conf
sudo ldconfig

%clean
rm -rf $RPM_BUILD_ROOT


%changelog
