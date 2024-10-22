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
echo "start prep"
echo "BUILDROOT = $RPM_BUILD_ROOT"
cd ../..
echo $PWD

MR_INSTALL_LIB_DIR="$RPM_BUILD_ROOT/usr/local/lib64/MeshLib/"
MR_INSTALL_RES_DIR="$RPM_BUILD_ROOT/usr/local/share/MeshLib/"
MR_INSTALL_INCLUDE_DIR="$RPM_BUILD_ROOT/usr/local/include/MeshLib/"

cmake --install ./build/Release/ --prefix "$RPM_BUILD_ROOT/usr/local"

# copy lib dir
cp -rL ./lib "${MR_INSTALL_LIB_DIR}"
cp -rL ./include "${MR_INSTALL_INCLUDE_DIR}"
echo "lib copy done"

# copy version file
cp build/Release/bin/mr.version "${MR_INSTALL_RES_DIR}"
echo "MR version copy done"

# copy udev rules
mkdir -p "${RPM_BUILD_ROOT}/usr/local/lib64/udev/rules.d/"
cp "./scripts/70-space-mouse-meshlib.rules" "${RPM_BUILD_ROOT}/usr/local/lib64/udev/rules.d/"

cd "${RPM_BUILD_ROOT}"
exit

%files
/usr/local/bin/MeshViewer
/usr/local/bin/meshconv
/usr/local/include/MeshLib/
/usr/local/lib64/MeshLib/
/usr/local/lib64/cmake/MeshLib/
/usr/local/lib64/udev/rules.d/70-space-mouse-meshlib.rules
/usr/local/share/MeshLib/

%post
if command -v udevadm 2>&1 >/dev/null ; then
  echo "Updating udev rules"
  udevadm control --reload-rules && udevadm trigger
fi

echo "Updating ldconfig"
cat << EOF > /etc/ld.so.conf.d/meshlib_libs.conf
/usr/local/lib64/MeshLib
/usr/local/lib64/MeshLib/lib
EOF
ldconfig

%clean
rm -rf $RPM_BUILD_ROOT

%changelog
