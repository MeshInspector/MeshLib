# exit if any command failed
#set -eo pipefail

MR_THIRDPARTY_DIR="thirdparty/"

if [ ! -f "./lib/libcpr.so" ]; then
 printf "Thirdparty build was not found. Building...\n"
 ./scripts/build_thirdparty.sh
fi

if [ ! -f "./build/Release/bin/libMRMesh.so" ]; then
 printf "Project release build was not found. Building...\n"
 export MESHLIB_BUILD_RELEASE="ON"
 export MESHLIB_BUILD_DEBUG="OFF"
 ./scripts/build_source.sh
fi

#modify rpm spec file and mr.version
version=0.0.0.0
if [ ${1} ]; then
  version=${1:1} #v1.2.3.4 -> 1.2.3.4
fi
echo $version > build/Release/bin/mr.version

VERSION_LINE_FIND="Version:"
VERSION_LINE="Version:        ${version}"
sed -i "s/$VERSION_LINE_FIND/$VERSION_LINE/" ./scripts/MeshLib-dev.spec


REQUIRES_LINE="Requires:"
req_counter=0
BASEDIR=$(dirname "$0")
requirements_file="$BASEDIR"/../requirements/fedora.txt
for req in `cat $requirements_file`
do
  if [ $req_counter -le 0 ]; then
  	REQUIRES_LINE="${REQUIRES_LINE} ${req}"
  else
  	REQUIRES_LINE="${REQUIRES_LINE}, ${req}"
  fi
  ((req_counter=req_counter+1))
done

REQUIRES_LINE_FIND="Requires:"
sed -i "s/$REQUIRES_LINE_FIND/$REQUIRES_LINE/" ./scripts/MeshLib-dev.spec

#create distr dirs
if [ -d "./rpmbuild/" ]; then
 rm -rf rpmbuild
fi

mkdir -p rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
rpmbuild -bb .${MRE_PREFIX}/scripts/MeshLib-dev.spec  --define "_topdir `pwd`/rpmbuild" --target x86_64
mv rpmbuild/RPMS/meshlib-dev.rpm .

rm -rf rpmbuild
