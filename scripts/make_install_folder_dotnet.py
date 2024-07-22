import os
import shutil
import sys

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
path_to_install_folder = os.path.join(base_path,'install')
path_to_includes = os.path.join(path_to_install_folder,'include')
path_to_libs = os.path.join(path_to_install_folder,'lib')
path_to_app = os.path.join(path_to_install_folder,'app')
path_to_sources = os.path.join(base_path,'source')
path_to_phmap = os.path.join(os.path.join(base_path,'thirdparty'),'parallel-hashmap')
path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')

include_extentions = ['.h','.hpp']
not_app_extentions = ['.lib','.obj','.pdb','.obj','.exp','.iobj','.ipdb']
lib_extentions = ['.lib','.pdb']
includes_src_dst = list()
includes_src_dst_thirdparty = list()

def vcpkg_dir():
	vcpkg_exe_dir = ""
	if len(sys.argv) > 2:
		vcpkg_exe_dir = sys.argv[2]
	else:
		vcpkg_exe_dir = os.popen("where vcpkg").read().strip()
		if "vcpkg.exe" not in vcpkg_exe_dir:
			vcpkg_exe_dir = "C:\\vcpkg"
		else:
			vcpkg_exe_dir = os.path.dirname( vcpkg_exe_dir )
	return os.path.join(os.path.join(vcpkg_exe_dir, "installed"),"x64-windows-meshlib")


vcpkg_dirercotry = vcpkg_dir()
print (vcpkg_dirercotry)