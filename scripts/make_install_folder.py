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
path_to_pybind11 = os.path.join(os.path.join(os.path.join(base_path,'thirdparty'),'pybind11'),'include')
path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')


include_extentions = ['.h','.hpp','.cuh']
not_app_extentions = ['.lib','.obj','.pdb','.obj','.exp','.iobj','.ipdb']
lib_extentions = ['.lib','.pdb']
includes_src_dst = list()
includes_src_dst_thirdparty = list()

def check_python_version():
	if (sys.version_info[0] < 3 or (sys.version_info[0] ==3 and sys.version_info[1] < 8)):
		print('Need python 3.8 or newer')
		print('Current python: ' + str(sys.version_info[0])+'.'+str(sys.version_info[1])+'.'+str(sys.version_info[2]))
		return False

	return True

def extention_is_one_of(file,extention_list):
	filename, file_extension = os.path.splitext(file)
	for ext in extention_list:
		if (file_extension == ext):
			return True
	return False


def create_directories():
	os.makedirs(path_to_install_folder,exist_ok=True)
	os.makedirs(path_to_includes,exist_ok=True)
	os.makedirs(path_to_libs,exist_ok=True)
	os.makedirs(path_to_app,exist_ok=True)

def append_incudes_list(path,thirdparty = False, subfolder = '' ):
	folder = os.walk(path)
	for address, dirs, files in folder:
		for file in files:
			if (subfolder and subfolder not in address[len(path):]):
				continue
			if (extention_is_one_of(file,include_extentions)):
				src = os.path.join(address,file)
				dst = os.path.join(path_to_includes + address[len(path):],file)
				if (thirdparty):
					includes_src_dst.append((src,dst))
				else:
					includes_src_dst_thirdparty.append((src,dst))

def prepare_includes_list():
	includes_src_dst.clear()
	includes_src_dst_thirdparty.clear()
	append_incudes_list(path_to_sources)
	append_incudes_list(path_to_phmap, True,'parallel_hashmap')
	append_incudes_list(path_to_pybind11, True)

def inject_copyright():
	copyright_header = open(path_to_copyright_header,'r').read()
	for src,dst in includes_src_dst:
		with open(dst, "r+") as f: s = f.read(); f.seek(0); f.write(copyright_header + '\n' + s)

def copy_includes():
	prepare_includes_list()
	for src,dst in includes_src_dst:
		dst_folder = os.path.dirname(dst)
		os.makedirs(dst_folder,exist_ok=True)
		shutil.copyfile(src, dst)
	inject_copyright()
	for src,dst in includes_src_dst_thirdparty:
		dst_folder = os.path.dirname(dst)
		os.makedirs(dst_folder,exist_ok=True)
		shutil.copyfile(src, dst)

def copy_app():
	shutil.copytree(os.path.join(path_to_sources,'x64'),path_to_app,dirs_exist_ok=True)
	folder = os.walk(path_to_app)
	for address, dirs, files in folder:
		for file in files:
			if (extention_is_one_of(file,not_app_extentions)):
				os.remove(os.path.join(address,file))

def add_version_file(version):
	app_pathes = [os.path.join(path_to_app,'Debug'),os.path.join(path_to_app,'Release')]
	for app_path in app_pathes:
		if os.path.isdir(app_path):
			mr_version_file= open(os.path.join(app_path,'mr.version'),"w")
			mr_version_file.write(version)
			mr_version_file.close()


def copy_lib():
	shutil.copytree(os.path.join(path_to_sources,'x64'),path_to_libs,dirs_exist_ok=True)	
	folder = os.walk(path_to_libs)
	for address, dirs, files in folder:
		for file in files:
			if ( not extention_is_one_of(file,lib_extentions)):
				os.remove(os.path.join(address,file))			

def main():
	if (not check_python_version()):
		print("Script failed!")
		return

	create_directories()
	if (not os.path.isdir(path_to_sources)):
		print(path_to_sources+" is not directory")
		print("Script failed!")
		return

	copy_includes()
	copy_app()
	copy_lib()
	version = "0.0.0.0"
	if len(sys.argv) > 1:
		version = sys.argv[1][1:]
	add_version_file(version)


main()