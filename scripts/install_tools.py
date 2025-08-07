import os
import shutil
import sys

base_path = ''
path_to_install_folder = ''
path_to_includes = ''
path_to_libs = ''
path_to_app = ''
path_to_sources = ''
path_to_copyright_header = ''
path_to_objects = ''

include_extensions = ['.h','.hpp','.cuh','.ipp','']
lib_extentions = ['.lib','.pdb']
includes_src_dst = list()
includes_src_dst_thirdparty = list()
path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')

def check_python_version():
	if (sys.version_info[0] < 3 or (sys.version_info[0] ==3 and sys.version_info[1] < 8)):
		print('Need python 3.8 or newer')
		print('Current python: ' + str(sys.version_info[0])+'.'+str(sys.version_info[1])+'.'+str(sys.version_info[2]))
		return False

	return True


def create_directories():
	os.makedirs(path_to_install_folder,exist_ok=True)
	os.makedirs(path_to_includes,exist_ok=True)
	os.makedirs(path_to_libs,exist_ok=True)
	os.makedirs(path_to_app,exist_ok=True)

def same_file_extension(filename,ext):
	return (not filename.startswith('.')) and (os.path.splitext(filename)[1] == ext)

# `skipped_dir_regexes` is a list of regular expressions, such as `foo/bar/(/.*)?`. Those are paths relative to `path`, always using forward slashes.
# Subdirectories with those names are ignored. Typically those should end with `/(/.*)?` to act recursively.
def append_includes_list(path, thirdparty = False, subfolder = '', skipped_dir_regexes = [], p = False):
	folder = os.walk(path)
	for address, dirs, files in folder:
		for file in files:
			if (subfolder and subfolder not in address[len(path):]):
				continue
			relpath = os.path.relpath(address, path).replace('\\', '/')
			if any(r.match(relpath) for r in skipped_dir_regexes):
				continue
			if p:
				print("### ", relpath)
			if (any(map(same_file_extension, [file for ext in include_extensions], include_extensions))):
				src = os.path.join(address,file)
				dst = os.path.join(path_to_includes + address[len(path):],file)
				if not thirdparty:
					includes_src_dst.append((src,dst))
				else:
					includes_src_dst_thirdparty.append((src,dst))

def prepare_includes_list():
	includes_src_dst.clear()
	append_includes_list(path_to_sources)

def inject_copyright():
	copyright_header = open(path_to_copyright_header,'r').read()
	for src,dst in includes_src_dst:
		with open(dst, "r+") as f: s = f.read(); f.seek(0); f.write(copyright_header + '\n' + s)

def add_version_file(version):
	app_pathes = [os.path.join(path_to_app,'Debug'),os.path.join(path_to_app,'Release')]
	for app_path in app_pathes:
		if os.path.isdir(app_path):
			mr_version_file= open(os.path.join(app_path,'mr.version'),"w")
			mr_version_file.write(version)
			mr_version_file.close()

def copy_includes():
	print('Not implemented')
def copy_app():
	print('Not implemented')
def copy_lib():
	print('Not implemented')

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
