import os
import shutil
import sys

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
path_to_install_folder = os.path.join(base_path,'install_dotnet')
path_to_includes = os.path.join(path_to_install_folder,'include')
path_to_libs = os.path.join(path_to_install_folder,'lib')
path_to_app = os.path.join(path_to_install_folder,'app')
path_to_sources = os.path.join(os.path.join(base_path,'source'), 'MRDotNet')
path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')
path_to_objects = os.path.join(os.path.join(base_path,'source'), 'x64')

lib_extentions = ['.lib','.pdb']
includes_src_dst = list()

excluded_modules = ['MRCommonPlugins', 'MRCuda', 'MRMeshC', 'MRViewer', 'MRMeshViewer', 'MRTest', 'MRTestC']

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
    
def append_includes_list(path):
    folder = os.walk(path)
    for address, dirs, files in folder:
        for file in files:
            if (file.endswith('.h')):
                src = os.path.join(address,file)
                dst = os.path.join(path_to_includes + address[len(path):],file)
                includes_src_dst.append((src,dst))
 
def prepare_includes_list():
	includes_src_dst.clear()
	append_includes_list(path_to_sources)
    
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
        
def copy_app():
    objFolder = os.walk(path_to_objects)    
    for address, dirs, files in objFolder:
        for file in files:
            if ((file.endswith('.dll') and not any(map(file.startswith, excluded_modules))) or (file == 'MRDotNetTest.exe')):
                src = os.path.join(address,file)
                dst = os.path.join(path_to_app,file)
                shutil.copyfile(src, dst)
                
def add_version_file(version):
	app_pathes = [os.path.join(path_to_app,'Debug'),os.path.join(path_to_app,'Release')]
	for app_path in app_pathes:
		if os.path.isdir(app_path):
			mr_version_file= open(os.path.join(app_path,'mr.version'),"w")
			mr_version_file.write(version)
			mr_version_file.close()
            
def copy_lib():
	objFolder = os.walk(path_to_objects)
	folder = os.walk(path_to_libs)
	for address, dirs, files in objFolder:
		for file in files:
			if (any(map(file.endswith, lib_extentions)) and not any(map(file.startswith, excluded_modules))):
				src = os.path.join(address,file)
				dst = os.path.join(path_to_libs,file)
				shutil.copyfile(src, dst)

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