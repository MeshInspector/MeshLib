import os
import shutil
import sys

import install_tools as it

it.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
it.path_to_install_folder = os.path.join(it.base_path,'install_dotnet')
it.path_to_includes = os.path.join(it.path_to_install_folder,'include')
it.path_to_libs = os.path.join(it.path_to_install_folder,'lib')
it.path_to_app = os.path.join(it.path_to_install_folder,'app')
it.path_to_sources = os.path.join(os.path.join(it.base_path,'source'), 'MRDotNet')
it.path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')
it.path_to_objects = os.path.join(os.path.join(it.base_path,'source'), 'x64')

excluded_modules = ['MRCommonPlugins', 'MRCuda', 'MRMeshC', 'MRViewer', 'MRMeshViewer', 'MRTest', 'MRTestC']

def create_directories():
	os.makedirs(it.path_to_install_folder,exist_ok=True)
	os.makedirs(it.path_to_includes,exist_ok=True)
	os.makedirs(it.path_to_libs,exist_ok=True)
	os.makedirs(os.path.join(it.path_to_libs, 'Debug'),exist_ok=True)
	os.makedirs(os.path.join(it.path_to_libs, 'Release'),exist_ok=True)
	os.makedirs(it.path_to_app,exist_ok=True)
	os.makedirs(os.path.join(it.path_to_app, 'Debug'),exist_ok=True)
	os.makedirs(os.path.join(it.path_to_app, 'Release'),exist_ok=True)
    
it.create_directories = create_directories
        
def copy_includes():
	it.prepare_includes_list()
	for src,dst in it.includes_src_dst:
		dst_folder = os.path.dirname(dst)
		os.makedirs(dst_folder,exist_ok=True)
		shutil.copyfile(src, dst)
	it.inject_copyright()
        
def copy_app():
    configs = ['Debug','Release'];
    for config in configs:
        path_to_objects = os.path.join(it.path_to_objects,config)
        objFolder = os.walk(path_to_objects)    
        path_to_app = os.path.join(it.path_to_app,config)
        for address, dirs, files in objFolder:
            for file in files:
                if ((file.endswith('.dll') and not any(map(file.startswith, excluded_modules)))):
                    src = os.path.join(address,file)
                    dst = os.path.join(path_to_app,file)
                    shutil.copyfile(src, dst)
            
def copy_lib():
    configs = ['Debug','Release'];
    for config in configs:
        path_to_objects = os.path.join(it.path_to_objects,config)
        objFolder = os.walk(path_to_objects)
        
        path_to_libs = os.path.join(it.path_to_libs,config)
        folder = os.walk(path_to_libs)
        for address, dirs, files in objFolder:
            for file in files:
                if (any(map(file.endswith, it.lib_extentions)) and not any(map(file.startswith, excluded_modules))):
                    src = os.path.join(address,file)
                    dst = os.path.join(path_to_libs,file)
                    shutil.copyfile(src, dst)
                
it.copy_includes = copy_includes
it.copy_app = copy_app
it.copy_lib = copy_lib
                
it.main()