import os
import shutil
import sys

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
path_to_objects = base_path
path_to_spec = os.path.join(base_path, 'Package.nuspec')
print( path_to_spec )
print('\n')

excluded_modules = ['MRCommonPlugins', 'MRCuda', 'MRMeshC', 'MRViewer', 'MRMeshViewer', 'MRTest', 'MRTestC']
path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')
copyright_header = open(path_to_copyright_header,'r').read()[3:]

f = open(path_to_spec, 'w')
f.write('<?xml version="1.0" encoding="utf-8"?>\n')
f.write('<package>\n')
f.write('\t<metadata>\n')

f.write('\t\t<id>MeshLib</id>\n')

f.write('\t\t<version>')
f.write(sys.argv[1][1:])
f.write('\t\t</version>\n')

f.write('\t\t<authors>AMV Consulting</authors>\n')
f.write('\t\t<owners>AMV Consulting</owners>\n')
f.write('\t\t<projectUrl>https://meshinspector.com</projectUrl>\n')
f.write('\t\t<description>Mesh processing library</description>\n')

f.write('\t\t<releaseNotes>https://github.com/MeshInspector/MeshLib/releases</releaseNotes>\n')

f.write('\t\t<copyright>')
f.write(copyright_header)
f.write('</copyright>\n')

f.write('\t</metadata>\n')

f.write('\t<files>\n')

folder = os.walk(path_to_objects)
for address, dirs, files in folder:
	for file in files:
		if ((file.endswith('.dll') and not any(map(file.startswith, excluded_modules)))):
			src = os.path.join(address,file)
			print(src)
			f.write('\t\t<file src="./source/x64/Release/')
			f.write(file)
			f.write('" target="lib/net6.0/"></file>\n')
            
f.write('\t</files>\n')
f.write('</package>\n')
f.close()
