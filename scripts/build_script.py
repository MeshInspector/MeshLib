import os
import sys

configuration = 'Release'
if (len(sys.argv)>1):
	if (sys.argv[1]=='Debug'):
		configuration = 'Debug'

build_cmd = '\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe\" source\\MeshLib.sln -p:Configuration=' + configuration

res = os.system(build_cmd)

sys.exit(res)
