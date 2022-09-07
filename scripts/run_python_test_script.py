import os
import sys
import platform

python_cmd = "py -3.10 "
platformSystem = platform.system()

if platformSystem == 'Linux':
	python_cmd = "python3 "

	os_name=""
	os_version=""
	if os.path.exists('/etc/os-release'):
		lines = open('/etc/os-release').read().split('\n')
		for line in lines:
			if line.startswith('NAME='):
				os_name=line.split('=')[-1].replace('"', '')
			if line.startswith('VERSION_ID='):
				os_version=line.split('=')[-1].replace('"', '')

	if "ubuntu" in os_name.lower():
		if os_version.startswith("20"):
			python_cmd = "python3.8 "
		elif os_version.startswith("22"):
			python_cmd = "python3.10 "
	elif "fedora" in os_name.lower():
		if os_version.startswith("35"):
			python_cmd = "python3.9 "

elif platformSystem == 'Darwin':
	python_cmd = "python3 "

if platformSystem == "Windows":
	import glob
	import shutil

	dest_dir = os.path.join(os.getcwd(), "meshlib")
	if not os.path.exists(dest_dir):
		print("Creating " + str(dest_dir))
		os.makedirs(dest_dir)
	print("Copying files...")
	for file in glob.glob(r'*.dll'):
		print(file)
		shutil.copy(file, dest_dir)
	for file in glob.glob(r'*py*'):
		print(file)
		shutil.copy(file, dest_dir)

directory = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) == 1:
    directory = os.path.join(directory, "..")
    directory = os.path.join(directory, "test_python")
else:
    directory = os.path.join(directory, sys.argv[1])

os.environ["MeshLibPyModulesPath"] = os.getcwd()
os.chdir(directory)

res = os.system(python_cmd + "-m pytest -s -v"  )

if (res != 0):
	sys.exit(1)
