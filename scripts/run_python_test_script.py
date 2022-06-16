import os
import sys
import platform

python_cmd = "py -3.10 "
platformSystem = platform.system()

if platformSystem == 'Linux':
	python_cmd = "python3.9 "
	if str(platform.python_version_tuple()[1]) == "10":
		platformRelease = platform.freedesktop_os_release()
		if platformRelease['VERSION_ID'].split(".")[0] == "22":
			python_cmd = "python3.10 "
elif platformSystem == 'Darwin':
	python_cmd = "python3 "

directory = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) == 1:
    directory = os.path.join(directory, "..")
    directory = os.path.join(directory, "python_test")
else:
    directory = os.path.join(directory, sys.argv[1])

print('Run scripts in folder :', directory)

res = 0
globalRes = 0
for filename in os.listdir(directory):
	if (filename.endswith(".py")):
		print("Run " + filename);
		res = os.system(python_cmd + os.path.join(directory,filename) )
		if (res != 0):
			globalRes = int(res)
			print(filename + ": Failed")
		else:
			print(filename + ": OK")

if (globalRes != 0):
	sys.exit(1)
