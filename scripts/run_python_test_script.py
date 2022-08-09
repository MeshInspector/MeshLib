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

res = os.system(python_cmd + "-m pytest -s -v" )

if (globalRes != 0):
	sys.exit(1)
