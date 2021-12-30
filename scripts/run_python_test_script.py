import os
import sys
import platform

python_cmd = "py -3.10 "
if (platform.system() == 'Linux' ):
	python_cmd = "python3.8 "

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
