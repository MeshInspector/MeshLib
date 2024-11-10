import os
import sys

if (len(sys.argv)<2):
	print("No path to the folder given")
	sys.exit(1)
    
if (len(sys.argv)<3):
	print("No path to distribute given")
	sys.exit(1)

src_folder = str(sys.argv[1])
src_folder2 = str("")
if (len(sys.argv)>3):
	src_folder2 = " " + str(sys.argv[2])
res_file = str(sys.argv[len(sys.argv)-1])

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
path_to_install_folder = os.path.join(base_path,src_folder)

zip_cmd = 'tar -a -c -f '+ ' ' + res_file + ' ' + src_folder + src_folder2

os.chdir(base_path)
res = os.system(zip_cmd)

sys.exit(res)