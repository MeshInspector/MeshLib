import os
import sys

if (len(sys.argv)<2):
	print("No path to distribute given")
	sys.exit(1)

res_file = str(sys.argv[1])

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
path_to_install_folder = os.path.join(base_path,'install')

zip_cmd = 'tar -a -c -f '+ ' ' + res_file + ' install' 

os.chdir(base_path)
res = os.system(zip_cmd)

sys.exit(res)