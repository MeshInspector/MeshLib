import os
import shutil
import subprocess
import sys

# The .pdb files dominate the developer distributive but are only needed when
# debugging a crash, so they ship as a companion archive that keeps the same
# internal layout: unpacking it over the main one restores the full folder.

if len(sys.argv) < 3:
	print("usage: split_install_pdb.py <install folder> <pdb archive>")
	sys.exit(1)

install_folder = sys.argv[1]
pdb_archive = os.path.abspath(sys.argv[2])

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(base_path)

stage_root = install_folder + '-pdb'
shutil.rmtree(stage_root, ignore_errors=True)
stage = os.path.join(stage_root, os.path.basename(install_folder))

moved = 0
total = 0
for address, dirs, files in os.walk(install_folder):
	for file in files:
		if not file.endswith('.pdb'):
			continue
		src = os.path.join(address, file)
		dst = os.path.join(stage, os.path.relpath(address, install_folder), file)
		os.makedirs(os.path.dirname(dst), exist_ok=True)
		total += os.path.getsize(src)
		shutil.move(src, dst)
		moved += 1

print("moved " + str(moved) + " .pdb files (" + str(total >> 20) + " MiB) into " + pdb_archive)
if moved == 0:
	print("error: no .pdb files found in " + install_folder)
	sys.exit(1)

res = subprocess.call(['tar', '-a', '-c', '-f', pdb_archive, '-C', stage_root, os.path.basename(install_folder)])
shutil.rmtree(stage_root, ignore_errors=True)
sys.exit(res)
