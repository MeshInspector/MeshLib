import os
import sys


configuration = 'Release'
if (len(sys.argv)>1):
	if (sys.argv[1]=='Debug'):
		configuration = 'Debug'
	if (sys.argv[1]=='Debug_vs2019'
		configuration = 'Debug_vs2019'
	if (sys.argv[1]=='Release_vs2019'
		configuration = 'Release_vs2019'


run_tests_cmd = 'source\\x64\\'+configuration+'\\MRTest.exe'

res = os.system(run_tests_cmd)

sys.exit(res)
