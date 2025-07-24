import os
import sys


configuration = 'Release'
if (len(sys.argv)>1):
	if (sys.argv[1]=='Debug'):
		configuration = 'Debug'

run_tests_cmd = 'source\\x64\\'+configuration+'\\MRTestC2.exe'

res = os.system(run_tests_cmd)

sys.exit(res)
