import os
import sys


configuration = 'Release'
if (len(sys.argv)>1):
	if (sys.argv[1]=='Debug'):
		configuration = 'Debug'

arch = sys.argv[2] if len(sys.argv) > 2 else 'x64'

run_tests_cmd = 'source\\'+arch+'\\'+configuration+'\\MRTestC2.exe'

res = os.system(run_tests_cmd)

sys.exit(res)
