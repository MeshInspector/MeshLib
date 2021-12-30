# Script prints all available functions in 'mrpy' with their descriptions

import mrmeshpy as mrmesh

funcs = dir (mrmesh)
for f in funcs:
    if not f.startswith('_'):
        print( help( "mrmeshpy." + f ) )
        print( "\n" )
