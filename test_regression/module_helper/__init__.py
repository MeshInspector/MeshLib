import os
import sys

# insert 0 to find mrpy.so in working directory and not in system
if "MeshLibPyModulesPath" in os.environ and not os.environ["MeshLibPyModulesPath"] in sys.path:
    sys.path.insert(0, os.environ["MeshLibPyModulesPath"])

is_new_binding = False

if os.environ.get('USE_MESHLIB2_PY', '0') != '0':
    is_new_binding = True
    from meshlib2 import mrmeshpy
    from meshlib2 import mrmeshnumpy
else:
    import meshlib.mrmeshpy
    import meshlib.mrmeshnumpy
