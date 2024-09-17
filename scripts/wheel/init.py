import pathlib as _pathlib

import meshlib.mrmeshpy as _mr

# override resources directory to the package's dir
_mr.SystemPath.overrideDirectory(_mr.SystemPathDirectory.Resources, _pathlib.Path(__file__).parent.resolve())
