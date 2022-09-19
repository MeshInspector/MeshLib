import setuptools
import pathlib
import argparse
import os
import platform
import sys


platform_system = platform.system()
print(platform_system)

VERSION = ""

if '--version' in sys.argv:
    index = sys.argv.index('--version')
    sys.argv.pop(index)  # Removes the '--foo'
    VERSION = sys.argv.pop(index)  # Returns the element after the '--foo'

PY_VERSION=str(sys.version_info[0]) + "." + str(sys.version_info[1])
LIBS_EXTENSION = ""
SYSTEM = ""

if platform_system == "Windows":
    LIBS_EXTENSION = ".pyd"
    SYSTEM = "Microsoft :: Windows"
elif platform_system == "Linux":
    LIBS_EXTENSION = ".so"
    SYSTEM = "POSIX :: Linux"
elif platform_system == "Darwin":
    LIBS_EXTENSION = ".so"
    SYSTEM = "MacOS"

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the readme file
long_description = (here / "readme.md").read_text(encoding="utf-8")

setuptools.setup(
    name="meshlib",
    version=VERSION,
    author="MeshLib Team",
    author_email="support@meshinspector.com",
    description="3d processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeshInspector/MeshLib",
    license_files=('LICENSE',),
    packages=['meshlib'],
    package_data={'meshlib': ['mrmeshnumpy.{ext}', 'mrmeshpy.{ext}', 'mrviewerpy.{ext}'.format(ext=LIBS_EXTENSION)]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: {}".format(PY_VERSION),
        "License :: Free for non-commercial use",
        "License :: Free For Educational Use",
        "Operating System :: {}".format(SYSTEM),
    ],
    python_requires='=={}.*'.format(PY_VERSION),
    install_requires=[
        'numpy>=1.21.0',
        'pytest>=7.1.0',
    ],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/MeshInspector/MeshLib/issues",
        "Source": "https://github.com/MeshInspector/MeshLib",
    },
)