# used in pair with create_wheel.sh
import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the readme file
long_description = (here / "readme.md").read_text(encoding="utf-8")

setuptools.setup(
    name="meshlib",
    version='$',
    author="MeshLib Team",
    author_email="support@meshinspector.com",
    description="3d processing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeshInspector/MeshLib",
    license_files=('LICENSE',),
    packages=['meshlib'],
    package_data={'meshlib': ['mrmeshnumpy.$', 'mrmeshpy.$', 'mrviewerpy.$']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: $",
        "License :: Free for non-commercial use",
        "License :: Free For Educational Use",
        "Operating System :: $",
    ],
    python_requires='==$.*',
    install_requires=[
        'numpy>=1.21.0',
        'pytest>=7.1.0',
    ],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/MeshInspector/MeshLib/issues",
        "Source": "https://github.com/MeshInspector/MeshLib",
    },
)