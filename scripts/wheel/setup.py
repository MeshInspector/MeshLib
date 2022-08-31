# used in pair with create_wheel.sh
import setuptools

setuptools.setup(
    name="meshlib",
    version='',
    author="MeshLib Team",
    author_email="support@meshinspector.com",
    description="Package to create MeshLib",
    packages=['meshlib'],
    package_data={'meshlib': ['mrmeshnumpy.so', 'mrmeshpy.so', 'mrviewerpy.so']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='',
    install_requires=[
        'numpy>=1.21.0',
        'pytest>=7.1.0',

    ],
)