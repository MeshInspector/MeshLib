### MeshLib docker images

The Dockerfiles in this directory (and the corresponding `meshlib/meshlib-*` images on [Docker Hub](https://hub.docker.com/u/meshlib)) are designed for MeshLib CI/CD only: they provide the build environments for the GitHub Actions workflows and are not intended for local development.

In particular, the images contain prebuilt thirdparty libraries outside the source tree, where the build does not find them by itself. The Linux images install them under `/usr/local/lib/MeshLib`. To build MeshLib from source in a container, point the build scripts at that prefix, as the CI workflows do:
```
$ MESHLIB_THIRDPARTY_ROOT_DIR=/usr/local/lib/MeshLib ./scripts/build_source.sh
```
The emscripten images keep them under `/usr/local/lib/emscripten/`, `emscripten-single/` and `emscripten-wasm64/`; link the matching one into the repository root first:
```
$ ln -s /usr/local/lib/emscripten/lib ./lib
$ ln -s /usr/local/lib/emscripten/include ./include
$ ln -s /usr/local/lib/emscripten/share ./share
```

Build an image locally:
```
$ docker build -f ./docker/ubuntu24Dockerfile -t meshlib/meshlib-ubuntu24 .
```
