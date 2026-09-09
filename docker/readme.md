### MeshLib docker images

The Dockerfiles in this directory (and the corresponding `meshlib/meshlib-*` images on [Docker Hub](https://hub.docker.com/u/meshlib)) are designed for MeshLib CI/CD only: they provide the build environments for the GitHub Actions workflows and are not intended for local development.

In particular, the images contain prebuilt thirdparty libraries in `/usr/local/lib/` (`meshlib-thirdparty-lib/` in the Linux images; `emscripten/`, `emscripten-single/` and `emscripten-wasm64/` in the emscripten ones) — outside the source tree, where the build does not find them by itself. If you still want to build MeshLib from source in a container, first link them into the repository root, as the CI workflows do:
```
$ ln -s /usr/local/lib/meshlib-thirdparty-lib/lib ./lib
$ ln -s /usr/local/lib/meshlib-thirdparty-lib/include ./include
$ ln -s /usr/local/lib/meshlib-thirdparty-lib/share ./share
```

Build an image locally:
```
$ docker build -f ./docker/ubuntu24Dockerfile -t meshlib/meshlib-ubuntu24 .
```
