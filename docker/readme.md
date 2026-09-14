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

#### The emscripten images carry a patched emsdk

`emscriptenDockerfile` applies `docker/patches/emsdk-4.0.19-proxying-26582.patch` to the emsdk it starts from. It is [emscripten-core/emscripten#26582](https://github.com/emscripten-core/emscripten/pull/26582): `emscripten_proxy_finish` signalled the proxy context's condition variable after releasing its mutex, so the waiting thread could return and destroy the context off its own stack before the signal landed, and the thread that should have woken waited forever. Every syscall is proxied to the main thread under `-pthread`, so any worker thread touching the filesystem could hang — about one 8-minute run in three on two cores.

The patch only takes effect because the cached `libc-mt*.a` archives are deleted along with it: `proxying.c` is compiled into libc, and the emsdk image ships that prebuilt. The cache warmup in the builder stage rebuilds them from the patched sources, and the final stage copies that cache into the image, so consumers get the fix without rebuilding anything.

Upstream shipped the fix in emsdk 5.0.5. When `EMSDK_VERSION` reaches it, `patch` will fail on the unmatched context and the image build will stop — delete the patch file and its `COPY`/`RUN` block at that point.
