The contents of this directory are the results of generating C bindings with `ENABLE_CUDA=0`, specifically the contents of `source/MeshLibC2Cuda`.

They are used purely for convenience in CI, when generating bindings on one platform and then compiling them on another, when the target platform doesn't have Cuda.
