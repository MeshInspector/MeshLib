import { ml, createCube, check, checkEq, checkApprox } from "./common.mjs";

// The module uses Emscripten's in-memory file system (no `NODERAWFS`), so the files live only inside the wasm module.
export function saveLoad()
{
    const cube = createCube();
    for (const ext of ["ply", "stl", "obj", "off", "ctm", "mrmesh"])
    {
        const path = "/tmp/cube." + ext;
        ml.MeshSave.toAnySupportedFormat(cube, path);
        const loaded = ml.MeshLoad.fromAnySupportedFormat(path);
        check(loaded instanceof ml.Mesh, "loaded " + ext);
        checkEq(loaded.topology.getValidFaces().count(), 12, "faces " + ext);
        checkApprox(loaded.volume(), 1.0, 0.001);
        ml.ffi.M.FS.unlink(path);
    }
}

export function loadError()
{
    let caught = false;
    try
    {
        ml.MeshLoad.fromAnySupportedFormat("/tmp/missing.ply");
    }
    catch (e)
    {
        caught = e instanceof ml.UnexpectedResultException && e.message.length > 0;
    }
    check(caught, "missing file");
}
