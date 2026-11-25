import { ml, createCube, check, checkEq, checkApprox } from "./common.mjs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { existsSync, mkdtempSync, rmSync } from "node:fs";

// The Node build of the module has the real file system (`-sNODERAWFS`), so the files land in the temporary directory.
export function saveLoad()
{
    const dir = mkdtempSync(join(tmpdir(), "meshlib-js-"));
    try
    {
        const cube = createCube();
        for (const ext of ["ply", "stl", "obj", "off", "ctm", "mrmesh"])
        {
            const path = join(dir, "cube." + ext);
            ml.MeshSave.toAnySupportedFormat(cube, path);
            check(existsSync(path), "saved " + ext);
            const loaded = ml.MeshLoad.fromAnySupportedFormat(path);
            check(loaded instanceof ml.Mesh, "loaded " + ext);
            checkEq(loaded.topology.getValidFaces().count(), 12, "faces " + ext);
            checkApprox(loaded.volume(), 1.0, 0.001);
        }
    }
    finally
    {
        rmSync(dir, {recursive: true, force: true});
    }
}

export function loadError()
{
    let caught = false;
    try
    {
        ml.MeshLoad.fromAnySupportedFormat(join(tmpdir(), "meshlib-js-missing-file.ply"));
    }
    catch (e)
    {
        caught = e instanceof ml.UnexpectedResultException && e.message.length > 0;
    }
    check(caught, "missing file");
}
