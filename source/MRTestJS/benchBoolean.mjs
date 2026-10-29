// Times mesh Booleans through the generated JS bindings; a scratch benchmark for comparing module builds.
// Usage: `node source/MRTestJS/benchBoolean.mjs path/to/MeshLibJS/js/index.mjs`.

import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { setBindings } from "./common.mjs";
import { meshBoolean } from "./testMeshBoolean.mjs";

const ml = await import(pathToFileURL(resolve(process.argv[2])).href);
await ml.init();
setBindings(ml);

function bench(name, fn, iters)
{
    const t = [];
    for (let i = 0; i < iters; i++)
    {
        const s = process.hrtime.bigint();
        fn();
        t.push(Number(process.hrtime.bigint() - s) / 1e6);
    }
    t.sort((a, b) => a - b);
    console.log(`${name}: min ${t[0].toFixed(1)} ms, median ${t[t.length >> 1].toFixed(1)} ms, max ${t[t.length - 1].toFixed(1)} ms, n=${iters}`);
}

for (const res of [64, 256])
{
    const a = ml.makeUVSphere(1.0, res, res);
    const b = ml.makeUVSphere(1.0, res, res);
    const params = new ml.BooleanParameters();
    params.rigidB2A = ml.AffineXf3f.translation(new ml.Vector3f(0.5, 0.3, 0.2));
    const tris = a.topology.getValidFaces().count();
    for (const [opName, op] of [["union", ml.BooleanOperation.Union], ["intersection", ml.BooleanOperation.Intersection]])
    {
        bench(`sphere ${res}x${res} (${tris} triangles) ${opName}`, () =>
        {
            const r = ml.boolean(a, b, op, params);
            if (!r.valid())
                throw new Error(r.errorString.toString());
            r.delete();
        }, res >= 256 ? 7 : 15);
    }
}

bench("testMeshBoolean.meshBoolean (128 unions of 8x8 tori)", meshBoolean, 7);
