// Runs the tests of the generated JS bindings, ported from MRTestJava. Every exported function of the test modules is a test.
// No test framework, to keep the bindings free of dependencies.
//
// Usage: `node source/MRTestJS/main.mjs [path/to/MeshLibJS/js/index.mjs]`, the default is `build/Release/MeshLibJS/js/index.mjs`.

import { pathToFileURL, fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import { setBindings } from "./common.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const bindingsPath = resolve(process.argv[2] ?? resolve(here, "../../build/Release/MeshLibJS/js/index.mjs"));
const ml = await import(pathToFileURL(bindingsPath).href);
await ml.init();
setBindings(ml);

const modules = ["testMath", "testBitSet", "testMesh", "testMeshBoolean", "testMeshFillHole", "testVoxels", "testPointCloud", "testMeshIo"];

const failures = [];
let passed = 0;
for (const name of modules)
{
    const mod = await import(`./${name}.mjs`);
    for (const [func, value] of Object.entries(mod))
    {
        if (typeof value !== "function")
            continue;
        const full = `${name}.${func}`;
        try
        {
            value();
            passed++;
            console.log(`ok    ${full}`);
        }
        catch (e)
        {
            failures.push(full);
            console.log(`FAIL  ${full}: ${e && e.stack ? e.stack : e}`);
        }
    }
}
console.log(`${passed} passed, ${failures.length} failed.`);
if (failures.length > 0)
{
    console.log("Failed: " + failures.join(", "));
    process.exit(1);
}
