// Shared helpers of the tests. `ml` is the bindings module, set by `main.mjs` after `init()`.

export let ml = null;

export function setBindings(bindings)
{
    ml = bindings;
}

export const PI_F = 3.14159265358979;

/** Creates a cube mesh with diagonal = 1. */
export function createCube()
{
    return ml.makeCube(ml.Vector3f.diagonal(1.0), ml.Vector3f.diagonal(-0.5));
}

/** Creates a torus with the given radii and resolutions. */
export function makeTorus(primaryRadius, secondaryRadius, primaryResolution, secondaryResolution)
{
    return ml.makeTorus(primaryRadius, secondaryRadius, primaryResolution, secondaryResolution, null);
}

export function check(condition, what)
{
    if (!condition)
        throw new Error("Check failed: " + what);
}

export function checkEq(actual, expected, what)
{
    if (actual != expected) // Loose: `bigint` vs `number`.
        throw new Error(what + ": expected " + expected + ", got " + actual);
}

export function checkApprox(actual, expected, eps)
{
    if (!(Math.abs(actual - expected) <= eps))
        throw new Error("expected " + expected + " (delta " + eps + "), got " + actual + " (delta " + Math.abs(actual - expected) + ")");
}

/** A `std::array<VertId, 3>` of three vertex indices. */
export function tri(a, b, c)
{
    const ret = new ml.std_array_MR_VertId_3();
    ret.setElems(0, new ml.VertId(a));
    ret.setElems(1, new ml.VertId(b));
    ret.setElems(2, new ml.VertId(c));
    return ret;
}
