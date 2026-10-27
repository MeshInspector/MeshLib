import { ml, check, checkEq, checkApprox } from "./common.mjs";

function checkVec(v, x, y, z)
{
    checkApprox(v.x, x, 0.0001);
    checkApprox(v.y, y, 0.0001);
    checkApprox(v.z, z, 0.0001);
}

function matrix(rows)
{
    return new ml.Matrix3f(new ml.Vector3f(...rows[0]), new ml.Vector3f(...rows[1]), new ml.Vector3f(...rows[2]));
}

function checkMatrix(m, rows)
{
    const actual = [[m.x.x, m.x.y, m.x.z], [m.y.x, m.y.y, m.y.z], [m.z.x, m.z.y, m.z.z]];
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            checkApprox(actual[i][j], rows[i][j], 0.001);
}

export function vector3()
{
    checkVec(ml.Vector3f.diagonal(3.0), 3.0, 3.0, 3.0);
    checkVec(ml.Vector3f.plusX(), 1.0, 0.0, 0.0);
    checkVec(ml.Vector3f.plusY(), 0.0, 1.0, 0.0);
    checkVec(ml.Vector3f.plusZ(), 0.0, 0.0, 1.0);
    checkVec(ml.opAdd(new ml.Vector3f(1.0, 2.0, 3.0), new ml.Vector3f(4.0, 5.0, 6.0)), 5.0, 7.0, 9.0);
    checkVec(ml.opSub(new ml.Vector3f(5.0, 7.0, 9.0), new ml.Vector3f(4.0, 5.0, 6.0)), 1.0, 2.0, 3.0);
    checkVec(ml.opMul(new ml.Vector3f(1.0, 2.0, 3.0), 2.0), 2.0, 4.0, 6.0);
    checkApprox(new ml.Vector3f(1.0, 2.0, 3.0).lengthSq(), 14.0, 0.0001);
    checkApprox(new ml.Vector3f(3.0, 4.0, 0.0).length(), 5.0, 0.0001);
    // The fields are accessors over wasm memory.
    const v = new ml.Vector3f(1.0, 2.0, 3.0);
    v.y = 7.5;
    checkVec(v, 1.0, 7.5, 3.0);
    check(ml.Vector3f.SIZE === 12, "SIZE");
    const copy = v.copy();
    v.x = -1;
    checkVec(copy, 1.0, 7.5, 3.0);
    check(v instanceof ml.CppStruct && v instanceof ml.CppObject, "CppStruct");
}

export function vector3i()
{
    const d = ml.Vector3i.diagonal(3);
    check(d.x === 3 && d.y === 3 && d.z === 3, "diagonal");
    const x = ml.Vector3i.plusX();
    check(x.x === 1 && x.y === 0 && x.z === 0, "plusX");
    const y = ml.Vector3i.plusY();
    check(y.x === 0 && y.y === 1 && y.z === 0, "plusY");
    const z = ml.Vector3i.plusZ();
    check(z.x === 0 && z.y === 0 && z.z === 1, "plusZ");
}

export function matrix3()
{
    checkMatrix(ml.Matrix3f.identity(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    {
        const m = ml.Matrix3f.rotation(new ml.Vector3f(1.0, 0.0, 0.0), 3.14159 / 2.0); // 90 degrees around the X axis.
        checkApprox(m.x.x, 1.0, 0.001);
        checkApprox(m.y.y, 0.0, 0.001);
        checkApprox(m.y.z, -1.0, 0.001);
        checkApprox(m.z.y, 1.0, 0.001);
        checkApprox(m.z.z, 0.0, 0.001);
    }
    checkMatrix(ml.Matrix3f.rotation(new ml.Vector3f(1.0, 0.0, 0.0), new ml.Vector3f(0.0, 1.0, 0.0)), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]);
    const a = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const b = matrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);
    checkMatrix(ml.opAdd(a, b), [[10, 10, 10], [10, 10, 10], [10, 10, 10]]);
    checkMatrix(ml.opSub(a, b), [[-8, -6, -4], [-2, 0, 2], [4, 6, 8]]);
    checkMatrix(ml.opMul(ml.Matrix3f.identity(), matrix([[2, 1, 0], [1, 2, 0], [0, 0, 3]])), [[2, 1, 0], [1, 2, 0], [0, 0, 3]]);
    const r = ml.opMul(matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]]), new ml.Vector3f(1.0, 2.0, 3.0));
    checkVec(r, 2.0, 4.0, 6.0);
    check(ml.opEqual(ml.Matrix3f.identity(), ml.Matrix3f.identity()), "equal");
    check(!ml.opEqual(ml.Matrix3f.identity(), matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]])), "not equal");
}

export function affineXf()
{
    {
        const result = new ml.AffineXf3f();
        check(ml.opEqual(result.A, ml.Matrix3f.identity()), "identity");
        checkVec(result.b, 0, 0, 0);
    }
    {
        const result = ml.AffineXf3f.translation(new ml.Vector3f(1.0, 2.0, 3.0));
        check(ml.opEqual(result.A, ml.Matrix3f.identity()), "identity");
        checkVec(result.b, 1, 2, 3);
    }
    {
        const linear = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        const result = ml.AffineXf3f.linear(linear);
        check(ml.opEqual(result.A, linear), "linear");
        checkVec(result.b, 0, 0, 0);
    }
    {
        const xf1 = ml.AffineXf3f.translation(new ml.Vector3f(1.0, 2.0, 3.0));
        const xf2 = ml.AffineXf3f.linear(matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]]));
        const result = ml.opMul(xf1, xf2);
        const expectedA = ml.opMul(xf1.A, xf2.A);
        const a = xf1.A;
        const b2 = xf2.b;
        const ex = a.x.x * b2.x + a.y.x * b2.y + a.z.x * b2.z + xf1.b.x;
        const ey = a.x.y * b2.x + a.y.y * b2.y + a.z.y * b2.z + xf1.b.y;
        const ez = a.x.z * b2.x + a.y.z * b2.y + a.z.z * b2.z + xf1.b.z;
        check(ml.opEqual(result.A, expectedA), "A");
        checkVec(result.b, ex, ey, ez);
    }
    {
        const xf = ml.AffineXf3f.translation(new ml.Vector3f(1.0, 2.0, 3.0));
        const point = new ml.Vector3f(4.0, 5.0, 6.0);
        const result = xf.call(point);
        checkVec(result, 5, 7, 9);
    }
}

export function box()
{
    const b = ml.Box3f.fromMinAndSize(ml.Vector3f.diagonal(-1.0), ml.Vector3f.diagonal(1.0));
    checkApprox(b.volume(), 1.0, 0.001);
    checkApprox(b.diagonal(), Math.sqrt(3), 0.01);
    check(b.valid(), "valid");
    checkVec(b.size(), 1, 1, 1);
    checkVec(b.center(), -0.5, -0.5, -0.5);
    check(!new ml.Box3f().valid(), "invalid");
    const bi = ml.Box3i.fromMinAndSize(ml.Vector3i.diagonal(-2), ml.Vector3i.diagonal(2));
    checkEq(bi.volume(), 8, "volume");
    check(bi.valid(), "valid");
    const size = bi.size();
    check(size.x === 2 && size.y === 2 && size.z === 2, "size");
    const center = bi.center();
    check(center.x === -1 && center.y === -1 && center.z === -1, "center");
    check(!new ml.Box3i().valid(), "invalid");
}

export function color()
{
    const rgba = (color, r, g, b, a) => color.r === r && color.g === g && color.b === b && color.a === a;
    check(rgba(new ml.Color(), 0, 0, 0, 255), "default");
    check(rgba(new ml.Color(128, 64, 32, 200), 128, 64, 32, 200), "int components");
    checkEq(new ml.Color(128, 64, 32, 200).getUInt32(), 0xC8204080, "getUInt32");
    checkEq(new ml.VertColors(10).size(), 10, "size");
}
