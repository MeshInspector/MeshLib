import { ml, createCube, check, checkEq } from "./common.mjs";

function cubePart()
{
    const mesh = createCube();
    return new ml.MeshPart(ml.ffi.RAW, new ml.MeshPart(mesh, null).release(), true, mesh);
}

export function offsetMesh()
{
    const mp = cubePart();
    const params = new ml.OffsetParameters();
    params.voxelSize = ml.suggestVoxelSize(mp, 10000.0);
    check(ml.offsetMesh(mp, 0.1, params) instanceof ml.Mesh, "offsetMesh");
    check(ml.doubleOffsetMesh(mp, 0.1, -0.2, params) instanceof ml.Mesh, "doubleOffsetMesh");
    check(ml.mcOffsetMesh(mp, 0.1, params, null) instanceof ml.Mesh, "mcOffsetMesh");
    const sharp = new ml.SharpOffsetParameters();
    sharp.voxelSize = ml.suggestVoxelSize(mp, 10000.0);
    check(ml.sharpOffsetMesh(mp, 0.1, sharp) instanceof ml.Mesh, "sharpOffsetMesh");
    const general = new ml.GeneralOffsetParameters();
    general.voxelSize = ml.suggestVoxelSize(mp, 10000.0);
    general.mode = ml.OffsetMode.Standard;
    check(ml.generalOffsetMesh(mp, 0.1, general) instanceof ml.Mesh, "generalOffsetMesh");
    const map = new ml.PartMapping();
    check(ml.thickenMesh(createCube(), 0.1, general, map) instanceof ml.Mesh, "thickenMesh");
}

export function progressCallback()
{
    const mp = cubePart();
    const params = new ml.OffsetParameters();
    params.voxelSize = ml.suggestVoxelSize(mp, 10000.0);
    let calls = 0;
    let last = -1.0;
    params.callBack = ml.std_function_bool_from_float.of(progress =>
    {
        calls++;
        check(progress >= last && progress <= 1.0, "progress goes up: " + progress);
        last = progress;
        return true;
    });
    check(ml.offsetMesh(mp, 0.1, params) instanceof ml.Mesh, "offsetMesh");
    check(calls > 0, "the callback was called");
    // A plain JS function is accepted too.
    calls = 0;
    params.callBack = (progress) => {calls++; return true;};
    check(ml.offsetMesh(mp, 0.1, params) instanceof ml.Mesh, "offsetMesh with a plain function");
    check(calls > 0, "the plain function was called");
}

export function cancellingCallback()
{
    const mp = cubePart();
    const params = new ml.OffsetParameters();
    params.voxelSize = ml.suggestVoxelSize(mp, 10000.0);
    params.callBack = ml.std_function_bool_from_float.of(progress => false);
    let cancelled = false;
    try
    {
        ml.offsetMesh(mp, 0.1, params);
    }
    catch (e)
    {
        cancelled = e instanceof ml.UnexpectedResultException;
    }
    check(cancelled, "cancelled");
}

export function callbackException()
{
    const f = ml.std_function_bool_from_float.of(progress => {throw new RangeError("from the function");});
    let caught = false;
    try
    {
        f.call(0.5);
    }
    catch (e)
    {
        caught = e instanceof RangeError && e.message === "from the function";
    }
    check(caught, "the exception of the function is rethrown");
}

export function casts()
{
    const objectMesh = new ml.ObjectMesh();
    const base = objectMesh.upcastToObject();
    check(!base.isOwned(), "the upcast is a reference");
    const back = ml.ObjectMesh.dynamicDowncastFromObject(base);
    check(back !== null && back.rawPointer() === objectMesh.rawPointer(), "dynamic downcast");
    check(ml.ObjectMesh.staticDowncastFromObject(base).rawPointer() === objectMesh.rawPointer(), "static downcast");
    const notAMesh = new ml.ObjectPoints();
    check(ml.ObjectMesh.dynamicDowncastFromObject(notAMesh.upcastToObject()) === null, "dynamic downcast of another class is null");
}

function createVolume()
{
    const sphereParams = new ml.SphereParams();
    sphereParams.radius = 1.0;
    sphereParams.numMeshVertices = 3000;
    const mesh = ml.makeSphere(sphereParams);
    const mp = new ml.MeshPart(ml.ffi.RAW, new ml.MeshPart(mesh, null).release(), true, mesh);
    const params = new ml.MeshToVolumeParams();
    params.voxelSize = ml.Vector3f.diagonal(0.1);
    return ml.meshToVolume(mp, params);
}

const dims = (d, x, y, z) => d.x === x && d.y === y && d.z === z;

export function vdbConversions()
{
    const volume = createVolume();
    check(volume.min > -0.001 && volume.min < 0.001, "min");
    check(volume.max > 2.999 && volume.max < 3.001, "max");
    check(dims(volume.dims, 26, 26, 26), "dims");
    const settings = new ml.GridToMeshSettings();
    settings.voxelSize = ml.Vector3f.diagonal(0.1);
    settings.isoValue = 1.0;
    const restored = ml.gridToMesh(volume.data, settings);
    const bbox = restored.computeBoundingBox(null);
    check(bbox.min.x > 0.199 && bbox.min.x < 0.201, "min x");
    check(bbox.max.x > 2.394 && bbox.max.x < 2.396, "max x");
    checkEq(restored.points.size(), 3748, "points");
}

export function resampling()
{
    const volume = createVolume();
    check(dims(ml.floatGridToVdbVolume(ml.resampled(volume.data, 2.0, null)).dims, 13, 13, 13), "uniform dims");
    check(dims(ml.floatGridToVdbVolume(ml.resampled(volume.data, new ml.Vector3f(2.0, 1.0, 0.5), null)).dims, 13, 27, 53), "dims");
    const box = new ml.Box3i(new ml.Vector3i(2, 5, 1), new ml.Vector3i(18, 13, 23));
    check(dims(ml.floatGridToVdbVolume(ml.cropped(volume.data, box, null)).dims, 16, 8, 22), "cropped dims");
}

export function accessors()
{
    const volume = createVolume();
    const p = new ml.Vector3i(0, 0, 0);
    check(ml.getValue(volume.data, p) === 3.0, "initial value");
    const d = volume.dims;
    const region = new ml.VoxelBitSet();
    region.resize(d.x * d.y * d.z, null);
    region.set(new ml.VoxelId(0), true);
    ml.setValue(volume.data, region, 1.0);
    check(ml.getValue(volume.data, p) === 1.0, "value after region set");
    ml.setValue(volume.data, p, 2.0);
    check(ml.getValue(volume.data, p) === 2.0, "value after point set");
}
