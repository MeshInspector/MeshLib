import { ml, PI_F, createCube, makeTorus, check, checkEq, checkApprox, tri } from "./common.mjs";

export function meshBoolean()
{
    const meshA = makeTorus(1.1, 0.5, 8, 8);
    const meshB = makeTorus(1.0, 0.2, 8, 8);
    {
        const rot = ml.Matrix3f.rotation(ml.Vector3f.plusZ(), ml.Vector3f.plusY());
        meshB.transform(ml.AffineXf3f.linear(rot), null);
    }

    const shiftStep = 0.2;
    const angleStep = PI_F;
    const baseAxis = [ml.Vector3f.plusX(), ml.Vector3f.plusY(), ml.Vector3f.plusZ()];
    for (let maskTrans = 0; maskTrans < 8; maskTrans++)
    {
        for (let maskRot = 0; maskRot < 8; maskRot++)
        {
            for (let shift = 0.01; shift < 0.2; shift += shiftStep)
            {
                let shiftVec = new ml.Vector3f(0.0, 0.0, 0.0);
                for (let i = 0; i < 3; i++)
                {
                    if ((maskTrans & (1 << i)) !== 0)
                        shiftVec = ml.opAdd(shiftVec, baseAxis[i]);
                }
                for (let angle = PI_F * 0.01; angle < PI_F * 7.0 / 18.0; angle += angleStep)
                {
                    let rotation = ml.Matrix3f.identity();
                    for (let i = 0; i < 3; i++)
                    {
                        if ((maskRot & (1 << i)) !== 0)
                            rotation = ml.opMul(ml.Matrix3f.rotation(baseAxis[i], angle), rotation);
                    }
                    const xf = ml.opMul(ml.AffineXf3f.translation(shiftVec), ml.AffineXf3f.linear(rotation));
                    const params = new ml.BooleanParameters();
                    params.rigidB2A = xf;
                    const resultAB = ml.boolean(meshA, meshB, ml.BooleanOperation.Union, params);
                    const resultBA = ml.boolean(meshB, meshA, ml.BooleanOperation.Union, params);
                    check(resultAB.valid(), "AB");
                    check(resultBA.valid(), "BA");
                    resultAB.delete();
                    resultBA.delete();
                }
            }
        }
    }
}

export function booleanMultipleEdgePropogationSort()
{
    const points = new ml.VertCoords(6);
    points.index_mut(new ml.VertId(0)).set(new ml.Vector3f(0.0, 0.0, 0.0));
    points.index_mut(new ml.VertId(1)).set(new ml.Vector3f(-0.5, 1.0, 0.0));
    points.index_mut(new ml.VertId(2)).set(new ml.Vector3f(0.5, 1.0, 0.0));
    points.index_mut(new ml.VertId(3)).set(new ml.Vector3f(0.0, 1.5, 0.5));
    points.index_mut(new ml.VertId(4)).set(new ml.Vector3f(-1.0, 1.5, 0.0));
    points.index_mut(new ml.VertId(5)).set(new ml.Vector3f(1.0, 1.5, 0.0));
    const triangulation = new ml.Triangulation(5);
    triangulation.index_mut(new ml.FaceId(0)).set(tri(0, 2, 1));
    triangulation.index_mut(new ml.FaceId(1)).set(tri(1, 2, 3));
    triangulation.index_mut(new ml.FaceId(2)).set(tri(3, 4, 1));
    triangulation.index_mut(new ml.FaceId(3)).set(tri(2, 5, 3));
    triangulation.index_mut(new ml.FaceId(4)).set(tri(3, 5, 4));
    const meshA = ml.Mesh.fromTriangles(points, triangulation, null, null);
    {
        const meshASup = new ml.Mesh(meshA);
        meshASup.points.index_mut(new ml.VertId(3)).set(new ml.Vector3f(0.0, 1.5, -0.5));
        const holes = meshA.topology.findHoleRepresentiveEdges(null);
        const border = ml.trackRightBoundaryLoop(meshA.topology, holes.at(0), null, ml.Turn.Rightmost);
        const borderVec = new ml.std_vector_std_vector_MR_EdgeId();
        borderVec.push_back(border);
        const mp = new ml.MeshPart(meshASup, meshASup.topology.getValidFaces());
        meshA.addMeshPart(mp, true, borderVec, borderVec, null, null);
    }

    const meshB = ml.makeCube(ml.Vector3f.diagonal(2.0), ml.Vector3f.diagonal(-0.5));
    meshB.transform(ml.AffineXf3f.translation(new ml.Vector3f(-1.5, -0.2, -0.5)), null);

    for (let i = 0; i < ml.BooleanOperation.Count; i++)
    {
        const resultAB = ml.boolean(meshA, meshB, i, new ml.BooleanParameters());
        const resultBA = ml.boolean(meshB, meshA, i, new ml.BooleanParameters());
        check(resultAB.valid(), "AB " + i);
        check(resultBA.valid(), "BA " + i);
    }
}

export function booleanMapper()
{
    const meshA = makeTorus(1.1, 0.5, 8, 8);
    const meshB = makeTorus(1.0, 0.2, 8, 8);
    {
        const rot = ml.Matrix3f.rotation(ml.Vector3f.plusZ(), ml.Vector3f.plusY());
        meshB.transform(ml.AffineXf3f.linear(rot), null);
    }

    const mapper = new ml.BooleanResultMapper();
    let result;
    {
        const parameters = new ml.BooleanParameters();
        parameters.mapper = mapper;
        result = ml.boolean(meshA, meshB, ml.BooleanOperation.Union, parameters);
    }
    check(result.valid(), "valid");

    const vMapA = mapper.map(meshA.topology.getValidVerts(), ml.BooleanResultMapper_MapObject.A);
    const vMapB = mapper.map(meshB.topology.getValidVerts(), ml.BooleanResultMapper_MapObject.B);
    checkEq(ml.opSub(vMapA, vMapB).count(), vMapA.count(), "disjoint verts");
    checkEq(vMapA.count(), 60, "verts A");
    checkEq(vMapB.count(), 48, "verts B");

    const fMapA = mapper.map(meshA.topology.getValidFaces(), ml.BooleanResultMapper_MapObject.A);
    const fMapB = mapper.map(meshB.topology.getValidFaces(), ml.BooleanResultMapper_MapObject.B);
    checkEq(ml.opSub(fMapA, fMapB).count(), fMapA.count(), "disjoint faces");
    checkEq(fMapA.count(), 224, "faces A");
    checkEq(fMapB.count(), 192, "faces B");

    const newFaces = mapper.newFaces();
    checkEq(newFaces.size(), 416, "new faces size");
    checkEq(newFaces.count(), 252, "new faces count");

    const mapsA = mapper.getMaps(ml.BooleanResultMapper_MapObject.A);
    checkEq(mapsA.old2newVerts.size(), 160, "old2newVerts A");
    checkEq(mapsA.cut2newFaces.size(), 280, "cut2newFaces A");
    checkEq(mapsA.cut2origin.size(), 280, "cut2origin A");
    const mapsB = mapper.getMaps(ml.BooleanResultMapper_MapObject.B);
    checkEq(mapsB.old2newVerts.size(), 160, "old2newVerts B");
    checkEq(mapsB.cut2newFaces.size(), 320, "cut2newFaces B");
    checkEq(mapsB.cut2origin.size(), 320, "cut2origin B");
}

export function meshCollide()
{
    const meshA = makeTorus(1.1, 0.5, 8, 8);
    const meshB = makeTorus(1.1, 0.2, 8, 8);
    const partA = new ml.MeshPart(meshA, null);
    const partB = new ml.MeshPart(meshB, null);
    check(ml.isInside(partB, partA, null), "B inside A");
    check(!ml.isInside(partA, partB, null), "A not inside B");
}

export function meshMeshDistance()
{
    const sphere1 = ml.makeUVSphere(1.0, 8, 8);
    const wholeSphere1 = new ml.MeshPart(sphere1, null);
    const d11 = ml.findDistance(wholeSphere1, wholeSphere1, null, 3.4e38);
    check(d11.distSq === 0.0, "self distance");
    const zShift = ml.AffineXf3f.translation(new ml.Vector3f(0.0, 0.0, 3.0));
    const d1z = ml.findDistance(wholeSphere1, wholeSphere1, zShift, 3.4e38);
    check(d1z.distSq === 1.0, "shifted distance");
    const sphere2 = ml.makeUVSphere(2.0, 8, 8);
    const wholeSphere2 = new ml.MeshPart(sphere2, null);
    const d12 = ml.findDistance(wholeSphere1, wholeSphere2, null, 3.4e38);
    const dist12 = Math.sqrt(d12.distSq);
    check(dist12 > 0.9 && dist12 < 1.0, "distance 12");
}

function createComponentsMesh()
{
    const meshA = ml.makeCube(ml.Vector3f.diagonal(1.0), ml.Vector3f.diagonal(-0.5));
    const meshB = ml.makeCube(ml.Vector3f.diagonal(0.1), ml.Vector3f.diagonal(1.0));
    const result = ml.boolean(meshA, meshB, ml.BooleanOperation.Union, new ml.BooleanParameters());
    const mesh = new ml.Mesh(result.mesh);
    // MeshPart(Mesh, FaceBitSet) does not keep the mesh alive, so the part owns it here.
    return new ml.MeshPart(ml.ffi.RAW, new ml.MeshPart(mesh, null).release(), true, mesh);
}

export function components()
{
    const m = createComponentsMesh();
    const map = ml.MeshComponents.getAllComponentsMap(m, null, null);
    checkEq(map.second, 2, "components");
    checkEq(map.first.size(), 24, "map size");
    checkEq(map.first.index(new ml.FaceId(0)).id_, 0, "face 0");
    checkEq(map.first.index(new ml.FaceId(12)).id_, 1, "face 12");

    const regions = ml.MeshComponents.getLargeByAreaRegions(m, map.first, map.second, 0.1);
    checkEq(regions.second, 1, "regions");
    check(regions.first.test(new ml.FaceId(0)), "face 0");
    check(!regions.first.test(new ml.FaceId(12)), "face 12");

    const components = ml.MeshComponents.getLargeByAreaComponents(m, 0.1, null);
    check(components.test(new ml.FaceId(0)), "face 0");
    check(!components.test(new ml.FaceId(12)), "face 12");

    // The `int *` output parameter is raw.
    const smallerComponents = ml.ffi.malloc(4);
    const largest = ml.MeshComponents.getLargestComponent(m, null, null, 0.1, smallerComponents);
    check(largest.test(new ml.FaceId(0)), "face 0");
    check(!largest.test(new ml.FaceId(12)), "face 12");
    checkEq(ml.ffi.M.HEAP32[smallerComponents >>> 2], 1, "smaller components");
    ml.ffi.free(smallerComponents);

    const component = ml.MeshComponents.getComponent(m, new ml.FaceId(12), null, null);
    check(component.test(new ml.FaceId(12)), "face 12");
    check(!component.test(new ml.FaceId(0)), "face 0");
}

export function expandShrink()
{
    const makeSphere = () =>
    {
        const params = new ml.SphereParams();
        params.numMeshVertices = 3000;
        params.radius = 1.0;
        return ml.makeSphere(params);
    };
    {
        const mesh = makeSphere();
        const top = mesh.topology;
        const region = ml.expand(top, new ml.FaceId(0), 3);
        checkEq(region.count(), 75, "expand from face");
        ml.expand(top, region, 3);
        const num = region.count();
        check(num == 274 || num == 284, "expand region: " + num); // With or without the FMA instruction.
        ml.shrink(top, region, 3);
        checkEq(region.count(), 75, "shrink");
    }
    {
        const mesh = makeSphere();
        const top = mesh.topology;
        const region = ml.expand(top, new ml.VertId(0), 3);
        checkEq(region.count(), 37, "expand from vert");
        ml.expand(top, region, 3);
        check(region.count() > 37, "expand region");
        ml.shrink(top, region, 3);
        checkEq(region.count(), 37, "shrink");
    }
}

export function fixSelfIntersections()
{
    const mesh = ml.makeTorusWithSelfIntersections(1.0, 0.2, 32, 16, null);
    checkEq(mesh.topology.getValidFaces().count(), 1024, "faces");
    let intersections = ml.SelfIntersections.getFaces(mesh, false, null);
    checkEq(intersections.count(), 128, "intersections");
    const settings = new ml.SelfIntersections.Settings();
    settings.method = ml.SelfIntersections.Settings_Method.CutAndFill;
    settings.touchIsIntersection = false;
    ml.SelfIntersections.fix(mesh, settings);
    const validFaces = mesh.topology.getValidFaces().count();
    check(validFaces == 1194 || validFaces == 1196, "valid faces after fix: " + validFaces);
    intersections = ml.SelfIntersections.getFaces(mesh, false, null);
    checkEq(intersections.count(), 0, "no intersections");
}

export function signedDistances()
{
    const meshA = createCube();
    const meshB = createCube();
    const shift = new ml.Vector3f(1.0, 2.0, 3.0);
    const params = new ml.MeshProjectionParameters();
    params.xf = ml.AffineXf3f.translation(shift);
    const results = ml.findSignedDistances(meshA, meshB, params, null);
    checkEq(results.size(), meshB.points.size(), "size");
    let maxDist = 0.0;
    for (let i = 0; i < results.size(); i++)
        maxDist = Math.max(maxDist, Math.abs(results.index(new ml.VertId(i))));
    checkApprox(maxDist, shift.length(), 1e-6);
}
