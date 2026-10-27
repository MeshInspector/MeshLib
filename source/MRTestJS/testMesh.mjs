import { ml, PI_F, createCube, makeTorus, check, checkEq, checkApprox } from "./common.mjs";

export function area()
{
    const mesh = createCube();
    let area = mesh.area(null);
    check(area > 5.999 && area < 6.001, "whole area");
    const faces = new ml.FaceBitSet();
    faces.resize(12, true);
    for (let i = 6; i < 12; i++)
        faces.set(new ml.FaceId(i), false);
    area = mesh.area(faces);
    check(area > 2.999 && area < 3.001, "region area");
    mesh.deleteFaces(faces, null);
    area = mesh.area();
    check(area > 2.999 && area < 3.001, "area after delete");
}

export function volume()
{
    const mesh = createCube();
    checkApprox(mesh.volume(), 1.0, 0.001);
}

export function shortEdges()
{
    const mesh = makeTorus(1.0, 0.05, 16, 16);
    const mp = new ml.MeshPart(mesh, null);
    const shortEdges = ml.findShortEdges(mp, 0.1, null);
    checkEq(shortEdges.count(), 256, "short edges");
}

export function incidentFaces()
{
    const mesh = createCube();
    const verts = new ml.VertBitSet();
    verts.resize(8, false);
    verts.set(new ml.VertId(0), true);
    checkEq(ml.getIncidentFaces(mesh.topology, verts).count(), 6, "incident faces from verts");
    const edges = new ml.UndirectedEdgeBitSet();
    edges.resize(12, false);
    edges.set(new ml.UndirectedEdgeId(0), true);
    checkEq(ml.getIncidentFaces(mesh.topology, edges).count(), 8, "incident faces from edges");
}

export function topology()
{
    const mesh = createCube();
    mesh.topology.pack(null, null, null, null);
    checkEq(mesh.topology.getValidVerts().count(), 8, "valid verts");
    checkEq(mesh.topology.getValidFaces().count(), 12, "valid faces");
    checkEq(mesh.topology.findHoleRepresentiveEdges(null).size(), 0, "holes");
    const v0 = new ml.VertId(-1), v1 = new ml.VertId(-1), v2 = new ml.VertId(-1);
    mesh.topology.getLeftTriVerts(new ml.EdgeId(0), v0, v1, v2);
    check(v0.id_ >= 0 && v1.id_ >= 0 && v2.id_ >= 0, "verts");
    checkEq(mesh.topology.findNumHoles(null), 0, "holes");
    checkEq(mesh.topology.faceSize(), 12, "face size");
    checkEq(mesh.topology.getTriangulation().size(), 12, "triangulation");
    // The reference keeps the mesh alive.
    const topology = createCube().topology;
    checkEq(topology.faceSize(), 12, "face size through a reference");
    check(!topology.isOwned(), "reference");
}

export function points()
{
    const mesh = createCube();
    const points = mesh.points;
    checkEq(points.size(), 8, "points");
    const p = points.index(new ml.VertId(0));
    check(Math.abs(p.x) === 0.5 && Math.abs(p.y) === 0.5 && Math.abs(p.z) === 0.5, "point");
    // Writing through the reference changes the mesh.
    points.index_mut(new ml.VertId(0)).x = 2.0;
    check(mesh.points.index(new ml.VertId(0)).x === 2.0, "write through the reference");
    mesh.invalidateCaches(true);
}

export function typedArrays()
{
    // The contiguous containers expose their scalars as typed arrays.
    const mesh = createCube();
    const points = mesh.points.toArray();
    check(points instanceof Float32Array && points.length === 24, "points as Float32Array");
    const view = mesh.points.view();
    check(view.length === 24 && view[0] === points[0], "view");
    const tris = mesh.topology.getTriangulation().toArray();
    check(tris instanceof Int32Array && tris.length === 36, "triangulation as Int32Array");
    // And are built from them.
    const coords = ml.VertCoords.fromArray(points);
    checkEq(coords.size(), 8, "fromArray");
    const triangulation = ml.Triangulation.fromArray(tris);
    checkEq(triangulation.size(), 12, "fromArray");
    const copy = ml.Mesh.fromTriangles(coords, triangulation);
    checkApprox(copy.volume(), mesh.volume(), 1e-6);
    coords.setFromArray(new Float32Array(points.map(x => x * 2)));
    const scaled = ml.Mesh.fromTriangles(coords, triangulation);
    checkApprox(scaled.volume(), 8.0, 0.001);
}

export function normals()
{
    const cube = createCube();
    checkEq(ml.computePerVertNormals(cube).size(), 8, "vert normals");
    checkEq(ml.computePerFaceNormals(cube).size(), 12, "face normals");
}

export function simpleObjects()
{
    const area = (mesh) => mesh.area(null);
    checkApprox(area(ml.makeCube(new ml.Vector3f(1.0, 1.0, 1.0), new ml.Vector3f(0.0, 0.0, 0.0))), 6.0, 0.001);
    checkApprox(area(ml.makeCylinderAdvanced(1.0, 0.5, 0.0, 2.0 * 3.1415, 1.0, 16)), 10.541437, 0.1);
    checkApprox(area(ml.makeTorus(2.0, 0.5, 32, 32, null)), 39.257, 0.1);
    checkApprox(area(ml.makeTorusWithSelfIntersections(2.0, 0.5, 16, 16, null)), 57.583, 0.01);
    const params = new ml.SphereParams();
    params.radius = 1.0;
    params.numMeshVertices = 100;
    checkApprox(area(ml.makeSphere(params)), 12.157, 0.1);
    checkApprox(area(ml.makeUVSphere(1.5, 32, 32)), 28.15, 0.1);
}

export function addNoise()
{
    const mesh = makeTorus(1.1, 0.5, 16, 16);
    ml.addNoise(mesh, null, null);
}

export function subdivide()
{
    const mesh = makeTorus(1.1, 0.5, 16, 16);
    const settings = new ml.SubdivideSettings();
    settings.maxDeviationAfterFlip = 0.5;
    ml.subdivideMesh(mesh, settings);
}

export function decimate()
{
    const mesh = ml.makeCylinderAdvanced(0.5, 0.5, 0.0, 20.0 / 180.0 * PI_F, 1.0, 16);
    // Select all faces.
    const region = new ml.FaceBitSet(mesh.topology.getValidFaces());
    const regionSaved = new ml.FaceBitSet(region);
    // Setup and run the decimator.
    const settings = new ml.DecimateSettings();
    settings.maxError = 0.001;
    settings.region = region;
    settings.maxTriangleAspectRatio = 80.0;
    const result = ml.decimateMesh(mesh, settings);
    // Compare the regions and the deleted vertices and faces.
    check(!ml.opEqual(regionSaved, region), "region changed");
    check(result.vertsDeleted > 0, "verts deleted");
    check(result.facesDeleted > 0, "faces deleted");
}

export function extrude()
{
    {
        const mesh = makeTorus(1.1, 0.5, 16, 16);
        const countBefore = mesh.points.size();
        const region = new ml.FaceBitSet();
        region.autoResizeSet(new ml.FaceId(0), true);
        region.autoResizeSet(new ml.FaceId(1), true);
        ml.makeDegenerateBandAroundRegion(mesh, region, null);
        mesh.invalidateCaches(true);
        // The number of vertices grows because of the vertex duplication along the boundary.
        check(mesh.points.size() != countBefore, "points added");
    }
    {
        const mesh = makeTorus(1.1, 0.5, 16, 16);
        const countBefore = mesh.points.size();
        ml.makeDegenerateBandAroundRegion(mesh, new ml.FaceBitSet(), null);
        mesh.invalidateCaches(true);
        checkEq(mesh.points.size(), countBefore, "points unchanged");
    }
}

export function uniteCloseVertices()
{
    const params = new ml.SphereParams();
    params.radius = 1.0;
    params.numMeshVertices = 3000;
    const mesh = ml.makeSphere(params);
    const vertMap = new ml.VertMap();
    const united = ml.MeshBuilder.uniteCloseVertices(mesh, 0.1, false, vertMap);
    checkEq(united, 2230, "united");
    checkEq(vertMap.index(new ml.VertId(1000)).id_, 42, "map");
}
