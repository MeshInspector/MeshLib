import { ml, makeTorus, check, checkEq } from "./common.mjs";

export function triangulation()
{
    const mesh = makeTorus(2.0, 1.0, 32, 32);
    const pc = ml.meshToPointCloud(mesh, ml.VertNormalsMode.AreaWeighted, null);
    const restored = ml.triangulatePointCloud(pc, null, null);
    check(restored.has_value(), "triangulatePointCloud failed");
    checkEq(restored.value().points.size(), 1024, "points");
    const top = restored.value().topology;
    checkEq(top.getValidVerts().count(), 1024, "valid verts");
    checkEq(top.findHoleRepresentiveEdges(null).size(), 0, "holes");
}

export function meshCollidePrecise()
{
    const meshA = makeTorus(1.1, 0.5, 8, 8);
    const meshB = makeTorus(1.1, 0.5, 8, 8);
    const rot = ml.Matrix3f.rotation(ml.Vector3f.plusZ(), new ml.Vector3f(0.1, 0.8, 0.2));
    meshB.transform(ml.AffineXf3f.linear(rot), null);
    const partA = new ml.MeshPart(meshA, null);
    const partB = new ml.MeshPart(meshB, null);
    const conv = ml.getVectorConverters(partA, partB, null);
    const intersections = ml.findCollidingEdgeTrisPrecise(partA, partB, conv.toInt, null, null);
    checkEq(intersections.size(), 152, "intersections");
    const contours = ml.orderIntersectionContours(meshA.topology, meshB.topology, intersections);
    checkEq(contours.size(), 4, "contours");
    const sizes = [];
    for (let i = 0; i < 4; i++)
        sizes.push(contours.index(i).size());
    check(sizes[0] == 69 || sizes[0] == 71, "contour 0: " + sizes[0]);
    check(sizes[1] == 9 || sizes[1] == 7, "contour 1: " + sizes[1]);
    const contoursA = new ml.OneMeshContours();
    const contoursB = new ml.OneMeshContours();
    ml.getOneMeshIntersectionContours(meshA, meshB, contours, contoursA, contoursB, conv, null, null, null);
    checkEq(contoursA.size(), 4, "contours A");
    checkEq(contoursB.size(), 4, "contours B");
    // the intersections of all contours are in one array
    checkEq(contoursA.intersections.size(), 156, "intersections in contours");
}
