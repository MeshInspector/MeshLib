import { ml, check, checkEq, tri } from "./common.mjs";

function createMeshWithHoles()
{
    const points = new ml.VertCoords(6);
    points.index_mut(new ml.VertId(0)).set(new ml.Vector3f(0.0, 0.0, 0.0));
    points.index_mut(new ml.VertId(1)).set(new ml.Vector3f(1.0, 0.0, 0.0));
    points.index_mut(new ml.VertId(2)).set(new ml.Vector3f(0.0, 1.0, 0.0));
    points.index_mut(new ml.VertId(3)).set(new ml.Vector3f(0.0, 0.0, 1.0));
    points.index_mut(new ml.VertId(4)).set(new ml.Vector3f(1.0, 0.0, 1.0));
    points.index_mut(new ml.VertId(5)).set(new ml.Vector3f(0.0, 1.0, 1.0));
    const triangulation = new ml.Triangulation(5);
    triangulation.index_mut(new ml.FaceId(0)).set(tri(0, 2, 1));
    triangulation.index_mut(new ml.FaceId(1)).set(tri(3, 4, 5));
    triangulation.index_mut(new ml.FaceId(2)).set(tri(0, 1, 3));
    triangulation.index_mut(new ml.FaceId(3)).set(tri(2, 5, 4));
    triangulation.index_mut(new ml.FaceId(4)).set(tri(2, 3, 5));
    return ml.Mesh.fromTriangles(points, triangulation, null, null);
}

export function meshFillHole()
{
    const mesh = createMeshWithHoles();
    const oldBdEdges = mesh.topology.findHoleRepresentiveEdges(null);
    checkEq(oldBdEdges.size(), 2, "holes");
    const newFaces = new ml.FaceBitSet();
    {
        const params = new ml.FillHoleParams();
        params.outNewFaces = newFaces;
        ml.fillHoles(mesh, oldBdEdges, params);
    }
    checkEq(newFaces.count(), 3, "new faces");
    checkEq(mesh.topology.findHoleRepresentiveEdges(null).size(), 0, "no holes");
}

export function meshFillHoleNicely()
{
    const mesh = createMeshWithHoles();
    const oldBdEdges = mesh.topology.findHoleRepresentiveEdges(null);
    checkEq(oldBdEdges.size(), 2, "holes");
    const settings = new ml.FillHoleNicelySettings();
    const patch = ml.fillHoleNicely(mesh, oldBdEdges.front(), settings);
    checkEq(patch.count(), 1887, "patch");
    checkEq(mesh.topology.findHoleRepresentiveEdges(null).size(), 1, "one hole left");
}

export function rightBoundary()
{
    const mesh = createMeshWithHoles();
    const loops = ml.findRightBoundary(mesh.topology, null, null);
    checkEq(loops.size(), 2, "loops");
    checkEq(loops.at(0).size(), 3, "loop 0");
    checkEq(loops.at(1).size(), 4, "loop 1");
}

export function findHoleComplicatingFaces()
{
    const mesh = createMeshWithHoles();
    checkEq(ml.findHoleComplicatingFaces(mesh).count(), 0, "complicating faces");
}
