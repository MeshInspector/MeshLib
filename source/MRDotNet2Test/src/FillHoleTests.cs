using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class FillHoleTests
    {
        private static void updateTri(Std.Mut_Array_MRVertId_3 tri, int v0, int v1, int v2)
        {
            tri.elems._0 = new VertId(v0);
            tri.elems._1 = new VertId(v1);
            tri.elems._2 = new VertId(v2);
        }

        private static Mesh CreateMeshWithHoles()
        {
            var points = new VertCoords();
            points.pushBack(new Vector3f(0, 0, 0));
            points.pushBack(new Vector3f(1, 0, 0));
            points.pushBack(new Vector3f(0, 1, 0));
            points.pushBack(new Vector3f(0, 0, 1));
            points.pushBack(new Vector3f(1, 0, 1));
            points.pushBack(new Vector3f(0, 1, 1));

            var triangles = new Triangulation(5);
            updateTri(triangles[new FaceId(0)], 0, 2, 1);
            updateTri(triangles[new FaceId(1)], 3, 4, 5);
            updateTri(triangles[new FaceId(2)], 0, 1, 3);
            updateTri(triangles[new FaceId(3)], 2, 5, 4);
            updateTri(triangles[new FaceId(4)], 2, 3, 5);

            return Mesh.fromTriangles(points, triangles);
        }

        [Test]
        public void TestFillHole()
        {
            var mesh = CreateMeshWithHoles();
            var holes = mesh.topology.findHoleRepresentiveEdges();
            Assert.That(holes.size(), Is.EqualTo(2) );

            var param = new FillHoleParams();

            fillHoles(mesh, holes, param);
            Assert.That(mesh.topology.findHoleRepresentiveEdges().size(), Is.EqualTo(0));
        }

        [Test]
        public void TestFillHoleNicely()
        {
            var mesh = CreateMeshWithHoles();
            var holes = mesh.topology.findHoleRepresentiveEdges();
            Assert.That(holes.size(), Is.EqualTo(2));

            var param = new FillHoleNicelySettings();

            var patch = fillHoleNicely(mesh, holes.at(0), param);

            Assert.That( patch.count, Is.EqualTo(1887) );
            Assert.That(mesh.topology.findHoleRepresentiveEdges().size(), Is.EqualTo(1));
        }

        [Test]
        public void TestRightBoundary()
        {
            var mesh = CreateMeshWithHoles();
            var loops = findRightBoundary(mesh.topology);
            Assert.That(loops.size(), Is.EqualTo(2));
            Assert.That(loops.at(0).size(), Is.EqualTo(3));
            Assert.That(loops.at(1).size(), Is.EqualTo(4));
        }

        [Test]
        public void TestFindHoleComplicatedFaces()
        {
            var mesh = CreateMeshWithHoles();
            var complicatedFaces = findHoleComplicatingFaces(mesh);
            Assert.That(complicatedFaces.count(), Is.EqualTo(0));
        }
    }
}
