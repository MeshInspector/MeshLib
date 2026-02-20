using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class FillHoleTests
    {
        private static Std.Array_MRVertId_3 makeTri(int v0, int v1, int v2)
        {
            // TODO: array constructor
            Std.Array_MRVertId_3 tri;
            tri.elems._0 = new VertId(v0);
            tri.elems._1 = new VertId(v1);
            tri.elems._2 = new VertId(v2);
            return tri;
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

            var triangles = new Triangulation();
            triangles.pushBack(makeTri(0, 2, 1));
            triangles.pushBack(makeTri(3, 4, 5));
            triangles.pushBack(makeTri(0, 1, 3));
            triangles.pushBack(makeTri(2, 5, 4));
            triangles.pushBack(makeTri(2, 3, 5));

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

            var patch = fillHoleNicely(mesh, holes[0], param);

            Assert.That( patch.count, Is.EqualTo(1887) );
            Assert.That(mesh.topology.findHoleRepresentiveEdges().size(), Is.EqualTo(1));
        }

        [Test]
        public void TestRightBoundary()
        {
            var mesh = CreateMeshWithHoles();
            var loops = findRightBoundary(mesh.topology);
            Assert.That(loops.size(), Is.EqualTo(2));
            Assert.That(loops[0].size(), Is.EqualTo(3));
            Assert.That(loops[1].size(), Is.EqualTo(4));
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
