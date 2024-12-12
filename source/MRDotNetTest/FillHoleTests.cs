using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class FillHoleTests
    {
        private static Mesh CreateMeshWithHoles()
        {
            List<Vector3f> points = new List<Vector3f>();
            points.Add(new Vector3f(0, 0, 0));
            points.Add(new Vector3f(1, 0, 0));
            points.Add(new Vector3f(0, 1, 0));
            points.Add(new Vector3f(0, 0, 1));
            points.Add(new Vector3f(1, 0, 1));
            points.Add(new Vector3f(0, 1, 1));

            List<ThreeVertIds> triangles = new List<ThreeVertIds>();
            triangles.Add(new ThreeVertIds(0, 2, 1));
            triangles.Add(new ThreeVertIds(3, 4, 5));
            triangles.Add(new ThreeVertIds(0, 1, 3));
            triangles.Add(new ThreeVertIds(2, 5, 4));
            triangles.Add(new ThreeVertIds(2, 3, 5));

            return Mesh.FromTriangles(points, triangles);
        }
        [Test]
        public void TestFillHole()
        {
            var mesh = CreateMeshWithHoles();
            Assert.That(mesh.HoleRepresentiveEdges.Count, Is.EqualTo(2) );

            var param = new FillHoleParams();
            param.OutNewFaces = new FaceBitSet();

            FillHoles(ref mesh, mesh.HoleRepresentiveEdges.ToList(), param);
            Assert.That(mesh.HoleRepresentiveEdges.Count, Is.EqualTo(0));
        }

        [Test]
        public void TestFillHoleNicely()
        {
            var mesh = CreateMeshWithHoles();
            Assert.That(mesh.HoleRepresentiveEdges.Count, Is.EqualTo(2));

            var param = new FillHoleNicelyParams();

            var patch = FillHoleNicely(ref mesh, mesh.HoleRepresentiveEdges[0], param);

            Assert.That( patch.Count, Is.EqualTo(1887) );
            Assert.That(mesh.HoleRepresentiveEdges.Count, Is.EqualTo(1));
        }

        [Test]
        public void TestRightBoundary()
        {
            var mesh = CreateMeshWithHoles();
            var loops = RegionBoundary.FindRightBoundary(mesh);
            Assert.That(loops.Count, Is.EqualTo(2));
            Assert.That(loops[0].Count, Is.EqualTo(3));
            Assert.That(loops[1].Count, Is.EqualTo(4));
        }

        [Test]
        public void TestFindHoleComplicatedFaces()
        {
            var mesh = CreateMeshWithHoles();
            var complicatedFaces = FindHoleComplicatingFaces(mesh);
            Assert.That(complicatedFaces.Count(), Is.EqualTo(0));
        }
    }
}
