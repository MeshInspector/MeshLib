using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class FillHoleTests
    {
        [Test]
        public void TestFillHole()
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

            var mesh = Mesh.FromTriangles(points, triangles);
            Assert.That(mesh.HoleRepresentiveEdges.Count, Is.EqualTo(2) );

            var param = new FillHoleParams();
            param.OutNewFaces = new BitSet();

            MeshFillHole.FillHoles(ref mesh, mesh.HoleRepresentiveEdges.ToList(), param);
            Assert.That(mesh.HoleRepresentiveEdges.Count, Is.EqualTo(0));
        }
    }
}
