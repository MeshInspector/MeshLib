using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class LaplacianTests
    {
        [Test]
        public void TestLaplacian()
        {
            var mesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var laplacian = new Laplacian(mesh);

            // initialize laplacian
            var triangulation = mesh.Topology.GetTriangulation();
            var i0 = triangulation.Front().Elems._0;
            var i1 = triangulation.Back().Elems._0;

            var ancV0 = mesh.Points.Index(i0);
            var ancV1 = mesh.Points.Index(i1);

            EdgeWeights edgeWeights = EdgeWeights.Unit;
            VertexMass vertexMass = VertexMass.Unit;
            Laplacian.RememberShape rememberShape = Laplacian.RememberShape.No;

            // fix specific vertices
            VertBitSet freeVerts = new VertBitSet();
            freeVerts.Resize(mesh.Topology.GetValidVerts().Count());
            freeVerts.Set(i0, true);
            freeVerts.Set(i1, true);

            laplacian.Init(freeVerts, edgeWeights, vertexMass, rememberShape);

            // apply laplacian
            laplacian.FixVertex(i0, ancV0);
            laplacian.FixVertex(i1, ancV1);

            laplacian.Apply();

            Assert.That(laplacian is not null);
        }
    }
}