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
            var mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var laplacian = new Laplacian(mesh);

            // initialize laplacian
            var triangulation = mesh.topology.getTriangulation();
            var i0 = triangulation.front().elems._0;
            var i1 = triangulation.back().elems._0;

            var ancV0 = mesh.points[i0];
            var ancV1 = mesh.points[i1];

            EdgeWeights edgeWeights = EdgeWeights.Unit;
            VertexMass vertexMass = VertexMass.Unit;
            Laplacian.RememberShape rememberShape = Laplacian.RememberShape.No;

            // fix specific vertices
            VertBitSet freeVerts = new VertBitSet();
            freeVerts.resize(mesh.topology.getValidVerts().count());
            freeVerts.set(i0, true);
            freeVerts.set(i1, true);

            laplacian.init(freeVerts, edgeWeights, vertexMass, rememberShape);

            // apply laplacian
            laplacian.fixVertex(i0, ancV0);
            laplacian.fixVertex(i1, ancV1);

            laplacian.apply();

            Assert.That(laplacian is not null);
        }
    }
}
