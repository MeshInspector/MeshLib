using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class LaplacianTests
    {
        [Test]
        public void TestLaplacian()
        {
            var mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var laplacian = new Laplacian(mesh);

            // initialize laplacian
            var i0 = mesh.Triangulation.FirstOrDefault().v0;
            var i1 = mesh.Triangulation.LastOrDefault().v0;

            var ancV0 = mesh.Points[i0.Id];
            var ancV1 = mesh.Points[i1.Id];

            EdgeWeights edgeWeights = EdgeWeights.Unit;
            VertexMass vertexMass = VertexMass.Unit;
            LaplacianRememberShape rememberShape = LaplacianRememberShape.No;

            // fix specific vertices
            VertBitSet freeVerts = new VertBitSet();
            freeVerts.Resize(mesh.ValidPoints.Count());
            freeVerts.Set(i0.Id, true);
            freeVerts.Set(i1.Id, true);

            laplacian.Init(freeVerts, edgeWeights, vertexMass, rememberShape);

            // apply laplacian
            laplacian.FixVertex(i0, ref ancV0);
            laplacian.FixVertex(i1, ref ancV1);

            laplacian.Apply();

            Assert.That(laplacian is not null);
        }
    }
}