using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class MeshNormalsTests
    {
        [Test]
        public void TestVertNormals()
        {
            var mesh = Mesh.MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );
            var normals = MeshNormals.ComputePerVertNormals( mesh );
            Assert.That( normals.Count, Is.EqualTo( 8 ) );
        }

        [Test]
        public void TestFaceNormals()
        {
            var mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var normals = MeshNormals.ComputePerFaceNormals(mesh);
            Assert.That(normals.Count, Is.EqualTo(12));
        }
    }
}
