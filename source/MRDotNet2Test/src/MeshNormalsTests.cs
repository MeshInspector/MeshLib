using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class MeshNormalsTests
    {
        [Test]
        public void TestVertNormals()
        {
            var mesh = MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );
            var normals = ComputePerVertNormals( mesh );
            Assert.That( normals.Size(), Is.EqualTo( 8 ) );
        }

        [Test]
        public void TestFaceNormals()
        {
            var mesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var normals = ComputePerFaceNormals(mesh);
            Assert.That(normals.Size(), Is.EqualTo(12));
        }
    }
}
