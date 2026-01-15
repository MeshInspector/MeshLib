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
            var mesh = makeCube( Vector3f.diagonal(1), Vector3f.diagonal(-0.5f) );
            var normals = computePerVertNormals( mesh );
            Assert.That( normals.size(), Is.EqualTo( 8 ) );
        }

        [Test]
        public void TestFaceNormals()
        {
            var mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var normals = computePerFaceNormals(mesh);
            Assert.That(normals.size(), Is.EqualTo(12));
        }
    }
}
