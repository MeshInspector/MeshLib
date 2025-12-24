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
            var mesh = MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) ).Value; // TODO: replace _Moved
            var normals = ComputePerVertNormals( mesh ).Value; // TODO: replace _Moved
            Assert.That( normals.Size(), Is.EqualTo( 8 ) );
        }

        [Test]
        public void TestFaceNormals()
        {
            var mesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)).Value; // TODO: replace _Moved
            var normals = ComputePerFaceNormals(mesh).Value; // TODO: replace _Moved
            Assert.That(normals.Size(), Is.EqualTo(12));
        }
    }
}
