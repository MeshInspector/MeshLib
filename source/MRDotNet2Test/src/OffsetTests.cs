using System;
using System.IO;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class OffsetTests
    {
        [Test]
        public void TestOffsets()
        {
            var mp = new MeshPart(MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)));

            var parameters = new OffsetParameters();
            parameters.voxelSize = SuggestVoxelSize(mp, 8000);

            var offset = OffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.points.Size() == 8792 );

            offset = McOffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.points.Size() == 8790 );

            offset = DoubleOffsetMesh(mp, 0.5f, -0.5f, parameters);
            Assert.That( offset.points.Size() == 2408 );

            var sharpParameters = new SharpOffsetParameters();
            sharpParameters.voxelSize = SuggestVoxelSize(mp, 8000);

            offset = SharpOffsetMesh(mp, 0.5f, sharpParameters);
            Assert.That( offset.points.Size() == 8790 );

            var generalParameters = new GeneralOffsetParameters();
            generalParameters.voxelSize = SuggestVoxelSize(mp, 8000);

            offset = GeneralOffsetMesh(mp, 0.5f, generalParameters);
            Assert.That( offset.points.Size() == 8790 );
        }

        [Test]
        public void TestThickenMesh()
        {
            var mp = new MeshPart(MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)));

            var parameters = new GeneralOffsetParameters();
            parameters.voxelSize = SuggestVoxelSize(mp, 8000);

            var offset = ThickenMesh(mp.mesh, 0.5f, parameters);

            Assert.That( offset.points.Size() == 8798 );
        }
    }
}
