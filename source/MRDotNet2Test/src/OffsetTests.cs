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
            parameters.VoxelSize = SuggestVoxelSize(mp, 8000);

            var offset = OffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.Points.Size() == 8792 );

            offset = McOffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.Points.Size() == 8790 );

            offset = DoubleOffsetMesh(mp, 0.5f, -0.5f, parameters);
            Assert.That( offset.Points.Size() == 2408 );

            var sharpParameters = new SharpOffsetParameters();
            sharpParameters.VoxelSize = SuggestVoxelSize(mp, 8000);

            offset = SharpOffsetMesh(mp, 0.5f, sharpParameters);
            Assert.That( offset.Points.Size() == 8790 );

            var generalParameters = new GeneralOffsetParameters();
            generalParameters.VoxelSize = SuggestVoxelSize(mp, 8000);

            offset = GeneralOffsetMesh(mp, 0.5f, generalParameters);
            Assert.That( offset.Points.Size() == 8790 );
        }

        [Test]
        public void TestThickenMesh()
        {
            var mp = new MeshPart(MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)));

            var parameters = new GeneralOffsetParameters();
            parameters.VoxelSize = SuggestVoxelSize(mp, 8000);

            var offset = ThickenMesh(mp.Mesh, 0.5f, parameters);

            Assert.That( offset.Points.Size() == 8798 );
        }
    }
}
