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
            var mp = new MeshPart(makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f)));

            var parameters = new OffsetParameters();
            parameters.voxelSize = suggestVoxelSize(mp, 8000);

            var offset = offsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.points.size() == 8792 );

            offset = mcOffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.points.size() == 8790 );

            offset = doubleOffsetMesh(mp, 0.5f, -0.5f, parameters);
            Assert.That( offset.points.size() == 2408 );

            var sharpParameters = new SharpOffsetParameters();
            sharpParameters.voxelSize = suggestVoxelSize(mp, 8000);

            offset = sharpOffsetMesh(mp, 0.5f, sharpParameters);
            Assert.That( offset.points.size() == 8790 );

            var generalParameters = new GeneralOffsetParameters();
            generalParameters.voxelSize = suggestVoxelSize(mp, 8000);

            offset = generalOffsetMesh(mp, 0.5f, generalParameters);
            Assert.That( offset.points.size() == 8790 );
        }

        [Test]
        public void TestThickenMesh()
        {
            var mp = new MeshPart(makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f)));

            var parameters = new GeneralOffsetParameters();
            parameters.voxelSize = suggestVoxelSize(mp, 8000);

            var offset = thickenMesh(mp.mesh, 0.5f, parameters);

            Assert.That( offset.points.size() == 8798 );
        }
    }
}
