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
            var mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));

            var parameters = new OffsetParameters();
            parameters.voxelSize = suggestVoxelSize(mesh, 8000);

            var offset = offsetMesh(mesh, 0.5f, parameters);
            Assert.That( offset.points.size() == 8792 );

            offset = mcOffsetMesh(mesh, 0.5f, parameters);
            Assert.That( offset.points.size() == 8790 );

            offset = doubleOffsetMesh(mesh, 0.5f, -0.5f, parameters);
            Assert.That( offset.points.size() == 2408 );

            var sharpParameters = new SharpOffsetParameters();
            sharpParameters.voxelSize = suggestVoxelSize(mesh, 8000);

            offset = sharpOffsetMesh(mesh, 0.5f, sharpParameters);
            Assert.That( offset.points.size() == 8790 );

            var generalParameters = new GeneralOffsetParameters();
            generalParameters.voxelSize = suggestVoxelSize(mesh, 8000);

            offset = generalOffsetMesh(mesh, 0.5f, generalParameters);
            Assert.That( offset.points.size() == 8790 );
        }

        [Test]
        public void TestThickenMesh()
        {
            var mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));

            var parameters = new GeneralOffsetParameters();
            parameters.voxelSize = suggestVoxelSize(mesh, 8000);

            var offset = thickenMesh(mesh, 0.5f, parameters);

            Assert.That( offset.points.size() == 8798 );
        }
    }
}
