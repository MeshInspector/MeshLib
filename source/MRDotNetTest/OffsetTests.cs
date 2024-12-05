using System;
using System.IO;
using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class OffsetTests
    {
        [Test]
        public void TestOffsets()
        {
            var mp = new MeshPart(Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)));

            var parameters = new OffsetParameters();
            parameters.voxelSize = Offset.SuggestVoxelSize(mp, 8000);
            
            var offset = Offset.OffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.Points.Count == 8792 );

            offset = Offset.McOffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.Points.Count == 8790 );

            offset = Offset.DoubleOffsetMesh(mp, 0.5f, -0.5f, parameters);
            Assert.That( offset.Points.Count == 2408 );

            var generalParameters = new GeneralOffsetParameters();

            offset = Offset.SharpOffsetMesh(mp, 0.5f, parameters, generalParameters);
            Assert.That( offset.Points.Count == 8790 );

            offset = Offset.GeneralOffsetMesh(mp, 0.5f, parameters, generalParameters);
            Assert.That( offset.Points.Count == 8790 );
        }

        [Test]
        public void TestThickenMesh()
        {
            var mp = new MeshPart(Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)));            ;

            var parameters = new OffsetParameters();
            parameters.voxelSize = Offset.SuggestVoxelSize(mp, 8000);

            var offset = Offset.ThickenMesh(mp.mesh, 0.5f, parameters, new GeneralOffsetParameters());

            Assert.That( offset.Points.Count == 8798 );
        }
    }
}
