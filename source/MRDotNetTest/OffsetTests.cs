using System;
using System.IO;
using NUnit.Framework;

using MR.DotNet;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class OffsetTests
    {
        [Test]
        public void TestOffset()
        {
            var mp = new MeshPart();
            mp.mesh = Mesh.MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );

            var parameters = new OffsetParameters();
            parameters.voxelSize = Offset.SuggestVoxelSize(mp, 8000);
            
            var offset = Offset.OffsetMesh(mp, 0.5f, parameters);
            Assert.That( offset.Points.Count == 8792 );
        }
    }
}
