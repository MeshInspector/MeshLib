using System;
using System.IO;
using NUnit.Framework;

using MR.DotNet;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class DecimateTests
    {
        [Test]
        public void TestDecimate()
        {
            var cylinder = Mesh.MakeCylinder( 0.5f, 0.0f, 20.0f / 180.0f * (float)Math.PI, 1.0f);

            BitSetReadOnly savedRegion = cylinder.ValidFaces.Clone();

            var parameters = new DecimateParameters();
            parameters.region = cylinder.ValidFaces.Clone() as BitSet;
            parameters.maxTriangleAspectRatio = 80;

            var decimateResult = MeshDecimate.Decimate(cylinder, parameters);
            Assert.That(parameters.region != savedRegion );
            Assert.That(decimateResult.facesDeleted > 0);
            Assert.That(decimateResult.vertsDeleted > 0);
        }
    }
}
