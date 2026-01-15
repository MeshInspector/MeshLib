using System;
using System.IO;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class DecimateTests
    {
        [Test]
        public void TestDecimate()
        {
            Mesh sphere = makeSphere( new SphereParams( 0.5f, 30000 ) );

            var savedRegion = new FaceBitSet((Const_BitSet)sphere.topology.getValidFaces());

            var parameters = new DecimateSettings();
            var region = new FaceBitSet((Const_BitSet)sphere.topology.getValidFaces());
            if ( region is not null )
                parameters.region = region;

            parameters.maxError = 1e-3f;
            parameters.maxTriangleAspectRatio = 80;

            var decimateResult = decimateMesh(sphere, parameters);
            Assert.That(parameters.region is not null && !parameters.region.Equals( savedRegion ) );
            Assert.That(decimateResult.facesDeleted > 0);
            Assert.That(decimateResult.vertsDeleted > 0);
        }

        [Test]
        public void TestRemesh()
        {
            var sphere = makeSphere( new SphereParams( 0.5f, 300 ) );
            Assert.That(sphere.topology.getValidFaces().count(), Is.EqualTo(596));
            var parameters = new RemeshSettings();
            parameters.targetEdgeLen = 0.1f;
            var remeshResult = remesh(sphere, parameters);
            Assert.That(sphere.topology.getValidFaces().count(), Is.EqualTo(716));
        }
    }
}
