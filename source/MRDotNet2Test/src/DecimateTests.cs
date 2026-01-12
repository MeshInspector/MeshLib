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
            Mesh sphere = MakeSphere( new SphereParams( 0.5f, 30000 ) );

            var savedRegion = new FaceBitSet((Const_BitSet)sphere.topology.GetValidFaces());

            var parameters = new DecimateSettings();
            var region = new FaceBitSet((Const_BitSet)sphere.topology.GetValidFaces());
            if ( region is not null )
                parameters.region = region;

            parameters.maxTriangleAspectRatio = 80;

            var decimateResult = DecimateMesh(sphere, parameters);
            Assert.That(parameters.region is not null && !parameters.region.Equals( savedRegion ) );
            Assert.That(decimateResult.facesDeleted > 0);
            Assert.That(decimateResult.vertsDeleted > 0);
        }

        [Test]
        public void TestRemesh()
        {
            var sphere = MakeSphere( new SphereParams( 0.5f, 300 ) );
            Assert.That(sphere.topology.GetValidFaces().Count(), Is.EqualTo(596));
            var parameters = new RemeshSettings();
            parameters.targetEdgeLen = 0.1f;
            var remeshResult = Remesh(sphere, parameters);
            Assert.That(sphere.topology.GetValidFaces().Count(), Is.EqualTo(716));
        }
    }
}
