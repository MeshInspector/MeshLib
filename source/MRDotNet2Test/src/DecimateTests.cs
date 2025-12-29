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

            var savedRegion = new FaceBitSet((Const_BitSet)sphere.Topology.GetValidFaces());

            var parameters = new DecimateSettings();
            var region = new FaceBitSet((Const_BitSet)sphere.Topology.GetValidFaces());
            if ( region is not null )
                parameters.Region = region;

            parameters.MaxTriangleAspectRatio = 80;

            var decimateResult = DecimateMesh(sphere, parameters);
            Assert.That(parameters.Region is not null && !parameters.Region.Equals( savedRegion ) );
            Assert.That(decimateResult.FacesDeleted > 0);
            Assert.That(decimateResult.VertsDeleted > 0);
        }

        [Test]
        public void TestRemesh()
        {
            var sphere = MakeSphere( new SphereParams( 0.5f, 300 ) );
            Assert.That(sphere.Topology.GetValidFaces().Count(), Is.EqualTo(596));
            var parameters = new RemeshSettings();
            parameters.TargetEdgeLen = 0.1f;
            var remeshResult = Remesh(sphere, parameters);
            Assert.That(sphere.Topology.GetValidFaces().Count(), Is.EqualTo(716));
        }
    }
}
