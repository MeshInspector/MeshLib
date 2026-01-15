using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class ICPTests
    {
        [Test]
        public void TestMultiwayICP()
        {
            var inputs = new Std.Vector_MRMeshOrPointsXf();
            Box3f maxBBox = new Box3f();

            inputs.pushBack( new MeshOrPointsXf(makeSphere(new SphereParams(1.0f, 1000)), new AffineXf3f()));
            Box3f bbox = inputs.at(0).obj.getObjBoundingBox();
            if (!maxBBox.valid() || bbox.volume() > maxBBox.volume())
                maxBBox = bbox;

            inputs.pushBack(new MeshOrPointsXf(makeSphere(new SphereParams(1.0f, 1000)), AffineXf3f.linear(Matrix3f.rotation(Vector3f.plusZ(), 0.1f))));
            bbox = inputs.at(1).obj.getObjBoundingBox();
            if (!maxBBox.valid() || bbox.volume() > maxBBox.volume())
                maxBBox = bbox;

            MultiwayICPSamplingParameters samplingParams = new MultiwayICPSamplingParameters();
            samplingParams.samplingVoxelSize = maxBBox.diagonal() * 0.03f;

            MultiwayICP icp = new MultiwayICP(new Vector_MRMeshOrPointsXf_MRObjId(Misc.Move(inputs)), samplingParams);
            ICPProperties iCPProperties = new ICPProperties();
            icp.setParams(iCPProperties);
            icp.updateAllPointPairs();

            Assert.That(icp.getNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.getNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.getMeanSqDistToPoint(), Is.EqualTo(0.00254).Within(1e-5));

            Console.WriteLine("Calculating transformations...");
            var xfs = icp.calculateTransformations();
            Assert.That(icp.getNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.getNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.getMeanSqDistToPoint(), Is.EqualTo(0).Within(1e-5));
        }

        [Test]
        public void TestICP()
        {
            var torusRef = makeTorus(2, 1, 32, 32);
            var torusMove = makeTorus(2, 1, 32, 32);

            var axis = Vector3f.plusX();
            var trans = new Vector3f(0.0f, 0.2f, 0.105f);
            var xf = new AffineXf3f(Matrix3f.rotation(axis, 0.2f), trans);

            MeshOrPointsXf flt = new MeshOrPointsXf(torusMove, xf );
            MeshOrPointsXf refer = new MeshOrPointsXf( torusRef, new AffineXf3f());

            var fltSamples = torusMove.topology.getValidVerts();
            var referSamples = torusRef.topology.getValidVerts();
            Assert.That(fltSamples is not null);
            Assert.That(referSamples is not null);

            if (fltSamples is null || referSamples is null)
                return;

            var icp = new ICP(flt, refer, fltSamples, referSamples);

            var newXf = icp.calculateTransformation();
            Console.WriteLine(icp.getStatusInfo());

            var diffXf = new AffineXf3f();
            diffXf.a -= newXf.a;
            diffXf.b -= newXf.b;

            Assert.That(Math.Abs(diffXf.a.x.x), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.x.y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.x.z), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.y.x), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.y.y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.y.z), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.z.x), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.z.y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.a.z.z), Is.LessThan(1e-6f));

            Assert.That(Math.Abs(diffXf.b.x), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.b.y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.b.z), Is.LessThan(1e-6f));

            var pairs = icp.getRef2FltPairs();
            Assert.That(pairs.size(), Is.EqualTo(1024));

            pairs = icp.getFlt2RefPairs();
            Assert.That(pairs.size(), Is.EqualTo(1024));
        }
    }
}
