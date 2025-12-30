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

            inputs.PushBack( new MeshOrPointsXf(MakeSphere(new SphereParams(1.0f, 1000)), new AffineXf3f()));
            Box3f bbox = inputs.At(0).obj.GetObjBoundingBox();
            if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                maxBBox = bbox;

            inputs.PushBack(new MeshOrPointsXf(MakeSphere(new SphereParams(1.0f, 1000)), AffineXf3f.Linear(Matrix3f.Rotation(Vector3f.PlusZ(), 0.1f))));
            bbox = inputs.At(1).obj.GetObjBoundingBox();
            if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                maxBBox = bbox;

            MultiwayICPSamplingParameters samplingParams = new MultiwayICPSamplingParameters();
            samplingParams.samplingVoxelSize = maxBBox.Diagonal() * 0.03f;

            MultiwayICP icp = new MultiwayICP(new Vector_MRMeshOrPointsXf_MRObjId(Misc.Move(inputs)), samplingParams);
            ICPProperties iCPProperties = new ICPProperties();
            icp.SetParams(iCPProperties);
            icp.UpdateAllPointPairs();

            Assert.That(icp.GetNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.GetNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.GetMeanSqDistToPoint(), Is.EqualTo(0.00254).Within(1e-5));

            Console.WriteLine("Calculating transformations...");
            var xfs = icp.CalculateTransformations();
            Assert.That(icp.GetNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.GetNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.GetMeanSqDistToPoint(), Is.EqualTo(0).Within(1e-5));
        }

        [Test]
        public void TestICP()
        {
            var torusRef = MakeTorus(2, 1, 32, 32);
            var torusMove = MakeTorus(2, 1, 32, 32);

            var axis = Vector3f.PlusX();
            var trans = new Vector3f(0.0f, 0.2f, 0.105f);
            var xf = new AffineXf3f(Matrix3f.Rotation(axis, 0.2f), trans);

            MeshOrPointsXf flt = new MeshOrPointsXf(torusMove, xf );
            MeshOrPointsXf refer = new MeshOrPointsXf( torusRef, new AffineXf3f());

            var fltSamples = torusMove.topology.GetValidVerts();
            var referSamples = torusRef.topology.GetValidVerts();
            Assert.That(fltSamples is not null);
            Assert.That(referSamples is not null);

            if (fltSamples is null || referSamples is null)
                return;

            var icp = new ICP(flt, refer, fltSamples, referSamples);

            var newXf = icp.CalculateTransformation();
            Console.WriteLine(icp.GetStatusInfo());

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

            var pairs = icp.GetRef2FltPairs();
            Assert.That(pairs.Size(), Is.EqualTo(1024));

            pairs = icp.GetFlt2RefPairs();
            Assert.That(pairs.Size(), Is.EqualTo(1024));
        }
    }
}
