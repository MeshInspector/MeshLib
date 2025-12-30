using System;
using System.IO;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class BooleanTests
    {
        [Test]
        public void TestOperations()
        {
            const float PI = 3.14159265f;
            Mesh meshA = MakeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = MakeTorus(1.0f, 0.2f, 8, 8);
            meshB.Transform( AffineXf3f.Linear( Matrix3f.Rotation( Vector3f.PlusZ(), Vector3f.PlusY() ) ) );

            const float shiftStep = 0.2f;
            const float angleStep = PI;
            var baseAxis = new Vector3f[] { Vector3f.PlusX(), Vector3f.PlusY(), Vector3f.PlusZ() };

            for (int maskTrans = 0; maskTrans < 8; ++maskTrans)
            {
                for (int maskRot = 0; maskRot < 8; ++maskRot)
                {
                    for (float shift = 0.01f; shift < 0.2f; shift += shiftStep)
                    {
                        Vector3f shiftVec = new Vector3f();
                        for (int i = 0; i < 3; ++i)
                            if ( ( maskTrans & ( 1 << i ) ) > 0 )
                                shiftVec += shift * baseAxis[i];

                        for (float angle = PI * 0.01f; angle < PI * 7.0f / 18.0f; angle += angleStep)
                        {
                            Matrix3f rotation = new Matrix3f();
                            for (int i = 0; i < 3; ++i)
                                if ( ( maskRot & (1 << i) ) > 0 )
                                    rotation = Matrix3f.Rotation( baseAxis[i], angle ) * rotation;

                            BooleanParameters parameters = new BooleanParameters();
                            parameters.rigidB2A = AffineXf3f.Translation(shiftVec) * AffineXf3f.Linear(rotation);

                            Assert.DoesNotThrow(() => Boolean(meshA, meshB, BooleanOperation.Union, parameters));
                            Assert.DoesNotThrow(() => Boolean(meshA, meshB, BooleanOperation.Intersection, parameters));
                        }
                    }
                }
            }
        }

        [Test]
        public void TestMapper()
        {
            Mesh meshA = MakeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = MakeTorus(1.0f, 0.2f, 8, 8);
            meshB.Transform(AffineXf3f.Linear(Matrix3f.Rotation(Vector3f.PlusZ(), Vector3f.PlusY())));

            var parameters = new BooleanParameters();
            parameters.mapper = new BooleanResultMapper();
            var booleanResult = Boolean(meshA, meshB, BooleanOperation.Union, parameters);

            var validPointsA = meshA.topology.GetValidVerts();
            var validPointsB = meshB.topology.GetValidVerts();
            var validFacesA = meshA.topology.GetValidFaces();
            var validFacesB = meshB.topology.GetValidFaces();

            var old2NewVerts = parameters.mapper.GetMaps(BooleanResultMapper.MapObject.A).old2newVerts;
            var vMapA = parameters.mapper.Map(validPointsA, BooleanResultMapper.MapObject.A);
            var vMapB = parameters.mapper.Map(validPointsB, BooleanResultMapper.MapObject.B);

            Assert.That(vMapA.Size(), Is.EqualTo(60) );
            Assert.That(vMapA.Count(), Is.EqualTo(60));
            Assert.That(vMapB.Size(), Is.EqualTo(204) );
            Assert.That(vMapB.Count(), Is.EqualTo(48));


            var fMapA = parameters.mapper.Map(validFacesA, BooleanResultMapper.MapObject.A);
            var fMapB = parameters.mapper.Map(validFacesB, BooleanResultMapper.MapObject.B);

            Assert.That(fMapA.Size(), Is.EqualTo(224) );
            Assert.That(fMapA.Count(), Is.EqualTo(224));
            Assert.That(fMapB.Size(), Is.EqualTo(416) );
            Assert.That(fMapB.Count(), Is.EqualTo(192));

            var newFaces = parameters.mapper.NewFaces();
            Assert.That(newFaces.Size(), Is.EqualTo(416) );
            Assert.That(newFaces.Count(), Is.EqualTo(252));

            var mapsA = parameters.mapper.GetMaps( BooleanResultMapper.MapObject.A );
            Assert.That(!mapsA.identity);
            Assert.That( mapsA.old2newVerts.Size(), Is.EqualTo(160) );
            Assert.That( mapsA.cut2newFaces.Size(), Is.EqualTo(348) );
            Assert.That( mapsA.cut2origin.Size(), Is.EqualTo(348) );

            var mapsB = parameters.mapper.GetMaps( BooleanResultMapper.MapObject.B );
            Assert.That(!mapsB.identity);
            Assert.That( mapsB.old2newVerts.Size(), Is.EqualTo(160) );
            Assert.That( mapsB.cut2newFaces.Size(), Is.EqualTo(384) );
            Assert.That( mapsB.cut2origin.Size(), Is.EqualTo(384) );
        }
    }
}
