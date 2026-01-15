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
            Mesh meshA = makeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = makeTorus(1.0f, 0.2f, 8, 8);
            meshB.transform( AffineXf3f.linear( Matrix3f.rotation( Vector3f.plusZ(), Vector3f.plusY() ) ) );

            const float shiftStep = 0.2f;
            const float angleStep = PI;
            var baseAxis = new Vector3f[] { Vector3f.plusX(), Vector3f.plusY(), Vector3f.plusZ() };

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
                                    rotation = Matrix3f.rotation( baseAxis[i], angle ) * rotation;

                            BooleanParameters parameters = new BooleanParameters();
                            parameters.rigidB2A = AffineXf3f.translation(shiftVec) * AffineXf3f.linear(rotation);

                            Assert.DoesNotThrow(() => boolean(meshA, meshB, BooleanOperation.Union, parameters));
                            Assert.DoesNotThrow(() => boolean(meshA, meshB, BooleanOperation.Intersection, parameters));
                        }
                    }
                }
            }
        }

        [Test]
        public void TestMapper()
        {
            Mesh meshA = makeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = makeTorus(1.0f, 0.2f, 8, 8);
            meshB.transform(AffineXf3f.linear(Matrix3f.rotation(Vector3f.plusZ(), Vector3f.plusY())));

            var parameters = new BooleanParameters();
            parameters.mapper = new BooleanResultMapper();
            var booleanResult = boolean(meshA, meshB, BooleanOperation.Union, parameters);

            var validPointsA = meshA.topology.getValidVerts();
            var validPointsB = meshB.topology.getValidVerts();
            var validFacesA = meshA.topology.getValidFaces();
            var validFacesB = meshB.topology.getValidFaces();

            var old2NewVerts = parameters.mapper.getMaps(BooleanResultMapper.MapObject.A).old2newVerts;
            var vMapA = parameters.mapper.map(validPointsA, BooleanResultMapper.MapObject.A);
            var vMapB = parameters.mapper.map(validPointsB, BooleanResultMapper.MapObject.B);

            Assert.That(vMapA.size(), Is.EqualTo(60) );
            Assert.That(vMapA.count(), Is.EqualTo(60));
            Assert.That(vMapB.size(), Is.EqualTo(204) );
            Assert.That(vMapB.count(), Is.EqualTo(48));


            var fMapA = parameters.mapper.map(validFacesA, BooleanResultMapper.MapObject.A);
            var fMapB = parameters.mapper.map(validFacesB, BooleanResultMapper.MapObject.B);

            Assert.That(fMapA.size(), Is.EqualTo(224) );
            Assert.That(fMapA.count(), Is.EqualTo(224));
            Assert.That(fMapB.size(), Is.EqualTo(416) );
            Assert.That(fMapB.count(), Is.EqualTo(192));

            var newFaces = parameters.mapper.newFaces();
            Assert.That(newFaces.size(), Is.EqualTo(416) );
            Assert.That(newFaces.count(), Is.EqualTo(252));

            var mapsA = parameters.mapper.getMaps( BooleanResultMapper.MapObject.A );
            Assert.That(!mapsA.identity);
            Assert.That( mapsA.old2newVerts.size(), Is.EqualTo(160) );
            Assert.That( mapsA.cut2newFaces.size(), Is.EqualTo(348) );
            Assert.That( mapsA.cut2origin.size(), Is.EqualTo(348) );

            var mapsB = parameters.mapper.getMaps( BooleanResultMapper.MapObject.B );
            Assert.That(!mapsB.identity);
            Assert.That( mapsB.old2newVerts.size(), Is.EqualTo(160) );
            Assert.That( mapsB.cut2newFaces.size(), Is.EqualTo(384) );
            Assert.That( mapsB.cut2origin.size(), Is.EqualTo(384) );
        }
    }
}
