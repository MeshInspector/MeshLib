using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{

    [TestFixture]
    internal class VdbConversionsTests
    {
        internal static VdbVolume CreateVolume()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var settings = new MeshToVolumeSettings();
            settings.voxelSize = Vector3f.Diagonal(0.1f);
            return MeshToVolume(mesh, settings);
        }

        [Test]
        public void TestConversions()
        {
            var vdbVolume = CreateVolume();
            Assert.That(vdbVolume.Min, Is.EqualTo(0).Within(0.001));
            Assert.That(vdbVolume.Max, Is.EqualTo(3).Within(0.001));
            Assert.That(vdbVolume.Dims.X, Is.EqualTo(26));
            Assert.That(vdbVolume.Dims.Y, Is.EqualTo(26));
            Assert.That(vdbVolume.Dims.Z, Is.EqualTo(26));

            var gridToMeshSettings = new GridToMeshSettings();
            gridToMeshSettings.voxelSize = Vector3f.Diagonal(0.1f);
            gridToMeshSettings.isoValue = 1;

            var restored = GridToMesh(vdbVolume.Data, gridToMeshSettings);

            Assert.That(restored.Points.Count, Is.EqualTo(3748));
            Assert.That(restored.BoundingBox.Min.X, Is.EqualTo(0.2).Within(0.001));
            Assert.That(restored.BoundingBox.Min.Y, Is.EqualTo(0.2).Within(0.001));
            Assert.That(restored.BoundingBox.Min.Z, Is.EqualTo(0.2).Within(0.001));

            Assert.That(restored.BoundingBox.Max.X, Is.EqualTo(2.395).Within(0.001));
            Assert.That(restored.BoundingBox.Max.Y, Is.EqualTo(2.395).Within(0.001));
            Assert.That(restored.BoundingBox.Max.Z, Is.EqualTo(2.395).Within(0.001));
        }

        [Test]
        public void TestSaveLoad()
        {
            var vdbVolume = CreateVolume();

            var tempFile = Path.GetTempFileName() + ".vdb";
            VoxelsSave.ToAnySupportedFormat(vdbVolume, tempFile);

            var restored = VoxelsLoad.FromAnySupportedFormat(tempFile);
            Assert.That(restored is not null);
            if (restored is null)
                return;

            Assert.That(restored.Count, Is.EqualTo(1));

            var readVolume = restored[0];
            Assert.That(readVolume.Dims.X, Is.EqualTo(26));
            Assert.That(readVolume.Dims.Y, Is.EqualTo(26));
            Assert.That(readVolume.Dims.Z, Is.EqualTo(26));
            Assert.That(readVolume.Min, Is.EqualTo(0).Within(0.001));
            Assert.That(readVolume.Max, Is.EqualTo(3).Within(0.001));
        }

        [Test]
        public void TestUniformResampling()
        {
            var vdbVolume = CreateVolume();
            var resampledGrid = Resampled(vdbVolume.Data, 2);
            var resampledVolume = FloatGridToVdbVolume(resampledGrid);

            Assert.That(resampledVolume.Dims.X, Is.EqualTo(13));
            Assert.That(resampledVolume.Dims.Y, Is.EqualTo(13));
            Assert.That(resampledVolume.Dims.Z, Is.EqualTo(13));
        }

        [Test]
        public void TestResampling()
        {
            var vdbVolume = CreateVolume();
            var resampledGrid = Resampled(vdbVolume.Data, new Vector3f( 2.0f, 1.0f, 0.5f ) );
            var resampledVolume = FloatGridToVdbVolume(resampledGrid);

            Assert.That(resampledVolume.Dims.X, Is.EqualTo(13));
            Assert.That(resampledVolume.Dims.Y, Is.EqualTo(27));
            Assert.That(resampledVolume.Dims.Z, Is.EqualTo(53));
        }

        [Test]
        public void TestCropping()
        {
            var vdbVolume = CreateVolume();
            var box = new Box3i();
            box.Min.X = 2;
            box.Min.Y = 5;
            box.Min.Z = 1;
            box.Max.X = 18;
            box.Max.Y = 13;
            box.Max.Z = 23;

            var croppedGrid = Cropped(vdbVolume.Data, box);
            var croppedVolume = FloatGridToVdbVolume(croppedGrid);

            Assert.That(croppedVolume.Dims.X, Is.EqualTo(16));
            Assert.That(croppedVolume.Dims.Y, Is.EqualTo(8));
            Assert.That(croppedVolume.Dims.Z, Is.EqualTo(22));
        }

        [Test]
        public void TestAccessors()
        {
            var vdbVolume = CreateVolume();
            var p = new Vector3i();
            Assert.That( GetValue( vdbVolume.Data, p ) == 3.0f );

            var region = new VoxelBitSet( vdbVolume.Dims.X * vdbVolume.Dims.Y * vdbVolume.Dims.Z );
            region.Set(0);
            SetValue( vdbVolume.Data, region, 1.0f );
            Assert.That( GetValue( vdbVolume.Data, p ) == 1.0f );

            SetValue( vdbVolume.Data, p, 2.0f );
            Assert.That( GetValue( vdbVolume.Data, p ) == 2.0f );
        }
    }
}