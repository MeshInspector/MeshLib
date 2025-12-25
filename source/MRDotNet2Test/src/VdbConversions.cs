using NUnit.Framework;
using static MR;

namespace MRTest
{

    [TestFixture]
    internal class VdbConversionsTests
    {
        internal static VdbVolume CreateVolume()
        {
            var mesh = MakeSphere(new SphereParams(1.0f, 3000));
            var settings = new MeshToVolumeParams();
            /*
             * TODO: fix struct field assignment
            settings.VoxelSize = Vector3f.Diagonal(0.1f);
             */
            settings.VoxelSize.X = 0.1f;
            settings.VoxelSize.Y = 0.1f;
            settings.VoxelSize.Z = 0.1f;
            return MeshToVolume(new MeshPart(mesh), settings);
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
            /*
             * TODO: fix struct field assignment
            gridToMeshSettings.VoxelSize = Vector3f.Diagonal(0.1f);
             */
            gridToMeshSettings.VoxelSize.X = 0.1f;
            gridToMeshSettings.VoxelSize.Y = 0.1f;
            gridToMeshSettings.VoxelSize.Z = 0.1f;
            gridToMeshSettings.IsoValue = 1;

            var restored = GridToMesh(vdbVolume.Data, gridToMeshSettings);
            var bbox = restored.GetBoundingBox();

            Assert.That(restored.Points.Size(), Is.EqualTo(3748));
            Assert.That(bbox.Min.X, Is.EqualTo(0.2).Within(0.001));
            Assert.That(bbox.Min.Y, Is.EqualTo(0.2).Within(0.001));
            Assert.That(bbox.Min.Z, Is.EqualTo(0.2).Within(0.001));

            Assert.That(bbox.Max.X, Is.EqualTo(2.395).Within(0.001));
            Assert.That(bbox.Max.Y, Is.EqualTo(2.395).Within(0.001));
            Assert.That(bbox.Max.Z, Is.EqualTo(2.395).Within(0.001));
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

            Assert.That(restored.Size(), Is.EqualTo(1));

            var readVolume = restored.At(0);
            Assert.That(readVolume.Dims.X, Is.EqualTo(26));
            Assert.That(readVolume.Dims.Y, Is.EqualTo(26));
            Assert.That(readVolume.Dims.Z, Is.EqualTo(26));
            Assert.That(readVolume.Min, Is.EqualTo(0).Within(0.001));
            Assert.That(readVolume.Max, Is.EqualTo(3).Within(0.001));

            File.Delete(tempFile);
        }

        [Test]
        public void TestUniformResampling()
        {
            var vdbVolume = CreateVolume();
            var resampledGrid = Resampled(vdbVolume.Data, 2);
            var resampledVolume = FloatGridToVdbVolume(resampledGrid);
            /*
             * TODO: fix struct field assignment
            resampledVolume.VoxelSize = vdbVolume.VoxelSize * 2;
             */
            resampledVolume.VoxelSize.X = vdbVolume.VoxelSize.X * 2;
            resampledVolume.VoxelSize.Y = vdbVolume.VoxelSize.Y * 2;
            resampledVolume.VoxelSize.Z = vdbVolume.VoxelSize.Z * 2;

            Assert.That(resampledVolume.Dims.X, Is.EqualTo(13));
            Assert.That(resampledVolume.Dims.Y, Is.EqualTo(13));
            Assert.That(resampledVolume.Dims.Z, Is.EqualTo(13));

            Assert.That(resampledVolume.VoxelSize.X, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(resampledVolume.VoxelSize.Y, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(resampledVolume.VoxelSize.Z, Is.EqualTo(0.2f).Within(0.001f));
        }

        [Test]
        public void TestResampling()
        {
            var vdbVolume = CreateVolume();
            var resampledGrid = Resampled(vdbVolume.Data, new Vector3f( 2.0f, 1.0f, 0.5f ) );
            var resampledVolume = FloatGridToVdbVolume(resampledGrid);

            resampledVolume.VoxelSize.X = vdbVolume.VoxelSize.X * 2;
            resampledVolume.VoxelSize.Y = vdbVolume.VoxelSize.Y * 1;
            resampledVolume.VoxelSize.Z = vdbVolume.VoxelSize.Z * 0.5f;

            Assert.That(resampledVolume.Dims.X, Is.EqualTo(13));
            Assert.That(resampledVolume.Dims.Y, Is.EqualTo(27));
            Assert.That(resampledVolume.Dims.Z, Is.EqualTo(53));

            Assert.That(resampledVolume.VoxelSize.X, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(resampledVolume.VoxelSize.Y, Is.EqualTo(0.1f).Within(0.001f));
            Assert.That(resampledVolume.VoxelSize.Z, Is.EqualTo(0.05f).Within(0.001f));

            var tempFile = Path.GetTempFileName() + ".vdb";
            VoxelsSave.ToAnySupportedFormat(resampledVolume, tempFile);

            var restored = VoxelsLoad.FromAnySupportedFormat(tempFile);
            Assert.That(restored is not null);
            if (restored is null)
                return;

            Assert.That(restored.Size(), Is.EqualTo(1));

            var readVolume = restored.At(0);
            Assert.That(readVolume.VoxelSize.X, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(readVolume.VoxelSize.Y, Is.EqualTo(0.1f).Within(0.001f));
            Assert.That(readVolume.VoxelSize.Z, Is.EqualTo(0.05f).Within(0.001f));

            File.Delete(tempFile);
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

            var region = new VoxelBitSet( (ulong)( vdbVolume.Dims.X * vdbVolume.Dims.Y * vdbVolume.Dims.Z ) );
            region.Set(new VoxelId(0));
            SetValue( vdbVolume.Data, region, 1.0f );
            Assert.That( GetValue( vdbVolume.Data, p ) == 1.0f );

            SetValue( vdbVolume.Data, p, 2.0f );
            Assert.That( GetValue( vdbVolume.Data, p ) == 2.0f );
        }
    }
}