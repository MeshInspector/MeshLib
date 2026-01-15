using NUnit.Framework;
using static MR;

namespace MRTest
{

    [TestFixture]
    internal class VdbConversionsTests
    {
        internal static VdbVolume CreateVolume()
        {
            var mesh = makeSphere(new SphereParams(1.0f, 3000));
            var settings = new MeshToVolumeParams();
            settings.voxelSize = Vector3f.diagonal(0.1f);
            return meshToVolume(new MeshPart(mesh), settings);
        }

        [Test]
        public void TestConversions()
        {
            var vdbVolume = CreateVolume();
            Assert.That(vdbVolume.min, Is.EqualTo(0).Within(0.001));
            Assert.That(vdbVolume.max, Is.EqualTo(3).Within(0.001));
            Assert.That(vdbVolume.dims.x, Is.EqualTo(26));
            Assert.That(vdbVolume.dims.y, Is.EqualTo(26));
            Assert.That(vdbVolume.dims.z, Is.EqualTo(26));

            var gridToMeshSettings = new GridToMeshSettings();
            /*
             * TODO: fix struct field assignment
            gridToMeshSettings.voxelSize = Vector3f.diagonal(0.1f);
             */
            gridToMeshSettings.voxelSize.x = 0.1f;
            gridToMeshSettings.voxelSize.y = 0.1f;
            gridToMeshSettings.voxelSize.z = 0.1f;
            gridToMeshSettings.isoValue = 1;

            var restored = gridToMesh(vdbVolume.data, gridToMeshSettings);
            var bbox = restored.getBoundingBox();

            Assert.That(restored.points.size(), Is.EqualTo(3748));
            Assert.That(bbox.min.x, Is.EqualTo(0.2).Within(0.001));
            Assert.That(bbox.min.y, Is.EqualTo(0.2).Within(0.001));
            Assert.That(bbox.min.z, Is.EqualTo(0.2).Within(0.001));

            Assert.That(bbox.max.x, Is.EqualTo(2.395).Within(0.001));
            Assert.That(bbox.max.y, Is.EqualTo(2.395).Within(0.001));
            Assert.That(bbox.max.z, Is.EqualTo(2.395).Within(0.001));
        }

        [Test]
        public void TestSaveLoad()
        {
            var vdbVolume = CreateVolume();

            var tempFile = Path.GetTempFileName() + ".vdb";
            VoxelsSave.toAnySupportedFormat(vdbVolume, tempFile);

            var restored = VoxelsLoad.fromAnySupportedFormat(tempFile);
            Assert.That(restored is not null);
            if (restored is null)
                return;

            Assert.That(restored.size(), Is.EqualTo(1));

            var readVolume = restored.at(0);
            Assert.That(readVolume.dims.x, Is.EqualTo(26));
            Assert.That(readVolume.dims.y, Is.EqualTo(26));
            Assert.That(readVolume.dims.z, Is.EqualTo(26));
            Assert.That(readVolume.min, Is.EqualTo(0).Within(0.001));
            Assert.That(readVolume.max, Is.EqualTo(3).Within(0.001));

            File.Delete(tempFile);
        }

        [Test]
        public void TestUniformResampling()
        {
            var vdbVolume = CreateVolume();
            var resampledGrid = resampled(vdbVolume.data, 2);
            var resampledVolume = floatGridToVdbVolume(resampledGrid);
            /*
             * TODO: fix struct field assignment
            resampledVolume.voxelSize = vdbVolume.voxelSize * 2;
             */
            resampledVolume.voxelSize.x = vdbVolume.voxelSize.x * 2;
            resampledVolume.voxelSize.y = vdbVolume.voxelSize.y * 2;
            resampledVolume.voxelSize.z = vdbVolume.voxelSize.z * 2;

            Assert.That(resampledVolume.dims.x, Is.EqualTo(13));
            Assert.That(resampledVolume.dims.y, Is.EqualTo(13));
            Assert.That(resampledVolume.dims.z, Is.EqualTo(13));

            Assert.That(resampledVolume.voxelSize.x, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(resampledVolume.voxelSize.y, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(resampledVolume.voxelSize.z, Is.EqualTo(0.2f).Within(0.001f));
        }

        [Test]
        public void TestResampling()
        {
            var vdbVolume = CreateVolume();
            var resampledGrid = resampled(vdbVolume.data, new Vector3f( 2.0f, 1.0f, 0.5f ) );
            var resampledVolume = floatGridToVdbVolume(resampledGrid);

            resampledVolume.voxelSize.x = vdbVolume.voxelSize.x * 2;
            resampledVolume.voxelSize.y = vdbVolume.voxelSize.y * 1;
            resampledVolume.voxelSize.z = vdbVolume.voxelSize.z * 0.5f;

            Assert.That(resampledVolume.dims.x, Is.EqualTo(13));
            Assert.That(resampledVolume.dims.y, Is.EqualTo(27));
            Assert.That(resampledVolume.dims.z, Is.EqualTo(53));

            Assert.That(resampledVolume.voxelSize.x, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(resampledVolume.voxelSize.y, Is.EqualTo(0.1f).Within(0.001f));
            Assert.That(resampledVolume.voxelSize.z, Is.EqualTo(0.05f).Within(0.001f));

            var tempFile = Path.GetTempFileName() + ".vdb";
            VoxelsSave.toAnySupportedFormat(resampledVolume, tempFile);

            var restored = VoxelsLoad.fromAnySupportedFormat(tempFile);
            Assert.That(restored is not null);
            if (restored is null)
                return;

            Assert.That(restored.size(), Is.EqualTo(1));

            var readVolume = restored.at(0);
            Assert.That(readVolume.voxelSize.x, Is.EqualTo(0.2f).Within(0.001f));
            Assert.That(readVolume.voxelSize.y, Is.EqualTo(0.1f).Within(0.001f));
            Assert.That(readVolume.voxelSize.z, Is.EqualTo(0.05f).Within(0.001f));

            File.Delete(tempFile);
        }

        [Test]
        public void TestCropping()
        {
            var vdbVolume = CreateVolume();
            var box = new Box3i();
            box.min.x = 2;
            box.min.y = 5;
            box.min.z = 1;
            box.max.x = 18;
            box.max.y = 13;
            box.max.z = 23;

            var croppedGrid = cropped(vdbVolume.data, box);
            var croppedVolume = floatGridToVdbVolume(croppedGrid);

            Assert.That(croppedVolume.dims.x, Is.EqualTo(16));
            Assert.That(croppedVolume.dims.y, Is.EqualTo(8));
            Assert.That(croppedVolume.dims.z, Is.EqualTo(22));
        }

        [Test]
        public void TestAccessors()
        {
            var vdbVolume = CreateVolume();
            var p = new Vector3i();
            Assert.That( getValue( vdbVolume.data, p ) == 3.0f );

            var region = new VoxelBitSet( (ulong)( vdbVolume.dims.x * vdbVolume.dims.y * vdbVolume.dims.z ) );
            region.set(new VoxelId(0));
            setValue( vdbVolume.data, region, 1.0f );
            Assert.That( getValue( vdbVolume.data, p ) == 1.0f );

            setValue( vdbVolume.data, p, 2.0f );
            Assert.That( getValue( vdbVolume.data, p ) == 2.0f );
        }
    }
}
