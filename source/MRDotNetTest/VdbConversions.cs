using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class VdbConversionsTests
    {
        [Test]
        public void TestConversions()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var settings = new MeshToVolumeSettings();
            settings.voxelSize = Vector3f.Diagonal(0.1f);

            var vdbVolume = MeshToVolume(mesh, settings);
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
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var settings = new MeshToVolumeSettings();
            settings.voxelSize = Vector3f.Diagonal(0.1f);

            var vdbVolume = MeshToVolume(mesh, settings);

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
    }
}