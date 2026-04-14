using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class PoitCloudTests
    {
        static PointCloud MakeCube()
        {
            var points = new PointCloud();
            points.addPoint(new Vector3f(0, 0, 0));
            points.addPoint(new Vector3f(0, 1, 0));
            points.addPoint(new Vector3f(1, 1, 0));
            points.addPoint(new Vector3f(1, 0, 0));
            points.addPoint(new Vector3f(0, 0, 1));
            points.addPoint(new Vector3f(0, 1, 1));
            points.addPoint(new Vector3f(1, 1, 1));
            points.addPoint(new Vector3f(1, 0, 1));
            return points;
        }

        [Test]
        public void TestPointCloud()
        {
            var points = MakeCube();

            Assert.That(points.points.size() == 8);
            Assert.That(points.normals.size() == 0);

            var bbox = points.getBoundingBox();
            Assert.That(bbox.min == new Vector3f(0, 0, 0));
            Assert.That(bbox.max == new Vector3f(1, 1, 1));
        }

        [Test]
        public void TestDisposing()
        {
            Assert.DoesNotThrow(() =>
            {
                var points = MakeCube();
                Assert.That(points.validPoints.count() == 8);
                points.Dispose();
            });
        }

        [Test]
        public void TestPointCloudNormals()
        {
            var points = new PointCloud();
            points.addPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1));
            points.addPoint(new Vector3f(0, 1, 0), new Vector3f(0, 0, 1));

            Assert.That(points.points.size() == 2);
            Assert.That(points.points.size() == 2);
        }

        [Test]
        public void TestSaveLoad()
        {
            var points = MakeCube();
            var tempFile = Path.GetTempFileName() + ".ply";
            PointsSave.toAnySupportedFormat(points, tempFile);

            var readPoints = PointsLoad.fromAnySupportedFormat(tempFile);
            Assert.That(points.points.size() == readPoints.points.size());
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".ply";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<Misc.UnexpectedResultException>(() => PointsLoad.fromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestTriangulation()
        {
            var mesh = makeTorus(2.0f, 1.0f, 32, 32);
            var pc = meshToPointCloud(mesh);
            var restored = triangulatePointCloud(pc, new TriangulationParameters()).value();
            Assert.That(restored is not null);
            if (restored is not null)
            {
                Assert.That(restored.points.size(), Is.EqualTo(1024));
                Assert.That(restored.topology.getValidVerts().count(), Is.EqualTo(1024));
                Assert.That(restored.topology.findHoleRepresentiveEdges().size() == 0);
            }
        }

        [Test]
        public void TestSaveLoadWithColors()
        {
            var points = MakeCube();
            var colors = new VertColors();
            colors.pushBack(new Color(1.0f, 0.0f, 0.0f));
            colors.pushBack(new Color(0.0f, 1.0f, 0.0f));
            colors.pushBack(new Color(0.0f, 0.0f, 1.0f));
            colors.pushBack(new Color(1.0f, 1.0f, 0.0f));
            colors.pushBack(new Color(1.0f, 0.0f, 1.0f));
            colors.pushBack(new Color(0.0f, 1.0f, 1.0f));
            colors.pushBack(new Color(1.0f, 1.0f, 1.0f));
            colors.pushBack(new Color(0.0f, 0.0f, 0.0f));

            var saveSettings = new SaveSettings();
            saveSettings.colors = colors;

            string path = Path.GetTempFileName() + ".ply";
            PointsSave.toAnySupportedFormat(points, path, saveSettings);

            var loadSettings = new PointsLoadSettings();
            loadSettings.colors = new VertColors();
            var readPoints = PointsLoad.fromAnySupportedFormat(path, loadSettings);
            Assert.That(points.points.size() == 8);

            var readColors = loadSettings.colors;
            Assert.That(colors.size() == readColors.size());
            for (ulong i = 0; i < colors.size(); i++)
            {
                Assert.That(colors[new VertId(i)] == readColors[new VertId(i)]);
            }

            File.Delete(path);
        }

        [Test]
        public void TestCachedPoints()
        {
            var points = MakeCube();
            Assert.That(points.points.size() == 8);
            points.addPoint(new Vector3f(0, 0, 0));
            Assert.That(points.points.size() == 9);
        }
    }
}
