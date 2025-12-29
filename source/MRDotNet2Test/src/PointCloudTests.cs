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
            points.AddPoint(new Vector3f(0, 0, 0));
            points.AddPoint(new Vector3f(0, 1, 0));
            points.AddPoint(new Vector3f(1, 1, 0));
            points.AddPoint(new Vector3f(1, 0, 0));
            points.AddPoint(new Vector3f(0, 0, 1));
            points.AddPoint(new Vector3f(0, 1, 1));
            points.AddPoint(new Vector3f(1, 1, 1));
            points.AddPoint(new Vector3f(1, 0, 1));
            return points;
        }

        [Test]
        public void TestPointCloud()
        {
            var points = MakeCube();

            Assert.That(points.Points.Size() == 8);
            Assert.That(points.Normals.Size() == 0);

            var bbox = points.GetBoundingBox();
            Assert.That(bbox.Min == new Vector3f(0, 0, 0));
            Assert.That(bbox.Max == new Vector3f(1, 1, 1));
        }

        [Test]
        public void TestDisposing()
        {
            Assert.DoesNotThrow(() =>
            {
                var points = MakeCube();
                Assert.That(points.ValidPoints.Count() == 8);
                points.Dispose();
            });
        }

        [Test]
        public void TestPointCloudNormals()
        {
            var points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1));
            points.AddPoint(new Vector3f(0, 1, 0), new Vector3f(0, 0, 1));

            Assert.That(points.Points.Size() == 2);
            Assert.That(points.Points.Size() == 2);
        }

        [Test]
        public void TestSaveLoad()
        {
            var points = MakeCube();
            var tempFile = Path.GetTempFileName() + ".ply";
            PointsSave.ToAnySupportedFormat(points, tempFile);

            var readPoints = PointsLoad.FromAnySupportedFormat(tempFile);
            Assert.That(points.Points.Size() == readPoints.Points.Size());
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".ply";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<Misc.UnexpectedResultException>(() => PointsLoad.FromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestTriangulation()
        {
            var mesh = MakeTorus(2.0f, 1.0f, 32, 32);
            var pc = MeshToPointCloud(mesh);
            var restored = TriangulatePointCloud(pc, new TriangulationParameters()).Value();
            Assert.That(restored is not null);
            if (restored is not null)
            {
                Assert.That(restored.Points.Size(), Is.EqualTo(1024));
                Assert.That(restored.Topology.GetValidVerts().Count(), Is.EqualTo(1024));
                Assert.That(restored.Topology.FindHoleRepresentiveEdges().Size() == 0);
            }
        }

        [Test]
        public void TestSaveLoadWithColors()
        {
            var points = MakeCube();
            var colors = new VertColors();
            colors.PushBack(new Color(1.0f, 0.0f, 0.0f));
            colors.PushBack(new Color(0.0f, 1.0f, 0.0f));
            colors.PushBack(new Color(0.0f, 0.0f, 1.0f));
            colors.PushBack(new Color(1.0f, 1.0f, 0.0f));
            colors.PushBack(new Color(1.0f, 0.0f, 1.0f));
            colors.PushBack(new Color(0.0f, 1.0f, 1.0f));
            colors.PushBack(new Color(1.0f, 1.0f, 1.0f));
            colors.PushBack(new Color(0.0f, 0.0f, 0.0f));

            var saveSettings = new SaveSettings();
            saveSettings.Colors = colors;
            
            string path = Path.GetTempFileName() + ".ply";
            PointsSave.ToAnySupportedFormat(points, path, saveSettings);

            var loadSettings = new PointsLoadSettings();
            loadSettings.Colors = new VertColors();
            var readPoints = PointsLoad.FromAnySupportedFormat(path, loadSettings);
            Assert.That(points.Points.Size() == 8);

            var readColors = loadSettings.Colors;
            Assert.That(colors.Size() == readColors.Size());
            for (ulong i = 0; i < colors.Size(); i++)
            {
                Assert.That(colors.Index(new VertId(i)) == readColors.Index(new VertId(i)));
            }

            File.Delete(path);
        }

        [Test]
        public void TestCachedPoints()
        {
            var points = MakeCube();
            Assert.That(points.Points.Size() == 8);
            points.AddPoint(new Vector3f(0, 0, 0));
            Assert.That(points.Points.Size() == 9);
        }
    }
}