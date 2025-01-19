using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
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

            Assert.That(points.Points.Count == 8);
            Assert.That(points.Normals.Count == 0);

            var bbox = points.BoundingBox;
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

            Assert.That(points.Points.Count == 2);
            Assert.That(points.Points.Count == 2);
        }

        [Test]
        public void TestNormalsError()
        {
            var points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1));
            Assert.Throws<InvalidOperationException>(() => points.AddPoint(new Vector3f(0, 0, 0)));

            points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0));
            Assert.Throws<InvalidOperationException>(() => points.AddPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1)));
        }

        [Test]
        public void TestSaveLoad()
        {
            var points = MakeCube();
            var tempFile = Path.GetTempFileName() + ".ply";
            PointsSave.ToAnySupportedFormat(points, tempFile);

            var readPoints = PointsLoad.FromAnySupportedFormat(tempFile);
            Assert.That(points.Points.Count == readPoints.Points.Count);
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".ply";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<SystemException>(() => PointsLoad.FromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestTriangulation()
        {
            var mesh = Mesh.MakeTorus(2.0f, 1.0f, 32, 32);
            var pc = Mesh.MeshToPointCloud(mesh);
            var restored = TriangulatePointCloud(pc, new TriangulationParameters());
            Assert.That(restored is not null);
            if (restored is not null)
            {
                Assert.That(restored.Points.Count, Is.EqualTo(1024));
                Assert.That(restored.ValidPoints.Count(), Is.EqualTo(1024));
                Assert.That(restored.HoleRepresentiveEdges.Count == 0);
            }
        }

        [Test]
        public void TestCreatingFromPointList()
        {
            var points = new List<Vector3f>(8);
            points.Add(new Vector3f(0, 0, 0));
            points.Add(new Vector3f(0, 1, 0));
            points.Add(new Vector3f(1, 1, 0));
            points.Add(new Vector3f(1, 0, 0));
            points.Add(new Vector3f(0, 0, 1));
            points.Add(new Vector3f(0, 1, 1));
            points.Add(new Vector3f(1, 1, 1));
            points.Add(new Vector3f(1, 0, 1));

            var pc = PointCloud.FromPoints(points);
            Assert.That(pc.Points.Count == 8);
            Assert.That(pc.Normals.Count == 0);
        }

        [Test]
        public void TestSaveLoadWithColors()
        {
            var points = MakeCube();
            var colors = new List<Color>(8);
            colors.Add(new Color(1.0f, 0.0f, 0.0f));
            colors.Add(new Color(0.0f, 1.0f, 0.0f));
            colors.Add(new Color(0.0f, 0.0f, 1.0f));
            colors.Add(new Color(1.0f, 1.0f, 0.0f));
            colors.Add(new Color(1.0f, 0.0f, 1.0f));
            colors.Add(new Color(0.0f, 1.0f, 1.0f));
            colors.Add(new Color(1.0f, 1.0f, 1.0f));
            colors.Add(new Color(0.0f, 0.0f, 0.0f));

            var saveSettings = new SaveSettings();
            saveSettings.colors = new VertColors(colors);
            
            string path = Path.GetTempFileName() + ".ply";
            PointsSave.ToAnySupportedFormat(points, path, saveSettings);

            var loadSettings = new PointsLoadSettings();
            loadSettings.colors = new VertColors();
            var readPoints = PointsLoad.FromAnySupportedFormat(path, loadSettings);
            Assert.That(points.Points.Count == 8);

            var readColors = loadSettings.colors.ToList();
            Assert.That(colors.Count == readColors.Count);
            for (int i = 0; i < colors.Count; i++)
            {
                Assert.That(colors[i] == readColors[i]);
            }

            File.Delete(path);
        }

        [Test]
        public void TestCachedPoints()
        {
            var points = MakeCube();
            Assert.That(points.Points.Count == 8);
            points.AddPoint(new Vector3f(0, 0, 0));
            Assert.That(points.Points.Count == 9);
        }
    }
}