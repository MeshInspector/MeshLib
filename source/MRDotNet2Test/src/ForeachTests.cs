using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class ForeachTests
    {
        [Test]
        public void TestStdVector()
        {
            var edges = new Std.Vector_MREdgeId();
            edges.pushBack(new EdgeId(3));
            edges.pushBack(new EdgeId(7));
            var ids = new List<int>();
            foreach (var e in edges)
                ids.Add(e.Value.id);
            Assert.That(ids, Is.EqualTo(new[] { 3, 7 }));

            int n = 0;
            foreach (var e in new Std.Vector_MREdgeId())
                ++n;
            Assert.That(n, Is.EqualTo(0));
        }

        [Test]
        public void TestMRVector()
        {
            var points = new VertCoords();
            points.pushBack(new Vector3f(1, 2, 3));
            points.pushBack(new Vector3f(4, 5, 6));
            float sum = 0;
            foreach (var p in points)
                sum += p.Value.x;
            Assert.That(sum, Is.EqualTo(5));
        }

        [Test]
        public void TestBitSet()
        {
            var bs = new FaceBitSet(10);
            bs.set(new FaceId(2));
            bs.set(new FaceId(8));
            var ids = new List<int>();
            foreach (var f in bs)
                ids.Add(f.id);
            Assert.That(ids, Is.EqualTo(new[] { 2, 8 }));
        }
    }
}
