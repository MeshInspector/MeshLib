using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR
{
    public partial class DotNet
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct VertId
        {
            public int Id;
            public VertId(int id = -1) { Id = id; }
            public bool Valid() => Id >= 0;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct FaceId
        {
            public int Id;
            public FaceId(int id = -1) { Id = id; }
            public bool Valid() => Id >= 0;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct EdgeId
        {
            public int Id;
            public EdgeId(int id = -1) { Id = id; }
            public bool Valid() => Id >= 0;
        }


        [StructLayout(LayoutKind.Sequential)]
        public struct UndirectedEdgeId
        {
            public int Id;
            public UndirectedEdgeId(int id = -1) { Id = id; }
            public bool Valid() => Id >= 0;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct RegionId
        {
            public int Id;
            public RegionId(int id = -1) { Id = id; }
            public bool Valid() => Id >= 0;
        }
    }
}
