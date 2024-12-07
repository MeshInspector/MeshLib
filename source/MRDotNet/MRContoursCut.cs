using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using static MR.DotNet;

namespace MR
{
    using OneMeshContours = List<OneMeshContour>;
    using static MR.DotNet.CoordinateConverters;
    using static MR.DotNet.Vector3f;

    public partial class DotNet
    {
        /// represents primitive type
        public enum VariantIndex
        {
            Face,
            Edge,
            Vertex
        };
        /// simple point on mesh, represented by primitive id and coordinate in mesh space
        public struct OneMeshIntersection
        {
            public VariantIndex variantIndex;
            public int index;
            public Vector3f coordinate;

            public OneMeshIntersection(VariantIndex variantIndex, int index, Vector3f coordinate)
            {
                this.variantIndex = variantIndex;
                this.index = index;
                this.coordinate = coordinate;
            }
        };
        /// one contour on mesh
        public struct OneMeshContour
        {
            public List<OneMeshIntersection> intersections;
            public bool closed;

            public OneMeshContour(List<OneMeshIntersection> intersections, bool closed)
            {
                this.intersections = intersections;
                this.closed = closed;
            }
        };


        [StructLayout(LayoutKind.Sequential)]
        internal struct MROneMeshIntersection
        {
            public int primitiveId = 0;
            public byte primitiveIdIndex = 0;
            public MRVector3f coordinate = new MRVector3f();
            public MROneMeshIntersection() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVectorOneMeshIntersection
        {
            public IntPtr data = IntPtr.Zero;
            public ulong size = 0;
            public IntPtr reserved = IntPtr.Zero;
            public MRVectorOneMeshIntersection() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MROneMeshContour
        {
            public MRVectorOneMeshIntersection intersections = new MRVectorOneMeshIntersection();
            //size of bool in C is 1, so use byte
            public byte closed = 0;
            public MROneMeshContour() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVariableEdgeTri
        {
            public EdgeId edge = new EdgeId();
            public FaceId tri = new FaceId();
            public bool isEdgeATriB = false;
            public MRVariableEdgeTri() { }
        };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MROneMeshContour mrOneMeshContoursGet(IntPtr contours, ulong index);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrOneMeshContoursSize(IntPtr contours);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrOneMeshContoursFree(IntPtr contours);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrGetOneMeshIntersectionContours(IntPtr meshA, IntPtr meshB,
                                                                     IntPtr continousContours,
                                                                     bool getMeshAIntersections,
                                                                     ref MRCoordinateConverters converters,
                                                                     IntPtr rigidB2A);
        /// converts ordered continuous contours of two meshes to OneMeshContours
        /// converters is required for better precision in case of degenerations
        /// note that contours should not have intersections
        static public OneMeshContours GetOneMeshIntersectionContours(Mesh meshA, Mesh meshB, ContinousContours contours, bool getMeshAIntersections,
            CoordinateConverters converters, AffineXf3f? rigidB2A = null)
        {
            var mrOneMeshContours = mrGetOneMeshIntersectionContours(meshA.mesh_, meshB.mesh_, contours.mrContours_, getMeshAIntersections, ref converters.conv_, rigidB2A is null ? IntPtr.Zero : rigidB2A.XfAddr());
            int contoursSize = (int)mrOneMeshContoursSize(mrOneMeshContours);
            var oneMeshContours = new OneMeshContours(contoursSize);
            for (int i = 0; i < contoursSize; i++)
            {
                var mrOneMeshContour = mrOneMeshContoursGet(mrOneMeshContours, (ulong)i);
                var oneMeshContour = new OneMeshContour
                {
                    intersections = new List<OneMeshIntersection>((int)mrOneMeshContour.intersections.size),
                    closed = mrOneMeshContour.closed > 0
                };

                for (int j = 0; j < (int)mrOneMeshContour.intersections.size; j++)
                {
                    var mrOneMeshIntersectionData = mrOneMeshContour.intersections.data;
                    int sizeOfOneMeshIntersection = Marshal.SizeOf(typeof(MROneMeshIntersection));
                    var mrOneMeshIntersection = (MROneMeshIntersection)Marshal.PtrToStructure(IntPtr.Add(mrOneMeshIntersectionData, j * sizeOfOneMeshIntersection), typeof(MROneMeshIntersection));
                    var oneMeshIntersection = new OneMeshIntersection();
                    oneMeshIntersection.variantIndex = (VariantIndex)mrOneMeshIntersection.primitiveIdIndex;
                    oneMeshIntersection.index = mrOneMeshIntersection.primitiveId;
                    oneMeshIntersection.coordinate = new Vector3f(mrOneMeshIntersection.coordinate);
                    oneMeshContour.intersections.Add(new OneMeshIntersection
                    {
                        variantIndex = (VariantIndex)mrOneMeshIntersection.primitiveIdIndex,
                        index = mrOneMeshIntersection.primitiveId,
                        coordinate = new Vector3f(mrOneMeshIntersection.coordinate)
                    });
                }
                oneMeshContours.Add(oneMeshContour);
            }
            return oneMeshContours;
        }
    }
}
