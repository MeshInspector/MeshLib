using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    using EdgeId = int;
    using FaceId = int;
    using ContinousContour = List<VariableEdgeTri>;
    using static MR.DotNet.ContoursCut;
    using static MR.DotNet.ContinousContours;

    [StructLayout(LayoutKind.Sequential)]
    public struct VariableEdgeTri
    {
        public EdgeId edge;
        public FaceId tri;
        public bool isEdgeATriB;
    };

    public class ContinousContours : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRContinuousContour
        {
            public IntPtr data;
            public ulong size;
            public IntPtr reserved;
        }

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRContinuousContour mrContinuousContoursGet(IntPtr contours, ulong index);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrContinuousContoursSize(IntPtr contours);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrContinuousContoursFree(IntPtr contours);

        internal ContinousContours(IntPtr mrContours)
        {
            mrContours_ = mrContours;            
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mrContours_ != IntPtr.Zero)
                {
                    mrContinuousContoursFree(mrContours_);
                }

                disposed = true;
            }
        }

        ~ContinousContours()
        {
            Dispose(false);
        }

        public ReadOnlyCollection<ContinousContour> Contours 
        { 
            get 
            {
                if ( contours_ is null )
                {
                    int contoursSize = (int)mrContinuousContoursSize(mrContours_);
                    contours_ = new List<ContinousContour>();
                    for (int i = 0; i < contoursSize; i++)
                    {
                        var mrContour = mrContinuousContoursGet(mrContours_, (ulong)i);

                        var contour = new ContinousContour();
                        int sizeOfVariableEdgeTri = Marshal.SizeOf(typeof(VariableEdgeTri));
                        for (int j = 0; j < (int)mrContour.size; j++)
                        {
                            var vetPtr = IntPtr.Add(mrContour.data, j * sizeOfVariableEdgeTri);
                            var vet = Marshal.PtrToStructure<VariableEdgeTri>(vetPtr);
                            contour.Add(vet);
                        }
                        contours_.Add(contour);
                    }
                }

                return contours_.AsReadOnly();
            }            
        }

        private List<ContinousContour>? contours_;
        
        internal IntPtr mrContours_;
    }
    public class IntersectionContour
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrOrderIntersectionContours( IntPtr topologyA, IntPtr topologyB, IntPtr intersections );
        /// combines individual intersections into ordered contours with the properties:
        /// a. left  of contours on mesh A is inside of mesh B,
        /// b. right of contours on mesh B is inside of mesh A,
        /// c. each intersected edge has origin inside meshes intersection and destination outside of it
        public static ContinousContours OrderIntersectionContours(Mesh meshA, Mesh meshB, PreciseCollisionResult intersections )
        {
            var mrContours = mrOrderIntersectionContours( meshA.meshTopology_, meshB.meshTopology_, intersections.nativeResult_ );
            return new ContinousContours(mrContours);
        }
    }
}
