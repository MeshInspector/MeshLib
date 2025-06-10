using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using static MR.DotNet;

namespace MR
{
    using MRContinuousContour = MRVectorVarEdgeTri;

    using ContinousContour = List<VarEdgeTri>;
    using PreciseCollisionResult = VectorVarEdgeTri;

    public partial class DotNet
    {
        public class ContinousContours : IDisposable
        {
            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern MRContinuousContour mrContinuousContoursGet(IntPtr contours, ulong index);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern ulong mrContinuousContoursSize(IntPtr contours);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
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
                    if (contours_ is null)
                    {
                        int contoursSize = (int)mrContinuousContoursSize(mrContours_);
                        contours_ = new List<ContinousContour>();
                        for (int i = 0; i < contoursSize; i++)
                        {
                            var mrContour = mrContinuousContoursGet(mrContours_, (ulong)i);
                            var contour = new VectorVarEdgeTri(mrContour);
                            contours_.Add(new ContinousContour(contour.List));
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
            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern IntPtr mrOrderIntersectionContours(IntPtr topologyA, IntPtr topologyB, IntPtr intersections);
            /// combines individual intersections into ordered contours with the properties:
            /// a. left  of contours on mesh A is inside of mesh B,
            /// b. right of contours on mesh B is inside of mesh A,
            /// c. each intersected edge has origin inside meshes intersection and destination outside of it
            public static ContinousContours OrderIntersectionContours(Mesh meshA, Mesh meshB, PreciseCollisionResult intersections)
            {
                var mrContours = mrOrderIntersectionContours(meshA.meshTopology_, meshB.meshTopology_, intersections.nativeVector_);
                return new ContinousContours(mrContours);
            }
        }
    }
}
