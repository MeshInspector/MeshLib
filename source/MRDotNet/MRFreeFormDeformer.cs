using System;
using static MR.DotNet.Box3f;
using static MR.DotNet.Vector3f;
using static MR.DotNet.Vector3i;
using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {
        public class FreeFormDeformer : IDisposable
        {
            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern IntPtr mrFreeFormDeformerNewFromMesh(IntPtr mesh, IntPtr region);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern void mrFreeFormDeformerFree(IntPtr deformer);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern void mrFreeFormDeformerInit(IntPtr deformer, ref MRVector3i resolution, ref MRBox3f initialBox);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern void mrFreeFormDeformerSetRefGridPointPosition(IntPtr deformer, ref MRVector3i coordOfPointInGrid, ref MRVector3f newPos);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrFreeFormDeformerGetRefGridPointPosition(IntPtr deformer, ref MRVector3i coordOfPointInGrid);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern void mrFreeFormDeformerApply(IntPtr deformer);

            internal IntPtr deformer_;

            public FreeFormDeformer(Mesh mesh, VertBitSet? region = null)
            {
                deformer_ = mrFreeFormDeformerNewFromMesh(mesh.mesh_, region is null ? (IntPtr)null : region.bs_);
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
                    if (deformer_ != IntPtr.Zero)
                    {
                        mrFreeFormDeformerFree(deformer_);
                    }
                    disposed = true;
                }
            }
            ~FreeFormDeformer()
            {
                mrFreeFormDeformerFree(deformer_);
            }

            /// Calculates all points' normalized positions in parallel
            /// sets ref grid by initialBox, if initialBox is invalid uses mesh bounding box instead
            public void Init(Vector3i resolution, Box3f initialBox)
            {
                mrFreeFormDeformerInit(deformer_, ref resolution.vec_, ref initialBox.boxRef());
            }

            /// Updates ref grid point position
            public void SetRefGridPointPosition(Vector3i coordOfPointInGrid, Vector3f newPos)
            {
                mrFreeFormDeformerSetRefGridPointPosition(deformer_, ref coordOfPointInGrid.vec_, ref newPos.vec_);
            }

            /// Gets ref grid point position
            public Vector3f mrFreeFormDeformerGetRefGridPointPosition(IntPtr deformer, Vector3i coordOfPointInGrid)
            {
                return new Vector3f(mrFreeFormDeformerGetRefGridPointPosition(deformer_, ref coordOfPointInGrid.vec_));
            }

            /// Apply updated grid to all mesh points in parallel
            /// ensure updating render object after using it
            public void Apply()
            {
                mrFreeFormDeformerApply(deformer_);
            }
        }
    }
}
