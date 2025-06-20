using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet;
using static MR.DotNet.Vector3f;

namespace MR
{
    public partial class DotNet
    {
        public enum LaplacianRememberShape
        {
            Yes = 0,
            No = 1
        };

        public enum VertexMass{
            /// <summary>
            /// all edges have same weight=1
            /// </summary>
            Unit,

            /// <summary>
            /// vertex mass depends on local geometry and proportional to the area of first-ring triangles
            /// </summary>
            NeiArea
        };

        public class Laplacian
        {
            #region C_FUNCTIONS
            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrLaplacianNew(IntPtr mesh);

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrLaplacianFree(IntPtr laplacian);

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrLaplacianInit(
                IntPtr laplacian,
                IntPtr freeVerts,
                EdgeWeights weights,
                VertexMass vmass,
                LaplacianRememberShape rem);

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrLaplacianFixVertex(
                IntPtr laplacian,
                VertId v,
                ref Vector3f fixedPos,
                bool smooth);

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrLaplacianApply(IntPtr laplacian);

            #endregion

            #region Contructors

            public Laplacian(Mesh mesh)
            {
                reference_ = mrLaplacianNew(mesh.varMesh());
            }

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (needToDispose_)
                {
                    if (reference_ != IntPtr.Zero)
                    {
                        mrLaplacianFree(reference_);
                        reference_ = IntPtr.Zero;
                    }
                }
            }

            #endregion

            #region Methods
            // Laplacian to smoothly deform a region preserving mesh fine details.
            // How to use:
            // 1. Initialize Laplacian for the region being deformed, here region properties are remembered.
            // 2. Change positions of some vertices within the region and call fixVertex for them.
            // 3. Optionally call updateSolver()
            // 4. Call apply() to change the remaining vertices within the region
            // Then steps 1-4 or 2-4 can be repeated.

            /// <summary>
            /// initialize Laplacian for the region being deformed, here region properties are remembered and precomputed;
            /// \param freeVerts must not include all vertices of a mesh connected component
            /// </summary>
            public void Init(
                    VertBitSet freeVerts,
                    EdgeWeights weights = EdgeWeights.Unit,
                    VertexMass vmass = VertexMass.Unit,
                    LaplacianRememberShape rem = LaplacianRememberShape.Yes)
            {
                mrLaplacianInit(reference_, freeVerts.bs_, weights, vmass, rem);
            }

            /// <summary>
            /// // sets position of given vertex after init and it must be fixed during apply (THIS METHOD CHANGES THE MESH);
            /// \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
            /// </summary>
            public void FixVertex(
                    VertId v,
                    ref Vector3f fixedPos,
                    bool smooth = true)
            {
                mrLaplacianFixVertex(reference_, v, ref fixedPos, smooth);
            }

            /// <summary>
            /// given fixed vertices, computes positions of remaining region vertices
            /// </summary>
            public void Apply() => 
                mrLaplacianApply(reference_);

            #endregion

            #region private fields

            internal IntPtr reference_;
            private bool needToDispose_ = true;

            #endregion
        }
    }
}
