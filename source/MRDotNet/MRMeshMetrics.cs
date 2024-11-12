using MR.DotNet;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    public class FillHoleMetric
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrFillHoleMetricFree(IntPtr metric);

        /// Computes combined metric after filling a hole
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern double mrCalcCombinedFillMetric( IntPtr mesh, IntPtr filledRegion, IntPtr metric );

        /// This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.
        /// It is rather fast to calculate, and it results in typically good triangulations.
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetCircumscribedMetric( IntPtr mesh );

        /// Same as mrGetCircumscribedFillMetric, but with extra penalty for the triangles having
        /// normals looking in the opposite side of plane containing left of (e).
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetPlaneFillMetric( IntPtr mesh, EdgeId e );

        /// Similar to mrGetPlaneFillMetric with extra penalty for the triangles having
        /// normals looking in the opposite side of plane containing left of (e),
        /// but the metric minimizes the sum of circumcircle radius times aspect ratio for all triangles in the triangulation.
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetPlaneNormalizedFillMetric( IntPtr mesh, EdgeId e );

        /// This metric minimizes the sum of triangleMetric for all triangles in the triangulation
        /// plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n
        /// Where\n
        /// triangleMetric is proportional to weighted triangle area and triangle aspect ratio\n
        /// edgeMetric grows with angle between triangles as ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4.
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetComplexFillMetric( IntPtr mesh, EdgeId e );

        /// This metric minimizes the maximal dihedral angle between the faces in the triangulation
        /// and on its boundary, and it avoids creating too degenerate triangles;
        ///  for planar holes it is the same as getCircumscribedMetric
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetUniversalMetric( IntPtr mesh );

        /// This metric is for triangulation construction with minimal summed area of triangles.
        /// Warning: this metric can produce degenerated triangles
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetMinAreaMetric( IntPtr mesh );

        private FillHoleMetric(IntPtr metric)
        {
            mrMetric_ = metric;
        }

        public FillHoleMetric() { mrMetric_ = IntPtr.Zero; }

        public static FillHoleMetric GetCircumscribedMetric(Mesh mesh)
        {
            return new FillHoleMetric(mrGetCircumscribedMetric(mesh.mesh_));
        }

        public static FillHoleMetric GetPlaneFillMetric(Mesh mesh, EdgeId e)
        {
            return new FillHoleMetric(mrGetPlaneFillMetric(mesh.mesh_, e));
        }

        public static FillHoleMetric GetPlaneNormalizedFillMetric(Mesh mesh, EdgeId e)
        {
            return new FillHoleMetric(mrGetPlaneNormalizedFillMetric(mesh.mesh_, e));
        }

        public static FillHoleMetric GetComplexFillMetric(Mesh mesh, EdgeId e)
        {
            return new FillHoleMetric(mrGetComplexFillMetric(mesh.mesh_, e));
        }

        public static FillHoleMetric GetUniversalMetric(Mesh mesh)
        {
            return new FillHoleMetric(mrGetUniversalMetric(mesh.mesh_));
        }

        public static FillHoleMetric GetMinAreaMetric(Mesh mesh)
        {
            return new FillHoleMetric(mrGetMinAreaMetric(mesh.mesh_));
        }

        ~FillHoleMetric()
        {
            mrFillHoleMetricFree(mrMetric_);
        }

        public double CalcCombinedFillMetric(Mesh mesh, BitSet filledRegion)
        {
            return mrCalcCombinedFillMetric(mesh.mesh_, filledRegion.bs_, mrMetric_);
        }

        internal IntPtr mrMetric_;
    }
}
