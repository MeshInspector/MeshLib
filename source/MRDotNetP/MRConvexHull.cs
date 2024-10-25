using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    public class ConvexHull
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMakeConvexHullFromMesh( IntPtr mesh );

        // computes the mesh of convex hull from given point cloud
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMakeConvexHullFromPointCloud( IntPtr pointCloud );

        // computes the mesh of convex hull from given mesh
        public static Mesh MakeConvexHull(Mesh mesh )
        {
            return new Mesh( mrMakeConvexHullFromMesh( mesh.mesh_ ) );
        }

        // computes the mesh of convex hull from given point cloud
        public static Mesh MakeConvexHull(PointCloud pointCloud )
        {
            return new Mesh( mrMakeConvexHullFromPointCloud( pointCloud.pc_ ) );
        }
    }
}
