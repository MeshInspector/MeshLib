using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace MR
{
    public partial class DotNet
    {
        public class MeshBuilder
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            unsafe private static extern MRVertMap* mrVertMapNew();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            unsafe private static extern void mrVertMapFree(MRVertMap* vertMap);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            unsafe private static extern int mrMeshBuilderUniteCloseVertices(IntPtr mesh, float closeDist, bool uniteOnlyBd, MRVertMap* optionalVertOldToNew);

            /// the function finds groups of mesh vertices located closer to each other than \param closeDist, and unites such vertices in one;
            /// then the mesh is rebuilt from the remaining triangles
            /// \param mesh is passed by ref. The function changes it
            /// \param optionalVertOldToNew is the mapping of vertices: before -> after
            /// \param uniteOnlyBd if true then only boundary vertices can be united, all internal vertices (even close ones) will remain
            /// \return the number of vertices united, 0 means no change in the mesh
            unsafe public static int UniteCloseVertices(ref Mesh mesh, float closeDist, bool uniteOnlyBd, List<VertId>? optionalVertOld2New = null)
            {
                if (optionalVertOld2New == null)
                    return mrMeshBuilderUniteCloseVertices(mesh.varMesh(), closeDist, uniteOnlyBd, null);

                MRVertMap* vertMap = mrVertMapNew();
                var res = mrMeshBuilderUniteCloseVertices(mesh.varMesh(), closeDist, uniteOnlyBd, vertMap);

                optionalVertOld2New.Clear();

                for (int i = 0; i < (int)vertMap->size; i++)
                {
                    var vertId = Marshal.ReadInt32(IntPtr.Add(vertMap->data, i * sizeof(int)));
                    optionalVertOld2New.Add(new VertId(vertId));
                }

                mrVertMapFree(vertMap);
                return res;
            }
        }
    }
}
