using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet.MeshComponents;

namespace MR.DotNet
{   
    public class MeshBuilder
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        unsafe private static extern MRVertMap* mrVertMapNew();

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        unsafe private static extern void mrVertMapFree(MRVertMap* vertMap);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        unsafe private static extern int mrMeshBuilderUniteCloseVertices(IntPtr mesh, float closeDist, bool uniteOnlyBd, MRVertMap* optionalVertOldToNew);

        unsafe public static int UniteCloseVertices( Mesh mesh, float closeDist, bool uniteOnlyBd, List<VertId>? optionalVertOld2New = null )
        {            
            if ( optionalVertOld2New == null )
                return mrMeshBuilderUniteCloseVertices(mesh.mesh_, closeDist, uniteOnlyBd, null);

            MRVertMap* vertMap = mrVertMapNew();
            var res = mrMeshBuilderUniteCloseVertices(mesh.mesh_, closeDist, uniteOnlyBd, vertMap);
            optionalVertOld2New.Clear();

            for (int i = 0; i < (int)vertMap->size; i++)
            {
                var vertId =  Marshal.ReadInt32(IntPtr.Add(vertMap->data, i * sizeof(int)));
                optionalVertOld2New.Add(new VertId(vertId));
            }

            mrVertMapFree(vertMap);
            return res;         
        }
    }
}
