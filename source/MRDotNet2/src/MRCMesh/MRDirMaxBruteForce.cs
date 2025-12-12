public static partial class MR
{
    /// finds the point having the largest projection on given direction by traversing all region points
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector3f dir, MR.Const_VertCoords points, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_3_const_MR_Vector3f_ref_MR_VertCoords", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_3_const_MR_Vector3f_ref_MR_VertCoords(MR.Const_Vector3f._Underlying *dir, MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *region);
        return __MR_findDirMaxBruteForce_3_const_MR_Vector3f_ref_MR_VertCoords(dir._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }

    /// finds the point having the largest projection on given direction by traversing all region points
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector2f dir, MR.Const_VertCoords2 points, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_3_const_MR_Vector2f_ref", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_3_const_MR_Vector2f_ref(MR.Const_Vector2f._Underlying *dir, MR.Const_VertCoords2._Underlying *points, MR.Const_VertBitSet._Underlying *region);
        return __MR_findDirMaxBruteForce_3_const_MR_Vector2f_ref(dir._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }

    /// finds the point in the cloud having the largest projection on given direction by traversing all valid points
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector3f dir, MR.Const_PointCloud cloud, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_3_const_MR_Vector3f_ref_MR_PointCloud", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_3_const_MR_Vector3f_ref_MR_PointCloud(MR.Const_Vector3f._Underlying *dir, MR.Const_PointCloud._Underlying *cloud, MR.Const_VertBitSet._Underlying *region);
        return __MR_findDirMaxBruteForce_3_const_MR_Vector3f_ref_MR_PointCloud(dir._UnderlyingPtr, cloud._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }

    /// finds the vertex in the polyline having the largest projection on given direction by traversing all valid vertices
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector3f dir, MR.Const_Polyline3 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_Polyline3", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_Polyline3(MR.Const_Vector3f._Underlying *dir, MR.Const_Polyline3._Underlying *polyline);
        return __MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_Polyline3(dir._UnderlyingPtr, polyline._UnderlyingPtr);
    }

    /// finds the vertex in the polyline having the largest projection on given direction by traversing all valid vertices
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector2f dir, MR.Const_Polyline2 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_2_const_MR_Vector2f_ref", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_2_const_MR_Vector2f_ref(MR.Const_Vector2f._Underlying *dir, MR.Const_Polyline2._Underlying *polyline);
        return __MR_findDirMaxBruteForce_2_const_MR_Vector2f_ref(dir._UnderlyingPtr, polyline._UnderlyingPtr);
    }

    /// finds the vertex in the mesh part having the largest projection on given direction by traversing all (region) faces
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector3f dir, MR.Const_MeshPart mp)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshPart", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshPart(MR.Const_Vector3f._Underlying *dir, MR.Const_MeshPart._Underlying *mp);
        return __MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshPart(dir._UnderlyingPtr, mp._UnderlyingPtr);
    }

    /// finds the vertex in the mesh part having the largest projection on given direction by traversing all (region) vertices
    /// Generated from function `MR::findDirMaxBruteForce`.
    public static unsafe MR.VertId FindDirMaxBruteForce(MR.Const_Vector3f dir, MR.Const_MeshVertPart mp)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshVertPart", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshVertPart(MR.Const_Vector3f._Underlying *dir, MR.Const_MeshVertPart._Underlying *mp);
        return __MR_findDirMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshVertPart(dir._UnderlyingPtr, mp._UnderlyingPtr);
    }

    /// finds the points having the smallest and the largest projections on given direction by traversing all region points
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector3f dir, MR.Const_VertCoords points, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_3_const_MR_Vector3f_ref_MR_VertCoords", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_3_const_MR_Vector3f_ref_MR_VertCoords(MR.Const_Vector3f._Underlying *dir, MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *region);
        return new(__MR_findDirMinMaxBruteForce_3_const_MR_Vector3f_ref_MR_VertCoords(dir._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true);
    }

    /// finds the points having the smallest and the largest projections on given direction by traversing all region points
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector2f dir, MR.Const_VertCoords2 points, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_3_const_MR_Vector2f_ref", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_3_const_MR_Vector2f_ref(MR.Const_Vector2f._Underlying *dir, MR.Const_VertCoords2._Underlying *points, MR.Const_VertBitSet._Underlying *region);
        return new(__MR_findDirMinMaxBruteForce_3_const_MR_Vector2f_ref(dir._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true);
    }

    /// finds the points in the cloud having the smallest and the largest projections on given direction by traversing all valid points
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector3f dir, MR.Const_PointCloud cloud, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_3_const_MR_Vector3f_ref_MR_PointCloud", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_3_const_MR_Vector3f_ref_MR_PointCloud(MR.Const_Vector3f._Underlying *dir, MR.Const_PointCloud._Underlying *cloud, MR.Const_VertBitSet._Underlying *region);
        return new(__MR_findDirMinMaxBruteForce_3_const_MR_Vector3f_ref_MR_PointCloud(dir._UnderlyingPtr, cloud._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true);
    }

    /// finds the vertex in the polyline having the smallest and the largest projections on given direction by traversing all valid vertices
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector3f dir, MR.Const_Polyline3 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_Polyline3", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_Polyline3(MR.Const_Vector3f._Underlying *dir, MR.Const_Polyline3._Underlying *polyline);
        return new(__MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_Polyline3(dir._UnderlyingPtr, polyline._UnderlyingPtr), is_owning: true);
    }

    /// finds the vertex in the polyline having the smallest and the largest projections on given direction by traversing all valid vertices
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector2f dir, MR.Const_Polyline2 polyline)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_2_const_MR_Vector2f_ref", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_2_const_MR_Vector2f_ref(MR.Const_Vector2f._Underlying *dir, MR.Const_Polyline2._Underlying *polyline);
        return new(__MR_findDirMinMaxBruteForce_2_const_MR_Vector2f_ref(dir._UnderlyingPtr, polyline._UnderlyingPtr), is_owning: true);
    }

    /// finds the vertices in the mesh part having the smallest and the largest projections on given direction by traversing all (region) faces
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector3f dir, MR.Const_MeshPart mp)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshPart", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshPart(MR.Const_Vector3f._Underlying *dir, MR.Const_MeshPart._Underlying *mp);
        return new(__MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshPart(dir._UnderlyingPtr, mp._UnderlyingPtr), is_owning: true);
    }

    /// finds the vertices in the mesh part having the smallest and the largest projections on given direction by traversing all (region) vertices
    /// Generated from function `MR::findDirMinMaxBruteForce`.
    public static unsafe MR.MinMaxArg_Float_MRVertId FindDirMinMaxBruteForce(MR.Const_Vector3f dir, MR.Const_MeshVertPart mp)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshVertPart", ExactSpelling = true)]
        extern static MR.MinMaxArg_Float_MRVertId._Underlying *__MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshVertPart(MR.Const_Vector3f._Underlying *dir, MR.Const_MeshVertPart._Underlying *mp);
        return new(__MR_findDirMinMaxBruteForce_2_const_MR_Vector3f_ref_MR_MeshVertPart(dir._UnderlyingPtr, mp._UnderlyingPtr), is_owning: true);
    }
}
