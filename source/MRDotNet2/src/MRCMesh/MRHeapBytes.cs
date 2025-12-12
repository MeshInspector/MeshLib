public static partial class MR
{
    /// returns the amount of memory given vector occupies on heap
    /// Generated from function `MR::heapBytes<MR::Color>`.
    public static unsafe ulong HeapBytes(MR.Std.Const_Vector_MRColor vec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_Color", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_Color(MR.Std.Const_Vector_MRColor._Underlying *vec);
        return __MR_heapBytes_MR_Color(vec._UnderlyingPtr);
    }

    /// returns the amount of memory given vector occupies on heap
    /// Generated from function `MR::heapBytes<std::shared_ptr<MR::Object>>`.
    public static unsafe ulong HeapBytes(MR.Std.Const_Vector_StdSharedPtrMRObject vec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_std_shared_ptr_MR_Object", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_std_shared_ptr_MR_Object(MR.Std.Const_Vector_StdSharedPtrMRObject._Underlying *vec);
        return __MR_heapBytes_std_shared_ptr_MR_Object(vec._UnderlyingPtr);
    }

    /// returns the amount of memory given vector occupies on heap
    /// Generated from function `MR::heapBytes<float>`.
    public static unsafe ulong HeapBytes(MR.Std.Const_Vector_Float vec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_float", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_float(MR.Std.Const_Vector_Float._Underlying *vec);
        return __MR_heapBytes_float(vec._UnderlyingPtr);
    }

    /// returns the amount of memory given vector occupies on heap
    /// Generated from function `MR::heapBytes<MR_uint64_t>`.
    public static unsafe ulong HeapBytes(MR.Std.Const_Vector_MRUint64T vec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_uint64_t", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_uint64_t(MR.Std.Const_Vector_MRUint64T._Underlying *vec);
        return __MR_heapBytes_uint64_t(vec._UnderlyingPtr);
    }

    /// Generated from function `MR::heapBytes<MR::MeshTexture, MR::TextureId>`.
    public static unsafe ulong HeapBytes(MR.Const_Vector_MRMeshTexture_MRTextureId vec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_MeshTexture_MR_TextureId", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_MeshTexture_MR_TextureId(MR.Const_Vector_MRMeshTexture_MRTextureId._Underlying *vec);
        return __MR_heapBytes_MR_MeshTexture_MR_TextureId(vec._UnderlyingPtr);
    }

    /// returns the amount of memory this smart pointer and its pointed object own together on heap
    /// Generated from function `MR::heapBytes<MR::Mesh>`.
    public static unsafe ulong HeapBytes(MR.Const_Mesh ptr)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_Mesh", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_Mesh(MR.Const_Mesh._UnderlyingShared *ptr);
        return __MR_heapBytes_MR_Mesh(ptr._UnderlyingSharedPtr);
    }

    /// returns the amount of memory this smart pointer and its pointed object own together on heap
    /// Generated from function `MR::heapBytes<MR::Object>`.
    public static unsafe ulong HeapBytes(MR.Const_Object ptr)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_Object", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_Object(MR.Const_Object._UnderlyingShared *ptr);
        return __MR_heapBytes_MR_Object(ptr._UnderlyingSharedPtr);
    }

    /// returns the amount of memory this smart pointer and its pointed object own together on heap
    /// Generated from function `MR::heapBytes<MR::PointCloud>`.
    public static unsafe ulong HeapBytes(MR.Const_PointCloud ptr)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_PointCloud", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_PointCloud(MR.Const_PointCloud._UnderlyingShared *ptr);
        return __MR_heapBytes_MR_PointCloud(ptr._UnderlyingSharedPtr);
    }

    /// returns the amount of memory this smart pointer and its pointed object own together on heap
    /// Generated from function `MR::heapBytes<MR::Polyline3>`.
    public static unsafe ulong HeapBytes(MR.Const_Polyline3 ptr)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_Polyline3", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_Polyline3(MR.Const_Polyline3._UnderlyingShared *ptr);
        return __MR_heapBytes_MR_Polyline3(ptr._UnderlyingSharedPtr);
    }
}
