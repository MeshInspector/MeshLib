public static partial class MR
{
    /// computes path distances in mesh vertices from given start vertices, stopping when maxDist is reached;
    /// considered paths can go either along edges or straightly within triangles
    /// Generated from function `MR::computeSurfaceDistances`.
    /// Parameter `maxDist` defaults to `3.40282347e38f`.
    /// Parameter `maxVertUpdates` defaults to `3`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSurfaceDistances(MR.Const_Mesh mesh, MR.Const_VertBitSet startVertices, float? maxDist = null, MR.Const_VertBitSet? region = null, int? maxVertUpdates = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfaceDistances_5_MR_VertBitSet", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSurfaceDistances_5_MR_VertBitSet(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *startVertices, float *maxDist, MR.Const_VertBitSet._Underlying *region, int *maxVertUpdates);
        float __deref_maxDist = maxDist.GetValueOrDefault();
        int __deref_maxVertUpdates = maxVertUpdates.GetValueOrDefault();
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSurfaceDistances_5_MR_VertBitSet(mesh._UnderlyingPtr, startVertices._UnderlyingPtr, maxDist.HasValue ? &__deref_maxDist : null, region is not null ? region._UnderlyingPtr : null, maxVertUpdates.HasValue ? &__deref_maxVertUpdates : null), is_owning: true));
    }

    /// computes path distances in mesh vertices from given start vertices, stopping when all targetVertices or maxDist is reached;
    /// considered paths can go either along edges or straightly within triangles
    /// Generated from function `MR::computeSurfaceDistances`.
    /// Parameter `maxDist` defaults to `3.40282347e38f`.
    /// Parameter `maxVertUpdates` defaults to `3`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSurfaceDistances(MR.Const_Mesh mesh, MR.Const_VertBitSet startVertices, MR.Const_VertBitSet targetVertices, float? maxDist = null, MR.Const_VertBitSet? region = null, int? maxVertUpdates = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfaceDistances_6_MR_VertBitSet", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSurfaceDistances_6_MR_VertBitSet(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *startVertices, MR.Const_VertBitSet._Underlying *targetVertices, float *maxDist, MR.Const_VertBitSet._Underlying *region, int *maxVertUpdates);
        float __deref_maxDist = maxDist.GetValueOrDefault();
        int __deref_maxVertUpdates = maxVertUpdates.GetValueOrDefault();
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSurfaceDistances_6_MR_VertBitSet(mesh._UnderlyingPtr, startVertices._UnderlyingPtr, targetVertices._UnderlyingPtr, maxDist.HasValue ? &__deref_maxDist : null, region is not null ? region._UnderlyingPtr : null, maxVertUpdates.HasValue ? &__deref_maxVertUpdates : null), is_owning: true));
    }

    /// computes path distances in mesh vertices from given start vertices with values in them, stopping when maxDist is reached;
    /// considered paths can go either along edges or straightly within triangles
    /// Generated from function `MR::computeSurfaceDistances`.
    /// Parameter `maxDist` defaults to `3.40282347e38f`.
    /// Parameter `maxVertUpdates` defaults to `3`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSurfaceDistances(MR.Const_Mesh mesh, MR.Phmap.Const_FlatHashMap_MRVertId_Float startVertices, float? maxDist = null, MR.Const_VertBitSet? region = null, int? maxVertUpdates = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfaceDistances_5_phmap_flat_hash_map_MR_VertId_float", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSurfaceDistances_5_phmap_flat_hash_map_MR_VertId_float(MR.Const_Mesh._Underlying *mesh, MR.Phmap.Const_FlatHashMap_MRVertId_Float._Underlying *startVertices, float *maxDist, MR.Const_VertBitSet._Underlying *region, int *maxVertUpdates);
        float __deref_maxDist = maxDist.GetValueOrDefault();
        int __deref_maxVertUpdates = maxVertUpdates.GetValueOrDefault();
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSurfaceDistances_5_phmap_flat_hash_map_MR_VertId_float(mesh._UnderlyingPtr, startVertices._UnderlyingPtr, maxDist.HasValue ? &__deref_maxDist : null, region is not null ? region._UnderlyingPtr : null, maxVertUpdates.HasValue ? &__deref_maxVertUpdates : null), is_owning: true));
    }

    /// computes path distance in mesh vertices from given start point, stopping when all vertices in the face where end is located are reached;
    /// \details considered paths can go either along edges or straightly within triangles
    /// \param endReached if pointer provided it will receive where a path from start to end exists
    /// Generated from function `MR::computeSurfaceDistances`.
    /// Parameter `maxVertUpdates` defaults to `3`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSurfaceDistances(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint end, MR.Const_VertBitSet? region = null, MR.Misc.InOut<bool>? endReached = null, int? maxVertUpdates = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfaceDistances_6_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSurfaceDistances_6_MR_MeshTriPoint(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *end, MR.Const_VertBitSet._Underlying *region, bool *endReached, int *maxVertUpdates);
        bool __value_endReached = endReached is not null ? endReached.Value : default(bool);
        int __deref_maxVertUpdates = maxVertUpdates.GetValueOrDefault();
        var __ret = __MR_computeSurfaceDistances_6_MR_MeshTriPoint(mesh._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, endReached is not null ? &__value_endReached : null, maxVertUpdates.HasValue ? &__deref_maxVertUpdates : null);
        if (endReached is not null) endReached.Value = __value_endReached;
        return MR.Misc.Move(new MR.VertScalars(__ret, is_owning: true));
    }

    /// computes path distances in mesh vertices from given start point, stopping when maxDist is reached;
    /// considered paths can go either along edges or straightly within triangles
    /// Generated from function `MR::computeSurfaceDistances`.
    /// Parameter `maxDist` defaults to `3.40282347e38f`.
    /// Parameter `maxVertUpdates` defaults to `3`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSurfaceDistances(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, float? maxDist = null, MR.Const_VertBitSet? region = null, int? maxVertUpdates = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfaceDistances_5_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSurfaceDistances_5_MR_MeshTriPoint(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, float *maxDist, MR.Const_VertBitSet._Underlying *region, int *maxVertUpdates);
        float __deref_maxDist = maxDist.GetValueOrDefault();
        int __deref_maxVertUpdates = maxVertUpdates.GetValueOrDefault();
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSurfaceDistances_5_MR_MeshTriPoint(mesh._UnderlyingPtr, start._UnderlyingPtr, maxDist.HasValue ? &__deref_maxDist : null, region is not null ? region._UnderlyingPtr : null, maxVertUpdates.HasValue ? &__deref_maxVertUpdates : null), is_owning: true));
    }

    /// computes path distances in mesh vertices from given start points, stopping when maxDist is reached;
    /// considered paths can go either along edges or straightly within triangles
    /// Generated from function `MR::computeSurfaceDistances`.
    /// Parameter `maxDist` defaults to `3.40282347e38f`.
    /// Parameter `maxVertUpdates` defaults to `3`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSurfaceDistances(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint starts, float? maxDist = null, MR.Const_VertBitSet? region = null, int? maxVertUpdates = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfaceDistances_5_std_vector_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSurfaceDistances_5_std_vector_MR_MeshTriPoint(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *starts, float *maxDist, MR.Const_VertBitSet._Underlying *region, int *maxVertUpdates);
        float __deref_maxDist = maxDist.GetValueOrDefault();
        int __deref_maxVertUpdates = maxVertUpdates.GetValueOrDefault();
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSurfaceDistances_5_std_vector_MR_MeshTriPoint(mesh._UnderlyingPtr, starts._UnderlyingPtr, maxDist.HasValue ? &__deref_maxDist : null, region is not null ? region._UnderlyingPtr : null, maxVertUpdates.HasValue ? &__deref_maxVertUpdates : null), is_owning: true));
    }
}
