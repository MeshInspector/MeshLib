public static partial class MR
{
    /// \brief Makes normals for valid points of given point cloud by directing them along the normal of best plane through the neighbours
    /// \param radius of neighborhood to consider
    /// \param orient OrientNormals::Smart here means orientation from best fit plane
    /// \return nullopt if progress returned false
    /// Generated from function `MR::makeUnorientedNormals`.
    /// Parameter `progress` defaults to `{}`.
    /// Parameter `orient` defaults to `OrientNormals::Smart`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertCoords> MakeUnorientedNormals(MR.Const_PointCloud pointCloud, float radius, MR.Std.Const_Function_BoolFuncFromFloat? progress = null, MR.OrientNormals? orient = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeUnorientedNormals_4_float", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_makeUnorientedNormals_4_float(MR.Const_PointCloud._Underlying *pointCloud, float radius, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress, MR.OrientNormals *orient);
        MR.OrientNormals __deref_orient = orient.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Optional_MRVertCoords(__MR_makeUnorientedNormals_4_float(pointCloud._UnderlyingPtr, radius, progress is not null ? progress._UnderlyingPtr : null, orient.HasValue ? &__deref_orient : null), is_owning: true));
    }

    /// \brief Makes normals for valid points of given point cloud by averaging neighbor triangle normals weighted by triangle's angle
    /// \triangs triangulation neighbours of each point
    /// \param orient OrientNormals::Smart here means orientation from normals of neigbour triangles
    /// \return nullopt if progress returned false
    /// Generated from function `MR::makeUnorientedNormals`.
    /// Parameter `progress` defaults to `{}`.
    /// Parameter `orient` defaults to `OrientNormals::Smart`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertCoords> MakeUnorientedNormals(MR.Const_PointCloud pointCloud, MR.Const_AllLocalTriangulations triangs, MR.Std.Const_Function_BoolFuncFromFloat? progress = null, MR.OrientNormals? orient = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeUnorientedNormals_4_MR_AllLocalTriangulations", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_makeUnorientedNormals_4_MR_AllLocalTriangulations(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_AllLocalTriangulations._Underlying *triangs, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress, MR.OrientNormals *orient);
        MR.OrientNormals __deref_orient = orient.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Optional_MRVertCoords(__MR_makeUnorientedNormals_4_MR_AllLocalTriangulations(pointCloud._UnderlyingPtr, triangs._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null, orient.HasValue ? &__deref_orient : null), is_owning: true));
    }

    /// \brief Makes normals for valid points of given point cloud by directing them along the normal of best plane through the neighbours
    /// \param closeVerts a buffer where for every valid point #i its neighbours are stored at indices [i*numNei; (i+1)*numNei)
    /// \param orient OrientNormals::Smart here means orientation from best fit plane
    /// \return nullopt if progress returned false
    /// Generated from function `MR::makeUnorientedNormals`.
    /// Parameter `progress` defaults to `{}`.
    /// Parameter `orient` defaults to `OrientNormals::Smart`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertCoords> MakeUnorientedNormals(MR.Const_PointCloud pointCloud, MR.Const_Buffer_MRVertId closeVerts, int numNei, MR.Std.Const_Function_BoolFuncFromFloat? progress = null, MR.OrientNormals? orient = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeUnorientedNormals_5", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_makeUnorientedNormals_5(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_Buffer_MRVertId._Underlying *closeVerts, int numNei, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress, MR.OrientNormals *orient);
        MR.OrientNormals __deref_orient = orient.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Optional_MRVertCoords(__MR_makeUnorientedNormals_5(pointCloud._UnderlyingPtr, closeVerts._UnderlyingPtr, numNei, progress is not null ? progress._UnderlyingPtr : null, orient.HasValue ? &__deref_orient : null), is_owning: true));
    }

    /// \brief Select orientation of given normals to make directions of close points consistent;
    /// \param radius of neighborhood to consider
    /// \return false if progress returned false
    /// Generated from function `MR::orientNormals`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe bool OrientNormals_(MR.Const_PointCloud pointCloud, MR.VertCoords normals, float radius, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientNormals_4_float", ExactSpelling = true)]
        extern static byte __MR_orientNormals_4_float(MR.Const_PointCloud._Underlying *pointCloud, MR.VertCoords._Underlying *normals, float radius, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return __MR_orientNormals_4_float(pointCloud._UnderlyingPtr, normals._UnderlyingPtr, radius, progress is not null ? progress._UnderlyingPtr : null) != 0;
    }

    /// \brief Select orientation of given normals to make directions of close points consistent;
    /// \param radius of neighborhood to consider
    /// \return false if progress returned false
    /// Unlike simple orientNormals this method constructs local triangulations around each point
    /// (with most neighbours within given radius and all neighbours within 2*radius)
    /// and considers all triangulation neighbors and not other points from the ball around each point.
    /// Generated from function `MR::orientNormals`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe bool OrientNormals_(MR.Const_PointCloud pointCloud, MR.VertCoords normals, MR.Const_AllLocalTriangulations triangs, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientNormals_4_MR_AllLocalTriangulations", ExactSpelling = true)]
        extern static byte __MR_orientNormals_4_MR_AllLocalTriangulations(MR.Const_PointCloud._Underlying *pointCloud, MR.VertCoords._Underlying *normals, MR.Const_AllLocalTriangulations._Underlying *triangs, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return __MR_orientNormals_4_MR_AllLocalTriangulations(pointCloud._UnderlyingPtr, normals._UnderlyingPtr, triangs._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null) != 0;
    }

    /// \brief Select orientation of given normals to make directions of close points consistent;
    /// \param closeVerts a buffer where for every valid point #i its neighbours are stored at indices [i*numNei; (i+1)*numNei)
    /// \return false if progress returned false
    /// Generated from function `MR::orientNormals`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe bool OrientNormals_(MR.Const_PointCloud pointCloud, MR.VertCoords normals, MR.Const_Buffer_MRVertId closeVerts, int numNei, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientNormals_5", ExactSpelling = true)]
        extern static byte __MR_orientNormals_5(MR.Const_PointCloud._Underlying *pointCloud, MR.VertCoords._Underlying *normals, MR.Const_Buffer_MRVertId._Underlying *closeVerts, int numNei, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return __MR_orientNormals_5(pointCloud._UnderlyingPtr, normals._UnderlyingPtr, closeVerts._UnderlyingPtr, numNei, progress is not null ? progress._UnderlyingPtr : null) != 0;
    }

    /// \brief Makes normals for valid points of given point cloud; directions of close points are selected to be consistent;
    /// \param radius of neighborhood to consider
    /// \return nullopt if progress returned false
    /// Generated from function `MR::makeOrientedNormals`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertCoords> MakeOrientedNormals(MR.Const_PointCloud pointCloud, float radius, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeOrientedNormals_float", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_makeOrientedNormals_float(MR.Const_PointCloud._Underlying *pointCloud, float radius, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.Std.Optional_MRVertCoords(__MR_makeOrientedNormals_float(pointCloud._UnderlyingPtr, radius, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /// \brief Makes normals for valid points of given point cloud; directions of close points are selected to be consistent;
    /// \triangs triangulation neighbours of each point, which are oriented during the call as well
    /// \return nullopt if progress returned false
    /// Generated from function `MR::makeOrientedNormals`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertCoords> MakeOrientedNormals(MR.Const_PointCloud pointCloud, MR.AllLocalTriangulations triangs, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeOrientedNormals_MR_AllLocalTriangulations", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_makeOrientedNormals_MR_AllLocalTriangulations(MR.Const_PointCloud._Underlying *pointCloud, MR.AllLocalTriangulations._Underlying *triangs, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.Std.Optional_MRVertCoords(__MR_makeOrientedNormals_MR_AllLocalTriangulations(pointCloud._UnderlyingPtr, triangs._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /// \brief Makes consistent normals for valid points of given point cloud
    /// \param avgNeighborhoodSize avg num of neighbors of each individual point
    //[[deprecated( "use makeOrientedNormals(...) instead" )]]
    /// Generated from function `MR::makeNormals`.
    /// Parameter `avgNeighborhoodSize` defaults to `48`.
    public static unsafe MR.Misc._Moved<MR.VertCoords> MakeNormals(MR.Const_PointCloud pointCloud, int? avgNeighborhoodSize = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeNormals", ExactSpelling = true)]
        extern static MR.VertCoords._Underlying *__MR_makeNormals(MR.Const_PointCloud._Underlying *pointCloud, int *avgNeighborhoodSize);
        int __deref_avgNeighborhoodSize = avgNeighborhoodSize.GetValueOrDefault();
        return MR.Misc.Move(new MR.VertCoords(__MR_makeNormals(pointCloud._UnderlyingPtr, avgNeighborhoodSize.HasValue ? &__deref_avgNeighborhoodSize : null), is_owning: true));
    }
}
