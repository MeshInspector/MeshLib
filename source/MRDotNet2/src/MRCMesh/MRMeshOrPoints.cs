public static partial class MR
{
    /// This class can hold either mesh part or point cloud.
    /// It is used for generic algorithms operating with either of them
    /// Generated from class `MR::MeshOrPoints`.
    /// This is the const half of the class.
    public class Const_MeshOrPoints : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOrPoints(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOrPoints_Destroy(_Underlying *_this);
            __MR_MeshOrPoints_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOrPoints() {Dispose(false);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe Const_MeshOrPoints(MR.Const_MeshOrPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_ConstructFromAnother(MR.MeshOrPoints._Underlying *_other);
            _UnderlyingPtr = __MR_MeshOrPoints_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe Const_MeshOrPoints(MR.Const_MeshPart mp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_MeshPart", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_MeshPart(MR.Const_MeshPart._Underlying *mp);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_MeshPart(mp._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator Const_MeshOrPoints(MR.Const_MeshPart mp) {return new(mp);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe Const_MeshOrPoints(MR.Const_PointCloudPart pcp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_PointCloudPart", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_PointCloudPart(MR.Const_PointCloudPart._Underlying *pcp);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_PointCloudPart(pcp._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator Const_MeshOrPoints(MR.Const_PointCloudPart pcp) {return new(pcp);}

        // these constructors are redundant for C++, but important for python bindings
        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe Const_MeshOrPoints(MR.Const_Mesh mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_Mesh", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_Mesh(MR.Const_Mesh._Underlying *mesh);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_Mesh(mesh._UnderlyingPtr);
        }

        // these constructors are redundant for C++, but important for python bindings
        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator Const_MeshOrPoints(MR.Const_Mesh mesh) {return new(mesh);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe Const_MeshOrPoints(MR.Const_PointCloud pc) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_PointCloud", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_PointCloud(MR.Const_PointCloud._Underlying *pc);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_PointCloud(pc._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator Const_MeshOrPoints(MR.Const_PointCloud pc) {return new(pc);}

        /// if this object holds a mesh part then returns pointer on it, otherwise returns nullptr
        /// Generated from method `MR::MeshOrPoints::asMeshPart`.
        public unsafe MR.Const_MeshPart? AsMeshPart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_asMeshPart", ExactSpelling = true)]
            extern static MR.Const_MeshPart._Underlying *__MR_MeshOrPoints_asMeshPart(_Underlying *_this);
            var __ret = __MR_MeshOrPoints_asMeshPart(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_MeshPart(__ret, is_owning: false) : null;
        }

        /// if this object holds a point cloud part then returns pointer on it, otherwise returns nullptr
        /// Generated from method `MR::MeshOrPoints::asPointCloudPart`.
        public unsafe MR.Const_PointCloudPart? AsPointCloudPart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_asPointCloudPart", ExactSpelling = true)]
            extern static MR.Const_PointCloudPart._Underlying *__MR_MeshOrPoints_asPointCloudPart(_Underlying *_this);
            var __ret = __MR_MeshOrPoints_asPointCloudPart(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_PointCloudPart(__ret, is_owning: false) : null;
        }

        /// returns the minimal bounding box containing all valid vertices of the object (and not only part of mesh);
        /// implemented via obj.getAABBTree()
        /// Generated from method `MR::MeshOrPoints::getObjBoundingBox`.
        public unsafe MR.Box3f GetObjBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_getObjBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_MeshOrPoints_getObjBoundingBox(_Underlying *_this);
            return __MR_MeshOrPoints_getObjBoundingBox(_UnderlyingPtr);
        }

        /// if AABBTree is already built does nothing otherwise builds and caches it
        /// Generated from method `MR::MeshOrPoints::cacheAABBTree`.
        public unsafe void CacheAABBTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_cacheAABBTree", ExactSpelling = true)]
            extern static void __MR_MeshOrPoints_cacheAABBTree(_Underlying *_this);
            __MR_MeshOrPoints_cacheAABBTree(_UnderlyingPtr);
        }

        /// passes through all valid vertices and finds the minimal bounding box containing all of them;
        /// if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::MeshOrPoints::computeBoundingBox`.
        public unsafe MR.Box3f ComputeBoundingBox(MR.Const_AffineXf3f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_computeBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_MeshOrPoints_computeBoundingBox(_Underlying *_this, MR.Const_AffineXf3f._Underlying *toWorld);
            return __MR_MeshOrPoints_computeBoundingBox(_UnderlyingPtr, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        /// Adds in existing PointAccumulator the elements of the contained object
        /// Generated from method `MR::MeshOrPoints::accumulate`.
        public unsafe void Accumulate(MR.PointAccumulator accum, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_accumulate", ExactSpelling = true)]
            extern static void __MR_MeshOrPoints_accumulate(_Underlying *_this, MR.PointAccumulator._Underlying *accum, MR.Const_AffineXf3f._Underlying *xf);
            __MR_MeshOrPoints_accumulate(_UnderlyingPtr, accum._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// performs sampling of vertices or points;
        /// subdivides bounding box of the object on voxels of approximately given size and returns at most one vertex per voxel;
        /// voxelSize is automatically increased to avoid more voxels than \param maxVoxels;
        /// if voxelSize == 0 then all valid points are returned;
        /// returns std::nullopt if it was terminated by the callback
        /// Generated from method `MR::MeshOrPoints::pointsGridSampling`.
        /// Parameter `maxVoxels` defaults to `500000`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> PointsGridSampling(float voxelSize, ulong? maxVoxels = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_pointsGridSampling", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_MeshOrPoints_pointsGridSampling(_Underlying *_this, float voxelSize, ulong *maxVoxels, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            ulong __deref_maxVoxels = maxVoxels.GetValueOrDefault();
            return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_MeshOrPoints_pointsGridSampling(_UnderlyingPtr, voxelSize, maxVoxels.HasValue ? &__deref_maxVoxels : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// gives access to points-vector (which can include invalid points as well)
        /// Generated from method `MR::MeshOrPoints::points`.
        public unsafe MR.Const_VertCoords Points()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_points", ExactSpelling = true)]
            extern static MR.Const_VertCoords._Underlying *__MR_MeshOrPoints_points(_Underlying *_this);
            return new(__MR_MeshOrPoints_points(_UnderlyingPtr), is_owning: false);
        }

        /// gives access to bit set of valid points
        /// Generated from method `MR::MeshOrPoints::validPoints`.
        public unsafe MR.Const_VertBitSet ValidPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_validPoints", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_MeshOrPoints_validPoints(_Underlying *_this);
            return new(__MR_MeshOrPoints_validPoints(_UnderlyingPtr), is_owning: false);
        }

        /// returns normals generating function: VertId->normal (or empty for point cloud without normals)
        /// Generated from method `MR::MeshOrPoints::normals`.
        public unsafe MR.Misc._Moved<MR.Std.Function_MRVector3fFuncFromMRVertId> Normals()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_normals", ExactSpelling = true)]
            extern static MR.Std.Function_MRVector3fFuncFromMRVertId._Underlying *__MR_MeshOrPoints_normals(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Function_MRVector3fFuncFromMRVertId(__MR_MeshOrPoints_normals(_UnderlyingPtr), is_owning: true));
        }

        /// returns weights generating function: VertId->float:
        /// for mesh it is double area of surrounding triangles, and for point cloud - nothing
        /// Generated from method `MR::MeshOrPoints::weights`.
        public unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMRVertId> Weights()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_weights", ExactSpelling = true)]
            extern static MR.Std.Function_FloatFuncFromMRVertId._Underlying *__MR_MeshOrPoints_weights(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMRVertId(__MR_MeshOrPoints_weights(_UnderlyingPtr), is_owning: true));
        }

        /// returns a function that finds projection (closest) points on this: Vector3f->ProjectionResult
        /// Generated from method `MR::MeshOrPoints::projector`.
        public unsafe MR.Misc._Moved<MR.Std.Function_MRMeshOrPointsProjectionResultFuncFromConstMRVector3fRef> Projector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_projector", ExactSpelling = true)]
            extern static MR.Std.Function_MRMeshOrPointsProjectionResultFuncFromConstMRVector3fRef._Underlying *__MR_MeshOrPoints_projector(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Function_MRMeshOrPointsProjectionResultFuncFromConstMRVector3fRef(__MR_MeshOrPoints_projector(_UnderlyingPtr), is_owning: true));
        }

        /// returns a function that updates previously known projection (closest) points on this,
        /// the update takes place only if newly found closest point is closer to p than sqrt(res.distSq) given on input
        /// The function returns true if the update has taken place.
        /// Generated from method `MR::MeshOrPoints::limitedProjector`.
        public unsafe MR.Misc._Moved<MR.Std.Function_BoolFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRef> LimitedProjector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_limitedProjector", ExactSpelling = true)]
            extern static MR.Std.Function_BoolFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRef._Underlying *__MR_MeshOrPoints_limitedProjector(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Function_BoolFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRef(__MR_MeshOrPoints_limitedProjector(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from class `MR::MeshOrPoints::ProjectionResult`.
        /// This is the const half of the class.
        public class Const_ProjectionResult : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ProjectionResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshOrPoints_ProjectionResult_Destroy(_Underlying *_this);
                __MR_MeshOrPoints_ProjectionResult_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ProjectionResult() {Dispose(false);}

            /// found closest point
            public unsafe MR.Const_Vector3f Point
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_Get_point", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_MeshOrPoints_ProjectionResult_Get_point(_Underlying *_this);
                    return new(__MR_MeshOrPoints_ProjectionResult_Get_point(_UnderlyingPtr), is_owning: false);
                }
            }

            /// normal at the closest point;
            /// for meshes it will be pseudonormal with the differentiation depending on closest point location (face/edge/vertex)
            public unsafe MR.Std.Const_Optional_MRVector3f Normal
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_Get_normal", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRVector3f._Underlying *__MR_MeshOrPoints_ProjectionResult_Get_normal(_Underlying *_this);
                    return new(__MR_MeshOrPoints_ProjectionResult_Get_normal(_UnderlyingPtr), is_owning: false);
                }
            }

            /// can be true only for meshes, if the closest point is located on the boundary of the mesh (or the current region)
            public unsafe bool IsBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_Get_isBd", ExactSpelling = true)]
                    extern static bool *__MR_MeshOrPoints_ProjectionResult_Get_isBd(_Underlying *_this);
                    return *__MR_MeshOrPoints_ProjectionResult_Get_isBd(_UnderlyingPtr);
                }
            }

            /// squared distance from query point to the closest point
            public unsafe float DistSq
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_Get_distSq", ExactSpelling = true)]
                    extern static float *__MR_MeshOrPoints_ProjectionResult_Get_distSq(_Underlying *_this);
                    return *__MR_MeshOrPoints_ProjectionResult_Get_distSq(_UnderlyingPtr);
                }
            }

            /// for point clouds it is the closest vertex,
            /// for meshes it is the closest vertex of the triangle with the closest point
            public unsafe MR.Const_VertId ClosestVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_Get_closestVert", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_MeshOrPoints_ProjectionResult_Get_closestVert(_Underlying *_this);
                    return new(__MR_MeshOrPoints_ProjectionResult_Get_closestVert(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ProjectionResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshOrPoints_ProjectionResult_DefaultConstruct();
            }

            /// Constructs `MR::MeshOrPoints::ProjectionResult` elementwise.
            public unsafe Const_ProjectionResult(MR.Vector3f point, MR._InOpt_Vector3f normal, bool isBd, float distSq, MR.VertId closestVert) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_ConstructFrom(MR.Vector3f point, MR.Vector3f *normal, byte isBd, float distSq, MR.VertId closestVert);
                _UnderlyingPtr = __MR_MeshOrPoints_ProjectionResult_ConstructFrom(point, normal.HasValue ? &normal.Object : null, isBd ? (byte)1 : (byte)0, distSq, closestVert);
            }

            /// Generated from constructor `MR::MeshOrPoints::ProjectionResult::ProjectionResult`.
            public unsafe Const_ProjectionResult(MR.MeshOrPoints.Const_ProjectionResult _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_ConstructFromAnother(MR.MeshOrPoints.ProjectionResult._Underlying *_other);
                _UnderlyingPtr = __MR_MeshOrPoints_ProjectionResult_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from conversion operator `MR::MeshOrPoints::ProjectionResult::operator bool`.
            public static unsafe explicit operator bool(MR.MeshOrPoints.Const_ProjectionResult _this)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_ConvertTo_bool", ExactSpelling = true)]
                extern static byte __MR_MeshOrPoints_ProjectionResult_ConvertTo_bool(MR.MeshOrPoints.Const_ProjectionResult._Underlying *_this);
                return __MR_MeshOrPoints_ProjectionResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
            }

            /// Generated from method `MR::MeshOrPoints::ProjectionResult::valid`.
            public unsafe bool Valid()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_valid", ExactSpelling = true)]
                extern static byte __MR_MeshOrPoints_ProjectionResult_valid(_Underlying *_this);
                return __MR_MeshOrPoints_ProjectionResult_valid(_UnderlyingPtr) != 0;
            }
        }

        /// Generated from class `MR::MeshOrPoints::ProjectionResult`.
        /// This is the non-const half of the class.
        public class ProjectionResult : Const_ProjectionResult
        {
            internal unsafe ProjectionResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// found closest point
            public new unsafe MR.Mut_Vector3f Point
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_GetMutable_point", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_MeshOrPoints_ProjectionResult_GetMutable_point(_Underlying *_this);
                    return new(__MR_MeshOrPoints_ProjectionResult_GetMutable_point(_UnderlyingPtr), is_owning: false);
                }
            }

            /// normal at the closest point;
            /// for meshes it will be pseudonormal with the differentiation depending on closest point location (face/edge/vertex)
            public new unsafe MR.Std.Optional_MRVector3f Normal
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_GetMutable_normal", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRVector3f._Underlying *__MR_MeshOrPoints_ProjectionResult_GetMutable_normal(_Underlying *_this);
                    return new(__MR_MeshOrPoints_ProjectionResult_GetMutable_normal(_UnderlyingPtr), is_owning: false);
                }
            }

            /// can be true only for meshes, if the closest point is located on the boundary of the mesh (or the current region)
            public new unsafe ref bool IsBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_GetMutable_isBd", ExactSpelling = true)]
                    extern static bool *__MR_MeshOrPoints_ProjectionResult_GetMutable_isBd(_Underlying *_this);
                    return ref *__MR_MeshOrPoints_ProjectionResult_GetMutable_isBd(_UnderlyingPtr);
                }
            }

            /// squared distance from query point to the closest point
            public new unsafe ref float DistSq
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_GetMutable_distSq", ExactSpelling = true)]
                    extern static float *__MR_MeshOrPoints_ProjectionResult_GetMutable_distSq(_Underlying *_this);
                    return ref *__MR_MeshOrPoints_ProjectionResult_GetMutable_distSq(_UnderlyingPtr);
                }
            }

            /// for point clouds it is the closest vertex,
            /// for meshes it is the closest vertex of the triangle with the closest point
            public new unsafe MR.Mut_VertId ClosestVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_GetMutable_closestVert", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_MeshOrPoints_ProjectionResult_GetMutable_closestVert(_Underlying *_this);
                    return new(__MR_MeshOrPoints_ProjectionResult_GetMutable_closestVert(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ProjectionResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshOrPoints_ProjectionResult_DefaultConstruct();
            }

            /// Constructs `MR::MeshOrPoints::ProjectionResult` elementwise.
            public unsafe ProjectionResult(MR.Vector3f point, MR._InOpt_Vector3f normal, bool isBd, float distSq, MR.VertId closestVert) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_ConstructFrom(MR.Vector3f point, MR.Vector3f *normal, byte isBd, float distSq, MR.VertId closestVert);
                _UnderlyingPtr = __MR_MeshOrPoints_ProjectionResult_ConstructFrom(point, normal.HasValue ? &normal.Object : null, isBd ? (byte)1 : (byte)0, distSq, closestVert);
            }

            /// Generated from constructor `MR::MeshOrPoints::ProjectionResult::ProjectionResult`.
            public unsafe ProjectionResult(MR.MeshOrPoints.Const_ProjectionResult _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_ConstructFromAnother(MR.MeshOrPoints.ProjectionResult._Underlying *_other);
                _UnderlyingPtr = __MR_MeshOrPoints_ProjectionResult_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshOrPoints::ProjectionResult::operator=`.
            public unsafe MR.MeshOrPoints.ProjectionResult Assign(MR.MeshOrPoints.Const_ProjectionResult _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ProjectionResult_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_MeshOrPoints_ProjectionResult_AssignFromAnother(_Underlying *_this, MR.MeshOrPoints.ProjectionResult._Underlying *_other);
                return new(__MR_MeshOrPoints_ProjectionResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ProjectionResult` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ProjectionResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ProjectionResult`/`Const_ProjectionResult` directly.
        public class _InOptMut_ProjectionResult
        {
            public ProjectionResult? Opt;

            public _InOptMut_ProjectionResult() {}
            public _InOptMut_ProjectionResult(ProjectionResult value) {Opt = value;}
            public static implicit operator _InOptMut_ProjectionResult(ProjectionResult value) {return new(value);}
        }

        /// This is used for optional parameters of class `ProjectionResult` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ProjectionResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ProjectionResult`/`Const_ProjectionResult` to pass it to the function.
        public class _InOptConst_ProjectionResult
        {
            public Const_ProjectionResult? Opt;

            public _InOptConst_ProjectionResult() {}
            public _InOptConst_ProjectionResult(Const_ProjectionResult value) {Opt = value;}
            public static implicit operator _InOptConst_ProjectionResult(Const_ProjectionResult value) {return new(value);}
        }
    }

    /// This class can hold either mesh part or point cloud.
    /// It is used for generic algorithms operating with either of them
    /// Generated from class `MR::MeshOrPoints`.
    /// This is the non-const half of the class.
    public class MeshOrPoints : Const_MeshOrPoints
    {
        internal unsafe MeshOrPoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe MeshOrPoints(MR.Const_MeshOrPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_ConstructFromAnother(MR.MeshOrPoints._Underlying *_other);
            _UnderlyingPtr = __MR_MeshOrPoints_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe MeshOrPoints(MR.Const_MeshPart mp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_MeshPart", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_MeshPart(MR.Const_MeshPart._Underlying *mp);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_MeshPart(mp._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator MeshOrPoints(MR.Const_MeshPart mp) {return new(mp);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe MeshOrPoints(MR.Const_PointCloudPart pcp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_PointCloudPart", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_PointCloudPart(MR.Const_PointCloudPart._Underlying *pcp);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_PointCloudPart(pcp._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator MeshOrPoints(MR.Const_PointCloudPart pcp) {return new(pcp);}

        // these constructors are redundant for C++, but important for python bindings
        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe MeshOrPoints(MR.Const_Mesh mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_Mesh", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_Mesh(MR.Const_Mesh._Underlying *mesh);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_Mesh(mesh._UnderlyingPtr);
        }

        // these constructors are redundant for C++, but important for python bindings
        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator MeshOrPoints(MR.Const_Mesh mesh) {return new(mesh);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public unsafe MeshOrPoints(MR.Const_PointCloud pc) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_Construct_MR_PointCloud", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_Construct_MR_PointCloud(MR.Const_PointCloud._Underlying *pc);
            _UnderlyingPtr = __MR_MeshOrPoints_Construct_MR_PointCloud(pc._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator MeshOrPoints(MR.Const_PointCloud pc) {return new(pc);}

        /// Generated from method `MR::MeshOrPoints::operator=`.
        public unsafe MR.MeshOrPoints Assign(MR.Const_MeshOrPoints _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPoints_AssignFromAnother(_Underlying *_this, MR.MeshOrPoints._Underlying *_other);
            return new(__MR_MeshOrPoints_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshOrPoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOrPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOrPoints`/`Const_MeshOrPoints` directly.
    public class _InOptMut_MeshOrPoints
    {
        public MeshOrPoints? Opt;

        public _InOptMut_MeshOrPoints() {}
        public _InOptMut_MeshOrPoints(MeshOrPoints value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOrPoints(MeshOrPoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOrPoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOrPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOrPoints`/`Const_MeshOrPoints` to pass it to the function.
    public class _InOptConst_MeshOrPoints
    {
        public Const_MeshOrPoints? Opt;

        public _InOptConst_MeshOrPoints() {}
        public _InOptConst_MeshOrPoints(Const_MeshOrPoints value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOrPoints(Const_MeshOrPoints value) {return new(value);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator _InOptConst_MeshOrPoints(MR.Const_MeshPart mp) {return new MR.MeshOrPoints(mp);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator _InOptConst_MeshOrPoints(MR.Const_PointCloudPart pcp) {return new MR.MeshOrPoints(pcp);}

        // these constructors are redundant for C++, but important for python bindings
        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator _InOptConst_MeshOrPoints(MR.Const_Mesh mesh) {return new MR.MeshOrPoints(mesh);}

        /// Generated from constructor `MR::MeshOrPoints::MeshOrPoints`.
        public static unsafe implicit operator _InOptConst_MeshOrPoints(MR.Const_PointCloud pc) {return new MR.MeshOrPoints(pc);}
    }

    /// an object and its transformation to global space with other objects
    /// Generated from class `MR::MeshOrPointsXf`.
    /// This is the const half of the class.
    public class Const_MeshOrPointsXf : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOrPointsXf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOrPointsXf_Destroy(_Underlying *_this);
            __MR_MeshOrPointsXf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOrPointsXf() {Dispose(false);}

        public unsafe MR.Const_MeshOrPoints Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_Get_obj", ExactSpelling = true)]
                extern static MR.Const_MeshOrPoints._Underlying *__MR_MeshOrPointsXf_Get_obj(_Underlying *_this);
                return new(__MR_MeshOrPointsXf_Get_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_AffineXf3f Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_Get_xf", ExactSpelling = true)]
                extern static MR.Const_AffineXf3f._Underlying *__MR_MeshOrPointsXf_Get_xf(_Underlying *_this);
                return new(__MR_MeshOrPointsXf_Get_xf(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::MeshOrPointsXf::MeshOrPointsXf`.
        public unsafe Const_MeshOrPointsXf(MR.Const_MeshOrPointsXf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPointsXf._Underlying *__MR_MeshOrPointsXf_ConstructFromAnother(MR.MeshOrPointsXf._Underlying *_other);
            _UnderlyingPtr = __MR_MeshOrPointsXf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Constructs `MR::MeshOrPointsXf` elementwise.
        public unsafe Const_MeshOrPointsXf(MR.Const_MeshOrPoints obj, MR.AffineXf3f xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshOrPointsXf._Underlying *__MR_MeshOrPointsXf_ConstructFrom(MR.MeshOrPoints._Underlying *obj, MR.AffineXf3f xf);
            _UnderlyingPtr = __MR_MeshOrPointsXf_ConstructFrom(obj._UnderlyingPtr, xf);
        }

        /// returns a function that finds projection (closest) points on this: Vector3f->ProjectionResult
        /// Generated from method `MR::MeshOrPointsXf::projector`.
        public unsafe MR.Misc._Moved<MR.Std.Function_MRMeshOrPointsProjectionResultFuncFromConstMRVector3fRef> Projector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_projector", ExactSpelling = true)]
            extern static MR.Std.Function_MRMeshOrPointsProjectionResultFuncFromConstMRVector3fRef._Underlying *__MR_MeshOrPointsXf_projector(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Function_MRMeshOrPointsProjectionResultFuncFromConstMRVector3fRef(__MR_MeshOrPointsXf_projector(_UnderlyingPtr), is_owning: true));
        }

        /// returns a function that updates previously known projection (closest) points on this,
        /// the update takes place only if newly found closest point is closer to p than sqrt(res.distSq) given on input
        /// Generated from method `MR::MeshOrPointsXf::limitedProjector`.
        public unsafe MR.Misc._Moved<MR.Std.Function_BoolFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRef> LimitedProjector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_limitedProjector", ExactSpelling = true)]
            extern static MR.Std.Function_BoolFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRef._Underlying *__MR_MeshOrPointsXf_limitedProjector(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Function_BoolFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRef(__MR_MeshOrPointsXf_limitedProjector(_UnderlyingPtr), is_owning: true));
        }
    }

    /// an object and its transformation to global space with other objects
    /// Generated from class `MR::MeshOrPointsXf`.
    /// This is the non-const half of the class.
    public class MeshOrPointsXf : Const_MeshOrPointsXf
    {
        internal unsafe MeshOrPointsXf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.MeshOrPoints Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_GetMutable_obj", ExactSpelling = true)]
                extern static MR.MeshOrPoints._Underlying *__MR_MeshOrPointsXf_GetMutable_obj(_Underlying *_this);
                return new(__MR_MeshOrPointsXf_GetMutable_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_AffineXf3f Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_GetMutable_xf", ExactSpelling = true)]
                extern static MR.Mut_AffineXf3f._Underlying *__MR_MeshOrPointsXf_GetMutable_xf(_Underlying *_this);
                return new(__MR_MeshOrPointsXf_GetMutable_xf(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::MeshOrPointsXf::MeshOrPointsXf`.
        public unsafe MeshOrPointsXf(MR.Const_MeshOrPointsXf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPointsXf._Underlying *__MR_MeshOrPointsXf_ConstructFromAnother(MR.MeshOrPointsXf._Underlying *_other);
            _UnderlyingPtr = __MR_MeshOrPointsXf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Constructs `MR::MeshOrPointsXf` elementwise.
        public unsafe MeshOrPointsXf(MR.Const_MeshOrPoints obj, MR.AffineXf3f xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshOrPointsXf._Underlying *__MR_MeshOrPointsXf_ConstructFrom(MR.MeshOrPoints._Underlying *obj, MR.AffineXf3f xf);
            _UnderlyingPtr = __MR_MeshOrPointsXf_ConstructFrom(obj._UnderlyingPtr, xf);
        }

        /// Generated from method `MR::MeshOrPointsXf::operator=`.
        public unsafe MR.MeshOrPointsXf Assign(MR.Const_MeshOrPointsXf _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOrPointsXf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshOrPointsXf._Underlying *__MR_MeshOrPointsXf_AssignFromAnother(_Underlying *_this, MR.MeshOrPointsXf._Underlying *_other);
            return new(__MR_MeshOrPointsXf_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshOrPointsXf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOrPointsXf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOrPointsXf`/`Const_MeshOrPointsXf` directly.
    public class _InOptMut_MeshOrPointsXf
    {
        public MeshOrPointsXf? Opt;

        public _InOptMut_MeshOrPointsXf() {}
        public _InOptMut_MeshOrPointsXf(MeshOrPointsXf value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOrPointsXf(MeshOrPointsXf value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOrPointsXf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOrPointsXf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOrPointsXf`/`Const_MeshOrPointsXf` to pass it to the function.
    public class _InOptConst_MeshOrPointsXf
    {
        public Const_MeshOrPointsXf? Opt;

        public _InOptConst_MeshOrPointsXf() {}
        public _InOptConst_MeshOrPointsXf(Const_MeshOrPointsXf value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOrPointsXf(Const_MeshOrPointsXf value) {return new(value);}
    }

    /// constructs MeshOrPoints from ObjectMesh or ObjectPoints, otherwise returns nullopt
    /// Generated from function `MR::getMeshOrPoints`.
    public static unsafe MR.Std.Optional_MRMeshOrPoints GetMeshOrPoints(MR.Const_Object? obj)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMeshOrPoints", ExactSpelling = true)]
        extern static MR.Std.Optional_MRMeshOrPoints._Underlying *__MR_getMeshOrPoints(MR.Const_Object._Underlying *obj);
        return new(__MR_getMeshOrPoints(obj is not null ? obj._UnderlyingPtr : null), is_owning: true);
    }

    /// Generated from function `MR::getMeshOrPointsXf`.
    public static unsafe MR.Std.Optional_MRMeshOrPointsXf GetMeshOrPointsXf(MR.Const_Object? obj)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMeshOrPointsXf", ExactSpelling = true)]
        extern static MR.Std.Optional_MRMeshOrPointsXf._Underlying *__MR_getMeshOrPointsXf(MR.Const_Object._Underlying *obj);
        return new(__MR_getMeshOrPointsXf(obj is not null ? obj._UnderlyingPtr : null), is_owning: true);
    }

    /// finds closest point on every object within given distance
    /// Generated from function `MR::projectOnAll`.
    /// Parameter `skipObjId` defaults to `{}`.
    public static unsafe void ProjectOnAll(MR.Const_Vector3f pt, MR.Const_AABBTreeObjects tree, float upDistLimitSq, MR.Std.Const_Function_VoidFuncFromMRObjIdMRMeshOrPointsProjectionResult callback, MR._InOpt_ObjId skipObjId = default)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_projectOnAll", ExactSpelling = true)]
        extern static void __MR_projectOnAll(MR.Const_Vector3f._Underlying *pt, MR.Const_AABBTreeObjects._Underlying *tree, float upDistLimitSq, MR.Std.Const_Function_VoidFuncFromMRObjIdMRMeshOrPointsProjectionResult._Underlying *callback, MR.ObjId *skipObjId);
        __MR_projectOnAll(pt._UnderlyingPtr, tree._UnderlyingPtr, upDistLimitSq, callback._UnderlyingPtr, skipObjId.HasValue ? &skipObjId.Object : null);
    }

    /// Projects a point onto an object, in world space. Returns `.valid() == false` if this object type isn't projectable onto.
    /// Generated from function `MR::projectWorldPointOntoObject`.
    public static unsafe MR.MeshOrPoints.ProjectionResult ProjectWorldPointOntoObject(MR.Const_Vector3f p, MR.Const_Object obj)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_projectWorldPointOntoObject", ExactSpelling = true)]
        extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_projectWorldPointOntoObject(MR.Const_Vector3f._Underlying *p, MR.Const_Object._Underlying *obj);
        return new(__MR_projectWorldPointOntoObject(p._UnderlyingPtr, obj._UnderlyingPtr), is_owning: true);
    }

    /// Recursively visits the objects and projects the point on each one. Returns the closest projection.
    /// If `root` is null, the scene root is used. Not passing `SceneRoot::get()` directly to avoid including that header.
    /// If `projectPred` is specified and false, will not project onto this object.
    /// If `recursePred` is specified and false, will not visit the children of this object.
    /// Generated from function `MR::projectWorldPointOntoObjectsRecursive`.
    /// Parameter `projectPred` defaults to `nullptr`.
    /// Parameter `recursePred` defaults to `nullptr`.
    public static unsafe MR.MeshOrPoints.ProjectionResult ProjectWorldPointOntoObjectsRecursive(MR.Const_Vector3f p, MR.Const_Object? root = null, MR.Std._ByValue_Function_BoolFuncFromConstMRObjectRef? projectPred = null, MR.Std._ByValue_Function_BoolFuncFromConstMRObjectRef? recursePred = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_projectWorldPointOntoObjectsRecursive", ExactSpelling = true)]
        extern static MR.MeshOrPoints.ProjectionResult._Underlying *__MR_projectWorldPointOntoObjectsRecursive(MR.Const_Vector3f._Underlying *p, MR.Const_Object._Underlying *root, MR.Misc._PassBy projectPred_pass_by, MR.Std.Function_BoolFuncFromConstMRObjectRef._Underlying *projectPred, MR.Misc._PassBy recursePred_pass_by, MR.Std.Function_BoolFuncFromConstMRObjectRef._Underlying *recursePred);
        return new(__MR_projectWorldPointOntoObjectsRecursive(p._UnderlyingPtr, root is not null ? root._UnderlyingPtr : null, projectPred is not null ? projectPred.PassByMode : MR.Misc._PassBy.default_arg, projectPred is not null && projectPred.Value is not null ? projectPred.Value._UnderlyingPtr : null, recursePred is not null ? recursePred.PassByMode : MR.Misc._PassBy.default_arg, recursePred is not null && recursePred.Value is not null ? recursePred.Value._UnderlyingPtr : null), is_owning: true);
    }
}
