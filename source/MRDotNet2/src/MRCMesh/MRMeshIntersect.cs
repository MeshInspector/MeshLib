public static partial class MR
{
    /// Generated from class `MR::MeshIntersectionResult`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MultiMeshIntersectionResult`
    /// This is the const half of the class.
    public class Const_MeshIntersectionResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshIntersectionResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshIntersectionResult_Destroy(_Underlying *_this);
            __MR_MeshIntersectionResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshIntersectionResult() {Dispose(false);}

        /// stores intersected face and global coordinates
        public unsafe MR.Const_PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_Get_proj", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MeshIntersectionResult_Get_proj(_Underlying *_this);
                return new(__MR_MeshIntersectionResult_Get_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores barycentric coordinates
        public unsafe MR.Const_MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_Get_mtp", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_MeshIntersectionResult_Get_mtp(_Underlying *_this);
                return new(__MR_MeshIntersectionResult_Get_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores the distance from ray origin to the intersection point in direction units
        public unsafe float DistanceAlongLine
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_Get_distanceAlongLine", ExactSpelling = true)]
                extern static float *__MR_MeshIntersectionResult_Get_distanceAlongLine(_Underlying *_this);
                return *__MR_MeshIntersectionResult_Get_distanceAlongLine(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshIntersectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshIntersectionResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshIntersectionResult` elementwise.
        public unsafe Const_MeshIntersectionResult(MR.Const_PointOnFace proj, MR.Const_MeshTriPoint mtp, float distanceAlongLine) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_ConstructFrom(MR.PointOnFace._Underlying *proj, MR.MeshTriPoint._Underlying *mtp, float distanceAlongLine);
            _UnderlyingPtr = __MR_MeshIntersectionResult_ConstructFrom(proj._UnderlyingPtr, mtp._UnderlyingPtr, distanceAlongLine);
        }

        /// Generated from constructor `MR::MeshIntersectionResult::MeshIntersectionResult`.
        public unsafe Const_MeshIntersectionResult(MR.Const_MeshIntersectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_ConstructFromAnother(MR.MeshIntersectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshIntersectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// check for validity
        /// Generated from conversion operator `MR::MeshIntersectionResult::operator bool`.
        public static unsafe explicit operator bool(MR.Const_MeshIntersectionResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_MeshIntersectionResult_ConvertTo_bool(MR.Const_MeshIntersectionResult._Underlying *_this);
            return __MR_MeshIntersectionResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::MeshIntersectionResult`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MultiMeshIntersectionResult`
    /// This is the non-const half of the class.
    public class MeshIntersectionResult : Const_MeshIntersectionResult
    {
        internal unsafe MeshIntersectionResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// stores intersected face and global coordinates
        public new unsafe MR.PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_GetMutable_proj", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MeshIntersectionResult_GetMutable_proj(_Underlying *_this);
                return new(__MR_MeshIntersectionResult_GetMutable_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores barycentric coordinates
        public new unsafe MR.MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_GetMutable_mtp", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_MeshIntersectionResult_GetMutable_mtp(_Underlying *_this);
                return new(__MR_MeshIntersectionResult_GetMutable_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores the distance from ray origin to the intersection point in direction units
        public new unsafe ref float DistanceAlongLine
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_GetMutable_distanceAlongLine", ExactSpelling = true)]
                extern static float *__MR_MeshIntersectionResult_GetMutable_distanceAlongLine(_Underlying *_this);
                return ref *__MR_MeshIntersectionResult_GetMutable_distanceAlongLine(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshIntersectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshIntersectionResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshIntersectionResult` elementwise.
        public unsafe MeshIntersectionResult(MR.Const_PointOnFace proj, MR.Const_MeshTriPoint mtp, float distanceAlongLine) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_ConstructFrom(MR.PointOnFace._Underlying *proj, MR.MeshTriPoint._Underlying *mtp, float distanceAlongLine);
            _UnderlyingPtr = __MR_MeshIntersectionResult_ConstructFrom(proj._UnderlyingPtr, mtp._UnderlyingPtr, distanceAlongLine);
        }

        /// Generated from constructor `MR::MeshIntersectionResult::MeshIntersectionResult`.
        public unsafe MeshIntersectionResult(MR.Const_MeshIntersectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_ConstructFromAnother(MR.MeshIntersectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshIntersectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshIntersectionResult::operator=`.
        public unsafe MR.MeshIntersectionResult Assign(MR.Const_MeshIntersectionResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshIntersectionResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MeshIntersectionResult_AssignFromAnother(_Underlying *_this, MR.MeshIntersectionResult._Underlying *_other);
            return new(__MR_MeshIntersectionResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshIntersectionResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshIntersectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshIntersectionResult`/`Const_MeshIntersectionResult` directly.
    public class _InOptMut_MeshIntersectionResult
    {
        public MeshIntersectionResult? Opt;

        public _InOptMut_MeshIntersectionResult() {}
        public _InOptMut_MeshIntersectionResult(MeshIntersectionResult value) {Opt = value;}
        public static implicit operator _InOptMut_MeshIntersectionResult(MeshIntersectionResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshIntersectionResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshIntersectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshIntersectionResult`/`Const_MeshIntersectionResult` to pass it to the function.
    public class _InOptConst_MeshIntersectionResult
    {
        public Const_MeshIntersectionResult? Opt;

        public _InOptConst_MeshIntersectionResult() {}
        public _InOptConst_MeshIntersectionResult(Const_MeshIntersectionResult value) {Opt = value;}
        public static implicit operator _InOptConst_MeshIntersectionResult(Const_MeshIntersectionResult value) {return new(value);}
    }

    /// Generated from class `MR::MultiRayMeshIntersectResult`.
    /// This is the const half of the class.
    public class Const_MultiRayMeshIntersectResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MultiRayMeshIntersectResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MultiRayMeshIntersectResult_Destroy(_Underlying *_this);
            __MR_MultiRayMeshIntersectResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MultiRayMeshIntersectResult() {Dispose(false);}

        ///< true if the ray has intersection with mesh part, false otherwise
        public unsafe ref void * IntersectingRays
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_Get_intersectingRays", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_Get_intersectingRays(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_Get_intersectingRays(_UnderlyingPtr);
            }
        }

        ///< distance along each ray till the intersection point or NaN if no intersection
        public unsafe ref void * RayDistances
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_Get_rayDistances", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_Get_rayDistances(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_Get_rayDistances(_UnderlyingPtr);
            }
        }

        ///< intersected triangles from mesh
        public unsafe ref void * IsectFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_Get_isectFaces", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_Get_isectFaces(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_Get_isectFaces(_UnderlyingPtr);
            }
        }

        ///< barycentric coordinates of the intersection point within intersected triangle or NaNs if no intersection
        public unsafe ref void * IsectBary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_Get_isectBary", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_Get_isectBary(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_Get_isectBary(_UnderlyingPtr);
            }
        }

        ///< intersection points or NaNs if no intersection
        public unsafe ref void * IsectPts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_Get_isectPts", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_Get_isectPts(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_Get_isectPts(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MultiRayMeshIntersectResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiRayMeshIntersectResult_DefaultConstruct();
        }

        /// Constructs `MR::MultiRayMeshIntersectResult` elementwise.
        public unsafe Const_MultiRayMeshIntersectResult(MR.BitSet? intersectingRays, MR.Std.Vector_Float? rayDistances, MR.Std.Vector_MRFaceId? isectFaces, MR.Std.Vector_MRTriPointf? isectBary, MR.Std.Vector_MRVector3f? isectPts) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_ConstructFrom(MR.BitSet._Underlying *intersectingRays, MR.Std.Vector_Float._Underlying *rayDistances, MR.Std.Vector_MRFaceId._Underlying *isectFaces, MR.Std.Vector_MRTriPointf._Underlying *isectBary, MR.Std.Vector_MRVector3f._Underlying *isectPts);
            _UnderlyingPtr = __MR_MultiRayMeshIntersectResult_ConstructFrom(intersectingRays is not null ? intersectingRays._UnderlyingPtr : null, rayDistances is not null ? rayDistances._UnderlyingPtr : null, isectFaces is not null ? isectFaces._UnderlyingPtr : null, isectBary is not null ? isectBary._UnderlyingPtr : null, isectPts is not null ? isectPts._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MultiRayMeshIntersectResult::MultiRayMeshIntersectResult`.
        public unsafe Const_MultiRayMeshIntersectResult(MR.Const_MultiRayMeshIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_ConstructFromAnother(MR.MultiRayMeshIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_MultiRayMeshIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MultiRayMeshIntersectResult`.
    /// This is the non-const half of the class.
    public class MultiRayMeshIntersectResult : Const_MultiRayMeshIntersectResult
    {
        internal unsafe MultiRayMeshIntersectResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< true if the ray has intersection with mesh part, false otherwise
        public new unsafe ref void * IntersectingRays
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_GetMutable_intersectingRays", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_GetMutable_intersectingRays(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_GetMutable_intersectingRays(_UnderlyingPtr);
            }
        }

        ///< distance along each ray till the intersection point or NaN if no intersection
        public new unsafe ref void * RayDistances
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_GetMutable_rayDistances", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_GetMutable_rayDistances(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_GetMutable_rayDistances(_UnderlyingPtr);
            }
        }

        ///< intersected triangles from mesh
        public new unsafe ref void * IsectFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_GetMutable_isectFaces", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_GetMutable_isectFaces(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_GetMutable_isectFaces(_UnderlyingPtr);
            }
        }

        ///< barycentric coordinates of the intersection point within intersected triangle or NaNs if no intersection
        public new unsafe ref void * IsectBary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_GetMutable_isectBary", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_GetMutable_isectBary(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_GetMutable_isectBary(_UnderlyingPtr);
            }
        }

        ///< intersection points or NaNs if no intersection
        public new unsafe ref void * IsectPts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_GetMutable_isectPts", ExactSpelling = true)]
                extern static void **__MR_MultiRayMeshIntersectResult_GetMutable_isectPts(_Underlying *_this);
                return ref *__MR_MultiRayMeshIntersectResult_GetMutable_isectPts(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MultiRayMeshIntersectResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiRayMeshIntersectResult_DefaultConstruct();
        }

        /// Constructs `MR::MultiRayMeshIntersectResult` elementwise.
        public unsafe MultiRayMeshIntersectResult(MR.BitSet? intersectingRays, MR.Std.Vector_Float? rayDistances, MR.Std.Vector_MRFaceId? isectFaces, MR.Std.Vector_MRTriPointf? isectBary, MR.Std.Vector_MRVector3f? isectPts) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_ConstructFrom(MR.BitSet._Underlying *intersectingRays, MR.Std.Vector_Float._Underlying *rayDistances, MR.Std.Vector_MRFaceId._Underlying *isectFaces, MR.Std.Vector_MRTriPointf._Underlying *isectBary, MR.Std.Vector_MRVector3f._Underlying *isectPts);
            _UnderlyingPtr = __MR_MultiRayMeshIntersectResult_ConstructFrom(intersectingRays is not null ? intersectingRays._UnderlyingPtr : null, rayDistances is not null ? rayDistances._UnderlyingPtr : null, isectFaces is not null ? isectFaces._UnderlyingPtr : null, isectBary is not null ? isectBary._UnderlyingPtr : null, isectPts is not null ? isectPts._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MultiRayMeshIntersectResult::MultiRayMeshIntersectResult`.
        public unsafe MultiRayMeshIntersectResult(MR.Const_MultiRayMeshIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_ConstructFromAnother(MR.MultiRayMeshIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_MultiRayMeshIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MultiRayMeshIntersectResult::operator=`.
        public unsafe MR.MultiRayMeshIntersectResult Assign(MR.Const_MultiRayMeshIntersectResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiRayMeshIntersectResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MultiRayMeshIntersectResult._Underlying *__MR_MultiRayMeshIntersectResult_AssignFromAnother(_Underlying *_this, MR.MultiRayMeshIntersectResult._Underlying *_other);
            return new(__MR_MultiRayMeshIntersectResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MultiRayMeshIntersectResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MultiRayMeshIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiRayMeshIntersectResult`/`Const_MultiRayMeshIntersectResult` directly.
    public class _InOptMut_MultiRayMeshIntersectResult
    {
        public MultiRayMeshIntersectResult? Opt;

        public _InOptMut_MultiRayMeshIntersectResult() {}
        public _InOptMut_MultiRayMeshIntersectResult(MultiRayMeshIntersectResult value) {Opt = value;}
        public static implicit operator _InOptMut_MultiRayMeshIntersectResult(MultiRayMeshIntersectResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MultiRayMeshIntersectResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MultiRayMeshIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiRayMeshIntersectResult`/`Const_MultiRayMeshIntersectResult` to pass it to the function.
    public class _InOptConst_MultiRayMeshIntersectResult
    {
        public Const_MultiRayMeshIntersectResult? Opt;

        public _InOptConst_MultiRayMeshIntersectResult() {}
        public _InOptConst_MultiRayMeshIntersectResult(Const_MultiRayMeshIntersectResult value) {Opt = value;}
        public static implicit operator _InOptConst_MultiRayMeshIntersectResult(Const_MultiRayMeshIntersectResult value) {return new(value);}
    }

    /// Generated from class `MR::MultiMeshIntersectionResult`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshIntersectionResult`
    /// This is the const half of the class.
    public class Const_MultiMeshIntersectionResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MultiMeshIntersectionResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MultiMeshIntersectionResult_Destroy(_Underlying *_this);
            __MR_MultiMeshIntersectionResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MultiMeshIntersectionResult() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_MeshIntersectionResult(Const_MultiMeshIntersectionResult self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_UpcastTo_MR_MeshIntersectionResult", ExactSpelling = true)]
            extern static MR.Const_MeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_UpcastTo_MR_MeshIntersectionResult(_Underlying *_this);
            MR.Const_MeshIntersectionResult ret = new(__MR_MultiMeshIntersectionResult_UpcastTo_MR_MeshIntersectionResult(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// the intersection found in this mesh
        public unsafe ref readonly void * Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_Get_mesh", ExactSpelling = true)]
                extern static void **__MR_MultiMeshIntersectionResult_Get_mesh(_Underlying *_this);
                return ref *__MR_MultiMeshIntersectionResult_Get_mesh(_UnderlyingPtr);
            }
        }

        /// stores intersected face and global coordinates
        public unsafe MR.Const_PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_Get_proj", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MultiMeshIntersectionResult_Get_proj(_Underlying *_this);
                return new(__MR_MultiMeshIntersectionResult_Get_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores barycentric coordinates
        public unsafe MR.Const_MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_Get_mtp", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_MultiMeshIntersectionResult_Get_mtp(_Underlying *_this);
                return new(__MR_MultiMeshIntersectionResult_Get_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores the distance from ray origin to the intersection point in direction units
        public unsafe float DistanceAlongLine
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_Get_distanceAlongLine", ExactSpelling = true)]
                extern static float *__MR_MultiMeshIntersectionResult_Get_distanceAlongLine(_Underlying *_this);
                return *__MR_MultiMeshIntersectionResult_Get_distanceAlongLine(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MultiMeshIntersectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiMeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiMeshIntersectionResult_DefaultConstruct();
        }

        /// Generated from constructor `MR::MultiMeshIntersectionResult::MultiMeshIntersectionResult`.
        public unsafe Const_MultiMeshIntersectionResult(MR.Const_MultiMeshIntersectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiMeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_ConstructFromAnother(MR.MultiMeshIntersectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_MultiMeshIntersectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// check for validity
        /// Generated from conversion operator `MR::MultiMeshIntersectionResult::operator bool`.
        public static unsafe explicit operator bool(MR.Const_MultiMeshIntersectionResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_MultiMeshIntersectionResult_ConvertTo_bool(MR.Const_MultiMeshIntersectionResult._Underlying *_this);
            return __MR_MultiMeshIntersectionResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::MultiMeshIntersectionResult`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshIntersectionResult`
    /// This is the non-const half of the class.
    public class MultiMeshIntersectionResult : Const_MultiMeshIntersectionResult
    {
        internal unsafe MultiMeshIntersectionResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.MeshIntersectionResult(MultiMeshIntersectionResult self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_UpcastTo_MR_MeshIntersectionResult", ExactSpelling = true)]
            extern static MR.MeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_UpcastTo_MR_MeshIntersectionResult(_Underlying *_this);
            MR.MeshIntersectionResult ret = new(__MR_MultiMeshIntersectionResult_UpcastTo_MR_MeshIntersectionResult(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// the intersection found in this mesh
        public new unsafe ref readonly void * Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_GetMutable_mesh", ExactSpelling = true)]
                extern static void **__MR_MultiMeshIntersectionResult_GetMutable_mesh(_Underlying *_this);
                return ref *__MR_MultiMeshIntersectionResult_GetMutable_mesh(_UnderlyingPtr);
            }
        }

        /// stores intersected face and global coordinates
        public new unsafe MR.PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_GetMutable_proj", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MultiMeshIntersectionResult_GetMutable_proj(_Underlying *_this);
                return new(__MR_MultiMeshIntersectionResult_GetMutable_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores barycentric coordinates
        public new unsafe MR.MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_GetMutable_mtp", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_MultiMeshIntersectionResult_GetMutable_mtp(_Underlying *_this);
                return new(__MR_MultiMeshIntersectionResult_GetMutable_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores the distance from ray origin to the intersection point in direction units
        public new unsafe ref float DistanceAlongLine
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_GetMutable_distanceAlongLine", ExactSpelling = true)]
                extern static float *__MR_MultiMeshIntersectionResult_GetMutable_distanceAlongLine(_Underlying *_this);
                return ref *__MR_MultiMeshIntersectionResult_GetMutable_distanceAlongLine(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MultiMeshIntersectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiMeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiMeshIntersectionResult_DefaultConstruct();
        }

        /// Generated from constructor `MR::MultiMeshIntersectionResult::MultiMeshIntersectionResult`.
        public unsafe MultiMeshIntersectionResult(MR.Const_MultiMeshIntersectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiMeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_ConstructFromAnother(MR.MultiMeshIntersectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_MultiMeshIntersectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MultiMeshIntersectionResult::operator=`.
        public unsafe MR.MultiMeshIntersectionResult Assign(MR.Const_MultiMeshIntersectionResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiMeshIntersectionResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MultiMeshIntersectionResult._Underlying *__MR_MultiMeshIntersectionResult_AssignFromAnother(_Underlying *_this, MR.MultiMeshIntersectionResult._Underlying *_other);
            return new(__MR_MultiMeshIntersectionResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MultiMeshIntersectionResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MultiMeshIntersectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiMeshIntersectionResult`/`Const_MultiMeshIntersectionResult` directly.
    public class _InOptMut_MultiMeshIntersectionResult
    {
        public MultiMeshIntersectionResult? Opt;

        public _InOptMut_MultiMeshIntersectionResult() {}
        public _InOptMut_MultiMeshIntersectionResult(MultiMeshIntersectionResult value) {Opt = value;}
        public static implicit operator _InOptMut_MultiMeshIntersectionResult(MultiMeshIntersectionResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MultiMeshIntersectionResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MultiMeshIntersectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiMeshIntersectionResult`/`Const_MultiMeshIntersectionResult` to pass it to the function.
    public class _InOptConst_MultiMeshIntersectionResult
    {
        public Const_MultiMeshIntersectionResult? Opt;

        public _InOptConst_MultiMeshIntersectionResult() {}
        public _InOptConst_MultiMeshIntersectionResult(Const_MultiMeshIntersectionResult value) {Opt = value;}
        public static implicit operator _InOptConst_MultiMeshIntersectionResult(Const_MultiMeshIntersectionResult value) {return new(value);}
    }

    /// Generated from class `MR::Line3Mesh<float>`.
    /// This is the const half of the class.
    public class Const_Line3Mesh_Float : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Line3Mesh_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_Destroy", ExactSpelling = true)]
            extern static void __MR_Line3Mesh_float_Destroy(_Underlying *_this);
            __MR_Line3Mesh_float_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Line3Mesh_Float() {Dispose(false);}

        ///< in the reference frame of mesh
        public unsafe MR.Const_Line3f Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_Get_line", ExactSpelling = true)]
                extern static MR.Const_Line3f._Underlying *__MR_Line3Mesh_float_Get_line(_Underlying *_this);
                return new(__MR_Line3Mesh_float_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< set it to a valid pointer for better performance
        public unsafe ref void * Prec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_Get_prec", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_Get_prec(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_Get_prec(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public unsafe ref readonly void * Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_Get_mesh", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_Get_mesh(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_Get_mesh(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public unsafe ref readonly void * Tree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_Get_tree", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_Get_tree(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_Get_tree(_UnderlyingPtr);
            }
        }

        ///< may remain nullptr, meaning all mesh
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_Get_region", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_Get_region(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_Get_region(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Line3Mesh_Float() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3Mesh_float_DefaultConstruct();
        }

        /// Constructs `MR::Line3Mesh<float>` elementwise.
        public unsafe Const_Line3Mesh_Float(MR.Const_Line3f line, MR.IntersectionPrecomputes_Float? prec, MR.Const_Mesh? mesh, MR.Const_AABBTree? tree, MR.Const_FaceBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_ConstructFrom", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_ConstructFrom(MR.Line3f._Underlying *line, MR.IntersectionPrecomputes_Float._Underlying *prec, MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTree._Underlying *tree, MR.Const_FaceBitSet._Underlying *region);
            _UnderlyingPtr = __MR_Line3Mesh_float_ConstructFrom(line._UnderlyingPtr, prec is not null ? prec._UnderlyingPtr : null, mesh is not null ? mesh._UnderlyingPtr : null, tree is not null ? tree._UnderlyingPtr : null, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Line3Mesh<float>::Line3Mesh`.
        public unsafe Const_Line3Mesh_Float(MR.Const_Line3Mesh_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_ConstructFromAnother(MR.Line3Mesh_Float._Underlying *_other);
            _UnderlyingPtr = __MR_Line3Mesh_float_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::Line3Mesh<float>`.
    /// This is the non-const half of the class.
    public class Line3Mesh_Float : Const_Line3Mesh_Float
    {
        internal unsafe Line3Mesh_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< in the reference frame of mesh
        public new unsafe MR.Line3f Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_GetMutable_line", ExactSpelling = true)]
                extern static MR.Line3f._Underlying *__MR_Line3Mesh_float_GetMutable_line(_Underlying *_this);
                return new(__MR_Line3Mesh_float_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< set it to a valid pointer for better performance
        public new unsafe ref void * Prec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_GetMutable_prec", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_GetMutable_prec(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_GetMutable_prec(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public new unsafe ref readonly void * Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_GetMutable_mesh", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_GetMutable_mesh(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_GetMutable_mesh(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public new unsafe ref readonly void * Tree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_GetMutable_tree", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_GetMutable_tree(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_GetMutable_tree(_UnderlyingPtr);
            }
        }

        ///< may remain nullptr, meaning all mesh
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_float_GetMutable_region(_Underlying *_this);
                return ref *__MR_Line3Mesh_float_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Line3Mesh_Float() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3Mesh_float_DefaultConstruct();
        }

        /// Constructs `MR::Line3Mesh<float>` elementwise.
        public unsafe Line3Mesh_Float(MR.Const_Line3f line, MR.IntersectionPrecomputes_Float? prec, MR.Const_Mesh? mesh, MR.Const_AABBTree? tree, MR.Const_FaceBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_ConstructFrom", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_ConstructFrom(MR.Line3f._Underlying *line, MR.IntersectionPrecomputes_Float._Underlying *prec, MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTree._Underlying *tree, MR.Const_FaceBitSet._Underlying *region);
            _UnderlyingPtr = __MR_Line3Mesh_float_ConstructFrom(line._UnderlyingPtr, prec is not null ? prec._UnderlyingPtr : null, mesh is not null ? mesh._UnderlyingPtr : null, tree is not null ? tree._UnderlyingPtr : null, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Line3Mesh<float>::Line3Mesh`.
        public unsafe Line3Mesh_Float(MR.Const_Line3Mesh_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_ConstructFromAnother(MR.Line3Mesh_Float._Underlying *_other);
            _UnderlyingPtr = __MR_Line3Mesh_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Line3Mesh<float>::operator=`.
        public unsafe MR.Line3Mesh_Float Assign(MR.Const_Line3Mesh_Float _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Line3Mesh_Float._Underlying *__MR_Line3Mesh_float_AssignFromAnother(_Underlying *_this, MR.Line3Mesh_Float._Underlying *_other);
            return new(__MR_Line3Mesh_float_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Line3Mesh_Float` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Line3Mesh_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3Mesh_Float`/`Const_Line3Mesh_Float` directly.
    public class _InOptMut_Line3Mesh_Float
    {
        public Line3Mesh_Float? Opt;

        public _InOptMut_Line3Mesh_Float() {}
        public _InOptMut_Line3Mesh_Float(Line3Mesh_Float value) {Opt = value;}
        public static implicit operator _InOptMut_Line3Mesh_Float(Line3Mesh_Float value) {return new(value);}
    }

    /// This is used for optional parameters of class `Line3Mesh_Float` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Line3Mesh_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3Mesh_Float`/`Const_Line3Mesh_Float` to pass it to the function.
    public class _InOptConst_Line3Mesh_Float
    {
        public Const_Line3Mesh_Float? Opt;

        public _InOptConst_Line3Mesh_Float() {}
        public _InOptConst_Line3Mesh_Float(Const_Line3Mesh_Float value) {Opt = value;}
        public static implicit operator _InOptConst_Line3Mesh_Float(Const_Line3Mesh_Float value) {return new(value);}
    }

    /// Generated from class `MR::Line3Mesh<double>`.
    /// This is the const half of the class.
    public class Const_Line3Mesh_Double : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Line3Mesh_Double(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_Destroy", ExactSpelling = true)]
            extern static void __MR_Line3Mesh_double_Destroy(_Underlying *_this);
            __MR_Line3Mesh_double_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Line3Mesh_Double() {Dispose(false);}

        ///< in the reference frame of mesh
        public unsafe MR.Const_Line3d Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_Get_line", ExactSpelling = true)]
                extern static MR.Const_Line3d._Underlying *__MR_Line3Mesh_double_Get_line(_Underlying *_this);
                return new(__MR_Line3Mesh_double_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< set it to a valid pointer for better performance
        public unsafe ref void * Prec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_Get_prec", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_Get_prec(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_Get_prec(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public unsafe ref readonly void * Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_Get_mesh", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_Get_mesh(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_Get_mesh(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public unsafe ref readonly void * Tree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_Get_tree", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_Get_tree(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_Get_tree(_UnderlyingPtr);
            }
        }

        ///< may remain nullptr, meaning all mesh
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_Get_region", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_Get_region(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_Get_region(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Line3Mesh_Double() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3Mesh_double_DefaultConstruct();
        }

        /// Constructs `MR::Line3Mesh<double>` elementwise.
        public unsafe Const_Line3Mesh_Double(MR.Const_Line3d line, MR.IntersectionPrecomputes_Double? prec, MR.Const_Mesh? mesh, MR.Const_AABBTree? tree, MR.Const_FaceBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_ConstructFrom", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_ConstructFrom(MR.Line3d._Underlying *line, MR.IntersectionPrecomputes_Double._Underlying *prec, MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTree._Underlying *tree, MR.Const_FaceBitSet._Underlying *region);
            _UnderlyingPtr = __MR_Line3Mesh_double_ConstructFrom(line._UnderlyingPtr, prec is not null ? prec._UnderlyingPtr : null, mesh is not null ? mesh._UnderlyingPtr : null, tree is not null ? tree._UnderlyingPtr : null, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Line3Mesh<double>::Line3Mesh`.
        public unsafe Const_Line3Mesh_Double(MR.Const_Line3Mesh_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_ConstructFromAnother(MR.Line3Mesh_Double._Underlying *_other);
            _UnderlyingPtr = __MR_Line3Mesh_double_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::Line3Mesh<double>`.
    /// This is the non-const half of the class.
    public class Line3Mesh_Double : Const_Line3Mesh_Double
    {
        internal unsafe Line3Mesh_Double(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< in the reference frame of mesh
        public new unsafe MR.Line3d Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_GetMutable_line", ExactSpelling = true)]
                extern static MR.Line3d._Underlying *__MR_Line3Mesh_double_GetMutable_line(_Underlying *_this);
                return new(__MR_Line3Mesh_double_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< set it to a valid pointer for better performance
        public new unsafe ref void * Prec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_GetMutable_prec", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_GetMutable_prec(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_GetMutable_prec(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public new unsafe ref readonly void * Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_GetMutable_mesh", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_GetMutable_mesh(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_GetMutable_mesh(_UnderlyingPtr);
            }
        }

        ///< must be set a valid pointer before use
        public new unsafe ref readonly void * Tree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_GetMutable_tree", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_GetMutable_tree(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_GetMutable_tree(_UnderlyingPtr);
            }
        }

        ///< may remain nullptr, meaning all mesh
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_Line3Mesh_double_GetMutable_region(_Underlying *_this);
                return ref *__MR_Line3Mesh_double_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Line3Mesh_Double() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3Mesh_double_DefaultConstruct();
        }

        /// Constructs `MR::Line3Mesh<double>` elementwise.
        public unsafe Line3Mesh_Double(MR.Const_Line3d line, MR.IntersectionPrecomputes_Double? prec, MR.Const_Mesh? mesh, MR.Const_AABBTree? tree, MR.Const_FaceBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_ConstructFrom", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_ConstructFrom(MR.Line3d._Underlying *line, MR.IntersectionPrecomputes_Double._Underlying *prec, MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTree._Underlying *tree, MR.Const_FaceBitSet._Underlying *region);
            _UnderlyingPtr = __MR_Line3Mesh_double_ConstructFrom(line._UnderlyingPtr, prec is not null ? prec._UnderlyingPtr : null, mesh is not null ? mesh._UnderlyingPtr : null, tree is not null ? tree._UnderlyingPtr : null, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Line3Mesh<double>::Line3Mesh`.
        public unsafe Line3Mesh_Double(MR.Const_Line3Mesh_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_ConstructFromAnother(MR.Line3Mesh_Double._Underlying *_other);
            _UnderlyingPtr = __MR_Line3Mesh_double_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Line3Mesh<double>::operator=`.
        public unsafe MR.Line3Mesh_Double Assign(MR.Const_Line3Mesh_Double _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3Mesh_double_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Line3Mesh_Double._Underlying *__MR_Line3Mesh_double_AssignFromAnother(_Underlying *_this, MR.Line3Mesh_Double._Underlying *_other);
            return new(__MR_Line3Mesh_double_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Line3Mesh_Double` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Line3Mesh_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3Mesh_Double`/`Const_Line3Mesh_Double` directly.
    public class _InOptMut_Line3Mesh_Double
    {
        public Line3Mesh_Double? Opt;

        public _InOptMut_Line3Mesh_Double() {}
        public _InOptMut_Line3Mesh_Double(Line3Mesh_Double value) {Opt = value;}
        public static implicit operator _InOptMut_Line3Mesh_Double(Line3Mesh_Double value) {return new(value);}
    }

    /// This is used for optional parameters of class `Line3Mesh_Double` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Line3Mesh_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3Mesh_Double`/`Const_Line3Mesh_Double` to pass it to the function.
    public class _InOptConst_Line3Mesh_Double
    {
        public Const_Line3Mesh_Double? Opt;

        public _InOptConst_Line3Mesh_Double() {}
        public _InOptConst_Line3Mesh_Double(Const_Line3Mesh_Double value) {Opt = value;}
        public static implicit operator _InOptConst_Line3Mesh_Double(Const_Line3Mesh_Double value) {return new(value);}
    }

    /// Finds ray and mesh intersection in float-precision.
    /// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
    /// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
    /// \p vadidFaces if given then all faces for which false is returned will be skipped
    /// Finds the closest intersection to ray origin (line param=0)
    /// or any intersection for better performance if \p !closestIntersect.
    /// Generated from function `MR::rayMeshIntersect`.
    /// Parameter `rayStart` defaults to `0.0f`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    /// Parameter `closestIntersect` defaults to `true`.
    /// Parameter `validFaces` defaults to `{}`.
    public static unsafe MR.MeshIntersectionResult RayMeshIntersect(MR.Const_MeshPart meshPart, MR.Const_Line3f line, float? rayStart = null, float? rayEnd = null, MR.Const_IntersectionPrecomputes_Float? prec = null, bool? closestIntersect = null, MR.Std.Const_Function_BoolFuncFromMRFaceId? validFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayMeshIntersect_MR_Line3f", ExactSpelling = true)]
        extern static MR.MeshIntersectionResult._Underlying *__MR_rayMeshIntersect_MR_Line3f(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Line3f._Underlying *line, float *rayStart, float *rayEnd, MR.Const_IntersectionPrecomputes_Float._Underlying *prec, byte *closestIntersect, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *validFaces);
        float __deref_rayStart = rayStart.GetValueOrDefault();
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        byte __deref_closestIntersect = closestIntersect.GetValueOrDefault() ? (byte)1 : (byte)0;
        return new(__MR_rayMeshIntersect_MR_Line3f(meshPart._UnderlyingPtr, line._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null, closestIntersect.HasValue ? &__deref_closestIntersect : null, validFaces is not null ? validFaces._UnderlyingPtr : null), is_owning: true);
    }

    /// Finds ray and mesh intersection in double-precision.
    /// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
    /// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
    /// \p vadidFaces if given then all faces for which false is returned will be skipped
    /// Finds the closest intersection to ray origin (line param=0)
    /// or any intersection for better performance if \p !closestIntersect.
    /// Generated from function `MR::rayMeshIntersect`.
    /// Parameter `rayStart` defaults to `0.0`.
    /// Parameter `rayEnd` defaults to `1.7976931348623157e308`.
    /// Parameter `closestIntersect` defaults to `true`.
    /// Parameter `validFaces` defaults to `{}`.
    public static unsafe MR.MeshIntersectionResult RayMeshIntersect(MR.Const_MeshPart meshPart, MR.Const_Line3d line, double? rayStart = null, double? rayEnd = null, MR.Const_IntersectionPrecomputes_Double? prec = null, bool? closestIntersect = null, MR.Std.Const_Function_BoolFuncFromMRFaceId? validFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayMeshIntersect_MR_Line3d", ExactSpelling = true)]
        extern static MR.MeshIntersectionResult._Underlying *__MR_rayMeshIntersect_MR_Line3d(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Line3d._Underlying *line, double *rayStart, double *rayEnd, MR.Const_IntersectionPrecomputes_Double._Underlying *prec, byte *closestIntersect, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *validFaces);
        double __deref_rayStart = rayStart.GetValueOrDefault();
        double __deref_rayEnd = rayEnd.GetValueOrDefault();
        byte __deref_closestIntersect = closestIntersect.GetValueOrDefault() ? (byte)1 : (byte)0;
        return new(__MR_rayMeshIntersect_MR_Line3d(meshPart._UnderlyingPtr, line._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null, closestIntersect.HasValue ? &__deref_closestIntersect : null, validFaces is not null ? validFaces._UnderlyingPtr : null), is_owning: true);
    }

    /// Finds intersections between a mesh and multiple rays in parallel (in float-precision).
    /// \p rayStart and \p rayEnd define the interval on all rays to detect an intersection.
    /// \p vadidFaces if given then all faces for which false is returned will be skipped
    /// Generated from function `MR::multiRayMeshIntersect`.
    /// Parameter `rayStart` defaults to `0.0f`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    /// Parameter `closestIntersect` defaults to `true`.
    /// Parameter `validFaces` defaults to `{}`.
    public static unsafe void MultiRayMeshIntersect(MR.Const_MeshPart meshPart, MR.Std.Const_Vector_MRVector3f origins, MR.Std.Const_Vector_MRVector3f dirs, MR.Const_MultiRayMeshIntersectResult result, float? rayStart = null, float? rayEnd = null, bool? closestIntersect = null, MR.Std.Const_Function_BoolFuncFromMRFaceId? validFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_multiRayMeshIntersect", ExactSpelling = true)]
        extern static void __MR_multiRayMeshIntersect(MR.Const_MeshPart._Underlying *meshPart, MR.Std.Const_Vector_MRVector3f._Underlying *origins, MR.Std.Const_Vector_MRVector3f._Underlying *dirs, MR.Const_MultiRayMeshIntersectResult._Underlying *result, float *rayStart, float *rayEnd, byte *closestIntersect, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *validFaces);
        float __deref_rayStart = rayStart.GetValueOrDefault();
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        byte __deref_closestIntersect = closestIntersect.GetValueOrDefault() ? (byte)1 : (byte)0;
        __MR_multiRayMeshIntersect(meshPart._UnderlyingPtr, origins._UnderlyingPtr, dirs._UnderlyingPtr, result._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, closestIntersect.HasValue ? &__deref_closestIntersect : null, validFaces is not null ? validFaces._UnderlyingPtr : null);
    }

    /// Intersects ray with many meshes. Finds any intersection (not the closest)
    /// \anchor rayMultiMeshAnyIntersectF
    /// Generated from function `MR::rayMultiMeshAnyIntersect`.
    /// Parameter `rayStart` defaults to `0.0f`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    public static unsafe MR.MultiMeshIntersectionResult RayMultiMeshAnyIntersect(MR.Std.Const_Vector_MRLine3MeshFloat lineMeshes, float? rayStart = null, float? rayEnd = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayMultiMeshAnyIntersect_std_vector_MR_Line3Mesh_float", ExactSpelling = true)]
        extern static MR.MultiMeshIntersectionResult._Underlying *__MR_rayMultiMeshAnyIntersect_std_vector_MR_Line3Mesh_float(MR.Std.Const_Vector_MRLine3MeshFloat._Underlying *lineMeshes, float *rayStart, float *rayEnd);
        float __deref_rayStart = rayStart.GetValueOrDefault();
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        return new(__MR_rayMultiMeshAnyIntersect_std_vector_MR_Line3Mesh_float(lineMeshes._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null), is_owning: true);
    }

    /// Same as \ref rayMultiMeshAnyIntersectF, but use double precision
    /// Generated from function `MR::rayMultiMeshAnyIntersect`.
    /// Parameter `rayStart` defaults to `0.0`.
    /// Parameter `rayEnd` defaults to `1.7976931348623157e308`.
    public static unsafe MR.MultiMeshIntersectionResult RayMultiMeshAnyIntersect(MR.Std.Const_Vector_MRLine3MeshDouble lineMeshes, double? rayStart = null, double? rayEnd = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayMultiMeshAnyIntersect_std_vector_MR_Line3Mesh_double", ExactSpelling = true)]
        extern static MR.MultiMeshIntersectionResult._Underlying *__MR_rayMultiMeshAnyIntersect_std_vector_MR_Line3Mesh_double(MR.Std.Const_Vector_MRLine3MeshDouble._Underlying *lineMeshes, double *rayStart, double *rayEnd);
        double __deref_rayStart = rayStart.GetValueOrDefault();
        double __deref_rayEnd = rayEnd.GetValueOrDefault();
        return new(__MR_rayMultiMeshAnyIntersect_std_vector_MR_Line3Mesh_double(lineMeshes._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null), is_owning: true);
    }

    /// Intersects ray with mesh. Finds all intersections
    /// \anchor rayMeshIntersectAllF
    /// Generated from function `MR::rayMeshIntersectAll`.
    /// Parameter `rayStart` defaults to `0.0f`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    public static unsafe void RayMeshIntersectAll(MR.Const_MeshPart meshPart, MR.Const_Line3f line, MR.Std._ByValue_Function_BoolFuncFromConstMRMeshIntersectionResultRef callback, float? rayStart = null, float? rayEnd = null, MR.Const_IntersectionPrecomputes_Float? prec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayMeshIntersectAll_MR_Line3f", ExactSpelling = true)]
        extern static void __MR_rayMeshIntersectAll_MR_Line3f(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Line3f._Underlying *line, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromConstMRMeshIntersectionResultRef._Underlying *callback, float *rayStart, float *rayEnd, MR.Const_IntersectionPrecomputes_Float._Underlying *prec);
        float __deref_rayStart = rayStart.GetValueOrDefault();
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        __MR_rayMeshIntersectAll_MR_Line3f(meshPart._UnderlyingPtr, line._UnderlyingPtr, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null);
    }

    /// Same as \ref rayMeshIntersectAllF, but use double precision
    /// Generated from function `MR::rayMeshIntersectAll`.
    /// Parameter `rayStart` defaults to `0.0`.
    /// Parameter `rayEnd` defaults to `1.7976931348623157e308`.
    public static unsafe void RayMeshIntersectAll(MR.Const_MeshPart meshPart, MR.Const_Line3d line, MR.Std._ByValue_Function_BoolFuncFromConstMRMeshIntersectionResultRef callback, double? rayStart = null, double? rayEnd = null, MR.Const_IntersectionPrecomputes_Double? prec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayMeshIntersectAll_MR_Line3d", ExactSpelling = true)]
        extern static void __MR_rayMeshIntersectAll_MR_Line3d(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Line3d._Underlying *line, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromConstMRMeshIntersectionResultRef._Underlying *callback, double *rayStart, double *rayEnd, MR.Const_IntersectionPrecomputes_Double._Underlying *prec);
        double __deref_rayStart = rayStart.GetValueOrDefault();
        double __deref_rayEnd = rayEnd.GetValueOrDefault();
        __MR_rayMeshIntersectAll_MR_Line3d(meshPart._UnderlyingPtr, line._UnderlyingPtr, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null);
    }

    /// given mesh part and arbitrary plane, outputs
    /// \param fs  triangles from boxes crossed or touched by the plane
    /// \param ues edges of these triangles
    /// \param vs  vertices of these triangles
    /// \param fsVec triangles from boxes crossed or touched by the plane in unspecified order
    /// Generated from function `MR::planeMeshIntersect`.
    public static unsafe void PlaneMeshIntersect(MR.Const_MeshPart meshPart, MR.Const_Plane3f plane, MR.FaceBitSet? fs, MR.UndirectedEdgeBitSet? ues, MR.VertBitSet? vs, MR.Std.Vector_MRFaceId? fsVec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_planeMeshIntersect", ExactSpelling = true)]
        extern static void __MR_planeMeshIntersect(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Plane3f._Underlying *plane, MR.FaceBitSet._Underlying *fs, MR.UndirectedEdgeBitSet._Underlying *ues, MR.VertBitSet._Underlying *vs, MR.Std.Vector_MRFaceId._Underlying *fsVec);
        __MR_planeMeshIntersect(meshPart._UnderlyingPtr, plane._UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null, ues is not null ? ues._UnderlyingPtr : null, vs is not null ? vs._UnderlyingPtr : null, fsVec is not null ? fsVec._UnderlyingPtr : null);
    }

    /// given mesh part and plane z=zLevel, outputs
    /// \param fs  triangles crossed or touched by the plane
    /// \param ues edges of these triangles
    /// \param vs  vertices of these triangles
    /// \param fsVec triangles crossed or touched by the plane in unspecified order
    /// Generated from function `MR::xyPlaneMeshIntersect`.
    public static unsafe void XyPlaneMeshIntersect(MR.Const_MeshPart meshPart, float zLevel, MR.FaceBitSet? fs, MR.UndirectedEdgeBitSet? ues, MR.VertBitSet? vs, MR.Std.Vector_MRFaceId? fsVec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xyPlaneMeshIntersect", ExactSpelling = true)]
        extern static void __MR_xyPlaneMeshIntersect(MR.Const_MeshPart._Underlying *meshPart, float zLevel, MR.FaceBitSet._Underlying *fs, MR.UndirectedEdgeBitSet._Underlying *ues, MR.VertBitSet._Underlying *vs, MR.Std.Vector_MRFaceId._Underlying *fsVec);
        __MR_xyPlaneMeshIntersect(meshPart._UnderlyingPtr, zLevel, fs is not null ? fs._UnderlyingPtr : null, ues is not null ? ues._UnderlyingPtr : null, vs is not null ? vs._UnderlyingPtr : null, fsVec is not null ? fsVec._UnderlyingPtr : null);
    }
}
