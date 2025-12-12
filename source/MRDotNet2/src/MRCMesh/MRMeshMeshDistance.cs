public static partial class MR
{
    /// Generated from class `MR::MeshMeshDistanceResult`.
    /// This is the const half of the class.
    public class Const_MeshMeshDistanceResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshMeshDistanceResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshMeshDistanceResult_Destroy(_Underlying *_this);
            __MR_MeshMeshDistanceResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshMeshDistanceResult() {Dispose(false);}

        /// two closest points: from meshes A and B respectively
        public unsafe MR.Const_PointOnFace A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_Get_a", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MeshMeshDistanceResult_Get_a(_Underlying *_this);
                return new(__MR_MeshMeshDistanceResult_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// two closest points: from meshes A and B respectively
        public unsafe MR.Const_PointOnFace B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_Get_b", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MeshMeshDistanceResult_Get_b(_Underlying *_this);
                return new(__MR_MeshMeshDistanceResult_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between a and b
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_MeshMeshDistanceResult_Get_distSq(_Underlying *_this);
                return *__MR_MeshMeshDistanceResult_Get_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshMeshDistanceResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshMeshDistanceResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshMeshDistanceResult` elementwise.
        public unsafe Const_MeshMeshDistanceResult(MR.Const_PointOnFace a, MR.Const_PointOnFace b, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_ConstructFrom(MR.PointOnFace._Underlying *a, MR.PointOnFace._Underlying *b, float distSq);
            _UnderlyingPtr = __MR_MeshMeshDistanceResult_ConstructFrom(a._UnderlyingPtr, b._UnderlyingPtr, distSq);
        }

        /// Generated from constructor `MR::MeshMeshDistanceResult::MeshMeshDistanceResult`.
        public unsafe Const_MeshMeshDistanceResult(MR.Const_MeshMeshDistanceResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_ConstructFromAnother(MR.MeshMeshDistanceResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshMeshDistanceResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshMeshDistanceResult`.
    /// This is the non-const half of the class.
    public class MeshMeshDistanceResult : Const_MeshMeshDistanceResult
    {
        internal unsafe MeshMeshDistanceResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// two closest points: from meshes A and B respectively
        public new unsafe MR.PointOnFace A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_GetMutable_a", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MeshMeshDistanceResult_GetMutable_a(_Underlying *_this);
                return new(__MR_MeshMeshDistanceResult_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// two closest points: from meshes A and B respectively
        public new unsafe MR.PointOnFace B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_GetMutable_b", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MeshMeshDistanceResult_GetMutable_b(_Underlying *_this);
                return new(__MR_MeshMeshDistanceResult_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between a and b
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_MeshMeshDistanceResult_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_MeshMeshDistanceResult_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshMeshDistanceResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshMeshDistanceResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshMeshDistanceResult` elementwise.
        public unsafe MeshMeshDistanceResult(MR.Const_PointOnFace a, MR.Const_PointOnFace b, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_ConstructFrom(MR.PointOnFace._Underlying *a, MR.PointOnFace._Underlying *b, float distSq);
            _UnderlyingPtr = __MR_MeshMeshDistanceResult_ConstructFrom(a._UnderlyingPtr, b._UnderlyingPtr, distSq);
        }

        /// Generated from constructor `MR::MeshMeshDistanceResult::MeshMeshDistanceResult`.
        public unsafe MeshMeshDistanceResult(MR.Const_MeshMeshDistanceResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_ConstructFromAnother(MR.MeshMeshDistanceResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshMeshDistanceResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshMeshDistanceResult::operator=`.
        public unsafe MR.MeshMeshDistanceResult Assign(MR.Const_MeshMeshDistanceResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshDistanceResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshDistanceResult._Underlying *__MR_MeshMeshDistanceResult_AssignFromAnother(_Underlying *_this, MR.MeshMeshDistanceResult._Underlying *_other);
            return new(__MR_MeshMeshDistanceResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshMeshDistanceResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshMeshDistanceResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshMeshDistanceResult`/`Const_MeshMeshDistanceResult` directly.
    public class _InOptMut_MeshMeshDistanceResult
    {
        public MeshMeshDistanceResult? Opt;

        public _InOptMut_MeshMeshDistanceResult() {}
        public _InOptMut_MeshMeshDistanceResult(MeshMeshDistanceResult value) {Opt = value;}
        public static implicit operator _InOptMut_MeshMeshDistanceResult(MeshMeshDistanceResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshMeshDistanceResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshMeshDistanceResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshMeshDistanceResult`/`Const_MeshMeshDistanceResult` to pass it to the function.
    public class _InOptConst_MeshMeshDistanceResult
    {
        public Const_MeshMeshDistanceResult? Opt;

        public _InOptConst_MeshMeshDistanceResult() {}
        public _InOptConst_MeshMeshDistanceResult(Const_MeshMeshDistanceResult value) {Opt = value;}
        public static implicit operator _InOptConst_MeshMeshDistanceResult(Const_MeshMeshDistanceResult value) {return new(value);}
    }

    /// Generated from class `MR::MeshMeshSignedDistanceResult`.
    /// This is the const half of the class.
    public class Const_MeshMeshSignedDistanceResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshMeshSignedDistanceResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshMeshSignedDistanceResult_Destroy(_Underlying *_this);
            __MR_MeshMeshSignedDistanceResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshMeshSignedDistanceResult() {Dispose(false);}

        /// two closest points: from meshes A and B respectively
        public unsafe MR.Const_PointOnFace A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_Get_a", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MeshMeshSignedDistanceResult_Get_a(_Underlying *_this);
                return new(__MR_MeshMeshSignedDistanceResult_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// two closest points: from meshes A and B respectively
        public unsafe MR.Const_PointOnFace B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_Get_b", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MeshMeshSignedDistanceResult_Get_b(_Underlying *_this);
                return new(__MR_MeshMeshSignedDistanceResult_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// signed distance between a and b, positive if meshes do not collide
        public unsafe float SignedDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_Get_signedDist", ExactSpelling = true)]
                extern static float *__MR_MeshMeshSignedDistanceResult_Get_signedDist(_Underlying *_this);
                return *__MR_MeshMeshSignedDistanceResult_Get_signedDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshMeshSignedDistanceResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshMeshSignedDistanceResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshMeshSignedDistanceResult` elementwise.
        public unsafe Const_MeshMeshSignedDistanceResult(MR.Const_PointOnFace a, MR.Const_PointOnFace b, float signedDist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_ConstructFrom(MR.PointOnFace._Underlying *a, MR.PointOnFace._Underlying *b, float signedDist);
            _UnderlyingPtr = __MR_MeshMeshSignedDistanceResult_ConstructFrom(a._UnderlyingPtr, b._UnderlyingPtr, signedDist);
        }

        /// Generated from constructor `MR::MeshMeshSignedDistanceResult::MeshMeshSignedDistanceResult`.
        public unsafe Const_MeshMeshSignedDistanceResult(MR.Const_MeshMeshSignedDistanceResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_ConstructFromAnother(MR.MeshMeshSignedDistanceResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshMeshSignedDistanceResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshMeshSignedDistanceResult`.
    /// This is the non-const half of the class.
    public class MeshMeshSignedDistanceResult : Const_MeshMeshSignedDistanceResult
    {
        internal unsafe MeshMeshSignedDistanceResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// two closest points: from meshes A and B respectively
        public new unsafe MR.PointOnFace A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_GetMutable_a", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MeshMeshSignedDistanceResult_GetMutable_a(_Underlying *_this);
                return new(__MR_MeshMeshSignedDistanceResult_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// two closest points: from meshes A and B respectively
        public new unsafe MR.PointOnFace B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_GetMutable_b", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MeshMeshSignedDistanceResult_GetMutable_b(_Underlying *_this);
                return new(__MR_MeshMeshSignedDistanceResult_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// signed distance between a and b, positive if meshes do not collide
        public new unsafe ref float SignedDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_GetMutable_signedDist", ExactSpelling = true)]
                extern static float *__MR_MeshMeshSignedDistanceResult_GetMutable_signedDist(_Underlying *_this);
                return ref *__MR_MeshMeshSignedDistanceResult_GetMutable_signedDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshMeshSignedDistanceResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshMeshSignedDistanceResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshMeshSignedDistanceResult` elementwise.
        public unsafe MeshMeshSignedDistanceResult(MR.Const_PointOnFace a, MR.Const_PointOnFace b, float signedDist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_ConstructFrom(MR.PointOnFace._Underlying *a, MR.PointOnFace._Underlying *b, float signedDist);
            _UnderlyingPtr = __MR_MeshMeshSignedDistanceResult_ConstructFrom(a._UnderlyingPtr, b._UnderlyingPtr, signedDist);
        }

        /// Generated from constructor `MR::MeshMeshSignedDistanceResult::MeshMeshSignedDistanceResult`.
        public unsafe MeshMeshSignedDistanceResult(MR.Const_MeshMeshSignedDistanceResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_ConstructFromAnother(MR.MeshMeshSignedDistanceResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshMeshSignedDistanceResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshMeshSignedDistanceResult::operator=`.
        public unsafe MR.MeshMeshSignedDistanceResult Assign(MR.Const_MeshMeshSignedDistanceResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshSignedDistanceResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_MeshMeshSignedDistanceResult_AssignFromAnother(_Underlying *_this, MR.MeshMeshSignedDistanceResult._Underlying *_other);
            return new(__MR_MeshMeshSignedDistanceResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshMeshSignedDistanceResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshMeshSignedDistanceResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshMeshSignedDistanceResult`/`Const_MeshMeshSignedDistanceResult` directly.
    public class _InOptMut_MeshMeshSignedDistanceResult
    {
        public MeshMeshSignedDistanceResult? Opt;

        public _InOptMut_MeshMeshSignedDistanceResult() {}
        public _InOptMut_MeshMeshSignedDistanceResult(MeshMeshSignedDistanceResult value) {Opt = value;}
        public static implicit operator _InOptMut_MeshMeshSignedDistanceResult(MeshMeshSignedDistanceResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshMeshSignedDistanceResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshMeshSignedDistanceResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshMeshSignedDistanceResult`/`Const_MeshMeshSignedDistanceResult` to pass it to the function.
    public class _InOptConst_MeshMeshSignedDistanceResult
    {
        public Const_MeshMeshSignedDistanceResult? Opt;

        public _InOptConst_MeshMeshSignedDistanceResult() {}
        public _InOptConst_MeshMeshSignedDistanceResult(Const_MeshMeshSignedDistanceResult value) {Opt = value;}
        public static implicit operator _InOptConst_MeshMeshSignedDistanceResult(Const_MeshMeshSignedDistanceResult value) {return new(value);}
    }

    /**
    * \brief computes minimal distance between two meshes or two mesh regions
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
    */
    /// Generated from function `MR::findDistance`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    public static unsafe MR.MeshMeshDistanceResult FindDistance(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null, float? upDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDistance", ExactSpelling = true)]
        extern static MR.MeshMeshDistanceResult._Underlying *__MR_findDistance(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, float *upDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        return new(__MR_findDistance(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief computes minimal distance between two meshes
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param upDistLimitSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points
    */
    /// Generated from function `MR::findSignedDistance`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    public static unsafe MR.MeshMeshSignedDistanceResult FindSignedDistance(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null, float? upDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSignedDistance_MR_MeshPart", ExactSpelling = true)]
        extern static MR.MeshMeshSignedDistanceResult._Underlying *__MR_findSignedDistance_MR_MeshPart(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, float *upDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        return new(__MR_findSignedDistance_MR_MeshPart(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief returns the maximum of the squared distances from each B-mesh vertex to A-mesh
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
    */
    /// Generated from function `MR::findMaxDistanceSqOneWay`.
    /// Parameter `maxDistanceSq` defaults to `3.40282347e38f`.
    public static unsafe float FindMaxDistanceSqOneWay(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null, float? maxDistanceSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMaxDistanceSqOneWay_MR_MeshPart", ExactSpelling = true)]
        extern static float __MR_findMaxDistanceSqOneWay_MR_MeshPart(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, float *maxDistanceSq);
        float __deref_maxDistanceSq = maxDistanceSq.GetValueOrDefault();
        return __MR_findMaxDistanceSqOneWay_MR_MeshPart(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, maxDistanceSq.HasValue ? &__deref_maxDistanceSq : null);
    }

    /**
    * \brief returns the squared Hausdorff distance between two meshes, that is
    the maximum of squared distances from each mesh vertex to the other mesh (in both directions)
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
    */
    /// Generated from function `MR::findMaxDistanceSq`.
    /// Parameter `maxDistanceSq` defaults to `3.40282347e38f`.
    public static unsafe float FindMaxDistanceSq(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null, float? maxDistanceSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMaxDistanceSq_MR_MeshPart", ExactSpelling = true)]
        extern static float __MR_findMaxDistanceSq_MR_MeshPart(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, float *maxDistanceSq);
        float __deref_maxDistanceSq = maxDistanceSq.GetValueOrDefault();
        return __MR_findMaxDistanceSq_MR_MeshPart(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, maxDistanceSq.HasValue ? &__deref_maxDistanceSq : null);
    }
}
