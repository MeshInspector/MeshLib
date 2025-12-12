public static partial class MR
{
    /// Generated from class `MR::PointAndDistance`.
    /// This is the const half of the class.
    public class Const_PointAndDistance : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointAndDistance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_Destroy", ExactSpelling = true)]
            extern static void __MR_PointAndDistance_Destroy(_Underlying *_this);
            __MR_PointAndDistance_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointAndDistance() {Dispose(false);}

        /// a cloud's point
        public unsafe MR.Const_VertId VId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_Get_vId", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PointAndDistance_Get_vId(_Underlying *_this);
                return new(__MR_PointAndDistance_Get_vId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the distance from input location to point vId considering point's weight
        public unsafe float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_Get_dist", ExactSpelling = true)]
                extern static float *__MR_PointAndDistance_Get_dist(_Underlying *_this);
                return *__MR_PointAndDistance_Get_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointAndDistance() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_DefaultConstruct();
            _UnderlyingPtr = __MR_PointAndDistance_DefaultConstruct();
        }

        /// Constructs `MR::PointAndDistance` elementwise.
        public unsafe Const_PointAndDistance(MR.VertId vId, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_ConstructFrom(MR.VertId vId, float dist);
            _UnderlyingPtr = __MR_PointAndDistance_ConstructFrom(vId, dist);
        }

        /// Generated from constructor `MR::PointAndDistance::PointAndDistance`.
        public unsafe Const_PointAndDistance(MR.Const_PointAndDistance _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_ConstructFromAnother(MR.PointAndDistance._Underlying *_other);
            _UnderlyingPtr = __MR_PointAndDistance_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PointAndDistance::operator bool`.
        public static unsafe explicit operator bool(MR.Const_PointAndDistance _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PointAndDistance_ConvertTo_bool(MR.Const_PointAndDistance._Underlying *_this);
            return __MR_PointAndDistance_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// check for validity, otherwise there is no point closer than maxBidirDist
        /// Generated from method `MR::PointAndDistance::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_valid", ExactSpelling = true)]
            extern static byte __MR_PointAndDistance_valid(_Underlying *_this);
            return __MR_PointAndDistance_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::PointAndDistance`.
    /// This is the non-const half of the class.
    public class PointAndDistance : Const_PointAndDistance
    {
        internal unsafe PointAndDistance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// a cloud's point
        public new unsafe MR.Mut_VertId VId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_GetMutable_vId", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PointAndDistance_GetMutable_vId(_Underlying *_this);
                return new(__MR_PointAndDistance_GetMutable_vId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the distance from input location to point vId considering point's weight
        public new unsafe ref float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_GetMutable_dist", ExactSpelling = true)]
                extern static float *__MR_PointAndDistance_GetMutable_dist(_Underlying *_this);
                return ref *__MR_PointAndDistance_GetMutable_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointAndDistance() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_DefaultConstruct();
            _UnderlyingPtr = __MR_PointAndDistance_DefaultConstruct();
        }

        /// Constructs `MR::PointAndDistance` elementwise.
        public unsafe PointAndDistance(MR.VertId vId, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_ConstructFrom(MR.VertId vId, float dist);
            _UnderlyingPtr = __MR_PointAndDistance_ConstructFrom(vId, dist);
        }

        /// Generated from constructor `MR::PointAndDistance::PointAndDistance`.
        public unsafe PointAndDistance(MR.Const_PointAndDistance _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_ConstructFromAnother(MR.PointAndDistance._Underlying *_other);
            _UnderlyingPtr = __MR_PointAndDistance_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointAndDistance::operator=`.
        public unsafe MR.PointAndDistance Assign(MR.Const_PointAndDistance _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAndDistance_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointAndDistance._Underlying *__MR_PointAndDistance_AssignFromAnother(_Underlying *_this, MR.PointAndDistance._Underlying *_other);
            return new(__MR_PointAndDistance_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointAndDistance` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointAndDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointAndDistance`/`Const_PointAndDistance` directly.
    public class _InOptMut_PointAndDistance
    {
        public PointAndDistance? Opt;

        public _InOptMut_PointAndDistance() {}
        public _InOptMut_PointAndDistance(PointAndDistance value) {Opt = value;}
        public static implicit operator _InOptMut_PointAndDistance(PointAndDistance value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointAndDistance` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointAndDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointAndDistance`/`Const_PointAndDistance` to pass it to the function.
    public class _InOptConst_PointAndDistance
    {
        public Const_PointAndDistance? Opt;

        public _InOptConst_PointAndDistance() {}
        public _InOptConst_PointAndDistance(Const_PointAndDistance value) {Opt = value;}
        public static implicit operator _InOptConst_PointAndDistance(Const_PointAndDistance value) {return new(value);}
    }

    /// Generated from class `MR::MeshPointAndDistance`.
    /// This is the const half of the class.
    public class Const_MeshPointAndDistance : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshPointAndDistance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshPointAndDistance_Destroy(_Underlying *_this);
            __MR_MeshPointAndDistance_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshPointAndDistance() {Dispose(false);}

        /// point location
        public unsafe MR.Const_Vector3f Loc
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_Get_loc", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshPointAndDistance_Get_loc(_Underlying *_this);
                return new(__MR_MeshPointAndDistance_Get_loc(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the corresponding point on mesh in barycentric representation
        public unsafe MR.Const_MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_Get_mtp", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_MeshPointAndDistance_Get_mtp(_Underlying *_this);
                return new(__MR_MeshPointAndDistance_Get_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// euclidean distance from input location to mtp
        public unsafe float EucledeanDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_Get_eucledeanDist", ExactSpelling = true)]
                extern static float *__MR_MeshPointAndDistance_Get_eucledeanDist(_Underlying *_this);
                return *__MR_MeshPointAndDistance_Get_eucledeanDist(_UnderlyingPtr);
            }
        }

        /// point's weight
        public unsafe float W
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_Get_w", ExactSpelling = true)]
                extern static float *__MR_MeshPointAndDistance_Get_w(_Underlying *_this);
                return *__MR_MeshPointAndDistance_Get_w(_UnderlyingPtr);
            }
        }

        /// either
        /// 1) bidirectional distances are computed, or
        /// 2) input location is locally outside of the surface (by pseudonormal)
        /// used for optimization
        public unsafe bool BidirectionalOrOutside
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_Get_bidirectionalOrOutside", ExactSpelling = true)]
                extern static bool *__MR_MeshPointAndDistance_Get_bidirectionalOrOutside(_Underlying *_this);
                return *__MR_MeshPointAndDistance_Get_bidirectionalOrOutside(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshPointAndDistance() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshPointAndDistance_DefaultConstruct();
        }

        /// Constructs `MR::MeshPointAndDistance` elementwise.
        public unsafe Const_MeshPointAndDistance(MR.Vector3f loc, MR.Const_MeshTriPoint mtp, float eucledeanDist, float w, bool bidirectionalOrOutside) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_ConstructFrom(MR.Vector3f loc, MR.MeshTriPoint._Underlying *mtp, float eucledeanDist, float w, byte bidirectionalOrOutside);
            _UnderlyingPtr = __MR_MeshPointAndDistance_ConstructFrom(loc, mtp._UnderlyingPtr, eucledeanDist, w, bidirectionalOrOutside ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::MeshPointAndDistance::MeshPointAndDistance`.
        public unsafe Const_MeshPointAndDistance(MR.Const_MeshPointAndDistance _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_ConstructFromAnother(MR.MeshPointAndDistance._Underlying *_other);
            _UnderlyingPtr = __MR_MeshPointAndDistance_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::MeshPointAndDistance::operator bool`.
        public static unsafe explicit operator bool(MR.Const_MeshPointAndDistance _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_MeshPointAndDistance_ConvertTo_bool(MR.Const_MeshPointAndDistance._Underlying *_this);
            return __MR_MeshPointAndDistance_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// bidirectional distance from input location to mtp considering point's weight
        /// Generated from method `MR::MeshPointAndDistance::bidirDist`.
        public unsafe float BidirDist()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_bidirDist", ExactSpelling = true)]
            extern static float __MR_MeshPointAndDistance_bidirDist(_Underlying *_this);
            return __MR_MeshPointAndDistance_bidirDist(_UnderlyingPtr);
        }

        /// the distance from input location to mtp considering point's weight and location inside/outside;
        /// dist() is continuous function of location unlike innerDist(),
        /// which makes 2*weight jump if the location moves through the surface
        /// Generated from method `MR::MeshPointAndDistance::dist`.
        public unsafe float Dist()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_dist", ExactSpelling = true)]
            extern static float __MR_MeshPointAndDistance_dist(_Underlying *_this);
            return __MR_MeshPointAndDistance_dist(_UnderlyingPtr);
        }

        /// check for validity, otherwise there is no point closer than maxBidirDist
        /// Generated from method `MR::MeshPointAndDistance::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_valid", ExactSpelling = true)]
            extern static byte __MR_MeshPointAndDistance_valid(_Underlying *_this);
            return __MR_MeshPointAndDistance_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::MeshPointAndDistance`.
    /// This is the non-const half of the class.
    public class MeshPointAndDistance : Const_MeshPointAndDistance
    {
        internal unsafe MeshPointAndDistance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// point location
        public new unsafe MR.Mut_Vector3f Loc
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_GetMutable_loc", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshPointAndDistance_GetMutable_loc(_Underlying *_this);
                return new(__MR_MeshPointAndDistance_GetMutable_loc(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the corresponding point on mesh in barycentric representation
        public new unsafe MR.MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_GetMutable_mtp", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_MeshPointAndDistance_GetMutable_mtp(_Underlying *_this);
                return new(__MR_MeshPointAndDistance_GetMutable_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// euclidean distance from input location to mtp
        public new unsafe ref float EucledeanDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_GetMutable_eucledeanDist", ExactSpelling = true)]
                extern static float *__MR_MeshPointAndDistance_GetMutable_eucledeanDist(_Underlying *_this);
                return ref *__MR_MeshPointAndDistance_GetMutable_eucledeanDist(_UnderlyingPtr);
            }
        }

        /// point's weight
        public new unsafe ref float W
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_GetMutable_w", ExactSpelling = true)]
                extern static float *__MR_MeshPointAndDistance_GetMutable_w(_Underlying *_this);
                return ref *__MR_MeshPointAndDistance_GetMutable_w(_UnderlyingPtr);
            }
        }

        /// either
        /// 1) bidirectional distances are computed, or
        /// 2) input location is locally outside of the surface (by pseudonormal)
        /// used for optimization
        public new unsafe ref bool BidirectionalOrOutside
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_GetMutable_bidirectionalOrOutside", ExactSpelling = true)]
                extern static bool *__MR_MeshPointAndDistance_GetMutable_bidirectionalOrOutside(_Underlying *_this);
                return ref *__MR_MeshPointAndDistance_GetMutable_bidirectionalOrOutside(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshPointAndDistance() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshPointAndDistance_DefaultConstruct();
        }

        /// Constructs `MR::MeshPointAndDistance` elementwise.
        public unsafe MeshPointAndDistance(MR.Vector3f loc, MR.Const_MeshTriPoint mtp, float eucledeanDist, float w, bool bidirectionalOrOutside) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_ConstructFrom(MR.Vector3f loc, MR.MeshTriPoint._Underlying *mtp, float eucledeanDist, float w, byte bidirectionalOrOutside);
            _UnderlyingPtr = __MR_MeshPointAndDistance_ConstructFrom(loc, mtp._UnderlyingPtr, eucledeanDist, w, bidirectionalOrOutside ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::MeshPointAndDistance::MeshPointAndDistance`.
        public unsafe MeshPointAndDistance(MR.Const_MeshPointAndDistance _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_ConstructFromAnother(MR.MeshPointAndDistance._Underlying *_other);
            _UnderlyingPtr = __MR_MeshPointAndDistance_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshPointAndDistance::operator=`.
        public unsafe MR.MeshPointAndDistance Assign(MR.Const_MeshPointAndDistance _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPointAndDistance_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshPointAndDistance._Underlying *__MR_MeshPointAndDistance_AssignFromAnother(_Underlying *_this, MR.MeshPointAndDistance._Underlying *_other);
            return new(__MR_MeshPointAndDistance_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshPointAndDistance` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshPointAndDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshPointAndDistance`/`Const_MeshPointAndDistance` directly.
    public class _InOptMut_MeshPointAndDistance
    {
        public MeshPointAndDistance? Opt;

        public _InOptMut_MeshPointAndDistance() {}
        public _InOptMut_MeshPointAndDistance(MeshPointAndDistance value) {Opt = value;}
        public static implicit operator _InOptMut_MeshPointAndDistance(MeshPointAndDistance value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshPointAndDistance` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshPointAndDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshPointAndDistance`/`Const_MeshPointAndDistance` to pass it to the function.
    public class _InOptConst_MeshPointAndDistance
    {
        public Const_MeshPointAndDistance? Opt;

        public _InOptConst_MeshPointAndDistance() {}
        public _InOptConst_MeshPointAndDistance(Const_MeshPointAndDistance value) {Opt = value;}
        public static implicit operator _InOptConst_MeshPointAndDistance(Const_MeshPointAndDistance value) {return new(value);}
    }

    /// Generated from class `MR::DistanceFromWeightedPointsParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceFromWeightedPointsComputeParams`
    /// This is the const half of the class.
    public class Const_DistanceFromWeightedPointsParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceFromWeightedPointsParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceFromWeightedPointsParams_Destroy(_Underlying *_this);
            __MR_DistanceFromWeightedPointsParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceFromWeightedPointsParams() {Dispose(false);}

        /// function returning the weight of each point, must be set by the user
        public unsafe MR.Std.Const_Function_FloatFuncFromMRVertId PointWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_Get_pointWeight", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromMRVertId._Underlying *__MR_DistanceFromWeightedPointsParams_Get_pointWeight(_Underlying *_this);
                return new(__MR_DistanceFromWeightedPointsParams_Get_pointWeight(_UnderlyingPtr), is_owning: false);
            }
        }

        /// maximal weight among all points in the cloud;
        /// if this value is imprecise, then more computations will be made by algorithm
        public unsafe float MaxWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_Get_maxWeight", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsParams_Get_maxWeight(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsParams_Get_maxWeight(_UnderlyingPtr);
            }
        }

        /// maximal magnitude of gradient of points' weight in the cloud, >=0;
        /// if maxWeightGrad < 1 then more search optimizations can be done
        public unsafe float MaxWeightGrad
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_Get_maxWeightGrad", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsParams_Get_maxWeightGrad(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsParams_Get_maxWeightGrad(_UnderlyingPtr);
            }
        }

        /// for points, it must always true;
        /// for triangles:
        ///   if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
        ///   if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
        public unsafe bool BidirectionalMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_Get_bidirectionalMode", ExactSpelling = true)]
                extern static bool *__MR_DistanceFromWeightedPointsParams_Get_bidirectionalMode(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsParams_Get_bidirectionalMode(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceFromWeightedPointsParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsParams_DefaultConstruct();
        }

        /// Constructs `MR::DistanceFromWeightedPointsParams` elementwise.
        public unsafe Const_DistanceFromWeightedPointsParams(MR.Std._ByValue_Function_FloatFuncFromMRVertId pointWeight, float maxWeight, float maxWeightGrad, bool bidirectionalMode) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_ConstructFrom(MR.Misc._PassBy pointWeight_pass_by, MR.Std.Function_FloatFuncFromMRVertId._Underlying *pointWeight, float maxWeight, float maxWeightGrad, byte bidirectionalMode);
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsParams_ConstructFrom(pointWeight.PassByMode, pointWeight.Value is not null ? pointWeight.Value._UnderlyingPtr : null, maxWeight, maxWeightGrad, bidirectionalMode ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::DistanceFromWeightedPointsParams::DistanceFromWeightedPointsParams`.
        public unsafe Const_DistanceFromWeightedPointsParams(MR._ByValue_DistanceFromWeightedPointsParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceFromWeightedPointsParams._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DistanceFromWeightedPointsParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceFromWeightedPointsComputeParams`
    /// This is the non-const half of the class.
    public class DistanceFromWeightedPointsParams : Const_DistanceFromWeightedPointsParams
    {
        internal unsafe DistanceFromWeightedPointsParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// function returning the weight of each point, must be set by the user
        public new unsafe MR.Std.Function_FloatFuncFromMRVertId PointWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_GetMutable_pointWeight", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromMRVertId._Underlying *__MR_DistanceFromWeightedPointsParams_GetMutable_pointWeight(_Underlying *_this);
                return new(__MR_DistanceFromWeightedPointsParams_GetMutable_pointWeight(_UnderlyingPtr), is_owning: false);
            }
        }

        /// maximal weight among all points in the cloud;
        /// if this value is imprecise, then more computations will be made by algorithm
        public new unsafe ref float MaxWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_GetMutable_maxWeight", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsParams_GetMutable_maxWeight(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsParams_GetMutable_maxWeight(_UnderlyingPtr);
            }
        }

        /// maximal magnitude of gradient of points' weight in the cloud, >=0;
        /// if maxWeightGrad < 1 then more search optimizations can be done
        public new unsafe ref float MaxWeightGrad
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_GetMutable_maxWeightGrad", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsParams_GetMutable_maxWeightGrad(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsParams_GetMutable_maxWeightGrad(_UnderlyingPtr);
            }
        }

        /// for points, it must always true;
        /// for triangles:
        ///   if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
        ///   if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
        public new unsafe ref bool BidirectionalMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_GetMutable_bidirectionalMode", ExactSpelling = true)]
                extern static bool *__MR_DistanceFromWeightedPointsParams_GetMutable_bidirectionalMode(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsParams_GetMutable_bidirectionalMode(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceFromWeightedPointsParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsParams_DefaultConstruct();
        }

        /// Constructs `MR::DistanceFromWeightedPointsParams` elementwise.
        public unsafe DistanceFromWeightedPointsParams(MR.Std._ByValue_Function_FloatFuncFromMRVertId pointWeight, float maxWeight, float maxWeightGrad, bool bidirectionalMode) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_ConstructFrom(MR.Misc._PassBy pointWeight_pass_by, MR.Std.Function_FloatFuncFromMRVertId._Underlying *pointWeight, float maxWeight, float maxWeightGrad, byte bidirectionalMode);
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsParams_ConstructFrom(pointWeight.PassByMode, pointWeight.Value is not null ? pointWeight.Value._UnderlyingPtr : null, maxWeight, maxWeightGrad, bidirectionalMode ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::DistanceFromWeightedPointsParams::DistanceFromWeightedPointsParams`.
        public unsafe DistanceFromWeightedPointsParams(MR._ByValue_DistanceFromWeightedPointsParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceFromWeightedPointsParams._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DistanceFromWeightedPointsParams::operator=`.
        public unsafe MR.DistanceFromWeightedPointsParams Assign(MR._ByValue_DistanceFromWeightedPointsParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceFromWeightedPointsParams._Underlying *_other);
            return new(__MR_DistanceFromWeightedPointsParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DistanceFromWeightedPointsParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DistanceFromWeightedPointsParams`/`Const_DistanceFromWeightedPointsParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DistanceFromWeightedPointsParams
    {
        internal readonly Const_DistanceFromWeightedPointsParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DistanceFromWeightedPointsParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DistanceFromWeightedPointsParams(Const_DistanceFromWeightedPointsParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DistanceFromWeightedPointsParams(Const_DistanceFromWeightedPointsParams arg) {return new(arg);}
        public _ByValue_DistanceFromWeightedPointsParams(MR.Misc._Moved<DistanceFromWeightedPointsParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DistanceFromWeightedPointsParams(MR.Misc._Moved<DistanceFromWeightedPointsParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DistanceFromWeightedPointsParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceFromWeightedPointsParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceFromWeightedPointsParams`/`Const_DistanceFromWeightedPointsParams` directly.
    public class _InOptMut_DistanceFromWeightedPointsParams
    {
        public DistanceFromWeightedPointsParams? Opt;

        public _InOptMut_DistanceFromWeightedPointsParams() {}
        public _InOptMut_DistanceFromWeightedPointsParams(DistanceFromWeightedPointsParams value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceFromWeightedPointsParams(DistanceFromWeightedPointsParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceFromWeightedPointsParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceFromWeightedPointsParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceFromWeightedPointsParams`/`Const_DistanceFromWeightedPointsParams` to pass it to the function.
    public class _InOptConst_DistanceFromWeightedPointsParams
    {
        public Const_DistanceFromWeightedPointsParams? Opt;

        public _InOptConst_DistanceFromWeightedPointsParams() {}
        public _InOptConst_DistanceFromWeightedPointsParams(Const_DistanceFromWeightedPointsParams value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceFromWeightedPointsParams(Const_DistanceFromWeightedPointsParams value) {return new(value);}
    }

    /// Generated from class `MR::DistanceFromWeightedPointsComputeParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceFromWeightedPointsParams`
    /// This is the const half of the class.
    public class Const_DistanceFromWeightedPointsComputeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceFromWeightedPointsComputeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceFromWeightedPointsComputeParams_Destroy(_Underlying *_this);
            __MR_DistanceFromWeightedPointsComputeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceFromWeightedPointsComputeParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_DistanceFromWeightedPointsParams(Const_DistanceFromWeightedPointsComputeParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_UpcastTo_MR_DistanceFromWeightedPointsParams", ExactSpelling = true)]
            extern static MR.Const_DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_UpcastTo_MR_DistanceFromWeightedPointsParams(_Underlying *_this);
            MR.Const_DistanceFromWeightedPointsParams ret = new(__MR_DistanceFromWeightedPointsComputeParams_UpcastTo_MR_DistanceFromWeightedPointsParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // default 0 here does not work for negative distances
        public unsafe float MinBidirDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Get_minBidirDist", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_Get_minBidirDist(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsComputeParams_Get_minBidirDist(_UnderlyingPtr);
            }
        }

        /// find the closest point only if weighted bidirectional distance to it is less than given value
        public unsafe float MaxBidirDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Get_maxBidirDist", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_Get_maxBidirDist(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsComputeParams_Get_maxBidirDist(_UnderlyingPtr);
            }
        }

        /// function returning the weight of each point, must be set by the user
        public unsafe MR.Std.Const_Function_FloatFuncFromMRVertId PointWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Get_pointWeight", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromMRVertId._Underlying *__MR_DistanceFromWeightedPointsComputeParams_Get_pointWeight(_Underlying *_this);
                return new(__MR_DistanceFromWeightedPointsComputeParams_Get_pointWeight(_UnderlyingPtr), is_owning: false);
            }
        }

        /// maximal weight among all points in the cloud;
        /// if this value is imprecise, then more computations will be made by algorithm
        public unsafe float MaxWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Get_maxWeight", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_Get_maxWeight(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsComputeParams_Get_maxWeight(_UnderlyingPtr);
            }
        }

        /// maximal magnitude of gradient of points' weight in the cloud, >=0;
        /// if maxWeightGrad < 1 then more search optimizations can be done
        public unsafe float MaxWeightGrad
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Get_maxWeightGrad", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_Get_maxWeightGrad(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsComputeParams_Get_maxWeightGrad(_UnderlyingPtr);
            }
        }

        /// for points, it must always true;
        /// for triangles:
        ///   if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
        ///   if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
        public unsafe bool BidirectionalMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_Get_bidirectionalMode", ExactSpelling = true)]
                extern static bool *__MR_DistanceFromWeightedPointsComputeParams_Get_bidirectionalMode(_Underlying *_this);
                return *__MR_DistanceFromWeightedPointsComputeParams_Get_bidirectionalMode(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceFromWeightedPointsComputeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsComputeParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsComputeParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::DistanceFromWeightedPointsComputeParams::DistanceFromWeightedPointsComputeParams`.
        public unsafe Const_DistanceFromWeightedPointsComputeParams(MR._ByValue_DistanceFromWeightedPointsComputeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsComputeParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceFromWeightedPointsComputeParams._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsComputeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DistanceFromWeightedPointsComputeParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceFromWeightedPointsParams`
    /// This is the non-const half of the class.
    public class DistanceFromWeightedPointsComputeParams : Const_DistanceFromWeightedPointsComputeParams
    {
        internal unsafe DistanceFromWeightedPointsComputeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.DistanceFromWeightedPointsParams(DistanceFromWeightedPointsComputeParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_UpcastTo_MR_DistanceFromWeightedPointsParams", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_UpcastTo_MR_DistanceFromWeightedPointsParams(_Underlying *_this);
            MR.DistanceFromWeightedPointsParams ret = new(__MR_DistanceFromWeightedPointsComputeParams_UpcastTo_MR_DistanceFromWeightedPointsParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // default 0 here does not work for negative distances
        public new unsafe ref float MinBidirDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_GetMutable_minBidirDist", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_minBidirDist(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_minBidirDist(_UnderlyingPtr);
            }
        }

        /// find the closest point only if weighted bidirectional distance to it is less than given value
        public new unsafe ref float MaxBidirDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxBidirDist", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxBidirDist(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxBidirDist(_UnderlyingPtr);
            }
        }

        /// function returning the weight of each point, must be set by the user
        public new unsafe MR.Std.Function_FloatFuncFromMRVertId PointWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_GetMutable_pointWeight", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromMRVertId._Underlying *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_pointWeight(_Underlying *_this);
                return new(__MR_DistanceFromWeightedPointsComputeParams_GetMutable_pointWeight(_UnderlyingPtr), is_owning: false);
            }
        }

        /// maximal weight among all points in the cloud;
        /// if this value is imprecise, then more computations will be made by algorithm
        public new unsafe ref float MaxWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxWeight", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxWeight(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxWeight(_UnderlyingPtr);
            }
        }

        /// maximal magnitude of gradient of points' weight in the cloud, >=0;
        /// if maxWeightGrad < 1 then more search optimizations can be done
        public new unsafe ref float MaxWeightGrad
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxWeightGrad", ExactSpelling = true)]
                extern static float *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxWeightGrad(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_maxWeightGrad(_UnderlyingPtr);
            }
        }

        /// for points, it must always true;
        /// for triangles:
        ///   if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
        ///   if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
        public new unsafe ref bool BidirectionalMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_GetMutable_bidirectionalMode", ExactSpelling = true)]
                extern static bool *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_bidirectionalMode(_Underlying *_this);
                return ref *__MR_DistanceFromWeightedPointsComputeParams_GetMutable_bidirectionalMode(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceFromWeightedPointsComputeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsComputeParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsComputeParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::DistanceFromWeightedPointsComputeParams::DistanceFromWeightedPointsComputeParams`.
        public unsafe DistanceFromWeightedPointsComputeParams(MR._ByValue_DistanceFromWeightedPointsComputeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsComputeParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceFromWeightedPointsComputeParams._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceFromWeightedPointsComputeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DistanceFromWeightedPointsComputeParams::operator=`.
        public unsafe MR.DistanceFromWeightedPointsComputeParams Assign(MR._ByValue_DistanceFromWeightedPointsComputeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceFromWeightedPointsComputeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceFromWeightedPointsComputeParams._Underlying *__MR_DistanceFromWeightedPointsComputeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceFromWeightedPointsComputeParams._Underlying *_other);
            return new(__MR_DistanceFromWeightedPointsComputeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DistanceFromWeightedPointsComputeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DistanceFromWeightedPointsComputeParams`/`Const_DistanceFromWeightedPointsComputeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DistanceFromWeightedPointsComputeParams
    {
        internal readonly Const_DistanceFromWeightedPointsComputeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DistanceFromWeightedPointsComputeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DistanceFromWeightedPointsComputeParams(Const_DistanceFromWeightedPointsComputeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DistanceFromWeightedPointsComputeParams(Const_DistanceFromWeightedPointsComputeParams arg) {return new(arg);}
        public _ByValue_DistanceFromWeightedPointsComputeParams(MR.Misc._Moved<DistanceFromWeightedPointsComputeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DistanceFromWeightedPointsComputeParams(MR.Misc._Moved<DistanceFromWeightedPointsComputeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DistanceFromWeightedPointsComputeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceFromWeightedPointsComputeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceFromWeightedPointsComputeParams`/`Const_DistanceFromWeightedPointsComputeParams` directly.
    public class _InOptMut_DistanceFromWeightedPointsComputeParams
    {
        public DistanceFromWeightedPointsComputeParams? Opt;

        public _InOptMut_DistanceFromWeightedPointsComputeParams() {}
        public _InOptMut_DistanceFromWeightedPointsComputeParams(DistanceFromWeightedPointsComputeParams value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceFromWeightedPointsComputeParams(DistanceFromWeightedPointsComputeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceFromWeightedPointsComputeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceFromWeightedPointsComputeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceFromWeightedPointsComputeParams`/`Const_DistanceFromWeightedPointsComputeParams` to pass it to the function.
    public class _InOptConst_DistanceFromWeightedPointsComputeParams
    {
        public Const_DistanceFromWeightedPointsComputeParams? Opt;

        public _InOptConst_DistanceFromWeightedPointsComputeParams() {}
        public _InOptConst_DistanceFromWeightedPointsComputeParams(Const_DistanceFromWeightedPointsComputeParams value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceFromWeightedPointsComputeParams(Const_DistanceFromWeightedPointsComputeParams value) {return new(value);}
    }

    /// consider a point cloud where each point has additive weight,
    /// and the distance to a point is considered equal to (euclidean distance - weight),
    /// finds the point with minimal distance to given 3D location
    /// Generated from function `MR::findClosestWeightedPoint`.
    public static unsafe MR.PointAndDistance FindClosestWeightedPoint(MR.Const_Vector3f loc, MR.Const_AABBTreePoints tree, MR.Const_DistanceFromWeightedPointsComputeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findClosestWeightedPoint", ExactSpelling = true)]
        extern static MR.PointAndDistance._Underlying *__MR_findClosestWeightedPoint(MR.Const_Vector3f._Underlying *loc, MR.Const_AABBTreePoints._Underlying *tree, MR.Const_DistanceFromWeightedPointsComputeParams._Underlying *params_);
        return new(__MR_findClosestWeightedPoint(loc._UnderlyingPtr, tree._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true);
    }

    /// consider a mesh where each vertex has additive weight, and this weight is linearly interpolated in mesh triangles,
    /// and the distance to a point is considered equal to (euclidean distance - weight),
    /// finds the point on given mesh with minimal distance to given 3D location
    /// Generated from function `MR::findClosestWeightedMeshPoint`.
    public static unsafe MR.MeshPointAndDistance FindClosestWeightedMeshPoint(MR.Const_Vector3f loc, MR.Const_Mesh mesh, MR.Const_DistanceFromWeightedPointsComputeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findClosestWeightedMeshPoint", ExactSpelling = true)]
        extern static MR.MeshPointAndDistance._Underlying *__MR_findClosestWeightedMeshPoint(MR.Const_Vector3f._Underlying *loc, MR.Const_Mesh._Underlying *mesh, MR.Const_DistanceFromWeightedPointsComputeParams._Underlying *params_);
        return new(__MR_findClosestWeightedMeshPoint(loc._UnderlyingPtr, mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true);
    }
}
