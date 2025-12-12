public static partial class MR
{
    /// Generated from class `MR::ICPPairData`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPGroupPair`
    ///     `MR::PointPair`
    /// This is the const half of the class.
    public class Const_ICPPairData : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_ICPPairData>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ICPPairData(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Destroy", ExactSpelling = true)]
            extern static void __MR_ICPPairData_Destroy(_Underlying *_this);
            __MR_ICPPairData_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ICPPairData() {Dispose(false);}

        /// coordinates of the source point after transforming in world space
        public unsafe MR.Const_Vector3f SrcPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Get_srcPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPPairData_Get_srcPoint(_Underlying *_this);
                return new(__MR_ICPPairData_Get_srcPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in source point after transforming in world space
        public unsafe MR.Const_Vector3f SrcNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Get_srcNorm", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPPairData_Get_srcNorm(_Underlying *_this);
                return new(__MR_ICPPairData_Get_srcNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the closest point on target after transforming in world space
        public unsafe MR.Const_Vector3f TgtPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Get_tgtPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPPairData_Get_tgtPoint(_Underlying *_this);
                return new(__MR_ICPPairData_Get_tgtPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in the target point after transforming in world space
        public unsafe MR.Const_Vector3f TgtNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Get_tgtNorm", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPPairData_Get_tgtNorm(_Underlying *_this);
                return new(__MR_ICPPairData_Get_tgtNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between source and target points
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_ICPPairData_Get_distSq(_Underlying *_this);
                return *__MR_ICPPairData_Get_distSq(_UnderlyingPtr);
            }
        }

        /// weight of the pair (to prioritize over other pairs)
        public unsafe float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_Get_weight", ExactSpelling = true)]
                extern static float *__MR_ICPPairData_Get_weight(_Underlying *_this);
                return *__MR_ICPPairData_Get_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ICPPairData() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPPairData_DefaultConstruct();
        }

        /// Constructs `MR::ICPPairData` elementwise.
        public unsafe Const_ICPPairData(MR.Vector3f srcPoint, MR.Vector3f srcNorm, MR.Vector3f tgtPoint, MR.Vector3f tgtNorm, float distSq, float weight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_ConstructFrom", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_ConstructFrom(MR.Vector3f srcPoint, MR.Vector3f srcNorm, MR.Vector3f tgtPoint, MR.Vector3f tgtNorm, float distSq, float weight);
            _UnderlyingPtr = __MR_ICPPairData_ConstructFrom(srcPoint, srcNorm, tgtPoint, tgtNorm, distSq, weight);
        }

        /// Generated from constructor `MR::ICPPairData::ICPPairData`.
        public unsafe Const_ICPPairData(MR.Const_ICPPairData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_ConstructFromAnother(MR.ICPPairData._Underlying *_other);
            _UnderlyingPtr = __MR_ICPPairData_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_ICPPairData _1, MR.Const_ICPPairData _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ICPPairData", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ICPPairData(MR.Const_ICPPairData._Underlying *_1, MR.Const_ICPPairData._Underlying *_2);
            return __MR_equal_MR_ICPPairData(_1._UnderlyingPtr, _2._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ICPPairData _1, MR.Const_ICPPairData _2)
        {
            return !(_1 == _2);
        }

        // IEquatable:

        public bool Equals(MR.Const_ICPPairData? _2)
        {
            if (_2 is null)
                return false;
            return this == _2;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_ICPPairData)
                return this == (MR.Const_ICPPairData)other;
            return false;
        }
    }

    /// Generated from class `MR::ICPPairData`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPGroupPair`
    ///     `MR::PointPair`
    /// This is the non-const half of the class.
    public class ICPPairData : Const_ICPPairData
    {
        internal unsafe ICPPairData(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// coordinates of the source point after transforming in world space
        public new unsafe MR.Mut_Vector3f SrcPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_GetMutable_srcPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPPairData_GetMutable_srcPoint(_Underlying *_this);
                return new(__MR_ICPPairData_GetMutable_srcPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in source point after transforming in world space
        public new unsafe MR.Mut_Vector3f SrcNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_GetMutable_srcNorm", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPPairData_GetMutable_srcNorm(_Underlying *_this);
                return new(__MR_ICPPairData_GetMutable_srcNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the closest point on target after transforming in world space
        public new unsafe MR.Mut_Vector3f TgtPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_GetMutable_tgtPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPPairData_GetMutable_tgtPoint(_Underlying *_this);
                return new(__MR_ICPPairData_GetMutable_tgtPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in the target point after transforming in world space
        public new unsafe MR.Mut_Vector3f TgtNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_GetMutable_tgtNorm", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPPairData_GetMutable_tgtNorm(_Underlying *_this);
                return new(__MR_ICPPairData_GetMutable_tgtNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between source and target points
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_ICPPairData_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_ICPPairData_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// weight of the pair (to prioritize over other pairs)
        public new unsafe ref float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_GetMutable_weight", ExactSpelling = true)]
                extern static float *__MR_ICPPairData_GetMutable_weight(_Underlying *_this);
                return ref *__MR_ICPPairData_GetMutable_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ICPPairData() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPPairData_DefaultConstruct();
        }

        /// Constructs `MR::ICPPairData` elementwise.
        public unsafe ICPPairData(MR.Vector3f srcPoint, MR.Vector3f srcNorm, MR.Vector3f tgtPoint, MR.Vector3f tgtNorm, float distSq, float weight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_ConstructFrom", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_ConstructFrom(MR.Vector3f srcPoint, MR.Vector3f srcNorm, MR.Vector3f tgtPoint, MR.Vector3f tgtNorm, float distSq, float weight);
            _UnderlyingPtr = __MR_ICPPairData_ConstructFrom(srcPoint, srcNorm, tgtPoint, tgtNorm, distSq, weight);
        }

        /// Generated from constructor `MR::ICPPairData::ICPPairData`.
        public unsafe ICPPairData(MR.Const_ICPPairData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_ConstructFromAnother(MR.ICPPairData._Underlying *_other);
            _UnderlyingPtr = __MR_ICPPairData_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ICPPairData::operator=`.
        public unsafe MR.ICPPairData Assign(MR.Const_ICPPairData _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPPairData_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPPairData_AssignFromAnother(_Underlying *_this, MR.ICPPairData._Underlying *_other);
            return new(__MR_ICPPairData_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ICPPairData` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ICPPairData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPPairData`/`Const_ICPPairData` directly.
    public class _InOptMut_ICPPairData
    {
        public ICPPairData? Opt;

        public _InOptMut_ICPPairData() {}
        public _InOptMut_ICPPairData(ICPPairData value) {Opt = value;}
        public static implicit operator _InOptMut_ICPPairData(ICPPairData value) {return new(value);}
    }

    /// This is used for optional parameters of class `ICPPairData` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ICPPairData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPPairData`/`Const_ICPPairData` to pass it to the function.
    public class _InOptConst_ICPPairData
    {
        public Const_ICPPairData? Opt;

        public _InOptConst_ICPPairData() {}
        public _InOptConst_ICPPairData(Const_ICPPairData value) {Opt = value;}
        public static implicit operator _InOptConst_ICPPairData(Const_ICPPairData value) {return new(value);}
    }

    /// Stores a pair of points: one samples on the source and the closest to it on the target
    /// Generated from class `MR::PointPair`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPPairData`
    /// This is the const half of the class.
    public class Const_PointPair : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_PointPair>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointPair(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Destroy", ExactSpelling = true)]
            extern static void __MR_PointPair_Destroy(_Underlying *_this);
            __MR_PointPair_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointPair() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ICPPairData(Const_PointPair self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_UpcastTo_MR_ICPPairData", ExactSpelling = true)]
            extern static MR.Const_ICPPairData._Underlying *__MR_PointPair_UpcastTo_MR_ICPPairData(_Underlying *_this);
            MR.Const_ICPPairData ret = new(__MR_PointPair_UpcastTo_MR_ICPPairData(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// id of the source point
        public unsafe MR.Const_VertId SrcVertId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_srcVertId", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PointPair_Get_srcVertId(_Underlying *_this);
                return new(__MR_PointPair_Get_srcVertId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// for point clouds it is the closest vertex on target,
        /// for meshes it is the closest vertex of the triangle with the closest point on target
        public unsafe MR.Const_VertId TgtCloseVert
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_tgtCloseVert", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PointPair_Get_tgtCloseVert(_Underlying *_this);
                return new(__MR_PointPair_Get_tgtCloseVert(_UnderlyingPtr), is_owning: false);
            }
        }

        /// cosine between normals in source and target points
        public unsafe float NormalsAngleCos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_normalsAngleCos", ExactSpelling = true)]
                extern static float *__MR_PointPair_Get_normalsAngleCos(_Underlying *_this);
                return *__MR_PointPair_Get_normalsAngleCos(_UnderlyingPtr);
            }
        }

        /// true if if the closest point on target is located on the boundary (only for meshes)
        public unsafe bool TgtOnBd
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_tgtOnBd", ExactSpelling = true)]
                extern static bool *__MR_PointPair_Get_tgtOnBd(_Underlying *_this);
                return *__MR_PointPair_Get_tgtOnBd(_UnderlyingPtr);
            }
        }

        /// coordinates of the source point after transforming in world space
        public unsafe MR.Const_Vector3f SrcPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_srcPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointPair_Get_srcPoint(_Underlying *_this);
                return new(__MR_PointPair_Get_srcPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in source point after transforming in world space
        public unsafe MR.Const_Vector3f SrcNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_srcNorm", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointPair_Get_srcNorm(_Underlying *_this);
                return new(__MR_PointPair_Get_srcNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the closest point on target after transforming in world space
        public unsafe MR.Const_Vector3f TgtPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_tgtPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointPair_Get_tgtPoint(_Underlying *_this);
                return new(__MR_PointPair_Get_tgtPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in the target point after transforming in world space
        public unsafe MR.Const_Vector3f TgtNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_tgtNorm", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointPair_Get_tgtNorm(_Underlying *_this);
                return new(__MR_PointPair_Get_tgtNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between source and target points
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_PointPair_Get_distSq(_Underlying *_this);
                return *__MR_PointPair_Get_distSq(_UnderlyingPtr);
            }
        }

        /// weight of the pair (to prioritize over other pairs)
        public unsafe float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_Get_weight", ExactSpelling = true)]
                extern static float *__MR_PointPair_Get_weight(_Underlying *_this);
                return *__MR_PointPair_Get_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointPair() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointPair._Underlying *__MR_PointPair_DefaultConstruct();
            _UnderlyingPtr = __MR_PointPair_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointPair::PointPair`.
        public unsafe Const_PointPair(MR.Const_PointPair _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointPair._Underlying *__MR_PointPair_ConstructFromAnother(MR.PointPair._Underlying *_other);
            _UnderlyingPtr = __MR_PointPair_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_PointPair _1, MR.Const_PointPair _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_PointPair", ExactSpelling = true)]
            extern static byte __MR_equal_MR_PointPair(MR.Const_PointPair._Underlying *_1, MR.Const_PointPair._Underlying *_2);
            return __MR_equal_MR_PointPair(_1._UnderlyingPtr, _2._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_PointPair _1, MR.Const_PointPair _2)
        {
            return !(_1 == _2);
        }

        // IEquatable:

        public bool Equals(MR.Const_PointPair? _2)
        {
            if (_2 is null)
                return false;
            return this == _2;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_PointPair)
                return this == (MR.Const_PointPair)other;
            return false;
        }
    }

    /// Stores a pair of points: one samples on the source and the closest to it on the target
    /// Generated from class `MR::PointPair`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPPairData`
    /// This is the non-const half of the class.
    public class PointPair : Const_PointPair
    {
        internal unsafe PointPair(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ICPPairData(PointPair self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_UpcastTo_MR_ICPPairData", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_PointPair_UpcastTo_MR_ICPPairData(_Underlying *_this);
            MR.ICPPairData ret = new(__MR_PointPair_UpcastTo_MR_ICPPairData(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// id of the source point
        public new unsafe MR.Mut_VertId SrcVertId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_srcVertId", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PointPair_GetMutable_srcVertId(_Underlying *_this);
                return new(__MR_PointPair_GetMutable_srcVertId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// for point clouds it is the closest vertex on target,
        /// for meshes it is the closest vertex of the triangle with the closest point on target
        public new unsafe MR.Mut_VertId TgtCloseVert
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_tgtCloseVert", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PointPair_GetMutable_tgtCloseVert(_Underlying *_this);
                return new(__MR_PointPair_GetMutable_tgtCloseVert(_UnderlyingPtr), is_owning: false);
            }
        }

        /// cosine between normals in source and target points
        public new unsafe ref float NormalsAngleCos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_normalsAngleCos", ExactSpelling = true)]
                extern static float *__MR_PointPair_GetMutable_normalsAngleCos(_Underlying *_this);
                return ref *__MR_PointPair_GetMutable_normalsAngleCos(_UnderlyingPtr);
            }
        }

        /// true if if the closest point on target is located on the boundary (only for meshes)
        public new unsafe ref bool TgtOnBd
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_tgtOnBd", ExactSpelling = true)]
                extern static bool *__MR_PointPair_GetMutable_tgtOnBd(_Underlying *_this);
                return ref *__MR_PointPair_GetMutable_tgtOnBd(_UnderlyingPtr);
            }
        }

        /// coordinates of the source point after transforming in world space
        public new unsafe MR.Mut_Vector3f SrcPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_srcPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointPair_GetMutable_srcPoint(_Underlying *_this);
                return new(__MR_PointPair_GetMutable_srcPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in source point after transforming in world space
        public new unsafe MR.Mut_Vector3f SrcNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_srcNorm", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointPair_GetMutable_srcNorm(_Underlying *_this);
                return new(__MR_PointPair_GetMutable_srcNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the closest point on target after transforming in world space
        public new unsafe MR.Mut_Vector3f TgtPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_tgtPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointPair_GetMutable_tgtPoint(_Underlying *_this);
                return new(__MR_PointPair_GetMutable_tgtPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in the target point after transforming in world space
        public new unsafe MR.Mut_Vector3f TgtNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_tgtNorm", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointPair_GetMutable_tgtNorm(_Underlying *_this);
                return new(__MR_PointPair_GetMutable_tgtNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between source and target points
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_PointPair_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_PointPair_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// weight of the pair (to prioritize over other pairs)
        public new unsafe ref float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_GetMutable_weight", ExactSpelling = true)]
                extern static float *__MR_PointPair_GetMutable_weight(_Underlying *_this);
                return ref *__MR_PointPair_GetMutable_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointPair() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointPair._Underlying *__MR_PointPair_DefaultConstruct();
            _UnderlyingPtr = __MR_PointPair_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointPair::PointPair`.
        public unsafe PointPair(MR.Const_PointPair _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointPair._Underlying *__MR_PointPair_ConstructFromAnother(MR.PointPair._Underlying *_other);
            _UnderlyingPtr = __MR_PointPair_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointPair::operator=`.
        public unsafe MR.PointPair Assign(MR.Const_PointPair _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPair_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointPair._Underlying *__MR_PointPair_AssignFromAnother(_Underlying *_this, MR.PointPair._Underlying *_other);
            return new(__MR_PointPair_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointPair` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointPair`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointPair`/`Const_PointPair` directly.
    public class _InOptMut_PointPair
    {
        public PointPair? Opt;

        public _InOptMut_PointPair() {}
        public _InOptMut_PointPair(PointPair value) {Opt = value;}
        public static implicit operator _InOptMut_PointPair(PointPair value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointPair` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointPair`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointPair`/`Const_PointPair` to pass it to the function.
    public class _InOptConst_PointPair
    {
        public Const_PointPair? Opt;

        public _InOptConst_PointPair() {}
        public _InOptConst_PointPair(Const_PointPair value) {Opt = value;}
        public static implicit operator _InOptConst_PointPair(Const_PointPair value) {return new(value);}
    }

    /// Simple interface for pairs holder
    /// Generated from class `MR::IPointPairs`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPGroupPairs`
    ///     `MR::PointPairs`
    /// This is the const half of the class.
    public class Const_IPointPairs : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IPointPairs(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_Destroy", ExactSpelling = true)]
            extern static void __MR_IPointPairs_Destroy(_Underlying *_this);
            __MR_IPointPairs_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IPointPairs() {Dispose(false);}

        ///< whether corresponding pair from vec must be considered during minimization
        public unsafe MR.Const_BitSet Active
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_Get_active", ExactSpelling = true)]
                extern static MR.Const_BitSet._Underlying *__MR_IPointPairs_Get_active(_Underlying *_this);
                return new(__MR_IPointPairs_Get_active(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from method `MR::IPointPairs::operator[]`.
        public unsafe MR.Const_ICPPairData Index(ulong _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_index_const", ExactSpelling = true)]
            extern static MR.Const_ICPPairData._Underlying *__MR_IPointPairs_index_const(_Underlying *_this, ulong _1);
            return new(__MR_IPointPairs_index_const(_UnderlyingPtr, _1), is_owning: false);
        }

        /// Generated from method `MR::IPointPairs::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_size", ExactSpelling = true)]
            extern static ulong __MR_IPointPairs_size(_Underlying *_this);
            return __MR_IPointPairs_size(_UnderlyingPtr);
        }
    }

    /// Simple interface for pairs holder
    /// Generated from class `MR::IPointPairs`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPGroupPairs`
    ///     `MR::PointPairs`
    /// This is the non-const half of the class.
    public class IPointPairs : Const_IPointPairs
    {
        internal unsafe IPointPairs(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< whether corresponding pair from vec must be considered during minimization
        public new unsafe MR.BitSet Active
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_GetMutable_active", ExactSpelling = true)]
                extern static MR.BitSet._Underlying *__MR_IPointPairs_GetMutable_active(_Underlying *_this);
                return new(__MR_IPointPairs_GetMutable_active(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from method `MR::IPointPairs::operator[]`.
        public unsafe new MR.ICPPairData Index(ulong _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_index", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_IPointPairs_index(_Underlying *_this, ulong _1);
            return new(__MR_IPointPairs_index(_UnderlyingPtr, _1), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IPointPairs` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IPointPairs`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IPointPairs`/`Const_IPointPairs` directly.
    public class _InOptMut_IPointPairs
    {
        public IPointPairs? Opt;

        public _InOptMut_IPointPairs() {}
        public _InOptMut_IPointPairs(IPointPairs value) {Opt = value;}
        public static implicit operator _InOptMut_IPointPairs(IPointPairs value) {return new(value);}
    }

    /// This is used for optional parameters of class `IPointPairs` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IPointPairs`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IPointPairs`/`Const_IPointPairs` to pass it to the function.
    public class _InOptConst_IPointPairs
    {
        public Const_IPointPairs? Opt;

        public _InOptConst_IPointPairs() {}
        public _InOptConst_IPointPairs(Const_IPointPairs value) {Opt = value;}
        public static implicit operator _InOptConst_IPointPairs(Const_IPointPairs value) {return new(value);}
    }

    /// Generated from class `MR::PointPairs`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointPairs`
    /// This is the const half of the class.
    public class Const_PointPairs : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointPairs(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_Destroy", ExactSpelling = true)]
            extern static void __MR_PointPairs_Destroy(_Underlying *_this);
            __MR_PointPairs_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointPairs() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_IPointPairs(Const_PointPairs self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_UpcastTo_MR_IPointPairs", ExactSpelling = true)]
            extern static MR.Const_IPointPairs._Underlying *__MR_PointPairs_UpcastTo_MR_IPointPairs(_Underlying *_this);
            MR.Const_IPointPairs ret = new(__MR_PointPairs_UpcastTo_MR_IPointPairs(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Downcasts:
        public static unsafe explicit operator Const_PointPairs?(MR.Const_IPointPairs parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_DynamicDowncastTo_MR_PointPairs", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointPairs_DynamicDowncastTo_MR_PointPairs(MR.Const_IPointPairs._Underlying *_this);
            var ptr = __MR_IPointPairs_DynamicDowncastTo_MR_PointPairs(parent._UnderlyingPtr);
            if (ptr is null) return null;
            Const_PointPairs ret = new(ptr, is_owning: false);
            ret._KeepAlive(parent);
            return ret;
        }

        ///< vector of all point pairs both active and not
        public unsafe MR.Std.Const_Vector_MRPointPair Vec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_Get_vec", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRPointPair._Underlying *__MR_PointPairs_Get_vec(_Underlying *_this);
                return new(__MR_PointPairs_Get_vec(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< whether corresponding pair from vec must be considered during minimization
        public unsafe MR.Const_BitSet Active
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_Get_active", ExactSpelling = true)]
                extern static MR.Const_BitSet._Underlying *__MR_PointPairs_Get_active(_Underlying *_this);
                return new(__MR_PointPairs_Get_active(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointPairs() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointPairs._Underlying *__MR_PointPairs_DefaultConstruct();
            _UnderlyingPtr = __MR_PointPairs_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointPairs::PointPairs`.
        public unsafe Const_PointPairs(MR._ByValue_PointPairs _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointPairs._Underlying *__MR_PointPairs_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointPairs._Underlying *_other);
            _UnderlyingPtr = __MR_PointPairs_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointPairs::operator[]`.
        public unsafe MR.Const_ICPPairData Index(ulong idx)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_index_const", ExactSpelling = true)]
            extern static MR.Const_ICPPairData._Underlying *__MR_PointPairs_index_const(_Underlying *_this, ulong idx);
            return new(__MR_PointPairs_index_const(_UnderlyingPtr, idx), is_owning: false);
        }

        /// Generated from method `MR::PointPairs::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_size", ExactSpelling = true)]
            extern static ulong __MR_PointPairs_size(_Underlying *_this);
            return __MR_PointPairs_size(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PointPairs`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointPairs`
    /// This is the non-const half of the class.
    public class PointPairs : Const_PointPairs
    {
        internal unsafe PointPairs(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.IPointPairs(PointPairs self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_UpcastTo_MR_IPointPairs", ExactSpelling = true)]
            extern static MR.IPointPairs._Underlying *__MR_PointPairs_UpcastTo_MR_IPointPairs(_Underlying *_this);
            MR.IPointPairs ret = new(__MR_PointPairs_UpcastTo_MR_IPointPairs(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Downcasts:
        public static unsafe explicit operator PointPairs?(MR.IPointPairs parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_DynamicDowncastTo_MR_PointPairs", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointPairs_DynamicDowncastTo_MR_PointPairs(MR.IPointPairs._Underlying *_this);
            var ptr = __MR_IPointPairs_DynamicDowncastTo_MR_PointPairs(parent._UnderlyingPtr);
            if (ptr is null) return null;
            PointPairs ret = new(ptr, is_owning: false);
            ret._KeepAlive(parent);
            return ret;
        }

        ///< vector of all point pairs both active and not
        public new unsafe MR.Std.Vector_MRPointPair Vec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_GetMutable_vec", ExactSpelling = true)]
                extern static MR.Std.Vector_MRPointPair._Underlying *__MR_PointPairs_GetMutable_vec(_Underlying *_this);
                return new(__MR_PointPairs_GetMutable_vec(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< whether corresponding pair from vec must be considered during minimization
        public new unsafe MR.BitSet Active
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_GetMutable_active", ExactSpelling = true)]
                extern static MR.BitSet._Underlying *__MR_PointPairs_GetMutable_active(_Underlying *_this);
                return new(__MR_PointPairs_GetMutable_active(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointPairs() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointPairs._Underlying *__MR_PointPairs_DefaultConstruct();
            _UnderlyingPtr = __MR_PointPairs_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointPairs::PointPairs`.
        public unsafe PointPairs(MR._ByValue_PointPairs _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointPairs._Underlying *__MR_PointPairs_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointPairs._Underlying *_other);
            _UnderlyingPtr = __MR_PointPairs_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointPairs::operator=`.
        public unsafe MR.PointPairs Assign(MR._ByValue_PointPairs _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointPairs._Underlying *__MR_PointPairs_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointPairs._Underlying *_other);
            return new(__MR_PointPairs_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::PointPairs::operator[]`.
        public unsafe new MR.ICPPairData Index(ulong idx)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointPairs_index", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_PointPairs_index(_Underlying *_this, ulong idx);
            return new(__MR_PointPairs_index(_UnderlyingPtr, idx), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointPairs` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointPairs`/`Const_PointPairs` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointPairs
    {
        internal readonly Const_PointPairs? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointPairs() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointPairs(Const_PointPairs new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointPairs(Const_PointPairs arg) {return new(arg);}
        public _ByValue_PointPairs(MR.Misc._Moved<PointPairs> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointPairs(MR.Misc._Moved<PointPairs> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointPairs` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointPairs`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointPairs`/`Const_PointPairs` directly.
    public class _InOptMut_PointPairs
    {
        public PointPairs? Opt;

        public _InOptMut_PointPairs() {}
        public _InOptMut_PointPairs(PointPairs value) {Opt = value;}
        public static implicit operator _InOptMut_PointPairs(PointPairs value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointPairs` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointPairs`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointPairs`/`Const_PointPairs` to pass it to the function.
    public class _InOptConst_PointPairs
    {
        public Const_PointPairs? Opt;

        public _InOptConst_PointPairs() {}
        public _InOptConst_PointPairs(Const_PointPairs value) {Opt = value;}
        public static implicit operator _InOptConst_PointPairs(Const_PointPairs value) {return new(value);}
    }

    /// Generated from class `MR::NumSum`.
    /// This is the const half of the class.
    public class Const_NumSum : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NumSum(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_Destroy", ExactSpelling = true)]
            extern static void __MR_NumSum_Destroy(_Underlying *_this);
            __MR_NumSum_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NumSum() {Dispose(false);}

        public unsafe int Num
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_Get_num", ExactSpelling = true)]
                extern static int *__MR_NumSum_Get_num(_Underlying *_this);
                return *__MR_NumSum_Get_num(_UnderlyingPtr);
            }
        }

        public unsafe double Sum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_Get_sum", ExactSpelling = true)]
                extern static double *__MR_NumSum_Get_sum(_Underlying *_this);
                return *__MR_NumSum_Get_sum(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NumSum() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_DefaultConstruct();
            _UnderlyingPtr = __MR_NumSum_DefaultConstruct();
        }

        /// Constructs `MR::NumSum` elementwise.
        public unsafe Const_NumSum(int num, double sum) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_ConstructFrom", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_ConstructFrom(int num, double sum);
            _UnderlyingPtr = __MR_NumSum_ConstructFrom(num, sum);
        }

        /// Generated from constructor `MR::NumSum::NumSum`.
        public unsafe Const_NumSum(MR.Const_NumSum _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_ConstructFromAnother(MR.NumSum._Underlying *_other);
            _UnderlyingPtr = __MR_NumSum_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NumSum::rootMeanSqF`.
        public unsafe float RootMeanSqF()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_rootMeanSqF", ExactSpelling = true)]
            extern static float __MR_NumSum_rootMeanSqF(_Underlying *_this);
            return __MR_NumSum_rootMeanSqF(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.NumSum operator+(MR.Const_NumSum a, MR.Const_NumSum b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NumSum", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_add_MR_NumSum(MR.Const_NumSum._Underlying *a, MR.Const_NumSum._Underlying *b);
            return new(__MR_add_MR_NumSum(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true);
        }
    }

    /// Generated from class `MR::NumSum`.
    /// This is the non-const half of the class.
    public class NumSum : Const_NumSum
    {
        internal unsafe NumSum(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref int Num
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_GetMutable_num", ExactSpelling = true)]
                extern static int *__MR_NumSum_GetMutable_num(_Underlying *_this);
                return ref *__MR_NumSum_GetMutable_num(_UnderlyingPtr);
            }
        }

        public new unsafe ref double Sum
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_GetMutable_sum", ExactSpelling = true)]
                extern static double *__MR_NumSum_GetMutable_sum(_Underlying *_this);
                return ref *__MR_NumSum_GetMutable_sum(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NumSum() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_DefaultConstruct();
            _UnderlyingPtr = __MR_NumSum_DefaultConstruct();
        }

        /// Constructs `MR::NumSum` elementwise.
        public unsafe NumSum(int num, double sum) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_ConstructFrom", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_ConstructFrom(int num, double sum);
            _UnderlyingPtr = __MR_NumSum_ConstructFrom(num, sum);
        }

        /// Generated from constructor `MR::NumSum::NumSum`.
        public unsafe NumSum(MR.Const_NumSum _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_ConstructFromAnother(MR.NumSum._Underlying *_other);
            _UnderlyingPtr = __MR_NumSum_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NumSum::operator=`.
        public unsafe MR.NumSum Assign(MR.Const_NumSum _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NumSum_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NumSum._Underlying *__MR_NumSum_AssignFromAnother(_Underlying *_this, MR.NumSum._Underlying *_other);
            return new(__MR_NumSum_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NumSum` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NumSum`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NumSum`/`Const_NumSum` directly.
    public class _InOptMut_NumSum
    {
        public NumSum? Opt;

        public _InOptMut_NumSum() {}
        public _InOptMut_NumSum(NumSum value) {Opt = value;}
        public static implicit operator _InOptMut_NumSum(NumSum value) {return new(value);}
    }

    /// This is used for optional parameters of class `NumSum` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NumSum`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NumSum`/`Const_NumSum` to pass it to the function.
    public class _InOptConst_NumSum
    {
        public Const_NumSum? Opt;

        public _InOptConst_NumSum() {}
        public _InOptConst_NumSum(Const_NumSum value) {Opt = value;}
        public static implicit operator _InOptConst_NumSum(Const_NumSum value) {return new(value);}
    }

    /// parameters of ICP algorithm
    /// Generated from class `MR::ICPProperties`.
    /// This is the const half of the class.
    public class Const_ICPProperties : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ICPProperties(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Destroy", ExactSpelling = true)]
            extern static void __MR_ICPProperties_Destroy(_Underlying *_this);
            __MR_ICPProperties_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ICPProperties() {Dispose(false);}

        /// The method how to update transformation from point pairs, see description of each option in ICPMethod
        public unsafe MR.ICPMethod Method
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_method", ExactSpelling = true)]
                extern static MR.ICPMethod *__MR_ICPProperties_Get_method(_Underlying *_this);
                return *__MR_ICPProperties_Get_method(_UnderlyingPtr);
            }
        }

        // [radians]
        public unsafe float P2plAngleLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_p2plAngleLimit", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_Get_p2plAngleLimit(_Underlying *_this);
                return *__MR_ICPProperties_Get_p2plAngleLimit(_UnderlyingPtr);
            }
        }

        /// Scaling during one iteration of ICPMethod::PointToPlane will be limited by this value.
        /// This is to reduce possible instability.
        public unsafe float P2plScaleLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_p2plScaleLimit", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_Get_p2plScaleLimit(_Underlying *_this);
                return *__MR_ICPProperties_Get_p2plScaleLimit(_UnderlyingPtr);
            }
        }

        // in [-1,1]
        public unsafe float CosThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_cosThreshold", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_Get_cosThreshold(_Underlying *_this);
                return *__MR_ICPProperties_Get_cosThreshold(_UnderlyingPtr);
            }
        }

        // [distance^2]
        public unsafe float DistThresholdSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_distThresholdSq", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_Get_distThresholdSq(_Underlying *_this);
                return *__MR_ICPProperties_Get_distThresholdSq(_UnderlyingPtr);
            }
        }

        // dimensionless
        public unsafe float FarDistFactor
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_farDistFactor", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_Get_farDistFactor(_Underlying *_this);
                return *__MR_ICPProperties_Get_farDistFactor(_UnderlyingPtr);
            }
        }

        /// Selects the group of transformations, where to find a solution (e.g. with scaling or without, with rotation or without, ...).
        /// See the description of each option in ICPMode.
        public unsafe MR.ICPMode IcpMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_icpMode", ExactSpelling = true)]
                extern static MR.ICPMode *__MR_ICPProperties_Get_icpMode(_Underlying *_this);
                return *__MR_ICPProperties_Get_icpMode(_UnderlyingPtr);
            }
        }

        /// Additional parameter for ICPMode::OrthogonalAxis and ICPMode::FixedAxis transformation groups.
        public unsafe MR.Const_Vector3f FixedRotationAxis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_fixedRotationAxis", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPProperties_Get_fixedRotationAxis(_Underlying *_this);
                return new(__MR_ICPProperties_Get_fixedRotationAxis(_UnderlyingPtr), is_owning: false);
            }
        }

        /// The maximum number of iterations that the algorithm can perform.
        /// Increase this parameter if you need a higher precision or if initial approximation is not very precise.
        public unsafe int IterLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_iterLimit", ExactSpelling = true)]
                extern static int *__MR_ICPProperties_Get_iterLimit(_Underlying *_this);
                return *__MR_ICPProperties_Get_iterLimit(_UnderlyingPtr);
            }
        }

        /// The algorithm will stop before making all (iterLimit) iterations, if there were
        /// consecutive (badIterStopCount) iterations, during which the average distance between points in active pairs did not diminish.
        public unsafe int BadIterStopCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_badIterStopCount", ExactSpelling = true)]
                extern static int *__MR_ICPProperties_Get_badIterStopCount(_Underlying *_this);
                return *__MR_ICPProperties_Get_badIterStopCount(_UnderlyingPtr);
            }
        }

        // [distance]
        public unsafe float ExitVal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_exitVal", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_Get_exitVal(_Underlying *_this);
                return *__MR_ICPProperties_Get_exitVal(_UnderlyingPtr);
            }
        }

        /// A pair of points is activated only if both points in the pair are mutually closest (reciprocity test passed),
        /// some papers recommend this mode for filtering out wrong pairs, but it can be too aggressive and deactivate (almost) all pairs.
        public unsafe bool MutualClosest
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_Get_mutualClosest", ExactSpelling = true)]
                extern static bool *__MR_ICPProperties_Get_mutualClosest(_Underlying *_this);
                return *__MR_ICPProperties_Get_mutualClosest(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ICPProperties() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPProperties_DefaultConstruct();
        }

        /// Constructs `MR::ICPProperties` elementwise.
        public unsafe Const_ICPProperties(MR.ICPMethod method, float p2plAngleLimit, float p2plScaleLimit, float cosThreshold, float distThresholdSq, float farDistFactor, MR.ICPMode icpMode, MR.Vector3f fixedRotationAxis, int iterLimit, int badIterStopCount, float exitVal, bool mutualClosest) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_ConstructFrom", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_ConstructFrom(MR.ICPMethod method, float p2plAngleLimit, float p2plScaleLimit, float cosThreshold, float distThresholdSq, float farDistFactor, MR.ICPMode icpMode, MR.Vector3f fixedRotationAxis, int iterLimit, int badIterStopCount, float exitVal, byte mutualClosest);
            _UnderlyingPtr = __MR_ICPProperties_ConstructFrom(method, p2plAngleLimit, p2plScaleLimit, cosThreshold, distThresholdSq, farDistFactor, icpMode, fixedRotationAxis, iterLimit, badIterStopCount, exitVal, mutualClosest ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::ICPProperties::ICPProperties`.
        public unsafe Const_ICPProperties(MR.Const_ICPProperties _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_ConstructFromAnother(MR.ICPProperties._Underlying *_other);
            _UnderlyingPtr = __MR_ICPProperties_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// parameters of ICP algorithm
    /// Generated from class `MR::ICPProperties`.
    /// This is the non-const half of the class.
    public class ICPProperties : Const_ICPProperties
    {
        internal unsafe ICPProperties(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// The method how to update transformation from point pairs, see description of each option in ICPMethod
        public new unsafe ref MR.ICPMethod Method
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_method", ExactSpelling = true)]
                extern static MR.ICPMethod *__MR_ICPProperties_GetMutable_method(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_method(_UnderlyingPtr);
            }
        }

        // [radians]
        public new unsafe ref float P2plAngleLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_p2plAngleLimit", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_GetMutable_p2plAngleLimit(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_p2plAngleLimit(_UnderlyingPtr);
            }
        }

        /// Scaling during one iteration of ICPMethod::PointToPlane will be limited by this value.
        /// This is to reduce possible instability.
        public new unsafe ref float P2plScaleLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_p2plScaleLimit", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_GetMutable_p2plScaleLimit(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_p2plScaleLimit(_UnderlyingPtr);
            }
        }

        // in [-1,1]
        public new unsafe ref float CosThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_cosThreshold", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_GetMutable_cosThreshold(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_cosThreshold(_UnderlyingPtr);
            }
        }

        // [distance^2]
        public new unsafe ref float DistThresholdSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_distThresholdSq", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_GetMutable_distThresholdSq(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_distThresholdSq(_UnderlyingPtr);
            }
        }

        // dimensionless
        public new unsafe ref float FarDistFactor
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_farDistFactor", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_GetMutable_farDistFactor(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_farDistFactor(_UnderlyingPtr);
            }
        }

        /// Selects the group of transformations, where to find a solution (e.g. with scaling or without, with rotation or without, ...).
        /// See the description of each option in ICPMode.
        public new unsafe ref MR.ICPMode IcpMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_icpMode", ExactSpelling = true)]
                extern static MR.ICPMode *__MR_ICPProperties_GetMutable_icpMode(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_icpMode(_UnderlyingPtr);
            }
        }

        /// Additional parameter for ICPMode::OrthogonalAxis and ICPMode::FixedAxis transformation groups.
        public new unsafe MR.Mut_Vector3f FixedRotationAxis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_fixedRotationAxis", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPProperties_GetMutable_fixedRotationAxis(_Underlying *_this);
                return new(__MR_ICPProperties_GetMutable_fixedRotationAxis(_UnderlyingPtr), is_owning: false);
            }
        }

        /// The maximum number of iterations that the algorithm can perform.
        /// Increase this parameter if you need a higher precision or if initial approximation is not very precise.
        public new unsafe ref int IterLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_iterLimit", ExactSpelling = true)]
                extern static int *__MR_ICPProperties_GetMutable_iterLimit(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_iterLimit(_UnderlyingPtr);
            }
        }

        /// The algorithm will stop before making all (iterLimit) iterations, if there were
        /// consecutive (badIterStopCount) iterations, during which the average distance between points in active pairs did not diminish.
        public new unsafe ref int BadIterStopCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_badIterStopCount", ExactSpelling = true)]
                extern static int *__MR_ICPProperties_GetMutable_badIterStopCount(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_badIterStopCount(_UnderlyingPtr);
            }
        }

        // [distance]
        public new unsafe ref float ExitVal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_exitVal", ExactSpelling = true)]
                extern static float *__MR_ICPProperties_GetMutable_exitVal(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_exitVal(_UnderlyingPtr);
            }
        }

        /// A pair of points is activated only if both points in the pair are mutually closest (reciprocity test passed),
        /// some papers recommend this mode for filtering out wrong pairs, but it can be too aggressive and deactivate (almost) all pairs.
        public new unsafe ref bool MutualClosest
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_GetMutable_mutualClosest", ExactSpelling = true)]
                extern static bool *__MR_ICPProperties_GetMutable_mutualClosest(_Underlying *_this);
                return ref *__MR_ICPProperties_GetMutable_mutualClosest(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ICPProperties() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPProperties_DefaultConstruct();
        }

        /// Constructs `MR::ICPProperties` elementwise.
        public unsafe ICPProperties(MR.ICPMethod method, float p2plAngleLimit, float p2plScaleLimit, float cosThreshold, float distThresholdSq, float farDistFactor, MR.ICPMode icpMode, MR.Vector3f fixedRotationAxis, int iterLimit, int badIterStopCount, float exitVal, bool mutualClosest) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_ConstructFrom", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_ConstructFrom(MR.ICPMethod method, float p2plAngleLimit, float p2plScaleLimit, float cosThreshold, float distThresholdSq, float farDistFactor, MR.ICPMode icpMode, MR.Vector3f fixedRotationAxis, int iterLimit, int badIterStopCount, float exitVal, byte mutualClosest);
            _UnderlyingPtr = __MR_ICPProperties_ConstructFrom(method, p2plAngleLimit, p2plScaleLimit, cosThreshold, distThresholdSq, farDistFactor, icpMode, fixedRotationAxis, iterLimit, badIterStopCount, exitVal, mutualClosest ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::ICPProperties::ICPProperties`.
        public unsafe ICPProperties(MR.Const_ICPProperties _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_ConstructFromAnother(MR.ICPProperties._Underlying *_other);
            _UnderlyingPtr = __MR_ICPProperties_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ICPProperties::operator=`.
        public unsafe MR.ICPProperties Assign(MR.Const_ICPProperties _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPProperties_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ICPProperties._Underlying *__MR_ICPProperties_AssignFromAnother(_Underlying *_this, MR.ICPProperties._Underlying *_other);
            return new(__MR_ICPProperties_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ICPProperties` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ICPProperties`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPProperties`/`Const_ICPProperties` directly.
    public class _InOptMut_ICPProperties
    {
        public ICPProperties? Opt;

        public _InOptMut_ICPProperties() {}
        public _InOptMut_ICPProperties(ICPProperties value) {Opt = value;}
        public static implicit operator _InOptMut_ICPProperties(ICPProperties value) {return new(value);}
    }

    /// This is used for optional parameters of class `ICPProperties` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ICPProperties`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPProperties`/`Const_ICPProperties` to pass it to the function.
    public class _InOptConst_ICPProperties
    {
        public Const_ICPProperties? Opt;

        public _InOptConst_ICPProperties() {}
        public _InOptConst_ICPProperties(Const_ICPProperties value) {Opt = value;}
        public static implicit operator _InOptConst_ICPProperties(Const_ICPProperties value) {return new(value);}
    }

    /// This class allows you to register two object with similar shape using
    /// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
    /// \snippet cpp-examples/MeshICP.dox.cpp 0
    /// Generated from class `MR::ICP`.
    /// This is the const half of the class.
    public class Const_ICP : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ICP(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Destroy", ExactSpelling = true)]
            extern static void __MR_ICP_Destroy(_Underlying *_this);
            __MR_ICP_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ICP() {Dispose(false);}

        /// Generated from constructor `MR::ICP::ICP`.
        public unsafe Const_ICP(MR._ByValue_ICP _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ICP._Underlying *_other);
            _UnderlyingPtr = __MR_ICP_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Constructs ICP framework with given sample points on both objects
        /// \param flt floating object
        /// \param ref reference object
        /// \param fltXf transformation from floating object space to global space
        /// \param refXf transformation from reference object space to global space
        /// \param fltSamples samples on floating object to find projections on the reference object during the algorithm
        /// \param refSamples samples on reference object to find projections on the floating object during the algorithm
        /// Generated from constructor `MR::ICP::ICP`.
        /// Parameter `fltSamples` defaults to `{}`.
        /// Parameter `refSamples` defaults to `{}`.
        public unsafe Const_ICP(MR.Const_MeshOrPoints flt, MR.Const_MeshOrPoints ref_, MR.Const_AffineXf3f fltXf, MR.Const_AffineXf3f refXf, MR.Const_VertBitSet? fltSamples = null, MR.Const_VertBitSet? refSamples = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_6", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_6(MR.Const_MeshOrPoints._Underlying *flt, MR.Const_MeshOrPoints._Underlying *ref_, MR.Const_AffineXf3f._Underlying *fltXf, MR.Const_AffineXf3f._Underlying *refXf, MR.Const_VertBitSet._Underlying *fltSamples, MR.Const_VertBitSet._Underlying *refSamples);
            _UnderlyingPtr = __MR_ICP_Construct_6(flt._UnderlyingPtr, ref_._UnderlyingPtr, fltXf._UnderlyingPtr, refXf._UnderlyingPtr, fltSamples is not null ? fltSamples._UnderlyingPtr : null, refSamples is not null ? refSamples._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ICP::ICP`.
        /// Parameter `fltSamples` defaults to `{}`.
        /// Parameter `refSamples` defaults to `{}`.
        public unsafe Const_ICP(MR.Const_MeshOrPointsXf flt, MR.Const_MeshOrPointsXf ref_, MR.Const_VertBitSet? fltSamples = null, MR.Const_VertBitSet? refSamples = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_4", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_4(MR.Const_MeshOrPointsXf._Underlying *flt, MR.Const_MeshOrPointsXf._Underlying *ref_, MR.Const_VertBitSet._Underlying *fltSamples, MR.Const_VertBitSet._Underlying *refSamples);
            _UnderlyingPtr = __MR_ICP_Construct_4(flt._UnderlyingPtr, ref_._UnderlyingPtr, fltSamples is not null ? fltSamples._UnderlyingPtr : null, refSamples is not null ? refSamples._UnderlyingPtr : null);
        }

        /// Constructs ICP framework with automatic points sampling on both objects
        /// \param flt floating object
        /// \param ref reference object
        /// \param fltXf transformation from floating object space to global space
        /// \param refXf transformation from reference object space to global space
        /// \param samplingVoxelSize approximate distance between samples on each of two objects
        /// Generated from constructor `MR::ICP::ICP`.
        public unsafe Const_ICP(MR.Const_MeshOrPoints flt, MR.Const_MeshOrPoints ref_, MR.Const_AffineXf3f fltXf, MR.Const_AffineXf3f refXf, float samplingVoxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_5", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_5(MR.Const_MeshOrPoints._Underlying *flt, MR.Const_MeshOrPoints._Underlying *ref_, MR.Const_AffineXf3f._Underlying *fltXf, MR.Const_AffineXf3f._Underlying *refXf, float samplingVoxelSize);
            _UnderlyingPtr = __MR_ICP_Construct_5(flt._UnderlyingPtr, ref_._UnderlyingPtr, fltXf._UnderlyingPtr, refXf._UnderlyingPtr, samplingVoxelSize);
        }

        /// Generated from constructor `MR::ICP::ICP`.
        public unsafe Const_ICP(MR.Const_MeshOrPointsXf flt, MR.Const_MeshOrPointsXf ref_, float samplingVoxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_3", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_3(MR.Const_MeshOrPointsXf._Underlying *flt, MR.Const_MeshOrPointsXf._Underlying *ref_, float samplingVoxelSize);
            _UnderlyingPtr = __MR_ICP_Construct_3(flt._UnderlyingPtr, ref_._UnderlyingPtr, samplingVoxelSize);
        }

        /// Generated from method `MR::ICP::getParams`.
        public unsafe MR.Const_ICPProperties GetParams()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getParams", ExactSpelling = true)]
            extern static MR.Const_ICPProperties._Underlying *__MR_ICP_getParams(_Underlying *_this);
            return new(__MR_ICP_getParams(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ICP::getStatusInfo`.
        public unsafe MR.Misc._Moved<MR.Std.String> GetStatusInfo()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getStatusInfo", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ICP_getStatusInfo(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ICP_getStatusInfo(_UnderlyingPtr), is_owning: true));
        }

        /// computes the number of samples able to form pairs
        /// Generated from method `MR::ICP::getNumSamples`.
        public unsafe ulong GetNumSamples()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getNumSamples", ExactSpelling = true)]
            extern static ulong __MR_ICP_getNumSamples(_Underlying *_this);
            return __MR_ICP_getNumSamples(_UnderlyingPtr);
        }

        /// computes the number of active point pairs
        /// Generated from method `MR::ICP::getNumActivePairs`.
        public unsafe ulong GetNumActivePairs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getNumActivePairs", ExactSpelling = true)]
            extern static ulong __MR_ICP_getNumActivePairs(_Underlying *_this);
            return __MR_ICP_getNumActivePairs(_UnderlyingPtr);
        }

        /// computes root-mean-square deviation between points
        /// Generated from method `MR::ICP::getMeanSqDistToPoint`.
        public unsafe float GetMeanSqDistToPoint()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getMeanSqDistToPoint", ExactSpelling = true)]
            extern static float __MR_ICP_getMeanSqDistToPoint(_Underlying *_this);
            return __MR_ICP_getMeanSqDistToPoint(_UnderlyingPtr);
        }

        /// computes root-mean-square deviation from points to target planes
        /// Generated from method `MR::ICP::getMeanSqDistToPlane`.
        public unsafe float GetMeanSqDistToPlane()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getMeanSqDistToPlane", ExactSpelling = true)]
            extern static float __MR_ICP_getMeanSqDistToPlane(_Underlying *_this);
            return __MR_ICP_getMeanSqDistToPlane(_UnderlyingPtr);
        }

        /// returns current pairs formed from samples on floating object and projections on reference object
        /// Generated from method `MR::ICP::getFlt2RefPairs`.
        public unsafe MR.Const_PointPairs GetFlt2RefPairs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getFlt2RefPairs", ExactSpelling = true)]
            extern static MR.Const_PointPairs._Underlying *__MR_ICP_getFlt2RefPairs(_Underlying *_this);
            return new(__MR_ICP_getFlt2RefPairs(_UnderlyingPtr), is_owning: false);
        }

        /// returns current pairs formed from samples on reference object and projections on floating object
        /// Generated from method `MR::ICP::getRef2FltPairs`.
        public unsafe MR.Const_PointPairs GetRef2FltPairs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_getRef2FltPairs", ExactSpelling = true)]
            extern static MR.Const_PointPairs._Underlying *__MR_ICP_getRef2FltPairs(_Underlying *_this);
            return new(__MR_ICP_getRef2FltPairs(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This class allows you to register two object with similar shape using
    /// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
    /// \snippet cpp-examples/MeshICP.dox.cpp 0
    /// Generated from class `MR::ICP`.
    /// This is the non-const half of the class.
    public class ICP : Const_ICP
    {
        internal unsafe ICP(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::ICP::ICP`.
        public unsafe ICP(MR._ByValue_ICP _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ICP._Underlying *_other);
            _UnderlyingPtr = __MR_ICP_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Constructs ICP framework with given sample points on both objects
        /// \param flt floating object
        /// \param ref reference object
        /// \param fltXf transformation from floating object space to global space
        /// \param refXf transformation from reference object space to global space
        /// \param fltSamples samples on floating object to find projections on the reference object during the algorithm
        /// \param refSamples samples on reference object to find projections on the floating object during the algorithm
        /// Generated from constructor `MR::ICP::ICP`.
        /// Parameter `fltSamples` defaults to `{}`.
        /// Parameter `refSamples` defaults to `{}`.
        public unsafe ICP(MR.Const_MeshOrPoints flt, MR.Const_MeshOrPoints ref_, MR.Const_AffineXf3f fltXf, MR.Const_AffineXf3f refXf, MR.Const_VertBitSet? fltSamples = null, MR.Const_VertBitSet? refSamples = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_6", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_6(MR.Const_MeshOrPoints._Underlying *flt, MR.Const_MeshOrPoints._Underlying *ref_, MR.Const_AffineXf3f._Underlying *fltXf, MR.Const_AffineXf3f._Underlying *refXf, MR.Const_VertBitSet._Underlying *fltSamples, MR.Const_VertBitSet._Underlying *refSamples);
            _UnderlyingPtr = __MR_ICP_Construct_6(flt._UnderlyingPtr, ref_._UnderlyingPtr, fltXf._UnderlyingPtr, refXf._UnderlyingPtr, fltSamples is not null ? fltSamples._UnderlyingPtr : null, refSamples is not null ? refSamples._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ICP::ICP`.
        /// Parameter `fltSamples` defaults to `{}`.
        /// Parameter `refSamples` defaults to `{}`.
        public unsafe ICP(MR.Const_MeshOrPointsXf flt, MR.Const_MeshOrPointsXf ref_, MR.Const_VertBitSet? fltSamples = null, MR.Const_VertBitSet? refSamples = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_4", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_4(MR.Const_MeshOrPointsXf._Underlying *flt, MR.Const_MeshOrPointsXf._Underlying *ref_, MR.Const_VertBitSet._Underlying *fltSamples, MR.Const_VertBitSet._Underlying *refSamples);
            _UnderlyingPtr = __MR_ICP_Construct_4(flt._UnderlyingPtr, ref_._UnderlyingPtr, fltSamples is not null ? fltSamples._UnderlyingPtr : null, refSamples is not null ? refSamples._UnderlyingPtr : null);
        }

        /// Constructs ICP framework with automatic points sampling on both objects
        /// \param flt floating object
        /// \param ref reference object
        /// \param fltXf transformation from floating object space to global space
        /// \param refXf transformation from reference object space to global space
        /// \param samplingVoxelSize approximate distance between samples on each of two objects
        /// Generated from constructor `MR::ICP::ICP`.
        public unsafe ICP(MR.Const_MeshOrPoints flt, MR.Const_MeshOrPoints ref_, MR.Const_AffineXf3f fltXf, MR.Const_AffineXf3f refXf, float samplingVoxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_5", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_5(MR.Const_MeshOrPoints._Underlying *flt, MR.Const_MeshOrPoints._Underlying *ref_, MR.Const_AffineXf3f._Underlying *fltXf, MR.Const_AffineXf3f._Underlying *refXf, float samplingVoxelSize);
            _UnderlyingPtr = __MR_ICP_Construct_5(flt._UnderlyingPtr, ref_._UnderlyingPtr, fltXf._UnderlyingPtr, refXf._UnderlyingPtr, samplingVoxelSize);
        }

        /// Generated from constructor `MR::ICP::ICP`.
        public unsafe ICP(MR.Const_MeshOrPointsXf flt, MR.Const_MeshOrPointsXf ref_, float samplingVoxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_Construct_3", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_Construct_3(MR.Const_MeshOrPointsXf._Underlying *flt, MR.Const_MeshOrPointsXf._Underlying *ref_, float samplingVoxelSize);
            _UnderlyingPtr = __MR_ICP_Construct_3(flt._UnderlyingPtr, ref_._UnderlyingPtr, samplingVoxelSize);
        }

        /// Generated from method `MR::ICP::operator=`.
        public unsafe MR.ICP Assign(MR._ByValue_ICP _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ICP._Underlying *__MR_ICP_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ICP._Underlying *_other);
            return new(__MR_ICP_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// tune algorithm params before run calculateTransformation()
        /// Generated from method `MR::ICP::setParams`.
        public unsafe void SetParams(MR.Const_ICPProperties prop)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setParams", ExactSpelling = true)]
            extern static void __MR_ICP_setParams(_Underlying *_this, MR.Const_ICPProperties._Underlying *prop);
            __MR_ICP_setParams(_UnderlyingPtr, prop._UnderlyingPtr);
        }

        /// Generated from method `MR::ICP::setCosineLimit`.
        public unsafe void SetCosineLimit(float cos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setCosineLimit", ExactSpelling = true)]
            extern static void __MR_ICP_setCosineLimit(_Underlying *_this, float cos);
            __MR_ICP_setCosineLimit(_UnderlyingPtr, cos);
        }

        /// Generated from method `MR::ICP::setDistanceLimit`.
        public unsafe void SetDistanceLimit(float dist)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setDistanceLimit", ExactSpelling = true)]
            extern static void __MR_ICP_setDistanceLimit(_Underlying *_this, float dist);
            __MR_ICP_setDistanceLimit(_UnderlyingPtr, dist);
        }

        /// Generated from method `MR::ICP::setBadIterCount`.
        public unsafe void SetBadIterCount(int iter)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setBadIterCount", ExactSpelling = true)]
            extern static void __MR_ICP_setBadIterCount(_Underlying *_this, int iter);
            __MR_ICP_setBadIterCount(_UnderlyingPtr, iter);
        }

        /// Generated from method `MR::ICP::setFarDistFactor`.
        public unsafe void SetFarDistFactor(float factor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setFarDistFactor", ExactSpelling = true)]
            extern static void __MR_ICP_setFarDistFactor(_Underlying *_this, float factor);
            __MR_ICP_setFarDistFactor(_UnderlyingPtr, factor);
        }

        /// select pairs with origin samples on floating object
        /// Generated from method `MR::ICP::setFltSamples`.
        public unsafe void SetFltSamples(MR.Const_VertBitSet fltSamples)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setFltSamples", ExactSpelling = true)]
            extern static void __MR_ICP_setFltSamples(_Underlying *_this, MR.Const_VertBitSet._Underlying *fltSamples);
            __MR_ICP_setFltSamples(_UnderlyingPtr, fltSamples._UnderlyingPtr);
        }

        /// Generated from method `MR::ICP::sampleFltPoints`.
        public unsafe void SampleFltPoints(float samplingVoxelSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_sampleFltPoints", ExactSpelling = true)]
            extern static void __MR_ICP_sampleFltPoints(_Underlying *_this, float samplingVoxelSize);
            __MR_ICP_sampleFltPoints(_UnderlyingPtr, samplingVoxelSize);
        }

        /// select pairs with origin samples on reference object
        /// Generated from method `MR::ICP::setRefSamples`.
        public unsafe void SetRefSamples(MR.Const_VertBitSet refSamples)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setRefSamples", ExactSpelling = true)]
            extern static void __MR_ICP_setRefSamples(_Underlying *_this, MR.Const_VertBitSet._Underlying *refSamples);
            __MR_ICP_setRefSamples(_UnderlyingPtr, refSamples._UnderlyingPtr);
        }

        /// Generated from method `MR::ICP::sampleRefPoints`.
        public unsafe void SampleRefPoints(float samplingVoxelSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_sampleRefPoints", ExactSpelling = true)]
            extern static void __MR_ICP_sampleRefPoints(_Underlying *_this, float samplingVoxelSize);
            __MR_ICP_sampleRefPoints(_UnderlyingPtr, samplingVoxelSize);
        }

        /// select pairs with origin samples on both objects
        /// Generated from method `MR::ICP::samplePoints`.
        public unsafe void SamplePoints(float samplingVoxelSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_samplePoints", ExactSpelling = true)]
            extern static void __MR_ICP_samplePoints(_Underlying *_this, float samplingVoxelSize);
            __MR_ICP_samplePoints(_UnderlyingPtr, samplingVoxelSize);
        }

        /// sets to-world transformations both for floating and reference objects
        /// Generated from method `MR::ICP::setXfs`.
        public unsafe void SetXfs(MR.Const_AffineXf3f fltXf, MR.Const_AffineXf3f refXf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setXfs", ExactSpelling = true)]
            extern static void __MR_ICP_setXfs(_Underlying *_this, MR.Const_AffineXf3f._Underlying *fltXf, MR.Const_AffineXf3f._Underlying *refXf);
            __MR_ICP_setXfs(_UnderlyingPtr, fltXf._UnderlyingPtr, refXf._UnderlyingPtr);
        }

        /// sets to-world transformation for the floating object
        /// Generated from method `MR::ICP::setFloatXf`.
        public unsafe void SetFloatXf(MR.Const_AffineXf3f fltXf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_setFloatXf", ExactSpelling = true)]
            extern static void __MR_ICP_setFloatXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *fltXf);
            __MR_ICP_setFloatXf(_UnderlyingPtr, fltXf._UnderlyingPtr);
        }

        /// automatically selects initial transformation for the floating object
        /// based on covariance matrices of both floating and reference objects;
        /// applies the transformation to the floating object and returns it
        /// Generated from method `MR::ICP::autoSelectFloatXf`.
        public unsafe MR.AffineXf3f AutoSelectFloatXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_autoSelectFloatXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ICP_autoSelectFloatXf(_Underlying *_this);
            return __MR_ICP_autoSelectFloatXf(_UnderlyingPtr);
        }

        /// recompute point pairs after manual change of transformations or parameters
        /// Generated from method `MR::ICP::updatePointPairs`.
        public unsafe void UpdatePointPairs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_updatePointPairs", ExactSpelling = true)]
            extern static void __MR_ICP_updatePointPairs(_Underlying *_this);
            __MR_ICP_updatePointPairs(_UnderlyingPtr);
        }

        /// runs ICP algorithm given input objects, transformations, and parameters;
        /// \return adjusted transformation of the floating object to match reference object
        /// Generated from method `MR::ICP::calculateTransformation`.
        public unsafe MR.AffineXf3f CalculateTransformation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICP_calculateTransformation", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ICP_calculateTransformation(_Underlying *_this);
            return __MR_ICP_calculateTransformation(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ICP` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ICP`/`Const_ICP` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ICP
    {
        internal readonly Const_ICP? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ICP(Const_ICP new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ICP(Const_ICP arg) {return new(arg);}
        public _ByValue_ICP(MR.Misc._Moved<ICP> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ICP(MR.Misc._Moved<ICP> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ICP` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ICP`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICP`/`Const_ICP` directly.
    public class _InOptMut_ICP
    {
        public ICP? Opt;

        public _InOptMut_ICP() {}
        public _InOptMut_ICP(ICP value) {Opt = value;}
        public static implicit operator _InOptMut_ICP(ICP value) {return new(value);}
    }

    /// This is used for optional parameters of class `ICP` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ICP`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICP`/`Const_ICP` to pass it to the function.
    public class _InOptConst_ICP
    {
        public Const_ICP? Opt;

        public _InOptConst_ICP() {}
        public _InOptConst_ICP(Const_ICP value) {Opt = value;}
        public static implicit operator _InOptConst_ICP(Const_ICP value) {return new(value);}
    }

    /// returns the number of samples able to form pairs
    /// Generated from function `MR::getNumSamples`.
    public static unsafe ulong GetNumSamples(MR.Const_IPointPairs pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getNumSamples", ExactSpelling = true)]
        extern static ulong __MR_getNumSamples(MR.Const_IPointPairs._Underlying *pairs);
        return __MR_getNumSamples(pairs._UnderlyingPtr);
    }

    /// computes the number of active pairs
    /// Generated from function `MR::getNumActivePairs`.
    public static unsafe ulong GetNumActivePairs(MR.Const_IPointPairs pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getNumActivePairs", ExactSpelling = true)]
        extern static ulong __MR_getNumActivePairs(MR.Const_IPointPairs._Underlying *pairs);
        return __MR_getNumActivePairs(pairs._UnderlyingPtr);
    }

    /// computes the number of active pairs and the sum of squared distances between points
    /// or the difference between the squared distances between points and inaccuracy
    /// Generated from function `MR::getSumSqDistToPoint`.
    public static unsafe MR.NumSum GetSumSqDistToPoint(MR.Const_IPointPairs pairs, double? inaccuracy = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getSumSqDistToPoint", ExactSpelling = true)]
        extern static MR.NumSum._Underlying *__MR_getSumSqDistToPoint(MR.Const_IPointPairs._Underlying *pairs, double *inaccuracy);
        double __deref_inaccuracy = inaccuracy.GetValueOrDefault();
        return new(__MR_getSumSqDistToPoint(pairs._UnderlyingPtr, inaccuracy.HasValue ? &__deref_inaccuracy : null), is_owning: true);
    }

    /// computes the number of active pairs and the sum of squared deviation from points to target planes
    /// or the difference between the squared distances between points to target planes and inaccuracy
    /// Generated from function `MR::getSumSqDistToPlane`.
    public static unsafe MR.NumSum GetSumSqDistToPlane(MR.Const_IPointPairs pairs, double? inaccuracy = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getSumSqDistToPlane", ExactSpelling = true)]
        extern static MR.NumSum._Underlying *__MR_getSumSqDistToPlane(MR.Const_IPointPairs._Underlying *pairs, double *inaccuracy);
        double __deref_inaccuracy = inaccuracy.GetValueOrDefault();
        return new(__MR_getSumSqDistToPlane(pairs._UnderlyingPtr, inaccuracy.HasValue ? &__deref_inaccuracy : null), is_owning: true);
    }

    /// computes root-mean-square deviation between points
    /// Generated from function `MR::getMeanSqDistToPoint`.
    public static unsafe float GetMeanSqDistToPoint(MR.Const_IPointPairs pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMeanSqDistToPoint", ExactSpelling = true)]
        extern static float __MR_getMeanSqDistToPoint(MR.Const_IPointPairs._Underlying *pairs);
        return __MR_getMeanSqDistToPoint(pairs._UnderlyingPtr);
    }

    /// computes root-mean-square deviation from points to target planes
    /// Generated from function `MR::getMeanSqDistToPlane`.
    public static unsafe float GetMeanSqDistToPlane(MR.Const_IPointPairs pairs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMeanSqDistToPlane", ExactSpelling = true)]
        extern static float __MR_getMeanSqDistToPlane(MR.Const_IPointPairs._Underlying *pairs);
        return __MR_getMeanSqDistToPlane(pairs._UnderlyingPtr);
    }

    /// returns status info string
    /// Generated from function `MR::getICPStatusInfo`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetICPStatusInfo(int iterations, MR.ICPExitType exitType)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getICPStatusInfo", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_getICPStatusInfo(int iterations, MR.ICPExitType exitType);
        return MR.Misc.Move(new MR.Std.String(__MR_getICPStatusInfo(iterations, exitType), is_owning: true));
    }

    /// given prepared (p2pl) object, finds the best transformation from it of given type with given limitations on rotation angle and global scale
    /// Generated from function `MR::getAligningXf`.
    public static unsafe MR.AffineXf3d GetAligningXf(MR.Const_PointToPlaneAligningTransform p2pl, MR.ICPMode mode, float angleLimit, float scaleLimit, MR.Const_Vector3f fixedRotationAxis)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getAligningXf", ExactSpelling = true)]
        extern static MR.AffineXf3d __MR_getAligningXf(MR.Const_PointToPlaneAligningTransform._Underlying *p2pl, MR.ICPMode mode, float angleLimit, float scaleLimit, MR.Const_Vector3f._Underlying *fixedRotationAxis);
        return __MR_getAligningXf(p2pl._UnderlyingPtr, mode, angleLimit, scaleLimit, fixedRotationAxis._UnderlyingPtr);
    }

    /// reset active bit if pair distance is further than maxDistSq
    /// Generated from function `MR::deactivateFarPairs`.
    public static unsafe ulong DeactivateFarPairs(MR.IPointPairs pairs, float maxDistSq)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deactivateFarPairs", ExactSpelling = true)]
        extern static ulong __MR_deactivateFarPairs(MR.IPointPairs._Underlying *pairs, float maxDistSq);
        return __MR_deactivateFarPairs(pairs._UnderlyingPtr, maxDistSq);
    }

    /// in each pair updates the target data and performs basic filtering (activation)
    /// Generated from function `MR::updatePointPairs`.
    public static unsafe void UpdatePointPairs(MR.PointPairs pairs, MR.Const_MeshOrPointsXf src, MR.Const_MeshOrPointsXf tgt, float cosThreshold, float distThresholdSq, bool mutualClosest)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_updatePointPairs", ExactSpelling = true)]
        extern static void __MR_updatePointPairs(MR.PointPairs._Underlying *pairs, MR.Const_MeshOrPointsXf._Underlying *src, MR.Const_MeshOrPointsXf._Underlying *tgt, float cosThreshold, float distThresholdSq, byte mutualClosest);
        __MR_updatePointPairs(pairs._UnderlyingPtr, src._UnderlyingPtr, tgt._UnderlyingPtr, cosThreshold, distThresholdSq, mutualClosest ? (byte)1 : (byte)0);
    }
}
