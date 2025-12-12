public static partial class MR
{
    /// Generated from class `MR::ICPGroupPair`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPPairData`
    /// This is the const half of the class.
    public class Const_ICPGroupPair : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ICPGroupPair(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Destroy", ExactSpelling = true)]
            extern static void __MR_ICPGroupPair_Destroy(_Underlying *_this);
            __MR_ICPGroupPair_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ICPGroupPair() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ICPPairData(Const_ICPGroupPair self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_UpcastTo_MR_ICPPairData", ExactSpelling = true)]
            extern static MR.Const_ICPPairData._Underlying *__MR_ICPGroupPair_UpcastTo_MR_ICPPairData(_Underlying *_this);
            MR.Const_ICPPairData ret = new(__MR_ICPGroupPair_UpcastTo_MR_ICPPairData(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.Const_ObjVertId SrcId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_srcId", ExactSpelling = true)]
                extern static MR.Const_ObjVertId._Underlying *__MR_ICPGroupPair_Get_srcId(_Underlying *_this);
                return new(__MR_ICPGroupPair_Get_srcId(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_ObjVertId TgtClosestId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_tgtClosestId", ExactSpelling = true)]
                extern static MR.Const_ObjVertId._Underlying *__MR_ICPGroupPair_Get_tgtClosestId(_Underlying *_this);
                return new(__MR_ICPGroupPair_Get_tgtClosestId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the source point after transforming in world space
        public unsafe MR.Const_Vector3f SrcPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_srcPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPGroupPair_Get_srcPoint(_Underlying *_this);
                return new(__MR_ICPGroupPair_Get_srcPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in source point after transforming in world space
        public unsafe MR.Const_Vector3f SrcNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_srcNorm", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPGroupPair_Get_srcNorm(_Underlying *_this);
                return new(__MR_ICPGroupPair_Get_srcNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the closest point on target after transforming in world space
        public unsafe MR.Const_Vector3f TgtPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_tgtPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPGroupPair_Get_tgtPoint(_Underlying *_this);
                return new(__MR_ICPGroupPair_Get_tgtPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in the target point after transforming in world space
        public unsafe MR.Const_Vector3f TgtNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_tgtNorm", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ICPGroupPair_Get_tgtNorm(_Underlying *_this);
                return new(__MR_ICPGroupPair_Get_tgtNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between source and target points
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_ICPGroupPair_Get_distSq(_Underlying *_this);
                return *__MR_ICPGroupPair_Get_distSq(_UnderlyingPtr);
            }
        }

        /// weight of the pair (to prioritize over other pairs)
        public unsafe float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_Get_weight", ExactSpelling = true)]
                extern static float *__MR_ICPGroupPair_Get_weight(_Underlying *_this);
                return *__MR_ICPGroupPair_Get_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ICPGroupPair() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPGroupPair._Underlying *__MR_ICPGroupPair_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPGroupPair_DefaultConstruct();
        }

        /// Generated from constructor `MR::ICPGroupPair::ICPGroupPair`.
        public unsafe Const_ICPGroupPair(MR.Const_ICPGroupPair _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPGroupPair._Underlying *__MR_ICPGroupPair_ConstructFromAnother(MR.ICPGroupPair._Underlying *_other);
            _UnderlyingPtr = __MR_ICPGroupPair_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ICPGroupPair`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ICPPairData`
    /// This is the non-const half of the class.
    public class ICPGroupPair : Const_ICPGroupPair
    {
        internal unsafe ICPGroupPair(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ICPPairData(ICPGroupPair self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_UpcastTo_MR_ICPPairData", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPGroupPair_UpcastTo_MR_ICPPairData(_Underlying *_this);
            MR.ICPPairData ret = new(__MR_ICPGroupPair_UpcastTo_MR_ICPPairData(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe MR.Mut_ObjVertId SrcId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_srcId", ExactSpelling = true)]
                extern static MR.Mut_ObjVertId._Underlying *__MR_ICPGroupPair_GetMutable_srcId(_Underlying *_this);
                return new(__MR_ICPGroupPair_GetMutable_srcId(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_ObjVertId TgtClosestId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_tgtClosestId", ExactSpelling = true)]
                extern static MR.Mut_ObjVertId._Underlying *__MR_ICPGroupPair_GetMutable_tgtClosestId(_Underlying *_this);
                return new(__MR_ICPGroupPair_GetMutable_tgtClosestId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the source point after transforming in world space
        public new unsafe MR.Mut_Vector3f SrcPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_srcPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPGroupPair_GetMutable_srcPoint(_Underlying *_this);
                return new(__MR_ICPGroupPair_GetMutable_srcPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in source point after transforming in world space
        public new unsafe MR.Mut_Vector3f SrcNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_srcNorm", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPGroupPair_GetMutable_srcNorm(_Underlying *_this);
                return new(__MR_ICPGroupPair_GetMutable_srcNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// coordinates of the closest point on target after transforming in world space
        public new unsafe MR.Mut_Vector3f TgtPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_tgtPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPGroupPair_GetMutable_tgtPoint(_Underlying *_this);
                return new(__MR_ICPGroupPair_GetMutable_tgtPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// normal in the target point after transforming in world space
        public new unsafe MR.Mut_Vector3f TgtNorm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_tgtNorm", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ICPGroupPair_GetMutable_tgtNorm(_Underlying *_this);
                return new(__MR_ICPGroupPair_GetMutable_tgtNorm(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance between source and target points
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_ICPGroupPair_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_ICPGroupPair_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// weight of the pair (to prioritize over other pairs)
        public new unsafe ref float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_GetMutable_weight", ExactSpelling = true)]
                extern static float *__MR_ICPGroupPair_GetMutable_weight(_Underlying *_this);
                return ref *__MR_ICPGroupPair_GetMutable_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ICPGroupPair() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPGroupPair._Underlying *__MR_ICPGroupPair_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPGroupPair_DefaultConstruct();
        }

        /// Generated from constructor `MR::ICPGroupPair::ICPGroupPair`.
        public unsafe ICPGroupPair(MR.Const_ICPGroupPair _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPGroupPair._Underlying *__MR_ICPGroupPair_ConstructFromAnother(MR.ICPGroupPair._Underlying *_other);
            _UnderlyingPtr = __MR_ICPGroupPair_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ICPGroupPair::operator=`.
        public unsafe MR.ICPGroupPair Assign(MR.Const_ICPGroupPair _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPair_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ICPGroupPair._Underlying *__MR_ICPGroupPair_AssignFromAnother(_Underlying *_this, MR.ICPGroupPair._Underlying *_other);
            return new(__MR_ICPGroupPair_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ICPGroupPair` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ICPGroupPair`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPGroupPair`/`Const_ICPGroupPair` directly.
    public class _InOptMut_ICPGroupPair
    {
        public ICPGroupPair? Opt;

        public _InOptMut_ICPGroupPair() {}
        public _InOptMut_ICPGroupPair(ICPGroupPair value) {Opt = value;}
        public static implicit operator _InOptMut_ICPGroupPair(ICPGroupPair value) {return new(value);}
    }

    /// This is used for optional parameters of class `ICPGroupPair` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ICPGroupPair`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPGroupPair`/`Const_ICPGroupPair` to pass it to the function.
    public class _InOptConst_ICPGroupPair
    {
        public Const_ICPGroupPair? Opt;

        public _InOptConst_ICPGroupPair() {}
        public _InOptConst_ICPGroupPair(Const_ICPGroupPair value) {Opt = value;}
        public static implicit operator _InOptConst_ICPGroupPair(Const_ICPGroupPair value) {return new(value);}
    }

    /// Generated from class `MR::ICPGroupPairs`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointPairs`
    /// This is the const half of the class.
    public class Const_ICPGroupPairs : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ICPGroupPairs(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_Destroy", ExactSpelling = true)]
            extern static void __MR_ICPGroupPairs_Destroy(_Underlying *_this);
            __MR_ICPGroupPairs_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ICPGroupPairs() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_IPointPairs(Const_ICPGroupPairs self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_UpcastTo_MR_IPointPairs", ExactSpelling = true)]
            extern static MR.Const_IPointPairs._Underlying *__MR_ICPGroupPairs_UpcastTo_MR_IPointPairs(_Underlying *_this);
            MR.Const_IPointPairs ret = new(__MR_ICPGroupPairs_UpcastTo_MR_IPointPairs(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Downcasts:
        public static unsafe explicit operator Const_ICPGroupPairs?(MR.Const_IPointPairs parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_DynamicDowncastTo_MR_ICPGroupPairs", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointPairs_DynamicDowncastTo_MR_ICPGroupPairs(MR.Const_IPointPairs._Underlying *_this);
            var ptr = __MR_IPointPairs_DynamicDowncastTo_MR_ICPGroupPairs(parent._UnderlyingPtr);
            if (ptr is null) return null;
            Const_ICPGroupPairs ret = new(ptr, is_owning: false);
            ret._KeepAlive(parent);
            return ret;
        }

        public unsafe MR.Std.Const_Vector_MRICPGroupPair Vec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_Get_vec", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRICPGroupPair._Underlying *__MR_ICPGroupPairs_Get_vec(_Underlying *_this);
                return new(__MR_ICPGroupPairs_Get_vec(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< whether corresponding pair from vec must be considered during minimization
        public unsafe MR.Const_BitSet Active
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_Get_active", ExactSpelling = true)]
                extern static MR.Const_BitSet._Underlying *__MR_ICPGroupPairs_Get_active(_Underlying *_this);
                return new(__MR_ICPGroupPairs_Get_active(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ICPGroupPairs() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPGroupPairs._Underlying *__MR_ICPGroupPairs_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPGroupPairs_DefaultConstruct();
        }

        /// Generated from constructor `MR::ICPGroupPairs::ICPGroupPairs`.
        public unsafe Const_ICPGroupPairs(MR._ByValue_ICPGroupPairs _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPGroupPairs._Underlying *__MR_ICPGroupPairs_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ICPGroupPairs._Underlying *_other);
            _UnderlyingPtr = __MR_ICPGroupPairs_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ICPGroupPairs::operator[]`.
        public unsafe MR.Const_ICPPairData Index(ulong idx)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_index_const", ExactSpelling = true)]
            extern static MR.Const_ICPPairData._Underlying *__MR_ICPGroupPairs_index_const(_Underlying *_this, ulong idx);
            return new(__MR_ICPGroupPairs_index_const(_UnderlyingPtr, idx), is_owning: false);
        }

        /// Generated from method `MR::ICPGroupPairs::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_size", ExactSpelling = true)]
            extern static ulong __MR_ICPGroupPairs_size(_Underlying *_this);
            return __MR_ICPGroupPairs_size(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ICPGroupPairs`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointPairs`
    /// This is the non-const half of the class.
    public class ICPGroupPairs : Const_ICPGroupPairs
    {
        internal unsafe ICPGroupPairs(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.IPointPairs(ICPGroupPairs self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_UpcastTo_MR_IPointPairs", ExactSpelling = true)]
            extern static MR.IPointPairs._Underlying *__MR_ICPGroupPairs_UpcastTo_MR_IPointPairs(_Underlying *_this);
            MR.IPointPairs ret = new(__MR_ICPGroupPairs_UpcastTo_MR_IPointPairs(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Downcasts:
        public static unsafe explicit operator ICPGroupPairs?(MR.IPointPairs parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointPairs_DynamicDowncastTo_MR_ICPGroupPairs", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointPairs_DynamicDowncastTo_MR_ICPGroupPairs(MR.IPointPairs._Underlying *_this);
            var ptr = __MR_IPointPairs_DynamicDowncastTo_MR_ICPGroupPairs(parent._UnderlyingPtr);
            if (ptr is null) return null;
            ICPGroupPairs ret = new(ptr, is_owning: false);
            ret._KeepAlive(parent);
            return ret;
        }

        public new unsafe MR.Std.Vector_MRICPGroupPair Vec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_GetMutable_vec", ExactSpelling = true)]
                extern static MR.Std.Vector_MRICPGroupPair._Underlying *__MR_ICPGroupPairs_GetMutable_vec(_Underlying *_this);
                return new(__MR_ICPGroupPairs_GetMutable_vec(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< whether corresponding pair from vec must be considered during minimization
        public new unsafe MR.BitSet Active
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_GetMutable_active", ExactSpelling = true)]
                extern static MR.BitSet._Underlying *__MR_ICPGroupPairs_GetMutable_active(_Underlying *_this);
                return new(__MR_ICPGroupPairs_GetMutable_active(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ICPGroupPairs() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ICPGroupPairs._Underlying *__MR_ICPGroupPairs_DefaultConstruct();
            _UnderlyingPtr = __MR_ICPGroupPairs_DefaultConstruct();
        }

        /// Generated from constructor `MR::ICPGroupPairs::ICPGroupPairs`.
        public unsafe ICPGroupPairs(MR._ByValue_ICPGroupPairs _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ICPGroupPairs._Underlying *__MR_ICPGroupPairs_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ICPGroupPairs._Underlying *_other);
            _UnderlyingPtr = __MR_ICPGroupPairs_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ICPGroupPairs::operator=`.
        public unsafe MR.ICPGroupPairs Assign(MR._ByValue_ICPGroupPairs _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ICPGroupPairs._Underlying *__MR_ICPGroupPairs_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ICPGroupPairs._Underlying *_other);
            return new(__MR_ICPGroupPairs_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ICPGroupPairs::operator[]`.
        public unsafe new MR.ICPPairData Index(ulong idx)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ICPGroupPairs_index", ExactSpelling = true)]
            extern static MR.ICPPairData._Underlying *__MR_ICPGroupPairs_index(_Underlying *_this, ulong idx);
            return new(__MR_ICPGroupPairs_index(_UnderlyingPtr, idx), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ICPGroupPairs` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ICPGroupPairs`/`Const_ICPGroupPairs` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ICPGroupPairs
    {
        internal readonly Const_ICPGroupPairs? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ICPGroupPairs() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ICPGroupPairs(Const_ICPGroupPairs new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ICPGroupPairs(Const_ICPGroupPairs arg) {return new(arg);}
        public _ByValue_ICPGroupPairs(MR.Misc._Moved<ICPGroupPairs> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ICPGroupPairs(MR.Misc._Moved<ICPGroupPairs> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ICPGroupPairs` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ICPGroupPairs`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPGroupPairs`/`Const_ICPGroupPairs` directly.
    public class _InOptMut_ICPGroupPairs
    {
        public ICPGroupPairs? Opt;

        public _InOptMut_ICPGroupPairs() {}
        public _InOptMut_ICPGroupPairs(ICPGroupPairs value) {Opt = value;}
        public static implicit operator _InOptMut_ICPGroupPairs(ICPGroupPairs value) {return new(value);}
    }

    /// This is used for optional parameters of class `ICPGroupPairs` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ICPGroupPairs`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ICPGroupPairs`/`Const_ICPGroupPairs` to pass it to the function.
    public class _InOptConst_ICPGroupPairs
    {
        public Const_ICPGroupPairs? Opt;

        public _InOptConst_ICPGroupPairs() {}
        public _InOptConst_ICPGroupPairs(Const_ICPGroupPairs value) {Opt = value;}
        public static implicit operator _InOptConst_ICPGroupPairs(Const_ICPGroupPairs value) {return new(value);}
    }

    /// structure to find leafs and groups of each in cascade mode
    /// Generated from class `MR::IICPTreeIndexer`.
    /// This is the const half of the class.
    public class Const_IICPTreeIndexer : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IICPTreeIndexer(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IICPTreeIndexer_Destroy", ExactSpelling = true)]
            extern static void __MR_IICPTreeIndexer_Destroy(_Underlying *_this);
            __MR_IICPTreeIndexer_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IICPTreeIndexer() {Dispose(false);}

        /// returns true if eI and eJ are from same node
        /// Generated from method `MR::IICPTreeIndexer::fromSameNode`.
        public unsafe bool FromSameNode(int l, MR.Const_Id_MRICPElemtTag eI, MR.Const_Id_MRICPElemtTag eJ)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IICPTreeIndexer_fromSameNode", ExactSpelling = true)]
            extern static byte __MR_IICPTreeIndexer_fromSameNode(_Underlying *_this, int l, MR.Id_MRICPElemtTag._Underlying *eI, MR.Id_MRICPElemtTag._Underlying *eJ);
            return __MR_IICPTreeIndexer_fromSameNode(_UnderlyingPtr, l, eI._UnderlyingPtr, eJ._UnderlyingPtr) != 0;
        }

        /// returns bitset of leaves of given node
        /// Generated from method `MR::IICPTreeIndexer::getElementLeaves`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetElementLeaves(int l, MR.Const_Id_MRICPElemtTag eId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IICPTreeIndexer_getElementLeaves", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_IICPTreeIndexer_getElementLeaves(_Underlying *_this, int l, MR.Id_MRICPElemtTag._Underlying *eId);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_IICPTreeIndexer_getElementLeaves(_UnderlyingPtr, l, eId._UnderlyingPtr), is_owning: true));
        }

        /// valid for l > 0, returns bitset of subnodes that is associated with eId
        /// should be valid for l == `getNumLayers`
        /// Generated from method `MR::IICPTreeIndexer::getElementNodes`.
        public unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> GetElementNodes(int l, MR.Const_Id_MRICPElemtTag eId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IICPTreeIndexer_getElementNodes", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_IICPTreeIndexer_getElementNodes(_Underlying *_this, int l, MR.Id_MRICPElemtTag._Underlying *eId);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_IICPTreeIndexer_getElementNodes(_UnderlyingPtr, l, eId._UnderlyingPtr), is_owning: true));
        }

        /// l == 0 - objs_.size()
        /// l == 1 - number of nodes one layer above objects
        /// l == 2 - number of nodes one layer above nodes lvl1
        /// ...
        /// l == `getNumLayers` - 1
        /// Generated from method `MR::IICPTreeIndexer::getNumElements`.
        public unsafe ulong GetNumElements(int l)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IICPTreeIndexer_getNumElements", ExactSpelling = true)]
            extern static ulong __MR_IICPTreeIndexer_getNumElements(_Underlying *_this, int l);
            return __MR_IICPTreeIndexer_getNumElements(_UnderlyingPtr, l);
        }

        /// Generated from method `MR::IICPTreeIndexer::getNumLayers`.
        public unsafe ulong GetNumLayers()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IICPTreeIndexer_getNumLayers", ExactSpelling = true)]
            extern static ulong __MR_IICPTreeIndexer_getNumLayers(_Underlying *_this);
            return __MR_IICPTreeIndexer_getNumLayers(_UnderlyingPtr);
        }
    }

    /// structure to find leafs and groups of each in cascade mode
    /// Generated from class `MR::IICPTreeIndexer`.
    /// This is the non-const half of the class.
    public class IICPTreeIndexer : Const_IICPTreeIndexer
    {
        internal unsafe IICPTreeIndexer(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}
    }

    /// This is used for optional parameters of class `IICPTreeIndexer` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IICPTreeIndexer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IICPTreeIndexer`/`Const_IICPTreeIndexer` directly.
    public class _InOptMut_IICPTreeIndexer
    {
        public IICPTreeIndexer? Opt;

        public _InOptMut_IICPTreeIndexer() {}
        public _InOptMut_IICPTreeIndexer(IICPTreeIndexer value) {Opt = value;}
        public static implicit operator _InOptMut_IICPTreeIndexer(IICPTreeIndexer value) {return new(value);}
    }

    /// This is used for optional parameters of class `IICPTreeIndexer` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IICPTreeIndexer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IICPTreeIndexer`/`Const_IICPTreeIndexer` to pass it to the function.
    public class _InOptConst_IICPTreeIndexer
    {
        public Const_IICPTreeIndexer? Opt;

        public _InOptConst_IICPTreeIndexer() {}
        public _InOptConst_IICPTreeIndexer(Const_IICPTreeIndexer value) {Opt = value;}
        public static implicit operator _InOptConst_IICPTreeIndexer(Const_IICPTreeIndexer value) {return new(value);}
    }

    /// Parameters that are used for sampling of the MultiwayICP objects
    /// Generated from class `MR::MultiwayICPSamplingParameters`.
    /// This is the const half of the class.
    public class Const_MultiwayICPSamplingParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MultiwayICPSamplingParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_MultiwayICPSamplingParameters_Destroy(_Underlying *_this);
            __MR_MultiwayICPSamplingParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MultiwayICPSamplingParameters() {Dispose(false);}

        /// sampling size of each object, 0 has special meaning "take all valid points"
        public unsafe float SamplingVoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_Get_samplingVoxelSize", ExactSpelling = true)]
                extern static float *__MR_MultiwayICPSamplingParameters_Get_samplingVoxelSize(_Underlying *_this);
                return *__MR_MultiwayICPSamplingParameters_Get_samplingVoxelSize(_UnderlyingPtr);
            }
        }

        /// size of maximum icp group to work with;
        /// if the number of objects exceeds this value, icp is applied in cascade mode;
        /// maxGroupSize = 1 means that every object is moved independently on half distance to the previous position of all other objects;
        /// maxGroupSize = 0 means that a big system of equations for all objects is solved (force no cascading)
        public unsafe int MaxGroupSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_Get_maxGroupSize", ExactSpelling = true)]
                extern static int *__MR_MultiwayICPSamplingParameters_Get_maxGroupSize(_Underlying *_this);
                return *__MR_MultiwayICPSamplingParameters_Get_maxGroupSize(_UnderlyingPtr);
            }
        }

        public unsafe MR.MultiwayICPSamplingParameters.CascadeMode CascadeMode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_Get_cascadeMode", ExactSpelling = true)]
                extern static MR.MultiwayICPSamplingParameters.CascadeMode *__MR_MultiwayICPSamplingParameters_Get_cascadeMode(_Underlying *_this);
                return *__MR_MultiwayICPSamplingParameters_Get_cascadeMode(_UnderlyingPtr);
            }
        }

        /// callback for progress reports
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MultiwayICPSamplingParameters_Get_cb(_Underlying *_this);
                return new(__MR_MultiwayICPSamplingParameters_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MultiwayICPSamplingParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiwayICPSamplingParameters_DefaultConstruct();
        }

        /// Constructs `MR::MultiwayICPSamplingParameters` elementwise.
        public unsafe Const_MultiwayICPSamplingParameters(float samplingVoxelSize, int maxGroupSize, MR.MultiwayICPSamplingParameters.CascadeMode cascadeMode, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_ConstructFrom(float samplingVoxelSize, int maxGroupSize, MR.MultiwayICPSamplingParameters.CascadeMode cascadeMode, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_MultiwayICPSamplingParameters_ConstructFrom(samplingVoxelSize, maxGroupSize, cascadeMode, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MultiwayICPSamplingParameters::MultiwayICPSamplingParameters`.
        public unsafe Const_MultiwayICPSamplingParameters(MR._ByValue_MultiwayICPSamplingParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MultiwayICPSamplingParameters._Underlying *_other);
            _UnderlyingPtr = __MR_MultiwayICPSamplingParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        public enum CascadeMode : int
        {
            Sequential = 0,
            /// separates objects on groups based on their index in ICPObjects (good if all objects about the size of all objects together)
            AABBTreeBased = 1,
        }
    }

    /// Parameters that are used for sampling of the MultiwayICP objects
    /// Generated from class `MR::MultiwayICPSamplingParameters`.
    /// This is the non-const half of the class.
    public class MultiwayICPSamplingParameters : Const_MultiwayICPSamplingParameters
    {
        internal unsafe MultiwayICPSamplingParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// sampling size of each object, 0 has special meaning "take all valid points"
        public new unsafe ref float SamplingVoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_GetMutable_samplingVoxelSize", ExactSpelling = true)]
                extern static float *__MR_MultiwayICPSamplingParameters_GetMutable_samplingVoxelSize(_Underlying *_this);
                return ref *__MR_MultiwayICPSamplingParameters_GetMutable_samplingVoxelSize(_UnderlyingPtr);
            }
        }

        /// size of maximum icp group to work with;
        /// if the number of objects exceeds this value, icp is applied in cascade mode;
        /// maxGroupSize = 1 means that every object is moved independently on half distance to the previous position of all other objects;
        /// maxGroupSize = 0 means that a big system of equations for all objects is solved (force no cascading)
        public new unsafe ref int MaxGroupSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_GetMutable_maxGroupSize", ExactSpelling = true)]
                extern static int *__MR_MultiwayICPSamplingParameters_GetMutable_maxGroupSize(_Underlying *_this);
                return ref *__MR_MultiwayICPSamplingParameters_GetMutable_maxGroupSize(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.MultiwayICPSamplingParameters.CascadeMode CascadeMode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_GetMutable_cascadeMode", ExactSpelling = true)]
                extern static MR.MultiwayICPSamplingParameters.CascadeMode *__MR_MultiwayICPSamplingParameters_GetMutable_cascadeMode(_Underlying *_this);
                return ref *__MR_MultiwayICPSamplingParameters_GetMutable_cascadeMode(_UnderlyingPtr);
            }
        }

        /// callback for progress reports
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MultiwayICPSamplingParameters_GetMutable_cb(_Underlying *_this);
                return new(__MR_MultiwayICPSamplingParameters_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MultiwayICPSamplingParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiwayICPSamplingParameters_DefaultConstruct();
        }

        /// Constructs `MR::MultiwayICPSamplingParameters` elementwise.
        public unsafe MultiwayICPSamplingParameters(float samplingVoxelSize, int maxGroupSize, MR.MultiwayICPSamplingParameters.CascadeMode cascadeMode, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_ConstructFrom(float samplingVoxelSize, int maxGroupSize, MR.MultiwayICPSamplingParameters.CascadeMode cascadeMode, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_MultiwayICPSamplingParameters_ConstructFrom(samplingVoxelSize, maxGroupSize, cascadeMode, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MultiwayICPSamplingParameters::MultiwayICPSamplingParameters`.
        public unsafe MultiwayICPSamplingParameters(MR._ByValue_MultiwayICPSamplingParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MultiwayICPSamplingParameters._Underlying *_other);
            _UnderlyingPtr = __MR_MultiwayICPSamplingParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MultiwayICPSamplingParameters::operator=`.
        public unsafe MR.MultiwayICPSamplingParameters Assign(MR._ByValue_MultiwayICPSamplingParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICPSamplingParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayICPSamplingParameters._Underlying *__MR_MultiwayICPSamplingParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MultiwayICPSamplingParameters._Underlying *_other);
            return new(__MR_MultiwayICPSamplingParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MultiwayICPSamplingParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MultiwayICPSamplingParameters`/`Const_MultiwayICPSamplingParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MultiwayICPSamplingParameters
    {
        internal readonly Const_MultiwayICPSamplingParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MultiwayICPSamplingParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MultiwayICPSamplingParameters(Const_MultiwayICPSamplingParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MultiwayICPSamplingParameters(Const_MultiwayICPSamplingParameters arg) {return new(arg);}
        public _ByValue_MultiwayICPSamplingParameters(MR.Misc._Moved<MultiwayICPSamplingParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MultiwayICPSamplingParameters(MR.Misc._Moved<MultiwayICPSamplingParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MultiwayICPSamplingParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MultiwayICPSamplingParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiwayICPSamplingParameters`/`Const_MultiwayICPSamplingParameters` directly.
    public class _InOptMut_MultiwayICPSamplingParameters
    {
        public MultiwayICPSamplingParameters? Opt;

        public _InOptMut_MultiwayICPSamplingParameters() {}
        public _InOptMut_MultiwayICPSamplingParameters(MultiwayICPSamplingParameters value) {Opt = value;}
        public static implicit operator _InOptMut_MultiwayICPSamplingParameters(MultiwayICPSamplingParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `MultiwayICPSamplingParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MultiwayICPSamplingParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiwayICPSamplingParameters`/`Const_MultiwayICPSamplingParameters` to pass it to the function.
    public class _InOptConst_MultiwayICPSamplingParameters
    {
        public Const_MultiwayICPSamplingParameters? Opt;

        public _InOptConst_MultiwayICPSamplingParameters() {}
        public _InOptConst_MultiwayICPSamplingParameters(Const_MultiwayICPSamplingParameters value) {Opt = value;}
        public static implicit operator _InOptConst_MultiwayICPSamplingParameters(Const_MultiwayICPSamplingParameters value) {return new(value);}
    }

    /// This class allows you to register many objects having similar parts
    /// and known initial approximations of orientations/locations using
    /// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
    /// \snippet cpp-samples/GlobalRegistration.cpp 0
    /// Generated from class `MR::MultiwayICP`.
    /// This is the const half of the class.
    public class Const_MultiwayICP : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MultiwayICP(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_Destroy", ExactSpelling = true)]
            extern static void __MR_MultiwayICP_Destroy(_Underlying *_this);
            __MR_MultiwayICP_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MultiwayICP() {Dispose(false);}

        /// Generated from constructor `MR::MultiwayICP::MultiwayICP`.
        public unsafe Const_MultiwayICP(MR._ByValue_MultiwayICP _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayICP._Underlying *__MR_MultiwayICP_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MultiwayICP._Underlying *_other);
            _UnderlyingPtr = __MR_MultiwayICP_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MultiwayICP::MultiwayICP`.
        public unsafe Const_MultiwayICP(MR.Const_Vector_MRMeshOrPointsXf_MRObjId objects, MR.Const_MultiwayICPSamplingParameters samplingParams) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_Construct", ExactSpelling = true)]
            extern static MR.MultiwayICP._Underlying *__MR_MultiwayICP_Construct(MR.Const_Vector_MRMeshOrPointsXf_MRObjId._Underlying *objects, MR.Const_MultiwayICPSamplingParameters._Underlying *samplingParams);
            _UnderlyingPtr = __MR_MultiwayICP_Construct(objects._UnderlyingPtr, samplingParams._UnderlyingPtr);
        }

        /// Generated from method `MR::MultiwayICP::getParams`.
        public unsafe MR.Const_ICPProperties GetParams()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getParams", ExactSpelling = true)]
            extern static MR.Const_ICPProperties._Underlying *__MR_MultiwayICP_getParams(_Underlying *_this);
            return new(__MR_MultiwayICP_getParams(_UnderlyingPtr), is_owning: false);
        }

        /// computes root-mean-square deviation between points
        /// or the standard deviation from given value if present
        /// Generated from method `MR::MultiwayICP::getMeanSqDistToPoint`.
        public unsafe float GetMeanSqDistToPoint(double? value = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getMeanSqDistToPoint", ExactSpelling = true)]
            extern static float __MR_MultiwayICP_getMeanSqDistToPoint(_Underlying *_this, double *value);
            double __deref_value = value.GetValueOrDefault();
            return __MR_MultiwayICP_getMeanSqDistToPoint(_UnderlyingPtr, value.HasValue ? &__deref_value : null);
        }

        /// computes root-mean-square deviation from points to target planes
        /// or the standard deviation from given value if present
        /// Generated from method `MR::MultiwayICP::getMeanSqDistToPlane`.
        public unsafe float GetMeanSqDistToPlane(double? value = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getMeanSqDistToPlane", ExactSpelling = true)]
            extern static float __MR_MultiwayICP_getMeanSqDistToPlane(_Underlying *_this, double *value);
            double __deref_value = value.GetValueOrDefault();
            return __MR_MultiwayICP_getMeanSqDistToPlane(_UnderlyingPtr, value.HasValue ? &__deref_value : null);
        }

        /// computes the number of samples able to form pairs
        /// Generated from method `MR::MultiwayICP::getNumSamples`.
        public unsafe ulong GetNumSamples()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getNumSamples", ExactSpelling = true)]
            extern static ulong __MR_MultiwayICP_getNumSamples(_Underlying *_this);
            return __MR_MultiwayICP_getNumSamples(_UnderlyingPtr);
        }

        /// computes the number of active point pairs
        /// Generated from method `MR::MultiwayICP::getNumActivePairs`.
        public unsafe ulong GetNumActivePairs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getNumActivePairs", ExactSpelling = true)]
            extern static ulong __MR_MultiwayICP_getNumActivePairs(_Underlying *_this);
            return __MR_MultiwayICP_getNumActivePairs(_UnderlyingPtr);
        }

        /// if in independent equations mode - creates separate equation system for each object
        /// otherwise creates single large equation system for all objects
        /// Generated from method `MR::MultiwayICP::devIndependentEquationsModeEnabled`.
        public unsafe bool DevIndependentEquationsModeEnabled()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_devIndependentEquationsModeEnabled", ExactSpelling = true)]
            extern static byte __MR_MultiwayICP_devIndependentEquationsModeEnabled(_Underlying *_this);
            return __MR_MultiwayICP_devIndependentEquationsModeEnabled(_UnderlyingPtr) != 0;
        }

        /// returns status info string
        /// Generated from method `MR::MultiwayICP::getStatusInfo`.
        public unsafe MR.Misc._Moved<MR.Std.String> GetStatusInfo()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getStatusInfo", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_MultiwayICP_getStatusInfo(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_MultiwayICP_getStatusInfo(_UnderlyingPtr), is_owning: true));
        }

        /// returns all pairs of all layers
        /// Generated from method `MR::MultiwayICP::getPairsPerLayer`.
        public unsafe MR.Const_Vector_MRVectorMRVectorMRICPGroupPairsMRIdMRICPElemtTagMRIdMRICPElemtTag_Int GetPairsPerLayer()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getPairsPerLayer", ExactSpelling = true)]
            extern static MR.Const_Vector_MRVectorMRVectorMRICPGroupPairsMRIdMRICPElemtTagMRIdMRICPElemtTag_Int._Underlying *__MR_MultiwayICP_getPairsPerLayer(_Underlying *_this);
            return new(__MR_MultiwayICP_getPairsPerLayer(_UnderlyingPtr), is_owning: false);
        }

        /// returns pointer to class that is used to navigate among layers of cascade registration
        /// if nullptr - cascade mode is not used
        /// Generated from method `MR::MultiwayICP::getCascadeIndexer`.
        public unsafe MR.Const_IICPTreeIndexer? GetCascadeIndexer()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_getCascadeIndexer", ExactSpelling = true)]
            extern static MR.Const_IICPTreeIndexer._Underlying *__MR_MultiwayICP_getCascadeIndexer(_Underlying *_this);
            var __ret = __MR_MultiwayICP_getCascadeIndexer(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_IICPTreeIndexer(__ret, is_owning: false) : null;
        }
    }

    /// This class allows you to register many objects having similar parts
    /// and known initial approximations of orientations/locations using
    /// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
    /// \snippet cpp-samples/GlobalRegistration.cpp 0
    /// Generated from class `MR::MultiwayICP`.
    /// This is the non-const half of the class.
    public class MultiwayICP : Const_MultiwayICP
    {
        internal unsafe MultiwayICP(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MultiwayICP::MultiwayICP`.
        public unsafe MultiwayICP(MR._ByValue_MultiwayICP _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayICP._Underlying *__MR_MultiwayICP_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MultiwayICP._Underlying *_other);
            _UnderlyingPtr = __MR_MultiwayICP_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MultiwayICP::MultiwayICP`.
        public unsafe MultiwayICP(MR.Const_Vector_MRMeshOrPointsXf_MRObjId objects, MR.Const_MultiwayICPSamplingParameters samplingParams) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_Construct", ExactSpelling = true)]
            extern static MR.MultiwayICP._Underlying *__MR_MultiwayICP_Construct(MR.Const_Vector_MRMeshOrPointsXf_MRObjId._Underlying *objects, MR.Const_MultiwayICPSamplingParameters._Underlying *samplingParams);
            _UnderlyingPtr = __MR_MultiwayICP_Construct(objects._UnderlyingPtr, samplingParams._UnderlyingPtr);
        }

        /// Generated from method `MR::MultiwayICP::operator=`.
        public unsafe MR.MultiwayICP Assign(MR._ByValue_MultiwayICP _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayICP._Underlying *__MR_MultiwayICP_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MultiwayICP._Underlying *_other);
            return new(__MR_MultiwayICP_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// runs ICP algorithm given input objects, transformations, and parameters;
        /// \return adjusted transformations of all objects to reach registered state
        /// the transformation of the last object is fixed and does not change here
        /// Generated from method `MR::MultiwayICP::calculateTransformations`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Vector_MRAffineXf3f_MRObjId> CalculateTransformations(MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_calculateTransformations", ExactSpelling = true)]
            extern static MR.Vector_MRAffineXf3f_MRObjId._Underlying *__MR_MultiwayICP_calculateTransformations(_Underlying *_this, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Vector_MRAffineXf3f_MRObjId(__MR_MultiwayICP_calculateTransformations(_UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// runs ICP algorithm given input objects, transformations, and parameters;
        /// \return adjusted transformations of all objects to reach registered state
        /// the transformation of the first object is fixed and does not change here
        /// Generated from method `MR::MultiwayICP::calculateTransformationsFixFirst`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Vector_MRAffineXf3f_MRObjId> CalculateTransformationsFixFirst(MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_calculateTransformationsFixFirst", ExactSpelling = true)]
            extern static MR.Vector_MRAffineXf3f_MRObjId._Underlying *__MR_MultiwayICP_calculateTransformationsFixFirst(_Underlying *_this, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Vector_MRAffineXf3f_MRObjId(__MR_MultiwayICP_calculateTransformationsFixFirst(_UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// select pairs with origin samples on all objects
        /// Generated from method `MR::MultiwayICP::resamplePoints`.
        public unsafe bool ResamplePoints(MR.Const_MultiwayICPSamplingParameters samplingParams)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_resamplePoints", ExactSpelling = true)]
            extern static byte __MR_MultiwayICP_resamplePoints(_Underlying *_this, MR.Const_MultiwayICPSamplingParameters._Underlying *samplingParams);
            return __MR_MultiwayICP_resamplePoints(_UnderlyingPtr, samplingParams._UnderlyingPtr) != 0;
        }

        /// in each pair updates the target data and performs basic filtering (activation)
        /// in cascade mode only useful for stats update
        /// Generated from method `MR::MultiwayICP::updateAllPointPairs`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe bool UpdateAllPointPairs(MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_updateAllPointPairs", ExactSpelling = true)]
            extern static byte __MR_MultiwayICP_updateAllPointPairs(_Underlying *_this, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return __MR_MultiwayICP_updateAllPointPairs(_UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null) != 0;
        }

        /// tune algorithm params before run calculateTransformations()
        /// Generated from method `MR::MultiwayICP::setParams`.
        public unsafe void SetParams(MR.Const_ICPProperties prop)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_setParams", ExactSpelling = true)]
            extern static void __MR_MultiwayICP_setParams(_Underlying *_this, MR.Const_ICPProperties._Underlying *prop);
            __MR_MultiwayICP_setParams(_UnderlyingPtr, prop._UnderlyingPtr);
        }

        /// sets callback that will be called for each iteration
        /// Generated from method `MR::MultiwayICP::setPerIterationCallback`.
        public unsafe void SetPerIterationCallback(MR.Std._ByValue_Function_VoidFuncFromInt callback)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_setPerIterationCallback", ExactSpelling = true)]
            extern static void __MR_MultiwayICP_setPerIterationCallback(_Underlying *_this, MR.Misc._PassBy callback_pass_by, MR.Std.Function_VoidFuncFromInt._Underlying *callback);
            __MR_MultiwayICP_setPerIterationCallback(_UnderlyingPtr, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MultiwayICP::devEnableIndependentEquationsMode`.
        public unsafe void DevEnableIndependentEquationsMode(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayICP_devEnableIndependentEquationsMode", ExactSpelling = true)]
            extern static void __MR_MultiwayICP_devEnableIndependentEquationsMode(_Underlying *_this, byte on);
            __MR_MultiwayICP_devEnableIndependentEquationsMode(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MultiwayICP` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MultiwayICP`/`Const_MultiwayICP` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MultiwayICP
    {
        internal readonly Const_MultiwayICP? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MultiwayICP(MR.Misc._Moved<MultiwayICP> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MultiwayICP(MR.Misc._Moved<MultiwayICP> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MultiwayICP` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MultiwayICP`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiwayICP`/`Const_MultiwayICP` directly.
    public class _InOptMut_MultiwayICP
    {
        public MultiwayICP? Opt;

        public _InOptMut_MultiwayICP() {}
        public _InOptMut_MultiwayICP(MultiwayICP value) {Opt = value;}
        public static implicit operator _InOptMut_MultiwayICP(MultiwayICP value) {return new(value);}
    }

    /// This is used for optional parameters of class `MultiwayICP` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MultiwayICP`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiwayICP`/`Const_MultiwayICP` to pass it to the function.
    public class _InOptConst_MultiwayICP
    {
        public Const_MultiwayICP? Opt;

        public _InOptConst_MultiwayICP() {}
        public _InOptConst_MultiwayICP(Const_MultiwayICP value) {Opt = value;}
        public static implicit operator _InOptConst_MultiwayICP(Const_MultiwayICP value) {return new(value);}
    }

    /// in each pair updates the target data and performs basic filtering (activation)
    /// Generated from function `MR::updateGroupPairs`.
    public static unsafe void UpdateGroupPairs(MR.ICPGroupPairs pairs, MR.Const_Vector_MRMeshOrPointsXf_MRObjId objs, MR.Std._ByValue_Function_VoidFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRefMRObjIdRef srcProjector, MR.Std._ByValue_Function_VoidFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRefMRObjIdRef tgtProjector, float cosThreshold, float distThresholdSq, bool mutualClosest)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_updateGroupPairs", ExactSpelling = true)]
        extern static void __MR_updateGroupPairs(MR.ICPGroupPairs._Underlying *pairs, MR.Const_Vector_MRMeshOrPointsXf_MRObjId._Underlying *objs, MR.Misc._PassBy srcProjector_pass_by, MR.Std.Function_VoidFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRefMRObjIdRef._Underlying *srcProjector, MR.Misc._PassBy tgtProjector_pass_by, MR.Std.Function_VoidFuncFromConstMRVector3fRefMRMeshOrPointsProjectionResultRefMRObjIdRef._Underlying *tgtProjector, float cosThreshold, float distThresholdSq, byte mutualClosest);
        __MR_updateGroupPairs(pairs._UnderlyingPtr, objs._UnderlyingPtr, srcProjector.PassByMode, srcProjector.Value is not null ? srcProjector.Value._UnderlyingPtr : null, tgtProjector.PassByMode, tgtProjector.Value is not null ? tgtProjector.Value._UnderlyingPtr : null, cosThreshold, distThresholdSq, mutualClosest ? (byte)1 : (byte)0);
    }
}
