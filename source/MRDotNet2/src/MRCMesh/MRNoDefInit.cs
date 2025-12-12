public static partial class MR
{
    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::UndirectedEdgeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::UndirectedEdgeId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.UndirectedEdgeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRUndirectedEdgeId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_UndirectedEdgeId(Const_NoDefInit_MRUndirectedEdgeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_UpcastTo_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_UpcastTo_MR_UndirectedEdgeId(_Underlying *_this);
            MR.Const_UndirectedEdgeId ret = new(__MR_NoDefInit_MR_UndirectedEdgeId_UpcastTo_MR_UndirectedEdgeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_UndirectedEdgeId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_UndirectedEdgeId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::UndirectedEdgeId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRUndirectedEdgeId(MR.Const_NoDefInit_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_ConstructFromAnother(MR.NoDefInit_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_UndirectedEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::UndirectedEdgeId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRUndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_UndirectedEdgeId_ConvertTo_int(MR.Const_NoDefInit_MRUndirectedEdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_UndirectedEdgeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::UndirectedEdgeId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRUndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_UndirectedEdgeId_ConvertTo_bool(MR.Const_NoDefInit_MRUndirectedEdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_UndirectedEdgeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_UndirectedEdgeId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_UndirectedEdgeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRUndirectedEdgeId _this, MR.UndirectedEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_UndirectedEdgeId_MR_UndirectedEdgeId(MR.Const_NoDefInit_MRUndirectedEdgeId._Underlying *_this, MR.UndirectedEdgeId b);
            return __MR_equal_MR_NoDefInit_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRUndirectedEdgeId _this, MR.UndirectedEdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::operator<`.
        public unsafe bool Less(MR.UndirectedEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_Underlying *_this, MR.UndirectedEdgeId b);
            return __MR_less_MR_NoDefInit_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.UndirectedEdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.UndirectedEdgeId)
                return this == (MR.UndirectedEdgeId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::UndirectedEdgeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::UndirectedEdgeId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRUndirectedEdgeId : Const_NoDefInit_MRUndirectedEdgeId
    {
        internal unsafe NoDefInit_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_UndirectedEdgeId(NoDefInit_MRUndirectedEdgeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_UpcastTo_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_UpcastTo_MR_UndirectedEdgeId(_Underlying *_this);
            MR.Mut_UndirectedEdgeId ret = new(__MR_NoDefInit_MR_UndirectedEdgeId_UpcastTo_MR_UndirectedEdgeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_UndirectedEdgeId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_UndirectedEdgeId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::UndirectedEdgeId>::NoDefInit`.
        public unsafe NoDefInit_MRUndirectedEdgeId(MR.Const_NoDefInit_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_ConstructFromAnother(MR.NoDefInit_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_UndirectedEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.NoDefInit_MRUndirectedEdgeId Assign(MR.Const_NoDefInit_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_NoDefInit_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_UndirectedEdgeId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_UndirectedEdgeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::operator-=`.
        public unsafe MR.Mut_UndirectedEdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_UndirectedEdgeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::UndirectedEdgeId>::operator+=`.
        public unsafe MR.Mut_UndirectedEdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_UndirectedEdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_NoDefInit_MR_UndirectedEdgeId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_UndirectedEdgeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRUndirectedEdgeId`/`Const_NoDefInit_MRUndirectedEdgeId` directly.
    public class _InOptMut_NoDefInit_MRUndirectedEdgeId
    {
        public NoDefInit_MRUndirectedEdgeId? Opt;

        public _InOptMut_NoDefInit_MRUndirectedEdgeId() {}
        public _InOptMut_NoDefInit_MRUndirectedEdgeId(NoDefInit_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRUndirectedEdgeId(NoDefInit_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRUndirectedEdgeId`/`Const_NoDefInit_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRUndirectedEdgeId
    {
        public Const_NoDefInit_MRUndirectedEdgeId? Opt;

        public _InOptConst_NoDefInit_MRUndirectedEdgeId() {}
        public _InOptConst_NoDefInit_MRUndirectedEdgeId(Const_NoDefInit_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRUndirectedEdgeId(Const_NoDefInit_MRUndirectedEdgeId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::FaceId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FaceId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRFaceId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.FaceId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_FaceId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRFaceId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_FaceId(Const_NoDefInit_MRFaceId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_UpcastTo_MR_FaceId", ExactSpelling = true)]
            extern static MR.Const_FaceId._Underlying *__MR_NoDefInit_MR_FaceId_UpcastTo_MR_FaceId(_Underlying *_this);
            MR.Const_FaceId ret = new(__MR_NoDefInit_MR_FaceId_UpcastTo_MR_FaceId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_FaceId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_FaceId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_NoDefInit_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::FaceId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRFaceId(MR.Const_NoDefInit_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_NoDefInit_MR_FaceId_ConstructFromAnother(MR.NoDefInit_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_FaceId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::FaceId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRFaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_FaceId_ConvertTo_int(MR.Const_NoDefInit_MRFaceId._Underlying *_this);
            return __MR_NoDefInit_MR_FaceId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::FaceId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRFaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_FaceId_ConvertTo_bool(MR.Const_NoDefInit_MRFaceId._Underlying *_this);
            return __MR_NoDefInit_MR_FaceId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_FaceId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_FaceId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRFaceId _this, MR.FaceId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_FaceId_MR_FaceId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_FaceId_MR_FaceId(MR.Const_NoDefInit_MRFaceId._Underlying *_this, MR.FaceId b);
            return __MR_equal_MR_NoDefInit_MR_FaceId_MR_FaceId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRFaceId _this, MR.FaceId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::operator<`.
        public unsafe bool Less(MR.FaceId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_FaceId_MR_FaceId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_FaceId_MR_FaceId(_Underlying *_this, MR.FaceId b);
            return __MR_less_MR_NoDefInit_MR_FaceId_MR_FaceId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.FaceId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.FaceId)
                return this == (MR.FaceId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::FaceId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FaceId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRFaceId : Const_NoDefInit_MRFaceId
    {
        internal unsafe NoDefInit_MRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_FaceId(NoDefInit_MRFaceId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_UpcastTo_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_NoDefInit_MR_FaceId_UpcastTo_MR_FaceId(_Underlying *_this);
            MR.Mut_FaceId ret = new(__MR_NoDefInit_MR_FaceId_UpcastTo_MR_FaceId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_FaceId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_FaceId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_NoDefInit_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::FaceId>::NoDefInit`.
        public unsafe NoDefInit_MRFaceId(MR.Const_NoDefInit_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_NoDefInit_MR_FaceId_ConstructFromAnother(MR.NoDefInit_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_FaceId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::operator=`.
        public unsafe MR.NoDefInit_MRFaceId Assign(MR.Const_NoDefInit_MRFaceId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_NoDefInit_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRFaceId._Underlying *_other);
            return new(__MR_NoDefInit_MR_FaceId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_FaceId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_FaceId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::operator-=`.
        public unsafe MR.Mut_FaceId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_NoDefInit_MR_FaceId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_FaceId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::FaceId>::operator+=`.
        public unsafe MR.Mut_FaceId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_FaceId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_NoDefInit_MR_FaceId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_FaceId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRFaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRFaceId`/`Const_NoDefInit_MRFaceId` directly.
    public class _InOptMut_NoDefInit_MRFaceId
    {
        public NoDefInit_MRFaceId? Opt;

        public _InOptMut_NoDefInit_MRFaceId() {}
        public _InOptMut_NoDefInit_MRFaceId(NoDefInit_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRFaceId(NoDefInit_MRFaceId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRFaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRFaceId`/`Const_NoDefInit_MRFaceId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRFaceId
    {
        public Const_NoDefInit_MRFaceId? Opt;

        public _InOptConst_NoDefInit_MRFaceId() {}
        public _InOptConst_NoDefInit_MRFaceId(Const_NoDefInit_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRFaceId(Const_NoDefInit_MRFaceId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::VertId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VertId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRVertId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.VertId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_VertId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRVertId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_VertId(Const_NoDefInit_MRVertId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_UpcastTo_MR_VertId", ExactSpelling = true)]
            extern static MR.Const_VertId._Underlying *__MR_NoDefInit_MR_VertId_UpcastTo_MR_VertId(_Underlying *_this);
            MR.Const_VertId ret = new(__MR_NoDefInit_MR_VertId_UpcastTo_MR_VertId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_VertId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_VertId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_NoDefInit_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::VertId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRVertId(MR.Const_NoDefInit_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_NoDefInit_MR_VertId_ConstructFromAnother(MR.NoDefInit_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_VertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::VertId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_VertId_ConvertTo_int(MR.Const_NoDefInit_MRVertId._Underlying *_this);
            return __MR_NoDefInit_MR_VertId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::VertId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_VertId_ConvertTo_bool(MR.Const_NoDefInit_MRVertId._Underlying *_this);
            return __MR_NoDefInit_MR_VertId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_VertId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_VertId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRVertId _this, MR.VertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_VertId_MR_VertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_VertId_MR_VertId(MR.Const_NoDefInit_MRVertId._Underlying *_this, MR.VertId b);
            return __MR_equal_MR_NoDefInit_MR_VertId_MR_VertId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRVertId _this, MR.VertId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::operator<`.
        public unsafe bool Less(MR.VertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_VertId_MR_VertId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_VertId_MR_VertId(_Underlying *_this, MR.VertId b);
            return __MR_less_MR_NoDefInit_MR_VertId_MR_VertId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.VertId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.VertId)
                return this == (MR.VertId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::VertId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VertId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRVertId : Const_NoDefInit_MRVertId
    {
        internal unsafe NoDefInit_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_VertId(NoDefInit_MRVertId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_UpcastTo_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_NoDefInit_MR_VertId_UpcastTo_MR_VertId(_Underlying *_this);
            MR.Mut_VertId ret = new(__MR_NoDefInit_MR_VertId_UpcastTo_MR_VertId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_VertId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_VertId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_NoDefInit_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::VertId>::NoDefInit`.
        public unsafe NoDefInit_MRVertId(MR.Const_NoDefInit_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_NoDefInit_MR_VertId_ConstructFromAnother(MR.NoDefInit_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_VertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::operator=`.
        public unsafe MR.NoDefInit_MRVertId Assign(MR.Const_NoDefInit_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_NoDefInit_MR_VertId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRVertId._Underlying *_other);
            return new(__MR_NoDefInit_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_VertId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_VertId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::operator-=`.
        public unsafe MR.Mut_VertId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_NoDefInit_MR_VertId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_VertId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::VertId>::operator+=`.
        public unsafe MR.Mut_VertId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VertId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_NoDefInit_MR_VertId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_VertId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRVertId`/`Const_NoDefInit_MRVertId` directly.
    public class _InOptMut_NoDefInit_MRVertId
    {
        public NoDefInit_MRVertId? Opt;

        public _InOptMut_NoDefInit_MRVertId() {}
        public _InOptMut_NoDefInit_MRVertId(NoDefInit_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRVertId(NoDefInit_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRVertId`/`Const_NoDefInit_MRVertId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRVertId
    {
        public Const_NoDefInit_MRVertId? Opt;

        public _InOptConst_NoDefInit_MRVertId() {}
        public _InOptConst_NoDefInit_MRVertId(Const_NoDefInit_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRVertId(Const_NoDefInit_MRVertId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::EdgeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgeId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MREdgeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.EdgeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_EdgeId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_EdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MREdgeId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_EdgeId(Const_NoDefInit_MREdgeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_UpcastTo_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Const_EdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_UpcastTo_MR_EdgeId(_Underlying *_this);
            MR.Const_EdgeId ret = new(__MR_NoDefInit_MR_EdgeId_UpcastTo_MR_EdgeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_EdgeId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_EdgeId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::EdgeId>::NoDefInit`.
        public unsafe Const_NoDefInit_MREdgeId(MR.Const_NoDefInit_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_ConstructFromAnother(MR.NoDefInit_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_EdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::EdgeId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MREdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_EdgeId_ConvertTo_int(MR.Const_NoDefInit_MREdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::EdgeId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MREdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_EdgeId_ConvertTo_bool(MR.Const_NoDefInit_MREdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::EdgeId>::operator MR::UndirectedEdgeId`.
        public static unsafe implicit operator MR.UndirectedEdgeId(MR.Const_NoDefInit_MREdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_ConvertTo_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_NoDefInit_MR_EdgeId_ConvertTo_MR_UndirectedEdgeId(MR.Const_NoDefInit_MREdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_ConvertTo_MR_UndirectedEdgeId(_this._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_EdgeId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_valid(_UnderlyingPtr) != 0;
        }

        // returns identifier of the edge with same ends but opposite orientation
        /// Generated from method `MR::NoDefInit<MR::EdgeId>::sym`.
        public unsafe MR.EdgeId Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_sym", ExactSpelling = true)]
            extern static MR.EdgeId __MR_NoDefInit_MR_EdgeId_sym(_Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_sym(_UnderlyingPtr);
        }

        // among each pair of sym-edges: one is always even and the other is odd
        /// Generated from method `MR::NoDefInit<MR::EdgeId>::even`.
        public unsafe bool Even()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_even", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_EdgeId_even(_Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_even(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::odd`.
        public unsafe bool Odd()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_odd", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_EdgeId_odd(_Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_odd(_UnderlyingPtr) != 0;
        }

        // returns unique identifier of the edge ignoring its direction
        /// Generated from method `MR::NoDefInit<MR::EdgeId>::undirected`.
        public unsafe MR.UndirectedEdgeId Undirected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_undirected", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_NoDefInit_MR_EdgeId_undirected(_Underlying *_this);
            return __MR_NoDefInit_MR_EdgeId_undirected(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MREdgeId _this, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_EdgeId_MR_EdgeId(MR.Const_NoDefInit_MREdgeId._Underlying *_this, MR.EdgeId b);
            return __MR_equal_MR_NoDefInit_MR_EdgeId_MR_EdgeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MREdgeId _this, MR.EdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::operator<`.
        public unsafe bool Less(MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_EdgeId_MR_EdgeId(_Underlying *_this, MR.EdgeId b);
            return __MR_less_MR_NoDefInit_MR_EdgeId_MR_EdgeId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.EdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.EdgeId)
                return this == (MR.EdgeId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::EdgeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgeId`
    /// This is the non-const half of the class.
    public class NoDefInit_MREdgeId : Const_NoDefInit_MREdgeId
    {
        internal unsafe NoDefInit_MREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_EdgeId(NoDefInit_MREdgeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_UpcastTo_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_UpcastTo_MR_EdgeId(_Underlying *_this);
            MR.Mut_EdgeId ret = new(__MR_NoDefInit_MR_EdgeId_UpcastTo_MR_EdgeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_EdgeId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_EdgeId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::EdgeId>::NoDefInit`.
        public unsafe NoDefInit_MREdgeId(MR.Const_NoDefInit_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_ConstructFromAnother(MR.NoDefInit_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_EdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::operator=`.
        public unsafe MR.NoDefInit_MREdgeId Assign(MR.Const_NoDefInit_MREdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MREdgeId._Underlying *_other);
            return new(__MR_NoDefInit_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_EdgeId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_EdgeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::operator-=`.
        public unsafe MR.Mut_EdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_EdgeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::EdgeId>::operator+=`.
        public unsafe MR.Mut_EdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_EdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_NoDefInit_MR_EdgeId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_EdgeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MREdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MREdgeId`/`Const_NoDefInit_MREdgeId` directly.
    public class _InOptMut_NoDefInit_MREdgeId
    {
        public NoDefInit_MREdgeId? Opt;

        public _InOptMut_NoDefInit_MREdgeId() {}
        public _InOptMut_NoDefInit_MREdgeId(NoDefInit_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MREdgeId(NoDefInit_MREdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MREdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MREdgeId`/`Const_NoDefInit_MREdgeId` to pass it to the function.
    public class _InOptConst_NoDefInit_MREdgeId
    {
        public Const_NoDefInit_MREdgeId? Opt;

        public _InOptConst_NoDefInit_MREdgeId() {}
        public _InOptConst_NoDefInit_MREdgeId(Const_NoDefInit_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MREdgeId(Const_NoDefInit_MREdgeId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::ObjId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRObjId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.ObjId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRObjId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_ObjId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_ObjId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRObjId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjId(Const_NoDefInit_MRObjId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_UpcastTo_MR_ObjId", ExactSpelling = true)]
            extern static MR.Const_ObjId._Underlying *__MR_NoDefInit_MR_ObjId_UpcastTo_MR_ObjId(_Underlying *_this);
            MR.Const_ObjId ret = new(__MR_NoDefInit_MR_ObjId_UpcastTo_MR_ObjId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_ObjId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_ObjId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_NoDefInit_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::ObjId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRObjId(MR.Const_NoDefInit_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_NoDefInit_MR_ObjId_ConstructFromAnother(MR.NoDefInit_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_ObjId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::ObjId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_ObjId_ConvertTo_int(MR.Const_NoDefInit_MRObjId._Underlying *_this);
            return __MR_NoDefInit_MR_ObjId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::ObjId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_ObjId_ConvertTo_bool(MR.Const_NoDefInit_MRObjId._Underlying *_this);
            return __MR_NoDefInit_MR_ObjId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_ObjId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_ObjId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRObjId _this, MR.ObjId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_ObjId_MR_ObjId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_ObjId_MR_ObjId(MR.Const_NoDefInit_MRObjId._Underlying *_this, MR.ObjId b);
            return __MR_equal_MR_NoDefInit_MR_ObjId_MR_ObjId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRObjId _this, MR.ObjId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::operator<`.
        public unsafe bool Less(MR.ObjId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_ObjId_MR_ObjId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_ObjId_MR_ObjId(_Underlying *_this, MR.ObjId b);
            return __MR_less_MR_NoDefInit_MR_ObjId_MR_ObjId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.ObjId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.ObjId)
                return this == (MR.ObjId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::ObjId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRObjId : Const_NoDefInit_MRObjId
    {
        internal unsafe NoDefInit_MRObjId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_ObjId(NoDefInit_MRObjId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_UpcastTo_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_NoDefInit_MR_ObjId_UpcastTo_MR_ObjId(_Underlying *_this);
            MR.Mut_ObjId ret = new(__MR_NoDefInit_MR_ObjId_UpcastTo_MR_ObjId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_ObjId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_ObjId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_NoDefInit_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::ObjId>::NoDefInit`.
        public unsafe NoDefInit_MRObjId(MR.Const_NoDefInit_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_NoDefInit_MR_ObjId_ConstructFromAnother(MR.NoDefInit_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_ObjId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::operator=`.
        public unsafe MR.NoDefInit_MRObjId Assign(MR.Const_NoDefInit_MRObjId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_NoDefInit_MR_ObjId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRObjId._Underlying *_other);
            return new(__MR_NoDefInit_MR_ObjId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_ObjId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_ObjId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::operator-=`.
        public unsafe MR.Mut_ObjId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_NoDefInit_MR_ObjId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_ObjId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::ObjId>::operator+=`.
        public unsafe MR.Mut_ObjId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_ObjId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_NoDefInit_MR_ObjId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_ObjId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRObjId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRObjId`/`Const_NoDefInit_MRObjId` directly.
    public class _InOptMut_NoDefInit_MRObjId
    {
        public NoDefInit_MRObjId? Opt;

        public _InOptMut_NoDefInit_MRObjId() {}
        public _InOptMut_NoDefInit_MRObjId(NoDefInit_MRObjId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRObjId(NoDefInit_MRObjId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRObjId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRObjId`/`Const_NoDefInit_MRObjId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRObjId
    {
        public Const_NoDefInit_MRObjId? Opt;

        public _InOptConst_NoDefInit_MRObjId() {}
        public _InOptConst_NoDefInit_MRObjId(Const_NoDefInit_MRObjId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRObjId(Const_NoDefInit_MRObjId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::GraphVertId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::GraphVertId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRGraphVertId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.GraphVertId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_GraphVertId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_GraphVertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRGraphVertId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_GraphVertId(Const_NoDefInit_MRGraphVertId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_UpcastTo_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Const_GraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_UpcastTo_MR_GraphVertId(_Underlying *_this);
            MR.Const_GraphVertId ret = new(__MR_NoDefInit_MR_GraphVertId_UpcastTo_MR_GraphVertId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_GraphVertId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_GraphVertId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::GraphVertId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRGraphVertId(MR.Const_NoDefInit_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_ConstructFromAnother(MR.NoDefInit_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphVertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::GraphVertId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRGraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_GraphVertId_ConvertTo_int(MR.Const_NoDefInit_MRGraphVertId._Underlying *_this);
            return __MR_NoDefInit_MR_GraphVertId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::GraphVertId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRGraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_GraphVertId_ConvertTo_bool(MR.Const_NoDefInit_MRGraphVertId._Underlying *_this);
            return __MR_NoDefInit_MR_GraphVertId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_GraphVertId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_GraphVertId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRGraphVertId _this, MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_GraphVertId_MR_GraphVertId(MR.Const_NoDefInit_MRGraphVertId._Underlying *_this, MR.GraphVertId b);
            return __MR_equal_MR_NoDefInit_MR_GraphVertId_MR_GraphVertId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRGraphVertId _this, MR.GraphVertId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::operator<`.
        public unsafe bool Less(MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_GraphVertId_MR_GraphVertId(_Underlying *_this, MR.GraphVertId b);
            return __MR_less_MR_NoDefInit_MR_GraphVertId_MR_GraphVertId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.GraphVertId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.GraphVertId)
                return this == (MR.GraphVertId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::GraphVertId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::GraphVertId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRGraphVertId : Const_NoDefInit_MRGraphVertId
    {
        internal unsafe NoDefInit_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_GraphVertId(NoDefInit_MRGraphVertId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_UpcastTo_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_UpcastTo_MR_GraphVertId(_Underlying *_this);
            MR.Mut_GraphVertId ret = new(__MR_NoDefInit_MR_GraphVertId_UpcastTo_MR_GraphVertId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_GraphVertId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_GraphVertId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::GraphVertId>::NoDefInit`.
        public unsafe NoDefInit_MRGraphVertId(MR.Const_NoDefInit_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_ConstructFromAnother(MR.NoDefInit_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphVertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::operator=`.
        public unsafe MR.NoDefInit_MRGraphVertId Assign(MR.Const_NoDefInit_MRGraphVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRGraphVertId._Underlying *_other);
            return new(__MR_NoDefInit_MR_GraphVertId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_GraphVertId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_GraphVertId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::operator-=`.
        public unsafe MR.Mut_GraphVertId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_GraphVertId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphVertId>::operator+=`.
        public unsafe MR.Mut_GraphVertId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphVertId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_NoDefInit_MR_GraphVertId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_GraphVertId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRGraphVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRGraphVertId`/`Const_NoDefInit_MRGraphVertId` directly.
    public class _InOptMut_NoDefInit_MRGraphVertId
    {
        public NoDefInit_MRGraphVertId? Opt;

        public _InOptMut_NoDefInit_MRGraphVertId() {}
        public _InOptMut_NoDefInit_MRGraphVertId(NoDefInit_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRGraphVertId(NoDefInit_MRGraphVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRGraphVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRGraphVertId`/`Const_NoDefInit_MRGraphVertId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRGraphVertId
    {
        public Const_NoDefInit_MRGraphVertId? Opt;

        public _InOptConst_NoDefInit_MRGraphVertId() {}
        public _InOptConst_NoDefInit_MRGraphVertId(Const_NoDefInit_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRGraphVertId(Const_NoDefInit_MRGraphVertId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::GraphEdgeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::GraphEdgeId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRGraphEdgeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.GraphEdgeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_GraphEdgeId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_GraphEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRGraphEdgeId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_GraphEdgeId(Const_NoDefInit_MRGraphEdgeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_UpcastTo_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Const_GraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_UpcastTo_MR_GraphEdgeId(_Underlying *_this);
            MR.Const_GraphEdgeId ret = new(__MR_NoDefInit_MR_GraphEdgeId_UpcastTo_MR_GraphEdgeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_GraphEdgeId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_GraphEdgeId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::GraphEdgeId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRGraphEdgeId(MR.Const_NoDefInit_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_ConstructFromAnother(MR.NoDefInit_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::GraphEdgeId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRGraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_GraphEdgeId_ConvertTo_int(MR.Const_NoDefInit_MRGraphEdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_GraphEdgeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::GraphEdgeId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRGraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_GraphEdgeId_ConvertTo_bool(MR.Const_NoDefInit_MRGraphEdgeId._Underlying *_this);
            return __MR_NoDefInit_MR_GraphEdgeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_GraphEdgeId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_GraphEdgeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRGraphEdgeId _this, MR.GraphEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_GraphEdgeId_MR_GraphEdgeId(MR.Const_NoDefInit_MRGraphEdgeId._Underlying *_this, MR.GraphEdgeId b);
            return __MR_equal_MR_NoDefInit_MR_GraphEdgeId_MR_GraphEdgeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRGraphEdgeId _this, MR.GraphEdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::operator<`.
        public unsafe bool Less(MR.GraphEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_GraphEdgeId_MR_GraphEdgeId(_Underlying *_this, MR.GraphEdgeId b);
            return __MR_less_MR_NoDefInit_MR_GraphEdgeId_MR_GraphEdgeId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.GraphEdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.GraphEdgeId)
                return this == (MR.GraphEdgeId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::GraphEdgeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::GraphEdgeId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRGraphEdgeId : Const_NoDefInit_MRGraphEdgeId
    {
        internal unsafe NoDefInit_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_GraphEdgeId(NoDefInit_MRGraphEdgeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_UpcastTo_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_UpcastTo_MR_GraphEdgeId(_Underlying *_this);
            MR.Mut_GraphEdgeId ret = new(__MR_NoDefInit_MR_GraphEdgeId_UpcastTo_MR_GraphEdgeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_GraphEdgeId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_GraphEdgeId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::GraphEdgeId>::NoDefInit`.
        public unsafe NoDefInit_MRGraphEdgeId(MR.Const_NoDefInit_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_ConstructFromAnother(MR.NoDefInit_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_GraphEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::operator=`.
        public unsafe MR.NoDefInit_MRGraphEdgeId Assign(MR.Const_NoDefInit_MRGraphEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRGraphEdgeId._Underlying *_other);
            return new(__MR_NoDefInit_MR_GraphEdgeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_GraphEdgeId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_GraphEdgeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::operator-=`.
        public unsafe MR.Mut_GraphEdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_GraphEdgeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::GraphEdgeId>::operator+=`.
        public unsafe MR.Mut_GraphEdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_GraphEdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_NoDefInit_MR_GraphEdgeId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_GraphEdgeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRGraphEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRGraphEdgeId`/`Const_NoDefInit_MRGraphEdgeId` directly.
    public class _InOptMut_NoDefInit_MRGraphEdgeId
    {
        public NoDefInit_MRGraphEdgeId? Opt;

        public _InOptMut_NoDefInit_MRGraphEdgeId() {}
        public _InOptMut_NoDefInit_MRGraphEdgeId(NoDefInit_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRGraphEdgeId(NoDefInit_MRGraphEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRGraphEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRGraphEdgeId`/`Const_NoDefInit_MRGraphEdgeId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRGraphEdgeId
    {
        public Const_NoDefInit_MRGraphEdgeId? Opt;

        public _InOptConst_NoDefInit_MRGraphEdgeId() {}
        public _InOptConst_NoDefInit_MRGraphEdgeId(Const_NoDefInit_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRGraphEdgeId(Const_NoDefInit_MRGraphEdgeId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::VoxelId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRVoxelId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.VoxelId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRVoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_VoxelId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRVoxelId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_VoxelId(Const_NoDefInit_MRVoxelId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_UpcastTo_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Const_VoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_UpcastTo_MR_VoxelId(_Underlying *_this);
            MR.Const_VoxelId ret = new(__MR_NoDefInit_MR_VoxelId_UpcastTo_MR_VoxelId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe ulong Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_Get_id_", ExactSpelling = true)]
                extern static ulong *__MR_NoDefInit_MR_VoxelId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_VoxelId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::VoxelId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRVoxelId(MR.Const_NoDefInit_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_ConstructFromAnother(MR.NoDefInit_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::VoxelId>::operator MR_uint64_t`.
        public static unsafe implicit operator ulong(MR.Const_NoDefInit_MRVoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_ConvertTo_uint64_t", ExactSpelling = true)]
            extern static ulong __MR_NoDefInit_MR_VoxelId_ConvertTo_uint64_t(MR.Const_NoDefInit_MRVoxelId._Underlying *_this);
            return __MR_NoDefInit_MR_VoxelId_ConvertTo_uint64_t(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::VoxelId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRVoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_VoxelId_ConvertTo_bool(MR.Const_NoDefInit_MRVoxelId._Underlying *_this);
            return __MR_NoDefInit_MR_VoxelId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_VoxelId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_VoxelId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRVoxelId _this, MR.VoxelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_VoxelId_MR_VoxelId(MR.Const_NoDefInit_MRVoxelId._Underlying *_this, MR.VoxelId b);
            return __MR_equal_MR_NoDefInit_MR_VoxelId_MR_VoxelId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRVoxelId _this, MR.VoxelId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::operator<`.
        public unsafe bool Less(MR.VoxelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_VoxelId_MR_VoxelId(_Underlying *_this, MR.VoxelId b);
            return __MR_less_MR_NoDefInit_MR_VoxelId_MR_VoxelId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.VoxelId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.VoxelId)
                return this == (MR.VoxelId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::VoxelId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRVoxelId : Const_NoDefInit_MRVoxelId
    {
        internal unsafe NoDefInit_MRVoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_VoxelId(NoDefInit_MRVoxelId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_UpcastTo_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_UpcastTo_MR_VoxelId(_Underlying *_this);
            MR.Mut_VoxelId ret = new(__MR_NoDefInit_MR_VoxelId_UpcastTo_MR_VoxelId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref ulong Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_GetMutable_id_", ExactSpelling = true)]
                extern static ulong *__MR_NoDefInit_MR_VoxelId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_VoxelId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::VoxelId>::NoDefInit`.
        public unsafe NoDefInit_MRVoxelId(MR.Const_NoDefInit_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_ConstructFromAnother(MR.NoDefInit_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::operator=`.
        public unsafe MR.NoDefInit_MRVoxelId Assign(MR.Const_NoDefInit_MRVoxelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRVoxelId._Underlying *_other);
            return new(__MR_NoDefInit_MR_VoxelId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::get`.
        public unsafe ref ulong Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_get", ExactSpelling = true)]
            extern static ulong *__MR_NoDefInit_MR_VoxelId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_VoxelId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::operator-=`.
        public unsafe MR.Mut_VoxelId SubAssign(ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_sub_assign(_Underlying *_this, ulong a);
            return new(__MR_NoDefInit_MR_VoxelId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::VoxelId>::operator+=`.
        public unsafe MR.Mut_VoxelId AddAssign(ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_VoxelId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_NoDefInit_MR_VoxelId_add_assign(_Underlying *_this, ulong a);
            return new(__MR_NoDefInit_MR_VoxelId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRVoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRVoxelId`/`Const_NoDefInit_MRVoxelId` directly.
    public class _InOptMut_NoDefInit_MRVoxelId
    {
        public NoDefInit_MRVoxelId? Opt;

        public _InOptMut_NoDefInit_MRVoxelId() {}
        public _InOptMut_NoDefInit_MRVoxelId(NoDefInit_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRVoxelId(NoDefInit_MRVoxelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRVoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRVoxelId`/`Const_NoDefInit_MRVoxelId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRVoxelId
    {
        public Const_NoDefInit_MRVoxelId? Opt;

        public _InOptConst_NoDefInit_MRVoxelId() {}
        public _InOptConst_NoDefInit_MRVoxelId(Const_NoDefInit_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRVoxelId(Const_NoDefInit_MRVoxelId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::PixelId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::PixelId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRPixelId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.PixelId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRPixelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_PixelId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_PixelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRPixelId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_PixelId(Const_NoDefInit_MRPixelId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_UpcastTo_MR_PixelId", ExactSpelling = true)]
            extern static MR.Const_PixelId._Underlying *__MR_NoDefInit_MR_PixelId_UpcastTo_MR_PixelId(_Underlying *_this);
            MR.Const_PixelId ret = new(__MR_NoDefInit_MR_PixelId_UpcastTo_MR_PixelId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_PixelId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_PixelId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_NoDefInit_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::PixelId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRPixelId(MR.Const_NoDefInit_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_NoDefInit_MR_PixelId_ConstructFromAnother(MR.NoDefInit_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_PixelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::PixelId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRPixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_PixelId_ConvertTo_int(MR.Const_NoDefInit_MRPixelId._Underlying *_this);
            return __MR_NoDefInit_MR_PixelId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::PixelId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRPixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_PixelId_ConvertTo_bool(MR.Const_NoDefInit_MRPixelId._Underlying *_this);
            return __MR_NoDefInit_MR_PixelId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_PixelId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_PixelId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRPixelId _this, MR.PixelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_PixelId_MR_PixelId(MR.Const_NoDefInit_MRPixelId._Underlying *_this, MR.PixelId b);
            return __MR_equal_MR_NoDefInit_MR_PixelId_MR_PixelId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRPixelId _this, MR.PixelId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::operator<`.
        public unsafe bool Less(MR.PixelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_PixelId_MR_PixelId(_Underlying *_this, MR.PixelId b);
            return __MR_less_MR_NoDefInit_MR_PixelId_MR_PixelId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.PixelId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.PixelId)
                return this == (MR.PixelId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::PixelId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::PixelId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRPixelId : Const_NoDefInit_MRPixelId
    {
        internal unsafe NoDefInit_MRPixelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_PixelId(NoDefInit_MRPixelId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_UpcastTo_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_NoDefInit_MR_PixelId_UpcastTo_MR_PixelId(_Underlying *_this);
            MR.Mut_PixelId ret = new(__MR_NoDefInit_MR_PixelId_UpcastTo_MR_PixelId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_PixelId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_PixelId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_NoDefInit_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::PixelId>::NoDefInit`.
        public unsafe NoDefInit_MRPixelId(MR.Const_NoDefInit_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_NoDefInit_MR_PixelId_ConstructFromAnother(MR.NoDefInit_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_PixelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::operator=`.
        public unsafe MR.NoDefInit_MRPixelId Assign(MR.Const_NoDefInit_MRPixelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_NoDefInit_MR_PixelId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRPixelId._Underlying *_other);
            return new(__MR_NoDefInit_MR_PixelId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_PixelId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_PixelId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::operator-=`.
        public unsafe MR.Mut_PixelId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_NoDefInit_MR_PixelId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_PixelId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::PixelId>::operator+=`.
        public unsafe MR.Mut_PixelId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_PixelId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_NoDefInit_MR_PixelId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_PixelId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRPixelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRPixelId`/`Const_NoDefInit_MRPixelId` directly.
    public class _InOptMut_NoDefInit_MRPixelId
    {
        public NoDefInit_MRPixelId? Opt;

        public _InOptMut_NoDefInit_MRPixelId() {}
        public _InOptMut_NoDefInit_MRPixelId(NoDefInit_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRPixelId(NoDefInit_MRPixelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRPixelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRPixelId`/`Const_NoDefInit_MRPixelId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRPixelId
    {
        public Const_NoDefInit_MRPixelId? Opt;

        public _InOptConst_NoDefInit_MRPixelId() {}
        public _InOptConst_NoDefInit_MRPixelId(Const_NoDefInit_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRPixelId(Const_NoDefInit_MRPixelId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::RegionId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RegionId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRRegionId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.RegionId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRRegionId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_RegionId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_RegionId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRRegionId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RegionId(Const_NoDefInit_MRRegionId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_UpcastTo_MR_RegionId", ExactSpelling = true)]
            extern static MR.Const_RegionId._Underlying *__MR_NoDefInit_MR_RegionId_UpcastTo_MR_RegionId(_Underlying *_this);
            MR.Const_RegionId ret = new(__MR_NoDefInit_MR_RegionId_UpcastTo_MR_RegionId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_RegionId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_RegionId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_NoDefInit_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::RegionId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRRegionId(MR.Const_NoDefInit_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_NoDefInit_MR_RegionId_ConstructFromAnother(MR.NoDefInit_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_RegionId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::RegionId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRRegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_RegionId_ConvertTo_int(MR.Const_NoDefInit_MRRegionId._Underlying *_this);
            return __MR_NoDefInit_MR_RegionId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::RegionId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRRegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_RegionId_ConvertTo_bool(MR.Const_NoDefInit_MRRegionId._Underlying *_this);
            return __MR_NoDefInit_MR_RegionId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_RegionId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_RegionId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRRegionId _this, MR.RegionId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_RegionId_MR_RegionId(MR.Const_NoDefInit_MRRegionId._Underlying *_this, MR.RegionId b);
            return __MR_equal_MR_NoDefInit_MR_RegionId_MR_RegionId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRRegionId _this, MR.RegionId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::operator<`.
        public unsafe bool Less(MR.RegionId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_RegionId_MR_RegionId(_Underlying *_this, MR.RegionId b);
            return __MR_less_MR_NoDefInit_MR_RegionId_MR_RegionId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.RegionId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.RegionId)
                return this == (MR.RegionId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::RegionId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RegionId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRRegionId : Const_NoDefInit_MRRegionId
    {
        internal unsafe NoDefInit_MRRegionId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_RegionId(NoDefInit_MRRegionId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_UpcastTo_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_NoDefInit_MR_RegionId_UpcastTo_MR_RegionId(_Underlying *_this);
            MR.Mut_RegionId ret = new(__MR_NoDefInit_MR_RegionId_UpcastTo_MR_RegionId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_RegionId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_RegionId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_NoDefInit_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::RegionId>::NoDefInit`.
        public unsafe NoDefInit_MRRegionId(MR.Const_NoDefInit_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_NoDefInit_MR_RegionId_ConstructFromAnother(MR.NoDefInit_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_RegionId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::operator=`.
        public unsafe MR.NoDefInit_MRRegionId Assign(MR.Const_NoDefInit_MRRegionId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_NoDefInit_MR_RegionId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRRegionId._Underlying *_other);
            return new(__MR_NoDefInit_MR_RegionId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_RegionId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_RegionId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::operator-=`.
        public unsafe MR.Mut_RegionId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_NoDefInit_MR_RegionId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_RegionId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::RegionId>::operator+=`.
        public unsafe MR.Mut_RegionId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_RegionId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_NoDefInit_MR_RegionId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_RegionId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRRegionId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRRegionId`/`Const_NoDefInit_MRRegionId` directly.
    public class _InOptMut_NoDefInit_MRRegionId
    {
        public NoDefInit_MRRegionId? Opt;

        public _InOptMut_NoDefInit_MRRegionId() {}
        public _InOptMut_NoDefInit_MRRegionId(NoDefInit_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRRegionId(NoDefInit_MRRegionId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRRegionId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRRegionId`/`Const_NoDefInit_MRRegionId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRRegionId
    {
        public Const_NoDefInit_MRRegionId? Opt;

        public _InOptConst_NoDefInit_MRRegionId() {}
        public _InOptConst_NoDefInit_MRRegionId(Const_NoDefInit_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRRegionId(Const_NoDefInit_MRRegionId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::NodeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NodeId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRNodeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.NodeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRNodeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_NodeId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_NodeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRNodeId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_NodeId(Const_NoDefInit_MRNodeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_UpcastTo_MR_NodeId", ExactSpelling = true)]
            extern static MR.Const_NodeId._Underlying *__MR_NoDefInit_MR_NodeId_UpcastTo_MR_NodeId(_Underlying *_this);
            MR.Const_NodeId ret = new(__MR_NoDefInit_MR_NodeId_UpcastTo_MR_NodeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_NodeId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_NodeId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_NoDefInit_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::NodeId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRNodeId(MR.Const_NoDefInit_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_NoDefInit_MR_NodeId_ConstructFromAnother(MR.NoDefInit_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_NodeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::NodeId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRNodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_NodeId_ConvertTo_int(MR.Const_NoDefInit_MRNodeId._Underlying *_this);
            return __MR_NoDefInit_MR_NodeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::NodeId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRNodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_NodeId_ConvertTo_bool(MR.Const_NoDefInit_MRNodeId._Underlying *_this);
            return __MR_NoDefInit_MR_NodeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_NodeId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_NodeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRNodeId _this, MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_NodeId_MR_NodeId(MR.Const_NoDefInit_MRNodeId._Underlying *_this, MR.NodeId b);
            return __MR_equal_MR_NoDefInit_MR_NodeId_MR_NodeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRNodeId _this, MR.NodeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::operator<`.
        public unsafe bool Less(MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_NodeId_MR_NodeId(_Underlying *_this, MR.NodeId b);
            return __MR_less_MR_NoDefInit_MR_NodeId_MR_NodeId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.NodeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.NodeId)
                return this == (MR.NodeId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::NodeId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NodeId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRNodeId : Const_NoDefInit_MRNodeId
    {
        internal unsafe NoDefInit_MRNodeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_NodeId(NoDefInit_MRNodeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_UpcastTo_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NoDefInit_MR_NodeId_UpcastTo_MR_NodeId(_Underlying *_this);
            MR.Mut_NodeId ret = new(__MR_NoDefInit_MR_NodeId_UpcastTo_MR_NodeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_NodeId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_NodeId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_NoDefInit_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::NodeId>::NoDefInit`.
        public unsafe NoDefInit_MRNodeId(MR.Const_NoDefInit_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_NoDefInit_MR_NodeId_ConstructFromAnother(MR.NoDefInit_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_NodeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::operator=`.
        public unsafe MR.NoDefInit_MRNodeId Assign(MR.Const_NoDefInit_MRNodeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_NoDefInit_MR_NodeId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRNodeId._Underlying *_other);
            return new(__MR_NoDefInit_MR_NodeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_NodeId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_NodeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::operator-=`.
        public unsafe MR.Mut_NodeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NoDefInit_MR_NodeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_NodeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::NodeId>::operator+=`.
        public unsafe MR.Mut_NodeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_NodeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NoDefInit_MR_NodeId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_NodeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRNodeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRNodeId`/`Const_NoDefInit_MRNodeId` directly.
    public class _InOptMut_NoDefInit_MRNodeId
    {
        public NoDefInit_MRNodeId? Opt;

        public _InOptMut_NoDefInit_MRNodeId() {}
        public _InOptMut_NoDefInit_MRNodeId(NoDefInit_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRNodeId(NoDefInit_MRNodeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRNodeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRNodeId`/`Const_NoDefInit_MRNodeId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRNodeId
    {
        public Const_NoDefInit_MRNodeId? Opt;

        public _InOptConst_NoDefInit_MRNodeId() {}
        public _InOptConst_NoDefInit_MRNodeId(Const_NoDefInit_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRNodeId(Const_NoDefInit_MRNodeId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::TextureId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::TextureId`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRTextureId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.TextureId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRTextureId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_TextureId_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_TextureId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRTextureId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_TextureId(Const_NoDefInit_MRTextureId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_UpcastTo_MR_TextureId", ExactSpelling = true)]
            extern static MR.Const_TextureId._Underlying *__MR_NoDefInit_MR_TextureId_UpcastTo_MR_TextureId(_Underlying *_this);
            MR.Const_TextureId ret = new(__MR_NoDefInit_MR_TextureId_UpcastTo_MR_TextureId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_TextureId_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_TextureId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_NoDefInit_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::TextureId>::NoDefInit`.
        public unsafe Const_NoDefInit_MRTextureId(MR.Const_NoDefInit_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_NoDefInit_MR_TextureId_ConstructFromAnother(MR.NoDefInit_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_TextureId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::TextureId>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRTextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_TextureId_ConvertTo_int(MR.Const_NoDefInit_MRTextureId._Underlying *_this);
            return __MR_NoDefInit_MR_TextureId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::TextureId>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRTextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_TextureId_ConvertTo_bool(MR.Const_NoDefInit_MRTextureId._Underlying *_this);
            return __MR_NoDefInit_MR_TextureId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_TextureId_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_TextureId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRTextureId _this, MR.TextureId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_TextureId_MR_TextureId(MR.Const_NoDefInit_MRTextureId._Underlying *_this, MR.TextureId b);
            return __MR_equal_MR_NoDefInit_MR_TextureId_MR_TextureId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRTextureId _this, MR.TextureId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::operator<`.
        public unsafe bool Less(MR.TextureId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_TextureId_MR_TextureId(_Underlying *_this, MR.TextureId b);
            return __MR_less_MR_NoDefInit_MR_TextureId_MR_TextureId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.TextureId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.TextureId)
                return this == (MR.TextureId)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::TextureId>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::TextureId`
    /// This is the non-const half of the class.
    public class NoDefInit_MRTextureId : Const_NoDefInit_MRTextureId
    {
        internal unsafe NoDefInit_MRTextureId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_TextureId(NoDefInit_MRTextureId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_UpcastTo_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_NoDefInit_MR_TextureId_UpcastTo_MR_TextureId(_Underlying *_this);
            MR.Mut_TextureId ret = new(__MR_NoDefInit_MR_TextureId_UpcastTo_MR_TextureId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_TextureId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_TextureId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_NoDefInit_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::TextureId>::NoDefInit`.
        public unsafe NoDefInit_MRTextureId(MR.Const_NoDefInit_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_NoDefInit_MR_TextureId_ConstructFromAnother(MR.NoDefInit_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_TextureId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::operator=`.
        public unsafe MR.NoDefInit_MRTextureId Assign(MR.Const_NoDefInit_MRTextureId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_NoDefInit_MR_TextureId_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRTextureId._Underlying *_other);
            return new(__MR_NoDefInit_MR_TextureId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_TextureId_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_TextureId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::operator-=`.
        public unsafe MR.Mut_TextureId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_NoDefInit_MR_TextureId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_TextureId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::TextureId>::operator+=`.
        public unsafe MR.Mut_TextureId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_TextureId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_NoDefInit_MR_TextureId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_TextureId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRTextureId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRTextureId`/`Const_NoDefInit_MRTextureId` directly.
    public class _InOptMut_NoDefInit_MRTextureId
    {
        public NoDefInit_MRTextureId? Opt;

        public _InOptMut_NoDefInit_MRTextureId() {}
        public _InOptMut_NoDefInit_MRTextureId(NoDefInit_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRTextureId(NoDefInit_MRTextureId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRTextureId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRTextureId`/`Const_NoDefInit_MRTextureId` to pass it to the function.
    public class _InOptConst_NoDefInit_MRTextureId
    {
        public Const_NoDefInit_MRTextureId? Opt;

        public _InOptConst_NoDefInit_MRTextureId() {}
        public _InOptConst_NoDefInit_MRTextureId(Const_NoDefInit_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRTextureId(Const_NoDefInit_MRTextureId value) {return new(value);}
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Id<MR::ICPElemtTag>`
    /// This is the const half of the class.
    public class Const_NoDefInit_MRIdMRICPElemtTag : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Id_MRICPElemtTag>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoDefInit_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_Destroy", ExactSpelling = true)]
            extern static void __MR_NoDefInit_MR_Id_MR_ICPElemtTag_Destroy(_Underlying *_this);
            __MR_NoDefInit_MR_Id_MR_ICPElemtTag_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoDefInit_MRIdMRICPElemtTag() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_Id_MRICPElemtTag(Const_NoDefInit_MRIdMRICPElemtTag self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_UpcastTo_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.Const_Id_MRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_UpcastTo_MR_Id_MR_ICPElemtTag(_Underlying *_this);
            MR.Const_Id_MRICPElemtTag ret = new(__MR_NoDefInit_MR_Id_MR_ICPElemtTag_UpcastTo_MR_Id_MR_ICPElemtTag(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_Get_id_(_Underlying *_this);
                return *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoDefInit_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::NoDefInit`.
        public unsafe Const_NoDefInit_MRIdMRICPElemtTag(MR.Const_NoDefInit_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.NoDefInit_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator int`.
        public static unsafe implicit operator int(MR.Const_NoDefInit_MRIdMRICPElemtTag _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConvertTo_int(MR.Const_NoDefInit_MRIdMRICPElemtTag._Underlying *_this);
            return __MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoDefInit_MRIdMRICPElemtTag _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConvertTo_bool(MR.Const_NoDefInit_MRIdMRICPElemtTag._Underlying *_this);
            return __MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_valid", ExactSpelling = true)]
            extern static byte __MR_NoDefInit_MR_Id_MR_ICPElemtTag_valid(_Underlying *_this);
            return __MR_NoDefInit_MR_Id_MR_ICPElemtTag_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator==`.
        public static unsafe bool operator==(MR.Const_NoDefInit_MRIdMRICPElemtTag _this, MR.Const_Id_MRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoDefInit_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoDefInit_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(MR.Const_NoDefInit_MRIdMRICPElemtTag._Underlying *_this, MR.Id_MRICPElemtTag._Underlying *b);
            return __MR_equal_MR_NoDefInit_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_this._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoDefInit_MRIdMRICPElemtTag _this, MR.Const_Id_MRICPElemtTag b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator<`.
        public unsafe bool Less(MR.Const_Id_MRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoDefInit_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoDefInit_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *b);
            return __MR_less_MR_NoDefInit_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        // IEquatable:

        public bool Equals(MR.Const_Id_MRICPElemtTag? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Id_MRICPElemtTag)
                return this == (MR.Const_Id_MRICPElemtTag)other;
            return false;
        }
    }

    // this class is similar to T, but does not make default initialization of the fields for best performance
    /// Generated from class `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Id<MR::ICPElemtTag>`
    /// This is the non-const half of the class.
    public class NoDefInit_MRIdMRICPElemtTag : Const_NoDefInit_MRIdMRICPElemtTag
    {
        internal unsafe NoDefInit_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Id_MRICPElemtTag(NoDefInit_MRIdMRICPElemtTag self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_UpcastTo_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_UpcastTo_MR_Id_MR_ICPElemtTag(_Underlying *_this);
            MR.Id_MRICPElemtTag ret = new(__MR_NoDefInit_MR_Id_MR_ICPElemtTag_UpcastTo_MR_Id_MR_ICPElemtTag(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoDefInit_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_NoDefInit_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::NoDefInit`.
        public unsafe NoDefInit_MRIdMRICPElemtTag(MR.Const_NoDefInit_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.NoDefInit_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_NoDefInit_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator=`.
        public unsafe MR.NoDefInit_MRIdMRICPElemtTag Assign(MR.Const_NoDefInit_MRIdMRICPElemtTag _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_AssignFromAnother(_Underlying *_this, MR.NoDefInit_MRIdMRICPElemtTag._Underlying *_other);
            return new(__MR_NoDefInit_MR_Id_MR_ICPElemtTag_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_get", ExactSpelling = true)]
            extern static int *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_get(_Underlying *_this);
            return ref *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator-=`.
        public unsafe MR.Id_MRICPElemtTag SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_sub_assign", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_Id_MR_ICPElemtTag_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>::operator+=`.
        public unsafe MR.Id_MRICPElemtTag AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoDefInit_MR_Id_MR_ICPElemtTag_add_assign", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_NoDefInit_MR_Id_MR_ICPElemtTag_add_assign(_Underlying *_this, int a);
            return new(__MR_NoDefInit_MR_Id_MR_ICPElemtTag_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoDefInit_MRIdMRICPElemtTag` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoDefInit_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRIdMRICPElemtTag`/`Const_NoDefInit_MRIdMRICPElemtTag` directly.
    public class _InOptMut_NoDefInit_MRIdMRICPElemtTag
    {
        public NoDefInit_MRIdMRICPElemtTag? Opt;

        public _InOptMut_NoDefInit_MRIdMRICPElemtTag() {}
        public _InOptMut_NoDefInit_MRIdMRICPElemtTag(NoDefInit_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptMut_NoDefInit_MRIdMRICPElemtTag(NoDefInit_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoDefInit_MRIdMRICPElemtTag` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoDefInit_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoDefInit_MRIdMRICPElemtTag`/`Const_NoDefInit_MRIdMRICPElemtTag` to pass it to the function.
    public class _InOptConst_NoDefInit_MRIdMRICPElemtTag
    {
        public Const_NoDefInit_MRIdMRICPElemtTag? Opt;

        public _InOptConst_NoDefInit_MRIdMRICPElemtTag() {}
        public _InOptConst_NoDefInit_MRIdMRICPElemtTag(Const_NoDefInit_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptConst_NoDefInit_MRIdMRICPElemtTag(Const_NoDefInit_MRIdMRICPElemtTag value) {return new(value);}
    }
}
