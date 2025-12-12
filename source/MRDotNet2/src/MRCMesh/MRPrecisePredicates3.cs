public static partial class MR
{
    /// Generated from class `MR::PreciseVertCoords`.
    /// This is the const half of the class.
    public class Const_PreciseVertCoords : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PreciseVertCoords(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_Destroy", ExactSpelling = true)]
            extern static void __MR_PreciseVertCoords_Destroy(_Underlying *_this);
            __MR_PreciseVertCoords_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PreciseVertCoords() {Dispose(false);}

        ///< unique id of the vertex (in both meshes)
        public unsafe MR.Const_VertId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_Get_id", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PreciseVertCoords_Get_id(_Underlying *_this);
                return new(__MR_PreciseVertCoords_Get_id(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< integer coordinates of the vertex
        public unsafe MR.Const_Vector3i Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_Get_pt", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_PreciseVertCoords_Get_pt(_Underlying *_this);
                return new(__MR_PreciseVertCoords_Get_pt(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PreciseVertCoords() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_DefaultConstruct();
            _UnderlyingPtr = __MR_PreciseVertCoords_DefaultConstruct();
        }

        /// Constructs `MR::PreciseVertCoords` elementwise.
        public unsafe Const_PreciseVertCoords(MR.VertId id, MR.Vector3i pt) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_ConstructFrom", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_ConstructFrom(MR.VertId id, MR.Vector3i pt);
            _UnderlyingPtr = __MR_PreciseVertCoords_ConstructFrom(id, pt);
        }

        /// Generated from constructor `MR::PreciseVertCoords::PreciseVertCoords`.
        public unsafe Const_PreciseVertCoords(MR.Const_PreciseVertCoords _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_ConstructFromAnother(MR.PreciseVertCoords._Underlying *_other);
            _UnderlyingPtr = __MR_PreciseVertCoords_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PreciseVertCoords`.
    /// This is the non-const half of the class.
    public class PreciseVertCoords : Const_PreciseVertCoords
    {
        internal unsafe PreciseVertCoords(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< unique id of the vertex (in both meshes)
        public new unsafe MR.Mut_VertId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_GetMutable_id", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PreciseVertCoords_GetMutable_id(_Underlying *_this);
                return new(__MR_PreciseVertCoords_GetMutable_id(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< integer coordinates of the vertex
        public new unsafe MR.Mut_Vector3i Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_GetMutable_pt", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_PreciseVertCoords_GetMutable_pt(_Underlying *_this);
                return new(__MR_PreciseVertCoords_GetMutable_pt(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PreciseVertCoords() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_DefaultConstruct();
            _UnderlyingPtr = __MR_PreciseVertCoords_DefaultConstruct();
        }

        /// Constructs `MR::PreciseVertCoords` elementwise.
        public unsafe PreciseVertCoords(MR.VertId id, MR.Vector3i pt) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_ConstructFrom", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_ConstructFrom(MR.VertId id, MR.Vector3i pt);
            _UnderlyingPtr = __MR_PreciseVertCoords_ConstructFrom(id, pt);
        }

        /// Generated from constructor `MR::PreciseVertCoords::PreciseVertCoords`.
        public unsafe PreciseVertCoords(MR.Const_PreciseVertCoords _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_ConstructFromAnother(MR.PreciseVertCoords._Underlying *_other);
            _UnderlyingPtr = __MR_PreciseVertCoords_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PreciseVertCoords::operator=`.
        public unsafe MR.PreciseVertCoords Assign(MR.Const_PreciseVertCoords _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoords._Underlying *__MR_PreciseVertCoords_AssignFromAnother(_Underlying *_this, MR.PreciseVertCoords._Underlying *_other);
            return new(__MR_PreciseVertCoords_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PreciseVertCoords` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PreciseVertCoords`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PreciseVertCoords`/`Const_PreciseVertCoords` directly.
    public class _InOptMut_PreciseVertCoords
    {
        public PreciseVertCoords? Opt;

        public _InOptMut_PreciseVertCoords() {}
        public _InOptMut_PreciseVertCoords(PreciseVertCoords value) {Opt = value;}
        public static implicit operator _InOptMut_PreciseVertCoords(PreciseVertCoords value) {return new(value);}
    }

    /// This is used for optional parameters of class `PreciseVertCoords` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PreciseVertCoords`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PreciseVertCoords`/`Const_PreciseVertCoords` to pass it to the function.
    public class _InOptConst_PreciseVertCoords
    {
        public Const_PreciseVertCoords? Opt;

        public _InOptConst_PreciseVertCoords() {}
        public _InOptConst_PreciseVertCoords(Const_PreciseVertCoords value) {Opt = value;}
        public static implicit operator _InOptConst_PreciseVertCoords(Const_PreciseVertCoords value) {return new(value);}
    }

    /// Generated from class `MR::TriangleSegmentIntersectResult`.
    /// This is the const half of the class.
    public class Const_TriangleSegmentIntersectResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TriangleSegmentIntersectResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_Destroy", ExactSpelling = true)]
            extern static void __MR_TriangleSegmentIntersectResult_Destroy(_Underlying *_this);
            __MR_TriangleSegmentIntersectResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TriangleSegmentIntersectResult() {Dispose(false);}

        ///< whether triangle and segment intersect
        public unsafe bool DoIntersect
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_Get_doIntersect", ExactSpelling = true)]
                extern static bool *__MR_TriangleSegmentIntersectResult_Get_doIntersect(_Underlying *_this);
                return *__MR_TriangleSegmentIntersectResult_Get_doIntersect(_UnderlyingPtr);
            }
        }

        ///< whether the plane with orientated triangle ABC has D point at the left
        public unsafe bool DIsLeftFromABC
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_Get_dIsLeftFromABC", ExactSpelling = true)]
                extern static bool *__MR_TriangleSegmentIntersectResult_Get_dIsLeftFromABC(_Underlying *_this);
                return *__MR_TriangleSegmentIntersectResult_Get_dIsLeftFromABC(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TriangleSegmentIntersectResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_DefaultConstruct();
            _UnderlyingPtr = __MR_TriangleSegmentIntersectResult_DefaultConstruct();
        }

        /// Constructs `MR::TriangleSegmentIntersectResult` elementwise.
        public unsafe Const_TriangleSegmentIntersectResult(bool doIntersect, bool dIsLeftFromABC) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_ConstructFrom(byte doIntersect, byte dIsLeftFromABC);
            _UnderlyingPtr = __MR_TriangleSegmentIntersectResult_ConstructFrom(doIntersect ? (byte)1 : (byte)0, dIsLeftFromABC ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::TriangleSegmentIntersectResult::TriangleSegmentIntersectResult`.
        public unsafe Const_TriangleSegmentIntersectResult(MR.Const_TriangleSegmentIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_ConstructFromAnother(MR.TriangleSegmentIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_TriangleSegmentIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::TriangleSegmentIntersectResult::operator bool`.
        public static unsafe explicit operator bool(MR.Const_TriangleSegmentIntersectResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_TriangleSegmentIntersectResult_ConvertTo_bool(MR.Const_TriangleSegmentIntersectResult._Underlying *_this);
            return __MR_TriangleSegmentIntersectResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::TriangleSegmentIntersectResult`.
    /// This is the non-const half of the class.
    public class TriangleSegmentIntersectResult : Const_TriangleSegmentIntersectResult
    {
        internal unsafe TriangleSegmentIntersectResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< whether triangle and segment intersect
        public new unsafe ref bool DoIntersect
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_GetMutable_doIntersect", ExactSpelling = true)]
                extern static bool *__MR_TriangleSegmentIntersectResult_GetMutable_doIntersect(_Underlying *_this);
                return ref *__MR_TriangleSegmentIntersectResult_GetMutable_doIntersect(_UnderlyingPtr);
            }
        }

        ///< whether the plane with orientated triangle ABC has D point at the left
        public new unsafe ref bool DIsLeftFromABC
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_GetMutable_dIsLeftFromABC", ExactSpelling = true)]
                extern static bool *__MR_TriangleSegmentIntersectResult_GetMutable_dIsLeftFromABC(_Underlying *_this);
                return ref *__MR_TriangleSegmentIntersectResult_GetMutable_dIsLeftFromABC(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TriangleSegmentIntersectResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_DefaultConstruct();
            _UnderlyingPtr = __MR_TriangleSegmentIntersectResult_DefaultConstruct();
        }

        /// Constructs `MR::TriangleSegmentIntersectResult` elementwise.
        public unsafe TriangleSegmentIntersectResult(bool doIntersect, bool dIsLeftFromABC) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_ConstructFrom(byte doIntersect, byte dIsLeftFromABC);
            _UnderlyingPtr = __MR_TriangleSegmentIntersectResult_ConstructFrom(doIntersect ? (byte)1 : (byte)0, dIsLeftFromABC ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::TriangleSegmentIntersectResult::TriangleSegmentIntersectResult`.
        public unsafe TriangleSegmentIntersectResult(MR.Const_TriangleSegmentIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_ConstructFromAnother(MR.TriangleSegmentIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_TriangleSegmentIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TriangleSegmentIntersectResult::operator=`.
        public unsafe MR.TriangleSegmentIntersectResult Assign(MR.Const_TriangleSegmentIntersectResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangleSegmentIntersectResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_TriangleSegmentIntersectResult_AssignFromAnother(_Underlying *_this, MR.TriangleSegmentIntersectResult._Underlying *_other);
            return new(__MR_TriangleSegmentIntersectResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TriangleSegmentIntersectResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TriangleSegmentIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriangleSegmentIntersectResult`/`Const_TriangleSegmentIntersectResult` directly.
    public class _InOptMut_TriangleSegmentIntersectResult
    {
        public TriangleSegmentIntersectResult? Opt;

        public _InOptMut_TriangleSegmentIntersectResult() {}
        public _InOptMut_TriangleSegmentIntersectResult(TriangleSegmentIntersectResult value) {Opt = value;}
        public static implicit operator _InOptMut_TriangleSegmentIntersectResult(TriangleSegmentIntersectResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `TriangleSegmentIntersectResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TriangleSegmentIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriangleSegmentIntersectResult`/`Const_TriangleSegmentIntersectResult` to pass it to the function.
    public class _InOptConst_TriangleSegmentIntersectResult
    {
        public Const_TriangleSegmentIntersectResult? Opt;

        public _InOptConst_TriangleSegmentIntersectResult() {}
        public _InOptConst_TriangleSegmentIntersectResult(Const_TriangleSegmentIntersectResult value) {Opt = value;}
        public static implicit operator _InOptConst_TriangleSegmentIntersectResult(Const_TriangleSegmentIntersectResult value) {return new(value);}
    }

    /// this struct contains coordinate converters float-int-float
    /// Generated from class `MR::CoordinateConverters`.
    /// This is the const half of the class.
    public class Const_CoordinateConverters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CoordinateConverters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_Destroy", ExactSpelling = true)]
            extern static void __MR_CoordinateConverters_Destroy(_Underlying *_this);
            __MR_CoordinateConverters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CoordinateConverters() {Dispose(false);}

        public unsafe MR.Std.Const_Function_MRVector3iFuncFromConstMRVector3fRef ToInt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_Get_toInt", ExactSpelling = true)]
                extern static MR.Std.Const_Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *__MR_CoordinateConverters_Get_toInt(_Underlying *_this);
                return new(__MR_CoordinateConverters_Get_toInt(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Function_MRVector3fFuncFromConstMRVector3iRef ToFloat
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_Get_toFloat", ExactSpelling = true)]
                extern static MR.Std.Const_Function_MRVector3fFuncFromConstMRVector3iRef._Underlying *__MR_CoordinateConverters_Get_toFloat(_Underlying *_this);
                return new(__MR_CoordinateConverters_Get_toFloat(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CoordinateConverters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_DefaultConstruct();
            _UnderlyingPtr = __MR_CoordinateConverters_DefaultConstruct();
        }

        /// Constructs `MR::CoordinateConverters` elementwise.
        public unsafe Const_CoordinateConverters(MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef toInt, MR.Std._ByValue_Function_MRVector3fFuncFromConstMRVector3iRef toFloat) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_ConstructFrom", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_ConstructFrom(MR.Misc._PassBy toInt_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *toInt, MR.Misc._PassBy toFloat_pass_by, MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef._Underlying *toFloat);
            _UnderlyingPtr = __MR_CoordinateConverters_ConstructFrom(toInt.PassByMode, toInt.Value is not null ? toInt.Value._UnderlyingPtr : null, toFloat.PassByMode, toFloat.Value is not null ? toFloat.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CoordinateConverters::CoordinateConverters`.
        public unsafe Const_CoordinateConverters(MR._ByValue_CoordinateConverters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CoordinateConverters._Underlying *_other);
            _UnderlyingPtr = __MR_CoordinateConverters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// this struct contains coordinate converters float-int-float
    /// Generated from class `MR::CoordinateConverters`.
    /// This is the non-const half of the class.
    public class CoordinateConverters : Const_CoordinateConverters
    {
        internal unsafe CoordinateConverters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef ToInt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_GetMutable_toInt", ExactSpelling = true)]
                extern static MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *__MR_CoordinateConverters_GetMutable_toInt(_Underlying *_this);
                return new(__MR_CoordinateConverters_GetMutable_toInt(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef ToFloat
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_GetMutable_toFloat", ExactSpelling = true)]
                extern static MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef._Underlying *__MR_CoordinateConverters_GetMutable_toFloat(_Underlying *_this);
                return new(__MR_CoordinateConverters_GetMutable_toFloat(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CoordinateConverters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_DefaultConstruct();
            _UnderlyingPtr = __MR_CoordinateConverters_DefaultConstruct();
        }

        /// Constructs `MR::CoordinateConverters` elementwise.
        public unsafe CoordinateConverters(MR.Std._ByValue_Function_MRVector3iFuncFromConstMRVector3fRef toInt, MR.Std._ByValue_Function_MRVector3fFuncFromConstMRVector3iRef toFloat) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_ConstructFrom", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_ConstructFrom(MR.Misc._PassBy toInt_pass_by, MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *toInt, MR.Misc._PassBy toFloat_pass_by, MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef._Underlying *toFloat);
            _UnderlyingPtr = __MR_CoordinateConverters_ConstructFrom(toInt.PassByMode, toInt.Value is not null ? toInt.Value._UnderlyingPtr : null, toFloat.PassByMode, toFloat.Value is not null ? toFloat.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CoordinateConverters::CoordinateConverters`.
        public unsafe CoordinateConverters(MR._ByValue_CoordinateConverters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CoordinateConverters._Underlying *_other);
            _UnderlyingPtr = __MR_CoordinateConverters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CoordinateConverters::operator=`.
        public unsafe MR.CoordinateConverters Assign(MR._ByValue_CoordinateConverters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CoordinateConverters._Underlying *__MR_CoordinateConverters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CoordinateConverters._Underlying *_other);
            return new(__MR_CoordinateConverters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `CoordinateConverters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CoordinateConverters`/`Const_CoordinateConverters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CoordinateConverters
    {
        internal readonly Const_CoordinateConverters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CoordinateConverters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_CoordinateConverters(Const_CoordinateConverters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CoordinateConverters(Const_CoordinateConverters arg) {return new(arg);}
        public _ByValue_CoordinateConverters(MR.Misc._Moved<CoordinateConverters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CoordinateConverters(MR.Misc._Moved<CoordinateConverters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CoordinateConverters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CoordinateConverters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CoordinateConverters`/`Const_CoordinateConverters` directly.
    public class _InOptMut_CoordinateConverters
    {
        public CoordinateConverters? Opt;

        public _InOptMut_CoordinateConverters() {}
        public _InOptMut_CoordinateConverters(CoordinateConverters value) {Opt = value;}
        public static implicit operator _InOptMut_CoordinateConverters(CoordinateConverters value) {return new(value);}
    }

    /// This is used for optional parameters of class `CoordinateConverters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CoordinateConverters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CoordinateConverters`/`Const_CoordinateConverters` to pass it to the function.
    public class _InOptConst_CoordinateConverters
    {
        public Const_CoordinateConverters? Opt;

        public _InOptConst_CoordinateConverters() {}
        public _InOptConst_CoordinateConverters(Const_CoordinateConverters value) {Opt = value;}
        public static implicit operator _InOptConst_CoordinateConverters(Const_CoordinateConverters value) {return new(value);}
    }

    /// returns true if the plane with orientated triangle ABC has 0 point at the left;
    /// uses simulation-of-simplicity to avoid "0 is exactly on plane"
    /// Generated from function `MR::orient3d`.
    public static unsafe bool Orient3d(MR.Const_Vector3i a, MR.Const_Vector3i b, MR.Const_Vector3i c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orient3d_3", ExactSpelling = true)]
        extern static byte __MR_orient3d_3(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b, MR.Const_Vector3i._Underlying *c);
        return __MR_orient3d_3(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr) != 0;
    }

    /// returns true if the plane with orientated triangle ABC has D point at the left;
    /// uses simulation-of-simplicity to avoid "D is exactly on plane"
    /// Generated from function `MR::orient3d`.
    public static unsafe bool Orient3d(MR.Const_Vector3i a, MR.Const_Vector3i b, MR.Const_Vector3i c, MR.Const_Vector3i d)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orient3d_4", ExactSpelling = true)]
        extern static byte __MR_orient3d_4(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b, MR.Const_Vector3i._Underlying *c, MR.Const_Vector3i._Underlying *d);
        return __MR_orient3d_4(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr) != 0;
    }

    /// first sorts the indices in ascending order, then calls the predicate for sorted points
    /// Generated from function `MR::orient3d`.
    public static unsafe bool Orient3d(MR.Std.Const_Array_MRPreciseVertCoords_4 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orient3d_1_std_array_MR_PreciseVertCoords_4", ExactSpelling = true)]
        extern static byte __MR_orient3d_1_std_array_MR_PreciseVertCoords_4(MR.Std.Const_Array_MRPreciseVertCoords_4._Underlying *vs);
        return __MR_orient3d_1_std_array_MR_PreciseVertCoords_4(vs._UnderlyingPtr) != 0;
    }

    /// Generated from function `MR::orient3d`.
    public static unsafe bool Orient3d(MR.Const_PreciseVertCoords? vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orient3d_1_const_MR_PreciseVertCoords_ptr", ExactSpelling = true)]
        extern static byte __MR_orient3d_1_const_MR_PreciseVertCoords_ptr(MR.Const_PreciseVertCoords._Underlying *vs);
        return __MR_orient3d_1_const_MR_PreciseVertCoords_ptr(vs is not null ? vs._UnderlyingPtr : null) != 0;
    }

    /// checks whether triangle ABC (indices 012) and segment DE (indices 34) intersect
    /// uses simulation-of-simplicity to avoid edge-segment intersections and co-planarity
    /// Generated from function `MR::doTriangleSegmentIntersect`.
    public static unsafe MR.TriangleSegmentIntersectResult DoTriangleSegmentIntersect(MR.Std.Const_Array_MRPreciseVertCoords_5 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_doTriangleSegmentIntersect", ExactSpelling = true)]
        extern static MR.TriangleSegmentIntersectResult._Underlying *__MR_doTriangleSegmentIntersect(MR.Std.Const_Array_MRPreciseVertCoords_5._Underlying *vs);
        return new(__MR_doTriangleSegmentIntersect(vs._UnderlyingPtr), is_owning: true);
    }

    /// given line segment s=01 and two triangles ta=234, tb=567 known to intersect it, finds the order of intersection using precise predicates:
    /// true:  s[0], s ^ ta, s ^ tb, s[1]
    /// false: s[0], s ^ tb, s ^ ta, s[1]
    /// segments ta and tb can have at most two shared points, all other points must be unique
    /// Generated from function `MR::segmentIntersectionOrder`.
    public static unsafe bool SegmentIntersectionOrder(MR.Std.Const_Array_MRPreciseVertCoords_8 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_segmentIntersectionOrder_std_array_MR_PreciseVertCoords_8", ExactSpelling = true)]
        extern static byte __MR_segmentIntersectionOrder_std_array_MR_PreciseVertCoords_8(MR.Std.Const_Array_MRPreciseVertCoords_8._Underlying *vs);
        return __MR_segmentIntersectionOrder_std_array_MR_PreciseVertCoords_8(vs._UnderlyingPtr) != 0;
    }

    /// creates converter from Vector3f to Vector3i in Box range (int diapason is mapped to box range)
    /// Generated from function `MR::getToIntConverter`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef> GetToIntConverter(MR.Const_Box3d box)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getToIntConverter", ExactSpelling = true)]
        extern static MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef._Underlying *__MR_getToIntConverter(MR.Const_Box3d._Underlying *box);
        return MR.Misc.Move(new MR.Std.Function_MRVector3iFuncFromConstMRVector3fRef(__MR_getToIntConverter(box._UnderlyingPtr), is_owning: true));
    }

    /// creates converter from Vector3i to Vector3f in Box range (int diapason is mapped to box range)
    /// Generated from function `MR::getToFloatConverter`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef> GetToFloatConverter(MR.Const_Box3d box)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getToFloatConverter", ExactSpelling = true)]
        extern static MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef._Underlying *__MR_getToFloatConverter(MR.Const_Box3d._Underlying *box);
        return MR.Misc.Move(new MR.Std.Function_MRVector3fFuncFromConstMRVector3iRef(__MR_getToFloatConverter(box._UnderlyingPtr), is_owning: true));
    }

    /// given two line segments AB and CD located in one plane,
    /// finds whether they intersect and if yes, computes their common point using integer-only arithmetic
    /// Generated from function `MR::findTwoSegmentsIntersection`.
    public static unsafe MR.Std.Optional_MRVector3i FindTwoSegmentsIntersection(MR.Const_Vector3i ai, MR.Const_Vector3i bi, MR.Const_Vector3i ci, MR.Const_Vector3i di)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwoSegmentsIntersection", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVector3i._Underlying *__MR_findTwoSegmentsIntersection(MR.Const_Vector3i._Underlying *ai, MR.Const_Vector3i._Underlying *bi, MR.Const_Vector3i._Underlying *ci, MR.Const_Vector3i._Underlying *di);
        return new(__MR_findTwoSegmentsIntersection(ai._UnderlyingPtr, bi._UnderlyingPtr, ci._UnderlyingPtr, di._UnderlyingPtr), is_owning: true);
    }

    /// finds intersection precise, using high precision int inside
    /// this function input should have intersection
    /// Generated from function `MR::findTriangleSegmentIntersectionPrecise`.
    public static unsafe MR.Vector3f FindTriangleSegmentIntersectionPrecise(MR.Const_Vector3f a, MR.Const_Vector3f b, MR.Const_Vector3f c, MR.Const_Vector3f d, MR.Const_Vector3f e, MR._ByValue_CoordinateConverters converters)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTriangleSegmentIntersectionPrecise", ExactSpelling = true)]
        extern static MR.Vector3f __MR_findTriangleSegmentIntersectionPrecise(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, MR.Const_Vector3f._Underlying *c, MR.Const_Vector3f._Underlying *d, MR.Const_Vector3f._Underlying *e, MR.Misc._PassBy converters_pass_by, MR.CoordinateConverters._Underlying *converters);
        return __MR_findTriangleSegmentIntersectionPrecise(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr, e._UnderlyingPtr, converters.PassByMode, converters.Value is not null ? converters.Value._UnderlyingPtr : null);
    }
}
