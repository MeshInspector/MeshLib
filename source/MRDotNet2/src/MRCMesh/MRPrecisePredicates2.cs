public static partial class MR
{
    /// Generated from class `MR::PreciseVertCoord`.
    /// This is the const half of the class.
    public class Const_PreciseVertCoord : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PreciseVertCoord(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_Destroy", ExactSpelling = true)]
            extern static void __MR_PreciseVertCoord_Destroy(_Underlying *_this);
            __MR_PreciseVertCoord_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PreciseVertCoord() {Dispose(false);}

        ///< unique id of the vertex (in both contours)
        public unsafe MR.Const_VertId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_Get_id", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PreciseVertCoord_Get_id(_Underlying *_this);
                return new(__MR_PreciseVertCoord_Get_id(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< coordinate
        public unsafe int Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_Get_pt", ExactSpelling = true)]
                extern static int *__MR_PreciseVertCoord_Get_pt(_Underlying *_this);
                return *__MR_PreciseVertCoord_Get_pt(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PreciseVertCoord() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_DefaultConstruct();
            _UnderlyingPtr = __MR_PreciseVertCoord_DefaultConstruct();
        }

        /// Constructs `MR::PreciseVertCoord` elementwise.
        public unsafe Const_PreciseVertCoord(MR.VertId id, int pt) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_ConstructFrom", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_ConstructFrom(MR.VertId id, int pt);
            _UnderlyingPtr = __MR_PreciseVertCoord_ConstructFrom(id, pt);
        }

        /// Generated from constructor `MR::PreciseVertCoord::PreciseVertCoord`.
        public unsafe Const_PreciseVertCoord(MR.Const_PreciseVertCoord _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_ConstructFromAnother(MR.PreciseVertCoord._Underlying *_other);
            _UnderlyingPtr = __MR_PreciseVertCoord_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PreciseVertCoord`.
    /// This is the non-const half of the class.
    public class PreciseVertCoord : Const_PreciseVertCoord
    {
        internal unsafe PreciseVertCoord(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< unique id of the vertex (in both contours)
        public new unsafe MR.Mut_VertId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_GetMutable_id", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PreciseVertCoord_GetMutable_id(_Underlying *_this);
                return new(__MR_PreciseVertCoord_GetMutable_id(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< coordinate
        public new unsafe ref int Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_GetMutable_pt", ExactSpelling = true)]
                extern static int *__MR_PreciseVertCoord_GetMutable_pt(_Underlying *_this);
                return ref *__MR_PreciseVertCoord_GetMutable_pt(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PreciseVertCoord() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_DefaultConstruct();
            _UnderlyingPtr = __MR_PreciseVertCoord_DefaultConstruct();
        }

        /// Constructs `MR::PreciseVertCoord` elementwise.
        public unsafe PreciseVertCoord(MR.VertId id, int pt) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_ConstructFrom", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_ConstructFrom(MR.VertId id, int pt);
            _UnderlyingPtr = __MR_PreciseVertCoord_ConstructFrom(id, pt);
        }

        /// Generated from constructor `MR::PreciseVertCoord::PreciseVertCoord`.
        public unsafe PreciseVertCoord(MR.Const_PreciseVertCoord _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_ConstructFromAnother(MR.PreciseVertCoord._Underlying *_other);
            _UnderlyingPtr = __MR_PreciseVertCoord_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PreciseVertCoord::operator=`.
        public unsafe MR.PreciseVertCoord Assign(MR.Const_PreciseVertCoord _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoord_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoord._Underlying *__MR_PreciseVertCoord_AssignFromAnother(_Underlying *_this, MR.PreciseVertCoord._Underlying *_other);
            return new(__MR_PreciseVertCoord_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PreciseVertCoord` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PreciseVertCoord`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PreciseVertCoord`/`Const_PreciseVertCoord` directly.
    public class _InOptMut_PreciseVertCoord
    {
        public PreciseVertCoord? Opt;

        public _InOptMut_PreciseVertCoord() {}
        public _InOptMut_PreciseVertCoord(PreciseVertCoord value) {Opt = value;}
        public static implicit operator _InOptMut_PreciseVertCoord(PreciseVertCoord value) {return new(value);}
    }

    /// This is used for optional parameters of class `PreciseVertCoord` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PreciseVertCoord`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PreciseVertCoord`/`Const_PreciseVertCoord` to pass it to the function.
    public class _InOptConst_PreciseVertCoord
    {
        public Const_PreciseVertCoord? Opt;

        public _InOptConst_PreciseVertCoord() {}
        public _InOptConst_PreciseVertCoord(Const_PreciseVertCoord value) {Opt = value;}
        public static implicit operator _InOptConst_PreciseVertCoord(Const_PreciseVertCoord value) {return new(value);}
    }

    /// Generated from class `MR::PreciseVertCoords2`.
    /// This is the const half of the class.
    public class Const_PreciseVertCoords2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PreciseVertCoords2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_Destroy", ExactSpelling = true)]
            extern static void __MR_PreciseVertCoords2_Destroy(_Underlying *_this);
            __MR_PreciseVertCoords2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PreciseVertCoords2() {Dispose(false);}

        ///< unique id of the vertex (in both contours)
        public unsafe MR.Const_VertId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_Get_id", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PreciseVertCoords2_Get_id(_Underlying *_this);
                return new(__MR_PreciseVertCoords2_Get_id(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< integer coordinates of the vertex
        public unsafe MR.Const_Vector2i Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_Get_pt", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_PreciseVertCoords2_Get_pt(_Underlying *_this);
                return new(__MR_PreciseVertCoords2_Get_pt(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PreciseVertCoords2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_DefaultConstruct();
            _UnderlyingPtr = __MR_PreciseVertCoords2_DefaultConstruct();
        }

        /// Constructs `MR::PreciseVertCoords2` elementwise.
        public unsafe Const_PreciseVertCoords2(MR.VertId id, MR.Vector2i pt) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_ConstructFrom", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_ConstructFrom(MR.VertId id, MR.Vector2i pt);
            _UnderlyingPtr = __MR_PreciseVertCoords2_ConstructFrom(id, pt);
        }

        /// Generated from constructor `MR::PreciseVertCoords2::PreciseVertCoords2`.
        public unsafe Const_PreciseVertCoords2(MR.Const_PreciseVertCoords2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_ConstructFromAnother(MR.PreciseVertCoords2._Underlying *_other);
            _UnderlyingPtr = __MR_PreciseVertCoords2_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PreciseVertCoords2`.
    /// This is the non-const half of the class.
    public class PreciseVertCoords2 : Const_PreciseVertCoords2
    {
        internal unsafe PreciseVertCoords2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< unique id of the vertex (in both contours)
        public new unsafe MR.Mut_VertId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_GetMutable_id", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PreciseVertCoords2_GetMutable_id(_Underlying *_this);
                return new(__MR_PreciseVertCoords2_GetMutable_id(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< integer coordinates of the vertex
        public new unsafe MR.Mut_Vector2i Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_GetMutable_pt", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_PreciseVertCoords2_GetMutable_pt(_Underlying *_this);
                return new(__MR_PreciseVertCoords2_GetMutable_pt(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PreciseVertCoords2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_DefaultConstruct();
            _UnderlyingPtr = __MR_PreciseVertCoords2_DefaultConstruct();
        }

        /// Constructs `MR::PreciseVertCoords2` elementwise.
        public unsafe PreciseVertCoords2(MR.VertId id, MR.Vector2i pt) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_ConstructFrom", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_ConstructFrom(MR.VertId id, MR.Vector2i pt);
            _UnderlyingPtr = __MR_PreciseVertCoords2_ConstructFrom(id, pt);
        }

        /// Generated from constructor `MR::PreciseVertCoords2::PreciseVertCoords2`.
        public unsafe PreciseVertCoords2(MR.Const_PreciseVertCoords2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_ConstructFromAnother(MR.PreciseVertCoords2._Underlying *_other);
            _UnderlyingPtr = __MR_PreciseVertCoords2_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PreciseVertCoords2::operator=`.
        public unsafe MR.PreciseVertCoords2 Assign(MR.Const_PreciseVertCoords2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PreciseVertCoords2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PreciseVertCoords2._Underlying *__MR_PreciseVertCoords2_AssignFromAnother(_Underlying *_this, MR.PreciseVertCoords2._Underlying *_other);
            return new(__MR_PreciseVertCoords2_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PreciseVertCoords2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PreciseVertCoords2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PreciseVertCoords2`/`Const_PreciseVertCoords2` directly.
    public class _InOptMut_PreciseVertCoords2
    {
        public PreciseVertCoords2? Opt;

        public _InOptMut_PreciseVertCoords2() {}
        public _InOptMut_PreciseVertCoords2(PreciseVertCoords2 value) {Opt = value;}
        public static implicit operator _InOptMut_PreciseVertCoords2(PreciseVertCoords2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `PreciseVertCoords2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PreciseVertCoords2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PreciseVertCoords2`/`Const_PreciseVertCoords2` to pass it to the function.
    public class _InOptConst_PreciseVertCoords2
    {
        public Const_PreciseVertCoords2? Opt;

        public _InOptConst_PreciseVertCoords2() {}
        public _InOptConst_PreciseVertCoords2(Const_PreciseVertCoords2 value) {Opt = value;}
        public static implicit operator _InOptConst_PreciseVertCoords2(Const_PreciseVertCoords2 value) {return new(value);}
    }

    /// Generated from class `MR::SegmentSegmentIntersectResult`.
    /// This is the const half of the class.
    public class Const_SegmentSegmentIntersectResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SegmentSegmentIntersectResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_Destroy", ExactSpelling = true)]
            extern static void __MR_SegmentSegmentIntersectResult_Destroy(_Underlying *_this);
            __MR_SegmentSegmentIntersectResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SegmentSegmentIntersectResult() {Dispose(false);}

        ///< whether the segments intersect
        public unsafe bool DoIntersect
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_Get_doIntersect", ExactSpelling = true)]
                extern static bool *__MR_SegmentSegmentIntersectResult_Get_doIntersect(_Underlying *_this);
                return *__MR_SegmentSegmentIntersectResult_Get_doIntersect(_UnderlyingPtr);
            }
        }

        ///< whether the directed line AB has C point at the left
        public unsafe bool CIsLeftFromAB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_Get_cIsLeftFromAB", ExactSpelling = true)]
                extern static bool *__MR_SegmentSegmentIntersectResult_Get_cIsLeftFromAB(_Underlying *_this);
                return *__MR_SegmentSegmentIntersectResult_Get_cIsLeftFromAB(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SegmentSegmentIntersectResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_DefaultConstruct();
            _UnderlyingPtr = __MR_SegmentSegmentIntersectResult_DefaultConstruct();
        }

        /// Constructs `MR::SegmentSegmentIntersectResult` elementwise.
        public unsafe Const_SegmentSegmentIntersectResult(bool doIntersect, bool cIsLeftFromAB) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_ConstructFrom(byte doIntersect, byte cIsLeftFromAB);
            _UnderlyingPtr = __MR_SegmentSegmentIntersectResult_ConstructFrom(doIntersect ? (byte)1 : (byte)0, cIsLeftFromAB ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::SegmentSegmentIntersectResult::SegmentSegmentIntersectResult`.
        public unsafe Const_SegmentSegmentIntersectResult(MR.Const_SegmentSegmentIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_ConstructFromAnother(MR.SegmentSegmentIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_SegmentSegmentIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::SegmentSegmentIntersectResult::operator bool`.
        public static unsafe explicit operator bool(MR.Const_SegmentSegmentIntersectResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_SegmentSegmentIntersectResult_ConvertTo_bool(MR.Const_SegmentSegmentIntersectResult._Underlying *_this);
            return __MR_SegmentSegmentIntersectResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::SegmentSegmentIntersectResult`.
    /// This is the non-const half of the class.
    public class SegmentSegmentIntersectResult : Const_SegmentSegmentIntersectResult
    {
        internal unsafe SegmentSegmentIntersectResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< whether the segments intersect
        public new unsafe ref bool DoIntersect
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_GetMutable_doIntersect", ExactSpelling = true)]
                extern static bool *__MR_SegmentSegmentIntersectResult_GetMutable_doIntersect(_Underlying *_this);
                return ref *__MR_SegmentSegmentIntersectResult_GetMutable_doIntersect(_UnderlyingPtr);
            }
        }

        ///< whether the directed line AB has C point at the left
        public new unsafe ref bool CIsLeftFromAB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_GetMutable_cIsLeftFromAB", ExactSpelling = true)]
                extern static bool *__MR_SegmentSegmentIntersectResult_GetMutable_cIsLeftFromAB(_Underlying *_this);
                return ref *__MR_SegmentSegmentIntersectResult_GetMutable_cIsLeftFromAB(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SegmentSegmentIntersectResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_DefaultConstruct();
            _UnderlyingPtr = __MR_SegmentSegmentIntersectResult_DefaultConstruct();
        }

        /// Constructs `MR::SegmentSegmentIntersectResult` elementwise.
        public unsafe SegmentSegmentIntersectResult(bool doIntersect, bool cIsLeftFromAB) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_ConstructFrom(byte doIntersect, byte cIsLeftFromAB);
            _UnderlyingPtr = __MR_SegmentSegmentIntersectResult_ConstructFrom(doIntersect ? (byte)1 : (byte)0, cIsLeftFromAB ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::SegmentSegmentIntersectResult::SegmentSegmentIntersectResult`.
        public unsafe SegmentSegmentIntersectResult(MR.Const_SegmentSegmentIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_ConstructFromAnother(MR.SegmentSegmentIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_SegmentSegmentIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SegmentSegmentIntersectResult::operator=`.
        public unsafe MR.SegmentSegmentIntersectResult Assign(MR.Const_SegmentSegmentIntersectResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmentSegmentIntersectResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_SegmentSegmentIntersectResult_AssignFromAnother(_Underlying *_this, MR.SegmentSegmentIntersectResult._Underlying *_other);
            return new(__MR_SegmentSegmentIntersectResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SegmentSegmentIntersectResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SegmentSegmentIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SegmentSegmentIntersectResult`/`Const_SegmentSegmentIntersectResult` directly.
    public class _InOptMut_SegmentSegmentIntersectResult
    {
        public SegmentSegmentIntersectResult? Opt;

        public _InOptMut_SegmentSegmentIntersectResult() {}
        public _InOptMut_SegmentSegmentIntersectResult(SegmentSegmentIntersectResult value) {Opt = value;}
        public static implicit operator _InOptMut_SegmentSegmentIntersectResult(SegmentSegmentIntersectResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `SegmentSegmentIntersectResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SegmentSegmentIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SegmentSegmentIntersectResult`/`Const_SegmentSegmentIntersectResult` to pass it to the function.
    public class _InOptConst_SegmentSegmentIntersectResult
    {
        public Const_SegmentSegmentIntersectResult? Opt;

        public _InOptConst_SegmentSegmentIntersectResult() {}
        public _InOptConst_SegmentSegmentIntersectResult(Const_SegmentSegmentIntersectResult value) {Opt = value;}
        public static implicit operator _InOptConst_SegmentSegmentIntersectResult(Const_SegmentSegmentIntersectResult value) {return new(value);}
    }

    /// this struct contains coordinate converters float-int-float
    /// Generated from class `MR::CoordinateConverters2`.
    /// This is the const half of the class.
    public class Const_CoordinateConverters2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CoordinateConverters2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_Destroy", ExactSpelling = true)]
            extern static void __MR_CoordinateConverters2_Destroy(_Underlying *_this);
            __MR_CoordinateConverters2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CoordinateConverters2() {Dispose(false);}

        public unsafe MR.Std.Const_Function_MRVector2iFuncFromConstMRVector2fRef ToInt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_Get_toInt", ExactSpelling = true)]
                extern static MR.Std.Const_Function_MRVector2iFuncFromConstMRVector2fRef._Underlying *__MR_CoordinateConverters2_Get_toInt(_Underlying *_this);
                return new(__MR_CoordinateConverters2_Get_toInt(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Function_MRVector2fFuncFromConstMRVector2iRef ToFloat
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_Get_toFloat", ExactSpelling = true)]
                extern static MR.Std.Const_Function_MRVector2fFuncFromConstMRVector2iRef._Underlying *__MR_CoordinateConverters2_Get_toFloat(_Underlying *_this);
                return new(__MR_CoordinateConverters2_Get_toFloat(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CoordinateConverters2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_DefaultConstruct();
            _UnderlyingPtr = __MR_CoordinateConverters2_DefaultConstruct();
        }

        /// Constructs `MR::CoordinateConverters2` elementwise.
        public unsafe Const_CoordinateConverters2(MR.Std._ByValue_Function_MRVector2iFuncFromConstMRVector2fRef toInt, MR.Std._ByValue_Function_MRVector2fFuncFromConstMRVector2iRef toFloat) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_ConstructFrom", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_ConstructFrom(MR.Misc._PassBy toInt_pass_by, MR.Std.Function_MRVector2iFuncFromConstMRVector2fRef._Underlying *toInt, MR.Misc._PassBy toFloat_pass_by, MR.Std.Function_MRVector2fFuncFromConstMRVector2iRef._Underlying *toFloat);
            _UnderlyingPtr = __MR_CoordinateConverters2_ConstructFrom(toInt.PassByMode, toInt.Value is not null ? toInt.Value._UnderlyingPtr : null, toFloat.PassByMode, toFloat.Value is not null ? toFloat.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CoordinateConverters2::CoordinateConverters2`.
        public unsafe Const_CoordinateConverters2(MR._ByValue_CoordinateConverters2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CoordinateConverters2._Underlying *_other);
            _UnderlyingPtr = __MR_CoordinateConverters2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// this struct contains coordinate converters float-int-float
    /// Generated from class `MR::CoordinateConverters2`.
    /// This is the non-const half of the class.
    public class CoordinateConverters2 : Const_CoordinateConverters2
    {
        internal unsafe CoordinateConverters2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_MRVector2iFuncFromConstMRVector2fRef ToInt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_GetMutable_toInt", ExactSpelling = true)]
                extern static MR.Std.Function_MRVector2iFuncFromConstMRVector2fRef._Underlying *__MR_CoordinateConverters2_GetMutable_toInt(_Underlying *_this);
                return new(__MR_CoordinateConverters2_GetMutable_toInt(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Function_MRVector2fFuncFromConstMRVector2iRef ToFloat
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_GetMutable_toFloat", ExactSpelling = true)]
                extern static MR.Std.Function_MRVector2fFuncFromConstMRVector2iRef._Underlying *__MR_CoordinateConverters2_GetMutable_toFloat(_Underlying *_this);
                return new(__MR_CoordinateConverters2_GetMutable_toFloat(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CoordinateConverters2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_DefaultConstruct();
            _UnderlyingPtr = __MR_CoordinateConverters2_DefaultConstruct();
        }

        /// Constructs `MR::CoordinateConverters2` elementwise.
        public unsafe CoordinateConverters2(MR.Std._ByValue_Function_MRVector2iFuncFromConstMRVector2fRef toInt, MR.Std._ByValue_Function_MRVector2fFuncFromConstMRVector2iRef toFloat) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_ConstructFrom", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_ConstructFrom(MR.Misc._PassBy toInt_pass_by, MR.Std.Function_MRVector2iFuncFromConstMRVector2fRef._Underlying *toInt, MR.Misc._PassBy toFloat_pass_by, MR.Std.Function_MRVector2fFuncFromConstMRVector2iRef._Underlying *toFloat);
            _UnderlyingPtr = __MR_CoordinateConverters2_ConstructFrom(toInt.PassByMode, toInt.Value is not null ? toInt.Value._UnderlyingPtr : null, toFloat.PassByMode, toFloat.Value is not null ? toFloat.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CoordinateConverters2::CoordinateConverters2`.
        public unsafe CoordinateConverters2(MR._ByValue_CoordinateConverters2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CoordinateConverters2._Underlying *_other);
            _UnderlyingPtr = __MR_CoordinateConverters2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CoordinateConverters2::operator=`.
        public unsafe MR.CoordinateConverters2 Assign(MR._ByValue_CoordinateConverters2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CoordinateConverters2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CoordinateConverters2._Underlying *__MR_CoordinateConverters2_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CoordinateConverters2._Underlying *_other);
            return new(__MR_CoordinateConverters2_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `CoordinateConverters2` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CoordinateConverters2`/`Const_CoordinateConverters2` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CoordinateConverters2
    {
        internal readonly Const_CoordinateConverters2? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CoordinateConverters2() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_CoordinateConverters2(Const_CoordinateConverters2 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CoordinateConverters2(Const_CoordinateConverters2 arg) {return new(arg);}
        public _ByValue_CoordinateConverters2(MR.Misc._Moved<CoordinateConverters2> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CoordinateConverters2(MR.Misc._Moved<CoordinateConverters2> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CoordinateConverters2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CoordinateConverters2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CoordinateConverters2`/`Const_CoordinateConverters2` directly.
    public class _InOptMut_CoordinateConverters2
    {
        public CoordinateConverters2? Opt;

        public _InOptMut_CoordinateConverters2() {}
        public _InOptMut_CoordinateConverters2(CoordinateConverters2 value) {Opt = value;}
        public static implicit operator _InOptMut_CoordinateConverters2(CoordinateConverters2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `CoordinateConverters2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CoordinateConverters2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CoordinateConverters2`/`Const_CoordinateConverters2` to pass it to the function.
    public class _InOptConst_CoordinateConverters2
    {
        public Const_CoordinateConverters2? Opt;

        public _InOptConst_CoordinateConverters2() {}
        public _InOptConst_CoordinateConverters2(Const_CoordinateConverters2 value) {Opt = value;}
        public static implicit operator _InOptConst_CoordinateConverters2(Const_CoordinateConverters2 value) {return new(value);}
    }

    /// return true if l is smaller than r,
    /// uses simulation-of-simplicity (assuming larger perturbations of points with smaller id) to avoid "coordinates are the same"
    /// Generated from function `MR::smaller`.
    public static unsafe bool Smaller(MR.Const_PreciseVertCoord l, MR.Const_PreciseVertCoord r)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_smaller", ExactSpelling = true)]
        extern static byte __MR_smaller(MR.Const_PreciseVertCoord._Underlying *l, MR.Const_PreciseVertCoord._Underlying *r);
        return __MR_smaller(l._UnderlyingPtr, r._UnderlyingPtr) != 0;
    }

    /// return true if the smallest rotation from vector (a) to vector (b) is in counter-clock-wise direction;
    /// uses simulation-of-simplicity (assuming perturbations a >> b) to avoid "vectors are collinear"
    /// Generated from function `MR::ccw`.
    public static unsafe bool Ccw(MR.Const_Vector2i a, MR.Const_Vector2i b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ccw_2", ExactSpelling = true)]
        extern static byte __MR_ccw_2(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
        return __MR_ccw_2(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
    }

    /// return true if the smallest rotation from vector (a-c) to vector (b-c) is in counter-clock-wise direction;
    /// uses simulation-of-simplicity (assuming perturbations a >> b >> c) to avoid "vectors are collinear"
    /// Generated from function `MR::ccw`.
    public static unsafe bool Ccw(MR.Const_Vector2i a, MR.Const_Vector2i b, MR.Const_Vector2i c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ccw_3", ExactSpelling = true)]
        extern static byte __MR_ccw_3(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b, MR.Const_Vector2i._Underlying *c);
        return __MR_ccw_3(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr) != 0;
    }

    /// return true if the smallest rotation from vector (vs[0]-vs[2]) to vector (vs[1]-vs[2]) is in counter-clock-wise direction;
    /// uses simulation-of-simplicity (assuming larger perturbations of points with smaller id) to avoid "vectors are collinear"
    /// Generated from function `MR::ccw`.
    public static unsafe bool Ccw(MR.Std.Const_Array_MRPreciseVertCoords2_3 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ccw_1_std_array_MR_PreciseVertCoords2_3", ExactSpelling = true)]
        extern static byte __MR_ccw_1_std_array_MR_PreciseVertCoords2_3(MR.Std.Const_Array_MRPreciseVertCoords2_3._Underlying *vs);
        return __MR_ccw_1_std_array_MR_PreciseVertCoords2_3(vs._UnderlyingPtr) != 0;
    }

    /// Generated from function `MR::ccw`.
    public static unsafe bool Ccw(MR.Const_PreciseVertCoords2? vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ccw_1_const_MR_PreciseVertCoords2_ptr", ExactSpelling = true)]
        extern static byte __MR_ccw_1_const_MR_PreciseVertCoords2_ptr(MR.Const_PreciseVertCoords2._Underlying *vs);
        return __MR_ccw_1_const_MR_PreciseVertCoords2_ptr(vs is not null ? vs._UnderlyingPtr : null) != 0;
    }

    /// given the line passing via points vs[0] and vs[1], which defines linear signed scalar distance field:
    /// zero on the line, positive for x where ccw(vs[0], vs[1], x) == true, and negative for x where ccw(vs[0], vs[1], x) == false
    /// finds whether sdistance(vs[2]) < sdistance(vs[3]);
    /// avoids equality of signed distances using simulation-of-simplicity approach (assuming larger perturbations of points with smaller id)
    /// Generated from function `MR::smaller2`.
    public static unsafe bool Smaller2(MR.Std.Const_Array_MRPreciseVertCoords2_4 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_smaller2", ExactSpelling = true)]
        extern static byte __MR_smaller2(MR.Std.Const_Array_MRPreciseVertCoords2_4._Underlying *vs);
        return __MR_smaller2(vs._UnderlyingPtr) != 0;
    }

    /// considers 3D points obtained from 2D inputs by moving each point on paraboloid: z = x*x+y*y;
    /// returns true if the plane with orientated triangle ABC has D point at the left;
    /// uses simulation-of-simplicity to avoid "D is exactly on plane"
    /// Generated from function `MR::orientParaboloid3d`.
    public static unsafe bool OrientParaboloid3d(MR.Const_Vector2i a, MR.Const_Vector2i b, MR.Const_Vector2i c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientParaboloid3d_3", ExactSpelling = true)]
        extern static byte __MR_orientParaboloid3d_3(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b, MR.Const_Vector2i._Underlying *c);
        return __MR_orientParaboloid3d_3(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr) != 0;
    }

    /// Generated from function `MR::orientParaboloid3d`.
    public static unsafe bool OrientParaboloid3d(MR.Const_Vector2i a, MR.Const_Vector2i b, MR.Const_Vector2i c, MR.Const_Vector2i d)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orientParaboloid3d_4", ExactSpelling = true)]
        extern static byte __MR_orientParaboloid3d_4(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b, MR.Const_Vector2i._Underlying *c, MR.Const_Vector2i._Underlying *d);
        return __MR_orientParaboloid3d_4(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr) != 0;
    }

    /// return true if 4th point in array lays inside circumcircle of first 3 points based triangle
    /// Generated from function `MR::inCircle`.
    public static unsafe bool InCircle(MR.Std.Const_Array_MRPreciseVertCoords2_4 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_inCircle_std_array_MR_PreciseVertCoords2_4", ExactSpelling = true)]
        extern static byte __MR_inCircle_std_array_MR_PreciseVertCoords2_4(MR.Std.Const_Array_MRPreciseVertCoords2_4._Underlying *vs);
        return __MR_inCircle_std_array_MR_PreciseVertCoords2_4(vs._UnderlyingPtr) != 0;
    }

    /// Generated from function `MR::inCircle`.
    public static unsafe bool InCircle(MR.Const_PreciseVertCoords2? vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_inCircle_const_MR_PreciseVertCoords2_ptr", ExactSpelling = true)]
        extern static byte __MR_inCircle_const_MR_PreciseVertCoords2_ptr(MR.Const_PreciseVertCoords2._Underlying *vs);
        return __MR_inCircle_const_MR_PreciseVertCoords2_ptr(vs is not null ? vs._UnderlyingPtr : null) != 0;
    }

    /// checks whether the segments AB (indices 01) and segments CD (indices 23) intersect;
    /// uses simulation-of-simplicity to avoid edge-segment intersections and co-planarity
    /// Generated from function `MR::doSegmentSegmentIntersect`.
    public static unsafe MR.SegmentSegmentIntersectResult DoSegmentSegmentIntersect(MR.Std.Const_Array_MRPreciseVertCoords2_4 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_doSegmentSegmentIntersect", ExactSpelling = true)]
        extern static MR.SegmentSegmentIntersectResult._Underlying *__MR_doSegmentSegmentIntersect(MR.Std.Const_Array_MRPreciseVertCoords2_4._Underlying *vs);
        return new(__MR_doSegmentSegmentIntersect(vs._UnderlyingPtr), is_owning: true);
    }

    /// given line segment s=01 and two other segments sa=23, sb=45 known to intersect it, finds the order of intersection using precise predicates:
    /// true:  s[0], s ^ sa, s ^ sb, s[1]
    /// false: s[0], s ^ sb, s ^ sa, s[1]
    /// segments sa and sb can have at most one shared point, all other points must be unique
    /// Generated from function `MR::segmentIntersectionOrder`.
    public static unsafe bool SegmentIntersectionOrder(MR.Std.Const_Array_MRPreciseVertCoords2_6 vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_segmentIntersectionOrder_std_array_MR_PreciseVertCoords2_6", ExactSpelling = true)]
        extern static byte __MR_segmentIntersectionOrder_std_array_MR_PreciseVertCoords2_6(MR.Std.Const_Array_MRPreciseVertCoords2_6._Underlying *vs);
        return __MR_segmentIntersectionOrder_std_array_MR_PreciseVertCoords2_6(vs._UnderlyingPtr) != 0;
    }

    /// finds intersection precise, using high precision int inside
    /// this function input should have intersection
    /// Generated from function `MR::findSegmentSegmentIntersectionPrecise`.
    public static unsafe MR.Vector2i FindSegmentSegmentIntersectionPrecise(MR.Const_Vector2i a, MR.Const_Vector2i b, MR.Const_Vector2i c, MR.Const_Vector2i d)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSegmentSegmentIntersectionPrecise_4", ExactSpelling = true)]
        extern static MR.Vector2i __MR_findSegmentSegmentIntersectionPrecise_4(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b, MR.Const_Vector2i._Underlying *c, MR.Const_Vector2i._Underlying *d);
        return __MR_findSegmentSegmentIntersectionPrecise_4(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr);
    }

    /// finds intersection precise, using high precision int inside
    /// this function input should have intersection
    /// Generated from function `MR::findSegmentSegmentIntersectionPrecise`.
    public static unsafe MR.Vector2f FindSegmentSegmentIntersectionPrecise(MR.Const_Vector2f a, MR.Const_Vector2f b, MR.Const_Vector2f c, MR.Const_Vector2f d, MR._ByValue_CoordinateConverters2 converters)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSegmentSegmentIntersectionPrecise_5", ExactSpelling = true)]
        extern static MR.Vector2f __MR_findSegmentSegmentIntersectionPrecise_5(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b, MR.Const_Vector2f._Underlying *c, MR.Const_Vector2f._Underlying *d, MR.Misc._PassBy converters_pass_by, MR.CoordinateConverters2._Underlying *converters);
        return __MR_findSegmentSegmentIntersectionPrecise_5(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr, converters.PassByMode, converters.Value is not null ? converters.Value._UnderlyingPtr : null);
    }
}
