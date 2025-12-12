public static partial class MR
{
    /// Generated from class `MR::PolylineProjectionResult2`.
    /// This is the const half of the class.
    public class Const_PolylineProjectionResult2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineProjectionResult2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineProjectionResult2_Destroy(_Underlying *_this);
            __MR_PolylineProjectionResult2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineProjectionResult2() {Dispose(false);}

        /// polyline's edge containing the closest point
        public unsafe MR.Const_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_Get_line", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_PolylineProjectionResult2_Get_line(_Underlying *_this);
                return new(__MR_PolylineProjectionResult2_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public unsafe MR.Const_Vector2f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_PolylineProjectionResult2_Get_point(_Underlying *_this);
                return new(__MR_PolylineProjectionResult2_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from pt to proj
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionResult2_Get_distSq(_Underlying *_this);
                return *__MR_PolylineProjectionResult2_Get_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineProjectionResult2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionResult2_DefaultConstruct();
        }

        /// Constructs `MR::PolylineProjectionResult2` elementwise.
        public unsafe Const_PolylineProjectionResult2(MR.UndirectedEdgeId line, MR.Vector2f point, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector2f point, float distSq);
            _UnderlyingPtr = __MR_PolylineProjectionResult2_ConstructFrom(line, point, distSq);
        }

        /// Generated from constructor `MR::PolylineProjectionResult2::PolylineProjectionResult2`.
        public unsafe Const_PolylineProjectionResult2(MR.Const_PolylineProjectionResult2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_ConstructFromAnother(MR.PolylineProjectionResult2._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionResult2_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PolylineProjectionResult2::operator bool`.
        public static unsafe explicit operator bool(MR.Const_PolylineProjectionResult2 _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PolylineProjectionResult2_ConvertTo_bool(MR.Const_PolylineProjectionResult2._Underlying *_this);
            return __MR_PolylineProjectionResult2_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// check for validity, otherwise the projection was not found
        /// Generated from method `MR::PolylineProjectionResult2::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_valid", ExactSpelling = true)]
            extern static byte __MR_PolylineProjectionResult2_valid(_Underlying *_this);
            return __MR_PolylineProjectionResult2_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::PolylineProjectionResult2`.
    /// This is the non-const half of the class.
    public class PolylineProjectionResult2 : Const_PolylineProjectionResult2
    {
        internal unsafe PolylineProjectionResult2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// polyline's edge containing the closest point
        public new unsafe MR.Mut_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_GetMutable_line", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_PolylineProjectionResult2_GetMutable_line(_Underlying *_this);
                return new(__MR_PolylineProjectionResult2_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public new unsafe MR.Mut_Vector2f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_PolylineProjectionResult2_GetMutable_point(_Underlying *_this);
                return new(__MR_PolylineProjectionResult2_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from pt to proj
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionResult2_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_PolylineProjectionResult2_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineProjectionResult2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionResult2_DefaultConstruct();
        }

        /// Constructs `MR::PolylineProjectionResult2` elementwise.
        public unsafe PolylineProjectionResult2(MR.UndirectedEdgeId line, MR.Vector2f point, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector2f point, float distSq);
            _UnderlyingPtr = __MR_PolylineProjectionResult2_ConstructFrom(line, point, distSq);
        }

        /// Generated from constructor `MR::PolylineProjectionResult2::PolylineProjectionResult2`.
        public unsafe PolylineProjectionResult2(MR.Const_PolylineProjectionResult2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_ConstructFromAnother(MR.PolylineProjectionResult2._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionResult2_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineProjectionResult2::operator=`.
        public unsafe MR.PolylineProjectionResult2 Assign(MR.Const_PolylineProjectionResult2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult2._Underlying *__MR_PolylineProjectionResult2_AssignFromAnother(_Underlying *_this, MR.PolylineProjectionResult2._Underlying *_other);
            return new(__MR_PolylineProjectionResult2_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineProjectionResult2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineProjectionResult2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionResult2`/`Const_PolylineProjectionResult2` directly.
    public class _InOptMut_PolylineProjectionResult2
    {
        public PolylineProjectionResult2? Opt;

        public _InOptMut_PolylineProjectionResult2() {}
        public _InOptMut_PolylineProjectionResult2(PolylineProjectionResult2 value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineProjectionResult2(PolylineProjectionResult2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineProjectionResult2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineProjectionResult2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionResult2`/`Const_PolylineProjectionResult2` to pass it to the function.
    public class _InOptConst_PolylineProjectionResult2
    {
        public Const_PolylineProjectionResult2? Opt;

        public _InOptConst_PolylineProjectionResult2() {}
        public _InOptConst_PolylineProjectionResult2(Const_PolylineProjectionResult2 value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineProjectionResult2(Const_PolylineProjectionResult2 value) {return new(value);}
    }

    /// Generated from class `MR::PolylineProjectionResult3`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PolylineProjectionResult3Arg`
    /// This is the const half of the class.
    public class Const_PolylineProjectionResult3 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineProjectionResult3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineProjectionResult3_Destroy(_Underlying *_this);
            __MR_PolylineProjectionResult3_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineProjectionResult3() {Dispose(false);}

        /// polyline's edge containing the closest point
        public unsafe MR.Const_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_Get_line", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_PolylineProjectionResult3_Get_line(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public unsafe MR.Const_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PolylineProjectionResult3_Get_point(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from pt to proj
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionResult3_Get_distSq(_Underlying *_this);
                return *__MR_PolylineProjectionResult3_Get_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineProjectionResult3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionResult3_DefaultConstruct();
        }

        /// Constructs `MR::PolylineProjectionResult3` elementwise.
        public unsafe Const_PolylineProjectionResult3(MR.UndirectedEdgeId line, MR.Vector3f point, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector3f point, float distSq);
            _UnderlyingPtr = __MR_PolylineProjectionResult3_ConstructFrom(line, point, distSq);
        }

        /// Generated from constructor `MR::PolylineProjectionResult3::PolylineProjectionResult3`.
        public unsafe Const_PolylineProjectionResult3(MR.Const_PolylineProjectionResult3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_ConstructFromAnother(MR.PolylineProjectionResult3._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionResult3_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PolylineProjectionResult3::operator bool`.
        public static unsafe explicit operator bool(MR.Const_PolylineProjectionResult3 _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PolylineProjectionResult3_ConvertTo_bool(MR.Const_PolylineProjectionResult3._Underlying *_this);
            return __MR_PolylineProjectionResult3_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// check for validity, otherwise the projection was not found
        /// Generated from method `MR::PolylineProjectionResult3::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_valid", ExactSpelling = true)]
            extern static byte __MR_PolylineProjectionResult3_valid(_Underlying *_this);
            return __MR_PolylineProjectionResult3_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::PolylineProjectionResult3`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PolylineProjectionResult3Arg`
    /// This is the non-const half of the class.
    public class PolylineProjectionResult3 : Const_PolylineProjectionResult3
    {
        internal unsafe PolylineProjectionResult3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// polyline's edge containing the closest point
        public new unsafe MR.Mut_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_GetMutable_line", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_PolylineProjectionResult3_GetMutable_line(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public new unsafe MR.Mut_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PolylineProjectionResult3_GetMutable_point(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from pt to proj
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionResult3_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_PolylineProjectionResult3_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineProjectionResult3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionResult3_DefaultConstruct();
        }

        /// Constructs `MR::PolylineProjectionResult3` elementwise.
        public unsafe PolylineProjectionResult3(MR.UndirectedEdgeId line, MR.Vector3f point, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector3f point, float distSq);
            _UnderlyingPtr = __MR_PolylineProjectionResult3_ConstructFrom(line, point, distSq);
        }

        /// Generated from constructor `MR::PolylineProjectionResult3::PolylineProjectionResult3`.
        public unsafe PolylineProjectionResult3(MR.Const_PolylineProjectionResult3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_ConstructFromAnother(MR.PolylineProjectionResult3._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionResult3_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineProjectionResult3::operator=`.
        public unsafe MR.PolylineProjectionResult3 Assign(MR.Const_PolylineProjectionResult3 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3_AssignFromAnother(_Underlying *_this, MR.PolylineProjectionResult3._Underlying *_other);
            return new(__MR_PolylineProjectionResult3_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineProjectionResult3` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineProjectionResult3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionResult3`/`Const_PolylineProjectionResult3` directly.
    public class _InOptMut_PolylineProjectionResult3
    {
        public PolylineProjectionResult3? Opt;

        public _InOptMut_PolylineProjectionResult3() {}
        public _InOptMut_PolylineProjectionResult3(PolylineProjectionResult3 value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineProjectionResult3(PolylineProjectionResult3 value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineProjectionResult3` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineProjectionResult3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionResult3`/`Const_PolylineProjectionResult3` to pass it to the function.
    public class _InOptConst_PolylineProjectionResult3
    {
        public Const_PolylineProjectionResult3? Opt;

        public _InOptConst_PolylineProjectionResult3() {}
        public _InOptConst_PolylineProjectionResult3(Const_PolylineProjectionResult3 value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineProjectionResult3(Const_PolylineProjectionResult3 value) {return new(value);}
    }

    /// Generated from class `MR::Polyline2ProjectionWithOffsetResult`.
    /// This is the const half of the class.
    public class Const_Polyline2ProjectionWithOffsetResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polyline2ProjectionWithOffsetResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_Destroy", ExactSpelling = true)]
            extern static void __MR_Polyline2ProjectionWithOffsetResult_Destroy(_Underlying *_this);
            __MR_Polyline2ProjectionWithOffsetResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polyline2ProjectionWithOffsetResult() {Dispose(false);}

        /// closest line id on polyline
        public unsafe MR.Const_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_Get_line", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_Polyline2ProjectionWithOffsetResult_Get_line(_Underlying *_this);
                return new(__MR_Polyline2ProjectionWithOffsetResult_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public unsafe MR.Const_Vector2f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_Polyline2ProjectionWithOffsetResult_Get_point(_Underlying *_this);
                return new(__MR_Polyline2ProjectionWithOffsetResult_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// distance from offset point to proj
        public unsafe float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_Get_dist", ExactSpelling = true)]
                extern static float *__MR_Polyline2ProjectionWithOffsetResult_Get_dist(_Underlying *_this);
                return *__MR_Polyline2ProjectionWithOffsetResult_Get_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polyline2ProjectionWithOffsetResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_DefaultConstruct();
            _UnderlyingPtr = __MR_Polyline2ProjectionWithOffsetResult_DefaultConstruct();
        }

        /// Constructs `MR::Polyline2ProjectionWithOffsetResult` elementwise.
        public unsafe Const_Polyline2ProjectionWithOffsetResult(MR.UndirectedEdgeId line, MR.Vector2f point, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector2f point, float dist);
            _UnderlyingPtr = __MR_Polyline2ProjectionWithOffsetResult_ConstructFrom(line, point, dist);
        }

        /// Generated from constructor `MR::Polyline2ProjectionWithOffsetResult::Polyline2ProjectionWithOffsetResult`.
        public unsafe Const_Polyline2ProjectionWithOffsetResult(MR.Const_Polyline2ProjectionWithOffsetResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_ConstructFromAnother(MR.Polyline2ProjectionWithOffsetResult._Underlying *_other);
            _UnderlyingPtr = __MR_Polyline2ProjectionWithOffsetResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::Polyline2ProjectionWithOffsetResult`.
    /// This is the non-const half of the class.
    public class Polyline2ProjectionWithOffsetResult : Const_Polyline2ProjectionWithOffsetResult
    {
        internal unsafe Polyline2ProjectionWithOffsetResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// closest line id on polyline
        public new unsafe MR.Mut_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_GetMutable_line", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_Polyline2ProjectionWithOffsetResult_GetMutable_line(_Underlying *_this);
                return new(__MR_Polyline2ProjectionWithOffsetResult_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public new unsafe MR.Mut_Vector2f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_Polyline2ProjectionWithOffsetResult_GetMutable_point(_Underlying *_this);
                return new(__MR_Polyline2ProjectionWithOffsetResult_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// distance from offset point to proj
        public new unsafe ref float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_GetMutable_dist", ExactSpelling = true)]
                extern static float *__MR_Polyline2ProjectionWithOffsetResult_GetMutable_dist(_Underlying *_this);
                return ref *__MR_Polyline2ProjectionWithOffsetResult_GetMutable_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polyline2ProjectionWithOffsetResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_DefaultConstruct();
            _UnderlyingPtr = __MR_Polyline2ProjectionWithOffsetResult_DefaultConstruct();
        }

        /// Constructs `MR::Polyline2ProjectionWithOffsetResult` elementwise.
        public unsafe Polyline2ProjectionWithOffsetResult(MR.UndirectedEdgeId line, MR.Vector2f point, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector2f point, float dist);
            _UnderlyingPtr = __MR_Polyline2ProjectionWithOffsetResult_ConstructFrom(line, point, dist);
        }

        /// Generated from constructor `MR::Polyline2ProjectionWithOffsetResult::Polyline2ProjectionWithOffsetResult`.
        public unsafe Polyline2ProjectionWithOffsetResult(MR.Const_Polyline2ProjectionWithOffsetResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_ConstructFromAnother(MR.Polyline2ProjectionWithOffsetResult._Underlying *_other);
            _UnderlyingPtr = __MR_Polyline2ProjectionWithOffsetResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Polyline2ProjectionWithOffsetResult::operator=`.
        public unsafe MR.Polyline2ProjectionWithOffsetResult Assign(MR.Const_Polyline2ProjectionWithOffsetResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2ProjectionWithOffsetResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_Polyline2ProjectionWithOffsetResult_AssignFromAnother(_Underlying *_this, MR.Polyline2ProjectionWithOffsetResult._Underlying *_other);
            return new(__MR_Polyline2ProjectionWithOffsetResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Polyline2ProjectionWithOffsetResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polyline2ProjectionWithOffsetResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polyline2ProjectionWithOffsetResult`/`Const_Polyline2ProjectionWithOffsetResult` directly.
    public class _InOptMut_Polyline2ProjectionWithOffsetResult
    {
        public Polyline2ProjectionWithOffsetResult? Opt;

        public _InOptMut_Polyline2ProjectionWithOffsetResult() {}
        public _InOptMut_Polyline2ProjectionWithOffsetResult(Polyline2ProjectionWithOffsetResult value) {Opt = value;}
        public static implicit operator _InOptMut_Polyline2ProjectionWithOffsetResult(Polyline2ProjectionWithOffsetResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polyline2ProjectionWithOffsetResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polyline2ProjectionWithOffsetResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polyline2ProjectionWithOffsetResult`/`Const_Polyline2ProjectionWithOffsetResult` to pass it to the function.
    public class _InOptConst_Polyline2ProjectionWithOffsetResult
    {
        public Const_Polyline2ProjectionWithOffsetResult? Opt;

        public _InOptConst_Polyline2ProjectionWithOffsetResult() {}
        public _InOptConst_Polyline2ProjectionWithOffsetResult(Const_Polyline2ProjectionWithOffsetResult value) {Opt = value;}
        public static implicit operator _InOptConst_Polyline2ProjectionWithOffsetResult(Const_Polyline2ProjectionWithOffsetResult value) {return new(value);}
    }

    /// Generated from class `MR::PolylineProjectionWithOffsetResult3`.
    /// This is the const half of the class.
    public class Const_PolylineProjectionWithOffsetResult3 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineProjectionWithOffsetResult3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineProjectionWithOffsetResult3_Destroy(_Underlying *_this);
            __MR_PolylineProjectionWithOffsetResult3_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineProjectionWithOffsetResult3() {Dispose(false);}

        /// closest line id on polyline
        public unsafe MR.Const_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_Get_line", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_PolylineProjectionWithOffsetResult3_Get_line(_Underlying *_this);
                return new(__MR_PolylineProjectionWithOffsetResult3_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public unsafe MR.Const_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PolylineProjectionWithOffsetResult3_Get_point(_Underlying *_this);
                return new(__MR_PolylineProjectionWithOffsetResult3_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// distance from offset point to proj
        public unsafe float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_Get_dist", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionWithOffsetResult3_Get_dist(_Underlying *_this);
                return *__MR_PolylineProjectionWithOffsetResult3_Get_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineProjectionWithOffsetResult3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionWithOffsetResult3_DefaultConstruct();
        }

        /// Constructs `MR::PolylineProjectionWithOffsetResult3` elementwise.
        public unsafe Const_PolylineProjectionWithOffsetResult3(MR.UndirectedEdgeId line, MR.Vector3f point, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector3f point, float dist);
            _UnderlyingPtr = __MR_PolylineProjectionWithOffsetResult3_ConstructFrom(line, point, dist);
        }

        /// Generated from constructor `MR::PolylineProjectionWithOffsetResult3::PolylineProjectionWithOffsetResult3`.
        public unsafe Const_PolylineProjectionWithOffsetResult3(MR.Const_PolylineProjectionWithOffsetResult3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_ConstructFromAnother(MR.PolylineProjectionWithOffsetResult3._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionWithOffsetResult3_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PolylineProjectionWithOffsetResult3`.
    /// This is the non-const half of the class.
    public class PolylineProjectionWithOffsetResult3 : Const_PolylineProjectionWithOffsetResult3
    {
        internal unsafe PolylineProjectionWithOffsetResult3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// closest line id on polyline
        public new unsafe MR.Mut_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_GetMutable_line", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_PolylineProjectionWithOffsetResult3_GetMutable_line(_Underlying *_this);
                return new(__MR_PolylineProjectionWithOffsetResult3_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public new unsafe MR.Mut_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PolylineProjectionWithOffsetResult3_GetMutable_point(_Underlying *_this);
                return new(__MR_PolylineProjectionWithOffsetResult3_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// distance from offset point to proj
        public new unsafe ref float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_GetMutable_dist", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionWithOffsetResult3_GetMutable_dist(_Underlying *_this);
                return ref *__MR_PolylineProjectionWithOffsetResult3_GetMutable_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineProjectionWithOffsetResult3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionWithOffsetResult3_DefaultConstruct();
        }

        /// Constructs `MR::PolylineProjectionWithOffsetResult3` elementwise.
        public unsafe PolylineProjectionWithOffsetResult3(MR.UndirectedEdgeId line, MR.Vector3f point, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_ConstructFrom(MR.UndirectedEdgeId line, MR.Vector3f point, float dist);
            _UnderlyingPtr = __MR_PolylineProjectionWithOffsetResult3_ConstructFrom(line, point, dist);
        }

        /// Generated from constructor `MR::PolylineProjectionWithOffsetResult3::PolylineProjectionWithOffsetResult3`.
        public unsafe PolylineProjectionWithOffsetResult3(MR.Const_PolylineProjectionWithOffsetResult3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_ConstructFromAnother(MR.PolylineProjectionWithOffsetResult3._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionWithOffsetResult3_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineProjectionWithOffsetResult3::operator=`.
        public unsafe MR.PolylineProjectionWithOffsetResult3 Assign(MR.Const_PolylineProjectionWithOffsetResult3 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionWithOffsetResult3_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_PolylineProjectionWithOffsetResult3_AssignFromAnother(_Underlying *_this, MR.PolylineProjectionWithOffsetResult3._Underlying *_other);
            return new(__MR_PolylineProjectionWithOffsetResult3_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineProjectionWithOffsetResult3` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineProjectionWithOffsetResult3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionWithOffsetResult3`/`Const_PolylineProjectionWithOffsetResult3` directly.
    public class _InOptMut_PolylineProjectionWithOffsetResult3
    {
        public PolylineProjectionWithOffsetResult3? Opt;

        public _InOptMut_PolylineProjectionWithOffsetResult3() {}
        public _InOptMut_PolylineProjectionWithOffsetResult3(PolylineProjectionWithOffsetResult3 value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineProjectionWithOffsetResult3(PolylineProjectionWithOffsetResult3 value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineProjectionWithOffsetResult3` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineProjectionWithOffsetResult3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionWithOffsetResult3`/`Const_PolylineProjectionWithOffsetResult3` to pass it to the function.
    public class _InOptConst_PolylineProjectionWithOffsetResult3
    {
        public Const_PolylineProjectionWithOffsetResult3? Opt;

        public _InOptConst_PolylineProjectionWithOffsetResult3() {}
        public _InOptConst_PolylineProjectionWithOffsetResult3(Const_PolylineProjectionWithOffsetResult3 value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineProjectionWithOffsetResult3(Const_PolylineProjectionWithOffsetResult3 value) {return new(value);}
    }

    /// Generated from class `MR::PolylineProjectionResult3Arg`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::PolylineProjectionResult3`
    /// This is the const half of the class.
    public class Const_PolylineProjectionResult3Arg : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineProjectionResult3Arg(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineProjectionResult3Arg_Destroy(_Underlying *_this);
            __MR_PolylineProjectionResult3Arg_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineProjectionResult3Arg() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_PolylineProjectionResult3(Const_PolylineProjectionResult3Arg self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_UpcastTo_MR_PolylineProjectionResult3", ExactSpelling = true)]
            extern static MR.Const_PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3Arg_UpcastTo_MR_PolylineProjectionResult3(_Underlying *_this);
            MR.Const_PolylineProjectionResult3 ret = new(__MR_PolylineProjectionResult3Arg_UpcastTo_MR_PolylineProjectionResult3(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // id of source point from which closest point was searched
        public unsafe MR.Const_VertId PointId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_Get_pointId", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PolylineProjectionResult3Arg_Get_pointId(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3Arg_Get_pointId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// polyline's edge containing the closest point
        public unsafe MR.Const_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_Get_line", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeId._Underlying *__MR_PolylineProjectionResult3Arg_Get_line(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3Arg_Get_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public unsafe MR.Const_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PolylineProjectionResult3Arg_Get_point(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3Arg_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from pt to proj
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionResult3Arg_Get_distSq(_Underlying *_this);
                return *__MR_PolylineProjectionResult3Arg_Get_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineProjectionResult3Arg() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3Arg._Underlying *__MR_PolylineProjectionResult3Arg_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionResult3Arg_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineProjectionResult3Arg::PolylineProjectionResult3Arg`.
        public unsafe Const_PolylineProjectionResult3Arg(MR.Const_PolylineProjectionResult3Arg _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3Arg._Underlying *__MR_PolylineProjectionResult3Arg_ConstructFromAnother(MR.PolylineProjectionResult3Arg._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionResult3Arg_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PolylineProjectionResult3Arg::operator bool`.
        public static unsafe explicit operator bool(MR.Const_PolylineProjectionResult3Arg _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PolylineProjectionResult3Arg_ConvertTo_bool(MR.Const_PolylineProjectionResult3Arg._Underlying *_this);
            return __MR_PolylineProjectionResult3Arg_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// check for validity, otherwise the projection was not found
        /// Generated from method `MR::PolylineProjectionResult3Arg::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_valid", ExactSpelling = true)]
            extern static byte __MR_PolylineProjectionResult3Arg_valid(_Underlying *_this);
            return __MR_PolylineProjectionResult3Arg_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::PolylineProjectionResult3Arg`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::PolylineProjectionResult3`
    /// This is the non-const half of the class.
    public class PolylineProjectionResult3Arg : Const_PolylineProjectionResult3Arg
    {
        internal unsafe PolylineProjectionResult3Arg(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.PolylineProjectionResult3(PolylineProjectionResult3Arg self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_UpcastTo_MR_PolylineProjectionResult3", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3._Underlying *__MR_PolylineProjectionResult3Arg_UpcastTo_MR_PolylineProjectionResult3(_Underlying *_this);
            MR.PolylineProjectionResult3 ret = new(__MR_PolylineProjectionResult3Arg_UpcastTo_MR_PolylineProjectionResult3(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // id of source point from which closest point was searched
        public new unsafe MR.Mut_VertId PointId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_GetMutable_pointId", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PolylineProjectionResult3Arg_GetMutable_pointId(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3Arg_GetMutable_pointId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// polyline's edge containing the closest point
        public new unsafe MR.Mut_UndirectedEdgeId Line
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_GetMutable_line", ExactSpelling = true)]
                extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_PolylineProjectionResult3Arg_GetMutable_line(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3Arg_GetMutable_line(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closest point on polyline, transformed by xf if it is given
        public new unsafe MR.Mut_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PolylineProjectionResult3Arg_GetMutable_point(_Underlying *_this);
                return new(__MR_PolylineProjectionResult3Arg_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from pt to proj
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_PolylineProjectionResult3Arg_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_PolylineProjectionResult3Arg_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineProjectionResult3Arg() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3Arg._Underlying *__MR_PolylineProjectionResult3Arg_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineProjectionResult3Arg_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineProjectionResult3Arg::PolylineProjectionResult3Arg`.
        public unsafe PolylineProjectionResult3Arg(MR.Const_PolylineProjectionResult3Arg _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3Arg._Underlying *__MR_PolylineProjectionResult3Arg_ConstructFromAnother(MR.PolylineProjectionResult3Arg._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineProjectionResult3Arg_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineProjectionResult3Arg::operator=`.
        public unsafe MR.PolylineProjectionResult3Arg Assign(MR.Const_PolylineProjectionResult3Arg _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineProjectionResult3Arg_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineProjectionResult3Arg._Underlying *__MR_PolylineProjectionResult3Arg_AssignFromAnother(_Underlying *_this, MR.PolylineProjectionResult3Arg._Underlying *_other);
            return new(__MR_PolylineProjectionResult3Arg_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineProjectionResult3Arg` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineProjectionResult3Arg`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionResult3Arg`/`Const_PolylineProjectionResult3Arg` directly.
    public class _InOptMut_PolylineProjectionResult3Arg
    {
        public PolylineProjectionResult3Arg? Opt;

        public _InOptMut_PolylineProjectionResult3Arg() {}
        public _InOptMut_PolylineProjectionResult3Arg(PolylineProjectionResult3Arg value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineProjectionResult3Arg(PolylineProjectionResult3Arg value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineProjectionResult3Arg` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineProjectionResult3Arg`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineProjectionResult3Arg`/`Const_PolylineProjectionResult3Arg` to pass it to the function.
    public class _InOptConst_PolylineProjectionResult3Arg
    {
        public Const_PolylineProjectionResult3Arg? Opt;

        public _InOptConst_PolylineProjectionResult3Arg() {}
        public _InOptConst_PolylineProjectionResult3Arg(Const_PolylineProjectionResult3Arg value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineProjectionResult3Arg(Const_PolylineProjectionResult3Arg value) {return new(value);}
    }

    /**
    * \brief computes the closest point on polyline to given point
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnPolyline2`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.PolylineProjectionResult2 FindProjectionOnPolyline2(MR.Const_Vector2f pt, MR.Const_Polyline2 polyline, float? upDistLimitSq = null, MR.Mut_AffineXf2f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPolyline2", ExactSpelling = true)]
        extern static MR.PolylineProjectionResult2._Underlying *__MR_findProjectionOnPolyline2(MR.Const_Vector2f._Underlying *pt, MR.Const_Polyline2._Underlying *polyline, float *upDistLimitSq, MR.Mut_AffineXf2f._Underlying *xf, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnPolyline2(pt._UnderlyingPtr, polyline._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on polyline to given point
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnPolyline`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.PolylineProjectionResult3 FindProjectionOnPolyline(MR.Const_Vector3f pt, MR.Const_Polyline3 polyline, float? upDistLimitSq = null, MR.Mut_AffineXf3f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPolyline_MR_Vector3f", ExactSpelling = true)]
        extern static MR.PolylineProjectionResult3._Underlying *__MR_findProjectionOnPolyline_MR_Vector3f(MR.Const_Vector3f._Underlying *pt, MR.Const_Polyline3._Underlying *polyline, float *upDistLimitSq, MR.Mut_AffineXf3f._Underlying *xf, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnPolyline_MR_Vector3f(pt._UnderlyingPtr, polyline._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief for each of points (pointsRegion) computes the closest point on polyline and returns the point for which maximum distance is reached,
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findMaxProjectionOnPolyline`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.PolylineProjectionResult3Arg FindMaxProjectionOnPolyline(MR.Const_VertCoords points, MR.Const_Polyline3 polyline, MR.Const_VertBitSet? pointsRegion = null, MR.Mut_AffineXf3f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMaxProjectionOnPolyline", ExactSpelling = true)]
        extern static MR.PolylineProjectionResult3Arg._Underlying *__MR_findMaxProjectionOnPolyline(MR.Const_VertCoords._Underlying *points, MR.Const_Polyline3._Underlying *polyline, MR.Const_VertBitSet._Underlying *pointsRegion, MR.Mut_AffineXf3f._Underlying *xf, float *loDistLimitSq);
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findMaxProjectionOnPolyline(points._UnderlyingPtr, polyline._UnderlyingPtr, pointsRegion is not null ? pointsRegion._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on polyline to given straight line
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnPolyline`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.PolylineProjectionResult3 FindProjectionOnPolyline(MR.Const_Line3f ln, MR.Const_Polyline3 polyline, float? upDistLimitSq = null, MR.Mut_AffineXf3f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPolyline_MR_Line3f", ExactSpelling = true)]
        extern static MR.PolylineProjectionResult3._Underlying *__MR_findProjectionOnPolyline_MR_Line3f(MR.Const_Line3f._Underlying *ln, MR.Const_Polyline3._Underlying *polyline, float *upDistLimitSq, MR.Mut_AffineXf3f._Underlying *xf, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnPolyline_MR_Line3f(ln._UnderlyingPtr, polyline._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on polyline to given point, respecting each edge offset
    * \param offsetPerEdge offset for each edge of polyline
    * \param upDistLimit upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimit and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimit low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnPolyline2WithOffset`.
    /// Parameter `upDistLimit` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimit` defaults to `0`.
    public static unsafe MR.Polyline2ProjectionWithOffsetResult FindProjectionOnPolyline2WithOffset(MR.Const_Vector2f pt, MR.Const_Polyline2 polyline, MR.Const_UndirectedEdgeScalars offsetPerEdge, float? upDistLimit = null, MR.Mut_AffineXf2f? xf = null, float? loDistLimit = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPolyline2WithOffset", ExactSpelling = true)]
        extern static MR.Polyline2ProjectionWithOffsetResult._Underlying *__MR_findProjectionOnPolyline2WithOffset(MR.Const_Vector2f._Underlying *pt, MR.Const_Polyline2._Underlying *polyline, MR.Const_UndirectedEdgeScalars._Underlying *offsetPerEdge, float *upDistLimit, MR.Mut_AffineXf2f._Underlying *xf, float *loDistLimit);
        float __deref_upDistLimit = upDistLimit.GetValueOrDefault();
        float __deref_loDistLimit = loDistLimit.GetValueOrDefault();
        return new(__MR_findProjectionOnPolyline2WithOffset(pt._UnderlyingPtr, polyline._UnderlyingPtr, offsetPerEdge._UnderlyingPtr, upDistLimit.HasValue ? &__deref_upDistLimit : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimit.HasValue ? &__deref_loDistLimit : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on polyline to given point, respecting each edge offset
    * \param offsetPerEdge offset for each edge of polyline
    * \param upDistLimit upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimit and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimit low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnPolylineWithOffset`.
    /// Parameter `upDistLimit` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimit` defaults to `0`.
    public static unsafe MR.PolylineProjectionWithOffsetResult3 FindProjectionOnPolylineWithOffset(MR.Const_Vector3f pt, MR.Const_Polyline3 polyline, MR.Const_UndirectedEdgeScalars offsetPerEdge, float? upDistLimit = null, MR.Mut_AffineXf3f? xf = null, float? loDistLimit = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPolylineWithOffset", ExactSpelling = true)]
        extern static MR.PolylineProjectionWithOffsetResult3._Underlying *__MR_findProjectionOnPolylineWithOffset(MR.Const_Vector3f._Underlying *pt, MR.Const_Polyline3._Underlying *polyline, MR.Const_UndirectedEdgeScalars._Underlying *offsetPerEdge, float *upDistLimit, MR.Mut_AffineXf3f._Underlying *xf, float *loDistLimit);
        float __deref_upDistLimit = upDistLimit.GetValueOrDefault();
        float __deref_loDistLimit = loDistLimit.GetValueOrDefault();
        return new(__MR_findProjectionOnPolylineWithOffset(pt._UnderlyingPtr, polyline._UnderlyingPtr, offsetPerEdge._UnderlyingPtr, upDistLimit.HasValue ? &__deref_upDistLimit : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimit.HasValue ? &__deref_loDistLimit : null), is_owning: true);
    }

    /// Finds all edges of given polyline that cross or touch given ball (center, radius)
    /// Generated from function `MR::findEdgesInBall`.
    public static unsafe void FindEdgesInBall(MR.Const_Polyline2 polyline, MR.Const_Vector2f center, float radius, MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdConstMRVector2fRefFloat foundCallback, MR.Mut_AffineXf2f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findEdgesInBall_MR_Polyline2", ExactSpelling = true)]
        extern static void __MR_findEdgesInBall_MR_Polyline2(MR.Const_Polyline2._Underlying *polyline, MR.Const_Vector2f._Underlying *center, float radius, MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdConstMRVector2fRefFloat._Underlying *foundCallback, MR.Mut_AffineXf2f._Underlying *xf);
        __MR_findEdgesInBall_MR_Polyline2(polyline._UnderlyingPtr, center._UnderlyingPtr, radius, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Finds all edges of given polyline that cross or touch given ball (center, radius)
    /// Generated from function `MR::findEdgesInBall`.
    public static unsafe void FindEdgesInBall(MR.Const_Polyline3 polyline, MR.Const_Vector3f center, float radius, MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdConstMRVector3fRefFloat foundCallback, MR.Mut_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findEdgesInBall_MR_Polyline3", ExactSpelling = true)]
        extern static void __MR_findEdgesInBall_MR_Polyline3(MR.Const_Polyline3._Underlying *polyline, MR.Const_Vector3f._Underlying *center, float radius, MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdConstMRVector3fRefFloat._Underlying *foundCallback, MR.Mut_AffineXf3f._Underlying *xf);
        __MR_findEdgesInBall_MR_Polyline3(polyline._UnderlyingPtr, center._UnderlyingPtr, radius, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Finds all edges of given mesh edges (specified by the tree) that cross or touch given ball (center, radius)
    /// Generated from function `MR::findMeshEdgesInBall`.
    public static unsafe void FindMeshEdgesInBall(MR.Const_Mesh mesh, MR.Const_AABBTreePolyline3 tree, MR.Const_Vector3f center, float radius, MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdConstMRVector3fRefFloat foundCallback, MR.Mut_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMeshEdgesInBall", ExactSpelling = true)]
        extern static void __MR_findMeshEdgesInBall(MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTreePolyline3._Underlying *tree, MR.Const_Vector3f._Underlying *center, float radius, MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdConstMRVector3fRefFloat._Underlying *foundCallback, MR.Mut_AffineXf3f._Underlying *xf);
        __MR_findMeshEdgesInBall(mesh._UnderlyingPtr, tree._UnderlyingPtr, center._UnderlyingPtr, radius, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /**
    * \brief computes the closest point on the mesh edges (specified by the tree) to given point
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnMeshEdges`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.PolylineProjectionResult3 FindProjectionOnMeshEdges(MR.Const_Vector3f pt, MR.Const_Mesh mesh, MR.Const_AABBTreePolyline3 tree, float? upDistLimitSq = null, MR.Mut_AffineXf3f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnMeshEdges_MR_Vector3f", ExactSpelling = true)]
        extern static MR.PolylineProjectionResult3._Underlying *__MR_findProjectionOnMeshEdges_MR_Vector3f(MR.Const_Vector3f._Underlying *pt, MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTreePolyline3._Underlying *tree, float *upDistLimitSq, MR.Mut_AffineXf3f._Underlying *xf, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnMeshEdges_MR_Vector3f(pt._UnderlyingPtr, mesh._UnderlyingPtr, tree._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on the mesh edges (specified by the tree) to given straight line
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exists returning upDistLimitSq and no valid point or edge
    * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    */
    /// Generated from function `MR::findProjectionOnMeshEdges`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.PolylineProjectionResult3 FindProjectionOnMeshEdges(MR.Const_Line3f ln, MR.Const_Mesh mesh, MR.Const_AABBTreePolyline3 tree, float? upDistLimitSq = null, MR.Mut_AffineXf3f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnMeshEdges_MR_Line3f", ExactSpelling = true)]
        extern static MR.PolylineProjectionResult3._Underlying *__MR_findProjectionOnMeshEdges_MR_Line3f(MR.Const_Line3f._Underlying *ln, MR.Const_Mesh._Underlying *mesh, MR.Const_AABBTreePolyline3._Underlying *tree, float *upDistLimitSq, MR.Mut_AffineXf3f._Underlying *xf, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnMeshEdges_MR_Line3f(ln._UnderlyingPtr, mesh._UnderlyingPtr, tree._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }
}
