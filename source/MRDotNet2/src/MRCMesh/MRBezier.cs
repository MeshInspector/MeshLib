public static partial class MR
{
    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve2f`.
    /// This is the const half of the class.
    public class Const_CubicBezierCurve2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CubicBezierCurve2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_Destroy", ExactSpelling = true)]
            extern static void __MR_CubicBezierCurve2f_Destroy(_Underlying *_this);
            __MR_CubicBezierCurve2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CubicBezierCurve2f() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_CubicBezierCurve2f_Get_elements();
                return *__MR_CubicBezierCurve2f_Get_elements();
            }
        }

        /// 4 control points
        public unsafe ref MR.ArrayVector2f4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_Get_p", ExactSpelling = true)]
                extern static MR.ArrayVector2f4 *__MR_CubicBezierCurve2f_Get_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve2f_Get_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CubicBezierCurve2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2f._Underlying *__MR_CubicBezierCurve2f_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve2f::CubicBezierCurve2f`.
        public unsafe Const_CubicBezierCurve2f(MR.Const_CubicBezierCurve2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2f._Underlying *__MR_CubicBezierCurve2f_ConstructFromAnother(MR.CubicBezierCurve2f._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes point on the curve from parameter value
        /// Generated from method `MR::CubicBezierCurve2f::getPoint`.
        public unsafe MR.Vector2f GetPoint(float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_getPoint", ExactSpelling = true)]
            extern static MR.Vector2f __MR_CubicBezierCurve2f_getPoint(_Underlying *_this, float t);
            return __MR_CubicBezierCurve2f_getPoint(_UnderlyingPtr, t);
        }

        /// computes weights of every control point for given parameter value, the sum of all weights is equal to 1
        /// Generated from method `MR::CubicBezierCurve2f::getWeights`.
        public static MR.Std.Array_Float_4 GetWeights(float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_getWeights", ExactSpelling = true)]
            extern static MR.Std.Array_Float_4 __MR_CubicBezierCurve2f_getWeights(float t);
            return __MR_CubicBezierCurve2f_getWeights(t);
        }
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve2f`.
    /// This is the non-const half of the class.
    public class CubicBezierCurve2f : Const_CubicBezierCurve2f
    {
        internal unsafe CubicBezierCurve2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// 4 control points
        new public unsafe ref MR.ArrayVector2f4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_GetMutable_p", ExactSpelling = true)]
                extern static MR.ArrayVector2f4 *__MR_CubicBezierCurve2f_GetMutable_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve2f_GetMutable_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CubicBezierCurve2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2f._Underlying *__MR_CubicBezierCurve2f_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve2f::CubicBezierCurve2f`.
        public unsafe CubicBezierCurve2f(MR.Const_CubicBezierCurve2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2f._Underlying *__MR_CubicBezierCurve2f_ConstructFromAnother(MR.CubicBezierCurve2f._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CubicBezierCurve2f::operator=`.
        public unsafe MR.CubicBezierCurve2f Assign(MR.Const_CubicBezierCurve2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2f._Underlying *__MR_CubicBezierCurve2f_AssignFromAnother(_Underlying *_this, MR.CubicBezierCurve2f._Underlying *_other);
            return new(__MR_CubicBezierCurve2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CubicBezierCurve2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CubicBezierCurve2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve2f`/`Const_CubicBezierCurve2f` directly.
    public class _InOptMut_CubicBezierCurve2f
    {
        public CubicBezierCurve2f? Opt;

        public _InOptMut_CubicBezierCurve2f() {}
        public _InOptMut_CubicBezierCurve2f(CubicBezierCurve2f value) {Opt = value;}
        public static implicit operator _InOptMut_CubicBezierCurve2f(CubicBezierCurve2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `CubicBezierCurve2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CubicBezierCurve2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve2f`/`Const_CubicBezierCurve2f` to pass it to the function.
    public class _InOptConst_CubicBezierCurve2f
    {
        public Const_CubicBezierCurve2f? Opt;

        public _InOptConst_CubicBezierCurve2f() {}
        public _InOptConst_CubicBezierCurve2f(Const_CubicBezierCurve2f value) {Opt = value;}
        public static implicit operator _InOptConst_CubicBezierCurve2f(Const_CubicBezierCurve2f value) {return new(value);}
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve2d`.
    /// This is the const half of the class.
    public class Const_CubicBezierCurve2d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CubicBezierCurve2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_Destroy", ExactSpelling = true)]
            extern static void __MR_CubicBezierCurve2d_Destroy(_Underlying *_this);
            __MR_CubicBezierCurve2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CubicBezierCurve2d() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_CubicBezierCurve2d_Get_elements();
                return *__MR_CubicBezierCurve2d_Get_elements();
            }
        }

        /// 4 control points
        public unsafe ref MR.ArrayVector2d4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_Get_p", ExactSpelling = true)]
                extern static MR.ArrayVector2d4 *__MR_CubicBezierCurve2d_Get_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve2d_Get_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CubicBezierCurve2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2d._Underlying *__MR_CubicBezierCurve2d_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve2d::CubicBezierCurve2d`.
        public unsafe Const_CubicBezierCurve2d(MR.Const_CubicBezierCurve2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2d._Underlying *__MR_CubicBezierCurve2d_ConstructFromAnother(MR.CubicBezierCurve2d._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes point on the curve from parameter value
        /// Generated from method `MR::CubicBezierCurve2d::getPoint`.
        public unsafe MR.Vector2d GetPoint(double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_getPoint", ExactSpelling = true)]
            extern static MR.Vector2d __MR_CubicBezierCurve2d_getPoint(_Underlying *_this, double t);
            return __MR_CubicBezierCurve2d_getPoint(_UnderlyingPtr, t);
        }

        /// computes weights of every control point for given parameter value, the sum of all weights is equal to 1
        /// Generated from method `MR::CubicBezierCurve2d::getWeights`.
        public static MR.Std.Array_Double_4 GetWeights(double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_getWeights", ExactSpelling = true)]
            extern static MR.Std.Array_Double_4 __MR_CubicBezierCurve2d_getWeights(double t);
            return __MR_CubicBezierCurve2d_getWeights(t);
        }
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve2d`.
    /// This is the non-const half of the class.
    public class CubicBezierCurve2d : Const_CubicBezierCurve2d
    {
        internal unsafe CubicBezierCurve2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// 4 control points
        new public unsafe ref MR.ArrayVector2d4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_GetMutable_p", ExactSpelling = true)]
                extern static MR.ArrayVector2d4 *__MR_CubicBezierCurve2d_GetMutable_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve2d_GetMutable_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CubicBezierCurve2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2d._Underlying *__MR_CubicBezierCurve2d_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve2d::CubicBezierCurve2d`.
        public unsafe CubicBezierCurve2d(MR.Const_CubicBezierCurve2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2d._Underlying *__MR_CubicBezierCurve2d_ConstructFromAnother(MR.CubicBezierCurve2d._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CubicBezierCurve2d::operator=`.
        public unsafe MR.CubicBezierCurve2d Assign(MR.Const_CubicBezierCurve2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve2d._Underlying *__MR_CubicBezierCurve2d_AssignFromAnother(_Underlying *_this, MR.CubicBezierCurve2d._Underlying *_other);
            return new(__MR_CubicBezierCurve2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CubicBezierCurve2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CubicBezierCurve2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve2d`/`Const_CubicBezierCurve2d` directly.
    public class _InOptMut_CubicBezierCurve2d
    {
        public CubicBezierCurve2d? Opt;

        public _InOptMut_CubicBezierCurve2d() {}
        public _InOptMut_CubicBezierCurve2d(CubicBezierCurve2d value) {Opt = value;}
        public static implicit operator _InOptMut_CubicBezierCurve2d(CubicBezierCurve2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `CubicBezierCurve2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CubicBezierCurve2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve2d`/`Const_CubicBezierCurve2d` to pass it to the function.
    public class _InOptConst_CubicBezierCurve2d
    {
        public Const_CubicBezierCurve2d? Opt;

        public _InOptConst_CubicBezierCurve2d() {}
        public _InOptConst_CubicBezierCurve2d(Const_CubicBezierCurve2d value) {Opt = value;}
        public static implicit operator _InOptConst_CubicBezierCurve2d(Const_CubicBezierCurve2d value) {return new(value);}
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve3f`.
    /// This is the const half of the class.
    public class Const_CubicBezierCurve3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CubicBezierCurve3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_Destroy", ExactSpelling = true)]
            extern static void __MR_CubicBezierCurve3f_Destroy(_Underlying *_this);
            __MR_CubicBezierCurve3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CubicBezierCurve3f() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_CubicBezierCurve3f_Get_elements();
                return *__MR_CubicBezierCurve3f_Get_elements();
            }
        }

        /// 4 control points
        public unsafe ref MR.ArrayVector3f4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_Get_p", ExactSpelling = true)]
                extern static MR.ArrayVector3f4 *__MR_CubicBezierCurve3f_Get_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve3f_Get_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CubicBezierCurve3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3f._Underlying *__MR_CubicBezierCurve3f_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve3f::CubicBezierCurve3f`.
        public unsafe Const_CubicBezierCurve3f(MR.Const_CubicBezierCurve3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3f._Underlying *__MR_CubicBezierCurve3f_ConstructFromAnother(MR.CubicBezierCurve3f._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes point on the curve from parameter value
        /// Generated from method `MR::CubicBezierCurve3f::getPoint`.
        public unsafe MR.Vector3f GetPoint(float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_getPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_CubicBezierCurve3f_getPoint(_Underlying *_this, float t);
            return __MR_CubicBezierCurve3f_getPoint(_UnderlyingPtr, t);
        }

        /// computes weights of every control point for given parameter value, the sum of all weights is equal to 1
        /// Generated from method `MR::CubicBezierCurve3f::getWeights`.
        public static MR.Std.Array_Float_4 GetWeights(float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_getWeights", ExactSpelling = true)]
            extern static MR.Std.Array_Float_4 __MR_CubicBezierCurve3f_getWeights(float t);
            return __MR_CubicBezierCurve3f_getWeights(t);
        }
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve3f`.
    /// This is the non-const half of the class.
    public class CubicBezierCurve3f : Const_CubicBezierCurve3f
    {
        internal unsafe CubicBezierCurve3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// 4 control points
        new public unsafe ref MR.ArrayVector3f4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_GetMutable_p", ExactSpelling = true)]
                extern static MR.ArrayVector3f4 *__MR_CubicBezierCurve3f_GetMutable_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve3f_GetMutable_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CubicBezierCurve3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3f._Underlying *__MR_CubicBezierCurve3f_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve3f::CubicBezierCurve3f`.
        public unsafe CubicBezierCurve3f(MR.Const_CubicBezierCurve3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3f._Underlying *__MR_CubicBezierCurve3f_ConstructFromAnother(MR.CubicBezierCurve3f._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CubicBezierCurve3f::operator=`.
        public unsafe MR.CubicBezierCurve3f Assign(MR.Const_CubicBezierCurve3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3f._Underlying *__MR_CubicBezierCurve3f_AssignFromAnother(_Underlying *_this, MR.CubicBezierCurve3f._Underlying *_other);
            return new(__MR_CubicBezierCurve3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CubicBezierCurve3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CubicBezierCurve3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve3f`/`Const_CubicBezierCurve3f` directly.
    public class _InOptMut_CubicBezierCurve3f
    {
        public CubicBezierCurve3f? Opt;

        public _InOptMut_CubicBezierCurve3f() {}
        public _InOptMut_CubicBezierCurve3f(CubicBezierCurve3f value) {Opt = value;}
        public static implicit operator _InOptMut_CubicBezierCurve3f(CubicBezierCurve3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `CubicBezierCurve3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CubicBezierCurve3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve3f`/`Const_CubicBezierCurve3f` to pass it to the function.
    public class _InOptConst_CubicBezierCurve3f
    {
        public Const_CubicBezierCurve3f? Opt;

        public _InOptConst_CubicBezierCurve3f() {}
        public _InOptConst_CubicBezierCurve3f(Const_CubicBezierCurve3f value) {Opt = value;}
        public static implicit operator _InOptConst_CubicBezierCurve3f(Const_CubicBezierCurve3f value) {return new(value);}
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve3d`.
    /// This is the const half of the class.
    public class Const_CubicBezierCurve3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CubicBezierCurve3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_Destroy", ExactSpelling = true)]
            extern static void __MR_CubicBezierCurve3d_Destroy(_Underlying *_this);
            __MR_CubicBezierCurve3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CubicBezierCurve3d() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_CubicBezierCurve3d_Get_elements();
                return *__MR_CubicBezierCurve3d_Get_elements();
            }
        }

        /// 4 control points
        public unsafe ref MR.ArrayVector3d4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_Get_p", ExactSpelling = true)]
                extern static MR.ArrayVector3d4 *__MR_CubicBezierCurve3d_Get_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve3d_Get_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CubicBezierCurve3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3d._Underlying *__MR_CubicBezierCurve3d_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve3d::CubicBezierCurve3d`.
        public unsafe Const_CubicBezierCurve3d(MR.Const_CubicBezierCurve3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3d._Underlying *__MR_CubicBezierCurve3d_ConstructFromAnother(MR.CubicBezierCurve3d._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes point on the curve from parameter value
        /// Generated from method `MR::CubicBezierCurve3d::getPoint`.
        public unsafe MR.Vector3d GetPoint(double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_getPoint", ExactSpelling = true)]
            extern static MR.Vector3d __MR_CubicBezierCurve3d_getPoint(_Underlying *_this, double t);
            return __MR_CubicBezierCurve3d_getPoint(_UnderlyingPtr, t);
        }

        /// computes weights of every control point for given parameter value, the sum of all weights is equal to 1
        /// Generated from method `MR::CubicBezierCurve3d::getWeights`.
        public static MR.Std.Array_Double_4 GetWeights(double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_getWeights", ExactSpelling = true)]
            extern static MR.Std.Array_Double_4 __MR_CubicBezierCurve3d_getWeights(double t);
            return __MR_CubicBezierCurve3d_getWeights(t);
        }
    }

    /// Cubic Bezier curve
    /// Generated from class `MR::CubicBezierCurve3d`.
    /// This is the non-const half of the class.
    public class CubicBezierCurve3d : Const_CubicBezierCurve3d
    {
        internal unsafe CubicBezierCurve3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// 4 control points
        new public unsafe ref MR.ArrayVector3d4 P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_GetMutable_p", ExactSpelling = true)]
                extern static MR.ArrayVector3d4 *__MR_CubicBezierCurve3d_GetMutable_p(_Underlying *_this);
                return ref *(__MR_CubicBezierCurve3d_GetMutable_p(_UnderlyingPtr));
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CubicBezierCurve3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3d._Underlying *__MR_CubicBezierCurve3d_DefaultConstruct();
            _UnderlyingPtr = __MR_CubicBezierCurve3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::CubicBezierCurve3d::CubicBezierCurve3d`.
        public unsafe CubicBezierCurve3d(MR.Const_CubicBezierCurve3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3d._Underlying *__MR_CubicBezierCurve3d_ConstructFromAnother(MR.CubicBezierCurve3d._Underlying *_other);
            _UnderlyingPtr = __MR_CubicBezierCurve3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CubicBezierCurve3d::operator=`.
        public unsafe MR.CubicBezierCurve3d Assign(MR.Const_CubicBezierCurve3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CubicBezierCurve3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CubicBezierCurve3d._Underlying *__MR_CubicBezierCurve3d_AssignFromAnother(_Underlying *_this, MR.CubicBezierCurve3d._Underlying *_other);
            return new(__MR_CubicBezierCurve3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CubicBezierCurve3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CubicBezierCurve3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve3d`/`Const_CubicBezierCurve3d` directly.
    public class _InOptMut_CubicBezierCurve3d
    {
        public CubicBezierCurve3d? Opt;

        public _InOptMut_CubicBezierCurve3d() {}
        public _InOptMut_CubicBezierCurve3d(CubicBezierCurve3d value) {Opt = value;}
        public static implicit operator _InOptMut_CubicBezierCurve3d(CubicBezierCurve3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `CubicBezierCurve3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CubicBezierCurve3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CubicBezierCurve3d`/`Const_CubicBezierCurve3d` to pass it to the function.
    public class _InOptConst_CubicBezierCurve3d
    {
        public Const_CubicBezierCurve3d? Opt;

        public _InOptConst_CubicBezierCurve3d() {}
        public _InOptConst_CubicBezierCurve3d(Const_CubicBezierCurve3d value) {Opt = value;}
        public static implicit operator _InOptConst_CubicBezierCurve3d(Const_CubicBezierCurve3d value) {return new(value);}
    }
}
