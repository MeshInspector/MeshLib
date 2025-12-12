public static partial class MR
{
    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm2f`.
    /// This is the const half of the class.
    public class Const_QuadraticForm2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_QuadraticForm2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_Destroy", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2f_Destroy(_Underlying *_this);
            __MR_QuadraticForm2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_QuadraticForm2f() {Dispose(false);}

        public unsafe MR.Const_SymMatrix2f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_Get_A", ExactSpelling = true)]
                extern static MR.Const_SymMatrix2f._Underlying *__MR_QuadraticForm2f_Get_A(_Underlying *_this);
                return new(__MR_QuadraticForm2f_Get_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_Get_c", ExactSpelling = true)]
                extern static float *__MR_QuadraticForm2f_Get_c(_Underlying *_this);
                return *__MR_QuadraticForm2f_Get_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_QuadraticForm2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm2f_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm2f` elementwise.
        public unsafe Const_QuadraticForm2f(MR.Const_SymMatrix2f A, float c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_ConstructFrom(MR.SymMatrix2f._Underlying *A, float c);
            _UnderlyingPtr = __MR_QuadraticForm2f_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm2f::QuadraticForm2f`.
        public unsafe Const_QuadraticForm2f(MR.Const_QuadraticForm2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_ConstructFromAnother(MR.QuadraticForm2f._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// evaluates the function at given x
        /// Generated from method `MR::QuadraticForm2f::eval`.
        public unsafe float Eval(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_eval", ExactSpelling = true)]
            extern static float __MR_QuadraticForm2f_eval(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_QuadraticForm2f_eval(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm2f`.
    /// This is the non-const half of the class.
    public class QuadraticForm2f : Const_QuadraticForm2f
    {
        internal unsafe QuadraticForm2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.SymMatrix2f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_GetMutable_A", ExactSpelling = true)]
                extern static MR.SymMatrix2f._Underlying *__MR_QuadraticForm2f_GetMutable_A(_Underlying *_this);
                return new(__MR_QuadraticForm2f_GetMutable_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_GetMutable_c", ExactSpelling = true)]
                extern static float *__MR_QuadraticForm2f_GetMutable_c(_Underlying *_this);
                return ref *__MR_QuadraticForm2f_GetMutable_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe QuadraticForm2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm2f_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm2f` elementwise.
        public unsafe QuadraticForm2f(MR.Const_SymMatrix2f A, float c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_ConstructFrom(MR.SymMatrix2f._Underlying *A, float c);
            _UnderlyingPtr = __MR_QuadraticForm2f_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm2f::QuadraticForm2f`.
        public unsafe QuadraticForm2f(MR.Const_QuadraticForm2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_ConstructFromAnother(MR.QuadraticForm2f._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm2f::operator=`.
        public unsafe MR.QuadraticForm2f Assign(MR.Const_QuadraticForm2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm2f._Underlying *__MR_QuadraticForm2f_AssignFromAnother(_Underlying *_this, MR.QuadraticForm2f._Underlying *_other);
            return new(__MR_QuadraticForm2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// adds to this squared distance to the origin point
        /// Generated from method `MR::QuadraticForm2f::addDistToOrigin`.
        public unsafe void AddDistToOrigin(float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_addDistToOrigin", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2f_addDistToOrigin(_Underlying *_this, float weight);
            __MR_QuadraticForm2f_addDistToOrigin(_UnderlyingPtr, weight);
        }

        /// adds to this squared distance to plane passing via origin with given unit normal
        /// Generated from method `MR::QuadraticForm2f::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector2f planeUnitNormal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_addDistToPlane_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2f_addDistToPlane_1(_Underlying *_this, MR.Const_Vector2f._Underlying *planeUnitNormal);
            __MR_QuadraticForm2f_addDistToPlane_1(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm2f::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector2f planeUnitNormal, float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_addDistToPlane_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2f_addDistToPlane_2(_Underlying *_this, MR.Const_Vector2f._Underlying *planeUnitNormal, float weight);
            __MR_QuadraticForm2f_addDistToPlane_2(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr, weight);
        }

        /// adds to this squared distance to line passing via origin with given unit direction
        /// Generated from method `MR::QuadraticForm2f::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector2f lineUnitDir)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_addDistToLine_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2f_addDistToLine_1(_Underlying *_this, MR.Const_Vector2f._Underlying *lineUnitDir);
            __MR_QuadraticForm2f_addDistToLine_1(_UnderlyingPtr, lineUnitDir._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm2f::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector2f lineUnitDir, float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2f_addDistToLine_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2f_addDistToLine_2(_Underlying *_this, MR.Const_Vector2f._Underlying *lineUnitDir, float weight);
            __MR_QuadraticForm2f_addDistToLine_2(_UnderlyingPtr, lineUnitDir._UnderlyingPtr, weight);
        }
    }

    /// This is used for optional parameters of class `QuadraticForm2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_QuadraticForm2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm2f`/`Const_QuadraticForm2f` directly.
    public class _InOptMut_QuadraticForm2f
    {
        public QuadraticForm2f? Opt;

        public _InOptMut_QuadraticForm2f() {}
        public _InOptMut_QuadraticForm2f(QuadraticForm2f value) {Opt = value;}
        public static implicit operator _InOptMut_QuadraticForm2f(QuadraticForm2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `QuadraticForm2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_QuadraticForm2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm2f`/`Const_QuadraticForm2f` to pass it to the function.
    public class _InOptConst_QuadraticForm2f
    {
        public Const_QuadraticForm2f? Opt;

        public _InOptConst_QuadraticForm2f() {}
        public _InOptConst_QuadraticForm2f(Const_QuadraticForm2f value) {Opt = value;}
        public static implicit operator _InOptConst_QuadraticForm2f(Const_QuadraticForm2f value) {return new(value);}
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm2d`.
    /// This is the const half of the class.
    public class Const_QuadraticForm2d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_QuadraticForm2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_Destroy", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2d_Destroy(_Underlying *_this);
            __MR_QuadraticForm2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_QuadraticForm2d() {Dispose(false);}

        public unsafe MR.Const_SymMatrix2d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_Get_A", ExactSpelling = true)]
                extern static MR.Const_SymMatrix2d._Underlying *__MR_QuadraticForm2d_Get_A(_Underlying *_this);
                return new(__MR_QuadraticForm2d_Get_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_Get_c", ExactSpelling = true)]
                extern static double *__MR_QuadraticForm2d_Get_c(_Underlying *_this);
                return *__MR_QuadraticForm2d_Get_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_QuadraticForm2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm2d_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm2d` elementwise.
        public unsafe Const_QuadraticForm2d(MR.Const_SymMatrix2d A, double c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_ConstructFrom(MR.SymMatrix2d._Underlying *A, double c);
            _UnderlyingPtr = __MR_QuadraticForm2d_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm2d::QuadraticForm2d`.
        public unsafe Const_QuadraticForm2d(MR.Const_QuadraticForm2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_ConstructFromAnother(MR.QuadraticForm2d._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// evaluates the function at given x
        /// Generated from method `MR::QuadraticForm2d::eval`.
        public unsafe double Eval(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_eval", ExactSpelling = true)]
            extern static double __MR_QuadraticForm2d_eval(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_QuadraticForm2d_eval(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm2d`.
    /// This is the non-const half of the class.
    public class QuadraticForm2d : Const_QuadraticForm2d
    {
        internal unsafe QuadraticForm2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.SymMatrix2d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_GetMutable_A", ExactSpelling = true)]
                extern static MR.SymMatrix2d._Underlying *__MR_QuadraticForm2d_GetMutable_A(_Underlying *_this);
                return new(__MR_QuadraticForm2d_GetMutable_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_GetMutable_c", ExactSpelling = true)]
                extern static double *__MR_QuadraticForm2d_GetMutable_c(_Underlying *_this);
                return ref *__MR_QuadraticForm2d_GetMutable_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe QuadraticForm2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm2d_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm2d` elementwise.
        public unsafe QuadraticForm2d(MR.Const_SymMatrix2d A, double c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_ConstructFrom(MR.SymMatrix2d._Underlying *A, double c);
            _UnderlyingPtr = __MR_QuadraticForm2d_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm2d::QuadraticForm2d`.
        public unsafe QuadraticForm2d(MR.Const_QuadraticForm2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_ConstructFromAnother(MR.QuadraticForm2d._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm2d::operator=`.
        public unsafe MR.QuadraticForm2d Assign(MR.Const_QuadraticForm2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm2d._Underlying *__MR_QuadraticForm2d_AssignFromAnother(_Underlying *_this, MR.QuadraticForm2d._Underlying *_other);
            return new(__MR_QuadraticForm2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// adds to this squared distance to the origin point
        /// Generated from method `MR::QuadraticForm2d::addDistToOrigin`.
        public unsafe void AddDistToOrigin(double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_addDistToOrigin", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2d_addDistToOrigin(_Underlying *_this, double weight);
            __MR_QuadraticForm2d_addDistToOrigin(_UnderlyingPtr, weight);
        }

        /// adds to this squared distance to plane passing via origin with given unit normal
        /// Generated from method `MR::QuadraticForm2d::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector2d planeUnitNormal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_addDistToPlane_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2d_addDistToPlane_1(_Underlying *_this, MR.Const_Vector2d._Underlying *planeUnitNormal);
            __MR_QuadraticForm2d_addDistToPlane_1(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm2d::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector2d planeUnitNormal, double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_addDistToPlane_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2d_addDistToPlane_2(_Underlying *_this, MR.Const_Vector2d._Underlying *planeUnitNormal, double weight);
            __MR_QuadraticForm2d_addDistToPlane_2(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr, weight);
        }

        /// adds to this squared distance to line passing via origin with given unit direction
        /// Generated from method `MR::QuadraticForm2d::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector2d lineUnitDir)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_addDistToLine_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2d_addDistToLine_1(_Underlying *_this, MR.Const_Vector2d._Underlying *lineUnitDir);
            __MR_QuadraticForm2d_addDistToLine_1(_UnderlyingPtr, lineUnitDir._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm2d::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector2d lineUnitDir, double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm2d_addDistToLine_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm2d_addDistToLine_2(_Underlying *_this, MR.Const_Vector2d._Underlying *lineUnitDir, double weight);
            __MR_QuadraticForm2d_addDistToLine_2(_UnderlyingPtr, lineUnitDir._UnderlyingPtr, weight);
        }
    }

    /// This is used for optional parameters of class `QuadraticForm2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_QuadraticForm2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm2d`/`Const_QuadraticForm2d` directly.
    public class _InOptMut_QuadraticForm2d
    {
        public QuadraticForm2d? Opt;

        public _InOptMut_QuadraticForm2d() {}
        public _InOptMut_QuadraticForm2d(QuadraticForm2d value) {Opt = value;}
        public static implicit operator _InOptMut_QuadraticForm2d(QuadraticForm2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `QuadraticForm2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_QuadraticForm2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm2d`/`Const_QuadraticForm2d` to pass it to the function.
    public class _InOptConst_QuadraticForm2d
    {
        public Const_QuadraticForm2d? Opt;

        public _InOptConst_QuadraticForm2d() {}
        public _InOptConst_QuadraticForm2d(Const_QuadraticForm2d value) {Opt = value;}
        public static implicit operator _InOptConst_QuadraticForm2d(Const_QuadraticForm2d value) {return new(value);}
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm3f`.
    /// This is the const half of the class.
    public class Const_QuadraticForm3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_QuadraticForm3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_Destroy", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3f_Destroy(_Underlying *_this);
            __MR_QuadraticForm3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_QuadraticForm3f() {Dispose(false);}

        public unsafe MR.Const_SymMatrix3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_Get_A", ExactSpelling = true)]
                extern static MR.Const_SymMatrix3f._Underlying *__MR_QuadraticForm3f_Get_A(_Underlying *_this);
                return new(__MR_QuadraticForm3f_Get_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_Get_c", ExactSpelling = true)]
                extern static float *__MR_QuadraticForm3f_Get_c(_Underlying *_this);
                return *__MR_QuadraticForm3f_Get_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_QuadraticForm3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm3f_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm3f` elementwise.
        public unsafe Const_QuadraticForm3f(MR.Const_SymMatrix3f A, float c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_ConstructFrom(MR.SymMatrix3f._Underlying *A, float c);
            _UnderlyingPtr = __MR_QuadraticForm3f_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm3f::QuadraticForm3f`.
        public unsafe Const_QuadraticForm3f(MR.Const_QuadraticForm3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_ConstructFromAnother(MR.QuadraticForm3f._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// evaluates the function at given x
        /// Generated from method `MR::QuadraticForm3f::eval`.
        public unsafe float Eval(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_eval", ExactSpelling = true)]
            extern static float __MR_QuadraticForm3f_eval(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_QuadraticForm3f_eval(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm3f`.
    /// This is the non-const half of the class.
    public class QuadraticForm3f : Const_QuadraticForm3f
    {
        internal unsafe QuadraticForm3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.SymMatrix3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_GetMutable_A", ExactSpelling = true)]
                extern static MR.SymMatrix3f._Underlying *__MR_QuadraticForm3f_GetMutable_A(_Underlying *_this);
                return new(__MR_QuadraticForm3f_GetMutable_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_GetMutable_c", ExactSpelling = true)]
                extern static float *__MR_QuadraticForm3f_GetMutable_c(_Underlying *_this);
                return ref *__MR_QuadraticForm3f_GetMutable_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe QuadraticForm3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm3f_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm3f` elementwise.
        public unsafe QuadraticForm3f(MR.Const_SymMatrix3f A, float c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_ConstructFrom(MR.SymMatrix3f._Underlying *A, float c);
            _UnderlyingPtr = __MR_QuadraticForm3f_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm3f::QuadraticForm3f`.
        public unsafe QuadraticForm3f(MR.Const_QuadraticForm3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_ConstructFromAnother(MR.QuadraticForm3f._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm3f::operator=`.
        public unsafe MR.QuadraticForm3f Assign(MR.Const_QuadraticForm3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_QuadraticForm3f_AssignFromAnother(_Underlying *_this, MR.QuadraticForm3f._Underlying *_other);
            return new(__MR_QuadraticForm3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// adds to this squared distance to the origin point
        /// Generated from method `MR::QuadraticForm3f::addDistToOrigin`.
        public unsafe void AddDistToOrigin(float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_addDistToOrigin", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3f_addDistToOrigin(_Underlying *_this, float weight);
            __MR_QuadraticForm3f_addDistToOrigin(_UnderlyingPtr, weight);
        }

        /// adds to this squared distance to plane passing via origin with given unit normal
        /// Generated from method `MR::QuadraticForm3f::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector3f planeUnitNormal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_addDistToPlane_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3f_addDistToPlane_1(_Underlying *_this, MR.Const_Vector3f._Underlying *planeUnitNormal);
            __MR_QuadraticForm3f_addDistToPlane_1(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm3f::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector3f planeUnitNormal, float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_addDistToPlane_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3f_addDistToPlane_2(_Underlying *_this, MR.Const_Vector3f._Underlying *planeUnitNormal, float weight);
            __MR_QuadraticForm3f_addDistToPlane_2(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr, weight);
        }

        /// adds to this squared distance to line passing via origin with given unit direction
        /// Generated from method `MR::QuadraticForm3f::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector3f lineUnitDir)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_addDistToLine_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3f_addDistToLine_1(_Underlying *_this, MR.Const_Vector3f._Underlying *lineUnitDir);
            __MR_QuadraticForm3f_addDistToLine_1(_UnderlyingPtr, lineUnitDir._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm3f::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector3f lineUnitDir, float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3f_addDistToLine_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3f_addDistToLine_2(_Underlying *_this, MR.Const_Vector3f._Underlying *lineUnitDir, float weight);
            __MR_QuadraticForm3f_addDistToLine_2(_UnderlyingPtr, lineUnitDir._UnderlyingPtr, weight);
        }
    }

    /// This is used for optional parameters of class `QuadraticForm3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_QuadraticForm3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm3f`/`Const_QuadraticForm3f` directly.
    public class _InOptMut_QuadraticForm3f
    {
        public QuadraticForm3f? Opt;

        public _InOptMut_QuadraticForm3f() {}
        public _InOptMut_QuadraticForm3f(QuadraticForm3f value) {Opt = value;}
        public static implicit operator _InOptMut_QuadraticForm3f(QuadraticForm3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `QuadraticForm3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_QuadraticForm3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm3f`/`Const_QuadraticForm3f` to pass it to the function.
    public class _InOptConst_QuadraticForm3f
    {
        public Const_QuadraticForm3f? Opt;

        public _InOptConst_QuadraticForm3f() {}
        public _InOptConst_QuadraticForm3f(Const_QuadraticForm3f value) {Opt = value;}
        public static implicit operator _InOptConst_QuadraticForm3f(Const_QuadraticForm3f value) {return new(value);}
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm3d`.
    /// This is the const half of the class.
    public class Const_QuadraticForm3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_QuadraticForm3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_Destroy", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3d_Destroy(_Underlying *_this);
            __MR_QuadraticForm3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_QuadraticForm3d() {Dispose(false);}

        public unsafe MR.Const_SymMatrix3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_Get_A", ExactSpelling = true)]
                extern static MR.Const_SymMatrix3d._Underlying *__MR_QuadraticForm3d_Get_A(_Underlying *_this);
                return new(__MR_QuadraticForm3d_Get_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_Get_c", ExactSpelling = true)]
                extern static double *__MR_QuadraticForm3d_Get_c(_Underlying *_this);
                return *__MR_QuadraticForm3d_Get_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_QuadraticForm3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm3d_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm3d` elementwise.
        public unsafe Const_QuadraticForm3d(MR.Const_SymMatrix3d A, double c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_ConstructFrom(MR.SymMatrix3d._Underlying *A, double c);
            _UnderlyingPtr = __MR_QuadraticForm3d_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm3d::QuadraticForm3d`.
        public unsafe Const_QuadraticForm3d(MR.Const_QuadraticForm3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_ConstructFromAnother(MR.QuadraticForm3d._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// evaluates the function at given x
        /// Generated from method `MR::QuadraticForm3d::eval`.
        public unsafe double Eval(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_eval", ExactSpelling = true)]
            extern static double __MR_QuadraticForm3d_eval(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_QuadraticForm3d_eval(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// quadratic form: f = x^T A x + c
    /// Generated from class `MR::QuadraticForm3d`.
    /// This is the non-const half of the class.
    public class QuadraticForm3d : Const_QuadraticForm3d
    {
        internal unsafe QuadraticForm3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.SymMatrix3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_GetMutable_A", ExactSpelling = true)]
                extern static MR.SymMatrix3d._Underlying *__MR_QuadraticForm3d_GetMutable_A(_Underlying *_this);
                return new(__MR_QuadraticForm3d_GetMutable_A(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_GetMutable_c", ExactSpelling = true)]
                extern static double *__MR_QuadraticForm3d_GetMutable_c(_Underlying *_this);
                return ref *__MR_QuadraticForm3d_GetMutable_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe QuadraticForm3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadraticForm3d_DefaultConstruct();
        }

        /// Constructs `MR::QuadraticForm3d` elementwise.
        public unsafe QuadraticForm3d(MR.Const_SymMatrix3d A, double c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_ConstructFrom", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_ConstructFrom(MR.SymMatrix3d._Underlying *A, double c);
            _UnderlyingPtr = __MR_QuadraticForm3d_ConstructFrom(A._UnderlyingPtr, c);
        }

        /// Generated from constructor `MR::QuadraticForm3d::QuadraticForm3d`.
        public unsafe QuadraticForm3d(MR.Const_QuadraticForm3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_ConstructFromAnother(MR.QuadraticForm3d._Underlying *_other);
            _UnderlyingPtr = __MR_QuadraticForm3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm3d::operator=`.
        public unsafe MR.QuadraticForm3d Assign(MR.Const_QuadraticForm3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.QuadraticForm3d._Underlying *__MR_QuadraticForm3d_AssignFromAnother(_Underlying *_this, MR.QuadraticForm3d._Underlying *_other);
            return new(__MR_QuadraticForm3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// adds to this squared distance to the origin point
        /// Generated from method `MR::QuadraticForm3d::addDistToOrigin`.
        public unsafe void AddDistToOrigin(double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_addDistToOrigin", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3d_addDistToOrigin(_Underlying *_this, double weight);
            __MR_QuadraticForm3d_addDistToOrigin(_UnderlyingPtr, weight);
        }

        /// adds to this squared distance to plane passing via origin with given unit normal
        /// Generated from method `MR::QuadraticForm3d::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector3d planeUnitNormal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_addDistToPlane_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3d_addDistToPlane_1(_Underlying *_this, MR.Const_Vector3d._Underlying *planeUnitNormal);
            __MR_QuadraticForm3d_addDistToPlane_1(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm3d::addDistToPlane`.
        public unsafe void AddDistToPlane(MR.Const_Vector3d planeUnitNormal, double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_addDistToPlane_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3d_addDistToPlane_2(_Underlying *_this, MR.Const_Vector3d._Underlying *planeUnitNormal, double weight);
            __MR_QuadraticForm3d_addDistToPlane_2(_UnderlyingPtr, planeUnitNormal._UnderlyingPtr, weight);
        }

        /// adds to this squared distance to line passing via origin with given unit direction
        /// Generated from method `MR::QuadraticForm3d::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector3d lineUnitDir)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_addDistToLine_1", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3d_addDistToLine_1(_Underlying *_this, MR.Const_Vector3d._Underlying *lineUnitDir);
            __MR_QuadraticForm3d_addDistToLine_1(_UnderlyingPtr, lineUnitDir._UnderlyingPtr);
        }

        /// Generated from method `MR::QuadraticForm3d::addDistToLine`.
        public unsafe void AddDistToLine(MR.Const_Vector3d lineUnitDir, double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadraticForm3d_addDistToLine_2", ExactSpelling = true)]
            extern static void __MR_QuadraticForm3d_addDistToLine_2(_Underlying *_this, MR.Const_Vector3d._Underlying *lineUnitDir, double weight);
            __MR_QuadraticForm3d_addDistToLine_2(_UnderlyingPtr, lineUnitDir._UnderlyingPtr, weight);
        }
    }

    /// This is used for optional parameters of class `QuadraticForm3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_QuadraticForm3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm3d`/`Const_QuadraticForm3d` directly.
    public class _InOptMut_QuadraticForm3d
    {
        public QuadraticForm3d? Opt;

        public _InOptMut_QuadraticForm3d() {}
        public _InOptMut_QuadraticForm3d(QuadraticForm3d value) {Opt = value;}
        public static implicit operator _InOptMut_QuadraticForm3d(QuadraticForm3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `QuadraticForm3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_QuadraticForm3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadraticForm3d`/`Const_QuadraticForm3d` to pass it to the function.
    public class _InOptConst_QuadraticForm3d
    {
        public Const_QuadraticForm3d? Opt;

        public _InOptConst_QuadraticForm3d() {}
        public _InOptConst_QuadraticForm3d(Const_QuadraticForm3d value) {Opt = value;}
        public static implicit operator _InOptConst_QuadraticForm3d(Const_QuadraticForm3d value) {return new(value);}
    }
}
