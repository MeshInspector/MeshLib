public static partial class MR
{
    /// Represents quadratic function f(x) = a*x*x + b*x + c
    /// Generated from class `MR::Parabolaf`.
    /// This is the const half of the class.
    public class Const_Parabolaf : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Parabolaf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_Destroy", ExactSpelling = true)]
            extern static void __MR_Parabolaf_Destroy(_Underlying *_this);
            __MR_Parabolaf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Parabolaf() {Dispose(false);}

        public unsafe float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_Get_a", ExactSpelling = true)]
                extern static float *__MR_Parabolaf_Get_a(_Underlying *_this);
                return *__MR_Parabolaf_Get_a(_UnderlyingPtr);
            }
        }

        public unsafe float B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_Get_b", ExactSpelling = true)]
                extern static float *__MR_Parabolaf_Get_b(_Underlying *_this);
                return *__MR_Parabolaf_Get_b(_UnderlyingPtr);
            }
        }

        public unsafe float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_Get_c", ExactSpelling = true)]
                extern static float *__MR_Parabolaf_Get_c(_Underlying *_this);
                return *__MR_Parabolaf_Get_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Parabolaf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_DefaultConstruct();
            _UnderlyingPtr = __MR_Parabolaf_DefaultConstruct();
        }

        /// Generated from constructor `MR::Parabolaf::Parabolaf`.
        public unsafe Const_Parabolaf(MR.Const_Parabolaf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_ConstructFromAnother(MR.Parabolaf._Underlying *_other);
            _UnderlyingPtr = __MR_Parabolaf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Parabolaf::Parabolaf`.
        public unsafe Const_Parabolaf(float a, float b, float c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_Construct", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_Construct(float a, float b, float c);
            _UnderlyingPtr = __MR_Parabolaf_Construct(a, b, c);
        }

        /// compute value of quadratic function at any x
        /// Generated from method `MR::Parabolaf::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_call", ExactSpelling = true)]
            extern static float __MR_Parabolaf_call(_Underlying *_this, float x);
            return __MR_Parabolaf_call(_UnderlyingPtr, x);
        }

        /// argument (x) where parabola reaches extremal value: minimum for a > 0, maximum for a < 0
        /// Generated from method `MR::Parabolaf::extremArg`.
        public unsafe float ExtremArg()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_extremArg", ExactSpelling = true)]
            extern static float __MR_Parabolaf_extremArg(_Underlying *_this);
            return __MR_Parabolaf_extremArg(_UnderlyingPtr);
        }

        /// value (y) where parabola reaches extremal value: minimum for a > 0, maximum for a < 0
        /// Generated from method `MR::Parabolaf::extremVal`.
        public unsafe float ExtremVal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_extremVal", ExactSpelling = true)]
            extern static float __MR_Parabolaf_extremVal(_Underlying *_this);
            return __MR_Parabolaf_extremVal(_UnderlyingPtr);
        }
    }

    /// Represents quadratic function f(x) = a*x*x + b*x + c
    /// Generated from class `MR::Parabolaf`.
    /// This is the non-const half of the class.
    public class Parabolaf : Const_Parabolaf
    {
        internal unsafe Parabolaf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_GetMutable_a", ExactSpelling = true)]
                extern static float *__MR_Parabolaf_GetMutable_a(_Underlying *_this);
                return ref *__MR_Parabolaf_GetMutable_a(_UnderlyingPtr);
            }
        }

        public new unsafe ref float B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_GetMutable_b", ExactSpelling = true)]
                extern static float *__MR_Parabolaf_GetMutable_b(_Underlying *_this);
                return ref *__MR_Parabolaf_GetMutable_b(_UnderlyingPtr);
            }
        }

        public new unsafe ref float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_GetMutable_c", ExactSpelling = true)]
                extern static float *__MR_Parabolaf_GetMutable_c(_Underlying *_this);
                return ref *__MR_Parabolaf_GetMutable_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Parabolaf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_DefaultConstruct();
            _UnderlyingPtr = __MR_Parabolaf_DefaultConstruct();
        }

        /// Generated from constructor `MR::Parabolaf::Parabolaf`.
        public unsafe Parabolaf(MR.Const_Parabolaf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_ConstructFromAnother(MR.Parabolaf._Underlying *_other);
            _UnderlyingPtr = __MR_Parabolaf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Parabolaf::Parabolaf`.
        public unsafe Parabolaf(float a, float b, float c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_Construct", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_Construct(float a, float b, float c);
            _UnderlyingPtr = __MR_Parabolaf_Construct(a, b, c);
        }

        /// Generated from method `MR::Parabolaf::operator=`.
        public unsafe MR.Parabolaf Assign(MR.Const_Parabolaf _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolaf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_Parabolaf_AssignFromAnother(_Underlying *_this, MR.Parabolaf._Underlying *_other);
            return new(__MR_Parabolaf_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Parabolaf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Parabolaf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Parabolaf`/`Const_Parabolaf` directly.
    public class _InOptMut_Parabolaf
    {
        public Parabolaf? Opt;

        public _InOptMut_Parabolaf() {}
        public _InOptMut_Parabolaf(Parabolaf value) {Opt = value;}
        public static implicit operator _InOptMut_Parabolaf(Parabolaf value) {return new(value);}
    }

    /// This is used for optional parameters of class `Parabolaf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Parabolaf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Parabolaf`/`Const_Parabolaf` to pass it to the function.
    public class _InOptConst_Parabolaf
    {
        public Const_Parabolaf? Opt;

        public _InOptConst_Parabolaf() {}
        public _InOptConst_Parabolaf(Const_Parabolaf value) {Opt = value;}
        public static implicit operator _InOptConst_Parabolaf(Const_Parabolaf value) {return new(value);}
    }

    /// Represents quadratic function f(x) = a*x*x + b*x + c
    /// Generated from class `MR::Parabolad`.
    /// This is the const half of the class.
    public class Const_Parabolad : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Parabolad(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_Destroy", ExactSpelling = true)]
            extern static void __MR_Parabolad_Destroy(_Underlying *_this);
            __MR_Parabolad_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Parabolad() {Dispose(false);}

        public unsafe double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_Get_a", ExactSpelling = true)]
                extern static double *__MR_Parabolad_Get_a(_Underlying *_this);
                return *__MR_Parabolad_Get_a(_UnderlyingPtr);
            }
        }

        public unsafe double B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_Get_b", ExactSpelling = true)]
                extern static double *__MR_Parabolad_Get_b(_Underlying *_this);
                return *__MR_Parabolad_Get_b(_UnderlyingPtr);
            }
        }

        public unsafe double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_Get_c", ExactSpelling = true)]
                extern static double *__MR_Parabolad_Get_c(_Underlying *_this);
                return *__MR_Parabolad_Get_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Parabolad() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_DefaultConstruct();
            _UnderlyingPtr = __MR_Parabolad_DefaultConstruct();
        }

        /// Generated from constructor `MR::Parabolad::Parabolad`.
        public unsafe Const_Parabolad(MR.Const_Parabolad _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_ConstructFromAnother(MR.Parabolad._Underlying *_other);
            _UnderlyingPtr = __MR_Parabolad_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Parabolad::Parabolad`.
        public unsafe Const_Parabolad(double a, double b, double c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_Construct", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_Construct(double a, double b, double c);
            _UnderlyingPtr = __MR_Parabolad_Construct(a, b, c);
        }

        /// compute value of quadratic function at any x
        /// Generated from method `MR::Parabolad::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_call", ExactSpelling = true)]
            extern static double __MR_Parabolad_call(_Underlying *_this, double x);
            return __MR_Parabolad_call(_UnderlyingPtr, x);
        }

        /// argument (x) where parabola reaches extremal value: minimum for a > 0, maximum for a < 0
        /// Generated from method `MR::Parabolad::extremArg`.
        public unsafe double ExtremArg()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_extremArg", ExactSpelling = true)]
            extern static double __MR_Parabolad_extremArg(_Underlying *_this);
            return __MR_Parabolad_extremArg(_UnderlyingPtr);
        }

        /// value (y) where parabola reaches extremal value: minimum for a > 0, maximum for a < 0
        /// Generated from method `MR::Parabolad::extremVal`.
        public unsafe double ExtremVal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_extremVal", ExactSpelling = true)]
            extern static double __MR_Parabolad_extremVal(_Underlying *_this);
            return __MR_Parabolad_extremVal(_UnderlyingPtr);
        }
    }

    /// Represents quadratic function f(x) = a*x*x + b*x + c
    /// Generated from class `MR::Parabolad`.
    /// This is the non-const half of the class.
    public class Parabolad : Const_Parabolad
    {
        internal unsafe Parabolad(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_GetMutable_a", ExactSpelling = true)]
                extern static double *__MR_Parabolad_GetMutable_a(_Underlying *_this);
                return ref *__MR_Parabolad_GetMutable_a(_UnderlyingPtr);
            }
        }

        public new unsafe ref double B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_GetMutable_b", ExactSpelling = true)]
                extern static double *__MR_Parabolad_GetMutable_b(_Underlying *_this);
                return ref *__MR_Parabolad_GetMutable_b(_UnderlyingPtr);
            }
        }

        public new unsafe ref double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_GetMutable_c", ExactSpelling = true)]
                extern static double *__MR_Parabolad_GetMutable_c(_Underlying *_this);
                return ref *__MR_Parabolad_GetMutable_c(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Parabolad() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_DefaultConstruct();
            _UnderlyingPtr = __MR_Parabolad_DefaultConstruct();
        }

        /// Generated from constructor `MR::Parabolad::Parabolad`.
        public unsafe Parabolad(MR.Const_Parabolad _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_ConstructFromAnother(MR.Parabolad._Underlying *_other);
            _UnderlyingPtr = __MR_Parabolad_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Parabolad::Parabolad`.
        public unsafe Parabolad(double a, double b, double c) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_Construct", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_Construct(double a, double b, double c);
            _UnderlyingPtr = __MR_Parabolad_Construct(a, b, c);
        }

        /// Generated from method `MR::Parabolad::operator=`.
        public unsafe MR.Parabolad Assign(MR.Const_Parabolad _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Parabolad_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_Parabolad_AssignFromAnother(_Underlying *_this, MR.Parabolad._Underlying *_other);
            return new(__MR_Parabolad_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Parabolad` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Parabolad`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Parabolad`/`Const_Parabolad` directly.
    public class _InOptMut_Parabolad
    {
        public Parabolad? Opt;

        public _InOptMut_Parabolad() {}
        public _InOptMut_Parabolad(Parabolad value) {Opt = value;}
        public static implicit operator _InOptMut_Parabolad(Parabolad value) {return new(value);}
    }

    /// This is used for optional parameters of class `Parabolad` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Parabolad`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Parabolad`/`Const_Parabolad` to pass it to the function.
    public class _InOptConst_Parabolad
    {
        public Const_Parabolad? Opt;

        public _InOptConst_Parabolad() {}
        public _InOptConst_Parabolad(Const_Parabolad value) {Opt = value;}
        public static implicit operator _InOptConst_Parabolad(Const_Parabolad value) {return new(value);}
    }
}
