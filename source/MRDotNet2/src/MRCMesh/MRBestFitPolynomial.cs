public static partial class MR
{
    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 0>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_0 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_0(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_0_Destroy(_Underlying *_this);
            __MR_Polynomial_float_0_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_0() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_0_Get_n();
                return *__MR_Polynomial_float_0_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_0() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_0_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_0_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 0>::Polynomial`.
        public unsafe Const_Polynomial_Float_0(MR._ByValue_Polynomial_Float_0 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_0_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_0._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_0_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 0>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_0_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_0_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 0>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_0> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_0_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_0(__MR_Polynomial_float_0_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 0>::intervalMin`.
        public unsafe float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_intervalMin", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_0_intervalMin(_Underlying *_this, float a, float b);
            return __MR_Polynomial_float_0_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 0>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_0 : Const_Polynomial_Float_0
    {
        internal unsafe Polynomial_Float_0(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_0() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_0_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_0_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 0>::Polynomial`.
        public unsafe Polynomial_Float_0(MR._ByValue_Polynomial_Float_0 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_0_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_0._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_0_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 0>::operator=`.
        public unsafe MR.Polynomial_Float_0 Assign(MR._ByValue_Polynomial_Float_0 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_0_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_0_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_0._Underlying *_other);
            return new(__MR_Polynomial_float_0_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_0` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_0`/`Const_Polynomial_Float_0` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_0
    {
        internal readonly Const_Polynomial_Float_0? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_0() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_0(Const_Polynomial_Float_0 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_0(Const_Polynomial_Float_0 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_0(MR.Misc._Moved<Polynomial_Float_0> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_0(MR.Misc._Moved<Polynomial_Float_0> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_0` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_0`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_0`/`Const_Polynomial_Float_0` directly.
    public class _InOptMut_Polynomial_Float_0
    {
        public Polynomial_Float_0? Opt;

        public _InOptMut_Polynomial_Float_0() {}
        public _InOptMut_Polynomial_Float_0(Polynomial_Float_0 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_0(Polynomial_Float_0 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_0` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_0`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_0`/`Const_Polynomial_Float_0` to pass it to the function.
    public class _InOptConst_Polynomial_Float_0
    {
        public Const_Polynomial_Float_0? Opt;

        public _InOptConst_Polynomial_Float_0() {}
        public _InOptConst_Polynomial_Float_0(Const_Polynomial_Float_0 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_0(Const_Polynomial_Float_0 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 1>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_1 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_1(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_1_Destroy(_Underlying *_this);
            __MR_Polynomial_float_1_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_1() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_1_Get_n();
                return *__MR_Polynomial_float_1_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_1() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_1._Underlying *__MR_Polynomial_float_1_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_1_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 1>::Polynomial`.
        public unsafe Const_Polynomial_Float_1(MR._ByValue_Polynomial_Float_1 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_1._Underlying *__MR_Polynomial_float_1_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_1._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_1_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 1>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_1_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_1_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 1>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Float> Solve(float tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Float._Underlying *__MR_Polynomial_float_1_solve(_Underlying *_this, float tol);
            return MR.Misc.Move(new MR.Std.Vector_Float(__MR_Polynomial_float_1_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 1>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_0> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_0._Underlying *__MR_Polynomial_float_1_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_0(__MR_Polynomial_float_1_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 1>::intervalMin`.
        public unsafe float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_intervalMin", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_1_intervalMin(_Underlying *_this, float a, float b);
            return __MR_Polynomial_float_1_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 1>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_1 : Const_Polynomial_Float_1
    {
        internal unsafe Polynomial_Float_1(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_1() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_1._Underlying *__MR_Polynomial_float_1_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_1_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 1>::Polynomial`.
        public unsafe Polynomial_Float_1(MR._ByValue_Polynomial_Float_1 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_1._Underlying *__MR_Polynomial_float_1_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_1._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_1_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 1>::operator=`.
        public unsafe MR.Polynomial_Float_1 Assign(MR._ByValue_Polynomial_Float_1 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_1_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_1._Underlying *__MR_Polynomial_float_1_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_1._Underlying *_other);
            return new(__MR_Polynomial_float_1_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_1` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_1`/`Const_Polynomial_Float_1` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_1
    {
        internal readonly Const_Polynomial_Float_1? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_1() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_1(Const_Polynomial_Float_1 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_1(Const_Polynomial_Float_1 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_1(MR.Misc._Moved<Polynomial_Float_1> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_1(MR.Misc._Moved<Polynomial_Float_1> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_1` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_1`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_1`/`Const_Polynomial_Float_1` directly.
    public class _InOptMut_Polynomial_Float_1
    {
        public Polynomial_Float_1? Opt;

        public _InOptMut_Polynomial_Float_1() {}
        public _InOptMut_Polynomial_Float_1(Polynomial_Float_1 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_1(Polynomial_Float_1 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_1` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_1`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_1`/`Const_Polynomial_Float_1` to pass it to the function.
    public class _InOptConst_Polynomial_Float_1
    {
        public Const_Polynomial_Float_1? Opt;

        public _InOptConst_Polynomial_Float_1() {}
        public _InOptConst_Polynomial_Float_1(Const_Polynomial_Float_1 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_1(Const_Polynomial_Float_1 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 2>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_2_Destroy(_Underlying *_this);
            __MR_Polynomial_float_2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_2() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_2_Get_n();
                return *__MR_Polynomial_float_2_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_2._Underlying *__MR_Polynomial_float_2_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_2_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 2>::Polynomial`.
        public unsafe Const_Polynomial_Float_2(MR._ByValue_Polynomial_Float_2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_2._Underlying *__MR_Polynomial_float_2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_2._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 2>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_2_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_2_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 2>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Float> Solve(float tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Float._Underlying *__MR_Polynomial_float_2_solve(_Underlying *_this, float tol);
            return MR.Misc.Move(new MR.Std.Vector_Float(__MR_Polynomial_float_2_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 2>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_1> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_1._Underlying *__MR_Polynomial_float_2_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_1(__MR_Polynomial_float_2_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 2>::intervalMin`.
        public unsafe float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_intervalMin", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_2_intervalMin(_Underlying *_this, float a, float b);
            return __MR_Polynomial_float_2_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 2>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_2 : Const_Polynomial_Float_2
    {
        internal unsafe Polynomial_Float_2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_2._Underlying *__MR_Polynomial_float_2_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_2_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 2>::Polynomial`.
        public unsafe Polynomial_Float_2(MR._ByValue_Polynomial_Float_2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_2._Underlying *__MR_Polynomial_float_2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_2._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 2>::operator=`.
        public unsafe MR.Polynomial_Float_2 Assign(MR._ByValue_Polynomial_Float_2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_2._Underlying *__MR_Polynomial_float_2_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_2._Underlying *_other);
            return new(__MR_Polynomial_float_2_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_2` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_2`/`Const_Polynomial_Float_2` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_2
    {
        internal readonly Const_Polynomial_Float_2? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_2() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_2(Const_Polynomial_Float_2 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_2(Const_Polynomial_Float_2 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_2(MR.Misc._Moved<Polynomial_Float_2> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_2(MR.Misc._Moved<Polynomial_Float_2> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_2`/`Const_Polynomial_Float_2` directly.
    public class _InOptMut_Polynomial_Float_2
    {
        public Polynomial_Float_2? Opt;

        public _InOptMut_Polynomial_Float_2() {}
        public _InOptMut_Polynomial_Float_2(Polynomial_Float_2 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_2(Polynomial_Float_2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_2`/`Const_Polynomial_Float_2` to pass it to the function.
    public class _InOptConst_Polynomial_Float_2
    {
        public Const_Polynomial_Float_2? Opt;

        public _InOptConst_Polynomial_Float_2() {}
        public _InOptConst_Polynomial_Float_2(Const_Polynomial_Float_2 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_2(Const_Polynomial_Float_2 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 3>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_3 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_3_Destroy(_Underlying *_this);
            __MR_Polynomial_float_3_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_3() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_3_Get_n();
                return *__MR_Polynomial_float_3_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_3._Underlying *__MR_Polynomial_float_3_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_3_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 3>::Polynomial`.
        public unsafe Const_Polynomial_Float_3(MR._ByValue_Polynomial_Float_3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_3._Underlying *__MR_Polynomial_float_3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_3._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 3>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_3_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_3_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 3>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Float> Solve(float tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Float._Underlying *__MR_Polynomial_float_3_solve(_Underlying *_this, float tol);
            return MR.Misc.Move(new MR.Std.Vector_Float(__MR_Polynomial_float_3_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 3>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_2> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_2._Underlying *__MR_Polynomial_float_3_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_2(__MR_Polynomial_float_3_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 3>::intervalMin`.
        public unsafe float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_intervalMin", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_3_intervalMin(_Underlying *_this, float a, float b);
            return __MR_Polynomial_float_3_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 3>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_3 : Const_Polynomial_Float_3
    {
        internal unsafe Polynomial_Float_3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_3._Underlying *__MR_Polynomial_float_3_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_3_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 3>::Polynomial`.
        public unsafe Polynomial_Float_3(MR._ByValue_Polynomial_Float_3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_3._Underlying *__MR_Polynomial_float_3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_3._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 3>::operator=`.
        public unsafe MR.Polynomial_Float_3 Assign(MR._ByValue_Polynomial_Float_3 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_3_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_3._Underlying *__MR_Polynomial_float_3_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_3._Underlying *_other);
            return new(__MR_Polynomial_float_3_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_3` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_3`/`Const_Polynomial_Float_3` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_3
    {
        internal readonly Const_Polynomial_Float_3? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_3() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_3(Const_Polynomial_Float_3 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_3(Const_Polynomial_Float_3 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_3(MR.Misc._Moved<Polynomial_Float_3> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_3(MR.Misc._Moved<Polynomial_Float_3> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_3` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_3`/`Const_Polynomial_Float_3` directly.
    public class _InOptMut_Polynomial_Float_3
    {
        public Polynomial_Float_3? Opt;

        public _InOptMut_Polynomial_Float_3() {}
        public _InOptMut_Polynomial_Float_3(Polynomial_Float_3 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_3(Polynomial_Float_3 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_3` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_3`/`Const_Polynomial_Float_3` to pass it to the function.
    public class _InOptConst_Polynomial_Float_3
    {
        public Const_Polynomial_Float_3? Opt;

        public _InOptConst_Polynomial_Float_3() {}
        public _InOptConst_Polynomial_Float_3(Const_Polynomial_Float_3 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_3(Const_Polynomial_Float_3 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 4>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_4 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_4(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_4_Destroy(_Underlying *_this);
            __MR_Polynomial_float_4_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_4() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_4_Get_n();
                return *__MR_Polynomial_float_4_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_4() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_4._Underlying *__MR_Polynomial_float_4_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_4_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 4>::Polynomial`.
        public unsafe Const_Polynomial_Float_4(MR._ByValue_Polynomial_Float_4 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_4._Underlying *__MR_Polynomial_float_4_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_4._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_4_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 4>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_4_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_4_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 4>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Float> Solve(float tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Float._Underlying *__MR_Polynomial_float_4_solve(_Underlying *_this, float tol);
            return MR.Misc.Move(new MR.Std.Vector_Float(__MR_Polynomial_float_4_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 4>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_3> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_3._Underlying *__MR_Polynomial_float_4_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_3(__MR_Polynomial_float_4_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 4>::intervalMin`.
        public unsafe float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_intervalMin", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_4_intervalMin(_Underlying *_this, float a, float b);
            return __MR_Polynomial_float_4_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 4>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_4 : Const_Polynomial_Float_4
    {
        internal unsafe Polynomial_Float_4(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_4() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_4._Underlying *__MR_Polynomial_float_4_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_4_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 4>::Polynomial`.
        public unsafe Polynomial_Float_4(MR._ByValue_Polynomial_Float_4 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_4._Underlying *__MR_Polynomial_float_4_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_4._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_4_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 4>::operator=`.
        public unsafe MR.Polynomial_Float_4 Assign(MR._ByValue_Polynomial_Float_4 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_4_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_4._Underlying *__MR_Polynomial_float_4_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_4._Underlying *_other);
            return new(__MR_Polynomial_float_4_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_4` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_4`/`Const_Polynomial_Float_4` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_4
    {
        internal readonly Const_Polynomial_Float_4? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_4() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_4(Const_Polynomial_Float_4 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_4(Const_Polynomial_Float_4 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_4(MR.Misc._Moved<Polynomial_Float_4> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_4(MR.Misc._Moved<Polynomial_Float_4> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_4` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_4`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_4`/`Const_Polynomial_Float_4` directly.
    public class _InOptMut_Polynomial_Float_4
    {
        public Polynomial_Float_4? Opt;

        public _InOptMut_Polynomial_Float_4() {}
        public _InOptMut_Polynomial_Float_4(Polynomial_Float_4 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_4(Polynomial_Float_4 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_4` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_4`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_4`/`Const_Polynomial_Float_4` to pass it to the function.
    public class _InOptConst_Polynomial_Float_4
    {
        public Const_Polynomial_Float_4? Opt;

        public _InOptConst_Polynomial_Float_4() {}
        public _InOptConst_Polynomial_Float_4(Const_Polynomial_Float_4 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_4(Const_Polynomial_Float_4 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 5>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_5 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_5(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_5_Destroy(_Underlying *_this);
            __MR_Polynomial_float_5_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_5() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_5_Get_n();
                return *__MR_Polynomial_float_5_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_5() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_5._Underlying *__MR_Polynomial_float_5_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_5_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 5>::Polynomial`.
        public unsafe Const_Polynomial_Float_5(MR._ByValue_Polynomial_Float_5 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_5._Underlying *__MR_Polynomial_float_5_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_5._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_5_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 5>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_5_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_5_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 5>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_4> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_4._Underlying *__MR_Polynomial_float_5_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_4(__MR_Polynomial_float_5_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<float, 5>::intervalMin`.
        public unsafe float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_intervalMin", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_5_intervalMin(_Underlying *_this, float a, float b);
            return __MR_Polynomial_float_5_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 5>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_5 : Const_Polynomial_Float_5
    {
        internal unsafe Polynomial_Float_5(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_5() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_5._Underlying *__MR_Polynomial_float_5_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_5_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 5>::Polynomial`.
        public unsafe Polynomial_Float_5(MR._ByValue_Polynomial_Float_5 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_5._Underlying *__MR_Polynomial_float_5_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_5._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_5_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 5>::operator=`.
        public unsafe MR.Polynomial_Float_5 Assign(MR._ByValue_Polynomial_Float_5 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_5_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_5._Underlying *__MR_Polynomial_float_5_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_5._Underlying *_other);
            return new(__MR_Polynomial_float_5_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_5` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_5`/`Const_Polynomial_Float_5` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_5
    {
        internal readonly Const_Polynomial_Float_5? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_5() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_5(Const_Polynomial_Float_5 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_5(Const_Polynomial_Float_5 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_5(MR.Misc._Moved<Polynomial_Float_5> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_5(MR.Misc._Moved<Polynomial_Float_5> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_5` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_5`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_5`/`Const_Polynomial_Float_5` directly.
    public class _InOptMut_Polynomial_Float_5
    {
        public Polynomial_Float_5? Opt;

        public _InOptMut_Polynomial_Float_5() {}
        public _InOptMut_Polynomial_Float_5(Polynomial_Float_5 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_5(Polynomial_Float_5 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_5` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_5`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_5`/`Const_Polynomial_Float_5` to pass it to the function.
    public class _InOptConst_Polynomial_Float_5
    {
        public Const_Polynomial_Float_5? Opt;

        public _InOptConst_Polynomial_Float_5() {}
        public _InOptConst_Polynomial_Float_5(Const_Polynomial_Float_5 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_5(Const_Polynomial_Float_5 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 6>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Float_6 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Float_6(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_float_6_Destroy(_Underlying *_this);
            __MR_Polynomial_float_6_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Float_6() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_float_6_Get_n();
                return *__MR_Polynomial_float_6_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Float_6() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_6._Underlying *__MR_Polynomial_float_6_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_6_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 6>::Polynomial`.
        public unsafe Const_Polynomial_Float_6(MR._ByValue_Polynomial_Float_6 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_6._Underlying *__MR_Polynomial_float_6_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_6._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_6_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 6>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_call", ExactSpelling = true)]
            extern static float __MR_Polynomial_float_6_call(_Underlying *_this, float x);
            return __MR_Polynomial_float_6_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<float, 6>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Float_5> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Float_5._Underlying *__MR_Polynomial_float_6_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Float_5(__MR_Polynomial_float_6_deriv(_UnderlyingPtr), is_owning: true));
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<float, 6>`.
    /// This is the non-const half of the class.
    public class Polynomial_Float_6 : Const_Polynomial_Float_6
    {
        internal unsafe Polynomial_Float_6(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Float_6() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Float_6._Underlying *__MR_Polynomial_float_6_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_float_6_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<float, 6>::Polynomial`.
        public unsafe Polynomial_Float_6(MR._ByValue_Polynomial_Float_6 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_6._Underlying *__MR_Polynomial_float_6_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_6._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_float_6_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<float, 6>::operator=`.
        public unsafe MR.Polynomial_Float_6 Assign(MR._ByValue_Polynomial_Float_6 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_float_6_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Float_6._Underlying *__MR_Polynomial_float_6_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Float_6._Underlying *_other);
            return new(__MR_Polynomial_float_6_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Float_6` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Float_6`/`Const_Polynomial_Float_6` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Float_6
    {
        internal readonly Const_Polynomial_Float_6? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Float_6() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Float_6(Const_Polynomial_Float_6 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Float_6(Const_Polynomial_Float_6 arg) {return new(arg);}
        public _ByValue_Polynomial_Float_6(MR.Misc._Moved<Polynomial_Float_6> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Float_6(MR.Misc._Moved<Polynomial_Float_6> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_6` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Float_6`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_6`/`Const_Polynomial_Float_6` directly.
    public class _InOptMut_Polynomial_Float_6
    {
        public Polynomial_Float_6? Opt;

        public _InOptMut_Polynomial_Float_6() {}
        public _InOptMut_Polynomial_Float_6(Polynomial_Float_6 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Float_6(Polynomial_Float_6 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Float_6` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Float_6`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Float_6`/`Const_Polynomial_Float_6` to pass it to the function.
    public class _InOptConst_Polynomial_Float_6
    {
        public Const_Polynomial_Float_6? Opt;

        public _InOptConst_Polynomial_Float_6() {}
        public _InOptConst_Polynomial_Float_6(Const_Polynomial_Float_6 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Float_6(Const_Polynomial_Float_6 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 0>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_0 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_0(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_0_Destroy(_Underlying *_this);
            __MR_Polynomial_double_0_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_0() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_0_Get_n();
                return *__MR_Polynomial_double_0_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_0() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_0_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_0_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 0>::Polynomial`.
        public unsafe Const_Polynomial_Double_0(MR._ByValue_Polynomial_Double_0 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_0_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_0._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_0_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 0>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_0_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_0_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 0>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_0> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_0_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_0(__MR_Polynomial_double_0_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 0>::intervalMin`.
        public unsafe double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_intervalMin", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_0_intervalMin(_Underlying *_this, double a, double b);
            return __MR_Polynomial_double_0_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 0>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_0 : Const_Polynomial_Double_0
    {
        internal unsafe Polynomial_Double_0(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_0() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_0_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_0_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 0>::Polynomial`.
        public unsafe Polynomial_Double_0(MR._ByValue_Polynomial_Double_0 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_0_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_0._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_0_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 0>::operator=`.
        public unsafe MR.Polynomial_Double_0 Assign(MR._ByValue_Polynomial_Double_0 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_0_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_0_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_0._Underlying *_other);
            return new(__MR_Polynomial_double_0_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_0` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_0`/`Const_Polynomial_Double_0` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_0
    {
        internal readonly Const_Polynomial_Double_0? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_0() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_0(Const_Polynomial_Double_0 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_0(Const_Polynomial_Double_0 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_0(MR.Misc._Moved<Polynomial_Double_0> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_0(MR.Misc._Moved<Polynomial_Double_0> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_0` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_0`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_0`/`Const_Polynomial_Double_0` directly.
    public class _InOptMut_Polynomial_Double_0
    {
        public Polynomial_Double_0? Opt;

        public _InOptMut_Polynomial_Double_0() {}
        public _InOptMut_Polynomial_Double_0(Polynomial_Double_0 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_0(Polynomial_Double_0 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_0` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_0`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_0`/`Const_Polynomial_Double_0` to pass it to the function.
    public class _InOptConst_Polynomial_Double_0
    {
        public Const_Polynomial_Double_0? Opt;

        public _InOptConst_Polynomial_Double_0() {}
        public _InOptConst_Polynomial_Double_0(Const_Polynomial_Double_0 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_0(Const_Polynomial_Double_0 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 1>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_1 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_1(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_1_Destroy(_Underlying *_this);
            __MR_Polynomial_double_1_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_1() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_1_Get_n();
                return *__MR_Polynomial_double_1_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_1() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_1._Underlying *__MR_Polynomial_double_1_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_1_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 1>::Polynomial`.
        public unsafe Const_Polynomial_Double_1(MR._ByValue_Polynomial_Double_1 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_1._Underlying *__MR_Polynomial_double_1_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_1._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_1_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 1>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_1_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_1_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 1>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Double> Solve(double tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Double._Underlying *__MR_Polynomial_double_1_solve(_Underlying *_this, double tol);
            return MR.Misc.Move(new MR.Std.Vector_Double(__MR_Polynomial_double_1_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 1>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_0> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_0._Underlying *__MR_Polynomial_double_1_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_0(__MR_Polynomial_double_1_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 1>::intervalMin`.
        public unsafe double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_intervalMin", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_1_intervalMin(_Underlying *_this, double a, double b);
            return __MR_Polynomial_double_1_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 1>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_1 : Const_Polynomial_Double_1
    {
        internal unsafe Polynomial_Double_1(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_1() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_1._Underlying *__MR_Polynomial_double_1_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_1_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 1>::Polynomial`.
        public unsafe Polynomial_Double_1(MR._ByValue_Polynomial_Double_1 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_1._Underlying *__MR_Polynomial_double_1_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_1._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_1_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 1>::operator=`.
        public unsafe MR.Polynomial_Double_1 Assign(MR._ByValue_Polynomial_Double_1 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_1_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_1._Underlying *__MR_Polynomial_double_1_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_1._Underlying *_other);
            return new(__MR_Polynomial_double_1_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_1` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_1`/`Const_Polynomial_Double_1` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_1
    {
        internal readonly Const_Polynomial_Double_1? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_1() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_1(Const_Polynomial_Double_1 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_1(Const_Polynomial_Double_1 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_1(MR.Misc._Moved<Polynomial_Double_1> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_1(MR.Misc._Moved<Polynomial_Double_1> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_1` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_1`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_1`/`Const_Polynomial_Double_1` directly.
    public class _InOptMut_Polynomial_Double_1
    {
        public Polynomial_Double_1? Opt;

        public _InOptMut_Polynomial_Double_1() {}
        public _InOptMut_Polynomial_Double_1(Polynomial_Double_1 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_1(Polynomial_Double_1 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_1` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_1`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_1`/`Const_Polynomial_Double_1` to pass it to the function.
    public class _InOptConst_Polynomial_Double_1
    {
        public Const_Polynomial_Double_1? Opt;

        public _InOptConst_Polynomial_Double_1() {}
        public _InOptConst_Polynomial_Double_1(Const_Polynomial_Double_1 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_1(Const_Polynomial_Double_1 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 2>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_2_Destroy(_Underlying *_this);
            __MR_Polynomial_double_2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_2() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_2_Get_n();
                return *__MR_Polynomial_double_2_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_2._Underlying *__MR_Polynomial_double_2_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_2_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 2>::Polynomial`.
        public unsafe Const_Polynomial_Double_2(MR._ByValue_Polynomial_Double_2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_2._Underlying *__MR_Polynomial_double_2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_2._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 2>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_2_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_2_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 2>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Double> Solve(double tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Double._Underlying *__MR_Polynomial_double_2_solve(_Underlying *_this, double tol);
            return MR.Misc.Move(new MR.Std.Vector_Double(__MR_Polynomial_double_2_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 2>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_1> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_1._Underlying *__MR_Polynomial_double_2_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_1(__MR_Polynomial_double_2_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 2>::intervalMin`.
        public unsafe double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_intervalMin", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_2_intervalMin(_Underlying *_this, double a, double b);
            return __MR_Polynomial_double_2_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 2>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_2 : Const_Polynomial_Double_2
    {
        internal unsafe Polynomial_Double_2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_2._Underlying *__MR_Polynomial_double_2_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_2_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 2>::Polynomial`.
        public unsafe Polynomial_Double_2(MR._ByValue_Polynomial_Double_2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_2._Underlying *__MR_Polynomial_double_2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_2._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 2>::operator=`.
        public unsafe MR.Polynomial_Double_2 Assign(MR._ByValue_Polynomial_Double_2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_2._Underlying *__MR_Polynomial_double_2_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_2._Underlying *_other);
            return new(__MR_Polynomial_double_2_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_2` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_2`/`Const_Polynomial_Double_2` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_2
    {
        internal readonly Const_Polynomial_Double_2? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_2() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_2(Const_Polynomial_Double_2 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_2(Const_Polynomial_Double_2 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_2(MR.Misc._Moved<Polynomial_Double_2> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_2(MR.Misc._Moved<Polynomial_Double_2> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_2`/`Const_Polynomial_Double_2` directly.
    public class _InOptMut_Polynomial_Double_2
    {
        public Polynomial_Double_2? Opt;

        public _InOptMut_Polynomial_Double_2() {}
        public _InOptMut_Polynomial_Double_2(Polynomial_Double_2 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_2(Polynomial_Double_2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_2`/`Const_Polynomial_Double_2` to pass it to the function.
    public class _InOptConst_Polynomial_Double_2
    {
        public Const_Polynomial_Double_2? Opt;

        public _InOptConst_Polynomial_Double_2() {}
        public _InOptConst_Polynomial_Double_2(Const_Polynomial_Double_2 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_2(Const_Polynomial_Double_2 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 3>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_3 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_3_Destroy(_Underlying *_this);
            __MR_Polynomial_double_3_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_3() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_3_Get_n();
                return *__MR_Polynomial_double_3_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_3._Underlying *__MR_Polynomial_double_3_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_3_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 3>::Polynomial`.
        public unsafe Const_Polynomial_Double_3(MR._ByValue_Polynomial_Double_3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_3._Underlying *__MR_Polynomial_double_3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_3._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 3>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_3_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_3_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 3>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Double> Solve(double tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Double._Underlying *__MR_Polynomial_double_3_solve(_Underlying *_this, double tol);
            return MR.Misc.Move(new MR.Std.Vector_Double(__MR_Polynomial_double_3_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 3>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_2> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_2._Underlying *__MR_Polynomial_double_3_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_2(__MR_Polynomial_double_3_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 3>::intervalMin`.
        public unsafe double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_intervalMin", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_3_intervalMin(_Underlying *_this, double a, double b);
            return __MR_Polynomial_double_3_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 3>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_3 : Const_Polynomial_Double_3
    {
        internal unsafe Polynomial_Double_3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_3._Underlying *__MR_Polynomial_double_3_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_3_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 3>::Polynomial`.
        public unsafe Polynomial_Double_3(MR._ByValue_Polynomial_Double_3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_3._Underlying *__MR_Polynomial_double_3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_3._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 3>::operator=`.
        public unsafe MR.Polynomial_Double_3 Assign(MR._ByValue_Polynomial_Double_3 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_3_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_3._Underlying *__MR_Polynomial_double_3_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_3._Underlying *_other);
            return new(__MR_Polynomial_double_3_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_3` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_3`/`Const_Polynomial_Double_3` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_3
    {
        internal readonly Const_Polynomial_Double_3? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_3() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_3(Const_Polynomial_Double_3 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_3(Const_Polynomial_Double_3 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_3(MR.Misc._Moved<Polynomial_Double_3> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_3(MR.Misc._Moved<Polynomial_Double_3> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_3` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_3`/`Const_Polynomial_Double_3` directly.
    public class _InOptMut_Polynomial_Double_3
    {
        public Polynomial_Double_3? Opt;

        public _InOptMut_Polynomial_Double_3() {}
        public _InOptMut_Polynomial_Double_3(Polynomial_Double_3 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_3(Polynomial_Double_3 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_3` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_3`/`Const_Polynomial_Double_3` to pass it to the function.
    public class _InOptConst_Polynomial_Double_3
    {
        public Const_Polynomial_Double_3? Opt;

        public _InOptConst_Polynomial_Double_3() {}
        public _InOptConst_Polynomial_Double_3(Const_Polynomial_Double_3 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_3(Const_Polynomial_Double_3 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 4>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_4 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_4(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_4_Destroy(_Underlying *_this);
            __MR_Polynomial_double_4_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_4() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_4_Get_n();
                return *__MR_Polynomial_double_4_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_4() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_4._Underlying *__MR_Polynomial_double_4_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_4_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 4>::Polynomial`.
        public unsafe Const_Polynomial_Double_4(MR._ByValue_Polynomial_Double_4 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_4._Underlying *__MR_Polynomial_double_4_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_4._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_4_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 4>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_4_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_4_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 4>::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_Double> Solve(double tol)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_solve", ExactSpelling = true)]
            extern static MR.Std.Vector_Double._Underlying *__MR_Polynomial_double_4_solve(_Underlying *_this, double tol);
            return MR.Misc.Move(new MR.Std.Vector_Double(__MR_Polynomial_double_4_solve(_UnderlyingPtr, tol), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 4>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_3> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_3._Underlying *__MR_Polynomial_double_4_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_3(__MR_Polynomial_double_4_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 4>::intervalMin`.
        public unsafe double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_intervalMin", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_4_intervalMin(_Underlying *_this, double a, double b);
            return __MR_Polynomial_double_4_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 4>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_4 : Const_Polynomial_Double_4
    {
        internal unsafe Polynomial_Double_4(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_4() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_4._Underlying *__MR_Polynomial_double_4_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_4_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 4>::Polynomial`.
        public unsafe Polynomial_Double_4(MR._ByValue_Polynomial_Double_4 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_4._Underlying *__MR_Polynomial_double_4_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_4._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_4_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 4>::operator=`.
        public unsafe MR.Polynomial_Double_4 Assign(MR._ByValue_Polynomial_Double_4 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_4_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_4._Underlying *__MR_Polynomial_double_4_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_4._Underlying *_other);
            return new(__MR_Polynomial_double_4_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_4` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_4`/`Const_Polynomial_Double_4` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_4
    {
        internal readonly Const_Polynomial_Double_4? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_4() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_4(Const_Polynomial_Double_4 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_4(Const_Polynomial_Double_4 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_4(MR.Misc._Moved<Polynomial_Double_4> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_4(MR.Misc._Moved<Polynomial_Double_4> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_4` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_4`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_4`/`Const_Polynomial_Double_4` directly.
    public class _InOptMut_Polynomial_Double_4
    {
        public Polynomial_Double_4? Opt;

        public _InOptMut_Polynomial_Double_4() {}
        public _InOptMut_Polynomial_Double_4(Polynomial_Double_4 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_4(Polynomial_Double_4 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_4` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_4`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_4`/`Const_Polynomial_Double_4` to pass it to the function.
    public class _InOptConst_Polynomial_Double_4
    {
        public Const_Polynomial_Double_4? Opt;

        public _InOptConst_Polynomial_Double_4() {}
        public _InOptConst_Polynomial_Double_4(Const_Polynomial_Double_4 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_4(Const_Polynomial_Double_4 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 5>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_5 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_5(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_5_Destroy(_Underlying *_this);
            __MR_Polynomial_double_5_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_5() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_5_Get_n();
                return *__MR_Polynomial_double_5_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_5() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_5._Underlying *__MR_Polynomial_double_5_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_5_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 5>::Polynomial`.
        public unsafe Const_Polynomial_Double_5(MR._ByValue_Polynomial_Double_5 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_5._Underlying *__MR_Polynomial_double_5_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_5._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_5_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 5>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_5_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_5_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 5>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_4> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_4._Underlying *__MR_Polynomial_double_5_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_4(__MR_Polynomial_double_5_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Polynomial<double, 5>::intervalMin`.
        public unsafe double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_intervalMin", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_5_intervalMin(_Underlying *_this, double a, double b);
            return __MR_Polynomial_double_5_intervalMin(_UnderlyingPtr, a, b);
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 5>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_5 : Const_Polynomial_Double_5
    {
        internal unsafe Polynomial_Double_5(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_5() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_5._Underlying *__MR_Polynomial_double_5_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_5_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 5>::Polynomial`.
        public unsafe Polynomial_Double_5(MR._ByValue_Polynomial_Double_5 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_5._Underlying *__MR_Polynomial_double_5_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_5._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_5_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 5>::operator=`.
        public unsafe MR.Polynomial_Double_5 Assign(MR._ByValue_Polynomial_Double_5 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_5_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_5._Underlying *__MR_Polynomial_double_5_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_5._Underlying *_other);
            return new(__MR_Polynomial_double_5_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_5` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_5`/`Const_Polynomial_Double_5` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_5
    {
        internal readonly Const_Polynomial_Double_5? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_5() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_5(Const_Polynomial_Double_5 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_5(Const_Polynomial_Double_5 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_5(MR.Misc._Moved<Polynomial_Double_5> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_5(MR.Misc._Moved<Polynomial_Double_5> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_5` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_5`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_5`/`Const_Polynomial_Double_5` directly.
    public class _InOptMut_Polynomial_Double_5
    {
        public Polynomial_Double_5? Opt;

        public _InOptMut_Polynomial_Double_5() {}
        public _InOptMut_Polynomial_Double_5(Polynomial_Double_5 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_5(Polynomial_Double_5 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_5` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_5`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_5`/`Const_Polynomial_Double_5` to pass it to the function.
    public class _InOptConst_Polynomial_Double_5
    {
        public Const_Polynomial_Double_5? Opt;

        public _InOptConst_Polynomial_Double_5() {}
        public _InOptConst_Polynomial_Double_5(Const_Polynomial_Double_5 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_5(Const_Polynomial_Double_5 value) {return new(value);}
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 6>`.
    /// This is the const half of the class.
    public class Const_Polynomial_Double_6 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polynomial_Double_6(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_Destroy", ExactSpelling = true)]
            extern static void __MR_Polynomial_double_6_Destroy(_Underlying *_this);
            __MR_Polynomial_double_6_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polynomial_Double_6() {Dispose(false);}

        public static unsafe ulong N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_Get_n", ExactSpelling = true)]
                extern static ulong *__MR_Polynomial_double_6_Get_n();
                return *__MR_Polynomial_double_6_Get_n();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polynomial_Double_6() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_6._Underlying *__MR_Polynomial_double_6_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_6_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 6>::Polynomial`.
        public unsafe Const_Polynomial_Double_6(MR._ByValue_Polynomial_Double_6 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_6._Underlying *__MR_Polynomial_double_6_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_6._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_6_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 6>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_call", ExactSpelling = true)]
            extern static double __MR_Polynomial_double_6_call(_Underlying *_this, double x);
            return __MR_Polynomial_double_6_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::Polynomial<double, 6>::deriv`.
        public unsafe MR.Misc._Moved<MR.Polynomial_Double_5> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_deriv", ExactSpelling = true)]
            extern static MR.Polynomial_Double_5._Underlying *__MR_Polynomial_double_6_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.Polynomial_Double_5(__MR_Polynomial_double_6_deriv(_UnderlyingPtr), is_owning: true));
        }
    }

    // Please note that these template classes are explicitly instantiated in the corresponding .cpp files.
    // The following degrees are instantiated: [2; 6].
    /// Generated from class `MR::Polynomial<double, 6>`.
    /// This is the non-const half of the class.
    public class Polynomial_Double_6 : Const_Polynomial_Double_6
    {
        internal unsafe Polynomial_Double_6(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polynomial_Double_6() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polynomial_Double_6._Underlying *__MR_Polynomial_double_6_DefaultConstruct();
            _UnderlyingPtr = __MR_Polynomial_double_6_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polynomial<double, 6>::Polynomial`.
        public unsafe Polynomial_Double_6(MR._ByValue_Polynomial_Double_6 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_6._Underlying *__MR_Polynomial_double_6_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_6._Underlying *_other);
            _UnderlyingPtr = __MR_Polynomial_double_6_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polynomial<double, 6>::operator=`.
        public unsafe MR.Polynomial_Double_6 Assign(MR._ByValue_Polynomial_Double_6 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polynomial_double_6_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polynomial_Double_6._Underlying *__MR_Polynomial_double_6_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polynomial_Double_6._Underlying *_other);
            return new(__MR_Polynomial_double_6_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polynomial_Double_6` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polynomial_Double_6`/`Const_Polynomial_Double_6` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polynomial_Double_6
    {
        internal readonly Const_Polynomial_Double_6? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polynomial_Double_6() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polynomial_Double_6(Const_Polynomial_Double_6 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polynomial_Double_6(Const_Polynomial_Double_6 arg) {return new(arg);}
        public _ByValue_Polynomial_Double_6(MR.Misc._Moved<Polynomial_Double_6> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polynomial_Double_6(MR.Misc._Moved<Polynomial_Double_6> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_6` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polynomial_Double_6`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_6`/`Const_Polynomial_Double_6` directly.
    public class _InOptMut_Polynomial_Double_6
    {
        public Polynomial_Double_6? Opt;

        public _InOptMut_Polynomial_Double_6() {}
        public _InOptMut_Polynomial_Double_6(Polynomial_Double_6 value) {Opt = value;}
        public static implicit operator _InOptMut_Polynomial_Double_6(Polynomial_Double_6 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polynomial_Double_6` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polynomial_Double_6`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polynomial_Double_6`/`Const_Polynomial_Double_6` to pass it to the function.
    public class _InOptConst_Polynomial_Double_6
    {
        public Const_Polynomial_Double_6? Opt;

        public _InOptConst_Polynomial_Double_6() {}
        public _InOptConst_Polynomial_Double_6(Const_Polynomial_Double_6 value) {Opt = value;}
        public static implicit operator _InOptConst_Polynomial_Double_6(Const_Polynomial_Double_6 value) {return new(value);}
    }

    /// This is a unifying interface for a polynomial of some degree, known only in runtime
    /// Generated from class `MR::PolynomialWrapper<float>`.
    /// This is the const half of the class.
    public class Const_PolynomialWrapper_Float : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolynomialWrapper_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_Destroy", ExactSpelling = true)]
            extern static void __MR_PolynomialWrapper_float_Destroy(_Underlying *_this);
            __MR_PolynomialWrapper_float_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolynomialWrapper_Float() {Dispose(false);}

        public unsafe MR.Std.Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 Poly
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_Get_poly", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_PolynomialWrapper_float_Get_poly(_Underlying *_this);
                return new(__MR_PolynomialWrapper_float_Get_poly(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::PolynomialWrapper<float>::PolynomialWrapper`.
        public unsafe Const_PolynomialWrapper_Float(MR._ByValue_PolynomialWrapper_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_PolynomialWrapper_float_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolynomialWrapper_Float._Underlying *_other);
            _UnderlyingPtr = __MR_PolynomialWrapper_float_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolynomialWrapper<float>::operator()`.
        public unsafe float Call(float x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_call", ExactSpelling = true)]
            extern static float __MR_PolynomialWrapper_float_call(_Underlying *_this, float x);
            return __MR_PolynomialWrapper_float_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::PolynomialWrapper<float>::deriv`.
        public unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_deriv", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_PolynomialWrapper_float_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_PolynomialWrapper_float_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PolynomialWrapper<float>::intervalMin`.
        public unsafe MR.Std.Optional_Float IntervalMin(float a, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_intervalMin", ExactSpelling = true)]
            extern static MR.Std.Optional_Float._Underlying *__MR_PolynomialWrapper_float_intervalMin(_Underlying *_this, float a, float b);
            return new(__MR_PolynomialWrapper_float_intervalMin(_UnderlyingPtr, a, b), is_owning: true);
        }
    }

    /// This is a unifying interface for a polynomial of some degree, known only in runtime
    /// Generated from class `MR::PolynomialWrapper<float>`.
    /// This is the non-const half of the class.
    public class PolynomialWrapper_Float : Const_PolynomialWrapper_Float
    {
        internal unsafe PolynomialWrapper_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 Poly
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_GetMutable_poly", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_PolynomialWrapper_float_GetMutable_poly(_Underlying *_this);
                return new(__MR_PolynomialWrapper_float_GetMutable_poly(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::PolynomialWrapper<float>::PolynomialWrapper`.
        public unsafe PolynomialWrapper_Float(MR._ByValue_PolynomialWrapper_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_PolynomialWrapper_float_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolynomialWrapper_Float._Underlying *_other);
            _UnderlyingPtr = __MR_PolynomialWrapper_float_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolynomialWrapper<float>::operator=`.
        public unsafe MR.PolynomialWrapper_Float Assign(MR._ByValue_PolynomialWrapper_Float _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_PolynomialWrapper_float_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PolynomialWrapper_Float._Underlying *_other);
            return new(__MR_PolynomialWrapper_float_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PolynomialWrapper_Float` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PolynomialWrapper_Float`/`Const_PolynomialWrapper_Float` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PolynomialWrapper_Float
    {
        internal readonly Const_PolynomialWrapper_Float? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PolynomialWrapper_Float(Const_PolynomialWrapper_Float new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PolynomialWrapper_Float(Const_PolynomialWrapper_Float arg) {return new(arg);}
        public _ByValue_PolynomialWrapper_Float(MR.Misc._Moved<PolynomialWrapper_Float> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PolynomialWrapper_Float(MR.Misc._Moved<PolynomialWrapper_Float> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PolynomialWrapper_Float` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolynomialWrapper_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolynomialWrapper_Float`/`Const_PolynomialWrapper_Float` directly.
    public class _InOptMut_PolynomialWrapper_Float
    {
        public PolynomialWrapper_Float? Opt;

        public _InOptMut_PolynomialWrapper_Float() {}
        public _InOptMut_PolynomialWrapper_Float(PolynomialWrapper_Float value) {Opt = value;}
        public static implicit operator _InOptMut_PolynomialWrapper_Float(PolynomialWrapper_Float value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolynomialWrapper_Float` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolynomialWrapper_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolynomialWrapper_Float`/`Const_PolynomialWrapper_Float` to pass it to the function.
    public class _InOptConst_PolynomialWrapper_Float
    {
        public Const_PolynomialWrapper_Float? Opt;

        public _InOptConst_PolynomialWrapper_Float() {}
        public _InOptConst_PolynomialWrapper_Float(Const_PolynomialWrapper_Float value) {Opt = value;}
        public static implicit operator _InOptConst_PolynomialWrapper_Float(Const_PolynomialWrapper_Float value) {return new(value);}
    }

    /// This is a unifying interface for a polynomial of some degree, known only in runtime
    /// Generated from class `MR::PolynomialWrapper<double>`.
    /// This is the const half of the class.
    public class Const_PolynomialWrapper_Double : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolynomialWrapper_Double(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_Destroy", ExactSpelling = true)]
            extern static void __MR_PolynomialWrapper_double_Destroy(_Underlying *_this);
            __MR_PolynomialWrapper_double_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolynomialWrapper_Double() {Dispose(false);}

        public unsafe MR.Std.Const_Variant_MRPolynomialDouble0_MRPolynomialDouble1_MRPolynomialDouble2_MRPolynomialDouble3_MRPolynomialDouble4_MRPolynomialDouble5_MRPolynomialDouble6 Poly
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_Get_poly", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRPolynomialDouble0_MRPolynomialDouble1_MRPolynomialDouble2_MRPolynomialDouble3_MRPolynomialDouble4_MRPolynomialDouble5_MRPolynomialDouble6._Underlying *__MR_PolynomialWrapper_double_Get_poly(_Underlying *_this);
                return new(__MR_PolynomialWrapper_double_Get_poly(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::PolynomialWrapper<double>::PolynomialWrapper`.
        public unsafe Const_PolynomialWrapper_Double(MR._ByValue_PolynomialWrapper_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Double._Underlying *__MR_PolynomialWrapper_double_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolynomialWrapper_Double._Underlying *_other);
            _UnderlyingPtr = __MR_PolynomialWrapper_double_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolynomialWrapper<double>::operator()`.
        public unsafe double Call(double x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_call", ExactSpelling = true)]
            extern static double __MR_PolynomialWrapper_double_call(_Underlying *_this, double x);
            return __MR_PolynomialWrapper_double_call(_UnderlyingPtr, x);
        }

        /// Generated from method `MR::PolynomialWrapper<double>::deriv`.
        public unsafe MR.Misc._Moved<MR.PolynomialWrapper_Double> Deriv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_deriv", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Double._Underlying *__MR_PolynomialWrapper_double_deriv(_Underlying *_this);
            return MR.Misc.Move(new MR.PolynomialWrapper_Double(__MR_PolynomialWrapper_double_deriv(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PolynomialWrapper<double>::intervalMin`.
        public unsafe MR.Std.Optional_Double IntervalMin(double a, double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_intervalMin", ExactSpelling = true)]
            extern static MR.Std.Optional_Double._Underlying *__MR_PolynomialWrapper_double_intervalMin(_Underlying *_this, double a, double b);
            return new(__MR_PolynomialWrapper_double_intervalMin(_UnderlyingPtr, a, b), is_owning: true);
        }
    }

    /// This is a unifying interface for a polynomial of some degree, known only in runtime
    /// Generated from class `MR::PolynomialWrapper<double>`.
    /// This is the non-const half of the class.
    public class PolynomialWrapper_Double : Const_PolynomialWrapper_Double
    {
        internal unsafe PolynomialWrapper_Double(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Variant_MRPolynomialDouble0_MRPolynomialDouble1_MRPolynomialDouble2_MRPolynomialDouble3_MRPolynomialDouble4_MRPolynomialDouble5_MRPolynomialDouble6 Poly
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_GetMutable_poly", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialDouble0_MRPolynomialDouble1_MRPolynomialDouble2_MRPolynomialDouble3_MRPolynomialDouble4_MRPolynomialDouble5_MRPolynomialDouble6._Underlying *__MR_PolynomialWrapper_double_GetMutable_poly(_Underlying *_this);
                return new(__MR_PolynomialWrapper_double_GetMutable_poly(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::PolynomialWrapper<double>::PolynomialWrapper`.
        public unsafe PolynomialWrapper_Double(MR._ByValue_PolynomialWrapper_Double _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Double._Underlying *__MR_PolynomialWrapper_double_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolynomialWrapper_Double._Underlying *_other);
            _UnderlyingPtr = __MR_PolynomialWrapper_double_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolynomialWrapper<double>::operator=`.
        public unsafe MR.PolynomialWrapper_Double Assign(MR._ByValue_PolynomialWrapper_Double _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolynomialWrapper_double_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Double._Underlying *__MR_PolynomialWrapper_double_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PolynomialWrapper_Double._Underlying *_other);
            return new(__MR_PolynomialWrapper_double_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PolynomialWrapper_Double` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PolynomialWrapper_Double`/`Const_PolynomialWrapper_Double` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PolynomialWrapper_Double
    {
        internal readonly Const_PolynomialWrapper_Double? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PolynomialWrapper_Double(Const_PolynomialWrapper_Double new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PolynomialWrapper_Double(Const_PolynomialWrapper_Double arg) {return new(arg);}
        public _ByValue_PolynomialWrapper_Double(MR.Misc._Moved<PolynomialWrapper_Double> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PolynomialWrapper_Double(MR.Misc._Moved<PolynomialWrapper_Double> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PolynomialWrapper_Double` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolynomialWrapper_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolynomialWrapper_Double`/`Const_PolynomialWrapper_Double` directly.
    public class _InOptMut_PolynomialWrapper_Double
    {
        public PolynomialWrapper_Double? Opt;

        public _InOptMut_PolynomialWrapper_Double() {}
        public _InOptMut_PolynomialWrapper_Double(PolynomialWrapper_Double value) {Opt = value;}
        public static implicit operator _InOptMut_PolynomialWrapper_Double(PolynomialWrapper_Double value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolynomialWrapper_Double` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolynomialWrapper_Double`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolynomialWrapper_Double`/`Const_PolynomialWrapper_Double` to pass it to the function.
    public class _InOptConst_PolynomialWrapper_Double
    {
        public Const_PolynomialWrapper_Double? Opt;

        public _InOptConst_PolynomialWrapper_Double() {}
        public _InOptConst_PolynomialWrapper_Double(Const_PolynomialWrapper_Double value) {Opt = value;}
        public static implicit operator _InOptConst_PolynomialWrapper_Double(Const_PolynomialWrapper_Double value) {return new(value);}
    }

    /// Generated from function `MR::canSolvePolynomial<MR_uint64_t>`.
    public static bool CanSolvePolynomial(ulong degree)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_canSolvePolynomial", ExactSpelling = true)]
        extern static byte __MR_canSolvePolynomial(ulong degree);
        return __MR_canSolvePolynomial(degree) != 0;
    }

    /// Generated from function `MR::canMinimizePolynomial<MR_uint64_t>`.
    public static bool CanMinimizePolynomial(ulong degree)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_canMinimizePolynomial", ExactSpelling = true)]
        extern static byte __MR_canMinimizePolynomial(ulong degree);
        return __MR_canMinimizePolynomial(degree) != 0;
    }
}
