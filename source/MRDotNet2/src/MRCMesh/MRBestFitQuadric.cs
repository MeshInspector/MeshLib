public static partial class MR
{
    /// Accumulate points and make best quadric approximation
    /// \details \f$ a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = z \f$
    /// Generated from class `MR::QuadricApprox`.
    /// This is the const half of the class.
    public class Const_QuadricApprox : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_QuadricApprox(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_Destroy", ExactSpelling = true)]
            extern static void __MR_QuadricApprox_Destroy(_Underlying *_this);
            __MR_QuadricApprox_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_QuadricApprox() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_QuadricApprox() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadricApprox._Underlying *__MR_QuadricApprox_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadricApprox_DefaultConstruct();
        }

        /// Generated from constructor `MR::QuadricApprox::QuadricApprox`.
        public unsafe Const_QuadricApprox(MR._ByValue_QuadricApprox _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadricApprox._Underlying *__MR_QuadricApprox_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.QuadricApprox._Underlying *_other);
            _UnderlyingPtr = __MR_QuadricApprox_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Accumulate points and make best quadric approximation
    /// \details \f$ a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = z \f$
    /// Generated from class `MR::QuadricApprox`.
    /// This is the non-const half of the class.
    public class QuadricApprox : Const_QuadricApprox
    {
        internal unsafe QuadricApprox(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe QuadricApprox() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_DefaultConstruct", ExactSpelling = true)]
            extern static MR.QuadricApprox._Underlying *__MR_QuadricApprox_DefaultConstruct();
            _UnderlyingPtr = __MR_QuadricApprox_DefaultConstruct();
        }

        /// Generated from constructor `MR::QuadricApprox::QuadricApprox`.
        public unsafe QuadricApprox(MR._ByValue_QuadricApprox _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.QuadricApprox._Underlying *__MR_QuadricApprox_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.QuadricApprox._Underlying *_other);
            _UnderlyingPtr = __MR_QuadricApprox_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::QuadricApprox::operator=`.
        public unsafe MR.QuadricApprox Assign(MR._ByValue_QuadricApprox _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_AssignFromAnother", ExactSpelling = true)]
            extern static MR.QuadricApprox._Underlying *__MR_QuadricApprox_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.QuadricApprox._Underlying *_other);
            return new(__MR_QuadricApprox_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Adds point to accumulation with weight
        /// Generated from method `MR::QuadricApprox::addPoint`.
        /// Parameter `weight` defaults to `1.0`.
        public unsafe void AddPoint(MR.Const_Vector3d point, double? weight = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_QuadricApprox_addPoint", ExactSpelling = true)]
            extern static void __MR_QuadricApprox_addPoint(_Underlying *_this, MR.Const_Vector3d._Underlying *point, double *weight);
            double __deref_weight = weight.GetValueOrDefault();
            __MR_QuadricApprox_addPoint(_UnderlyingPtr, point._UnderlyingPtr, weight.HasValue ? &__deref_weight : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `QuadricApprox` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `QuadricApprox`/`Const_QuadricApprox` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_QuadricApprox
    {
        internal readonly Const_QuadricApprox? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_QuadricApprox() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_QuadricApprox(Const_QuadricApprox new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_QuadricApprox(Const_QuadricApprox arg) {return new(arg);}
        public _ByValue_QuadricApprox(MR.Misc._Moved<QuadricApprox> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_QuadricApprox(MR.Misc._Moved<QuadricApprox> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `QuadricApprox` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_QuadricApprox`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadricApprox`/`Const_QuadricApprox` directly.
    public class _InOptMut_QuadricApprox
    {
        public QuadricApprox? Opt;

        public _InOptMut_QuadricApprox() {}
        public _InOptMut_QuadricApprox(QuadricApprox value) {Opt = value;}
        public static implicit operator _InOptMut_QuadricApprox(QuadricApprox value) {return new(value);}
    }

    /// This is used for optional parameters of class `QuadricApprox` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_QuadricApprox`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `QuadricApprox`/`Const_QuadricApprox` to pass it to the function.
    public class _InOptConst_QuadricApprox
    {
        public Const_QuadricApprox? Opt;

        public _InOptConst_QuadricApprox() {}
        public _InOptConst_QuadricApprox(Const_QuadricApprox value) {Opt = value;}
        public static implicit operator _InOptConst_QuadricApprox(Const_QuadricApprox value) {return new(value);}
    }
}
