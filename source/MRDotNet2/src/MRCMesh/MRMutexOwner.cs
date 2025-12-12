public static partial class MR
{
    /// This class exists to provide copy and move constructors and assignment operations for std::mutex
    /// which actually does nothing
    /// Generated from class `MR::MutexOwner`.
    /// This is the const half of the class.
    public class Const_MutexOwner : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MutexOwner(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MutexOwner_Destroy", ExactSpelling = true)]
            extern static void __MR_MutexOwner_Destroy(_Underlying *_this);
            __MR_MutexOwner_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MutexOwner() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MutexOwner() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MutexOwner_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MutexOwner._Underlying *__MR_MutexOwner_DefaultConstruct();
            _UnderlyingPtr = __MR_MutexOwner_DefaultConstruct();
        }

        /// Generated from constructor `MR::MutexOwner::MutexOwner`.
        public unsafe Const_MutexOwner(MR._ByValue_MutexOwner _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MutexOwner_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MutexOwner._Underlying *__MR_MutexOwner_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MutexOwner._Underlying *_other);
            _UnderlyingPtr = __MR_MutexOwner_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// This class exists to provide copy and move constructors and assignment operations for std::mutex
    /// which actually does nothing
    /// Generated from class `MR::MutexOwner`.
    /// This is the non-const half of the class.
    public class MutexOwner : Const_MutexOwner
    {
        internal unsafe MutexOwner(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MutexOwner() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MutexOwner_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MutexOwner._Underlying *__MR_MutexOwner_DefaultConstruct();
            _UnderlyingPtr = __MR_MutexOwner_DefaultConstruct();
        }

        /// Generated from constructor `MR::MutexOwner::MutexOwner`.
        public unsafe MutexOwner(MR._ByValue_MutexOwner _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MutexOwner_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MutexOwner._Underlying *__MR_MutexOwner_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MutexOwner._Underlying *_other);
            _UnderlyingPtr = __MR_MutexOwner_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MutexOwner::operator=`.
        public unsafe MR.MutexOwner Assign(MR._ByValue_MutexOwner _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MutexOwner_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MutexOwner._Underlying *__MR_MutexOwner_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MutexOwner._Underlying *_other);
            return new(__MR_MutexOwner_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MutexOwner` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MutexOwner`/`Const_MutexOwner` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MutexOwner
    {
        internal readonly Const_MutexOwner? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MutexOwner() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MutexOwner(Const_MutexOwner new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MutexOwner(Const_MutexOwner arg) {return new(arg);}
        public _ByValue_MutexOwner(MR.Misc._Moved<MutexOwner> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MutexOwner(MR.Misc._Moved<MutexOwner> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MutexOwner` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MutexOwner`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MutexOwner`/`Const_MutexOwner` directly.
    public class _InOptMut_MutexOwner
    {
        public MutexOwner? Opt;

        public _InOptMut_MutexOwner() {}
        public _InOptMut_MutexOwner(MutexOwner value) {Opt = value;}
        public static implicit operator _InOptMut_MutexOwner(MutexOwner value) {return new(value);}
    }

    /// This is used for optional parameters of class `MutexOwner` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MutexOwner`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MutexOwner`/`Const_MutexOwner` to pass it to the function.
    public class _InOptConst_MutexOwner
    {
        public Const_MutexOwner? Opt;

        public _InOptConst_MutexOwner() {}
        public _InOptConst_MutexOwner(Const_MutexOwner value) {Opt = value;}
        public static implicit operator _InOptConst_MutexOwner(Const_MutexOwner value) {return new(value);}
    }
}
