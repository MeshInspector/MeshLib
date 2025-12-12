public static partial class MR
{
    /// Generated from class `MR::NoInit`.
    /// This is the const half of the class.
    public class Const_NoInit : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoInit(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInit_Destroy", ExactSpelling = true)]
            extern static void __MR_NoInit_Destroy(_Underlying *_this);
            __MR_NoInit_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoInit() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoInit() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInit_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoInit._Underlying *__MR_NoInit_DefaultConstruct();
            _UnderlyingPtr = __MR_NoInit_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoInit::NoInit`.
        public unsafe Const_NoInit(MR.Const_NoInit _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInit_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoInit._Underlying *__MR_NoInit_ConstructFromAnother(MR.NoInit._Underlying *_other);
            _UnderlyingPtr = __MR_NoInit_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::NoInit`.
    /// This is the non-const half of the class.
    public class NoInit : Const_NoInit
    {
        internal unsafe NoInit(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoInit() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInit_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoInit._Underlying *__MR_NoInit_DefaultConstruct();
            _UnderlyingPtr = __MR_NoInit_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoInit::NoInit`.
        public unsafe NoInit(MR.Const_NoInit _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInit_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoInit._Underlying *__MR_NoInit_ConstructFromAnother(MR.NoInit._Underlying *_other);
            _UnderlyingPtr = __MR_NoInit_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoInit::operator=`.
        public unsafe MR.NoInit Assign(MR.Const_NoInit _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInit_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoInit._Underlying *__MR_NoInit_AssignFromAnother(_Underlying *_this, MR.NoInit._Underlying *_other);
            return new(__MR_NoInit_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoInit` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoInit`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoInit`/`Const_NoInit` directly.
    public class _InOptMut_NoInit
    {
        public NoInit? Opt;

        public _InOptMut_NoInit() {}
        public _InOptMut_NoInit(NoInit value) {Opt = value;}
        public static implicit operator _InOptMut_NoInit(NoInit value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoInit` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoInit`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoInit`/`Const_NoInit` to pass it to the function.
    public class _InOptConst_NoInit
    {
        public Const_NoInit? Opt;

        public _InOptConst_NoInit() {}
        public _InOptConst_NoInit(Const_NoInit value) {Opt = value;}
        public static implicit operator _InOptConst_NoInit(Const_NoInit value) {return new(value);}
    }
}
