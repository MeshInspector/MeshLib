public static partial class MR
{
    /// Generated from class `MR::ObjectFactoryBase`.
    /// This is the const half of the class.
    public class Const_ObjectFactoryBase : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjectFactoryBase(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectFactoryBase_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjectFactoryBase_Destroy(_Underlying *_this);
            __MR_ObjectFactoryBase_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectFactoryBase() {Dispose(false);}

        /// Generated from constructor `MR::ObjectFactoryBase::ObjectFactoryBase`.
        public unsafe Const_ObjectFactoryBase(MR._ByValue_ObjectFactoryBase _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectFactoryBase_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectFactoryBase._Underlying *__MR_ObjectFactoryBase_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectFactoryBase._Underlying *_other);
            _UnderlyingPtr = __MR_ObjectFactoryBase_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ObjectFactoryBase`.
    /// This is the non-const half of the class.
    public class ObjectFactoryBase : Const_ObjectFactoryBase
    {
        internal unsafe ObjectFactoryBase(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::ObjectFactoryBase::ObjectFactoryBase`.
        public unsafe ObjectFactoryBase(MR._ByValue_ObjectFactoryBase _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectFactoryBase_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectFactoryBase._Underlying *__MR_ObjectFactoryBase_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectFactoryBase._Underlying *_other);
            _UnderlyingPtr = __MR_ObjectFactoryBase_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectFactoryBase` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectFactoryBase`/`Const_ObjectFactoryBase` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectFactoryBase
    {
        internal readonly Const_ObjectFactoryBase? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectFactoryBase(Const_ObjectFactoryBase new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ObjectFactoryBase(Const_ObjectFactoryBase arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectFactoryBase` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectFactoryBase`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectFactoryBase`/`Const_ObjectFactoryBase` directly.
    public class _InOptMut_ObjectFactoryBase
    {
        public ObjectFactoryBase? Opt;

        public _InOptMut_ObjectFactoryBase() {}
        public _InOptMut_ObjectFactoryBase(ObjectFactoryBase value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectFactoryBase(ObjectFactoryBase value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectFactoryBase` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectFactoryBase`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectFactoryBase`/`Const_ObjectFactoryBase` to pass it to the function.
    public class _InOptConst_ObjectFactoryBase
    {
        public Const_ObjectFactoryBase? Opt;

        public _InOptConst_ObjectFactoryBase() {}
        public _InOptConst_ObjectFactoryBase(Const_ObjectFactoryBase value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectFactoryBase(Const_ObjectFactoryBase value) {return new(value);}
    }

    /// the function to create new object instance by registered class name
    /// Generated from function `MR::createObject`.
    public static unsafe MR.Misc._Moved<MR.Object> CreateObject(ReadOnlySpan<char> className)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_createObject", ExactSpelling = true)]
        extern static MR.Object._UnderlyingShared *__MR_createObject(byte *className, byte *className_end);
        byte[] __bytes_className = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(className.Length)];
        int __len_className = System.Text.Encoding.UTF8.GetBytes(className, __bytes_className);
        fixed (byte *__ptr_className = __bytes_className)
        {
            return MR.Misc.Move(new MR.Object(__MR_createObject(__ptr_className, __ptr_className + __len_className), is_owning: true));
        }
    }
}
