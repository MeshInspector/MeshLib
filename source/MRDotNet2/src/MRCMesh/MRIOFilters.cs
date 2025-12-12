public static partial class MR
{
    /// Generated from class `MR::IOFilter`.
    /// This is the const half of the class.
    public class Const_IOFilter : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IOFilter(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_Destroy", ExactSpelling = true)]
            extern static void __MR_IOFilter_Destroy(_Underlying *_this);
            __MR_IOFilter_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IOFilter() {Dispose(false);}

        public unsafe MR.Std.Const_String Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_IOFilter_Get_name(_Underlying *_this);
                return new(__MR_IOFilter_Get_name(_UnderlyingPtr), is_owning: false);
            }
        }

        // "*.ext" or "*.ext1;*.ext2;*.ext3"
        public unsafe MR.Std.Const_String Extensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_Get_extensions", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_IOFilter_Get_extensions(_Underlying *_this);
                return new(__MR_IOFilter_Get_extensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IOFilter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_DefaultConstruct();
            _UnderlyingPtr = __MR_IOFilter_DefaultConstruct();
        }

        /// Generated from constructor `MR::IOFilter::IOFilter`.
        public unsafe Const_IOFilter(MR._ByValue_IOFilter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.IOFilter._Underlying *_other);
            _UnderlyingPtr = __MR_IOFilter_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::IOFilter::IOFilter`.
        public unsafe Const_IOFilter(ReadOnlySpan<char> _name, ReadOnlySpan<char> _ext) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_Construct", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_Construct(byte *_name, byte *_name_end, byte *_ext, byte *_ext_end);
            byte[] __bytes__name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(_name.Length)];
            int __len__name = System.Text.Encoding.UTF8.GetBytes(_name, __bytes__name);
            fixed (byte *__ptr__name = __bytes__name)
            {
                byte[] __bytes__ext = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(_ext.Length)];
                int __len__ext = System.Text.Encoding.UTF8.GetBytes(_ext, __bytes__ext);
                fixed (byte *__ptr__ext = __bytes__ext)
                {
                    _UnderlyingPtr = __MR_IOFilter_Construct(__ptr__name, __ptr__name + __len__name, __ptr__ext, __ptr__ext + __len__ext);
                }
            }
        }

        /// Generated from method `MR::IOFilter::isSupportedExtension`.
        public unsafe bool IsSupportedExtension(ReadOnlySpan<char> ext)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_isSupportedExtension", ExactSpelling = true)]
            extern static byte __MR_IOFilter_isSupportedExtension(_Underlying *_this, byte *ext, byte *ext_end);
            byte[] __bytes_ext = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(ext.Length)];
            int __len_ext = System.Text.Encoding.UTF8.GetBytes(ext, __bytes_ext);
            fixed (byte *__ptr_ext = __bytes_ext)
            {
                return __MR_IOFilter_isSupportedExtension(_UnderlyingPtr, __ptr_ext, __ptr_ext + __len_ext) != 0;
            }
        }
    }

    /// Generated from class `MR::IOFilter`.
    /// This is the non-const half of the class.
    public class IOFilter : Const_IOFilter
    {
        internal unsafe IOFilter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.String Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_GetMutable_name", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_IOFilter_GetMutable_name(_Underlying *_this);
                return new(__MR_IOFilter_GetMutable_name(_UnderlyingPtr), is_owning: false);
            }
        }

        // "*.ext" or "*.ext1;*.ext2;*.ext3"
        public new unsafe MR.Std.String Extensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_GetMutable_extensions", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_IOFilter_GetMutable_extensions(_Underlying *_this);
                return new(__MR_IOFilter_GetMutable_extensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe IOFilter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_DefaultConstruct();
            _UnderlyingPtr = __MR_IOFilter_DefaultConstruct();
        }

        /// Generated from constructor `MR::IOFilter::IOFilter`.
        public unsafe IOFilter(MR._ByValue_IOFilter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.IOFilter._Underlying *_other);
            _UnderlyingPtr = __MR_IOFilter_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::IOFilter::IOFilter`.
        public unsafe IOFilter(ReadOnlySpan<char> _name, ReadOnlySpan<char> _ext) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_Construct", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_Construct(byte *_name, byte *_name_end, byte *_ext, byte *_ext_end);
            byte[] __bytes__name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(_name.Length)];
            int __len__name = System.Text.Encoding.UTF8.GetBytes(_name, __bytes__name);
            fixed (byte *__ptr__name = __bytes__name)
            {
                byte[] __bytes__ext = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(_ext.Length)];
                int __len__ext = System.Text.Encoding.UTF8.GetBytes(_ext, __bytes__ext);
                fixed (byte *__ptr__ext = __bytes__ext)
                {
                    _UnderlyingPtr = __MR_IOFilter_Construct(__ptr__name, __ptr__name + __len__name, __ptr__ext, __ptr__ext + __len__ext);
                }
            }
        }

        /// Generated from method `MR::IOFilter::operator=`.
        public unsafe MR.IOFilter Assign(MR._ByValue_IOFilter _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IOFilter_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IOFilter._Underlying *__MR_IOFilter_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.IOFilter._Underlying *_other);
            return new(__MR_IOFilter_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `IOFilter` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `IOFilter`/`Const_IOFilter` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_IOFilter
    {
        internal readonly Const_IOFilter? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_IOFilter() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_IOFilter(Const_IOFilter new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_IOFilter(Const_IOFilter arg) {return new(arg);}
        public _ByValue_IOFilter(MR.Misc._Moved<IOFilter> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_IOFilter(MR.Misc._Moved<IOFilter> arg) {return new(arg);}
    }

    /// This is used as a function parameter when the underlying function receives an optional `IOFilter` by value,
    ///   and also has a default argument, meaning it has two different null states.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `IOFilter`/`Const_IOFilter` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument.
    /// * Pass `MR.Misc.NullOptType` to pass no object.
    public class _ByValueOptOpt_IOFilter
    {
        internal readonly Const_IOFilter? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValueOptOpt_IOFilter() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValueOptOpt_IOFilter(Const_IOFilter new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValueOptOpt_IOFilter(Const_IOFilter arg) {return new(arg);}
        public _ByValueOptOpt_IOFilter(MR.Misc._Moved<IOFilter> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValueOptOpt_IOFilter(MR.Misc._Moved<IOFilter> arg) {return new(arg);}
        public _ByValueOptOpt_IOFilter(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
        public static implicit operator _ByValueOptOpt_IOFilter(MR.Misc.NullOptType nullopt) {return new(nullopt);}
    }

    /// This is used for optional parameters of class `IOFilter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IOFilter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IOFilter`/`Const_IOFilter` directly.
    public class _InOptMut_IOFilter
    {
        public IOFilter? Opt;

        public _InOptMut_IOFilter() {}
        public _InOptMut_IOFilter(IOFilter value) {Opt = value;}
        public static implicit operator _InOptMut_IOFilter(IOFilter value) {return new(value);}
    }

    /// This is used for optional parameters of class `IOFilter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IOFilter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IOFilter`/`Const_IOFilter` to pass it to the function.
    public class _InOptConst_IOFilter
    {
        public Const_IOFilter? Opt;

        public _InOptConst_IOFilter() {}
        public _InOptConst_IOFilter(Const_IOFilter value) {Opt = value;}
        public static implicit operator _InOptConst_IOFilter(Const_IOFilter value) {return new(value);}
    }

    /// find a corresponding filter for a given extension
    /// Generated from function `MR::findFilter`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRIOFilter> FindFilter(MR.Std.Const_Vector_MRIOFilter filters, ReadOnlySpan<char> extension)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findFilter", ExactSpelling = true)]
        extern static MR.Std.Optional_MRIOFilter._Underlying *__MR_findFilter(MR.Std.Const_Vector_MRIOFilter._Underlying *filters, byte *extension, byte *extension_end);
        byte[] __bytes_extension = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(extension.Length)];
        int __len_extension = System.Text.Encoding.UTF8.GetBytes(extension, __bytes_extension);
        fixed (byte *__ptr_extension = __bytes_extension)
        {
            return MR.Misc.Move(new MR.Std.Optional_MRIOFilter(__MR_findFilter(filters._UnderlyingPtr, __ptr_extension, __ptr_extension + __len_extension), is_owning: true));
        }
    }
}
