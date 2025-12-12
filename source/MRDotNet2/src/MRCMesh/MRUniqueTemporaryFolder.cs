public static partial class MR
{
    /// helper class to create a temporary folder; the folder will be removed on the object's destruction
    /// Generated from class `MR::UniqueTemporaryFolder`.
    /// This is the const half of the class.
    public class Const_UniqueTemporaryFolder : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UniqueTemporaryFolder(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_Destroy", ExactSpelling = true)]
            extern static void __MR_UniqueTemporaryFolder_Destroy(_Underlying *_this);
            __MR_UniqueTemporaryFolder_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UniqueTemporaryFolder() {Dispose(false);}

        /// creates new folder in temp directory
        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        /// Parameter `onPreTempFolderDelete` defaults to `{}`.
        public unsafe Const_UniqueTemporaryFolder(MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef? onPreTempFolderDelete = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_Construct", ExactSpelling = true)]
            extern static MR.UniqueTemporaryFolder._Underlying *__MR_UniqueTemporaryFolder_Construct(MR.Misc._PassBy onPreTempFolderDelete_pass_by, MR.Std.Function_VoidFuncFromConstStdFilesystemPathRef._Underlying *onPreTempFolderDelete);
            _UnderlyingPtr = __MR_UniqueTemporaryFolder_Construct(onPreTempFolderDelete is not null ? onPreTempFolderDelete.PassByMode : MR.Misc._PassBy.default_arg, onPreTempFolderDelete is not null && onPreTempFolderDelete.Value is not null ? onPreTempFolderDelete.Value._UnderlyingPtr : null);
        }

        /// creates new folder in temp directory
        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        /// Parameter `onPreTempFolderDelete` defaults to `{}`.
        public static unsafe implicit operator Const_UniqueTemporaryFolder(MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef? onPreTempFolderDelete) {return new(onPreTempFolderDelete);}

        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        public unsafe Const_UniqueTemporaryFolder(MR._ByValue_UniqueTemporaryFolder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniqueTemporaryFolder._Underlying *__MR_UniqueTemporaryFolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniqueTemporaryFolder._Underlying *_other);
            _UnderlyingPtr = __MR_UniqueTemporaryFolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from conversion operator `MR::UniqueTemporaryFolder::operator bool`.
        public static unsafe explicit operator bool(MR.Const_UniqueTemporaryFolder _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_UniqueTemporaryFolder_ConvertTo_bool(MR.Const_UniqueTemporaryFolder._Underlying *_this);
            return __MR_UniqueTemporaryFolder_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from conversion operator `MR::UniqueTemporaryFolder::operator const std::filesystem::path &`.
        public static unsafe implicit operator MR.Std.Filesystem.Const_Path(MR.Const_UniqueTemporaryFolder _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_ConvertTo_const_std_filesystem_path_ref", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_UniqueTemporaryFolder_ConvertTo_const_std_filesystem_path_ref(MR.Const_UniqueTemporaryFolder._Underlying *_this);
            return new(__MR_UniqueTemporaryFolder_ConvertTo_const_std_filesystem_path_ref(_this._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UniqueTemporaryFolder::operator/`.
        public static unsafe MR.Misc._Moved<MR.Std.Filesystem.Path> operator/(MR.Const_UniqueTemporaryFolder _this, ReadOnlySpan<char> child)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_UniqueTemporaryFolder_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Path._Underlying *__MR_div_MR_UniqueTemporaryFolder_std_filesystem_path(MR.Const_UniqueTemporaryFolder._Underlying *_this, byte *child, byte *child_end);
            byte[] __bytes_child = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(child.Length)];
            int __len_child = System.Text.Encoding.UTF8.GetBytes(child, __bytes_child);
            fixed (byte *__ptr_child = __bytes_child)
            {
                return MR.Misc.Move(new MR.Std.Filesystem.Path(__MR_div_MR_UniqueTemporaryFolder_std_filesystem_path(_this._UnderlyingPtr, __ptr_child, __ptr_child + __len_child), is_owning: true));
            }
        }
    }

    /// helper class to create a temporary folder; the folder will be removed on the object's destruction
    /// Generated from class `MR::UniqueTemporaryFolder`.
    /// This is the non-const half of the class.
    public class UniqueTemporaryFolder : Const_UniqueTemporaryFolder
    {
        internal unsafe UniqueTemporaryFolder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// creates new folder in temp directory
        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        /// Parameter `onPreTempFolderDelete` defaults to `{}`.
        public unsafe UniqueTemporaryFolder(MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef? onPreTempFolderDelete = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_Construct", ExactSpelling = true)]
            extern static MR.UniqueTemporaryFolder._Underlying *__MR_UniqueTemporaryFolder_Construct(MR.Misc._PassBy onPreTempFolderDelete_pass_by, MR.Std.Function_VoidFuncFromConstStdFilesystemPathRef._Underlying *onPreTempFolderDelete);
            _UnderlyingPtr = __MR_UniqueTemporaryFolder_Construct(onPreTempFolderDelete is not null ? onPreTempFolderDelete.PassByMode : MR.Misc._PassBy.default_arg, onPreTempFolderDelete is not null && onPreTempFolderDelete.Value is not null ? onPreTempFolderDelete.Value._UnderlyingPtr : null);
        }

        /// creates new folder in temp directory
        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        /// Parameter `onPreTempFolderDelete` defaults to `{}`.
        public static unsafe implicit operator UniqueTemporaryFolder(MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef? onPreTempFolderDelete) {return new(onPreTempFolderDelete);}

        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        public unsafe UniqueTemporaryFolder(MR._ByValue_UniqueTemporaryFolder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniqueTemporaryFolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniqueTemporaryFolder._Underlying *__MR_UniqueTemporaryFolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniqueTemporaryFolder._Underlying *_other);
            _UnderlyingPtr = __MR_UniqueTemporaryFolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UniqueTemporaryFolder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UniqueTemporaryFolder`/`Const_UniqueTemporaryFolder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UniqueTemporaryFolder
    {
        internal readonly Const_UniqueTemporaryFolder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UniqueTemporaryFolder() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UniqueTemporaryFolder(Const_UniqueTemporaryFolder new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UniqueTemporaryFolder(Const_UniqueTemporaryFolder arg) {return new(arg);}

        /// creates new folder in temp directory
        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        /// Parameter `onPreTempFolderDelete` defaults to `{}`.
        public static unsafe implicit operator _ByValue_UniqueTemporaryFolder(MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef? onPreTempFolderDelete) {return new MR.UniqueTemporaryFolder(onPreTempFolderDelete);}
    }

    /// This is used for optional parameters of class `UniqueTemporaryFolder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UniqueTemporaryFolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniqueTemporaryFolder`/`Const_UniqueTemporaryFolder` directly.
    public class _InOptMut_UniqueTemporaryFolder
    {
        public UniqueTemporaryFolder? Opt;

        public _InOptMut_UniqueTemporaryFolder() {}
        public _InOptMut_UniqueTemporaryFolder(UniqueTemporaryFolder value) {Opt = value;}
        public static implicit operator _InOptMut_UniqueTemporaryFolder(UniqueTemporaryFolder value) {return new(value);}
    }

    /// This is used for optional parameters of class `UniqueTemporaryFolder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UniqueTemporaryFolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniqueTemporaryFolder`/`Const_UniqueTemporaryFolder` to pass it to the function.
    public class _InOptConst_UniqueTemporaryFolder
    {
        public Const_UniqueTemporaryFolder? Opt;

        public _InOptConst_UniqueTemporaryFolder() {}
        public _InOptConst_UniqueTemporaryFolder(Const_UniqueTemporaryFolder value) {Opt = value;}
        public static implicit operator _InOptConst_UniqueTemporaryFolder(Const_UniqueTemporaryFolder value) {return new(value);}

        /// creates new folder in temp directory
        /// Generated from constructor `MR::UniqueTemporaryFolder::UniqueTemporaryFolder`.
        /// Parameter `onPreTempFolderDelete` defaults to `{}`.
        public static unsafe implicit operator _InOptConst_UniqueTemporaryFolder(MR.Std._ByValue_Function_VoidFuncFromConstStdFilesystemPathRef? onPreTempFolderDelete) {return new MR.UniqueTemporaryFolder(onPreTempFolderDelete);}
    }
}
