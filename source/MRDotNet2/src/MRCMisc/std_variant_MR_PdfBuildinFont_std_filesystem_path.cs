public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 2 objects: `MR::PdfBuildinFont`, `std::filesystem::path`.
        /// This is the const half of the class.
        public class Const_Variant_MRPdfBuildinFont_StdFilesystemPath : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MRPdfBuildinFont_StdFilesystemPath(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Destroy(_Underlying *_this);
                __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MRPdfBuildinFont_StdFilesystemPath() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MRPdfBuildinFont_StdFilesystemPath() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MRPdfBuildinFont_StdFilesystemPath(MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Index(_Underlying *_this);
                return __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::PdfBuildinFont`.
            public unsafe Const_Variant_MRPdfBuildinFont_StdFilesystemPath(MR.PdfBuildinFont value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_MR_PdfBuildinFont", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_MR_PdfBuildinFont(MR.PdfBuildinFont value);
                _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_MR_PdfBuildinFont(value);
            }

            /// Constructs the variant storing the element 1, of type `std::filesystem::path`.
            public unsafe Const_Variant_MRPdfBuildinFont_StdFilesystemPath(ReadOnlySpan<char> value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_std_filesystem_path", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_std_filesystem_path(byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_std_filesystem_path(__ptr_value, __ptr_value + __len_value);
                }
            }

            /// Returns the element 0, of type `MR::PdfBuildinFont`, read-only. If it's not the active element, returns null.
            public unsafe MR.PdfBuildinFont? GetMRPdfBuildinFont()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Get_MR_PdfBuildinFont", ExactSpelling = true)]
                extern static MR.PdfBuildinFont *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Get_MR_PdfBuildinFont(_Underlying *_this);
                var __ret = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Get_MR_PdfBuildinFont(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// Returns the element 1, of type `std::filesystem::path`, read-only. If it's not the active element, returns null.
            public unsafe MR.Std.Filesystem.Const_Path? GetStdFilesystemPath()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Get_std_filesystem_path", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Get_std_filesystem_path(_Underlying *_this);
                var __ret = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_Get_std_filesystem_path(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Filesystem.Const_Path(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 2 objects: `MR::PdfBuildinFont`, `std::filesystem::path`.
        /// This is the non-const half of the class.
        public class Variant_MRPdfBuildinFont_StdFilesystemPath : Const_Variant_MRPdfBuildinFont_StdFilesystemPath
        {
            internal unsafe Variant_MRPdfBuildinFont_StdFilesystemPath(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MRPdfBuildinFont_StdFilesystemPath() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MRPdfBuildinFont_StdFilesystemPath(MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *other);
                __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `MR::PdfBuildinFont`.
            public unsafe Variant_MRPdfBuildinFont_StdFilesystemPath(MR.PdfBuildinFont value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_MR_PdfBuildinFont", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_MR_PdfBuildinFont(MR.PdfBuildinFont value);
                _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_MR_PdfBuildinFont(value);
            }

            /// Constructs the variant storing the element 1, of type `std::filesystem::path`.
            public unsafe Variant_MRPdfBuildinFont_StdFilesystemPath(ReadOnlySpan<char> value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_std_filesystem_path", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_std_filesystem_path(byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    _UnderlyingPtr = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_ConstructAs_std_filesystem_path(__ptr_value, __ptr_value + __len_value);
                }
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::PdfBuildinFont`.
            public unsafe void AssignAsMRPdfBuildinFont(MR.PdfBuildinFont value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignAs_MR_PdfBuildinFont", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignAs_MR_PdfBuildinFont(_Underlying *_this, MR.PdfBuildinFont value);
                __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignAs_MR_PdfBuildinFont(_UnderlyingPtr, value);
            }

            /// Assigns to the variant, making it store the element 1, of type `std::filesystem::path`.
            public unsafe void AssignAsStdFilesystemPath(ReadOnlySpan<char> value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignAs_std_filesystem_path", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignAs_std_filesystem_path(_Underlying *_this, byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_AssignAs_std_filesystem_path(_UnderlyingPtr, __ptr_value, __ptr_value + __len_value);
                }
            }

            /// Returns the element 0, of type `MR::PdfBuildinFont`, mutable. If it's not the active element, returns null.
            public unsafe MR.Misc.Ref<MR.PdfBuildinFont>? GetMutableMRPdfBuildinFont()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_GetMutable_MR_PdfBuildinFont", ExactSpelling = true)]
                extern static MR.PdfBuildinFont *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_GetMutable_MR_PdfBuildinFont(_Underlying *_this);
                var __ret = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_GetMutable_MR_PdfBuildinFont(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<MR.PdfBuildinFont>(__ret) : null;
            }

            /// Returns the element 1, of type `std::filesystem::path`, mutable. If it's not the active element, returns null.
            public unsafe MR.Std.Filesystem.Path? GetMutableStdFilesystemPath()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_GetMutable_std_filesystem_path", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Path._Underlying *__MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_GetMutable_std_filesystem_path(_Underlying *_this);
                var __ret = __MR_std_variant_MR_PdfBuildinFont_std_filesystem_path_GetMutable_std_filesystem_path(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Filesystem.Path(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_MRPdfBuildinFont_StdFilesystemPath` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_MRPdfBuildinFont_StdFilesystemPath`/`Const_Variant_MRPdfBuildinFont_StdFilesystemPath` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath
        {
            internal readonly Const_Variant_MRPdfBuildinFont_StdFilesystemPath? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath(Const_Variant_MRPdfBuildinFont_StdFilesystemPath new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath(Const_Variant_MRPdfBuildinFont_StdFilesystemPath arg) {return new(arg);}
            public _ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath(MR.Misc._Moved<Variant_MRPdfBuildinFont_StdFilesystemPath> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath(MR.Misc._Moved<Variant_MRPdfBuildinFont_StdFilesystemPath> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_MRPdfBuildinFont_StdFilesystemPath` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MRPdfBuildinFont_StdFilesystemPath`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRPdfBuildinFont_StdFilesystemPath`/`Const_Variant_MRPdfBuildinFont_StdFilesystemPath` directly.
        public class _InOptMut_Variant_MRPdfBuildinFont_StdFilesystemPath
        {
            public Variant_MRPdfBuildinFont_StdFilesystemPath? Opt;

            public _InOptMut_Variant_MRPdfBuildinFont_StdFilesystemPath() {}
            public _InOptMut_Variant_MRPdfBuildinFont_StdFilesystemPath(Variant_MRPdfBuildinFont_StdFilesystemPath value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MRPdfBuildinFont_StdFilesystemPath(Variant_MRPdfBuildinFont_StdFilesystemPath value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MRPdfBuildinFont_StdFilesystemPath` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MRPdfBuildinFont_StdFilesystemPath`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRPdfBuildinFont_StdFilesystemPath`/`Const_Variant_MRPdfBuildinFont_StdFilesystemPath` to pass it to the function.
        public class _InOptConst_Variant_MRPdfBuildinFont_StdFilesystemPath
        {
            public Const_Variant_MRPdfBuildinFont_StdFilesystemPath? Opt;

            public _InOptConst_Variant_MRPdfBuildinFont_StdFilesystemPath() {}
            public _InOptConst_Variant_MRPdfBuildinFont_StdFilesystemPath(Const_Variant_MRPdfBuildinFont_StdFilesystemPath value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MRPdfBuildinFont_StdFilesystemPath(Const_Variant_MRPdfBuildinFont_StdFilesystemPath value) {return new(value);}
        }
    }
}
