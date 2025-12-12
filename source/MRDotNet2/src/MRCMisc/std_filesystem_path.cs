public static partial class MR
{
    public static partial class Std
    {
        public static partial class Filesystem
        {
            /// Stores a filesystem path.
            /// This is the const half of the class.
            public class Const_Path : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Path(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_Destroy", ExactSpelling = true)]
                    extern static void __MR_std_filesystem_path_Destroy(_Underlying *_this);
                    __MR_std_filesystem_path_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Path() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Path() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_std_filesystem_path_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_filesystem_path_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Const_Path(MR.Std.Filesystem._ByValue_Path other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_std_filesystem_path_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Filesystem.Path._Underlying *other);
                    _UnderlyingPtr = __MR_std_filesystem_path_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
                }

                /// Constructs a new instance.
                public unsafe Const_Path(ReadOnlySpan<char> other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_std_filesystem_path_ConstructFrom(byte *other, byte *other_end);
                    byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                    int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                    fixed (byte *__ptr_other = __bytes_other)
                    {
                        _UnderlyingPtr = __MR_std_filesystem_path_ConstructFrom(__ptr_other, __ptr_other + __len_other);
                    }
                }

                /// Constructs a new instance.
                public static unsafe implicit operator Const_Path(ReadOnlySpan<char> other) {return new(other);}
                public static unsafe implicit operator Const_Path(string other) {return new(other);}

                /// Get the contents as a UTF8-encoded string.
                public unsafe MR.Misc._Moved<MR.Std.String> GetString()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_GetString", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_std_filesystem_path_GetString(_Underlying *_this);
                    return MR.Misc.Move(new MR.Std.String(__MR_std_filesystem_path_GetString(_UnderlyingPtr), is_owning: true));
                }

                // Custom extras:

                public static unsafe implicit operator string(MR.Std.Filesystem.Const_Path self)
                {
                    return self.GetString().Value;
                }
            }

            /// Stores a filesystem path.
            /// This is the non-const half of the class.
            public class Path : Const_Path
            {
                internal unsafe Path(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Path() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_std_filesystem_path_DefaultConstruct();
                    _UnderlyingPtr = __MR_std_filesystem_path_DefaultConstruct();
                }

                /// Constructs a copy of another instance. The source remains alive.
                public unsafe Path(MR.Std.Filesystem._ByValue_Path other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_std_filesystem_path_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Filesystem.Path._Underlying *other);
                    _UnderlyingPtr = __MR_std_filesystem_path_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
                }

                /// Constructs a new instance.
                public unsafe Path(ReadOnlySpan<char> other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_std_filesystem_path_ConstructFrom(byte *other, byte *other_end);
                    byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                    int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                    fixed (byte *__ptr_other = __bytes_other)
                    {
                        _UnderlyingPtr = __MR_std_filesystem_path_ConstructFrom(__ptr_other, __ptr_other + __len_other);
                    }
                }

                /// Constructs a new instance.
                public static unsafe implicit operator Path(ReadOnlySpan<char> other) {return new(other);}
                public static unsafe implicit operator Path(string other) {return new(other);}

                /// Assigns the contents from another instance. Both objects remain alive after the call.
                public unsafe void Assign(MR.Std.Filesystem._ByValue_Path other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_AssignFromAnother", ExactSpelling = true)]
                    extern static void __MR_std_filesystem_path_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Filesystem.Path._Underlying *other);
                    __MR_std_filesystem_path_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
                }

                /// Assigns the contents.
                public unsafe void Assign(ReadOnlySpan<char> other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_filesystem_path_AssignFrom", ExactSpelling = true)]
                    extern static void __MR_std_filesystem_path_AssignFrom(_Underlying *_this, byte *other, byte *other_end);
                    byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                    int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                    fixed (byte *__ptr_other = __bytes_other)
                    {
                        __MR_std_filesystem_path_AssignFrom(_UnderlyingPtr, __ptr_other, __ptr_other + __len_other);
                    }
                }

                // Custom extras:

                public static unsafe implicit operator string(MR.Std.Filesystem.Path self)
                {
                    return self.GetString().Value;
                }
            }

            /// This is used as a function parameter when the underlying function receives `Path` by value.
            /// Usage:
            /// * Pass `new()` to default-construct the instance.
            /// * Pass an instance of `Path`/`Const_Path` to copy it into the function.
            /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
            ///   Be careful if your input isn't a unique reference to this object.
            /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
            public class _ByValue_Path
            {
                internal readonly Const_Path? Value;
                internal readonly MR.Misc._PassBy PassByMode;
                public _ByValue_Path() {PassByMode = MR.Misc._PassBy.default_construct;}
                public _ByValue_Path(Const_Path new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
                public static implicit operator _ByValue_Path(Const_Path arg) {return new(arg);}
                public _ByValue_Path(MR.Misc._Moved<Path> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
                public static implicit operator _ByValue_Path(MR.Misc._Moved<Path> arg) {return new(arg);}

                /// Constructs a new instance.
                public static unsafe implicit operator _ByValue_Path(ReadOnlySpan<char> other) {return new MR.Std.Filesystem.Path(other);}
                public static unsafe implicit operator _ByValue_Path(string other) {return new(other);}
            }

            /// This is used for optional parameters of class `Path` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Path`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Path`/`Const_Path` directly.
            public class _InOptMut_Path
            {
                public Path? Opt;

                public _InOptMut_Path() {}
                public _InOptMut_Path(Path value) {Opt = value;}
                public static implicit operator _InOptMut_Path(Path value) {return new(value);}
            }

            /// This is used for optional parameters of class `Path` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Path`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Path`/`Const_Path` to pass it to the function.
            public class _InOptConst_Path
            {
                public Const_Path? Opt;

                public _InOptConst_Path() {}
                public _InOptConst_Path(Const_Path value) {Opt = value;}
                public static implicit operator _InOptConst_Path(Const_Path value) {return new(value);}

                /// Constructs a new instance.
                public static unsafe implicit operator _InOptConst_Path(ReadOnlySpan<char> other) {return new MR.Std.Filesystem.Path(other);}
                public static unsafe implicit operator _InOptConst_Path(string other) {return new(other);}
            }
        }
    }
}
