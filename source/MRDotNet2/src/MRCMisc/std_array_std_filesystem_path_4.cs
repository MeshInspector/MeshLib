public static partial class MR
{
    public static partial class Std
    {
        /// A fixed-size array of `std::filesystem::path` of size 4.
        /// This is the const half of the class.
        public class Const_Array_StdFilesystemPath_4 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Array_StdFilesystemPath_4(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_Destroy", ExactSpelling = true)]
                extern static void __MR_std_array_std_filesystem_path_4_Destroy(_Underlying *_this);
                __MR_std_array_std_filesystem_path_4_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Array_StdFilesystemPath_4() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Array_StdFilesystemPath_4() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_StdFilesystemPath_4._Underlying *__MR_std_array_std_filesystem_path_4_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_std_filesystem_path_4_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Array_StdFilesystemPath_4(MR.Std._ByValue_Array_StdFilesystemPath_4 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_StdFilesystemPath_4._Underlying *__MR_std_array_std_filesystem_path_4_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Array_StdFilesystemPath_4._Underlying *other);
                _UnderlyingPtr = __MR_std_array_std_filesystem_path_4_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Std.Filesystem.Const_Path At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_At", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_std_array_std_filesystem_path_4_At(_Underlying *_this, ulong i);
                return new(__MR_std_array_std_filesystem_path_4_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, read-only.
            public unsafe MR.Std.Filesystem.Const_Path? Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_Data", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_std_array_std_filesystem_path_4_Data(_Underlying *_this);
                var __ret = __MR_std_array_std_filesystem_path_4_Data(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Filesystem.Const_Path(__ret, is_owning: false) : null;
            }
        }

        /// A fixed-size array of `std::filesystem::path` of size 4.
        /// This is the non-const half of the class.
        public class Array_StdFilesystemPath_4 : Const_Array_StdFilesystemPath_4
        {
            internal unsafe Array_StdFilesystemPath_4(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Array_StdFilesystemPath_4() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_StdFilesystemPath_4._Underlying *__MR_std_array_std_filesystem_path_4_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_std_filesystem_path_4_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Array_StdFilesystemPath_4(MR.Std._ByValue_Array_StdFilesystemPath_4 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_StdFilesystemPath_4._Underlying *__MR_std_array_std_filesystem_path_4_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Array_StdFilesystemPath_4._Underlying *other);
                _UnderlyingPtr = __MR_std_array_std_filesystem_path_4_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Array_StdFilesystemPath_4 other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_array_std_filesystem_path_4_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Array_StdFilesystemPath_4._Underlying *other);
                __MR_std_array_std_filesystem_path_4_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.Std.Filesystem.Path MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_MutableAt", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Path._Underlying *__MR_std_array_std_filesystem_path_4_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_array_std_filesystem_path_4_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, mutable.
            public unsafe MR.Std.Filesystem.Path? MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_std_filesystem_path_4_MutableData", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Path._Underlying *__MR_std_array_std_filesystem_path_4_MutableData(_Underlying *_this);
                var __ret = __MR_std_array_std_filesystem_path_4_MutableData(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Filesystem.Path(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Array_StdFilesystemPath_4` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Array_StdFilesystemPath_4`/`Const_Array_StdFilesystemPath_4` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Array_StdFilesystemPath_4
        {
            internal readonly Const_Array_StdFilesystemPath_4? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Array_StdFilesystemPath_4() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Array_StdFilesystemPath_4(Const_Array_StdFilesystemPath_4 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Array_StdFilesystemPath_4(Const_Array_StdFilesystemPath_4 arg) {return new(arg);}
            public _ByValue_Array_StdFilesystemPath_4(MR.Misc._Moved<Array_StdFilesystemPath_4> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Array_StdFilesystemPath_4(MR.Misc._Moved<Array_StdFilesystemPath_4> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Array_StdFilesystemPath_4` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Array_StdFilesystemPath_4`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_StdFilesystemPath_4`/`Const_Array_StdFilesystemPath_4` directly.
        public class _InOptMut_Array_StdFilesystemPath_4
        {
            public Array_StdFilesystemPath_4? Opt;

            public _InOptMut_Array_StdFilesystemPath_4() {}
            public _InOptMut_Array_StdFilesystemPath_4(Array_StdFilesystemPath_4 value) {Opt = value;}
            public static implicit operator _InOptMut_Array_StdFilesystemPath_4(Array_StdFilesystemPath_4 value) {return new(value);}
        }

        /// This is used for optional parameters of class `Array_StdFilesystemPath_4` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Array_StdFilesystemPath_4`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_StdFilesystemPath_4`/`Const_Array_StdFilesystemPath_4` to pass it to the function.
        public class _InOptConst_Array_StdFilesystemPath_4
        {
            public Const_Array_StdFilesystemPath_4? Opt;

            public _InOptConst_Array_StdFilesystemPath_4() {}
            public _InOptConst_Array_StdFilesystemPath_4(Const_Array_StdFilesystemPath_4 value) {Opt = value;}
            public static implicit operator _InOptConst_Array_StdFilesystemPath_4(Const_Array_StdFilesystemPath_4 value) {return new(value);}
        }
    }
}
