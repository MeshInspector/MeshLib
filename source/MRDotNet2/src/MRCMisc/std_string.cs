public static partial class MR
{
    public static partial class Std
    {
        /// A heap-allocated null-terminated string.
        /// This is the const half of the class.
        public class Const_String : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_String(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_Destroy", ExactSpelling = true)]
                extern static void __MR_std_string_Destroy(_Underlying *_this);
                __MR_std_string_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_String() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_String() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_string_DefaultConstruct();
                _UnderlyingPtr = __MR_std_string_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_String(MR.Std._ByValue_String other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.String._Underlying *other);
                _UnderlyingPtr = __MR_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Const_String(ReadOnlySpan<char> other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_string_ConstructFrom(byte *other, byte *other_end);
                byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                fixed (byte *__ptr_other = __bytes_other)
                {
                    _UnderlyingPtr = __MR_std_string_ConstructFrom(__ptr_other, __ptr_other + __len_other);
                }
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_String(ReadOnlySpan<char> other) {return new(other);}
            public static unsafe implicit operator Const_String(string other) {return new(other);}

            /// The number of characters in the string, excluding the null-terminator.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_Size", ExactSpelling = true)]
                extern static ulong __MR_std_string_Size(_Underlying *_this);
                return __MR_std_string_Size(_UnderlyingPtr);
            }

            /// Returns the string contents, which are always null-terminated.
            /// Returns a read-only pointer.
            public unsafe byte *Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_Data", ExactSpelling = true)]
                extern static byte *__MR_std_string_Data(_Underlying *_this);
                return __MR_std_string_Data(_UnderlyingPtr);
            }

            /// Returns a pointer to the end of string, to its null-terminator.
            /// Returns a read-only pointer.
            public unsafe byte *DataEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_DataEnd", ExactSpelling = true)]
                extern static byte *__MR_std_string_DataEnd(_Underlying *_this);
                return __MR_std_string_DataEnd(_UnderlyingPtr);
            }

            // Custom extras:

            public static unsafe implicit operator ReadOnlySpan<byte>(MR.Std.Const_String self)
            {
                return new(self.Data(), checked((int)self.Size()));
            }

            public static unsafe implicit operator string(MR.Std.Const_String self)
            {
                return System.Text.Encoding.UTF8.GetString(self.Data(), checked((int)self.Size()));
            }
        }

        /// A heap-allocated null-terminated string.
        /// This is the non-const half of the class.
        public class String : Const_String
        {
            internal unsafe String(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe String() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_string_DefaultConstruct();
                _UnderlyingPtr = __MR_std_string_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe String(MR.Std._ByValue_String other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.String._Underlying *other);
                _UnderlyingPtr = __MR_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe String(ReadOnlySpan<char> other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_string_ConstructFrom(byte *other, byte *other_end);
                byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                fixed (byte *__ptr_other = __bytes_other)
                {
                    _UnderlyingPtr = __MR_std_string_ConstructFrom(__ptr_other, __ptr_other + __len_other);
                }
            }

            /// Constructs a new instance.
            public static unsafe implicit operator String(ReadOnlySpan<char> other) {return new(other);}
            public static unsafe implicit operator String(string other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_String other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_string_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.String._Underlying *other);
                __MR_std_string_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents.
            public unsafe void Assign(ReadOnlySpan<char> other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_string_AssignFrom(_Underlying *_this, byte *other, byte *other_end);
                byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                fixed (byte *__ptr_other = __bytes_other)
                {
                    __MR_std_string_AssignFrom(_UnderlyingPtr, __ptr_other, __ptr_other + __len_other);
                }
            }

            /// Returns the string contents, which are always null-terminated. This version returns a non-const pointer.
            /// Returns a read-only pointer.
            public unsafe byte *MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_MutableData", ExactSpelling = true)]
                extern static byte *__MR_std_string_MutableData(_Underlying *_this);
                return __MR_std_string_MutableData(_UnderlyingPtr);
            }

            /// Returns a pointer to the end of string, to its null-terminator. This version returns a non-const pointer.
            /// Returns a mutable pointer.
            public unsafe byte *MutableDataEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_MutableDataEnd", ExactSpelling = true)]
                extern static byte *__MR_std_string_MutableDataEnd(_Underlying *_this);
                return __MR_std_string_MutableDataEnd(_UnderlyingPtr);
            }

            // Custom extras:

            public static unsafe implicit operator Span<byte>(MR.Std.String s)
            {
                return new(s.MutableData(), checked((int)s.Size()));
            }
        }

        /// This is used as a function parameter when the underlying function receives `String` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `String`/`Const_String` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_String
        {
            internal readonly Const_String? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_String() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_String(Const_String new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_String(Const_String arg) {return new(arg);}
            public _ByValue_String(MR.Misc._Moved<String> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_String(MR.Misc._Moved<String> arg) {return new(arg);}

            /// Constructs a new instance.
            public static unsafe implicit operator _ByValue_String(ReadOnlySpan<char> other) {return new MR.Std.String(other);}
            public static unsafe implicit operator _ByValue_String(string other) {return new(other);}
        }

        /// This is used as a function parameter when the underlying function receives an optional `String` by value,
        ///   and also has a default argument, meaning it has two different null states.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `String`/`Const_String` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument.
        /// * Pass `MR.Misc.NullOptType` to pass no object.
        public class _ByValueOptOpt_String
        {
            internal readonly Const_String? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValueOptOpt_String() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValueOptOpt_String(Const_String new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValueOptOpt_String(Const_String arg) {return new(arg);}
            public _ByValueOptOpt_String(MR.Misc._Moved<String> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValueOptOpt_String(MR.Misc._Moved<String> arg) {return new(arg);}
            public _ByValueOptOpt_String(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
            public static implicit operator _ByValueOptOpt_String(MR.Misc.NullOptType nullopt) {return new(nullopt);}

            /// Constructs a new instance.
            public static unsafe implicit operator _ByValueOptOpt_String(ReadOnlySpan<char> other) {return new MR.Std.String(other);}
            public static unsafe implicit operator _ByValueOptOpt_String(string other) {return new(other);}
        }

        /// This is used for optional parameters of class `String` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_String`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `String`/`Const_String` directly.
        public class _InOptMut_String
        {
            public String? Opt;

            public _InOptMut_String() {}
            public _InOptMut_String(String value) {Opt = value;}
            public static implicit operator _InOptMut_String(String value) {return new(value);}
        }

        /// This is used for optional parameters of class `String` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_String`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `String`/`Const_String` to pass it to the function.
        public class _InOptConst_String
        {
            public Const_String? Opt;

            public _InOptConst_String() {}
            public _InOptConst_String(Const_String value) {Opt = value;}
            public static implicit operator _InOptConst_String(Const_String value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_String(ReadOnlySpan<char> other) {return new MR.Std.String(other);}
            public static unsafe implicit operator _InOptConst_String(string other) {return new(other);}
        }
    }
}
