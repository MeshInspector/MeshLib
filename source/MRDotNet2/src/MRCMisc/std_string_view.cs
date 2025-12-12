public static partial class MR
{
    public static partial class Std
    {
        /// A non-owning string view. Not necessarily null-terminated.
        /// This is the const half of the class.
        public class Const_StringView : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_StringView(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_Destroy", ExactSpelling = true)]
                extern static void __MR_std_string_view_Destroy(_Underlying *_this);
                __MR_std_string_view_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_StringView() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_StringView() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_std_string_view_DefaultConstruct();
                _UnderlyingPtr = __MR_std_string_view_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_StringView(MR.Std.Const_StringView other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_std_string_view_ConstructFromAnother(MR.Std.StringView._Underlying *other);
                _UnderlyingPtr = __MR_std_string_view_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_StringView(ReadOnlySpan<char> other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_std_string_view_ConstructFrom(byte *other, byte *other_end);
                byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                fixed (byte *__ptr_other = __bytes_other)
                {
                    _UnderlyingPtr = __MR_std_string_view_ConstructFrom(__ptr_other, __ptr_other + __len_other);
                }
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_StringView(ReadOnlySpan<char> other) {return new(other);}
            public static unsafe implicit operator Const_StringView(string other) {return new(other);}

            /// The number of characters in the string.
            public unsafe ulong Size()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_Size", ExactSpelling = true)]
                extern static ulong __MR_std_string_view_Size(_Underlying *_this);
                return __MR_std_string_view_Size(_UnderlyingPtr);
            }

            /// Returns the string contents, NOT necessarily null-terminated.
            /// Returns a read-only pointer.
            public unsafe byte *Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_Data", ExactSpelling = true)]
                extern static byte *__MR_std_string_view_Data(_Underlying *_this);
                return __MR_std_string_view_Data(_UnderlyingPtr);
            }

            /// Returns a pointer to the end of string. Not dereferencable.
            /// Returns a read-only pointer.
            public unsafe byte *DataEnd()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_DataEnd", ExactSpelling = true)]
                extern static byte *__MR_std_string_view_DataEnd(_Underlying *_this);
                return __MR_std_string_view_DataEnd(_UnderlyingPtr);
            }

            // Custom extras:

            public static unsafe implicit operator ReadOnlySpan<byte>(MR.Std.Const_StringView self)
            {
                return new(self.Data(), checked((int)self.Size()));
            }

            public static unsafe implicit operator string(MR.Std.Const_StringView self)
            {
                return System.Text.Encoding.UTF8.GetString(self.Data(), checked((int)self.Size()));
            }
        }

        /// A non-owning string view. Not necessarily null-terminated.
        /// This is the non-const half of the class.
        public class StringView : Const_StringView
        {
            internal unsafe StringView(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe StringView() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_std_string_view_DefaultConstruct();
                _UnderlyingPtr = __MR_std_string_view_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe StringView(MR.Std.Const_StringView other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_std_string_view_ConstructFromAnother(MR.Std.StringView._Underlying *other);
                _UnderlyingPtr = __MR_std_string_view_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe StringView(ReadOnlySpan<char> other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_std_string_view_ConstructFrom(byte *other, byte *other_end);
                byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                fixed (byte *__ptr_other = __bytes_other)
                {
                    _UnderlyingPtr = __MR_std_string_view_ConstructFrom(__ptr_other, __ptr_other + __len_other);
                }
            }

            /// Constructs a new instance.
            public static unsafe implicit operator StringView(ReadOnlySpan<char> other) {return new(other);}
            public static unsafe implicit operator StringView(string other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_StringView other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_string_view_AssignFromAnother(_Underlying *_this, MR.Std.StringView._Underlying *other);
                __MR_std_string_view_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(ReadOnlySpan<char> other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_string_view_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_string_view_AssignFrom(_Underlying *_this, byte *other, byte *other_end);
                byte[] __bytes_other = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(other.Length)];
                int __len_other = System.Text.Encoding.UTF8.GetBytes(other, __bytes_other);
                fixed (byte *__ptr_other = __bytes_other)
                {
                    __MR_std_string_view_AssignFrom(_UnderlyingPtr, __ptr_other, __ptr_other + __len_other);
                }
            }
        }

        /// This is used for optional parameters of class `StringView` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_StringView`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `StringView`/`Const_StringView` directly.
        public class _InOptMut_StringView
        {
            public StringView? Opt;

            public _InOptMut_StringView() {}
            public _InOptMut_StringView(StringView value) {Opt = value;}
            public static implicit operator _InOptMut_StringView(StringView value) {return new(value);}
        }

        /// This is used for optional parameters of class `StringView` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_StringView`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `StringView`/`Const_StringView` to pass it to the function.
        public class _InOptConst_StringView
        {
            public Const_StringView? Opt;

            public _InOptConst_StringView() {}
            public _InOptConst_StringView(Const_StringView value) {Opt = value;}
            public static implicit operator _InOptConst_StringView(Const_StringView value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_StringView(ReadOnlySpan<char> other) {return new MR.Std.StringView(other);}
            public static unsafe implicit operator _InOptConst_StringView(string other) {return new(other);}
        }
    }
}
