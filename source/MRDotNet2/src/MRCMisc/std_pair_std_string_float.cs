public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `std::string` and `std::string`.
        /// This is the const half of the class.
        public class Const_Pair_StdString_Float : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_StdString_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_std_string_float_Destroy(_Underlying *_this);
                __MR_std_pair_std_string_float_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_StdString_Float() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_StdString_Float() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_StdString_Float._Underlying *__MR_std_pair_std_string_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_std_string_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_StdString_Float(MR.Std._ByValue_Pair_StdString_Float other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_StdString_Float._Underlying *__MR_std_pair_std_string_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_StdString_Float._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_std_string_float_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_StdString_Float(ReadOnlySpan<char> first, float second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_StdString_Float._Underlying *__MR_std_pair_std_string_float_Construct(byte *first, byte *first_end, float second);
                byte[] __bytes_first = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(first.Length)];
                int __len_first = System.Text.Encoding.UTF8.GetBytes(first, __bytes_first);
                fixed (byte *__ptr_first = __bytes_first)
                {
                    _UnderlyingPtr = __MR_std_pair_std_string_float_Construct(__ptr_first, __ptr_first + __len_first, second);
                }
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Std.Const_String First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_First", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_std_pair_std_string_float_First(_Underlying *_this);
                return new(__MR_std_pair_std_string_float_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe float Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_Second", ExactSpelling = true)]
                extern static float *__MR_std_pair_std_string_float_Second(_Underlying *_this);
                return *__MR_std_pair_std_string_float_Second(_UnderlyingPtr);
            }
        }

        /// Stores two objects: `std::string` and `std::string`.
        /// This is the non-const half of the class.
        public class Pair_StdString_Float : Const_Pair_StdString_Float
        {
            internal unsafe Pair_StdString_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_StdString_Float() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_StdString_Float._Underlying *__MR_std_pair_std_string_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_std_string_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_StdString_Float(MR.Std._ByValue_Pair_StdString_Float other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_StdString_Float._Underlying *__MR_std_pair_std_string_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_StdString_Float._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_std_string_float_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Pair_StdString_Float other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_std_string_float_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Pair_StdString_Float._Underlying *other);
                __MR_std_pair_std_string_float_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_StdString_Float(ReadOnlySpan<char> first, float second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_StdString_Float._Underlying *__MR_std_pair_std_string_float_Construct(byte *first, byte *first_end, float second);
                byte[] __bytes_first = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(first.Length)];
                int __len_first = System.Text.Encoding.UTF8.GetBytes(first, __bytes_first);
                fixed (byte *__ptr_first = __bytes_first)
                {
                    _UnderlyingPtr = __MR_std_pair_std_string_float_Construct(__ptr_first, __ptr_first + __len_first, second);
                }
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Std.String MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_MutableFirst", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_pair_std_string_float_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_std_string_float_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe ref float MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_std_string_float_MutableSecond", ExactSpelling = true)]
                extern static float *__MR_std_pair_std_string_float_MutableSecond(_Underlying *_this);
                return ref *__MR_std_pair_std_string_float_MutableSecond(_UnderlyingPtr);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Pair_StdString_Float` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Pair_StdString_Float`/`Const_Pair_StdString_Float` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Pair_StdString_Float
        {
            internal readonly Const_Pair_StdString_Float? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Pair_StdString_Float() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Pair_StdString_Float(Const_Pair_StdString_Float new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Pair_StdString_Float(Const_Pair_StdString_Float arg) {return new(arg);}
            public _ByValue_Pair_StdString_Float(MR.Misc._Moved<Pair_StdString_Float> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Pair_StdString_Float(MR.Misc._Moved<Pair_StdString_Float> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Pair_StdString_Float` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_StdString_Float`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_StdString_Float`/`Const_Pair_StdString_Float` directly.
        public class _InOptMut_Pair_StdString_Float
        {
            public Pair_StdString_Float? Opt;

            public _InOptMut_Pair_StdString_Float() {}
            public _InOptMut_Pair_StdString_Float(Pair_StdString_Float value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_StdString_Float(Pair_StdString_Float value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_StdString_Float` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_StdString_Float`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_StdString_Float`/`Const_Pair_StdString_Float` to pass it to the function.
        public class _InOptConst_Pair_StdString_Float
        {
            public Const_Pair_StdString_Float? Opt;

            public _InOptConst_Pair_StdString_Float() {}
            public _InOptConst_Pair_StdString_Float(Const_Pair_StdString_Float value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_StdString_Float(Const_Pair_StdString_Float value) {return new(value);}
        }
    }
}
