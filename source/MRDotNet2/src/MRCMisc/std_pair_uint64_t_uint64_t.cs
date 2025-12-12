public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR_uint64_t` and `MR_uint64_t`.
        /// This is the const half of the class.
        public class Const_Pair_MRUint64T_MRUint64T : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRUint64T_MRUint64T(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_uint64_t_uint64_t_Destroy(_Underlying *_this);
                __MR_std_pair_uint64_t_uint64_t_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRUint64T_MRUint64T() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRUint64T_MRUint64T() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_std_pair_uint64_t_uint64_t_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_uint64_t_uint64_t_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRUint64T_MRUint64T(MR.Std.Const_Pair_MRUint64T_MRUint64T other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_std_pair_uint64_t_uint64_t_ConstructFromAnother(MR.Std.Pair_MRUint64T_MRUint64T._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_uint64_t_uint64_t_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRUint64T_MRUint64T(ulong first, ulong second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_std_pair_uint64_t_uint64_t_Construct(ulong first, ulong second);
                _UnderlyingPtr = __MR_std_pair_uint64_t_uint64_t_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe ulong First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_First", ExactSpelling = true)]
                extern static ulong *__MR_std_pair_uint64_t_uint64_t_First(_Underlying *_this);
                return *__MR_std_pair_uint64_t_uint64_t_First(_UnderlyingPtr);
            }

            /// The second of the two elements, read-only.
            public unsafe ulong Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_Second", ExactSpelling = true)]
                extern static ulong *__MR_std_pair_uint64_t_uint64_t_Second(_Underlying *_this);
                return *__MR_std_pair_uint64_t_uint64_t_Second(_UnderlyingPtr);
            }
        }

        /// Stores two objects: `MR_uint64_t` and `MR_uint64_t`.
        /// This is the non-const half of the class.
        public class Pair_MRUint64T_MRUint64T : Const_Pair_MRUint64T_MRUint64T
        {
            internal unsafe Pair_MRUint64T_MRUint64T(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRUint64T_MRUint64T() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_std_pair_uint64_t_uint64_t_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_uint64_t_uint64_t_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRUint64T_MRUint64T(MR.Std.Const_Pair_MRUint64T_MRUint64T other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_std_pair_uint64_t_uint64_t_ConstructFromAnother(MR.Std.Pair_MRUint64T_MRUint64T._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_uint64_t_uint64_t_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_MRUint64T_MRUint64T other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_uint64_t_uint64_t_AssignFromAnother(_Underlying *_this, MR.Std.Pair_MRUint64T_MRUint64T._Underlying *other);
                __MR_std_pair_uint64_t_uint64_t_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRUint64T_MRUint64T(ulong first, ulong second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_std_pair_uint64_t_uint64_t_Construct(ulong first, ulong second);
                _UnderlyingPtr = __MR_std_pair_uint64_t_uint64_t_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe ref ulong MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_MutableFirst", ExactSpelling = true)]
                extern static ulong *__MR_std_pair_uint64_t_uint64_t_MutableFirst(_Underlying *_this);
                return ref *__MR_std_pair_uint64_t_uint64_t_MutableFirst(_UnderlyingPtr);
            }

            /// The second of the two elements, mutable.
            public unsafe ref ulong MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_uint64_t_uint64_t_MutableSecond", ExactSpelling = true)]
                extern static ulong *__MR_std_pair_uint64_t_uint64_t_MutableSecond(_Underlying *_this);
                return ref *__MR_std_pair_uint64_t_uint64_t_MutableSecond(_UnderlyingPtr);
            }
        }

        /// This is used for optional parameters of class `Pair_MRUint64T_MRUint64T` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRUint64T_MRUint64T`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRUint64T_MRUint64T`/`Const_Pair_MRUint64T_MRUint64T` directly.
        public class _InOptMut_Pair_MRUint64T_MRUint64T
        {
            public Pair_MRUint64T_MRUint64T? Opt;

            public _InOptMut_Pair_MRUint64T_MRUint64T() {}
            public _InOptMut_Pair_MRUint64T_MRUint64T(Pair_MRUint64T_MRUint64T value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRUint64T_MRUint64T(Pair_MRUint64T_MRUint64T value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRUint64T_MRUint64T` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRUint64T_MRUint64T`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRUint64T_MRUint64T`/`Const_Pair_MRUint64T_MRUint64T` to pass it to the function.
        public class _InOptConst_Pair_MRUint64T_MRUint64T
        {
            public Const_Pair_MRUint64T_MRUint64T? Opt;

            public _InOptConst_Pair_MRUint64T_MRUint64T() {}
            public _InOptConst_Pair_MRUint64T_MRUint64T(Const_Pair_MRUint64T_MRUint64T value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRUint64T_MRUint64T(Const_Pair_MRUint64T_MRUint64T value) {return new(value);}
        }
    }
}
