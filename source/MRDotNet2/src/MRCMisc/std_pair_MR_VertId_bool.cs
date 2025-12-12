public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::VertId` and `MR::VertId`.
        /// This is the const half of the class.
        public class Const_Pair_MRVertId_Bool : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRVertId_Bool(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_VertId_bool_Destroy(_Underlying *_this);
                __MR_std_pair_MR_VertId_bool_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRVertId_Bool() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRVertId_Bool() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_std_pair_MR_VertId_bool_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_VertId_bool_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRVertId_Bool(MR.Std.Const_Pair_MRVertId_Bool other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_std_pair_MR_VertId_bool_ConstructFromAnother(MR.Std.Pair_MRVertId_Bool._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_bool_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRVertId_Bool(MR.VertId first, bool second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_std_pair_MR_VertId_bool_Construct(MR.VertId first, byte second);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_bool_Construct(first, second ? (byte)1 : (byte)0);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_VertId First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_First", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_std_pair_MR_VertId_bool_First(_Underlying *_this);
                return new(__MR_std_pair_MR_VertId_bool_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe bool Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_Second", ExactSpelling = true)]
                extern static bool *__MR_std_pair_MR_VertId_bool_Second(_Underlying *_this);
                return *__MR_std_pair_MR_VertId_bool_Second(_UnderlyingPtr);
            }
        }

        /// Stores two objects: `MR::VertId` and `MR::VertId`.
        /// This is the non-const half of the class.
        public class Pair_MRVertId_Bool : Const_Pair_MRVertId_Bool
        {
            internal unsafe Pair_MRVertId_Bool(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRVertId_Bool() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_std_pair_MR_VertId_bool_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_VertId_bool_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRVertId_Bool(MR.Std.Const_Pair_MRVertId_Bool other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_std_pair_MR_VertId_bool_ConstructFromAnother(MR.Std.Pair_MRVertId_Bool._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_bool_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_MRVertId_Bool other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_VertId_bool_AssignFromAnother(_Underlying *_this, MR.Std.Pair_MRVertId_Bool._Underlying *other);
                __MR_std_pair_MR_VertId_bool_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRVertId_Bool(MR.VertId first, bool second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_Bool._Underlying *__MR_std_pair_MR_VertId_bool_Construct(MR.VertId first, byte second);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_bool_Construct(first, second ? (byte)1 : (byte)0);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Mut_VertId MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_MutableFirst", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_std_pair_MR_VertId_bool_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_VertId_bool_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe ref bool MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_bool_MutableSecond", ExactSpelling = true)]
                extern static bool *__MR_std_pair_MR_VertId_bool_MutableSecond(_Underlying *_this);
                return ref *__MR_std_pair_MR_VertId_bool_MutableSecond(_UnderlyingPtr);
            }
        }

        /// This is used for optional parameters of class `Pair_MRVertId_Bool` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRVertId_Bool`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVertId_Bool`/`Const_Pair_MRVertId_Bool` directly.
        public class _InOptMut_Pair_MRVertId_Bool
        {
            public Pair_MRVertId_Bool? Opt;

            public _InOptMut_Pair_MRVertId_Bool() {}
            public _InOptMut_Pair_MRVertId_Bool(Pair_MRVertId_Bool value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRVertId_Bool(Pair_MRVertId_Bool value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRVertId_Bool` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRVertId_Bool`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVertId_Bool`/`Const_Pair_MRVertId_Bool` to pass it to the function.
        public class _InOptConst_Pair_MRVertId_Bool
        {
            public Const_Pair_MRVertId_Bool? Opt;

            public _InOptConst_Pair_MRVertId_Bool() {}
            public _InOptConst_Pair_MRVertId_Bool(Const_Pair_MRVertId_Bool value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRVertId_Bool(Const_Pair_MRVertId_Bool value) {return new(value);}
        }
    }
}
