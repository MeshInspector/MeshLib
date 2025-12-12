public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::VertId` and `MR::VertId`.
        /// This is the const half of the class.
        public class Const_Pair_MRVertId_MRVertId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRVertId_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_VertId_MR_VertId_Destroy(_Underlying *_this);
                __MR_std_pair_MR_VertId_MR_VertId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRVertId_MRVertId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRVertId_MRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_VertId_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRVertId_MRVertId(MR.Std.Const_Pair_MRVertId_MRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_ConstructFromAnother(MR.Std.Pair_MRVertId_MRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_MR_VertId_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRVertId_MRVertId(MR.VertId first, MR.VertId second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_Construct(MR.VertId first, MR.VertId second);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_MR_VertId_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_VertId First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_First", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_First(_Underlying *_this);
                return new(__MR_std_pair_MR_VertId_MR_VertId_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_VertId Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_Second", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_VertId_MR_VertId_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::VertId` and `MR::VertId`.
        /// This is the non-const half of the class.
        public class Pair_MRVertId_MRVertId : Const_Pair_MRVertId_MRVertId
        {
            internal unsafe Pair_MRVertId_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRVertId_MRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_VertId_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRVertId_MRVertId(MR.Std.Const_Pair_MRVertId_MRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_ConstructFromAnother(MR.Std.Pair_MRVertId_MRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_MR_VertId_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_MRVertId_MRVertId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_VertId_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Std.Pair_MRVertId_MRVertId._Underlying *other);
                __MR_std_pair_MR_VertId_MR_VertId_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRVertId_MRVertId(MR.VertId first, MR.VertId second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_Construct(MR.VertId first, MR.VertId second);
                _UnderlyingPtr = __MR_std_pair_MR_VertId_MR_VertId_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Mut_VertId MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_MutableFirst", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_VertId_MR_VertId_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.Mut_VertId MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_VertId_MR_VertId_MutableSecond", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_std_pair_MR_VertId_MR_VertId_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_VertId_MR_VertId_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Pair_MRVertId_MRVertId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRVertId_MRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVertId_MRVertId`/`Const_Pair_MRVertId_MRVertId` directly.
        public class _InOptMut_Pair_MRVertId_MRVertId
        {
            public Pair_MRVertId_MRVertId? Opt;

            public _InOptMut_Pair_MRVertId_MRVertId() {}
            public _InOptMut_Pair_MRVertId_MRVertId(Pair_MRVertId_MRVertId value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRVertId_MRVertId(Pair_MRVertId_MRVertId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRVertId_MRVertId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRVertId_MRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVertId_MRVertId`/`Const_Pair_MRVertId_MRVertId` to pass it to the function.
        public class _InOptConst_Pair_MRVertId_MRVertId
        {
            public Const_Pair_MRVertId_MRVertId? Opt;

            public _InOptConst_Pair_MRVertId_MRVertId() {}
            public _InOptConst_Pair_MRVertId_MRVertId(Const_Pair_MRVertId_MRVertId value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRVertId_MRVertId(Const_Pair_MRVertId_MRVertId value) {return new(value);}
        }
    }
}
