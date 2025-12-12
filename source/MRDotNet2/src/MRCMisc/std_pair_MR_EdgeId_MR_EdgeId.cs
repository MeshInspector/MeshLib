public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::EdgeId` and `MR::EdgeId`.
        /// This is the const half of the class.
        public class Const_Pair_MREdgeId_MREdgeId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MREdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_EdgeId_MR_EdgeId_Destroy(_Underlying *_this);
                __MR_std_pair_MR_EdgeId_MR_EdgeId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MREdgeId_MREdgeId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MREdgeId_MREdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MREdgeId_MREdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MREdgeId_MREdgeId(MR.Std.Const_Pair_MREdgeId_MREdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MREdgeId_MREdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Std.Pair_MREdgeId_MREdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_EdgeId_MR_EdgeId_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MREdgeId_MREdgeId(MR.EdgeId first, MR.EdgeId second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MREdgeId_MREdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_Construct(MR.EdgeId first, MR.EdgeId second);
                _UnderlyingPtr = __MR_std_pair_MR_EdgeId_MR_EdgeId_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_EdgeId First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_First", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_First(_Underlying *_this);
                return new(__MR_std_pair_MR_EdgeId_MR_EdgeId_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_EdgeId Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_Second", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_EdgeId_MR_EdgeId_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::EdgeId` and `MR::EdgeId`.
        /// This is the non-const half of the class.
        public class Pair_MREdgeId_MREdgeId : Const_Pair_MREdgeId_MREdgeId
        {
            internal unsafe Pair_MREdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MREdgeId_MREdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MREdgeId_MREdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MREdgeId_MREdgeId(MR.Std.Const_Pair_MREdgeId_MREdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MREdgeId_MREdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Std.Pair_MREdgeId_MREdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_EdgeId_MR_EdgeId_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_MREdgeId_MREdgeId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_EdgeId_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.Std.Pair_MREdgeId_MREdgeId._Underlying *other);
                __MR_std_pair_MR_EdgeId_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MREdgeId_MREdgeId(MR.EdgeId first, MR.EdgeId second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MREdgeId_MREdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_Construct(MR.EdgeId first, MR.EdgeId second);
                _UnderlyingPtr = __MR_std_pair_MR_EdgeId_MR_EdgeId_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Mut_EdgeId MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_MutableFirst", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_EdgeId_MR_EdgeId_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.Mut_EdgeId MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_EdgeId_MR_EdgeId_MutableSecond", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_std_pair_MR_EdgeId_MR_EdgeId_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_EdgeId_MR_EdgeId_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Pair_MREdgeId_MREdgeId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MREdgeId_MREdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MREdgeId_MREdgeId`/`Const_Pair_MREdgeId_MREdgeId` directly.
        public class _InOptMut_Pair_MREdgeId_MREdgeId
        {
            public Pair_MREdgeId_MREdgeId? Opt;

            public _InOptMut_Pair_MREdgeId_MREdgeId() {}
            public _InOptMut_Pair_MREdgeId_MREdgeId(Pair_MREdgeId_MREdgeId value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MREdgeId_MREdgeId(Pair_MREdgeId_MREdgeId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MREdgeId_MREdgeId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MREdgeId_MREdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MREdgeId_MREdgeId`/`Const_Pair_MREdgeId_MREdgeId` to pass it to the function.
        public class _InOptConst_Pair_MREdgeId_MREdgeId
        {
            public Const_Pair_MREdgeId_MREdgeId? Opt;

            public _InOptConst_Pair_MREdgeId_MREdgeId() {}
            public _InOptConst_Pair_MREdgeId_MREdgeId(Const_Pair_MREdgeId_MREdgeId value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MREdgeId_MREdgeId(Const_Pair_MREdgeId_MREdgeId value) {return new(value);}
        }
    }
}
