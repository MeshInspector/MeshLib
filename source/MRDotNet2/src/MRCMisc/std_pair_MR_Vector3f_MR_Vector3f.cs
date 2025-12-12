public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::Vector3f` and `MR::Vector3f`.
        /// This is the const half of the class.
        public class Const_Pair_MRVector3f_MRVector3f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRVector3f_MRVector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_Vector3f_MR_Vector3f_Destroy(_Underlying *_this);
                __MR_std_pair_MR_Vector3f_MR_Vector3f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRVector3f_MRVector3f() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRVector3f_MRVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_Vector3f_MR_Vector3f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRVector3f_MRVector3f(MR.Std.Const_Pair_MRVector3f_MRVector3f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_ConstructFromAnother(MR.Std.Pair_MRVector3f_MRVector3f._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3f_MR_Vector3f_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRVector3f_MRVector3f(MR.Vector3f first, MR.Vector3f second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_Construct(MR.Vector3f first, MR.Vector3f second);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3f_MR_Vector3f_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_Vector3f First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_First", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_First(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3f_MR_Vector3f_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_Vector3f Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_Second", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3f_MR_Vector3f_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::Vector3f` and `MR::Vector3f`.
        /// This is the non-const half of the class.
        public class Pair_MRVector3f_MRVector3f : Const_Pair_MRVector3f_MRVector3f
        {
            internal unsafe Pair_MRVector3f_MRVector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRVector3f_MRVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_Vector3f_MR_Vector3f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRVector3f_MRVector3f(MR.Std.Const_Pair_MRVector3f_MRVector3f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_ConstructFromAnother(MR.Std.Pair_MRVector3f_MRVector3f._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3f_MR_Vector3f_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_MRVector3f_MRVector3f other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_Vector3f_MR_Vector3f_AssignFromAnother(_Underlying *_this, MR.Std.Pair_MRVector3f_MRVector3f._Underlying *other);
                __MR_std_pair_MR_Vector3f_MR_Vector3f_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRVector3f_MRVector3f(MR.Vector3f first, MR.Vector3f second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_Construct(MR.Vector3f first, MR.Vector3f second);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3f_MR_Vector3f_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Mut_Vector3f MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_MutableFirst", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3f_MR_Vector3f_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.Mut_Vector3f MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3f_MR_Vector3f_MutableSecond", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_std_pair_MR_Vector3f_MR_Vector3f_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3f_MR_Vector3f_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Pair_MRVector3f_MRVector3f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRVector3f_MRVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVector3f_MRVector3f`/`Const_Pair_MRVector3f_MRVector3f` directly.
        public class _InOptMut_Pair_MRVector3f_MRVector3f
        {
            public Pair_MRVector3f_MRVector3f? Opt;

            public _InOptMut_Pair_MRVector3f_MRVector3f() {}
            public _InOptMut_Pair_MRVector3f_MRVector3f(Pair_MRVector3f_MRVector3f value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRVector3f_MRVector3f(Pair_MRVector3f_MRVector3f value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRVector3f_MRVector3f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRVector3f_MRVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVector3f_MRVector3f`/`Const_Pair_MRVector3f_MRVector3f` to pass it to the function.
        public class _InOptConst_Pair_MRVector3f_MRVector3f
        {
            public Const_Pair_MRVector3f_MRVector3f? Opt;

            public _InOptConst_Pair_MRVector3f_MRVector3f() {}
            public _InOptConst_Pair_MRVector3f_MRVector3f(Const_Pair_MRVector3f_MRVector3f value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRVector3f_MRVector3f(Const_Pair_MRVector3f_MRVector3f value) {return new(value);}
        }
    }
}
