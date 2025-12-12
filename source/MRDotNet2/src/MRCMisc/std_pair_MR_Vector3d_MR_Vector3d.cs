public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::Vector3d` and `MR::Vector3d`.
        /// This is the const half of the class.
        public class Const_Pair_MRVector3d_MRVector3d : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRVector3d_MRVector3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_Vector3d_MR_Vector3d_Destroy(_Underlying *_this);
                __MR_std_pair_MR_Vector3d_MR_Vector3d_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRVector3d_MRVector3d() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRVector3d_MRVector3d() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_Vector3d_MR_Vector3d_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRVector3d_MRVector3d(MR.Std.Const_Pair_MRVector3d_MRVector3d other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_ConstructFromAnother(MR.Std.Pair_MRVector3d_MRVector3d._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3d_MR_Vector3d_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRVector3d_MRVector3d(MR.Vector3d first, MR.Vector3d second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_Construct(MR.Vector3d first, MR.Vector3d second);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3d_MR_Vector3d_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_Vector3d First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_First", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_First(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3d_MR_Vector3d_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_Vector3d Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_Second", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3d_MR_Vector3d_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::Vector3d` and `MR::Vector3d`.
        /// This is the non-const half of the class.
        public class Pair_MRVector3d_MRVector3d : Const_Pair_MRVector3d_MRVector3d
        {
            internal unsafe Pair_MRVector3d_MRVector3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRVector3d_MRVector3d() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_Vector3d_MR_Vector3d_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRVector3d_MRVector3d(MR.Std.Const_Pair_MRVector3d_MRVector3d other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_ConstructFromAnother(MR.Std.Pair_MRVector3d_MRVector3d._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3d_MR_Vector3d_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_MRVector3d_MRVector3d other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_Vector3d_MR_Vector3d_AssignFromAnother(_Underlying *_this, MR.Std.Pair_MRVector3d_MRVector3d._Underlying *other);
                __MR_std_pair_MR_Vector3d_MR_Vector3d_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRVector3d_MRVector3d(MR.Vector3d first, MR.Vector3d second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_Construct(MR.Vector3d first, MR.Vector3d second);
                _UnderlyingPtr = __MR_std_pair_MR_Vector3d_MR_Vector3d_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Mut_Vector3d MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_MutableFirst", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3d_MR_Vector3d_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.Mut_Vector3d MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Vector3d_MR_Vector3d_MutableSecond", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_std_pair_MR_Vector3d_MR_Vector3d_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_Vector3d_MR_Vector3d_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Pair_MRVector3d_MRVector3d` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRVector3d_MRVector3d`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVector3d_MRVector3d`/`Const_Pair_MRVector3d_MRVector3d` directly.
        public class _InOptMut_Pair_MRVector3d_MRVector3d
        {
            public Pair_MRVector3d_MRVector3d? Opt;

            public _InOptMut_Pair_MRVector3d_MRVector3d() {}
            public _InOptMut_Pair_MRVector3d_MRVector3d(Pair_MRVector3d_MRVector3d value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRVector3d_MRVector3d(Pair_MRVector3d_MRVector3d value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRVector3d_MRVector3d` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRVector3d_MRVector3d`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRVector3d_MRVector3d`/`Const_Pair_MRVector3d_MRVector3d` to pass it to the function.
        public class _InOptConst_Pair_MRVector3d_MRVector3d
        {
            public Const_Pair_MRVector3d_MRVector3d? Opt;

            public _InOptConst_Pair_MRVector3d_MRVector3d() {}
            public _InOptConst_Pair_MRVector3d_MRVector3d(Const_Pair_MRVector3d_MRVector3d value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRVector3d_MRVector3d(Const_Pair_MRVector3d_MRVector3d value) {return new(value);}
        }
    }
}
