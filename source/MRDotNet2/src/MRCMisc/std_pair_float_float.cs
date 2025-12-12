public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `float` and `float`.
        /// This is the const half of the class.
        public class Const_Pair_Float_Float : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_Float_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_float_float_Destroy(_Underlying *_this);
                __MR_std_pair_float_float_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_Float_Float() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_Float_Float() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_Float_Float._Underlying *__MR_std_pair_float_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_float_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_Float_Float(MR.Std.Const_Pair_Float_Float other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_Float_Float._Underlying *__MR_std_pair_float_float_ConstructFromAnother(MR.Std.Pair_Float_Float._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_float_float_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_Float_Float(float first, float second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_Float_Float._Underlying *__MR_std_pair_float_float_Construct(float first, float second);
                _UnderlyingPtr = __MR_std_pair_float_float_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe float First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_First", ExactSpelling = true)]
                extern static float *__MR_std_pair_float_float_First(_Underlying *_this);
                return *__MR_std_pair_float_float_First(_UnderlyingPtr);
            }

            /// The second of the two elements, read-only.
            public unsafe float Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_Second", ExactSpelling = true)]
                extern static float *__MR_std_pair_float_float_Second(_Underlying *_this);
                return *__MR_std_pair_float_float_Second(_UnderlyingPtr);
            }
        }

        /// Stores two objects: `float` and `float`.
        /// This is the non-const half of the class.
        public class Pair_Float_Float : Const_Pair_Float_Float
        {
            internal unsafe Pair_Float_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_Float_Float() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_Float_Float._Underlying *__MR_std_pair_float_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_float_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_Float_Float(MR.Std.Const_Pair_Float_Float other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_Float_Float._Underlying *__MR_std_pair_float_float_ConstructFromAnother(MR.Std.Pair_Float_Float._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_float_float_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_Float_Float other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_float_float_AssignFromAnother(_Underlying *_this, MR.Std.Pair_Float_Float._Underlying *other);
                __MR_std_pair_float_float_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_Float_Float(float first, float second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_Float_Float._Underlying *__MR_std_pair_float_float_Construct(float first, float second);
                _UnderlyingPtr = __MR_std_pair_float_float_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe ref float MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_MutableFirst", ExactSpelling = true)]
                extern static float *__MR_std_pair_float_float_MutableFirst(_Underlying *_this);
                return ref *__MR_std_pair_float_float_MutableFirst(_UnderlyingPtr);
            }

            /// The second of the two elements, mutable.
            public unsafe ref float MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_float_float_MutableSecond", ExactSpelling = true)]
                extern static float *__MR_std_pair_float_float_MutableSecond(_Underlying *_this);
                return ref *__MR_std_pair_float_float_MutableSecond(_UnderlyingPtr);
            }
        }

        /// This is used for optional parameters of class `Pair_Float_Float` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_Float_Float`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_Float_Float`/`Const_Pair_Float_Float` directly.
        public class _InOptMut_Pair_Float_Float
        {
            public Pair_Float_Float? Opt;

            public _InOptMut_Pair_Float_Float() {}
            public _InOptMut_Pair_Float_Float(Pair_Float_Float value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_Float_Float(Pair_Float_Float value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_Float_Float` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_Float_Float`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_Float_Float`/`Const_Pair_Float_Float` to pass it to the function.
        public class _InOptConst_Pair_Float_Float
        {
            public Const_Pair_Float_Float? Opt;

            public _InOptConst_Pair_Float_Float() {}
            public _InOptConst_Pair_Float_Float(Const_Pair_Float_Float value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_Float_Float(Const_Pair_Float_Float value) {return new(value);}
        }
    }
}
