public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `double` and `double`.
        /// This is the const half of the class.
        public class Const_Pair_Double_Int : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_Double_Int(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_double_int_Destroy(_Underlying *_this);
                __MR_std_pair_double_int_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_Double_Int() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_Double_Int() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_Double_Int._Underlying *__MR_std_pair_double_int_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_double_int_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_Double_Int(MR.Std.Const_Pair_Double_Int other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_Double_Int._Underlying *__MR_std_pair_double_int_ConstructFromAnother(MR.Std.Pair_Double_Int._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_double_int_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_Double_Int(double first, int second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_Double_Int._Underlying *__MR_std_pair_double_int_Construct(double first, int second);
                _UnderlyingPtr = __MR_std_pair_double_int_Construct(first, second);
            }

            /// The first of the two elements, read-only.
            public unsafe double First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_First", ExactSpelling = true)]
                extern static double *__MR_std_pair_double_int_First(_Underlying *_this);
                return *__MR_std_pair_double_int_First(_UnderlyingPtr);
            }

            /// The second of the two elements, read-only.
            public unsafe int Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_Second", ExactSpelling = true)]
                extern static int *__MR_std_pair_double_int_Second(_Underlying *_this);
                return *__MR_std_pair_double_int_Second(_UnderlyingPtr);
            }
        }

        /// Stores two objects: `double` and `double`.
        /// This is the non-const half of the class.
        public class Pair_Double_Int : Const_Pair_Double_Int
        {
            internal unsafe Pair_Double_Int(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_Double_Int() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_Double_Int._Underlying *__MR_std_pair_double_int_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_double_int_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_Double_Int(MR.Std.Const_Pair_Double_Int other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_Double_Int._Underlying *__MR_std_pair_double_int_ConstructFromAnother(MR.Std.Pair_Double_Int._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_double_int_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Pair_Double_Int other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_double_int_AssignFromAnother(_Underlying *_this, MR.Std.Pair_Double_Int._Underlying *other);
                __MR_std_pair_double_int_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_Double_Int(double first, int second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_Double_Int._Underlying *__MR_std_pair_double_int_Construct(double first, int second);
                _UnderlyingPtr = __MR_std_pair_double_int_Construct(first, second);
            }

            /// The first of the two elements, mutable.
            public unsafe ref double MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_MutableFirst", ExactSpelling = true)]
                extern static double *__MR_std_pair_double_int_MutableFirst(_Underlying *_this);
                return ref *__MR_std_pair_double_int_MutableFirst(_UnderlyingPtr);
            }

            /// The second of the two elements, mutable.
            public unsafe ref int MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_double_int_MutableSecond", ExactSpelling = true)]
                extern static int *__MR_std_pair_double_int_MutableSecond(_Underlying *_this);
                return ref *__MR_std_pair_double_int_MutableSecond(_UnderlyingPtr);
            }
        }

        /// This is used for optional parameters of class `Pair_Double_Int` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_Double_Int`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_Double_Int`/`Const_Pair_Double_Int` directly.
        public class _InOptMut_Pair_Double_Int
        {
            public Pair_Double_Int? Opt;

            public _InOptMut_Pair_Double_Int() {}
            public _InOptMut_Pair_Double_Int(Pair_Double_Int value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_Double_Int(Pair_Double_Int value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_Double_Int` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_Double_Int`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_Double_Int`/`Const_Pair_Double_Int` to pass it to the function.
        public class _InOptConst_Pair_Double_Int
        {
            public Const_Pair_Double_Int? Opt;

            public _InOptConst_Pair_Double_Int() {}
            public _InOptConst_Pair_Double_Int(Const_Pair_Double_Int value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_Double_Int(Const_Pair_Double_Int value) {return new(value);}
        }
    }
}
