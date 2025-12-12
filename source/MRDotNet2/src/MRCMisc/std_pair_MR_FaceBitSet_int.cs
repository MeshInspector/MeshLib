public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::FaceBitSet` and `MR::FaceBitSet`.
        /// This is the const half of the class.
        public class Const_Pair_MRFaceBitSet_Int : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRFaceBitSet_Int(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_FaceBitSet_int_Destroy(_Underlying *_this);
                __MR_std_pair_MR_FaceBitSet_int_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRFaceBitSet_Int() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRFaceBitSet_Int() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_std_pair_MR_FaceBitSet_int_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_FaceBitSet_int_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRFaceBitSet_Int(MR.Std._ByValue_Pair_MRFaceBitSet_Int other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_std_pair_MR_FaceBitSet_int_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRFaceBitSet_Int._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_FaceBitSet_int_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRFaceBitSet_Int(MR._ByValue_FaceBitSet first, int second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_std_pair_MR_FaceBitSet_int_Construct(MR.Misc._PassBy first_pass_by, MR.FaceBitSet._Underlying *first, int second);
                _UnderlyingPtr = __MR_std_pair_MR_FaceBitSet_int_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_FaceBitSet First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_First", ExactSpelling = true)]
                extern static MR.Const_FaceBitSet._Underlying *__MR_std_pair_MR_FaceBitSet_int_First(_Underlying *_this);
                return new(__MR_std_pair_MR_FaceBitSet_int_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe int Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_Second", ExactSpelling = true)]
                extern static int *__MR_std_pair_MR_FaceBitSet_int_Second(_Underlying *_this);
                return *__MR_std_pair_MR_FaceBitSet_int_Second(_UnderlyingPtr);
            }
        }

        /// Stores two objects: `MR::FaceBitSet` and `MR::FaceBitSet`.
        /// This is the non-const half of the class.
        public class Pair_MRFaceBitSet_Int : Const_Pair_MRFaceBitSet_Int
        {
            internal unsafe Pair_MRFaceBitSet_Int(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRFaceBitSet_Int() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_std_pair_MR_FaceBitSet_int_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_FaceBitSet_int_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRFaceBitSet_Int(MR.Std._ByValue_Pair_MRFaceBitSet_Int other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_std_pair_MR_FaceBitSet_int_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRFaceBitSet_Int._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_FaceBitSet_int_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Pair_MRFaceBitSet_Int other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_FaceBitSet_int_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRFaceBitSet_Int._Underlying *other);
                __MR_std_pair_MR_FaceBitSet_int_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRFaceBitSet_Int(MR._ByValue_FaceBitSet first, int second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_std_pair_MR_FaceBitSet_int_Construct(MR.Misc._PassBy first_pass_by, MR.FaceBitSet._Underlying *first, int second);
                _UnderlyingPtr = __MR_std_pair_MR_FaceBitSet_int_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.FaceBitSet MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_MutableFirst", ExactSpelling = true)]
                extern static MR.FaceBitSet._Underlying *__MR_std_pair_MR_FaceBitSet_int_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_FaceBitSet_int_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe ref int MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_FaceBitSet_int_MutableSecond", ExactSpelling = true)]
                extern static int *__MR_std_pair_MR_FaceBitSet_int_MutableSecond(_Underlying *_this);
                return ref *__MR_std_pair_MR_FaceBitSet_int_MutableSecond(_UnderlyingPtr);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Pair_MRFaceBitSet_Int` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Pair_MRFaceBitSet_Int`/`Const_Pair_MRFaceBitSet_Int` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Pair_MRFaceBitSet_Int
        {
            internal readonly Const_Pair_MRFaceBitSet_Int? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Pair_MRFaceBitSet_Int() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Pair_MRFaceBitSet_Int(Const_Pair_MRFaceBitSet_Int new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Pair_MRFaceBitSet_Int(Const_Pair_MRFaceBitSet_Int arg) {return new(arg);}
            public _ByValue_Pair_MRFaceBitSet_Int(MR.Misc._Moved<Pair_MRFaceBitSet_Int> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Pair_MRFaceBitSet_Int(MR.Misc._Moved<Pair_MRFaceBitSet_Int> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Pair_MRFaceBitSet_Int` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRFaceBitSet_Int`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRFaceBitSet_Int`/`Const_Pair_MRFaceBitSet_Int` directly.
        public class _InOptMut_Pair_MRFaceBitSet_Int
        {
            public Pair_MRFaceBitSet_Int? Opt;

            public _InOptMut_Pair_MRFaceBitSet_Int() {}
            public _InOptMut_Pair_MRFaceBitSet_Int(Pair_MRFaceBitSet_Int value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRFaceBitSet_Int(Pair_MRFaceBitSet_Int value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRFaceBitSet_Int` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRFaceBitSet_Int`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRFaceBitSet_Int`/`Const_Pair_MRFaceBitSet_Int` to pass it to the function.
        public class _InOptConst_Pair_MRFaceBitSet_Int
        {
            public Const_Pair_MRFaceBitSet_Int? Opt;

            public _InOptConst_Pair_MRFaceBitSet_Int() {}
            public _InOptConst_Pair_MRFaceBitSet_Int(Const_Pair_MRFaceBitSet_Int value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRFaceBitSet_Int(Const_Pair_MRFaceBitSet_Int value) {return new(value);}
        }
    }
}
