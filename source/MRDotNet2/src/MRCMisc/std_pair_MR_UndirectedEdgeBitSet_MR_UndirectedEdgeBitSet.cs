public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::UndirectedEdgeBitSet` and `MR::UndirectedEdgeBitSet`.
        /// This is the const half of the class.
        public class Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Destroy(_Underlying *_this);
                __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(MR.Std._ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(MR._ByValue_UndirectedEdgeBitSet first, MR._ByValue_UndirectedEdgeBitSet second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Construct(MR.Misc._PassBy first_pass_by, MR.UndirectedEdgeBitSet._Underlying *first, MR.Misc._PassBy second_pass_by, MR.UndirectedEdgeBitSet._Underlying *second);
                _UnderlyingPtr = __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second.PassByMode, second.Value is not null ? second.Value._UnderlyingPtr : null);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_UndirectedEdgeBitSet First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_First", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_First(_Underlying *_this);
                return new(__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_UndirectedEdgeBitSet Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Second", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::UndirectedEdgeBitSet` and `MR::UndirectedEdgeBitSet`.
        /// This is the non-const half of the class.
        public class Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet : Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet
        {
            internal unsafe Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(MR.Std._ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *other);
                __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(MR._ByValue_UndirectedEdgeBitSet first, MR._ByValue_UndirectedEdgeBitSet second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Construct(MR.Misc._PassBy first_pass_by, MR.UndirectedEdgeBitSet._Underlying *first, MR.Misc._PassBy second_pass_by, MR.UndirectedEdgeBitSet._Underlying *second);
                _UnderlyingPtr = __MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second.PassByMode, second.Value is not null ? second.Value._UnderlyingPtr : null);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.UndirectedEdgeBitSet MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_MutableFirst", ExactSpelling = true)]
                extern static MR.UndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.UndirectedEdgeBitSet MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_MutableSecond", ExactSpelling = true)]
                extern static MR.UndirectedEdgeBitSet._Underlying *__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_UndirectedEdgeBitSet_MR_UndirectedEdgeBitSet_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet`/`Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet
        {
            internal readonly Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet arg) {return new(arg);}
            public _ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(MR.Misc._Moved<Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(MR.Misc._Moved<Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet`/`Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet` directly.
        public class _InOptMut_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet
        {
            public Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet? Opt;

            public _InOptMut_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet() {}
            public _InOptMut_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet`/`Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet` to pass it to the function.
        public class _InOptConst_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet
        {
            public Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet? Opt;

            public _InOptConst_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet() {}
            public _InOptConst_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet(Const_Pair_MRUndirectedEdgeBitSet_MRUndirectedEdgeBitSet value) {return new(value);}
        }
    }
}
