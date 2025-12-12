public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::DistanceMap` and `MR::DistanceMap`.
        /// This is the const half of the class.
        public class Const_Pair_MRDistanceMap_MRDistanceMap : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRDistanceMap_MRDistanceMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_DistanceMap_MR_DistanceMap_Destroy(_Underlying *_this);
                __MR_std_pair_MR_DistanceMap_MR_DistanceMap_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRDistanceMap_MRDistanceMap() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRDistanceMap_MRDistanceMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_DistanceMap_MR_DistanceMap_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRDistanceMap_MRDistanceMap(MR.Std._ByValue_Pair_MRDistanceMap_MRDistanceMap other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_DistanceMap_MR_DistanceMap_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRDistanceMap_MRDistanceMap(MR._ByValue_DistanceMap first, MR._ByValue_DistanceMap second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_Construct(MR.Misc._PassBy first_pass_by, MR.DistanceMap._Underlying *first, MR.Misc._PassBy second_pass_by, MR.DistanceMap._Underlying *second);
                _UnderlyingPtr = __MR_std_pair_MR_DistanceMap_MR_DistanceMap_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second.PassByMode, second.Value is not null ? second.Value._UnderlyingPtr : null);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_DistanceMap First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_First", ExactSpelling = true)]
                extern static MR.Const_DistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_First(_Underlying *_this);
                return new(__MR_std_pair_MR_DistanceMap_MR_DistanceMap_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_DistanceMap Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_Second", ExactSpelling = true)]
                extern static MR.Const_DistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_DistanceMap_MR_DistanceMap_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::DistanceMap` and `MR::DistanceMap`.
        /// This is the non-const half of the class.
        public class Pair_MRDistanceMap_MRDistanceMap : Const_Pair_MRDistanceMap_MRDistanceMap
        {
            internal unsafe Pair_MRDistanceMap_MRDistanceMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRDistanceMap_MRDistanceMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_DistanceMap_MR_DistanceMap_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRDistanceMap_MRDistanceMap(MR.Std._ByValue_Pair_MRDistanceMap_MRDistanceMap other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_DistanceMap_MR_DistanceMap_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Pair_MRDistanceMap_MRDistanceMap other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_DistanceMap_MR_DistanceMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *other);
                __MR_std_pair_MR_DistanceMap_MR_DistanceMap_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRDistanceMap_MRDistanceMap(MR._ByValue_DistanceMap first, MR._ByValue_DistanceMap second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_Construct(MR.Misc._PassBy first_pass_by, MR.DistanceMap._Underlying *first, MR.Misc._PassBy second_pass_by, MR.DistanceMap._Underlying *second);
                _UnderlyingPtr = __MR_std_pair_MR_DistanceMap_MR_DistanceMap_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second.PassByMode, second.Value is not null ? second.Value._UnderlyingPtr : null);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.DistanceMap MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_MutableFirst", ExactSpelling = true)]
                extern static MR.DistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_DistanceMap_MR_DistanceMap_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.DistanceMap MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_DistanceMap_MR_DistanceMap_MutableSecond", ExactSpelling = true)]
                extern static MR.DistanceMap._Underlying *__MR_std_pair_MR_DistanceMap_MR_DistanceMap_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_DistanceMap_MR_DistanceMap_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Pair_MRDistanceMap_MRDistanceMap` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Pair_MRDistanceMap_MRDistanceMap`/`Const_Pair_MRDistanceMap_MRDistanceMap` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Pair_MRDistanceMap_MRDistanceMap
        {
            internal readonly Const_Pair_MRDistanceMap_MRDistanceMap? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Pair_MRDistanceMap_MRDistanceMap() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Pair_MRDistanceMap_MRDistanceMap(Const_Pair_MRDistanceMap_MRDistanceMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Pair_MRDistanceMap_MRDistanceMap(Const_Pair_MRDistanceMap_MRDistanceMap arg) {return new(arg);}
            public _ByValue_Pair_MRDistanceMap_MRDistanceMap(MR.Misc._Moved<Pair_MRDistanceMap_MRDistanceMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Pair_MRDistanceMap_MRDistanceMap(MR.Misc._Moved<Pair_MRDistanceMap_MRDistanceMap> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Pair_MRDistanceMap_MRDistanceMap` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRDistanceMap_MRDistanceMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRDistanceMap_MRDistanceMap`/`Const_Pair_MRDistanceMap_MRDistanceMap` directly.
        public class _InOptMut_Pair_MRDistanceMap_MRDistanceMap
        {
            public Pair_MRDistanceMap_MRDistanceMap? Opt;

            public _InOptMut_Pair_MRDistanceMap_MRDistanceMap() {}
            public _InOptMut_Pair_MRDistanceMap_MRDistanceMap(Pair_MRDistanceMap_MRDistanceMap value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRDistanceMap_MRDistanceMap(Pair_MRDistanceMap_MRDistanceMap value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRDistanceMap_MRDistanceMap` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRDistanceMap_MRDistanceMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRDistanceMap_MRDistanceMap`/`Const_Pair_MRDistanceMap_MRDistanceMap` to pass it to the function.
        public class _InOptConst_Pair_MRDistanceMap_MRDistanceMap
        {
            public Const_Pair_MRDistanceMap_MRDistanceMap? Opt;

            public _InOptConst_Pair_MRDistanceMap_MRDistanceMap() {}
            public _InOptConst_Pair_MRDistanceMap_MRDistanceMap(Const_Pair_MRDistanceMap_MRDistanceMap value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRDistanceMap_MRDistanceMap(Const_Pair_MRDistanceMap_MRDistanceMap value) {return new(value);}
        }
    }
}
