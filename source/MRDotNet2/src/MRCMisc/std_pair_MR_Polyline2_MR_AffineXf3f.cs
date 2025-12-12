public static partial class MR
{
    public static partial class Std
    {
        /// Stores two objects: `MR::Polyline2` and `MR::Polyline2`.
        /// This is the const half of the class.
        public class Const_Pair_MRPolyline2_MRAffineXf3f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Pair_MRPolyline2_MRAffineXf3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_Destroy", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_Polyline2_MR_AffineXf3f_Destroy(_Underlying *_this);
                __MR_std_pair_MR_Polyline2_MR_AffineXf3f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Pair_MRPolyline2_MRAffineXf3f() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Pair_MRPolyline2_MRAffineXf3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_Polyline2_MR_AffineXf3f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Pair_MRPolyline2_MRAffineXf3f(MR.Std._ByValue_Pair_MRPolyline2_MRAffineXf3f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_Polyline2_MR_AffineXf3f_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Const_Pair_MRPolyline2_MRAffineXf3f(MR._ByValue_Polyline2 first, MR.AffineXf3f second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_Construct(MR.Misc._PassBy first_pass_by, MR.Polyline2._Underlying *first, MR.AffineXf3f second);
                _UnderlyingPtr = __MR_std_pair_MR_Polyline2_MR_AffineXf3f_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second);
            }

            /// The first of the two elements, read-only.
            public unsafe MR.Const_Polyline2 First()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_First", ExactSpelling = true)]
                extern static MR.Const_Polyline2._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_First(_Underlying *_this);
                return new(__MR_std_pair_MR_Polyline2_MR_AffineXf3f_First(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, read-only.
            public unsafe MR.Const_AffineXf3f Second()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_Second", ExactSpelling = true)]
                extern static MR.Const_AffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_Second(_Underlying *_this);
                return new(__MR_std_pair_MR_Polyline2_MR_AffineXf3f_Second(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Stores two objects: `MR::Polyline2` and `MR::Polyline2`.
        /// This is the non-const half of the class.
        public class Pair_MRPolyline2_MRAffineXf3f : Const_Pair_MRPolyline2_MRAffineXf3f
        {
            internal unsafe Pair_MRPolyline2_MRAffineXf3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Pair_MRPolyline2_MRAffineXf3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_pair_MR_Polyline2_MR_AffineXf3f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Pair_MRPolyline2_MRAffineXf3f(MR.Std._ByValue_Pair_MRPolyline2_MRAffineXf3f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *other);
                _UnderlyingPtr = __MR_std_pair_MR_Polyline2_MR_AffineXf3f_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Pair_MRPolyline2_MRAffineXf3f other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_pair_MR_Polyline2_MR_AffineXf3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *other);
                __MR_std_pair_MR_Polyline2_MR_AffineXf3f_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the pair elementwise.
            public unsafe Pair_MRPolyline2_MRAffineXf3f(MR._ByValue_Polyline2 first, MR.AffineXf3f second) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_Construct", ExactSpelling = true)]
                extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_Construct(MR.Misc._PassBy first_pass_by, MR.Polyline2._Underlying *first, MR.AffineXf3f second);
                _UnderlyingPtr = __MR_std_pair_MR_Polyline2_MR_AffineXf3f_Construct(first.PassByMode, first.Value is not null ? first.Value._UnderlyingPtr : null, second);
            }

            /// The first of the two elements, mutable.
            public unsafe MR.Polyline2 MutableFirst()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_MutableFirst", ExactSpelling = true)]
                extern static MR.Polyline2._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_MutableFirst(_Underlying *_this);
                return new(__MR_std_pair_MR_Polyline2_MR_AffineXf3f_MutableFirst(_UnderlyingPtr), is_owning: false);
            }

            /// The second of the two elements, mutable.
            public unsafe MR.Mut_AffineXf3f MutableSecond()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_pair_MR_Polyline2_MR_AffineXf3f_MutableSecond", ExactSpelling = true)]
                extern static MR.Mut_AffineXf3f._Underlying *__MR_std_pair_MR_Polyline2_MR_AffineXf3f_MutableSecond(_Underlying *_this);
                return new(__MR_std_pair_MR_Polyline2_MR_AffineXf3f_MutableSecond(_UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Pair_MRPolyline2_MRAffineXf3f` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Pair_MRPolyline2_MRAffineXf3f`/`Const_Pair_MRPolyline2_MRAffineXf3f` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Pair_MRPolyline2_MRAffineXf3f
        {
            internal readonly Const_Pair_MRPolyline2_MRAffineXf3f? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Pair_MRPolyline2_MRAffineXf3f() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Pair_MRPolyline2_MRAffineXf3f(Const_Pair_MRPolyline2_MRAffineXf3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Pair_MRPolyline2_MRAffineXf3f(Const_Pair_MRPolyline2_MRAffineXf3f arg) {return new(arg);}
            public _ByValue_Pair_MRPolyline2_MRAffineXf3f(MR.Misc._Moved<Pair_MRPolyline2_MRAffineXf3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Pair_MRPolyline2_MRAffineXf3f(MR.Misc._Moved<Pair_MRPolyline2_MRAffineXf3f> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Pair_MRPolyline2_MRAffineXf3f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Pair_MRPolyline2_MRAffineXf3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRPolyline2_MRAffineXf3f`/`Const_Pair_MRPolyline2_MRAffineXf3f` directly.
        public class _InOptMut_Pair_MRPolyline2_MRAffineXf3f
        {
            public Pair_MRPolyline2_MRAffineXf3f? Opt;

            public _InOptMut_Pair_MRPolyline2_MRAffineXf3f() {}
            public _InOptMut_Pair_MRPolyline2_MRAffineXf3f(Pair_MRPolyline2_MRAffineXf3f value) {Opt = value;}
            public static implicit operator _InOptMut_Pair_MRPolyline2_MRAffineXf3f(Pair_MRPolyline2_MRAffineXf3f value) {return new(value);}
        }

        /// This is used for optional parameters of class `Pair_MRPolyline2_MRAffineXf3f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Pair_MRPolyline2_MRAffineXf3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Pair_MRPolyline2_MRAffineXf3f`/`Const_Pair_MRPolyline2_MRAffineXf3f` to pass it to the function.
        public class _InOptConst_Pair_MRPolyline2_MRAffineXf3f
        {
            public Const_Pair_MRPolyline2_MRAffineXf3f? Opt;

            public _InOptConst_Pair_MRPolyline2_MRAffineXf3f() {}
            public _InOptConst_Pair_MRPolyline2_MRAffineXf3f(Const_Pair_MRPolyline2_MRAffineXf3f value) {Opt = value;}
            public static implicit operator _InOptConst_Pair_MRPolyline2_MRAffineXf3f(Const_Pair_MRPolyline2_MRAffineXf3f value) {return new(value);}
        }
    }
}
