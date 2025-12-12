public static partial class MR
{
    /// This class can be used to solve the problem of multiple 3D objects alignment,
    /// by first collecting weighted links between pairs of points from different objects,
    /// and then solving for transformations minimizing weighted average of link penalties
    /// Generated from class `MR::MultiwayAligningTransform`.
    /// This is the const half of the class.
    public class Const_MultiwayAligningTransform : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MultiwayAligningTransform(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Destroy", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_Destroy(_Underlying *_this);
            __MR_MultiwayAligningTransform_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MultiwayAligningTransform() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MultiwayAligningTransform() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiwayAligningTransform_DefaultConstruct();
        }

        /// Generated from constructor `MR::MultiwayAligningTransform::MultiwayAligningTransform`.
        public unsafe Const_MultiwayAligningTransform(MR._ByValue_MultiwayAligningTransform _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MultiwayAligningTransform._Underlying *_other);
            _UnderlyingPtr = __MR_MultiwayAligningTransform_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// initializes internal data to start registering given number of objects
        /// Generated from constructor `MR::MultiwayAligningTransform::MultiwayAligningTransform`.
        public unsafe Const_MultiwayAligningTransform(int numObjs) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Construct", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_Construct(int numObjs);
            _UnderlyingPtr = __MR_MultiwayAligningTransform_Construct(numObjs);
        }

        /// finds the solution consisting of all objects transformations (numObj),
        /// that minimizes the summed weighted squared distance among accumulated links;
        /// the transform of the last object is always identity (it is fixed)
        /// Generated from method `MR::MultiwayAligningTransform::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRRigidXf3d> Solve(MR.MultiwayAligningTransform.Const_Stabilizer stab)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_solve_1", ExactSpelling = true)]
            extern static MR.Std.Vector_MRRigidXf3d._Underlying *__MR_MultiwayAligningTransform_solve_1(_Underlying *_this, MR.MultiwayAligningTransform.Const_Stabilizer._Underlying *stab);
            return MR.Misc.Move(new MR.Std.Vector_MRRigidXf3d(__MR_MultiwayAligningTransform_solve_1(_UnderlyingPtr, stab._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::MultiwayAligningTransform::solve`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRRigidXf3d> Solve()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_solve_0", ExactSpelling = true)]
            extern static MR.Std.Vector_MRRigidXf3d._Underlying *__MR_MultiwayAligningTransform_solve_0(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRRigidXf3d(__MR_MultiwayAligningTransform_solve_0(_UnderlyingPtr), is_owning: true));
        }

        /// small stabilizer allows one to convert the linear system from under-determined to determined (e.g. too few linearly independent pairs for some object);
        /// large stabilizer results in suboptimal found transformations (huge stabilizier => zero transforamtions)
        /// Generated from class `MR::MultiwayAligningTransform::Stabilizer`.
        /// This is the const half of the class.
        public class Const_Stabilizer : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Stabilizer(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_Destroy", ExactSpelling = true)]
                extern static void __MR_MultiwayAligningTransform_Stabilizer_Destroy(_Underlying *_this);
                __MR_MultiwayAligningTransform_Stabilizer_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Stabilizer() {Dispose(false);}

            // length units
            public unsafe double Rot
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_Get_rot", ExactSpelling = true)]
                    extern static double *__MR_MultiwayAligningTransform_Stabilizer_Get_rot(_Underlying *_this);
                    return *__MR_MultiwayAligningTransform_Stabilizer_Get_rot(_UnderlyingPtr);
                }
            }

            // dimensionless
            public unsafe double Shift
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_Get_shift", ExactSpelling = true)]
                    extern static double *__MR_MultiwayAligningTransform_Stabilizer_Get_shift(_Underlying *_this);
                    return *__MR_MultiwayAligningTransform_Stabilizer_Get_shift(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Stabilizer() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_DefaultConstruct();
                _UnderlyingPtr = __MR_MultiwayAligningTransform_Stabilizer_DefaultConstruct();
            }

            /// Constructs `MR::MultiwayAligningTransform::Stabilizer` elementwise.
            public unsafe Const_Stabilizer(double rot, double shift) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_ConstructFrom", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_ConstructFrom(double rot, double shift);
                _UnderlyingPtr = __MR_MultiwayAligningTransform_Stabilizer_ConstructFrom(rot, shift);
            }

            /// Generated from constructor `MR::MultiwayAligningTransform::Stabilizer::Stabilizer`.
            public unsafe Const_Stabilizer(MR.MultiwayAligningTransform.Const_Stabilizer _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_ConstructFromAnother(MR.MultiwayAligningTransform.Stabilizer._Underlying *_other);
                _UnderlyingPtr = __MR_MultiwayAligningTransform_Stabilizer_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// small stabilizer allows one to convert the linear system from under-determined to determined (e.g. too few linearly independent pairs for some object);
        /// large stabilizer results in suboptimal found transformations (huge stabilizier => zero transforamtions)
        /// Generated from class `MR::MultiwayAligningTransform::Stabilizer`.
        /// This is the non-const half of the class.
        public class Stabilizer : Const_Stabilizer
        {
            internal unsafe Stabilizer(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // length units
            public new unsafe ref double Rot
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_GetMutable_rot", ExactSpelling = true)]
                    extern static double *__MR_MultiwayAligningTransform_Stabilizer_GetMutable_rot(_Underlying *_this);
                    return ref *__MR_MultiwayAligningTransform_Stabilizer_GetMutable_rot(_UnderlyingPtr);
                }
            }

            // dimensionless
            public new unsafe ref double Shift
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_GetMutable_shift", ExactSpelling = true)]
                    extern static double *__MR_MultiwayAligningTransform_Stabilizer_GetMutable_shift(_Underlying *_this);
                    return ref *__MR_MultiwayAligningTransform_Stabilizer_GetMutable_shift(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Stabilizer() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_DefaultConstruct();
                _UnderlyingPtr = __MR_MultiwayAligningTransform_Stabilizer_DefaultConstruct();
            }

            /// Constructs `MR::MultiwayAligningTransform::Stabilizer` elementwise.
            public unsafe Stabilizer(double rot, double shift) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_ConstructFrom", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_ConstructFrom(double rot, double shift);
                _UnderlyingPtr = __MR_MultiwayAligningTransform_Stabilizer_ConstructFrom(rot, shift);
            }

            /// Generated from constructor `MR::MultiwayAligningTransform::Stabilizer::Stabilizer`.
            public unsafe Stabilizer(MR.MultiwayAligningTransform.Const_Stabilizer _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_ConstructFromAnother(MR.MultiwayAligningTransform.Stabilizer._Underlying *_other);
                _UnderlyingPtr = __MR_MultiwayAligningTransform_Stabilizer_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MultiwayAligningTransform::Stabilizer::operator=`.
            public unsafe MR.MultiwayAligningTransform.Stabilizer Assign(MR.MultiwayAligningTransform.Const_Stabilizer _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Stabilizer_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MultiwayAligningTransform.Stabilizer._Underlying *__MR_MultiwayAligningTransform_Stabilizer_AssignFromAnother(_Underlying *_this, MR.MultiwayAligningTransform.Stabilizer._Underlying *_other);
                return new(__MR_MultiwayAligningTransform_Stabilizer_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Stabilizer` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Stabilizer`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Stabilizer`/`Const_Stabilizer` directly.
        public class _InOptMut_Stabilizer
        {
            public Stabilizer? Opt;

            public _InOptMut_Stabilizer() {}
            public _InOptMut_Stabilizer(Stabilizer value) {Opt = value;}
            public static implicit operator _InOptMut_Stabilizer(Stabilizer value) {return new(value);}
        }

        /// This is used for optional parameters of class `Stabilizer` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Stabilizer`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Stabilizer`/`Const_Stabilizer` to pass it to the function.
        public class _InOptConst_Stabilizer
        {
            public Const_Stabilizer? Opt;

            public _InOptConst_Stabilizer() {}
            public _InOptConst_Stabilizer(Const_Stabilizer value) {Opt = value;}
            public static implicit operator _InOptConst_Stabilizer(Const_Stabilizer value) {return new(value);}
        }
    }

    /// This class can be used to solve the problem of multiple 3D objects alignment,
    /// by first collecting weighted links between pairs of points from different objects,
    /// and then solving for transformations minimizing weighted average of link penalties
    /// Generated from class `MR::MultiwayAligningTransform`.
    /// This is the non-const half of the class.
    public class MultiwayAligningTransform : Const_MultiwayAligningTransform
    {
        internal unsafe MultiwayAligningTransform(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MultiwayAligningTransform() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_DefaultConstruct();
            _UnderlyingPtr = __MR_MultiwayAligningTransform_DefaultConstruct();
        }

        /// Generated from constructor `MR::MultiwayAligningTransform::MultiwayAligningTransform`.
        public unsafe MultiwayAligningTransform(MR._ByValue_MultiwayAligningTransform _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MultiwayAligningTransform._Underlying *_other);
            _UnderlyingPtr = __MR_MultiwayAligningTransform_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// initializes internal data to start registering given number of objects
        /// Generated from constructor `MR::MultiwayAligningTransform::MultiwayAligningTransform`.
        public unsafe MultiwayAligningTransform(int numObjs) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_Construct", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_Construct(int numObjs);
            _UnderlyingPtr = __MR_MultiwayAligningTransform_Construct(numObjs);
        }

        /// Generated from method `MR::MultiwayAligningTransform::operator=`.
        public unsafe MR.MultiwayAligningTransform Assign(MR._ByValue_MultiwayAligningTransform _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MultiwayAligningTransform._Underlying *__MR_MultiwayAligningTransform_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MultiwayAligningTransform._Underlying *_other);
            return new(__MR_MultiwayAligningTransform_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// reinitializes internal data to start registering given number of objects
        /// Generated from method `MR::MultiwayAligningTransform::reset`.
        public unsafe void Reset(int numObjs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_reset", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_reset(_Underlying *_this, int numObjs);
            __MR_MultiwayAligningTransform_reset(_UnderlyingPtr, numObjs);
        }

        /// appends a 3D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
        /// with link penalty equal to weight (w) times squared distance between two points
        /// Generated from method `MR::MultiwayAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(int objA, MR.Const_Vector3d pA, int objB, MR.Const_Vector3d pB, double? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_add_5_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_add_5_MR_Vector3d(_Underlying *_this, int objA, MR.Const_Vector3d._Underlying *pA, int objB, MR.Const_Vector3d._Underlying *pB, double *w);
            double __deref_w = w.GetValueOrDefault();
            __MR_MultiwayAligningTransform_add_5_MR_Vector3d(_UnderlyingPtr, objA, pA._UnderlyingPtr, objB, pB._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// appends a 3D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
        /// with link penalty equal to weight (w) times squared distance between two points
        /// Generated from method `MR::MultiwayAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(int objA, MR.Const_Vector3f pA, int objB, MR.Const_Vector3f pB, float? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_add_5_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_add_5_MR_Vector3f(_Underlying *_this, int objA, MR.Const_Vector3f._Underlying *pA, int objB, MR.Const_Vector3f._Underlying *pB, float *w);
            float __deref_w = w.GetValueOrDefault();
            __MR_MultiwayAligningTransform_add_5_MR_Vector3f(_UnderlyingPtr, objA, pA._UnderlyingPtr, objB, pB._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// appends a 1D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
        /// with link penalty equal to weight (w) times squared distance between their projections on given direction (n);
        /// for a point on last fixed object, it is equivalent to point-to-plane link with the plane through that fixed point with normal (n)
        /// Generated from method `MR::MultiwayAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(int objA, MR.Const_Vector3d pA, int objB, MR.Const_Vector3d pB, MR.Const_Vector3d n, double? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_add_6_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_add_6_MR_Vector3d(_Underlying *_this, int objA, MR.Const_Vector3d._Underlying *pA, int objB, MR.Const_Vector3d._Underlying *pB, MR.Const_Vector3d._Underlying *n, double *w);
            double __deref_w = w.GetValueOrDefault();
            __MR_MultiwayAligningTransform_add_6_MR_Vector3d(_UnderlyingPtr, objA, pA._UnderlyingPtr, objB, pB._UnderlyingPtr, n._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// appends a 1D link into consideration: one point (pA) from (objA), and the other point (pB) from (objB)
        /// with link penalty equal to weight (w) times squared distance between their projections on given direction (n);
        /// for a point on last fixed object, it is equivalent to point-to-plane link with the plane through that fixed point with normal (n)
        /// Generated from method `MR::MultiwayAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(int objA, MR.Const_Vector3f pA, int objB, MR.Const_Vector3f pB, MR.Const_Vector3f n, float? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_add_6_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_add_6_MR_Vector3f(_Underlying *_this, int objA, MR.Const_Vector3f._Underlying *pA, int objB, MR.Const_Vector3f._Underlying *pB, MR.Const_Vector3f._Underlying *n, float *w);
            float __deref_w = w.GetValueOrDefault();
            __MR_MultiwayAligningTransform_add_6_MR_Vector3f(_UnderlyingPtr, objA, pA._UnderlyingPtr, objB, pB._UnderlyingPtr, n._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// appends links accumulated in (r) into this
        /// Generated from method `MR::MultiwayAligningTransform::add`.
        public unsafe void Add(MR.Const_MultiwayAligningTransform r)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MultiwayAligningTransform_add_1", ExactSpelling = true)]
            extern static void __MR_MultiwayAligningTransform_add_1(_Underlying *_this, MR.Const_MultiwayAligningTransform._Underlying *r);
            __MR_MultiwayAligningTransform_add_1(_UnderlyingPtr, r._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MultiwayAligningTransform` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MultiwayAligningTransform`/`Const_MultiwayAligningTransform` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MultiwayAligningTransform
    {
        internal readonly Const_MultiwayAligningTransform? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MultiwayAligningTransform() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MultiwayAligningTransform(MR.Misc._Moved<MultiwayAligningTransform> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MultiwayAligningTransform(MR.Misc._Moved<MultiwayAligningTransform> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MultiwayAligningTransform` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MultiwayAligningTransform`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiwayAligningTransform`/`Const_MultiwayAligningTransform` directly.
    public class _InOptMut_MultiwayAligningTransform
    {
        public MultiwayAligningTransform? Opt;

        public _InOptMut_MultiwayAligningTransform() {}
        public _InOptMut_MultiwayAligningTransform(MultiwayAligningTransform value) {Opt = value;}
        public static implicit operator _InOptMut_MultiwayAligningTransform(MultiwayAligningTransform value) {return new(value);}
    }

    /// This is used for optional parameters of class `MultiwayAligningTransform` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MultiwayAligningTransform`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MultiwayAligningTransform`/`Const_MultiwayAligningTransform` to pass it to the function.
    public class _InOptConst_MultiwayAligningTransform
    {
        public Const_MultiwayAligningTransform? Opt;

        public _InOptConst_MultiwayAligningTransform() {}
        public _InOptConst_MultiwayAligningTransform(Const_MultiwayAligningTransform value) {Opt = value;}
        public static implicit operator _InOptConst_MultiwayAligningTransform(Const_MultiwayAligningTransform value) {return new(value);}
    }
}
