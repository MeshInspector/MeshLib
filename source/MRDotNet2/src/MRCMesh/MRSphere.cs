public static partial class MR
{
    /// Generated from class `MR::Sphere2f`.
    /// This is the const half of the class.
    public class Const_Sphere2f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Sphere2f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Sphere2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_Destroy", ExactSpelling = true)]
            extern static void __MR_Sphere2f_Destroy(_Underlying *_this);
            __MR_Sphere2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Sphere2f() {Dispose(false);}

        public unsafe MR.Const_Vector2f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_Sphere2f_Get_center(_Underlying *_this);
                return new(__MR_Sphere2f_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_Get_radius", ExactSpelling = true)]
                extern static float *__MR_Sphere2f_Get_radius(_Underlying *_this);
                return *__MR_Sphere2f_Get_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Sphere2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere2f::Sphere2f`.
        public unsafe Const_Sphere2f(MR.Const_Sphere2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_ConstructFromAnother(MR.Sphere2f._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere2f::Sphere2f`.
        public unsafe Const_Sphere2f(MR.Const_Vector2f c, float r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_Construct", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_Construct(MR.Const_Vector2f._Underlying *c, float r);
            _UnderlyingPtr = __MR_Sphere2f_Construct(c._UnderlyingPtr, r);
        }

        /// finds the closest point on sphere
        /// Generated from method `MR::Sphere2f::project`.
        public unsafe MR.Vector2f Project(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_project", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Sphere2f_project(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_Sphere2f_project(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns signed distance from given point to this sphere:
        /// positive - outside, zero - on sphere, negative - inside
        /// Generated from method `MR::Sphere2f::distance`.
        public unsafe float Distance(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_distance", ExactSpelling = true)]
            extern static float __MR_Sphere2f_distance(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_Sphere2f_distance(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns squared distance from given point to this sphere
        /// Generated from method `MR::Sphere2f::distanceSq`.
        public unsafe float DistanceSq(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_distanceSq", ExactSpelling = true)]
            extern static float __MR_Sphere2f_distanceSq(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_Sphere2f_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Sphere2f a, MR.Const_Sphere2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Sphere2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Sphere2f(MR.Const_Sphere2f._Underlying *a, MR.Const_Sphere2f._Underlying *b);
            return __MR_equal_MR_Sphere2f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Sphere2f a, MR.Const_Sphere2f b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_Sphere2f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Sphere2f)
                return this == (MR.Const_Sphere2f)other;
            return false;
        }
    }

    /// Generated from class `MR::Sphere2f`.
    /// This is the non-const half of the class.
    public class Sphere2f : Const_Sphere2f
    {
        internal unsafe Sphere2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector2f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_Sphere2f_GetMutable_center(_Underlying *_this);
                return new(__MR_Sphere2f_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_Sphere2f_GetMutable_radius(_Underlying *_this);
                return ref *__MR_Sphere2f_GetMutable_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Sphere2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere2f::Sphere2f`.
        public unsafe Sphere2f(MR.Const_Sphere2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_ConstructFromAnother(MR.Sphere2f._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere2f::Sphere2f`.
        public unsafe Sphere2f(MR.Const_Vector2f c, float r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_Construct", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_Construct(MR.Const_Vector2f._Underlying *c, float r);
            _UnderlyingPtr = __MR_Sphere2f_Construct(c._UnderlyingPtr, r);
        }

        /// Generated from method `MR::Sphere2f::operator=`.
        public unsafe MR.Sphere2f Assign(MR.Const_Sphere2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Sphere2f._Underlying *__MR_Sphere2f_AssignFromAnother(_Underlying *_this, MR.Sphere2f._Underlying *_other);
            return new(__MR_Sphere2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Sphere2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Sphere2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere2f`/`Const_Sphere2f` directly.
    public class _InOptMut_Sphere2f
    {
        public Sphere2f? Opt;

        public _InOptMut_Sphere2f() {}
        public _InOptMut_Sphere2f(Sphere2f value) {Opt = value;}
        public static implicit operator _InOptMut_Sphere2f(Sphere2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Sphere2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Sphere2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere2f`/`Const_Sphere2f` to pass it to the function.
    public class _InOptConst_Sphere2f
    {
        public Const_Sphere2f? Opt;

        public _InOptConst_Sphere2f() {}
        public _InOptConst_Sphere2f(Const_Sphere2f value) {Opt = value;}
        public static implicit operator _InOptConst_Sphere2f(Const_Sphere2f value) {return new(value);}
    }

    /// Generated from class `MR::Sphere2d`.
    /// This is the const half of the class.
    public class Const_Sphere2d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Sphere2d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Sphere2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_Destroy", ExactSpelling = true)]
            extern static void __MR_Sphere2d_Destroy(_Underlying *_this);
            __MR_Sphere2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Sphere2d() {Dispose(false);}

        public unsafe MR.Const_Vector2d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_Sphere2d_Get_center(_Underlying *_this);
                return new(__MR_Sphere2d_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe double Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_Get_radius", ExactSpelling = true)]
                extern static double *__MR_Sphere2d_Get_radius(_Underlying *_this);
                return *__MR_Sphere2d_Get_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Sphere2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere2d::Sphere2d`.
        public unsafe Const_Sphere2d(MR.Const_Sphere2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_ConstructFromAnother(MR.Sphere2d._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere2d::Sphere2d`.
        public unsafe Const_Sphere2d(MR.Const_Vector2d c, double r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_Construct", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_Construct(MR.Const_Vector2d._Underlying *c, double r);
            _UnderlyingPtr = __MR_Sphere2d_Construct(c._UnderlyingPtr, r);
        }

        /// finds the closest point on sphere
        /// Generated from method `MR::Sphere2d::project`.
        public unsafe MR.Vector2d Project(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_project", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Sphere2d_project(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_Sphere2d_project(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns signed distance from given point to this sphere:
        /// positive - outside, zero - on sphere, negative - inside
        /// Generated from method `MR::Sphere2d::distance`.
        public unsafe double Distance(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_distance", ExactSpelling = true)]
            extern static double __MR_Sphere2d_distance(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_Sphere2d_distance(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns squared distance from given point to this sphere
        /// Generated from method `MR::Sphere2d::distanceSq`.
        public unsafe double DistanceSq(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_distanceSq", ExactSpelling = true)]
            extern static double __MR_Sphere2d_distanceSq(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_Sphere2d_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Sphere2d a, MR.Const_Sphere2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Sphere2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Sphere2d(MR.Const_Sphere2d._Underlying *a, MR.Const_Sphere2d._Underlying *b);
            return __MR_equal_MR_Sphere2d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Sphere2d a, MR.Const_Sphere2d b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_Sphere2d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Sphere2d)
                return this == (MR.Const_Sphere2d)other;
            return false;
        }
    }

    /// Generated from class `MR::Sphere2d`.
    /// This is the non-const half of the class.
    public class Sphere2d : Const_Sphere2d
    {
        internal unsafe Sphere2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector2d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_Sphere2d_GetMutable_center(_Underlying *_this);
                return new(__MR_Sphere2d_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref double Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_GetMutable_radius", ExactSpelling = true)]
                extern static double *__MR_Sphere2d_GetMutable_radius(_Underlying *_this);
                return ref *__MR_Sphere2d_GetMutable_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Sphere2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere2d::Sphere2d`.
        public unsafe Sphere2d(MR.Const_Sphere2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_ConstructFromAnother(MR.Sphere2d._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere2d::Sphere2d`.
        public unsafe Sphere2d(MR.Const_Vector2d c, double r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_Construct", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_Construct(MR.Const_Vector2d._Underlying *c, double r);
            _UnderlyingPtr = __MR_Sphere2d_Construct(c._UnderlyingPtr, r);
        }

        /// Generated from method `MR::Sphere2d::operator=`.
        public unsafe MR.Sphere2d Assign(MR.Const_Sphere2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Sphere2d._Underlying *__MR_Sphere2d_AssignFromAnother(_Underlying *_this, MR.Sphere2d._Underlying *_other);
            return new(__MR_Sphere2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Sphere2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Sphere2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere2d`/`Const_Sphere2d` directly.
    public class _InOptMut_Sphere2d
    {
        public Sphere2d? Opt;

        public _InOptMut_Sphere2d() {}
        public _InOptMut_Sphere2d(Sphere2d value) {Opt = value;}
        public static implicit operator _InOptMut_Sphere2d(Sphere2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Sphere2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Sphere2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere2d`/`Const_Sphere2d` to pass it to the function.
    public class _InOptConst_Sphere2d
    {
        public Const_Sphere2d? Opt;

        public _InOptConst_Sphere2d() {}
        public _InOptConst_Sphere2d(Const_Sphere2d value) {Opt = value;}
        public static implicit operator _InOptConst_Sphere2d(Const_Sphere2d value) {return new(value);}
    }

    /// Generated from class `MR::Sphere3f`.
    /// This is the const half of the class.
    public class Const_Sphere3f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Sphere3f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Sphere3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Sphere3f_Destroy(_Underlying *_this);
            __MR_Sphere3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Sphere3f() {Dispose(false);}

        public unsafe MR.Const_Vector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Sphere3f_Get_center(_Underlying *_this);
                return new(__MR_Sphere3f_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_Get_radius", ExactSpelling = true)]
                extern static float *__MR_Sphere3f_Get_radius(_Underlying *_this);
                return *__MR_Sphere3f_Get_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Sphere3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere3f::Sphere3f`.
        public unsafe Const_Sphere3f(MR.Const_Sphere3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_ConstructFromAnother(MR.Sphere3f._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere3f::Sphere3f`.
        public unsafe Const_Sphere3f(MR.Const_Vector3f c, float r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_Construct", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_Construct(MR.Const_Vector3f._Underlying *c, float r);
            _UnderlyingPtr = __MR_Sphere3f_Construct(c._UnderlyingPtr, r);
        }

        /// finds the closest point on sphere
        /// Generated from method `MR::Sphere3f::project`.
        public unsafe MR.Vector3f Project(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_project", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Sphere3f_project(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_Sphere3f_project(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns signed distance from given point to this sphere:
        /// positive - outside, zero - on sphere, negative - inside
        /// Generated from method `MR::Sphere3f::distance`.
        public unsafe float Distance(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_distance", ExactSpelling = true)]
            extern static float __MR_Sphere3f_distance(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_Sphere3f_distance(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns squared distance from given point to this sphere
        /// Generated from method `MR::Sphere3f::distanceSq`.
        public unsafe float DistanceSq(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_distanceSq", ExactSpelling = true)]
            extern static float __MR_Sphere3f_distanceSq(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_Sphere3f_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Sphere3f a, MR.Const_Sphere3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Sphere3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Sphere3f(MR.Const_Sphere3f._Underlying *a, MR.Const_Sphere3f._Underlying *b);
            return __MR_equal_MR_Sphere3f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Sphere3f a, MR.Const_Sphere3f b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_Sphere3f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Sphere3f)
                return this == (MR.Const_Sphere3f)other;
            return false;
        }
    }

    /// Generated from class `MR::Sphere3f`.
    /// This is the non-const half of the class.
    public class Sphere3f : Const_Sphere3f
    {
        internal unsafe Sphere3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Sphere3f_GetMutable_center(_Underlying *_this);
                return new(__MR_Sphere3f_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_Sphere3f_GetMutable_radius(_Underlying *_this);
                return ref *__MR_Sphere3f_GetMutable_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Sphere3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere3f::Sphere3f`.
        public unsafe Sphere3f(MR.Const_Sphere3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_ConstructFromAnother(MR.Sphere3f._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere3f::Sphere3f`.
        public unsafe Sphere3f(MR.Const_Vector3f c, float r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_Construct", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_Construct(MR.Const_Vector3f._Underlying *c, float r);
            _UnderlyingPtr = __MR_Sphere3f_Construct(c._UnderlyingPtr, r);
        }

        /// Generated from method `MR::Sphere3f::operator=`.
        public unsafe MR.Sphere3f Assign(MR.Const_Sphere3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Sphere3f_AssignFromAnother(_Underlying *_this, MR.Sphere3f._Underlying *_other);
            return new(__MR_Sphere3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Sphere3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Sphere3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere3f`/`Const_Sphere3f` directly.
    public class _InOptMut_Sphere3f
    {
        public Sphere3f? Opt;

        public _InOptMut_Sphere3f() {}
        public _InOptMut_Sphere3f(Sphere3f value) {Opt = value;}
        public static implicit operator _InOptMut_Sphere3f(Sphere3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Sphere3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Sphere3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere3f`/`Const_Sphere3f` to pass it to the function.
    public class _InOptConst_Sphere3f
    {
        public Const_Sphere3f? Opt;

        public _InOptConst_Sphere3f() {}
        public _InOptConst_Sphere3f(Const_Sphere3f value) {Opt = value;}
        public static implicit operator _InOptConst_Sphere3f(Const_Sphere3f value) {return new(value);}
    }

    /// Generated from class `MR::Sphere3d`.
    /// This is the const half of the class.
    public class Const_Sphere3d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Sphere3d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Sphere3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Sphere3d_Destroy(_Underlying *_this);
            __MR_Sphere3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Sphere3d() {Dispose(false);}

        public unsafe MR.Const_Vector3d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_Sphere3d_Get_center(_Underlying *_this);
                return new(__MR_Sphere3d_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe double Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_Get_radius", ExactSpelling = true)]
                extern static double *__MR_Sphere3d_Get_radius(_Underlying *_this);
                return *__MR_Sphere3d_Get_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Sphere3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere3d::Sphere3d`.
        public unsafe Const_Sphere3d(MR.Const_Sphere3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_ConstructFromAnother(MR.Sphere3d._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere3d::Sphere3d`.
        public unsafe Const_Sphere3d(MR.Const_Vector3d c, double r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_Construct", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_Construct(MR.Const_Vector3d._Underlying *c, double r);
            _UnderlyingPtr = __MR_Sphere3d_Construct(c._UnderlyingPtr, r);
        }

        /// finds the closest point on sphere
        /// Generated from method `MR::Sphere3d::project`.
        public unsafe MR.Vector3d Project(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_project", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Sphere3d_project(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_Sphere3d_project(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns signed distance from given point to this sphere:
        /// positive - outside, zero - on sphere, negative - inside
        /// Generated from method `MR::Sphere3d::distance`.
        public unsafe double Distance(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_distance", ExactSpelling = true)]
            extern static double __MR_Sphere3d_distance(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_Sphere3d_distance(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns squared distance from given point to this sphere
        /// Generated from method `MR::Sphere3d::distanceSq`.
        public unsafe double DistanceSq(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_distanceSq", ExactSpelling = true)]
            extern static double __MR_Sphere3d_distanceSq(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_Sphere3d_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Sphere3d a, MR.Const_Sphere3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Sphere3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Sphere3d(MR.Const_Sphere3d._Underlying *a, MR.Const_Sphere3d._Underlying *b);
            return __MR_equal_MR_Sphere3d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Sphere3d a, MR.Const_Sphere3d b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_Sphere3d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Sphere3d)
                return this == (MR.Const_Sphere3d)other;
            return false;
        }
    }

    /// Generated from class `MR::Sphere3d`.
    /// This is the non-const half of the class.
    public class Sphere3d : Const_Sphere3d
    {
        internal unsafe Sphere3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_Sphere3d_GetMutable_center(_Underlying *_this);
                return new(__MR_Sphere3d_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref double Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_GetMutable_radius", ExactSpelling = true)]
                extern static double *__MR_Sphere3d_GetMutable_radius(_Underlying *_this);
                return ref *__MR_Sphere3d_GetMutable_radius(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Sphere3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Sphere3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Sphere3d::Sphere3d`.
        public unsafe Sphere3d(MR.Const_Sphere3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_ConstructFromAnother(MR.Sphere3d._Underlying *_other);
            _UnderlyingPtr = __MR_Sphere3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Sphere3d::Sphere3d`.
        public unsafe Sphere3d(MR.Const_Vector3d c, double r) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_Construct", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_Construct(MR.Const_Vector3d._Underlying *c, double r);
            _UnderlyingPtr = __MR_Sphere3d_Construct(c._UnderlyingPtr, r);
        }

        /// Generated from method `MR::Sphere3d::operator=`.
        public unsafe MR.Sphere3d Assign(MR.Const_Sphere3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Sphere3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Sphere3d._Underlying *__MR_Sphere3d_AssignFromAnother(_Underlying *_this, MR.Sphere3d._Underlying *_other);
            return new(__MR_Sphere3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Sphere3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Sphere3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere3d`/`Const_Sphere3d` directly.
    public class _InOptMut_Sphere3d
    {
        public Sphere3d? Opt;

        public _InOptMut_Sphere3d() {}
        public _InOptMut_Sphere3d(Sphere3d value) {Opt = value;}
        public static implicit operator _InOptMut_Sphere3d(Sphere3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Sphere3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Sphere3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Sphere3d`/`Const_Sphere3d` to pass it to the function.
    public class _InOptConst_Sphere3d
    {
        public Const_Sphere3d? Opt;

        public _InOptConst_Sphere3d() {}
        public _InOptConst_Sphere3d(Const_Sphere3d value) {Opt = value;}
        public static implicit operator _InOptConst_Sphere3d(Const_Sphere3d value) {return new(value);}
    }
}
