public static partial class MR
{
    /// 3-dimensional plane: dot(n,x) - d = 0
    /// Generated from class `MR::Plane3f`.
    /// This is the const half of the class.
    public class Const_Plane3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Plane3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Plane3f_Destroy(_Underlying *_this);
            __MR_Plane3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Plane3f() {Dispose(false);}

        public unsafe MR.Const_Vector3f N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Get_n", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Plane3f_Get_n(_Underlying *_this);
                return new(__MR_Plane3f_Get_n(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Get_d", ExactSpelling = true)]
                extern static float *__MR_Plane3f_Get_d(_Underlying *_this);
                return *__MR_Plane3f_Get_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Plane3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Plane3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Plane3f::Plane3f`.
        public unsafe Const_Plane3f(MR.Const_Plane3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_ConstructFromAnother(MR.Plane3f._Underlying *_other);
            _UnderlyingPtr = __MR_Plane3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Plane3f::Plane3f`.
        public unsafe Const_Plane3f(MR.Const_Vector3f n, float d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Construct", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_Construct(MR.Const_Vector3f._Underlying *n, float d);
            _UnderlyingPtr = __MR_Plane3f_Construct(n._UnderlyingPtr, d);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Plane3f::Plane3f`.
        public unsafe Const_Plane3f(MR.Const_Plane3d p) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Construct_double", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_Construct_double(MR.Const_Plane3d._Underlying *p);
            _UnderlyingPtr = __MR_Plane3f_Construct_double(p._UnderlyingPtr);
        }

        /// Generated from method `MR::Plane3f::fromDirAndPt`.
        public static unsafe MR.Plane3f FromDirAndPt(MR.Const_Vector3f n, MR.Const_Vector3f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_fromDirAndPt", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_fromDirAndPt(MR.Const_Vector3f._Underlying *n, MR.Const_Vector3f._Underlying *p);
            return new(__MR_Plane3f_fromDirAndPt(n._UnderlyingPtr, p._UnderlyingPtr), is_owning: true);
        }

        /// returns distance from given point to this plane (if n is a unit vector)
        /// Generated from method `MR::Plane3f::distance`.
        public unsafe float Distance(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_distance", ExactSpelling = true)]
            extern static float __MR_Plane3f_distance(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_Plane3f_distance(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns same plane represented with flipped direction of n-vector
        /// Generated from method `MR::Plane3f::operator-`.
        public static unsafe MR.Plane3f operator-(MR.Const_Plane3f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Plane3f", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_neg_MR_Plane3f(MR.Const_Plane3f._Underlying *_this);
            return new(__MR_neg_MR_Plane3f(_this._UnderlyingPtr), is_owning: true);
        }

        /// returns same representation
        /// Generated from method `MR::Plane3f::operator+`.
        public static unsafe MR.Const_Plane3f operator+(MR.Const_Plane3f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Plane3f", ExactSpelling = true)]
            extern static MR.Const_Plane3f._Underlying *__MR_pos_MR_Plane3f(MR.Const_Plane3f._Underlying *_this);
            return new(__MR_pos_MR_Plane3f(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns same plane represented with unit n-vector
        /// Generated from method `MR::Plane3f::normalized`.
        public unsafe MR.Plane3f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_normalized", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_normalized(_Underlying *_this);
            return new(__MR_Plane3f_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// finds the closest point on plane
        /// Generated from method `MR::Plane3f::project`.
        public unsafe MR.Vector3f Project(MR.Const_Vector3f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_project", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Plane3f_project(_Underlying *_this, MR.Const_Vector3f._Underlying *p);
            return __MR_Plane3f_project(_UnderlyingPtr, p._UnderlyingPtr);
        }
    }

    /// 3-dimensional plane: dot(n,x) - d = 0
    /// Generated from class `MR::Plane3f`.
    /// This is the non-const half of the class.
    public class Plane3f : Const_Plane3f
    {
        internal unsafe Plane3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_GetMutable_n", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Plane3f_GetMutable_n(_Underlying *_this);
                return new(__MR_Plane3f_GetMutable_n(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_GetMutable_d", ExactSpelling = true)]
                extern static float *__MR_Plane3f_GetMutable_d(_Underlying *_this);
                return ref *__MR_Plane3f_GetMutable_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Plane3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Plane3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Plane3f::Plane3f`.
        public unsafe Plane3f(MR.Const_Plane3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_ConstructFromAnother(MR.Plane3f._Underlying *_other);
            _UnderlyingPtr = __MR_Plane3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Plane3f::Plane3f`.
        public unsafe Plane3f(MR.Const_Vector3f n, float d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Construct", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_Construct(MR.Const_Vector3f._Underlying *n, float d);
            _UnderlyingPtr = __MR_Plane3f_Construct(n._UnderlyingPtr, d);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Plane3f::Plane3f`.
        public unsafe Plane3f(MR.Const_Plane3d p) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_Construct_double", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_Construct_double(MR.Const_Plane3d._Underlying *p);
            _UnderlyingPtr = __MR_Plane3f_Construct_double(p._UnderlyingPtr);
        }

        /// Generated from method `MR::Plane3f::operator=`.
        public unsafe MR.Plane3f Assign(MR.Const_Plane3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Plane3f_AssignFromAnother(_Underlying *_this, MR.Plane3f._Underlying *_other);
            return new(__MR_Plane3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Plane3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Plane3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Plane3f`/`Const_Plane3f` directly.
    public class _InOptMut_Plane3f
    {
        public Plane3f? Opt;

        public _InOptMut_Plane3f() {}
        public _InOptMut_Plane3f(Plane3f value) {Opt = value;}
        public static implicit operator _InOptMut_Plane3f(Plane3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Plane3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Plane3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Plane3f`/`Const_Plane3f` to pass it to the function.
    public class _InOptConst_Plane3f
    {
        public Const_Plane3f? Opt;

        public _InOptConst_Plane3f() {}
        public _InOptConst_Plane3f(Const_Plane3f value) {Opt = value;}
        public static implicit operator _InOptConst_Plane3f(Const_Plane3f value) {return new(value);}
    }

    /// 3-dimensional plane: dot(n,x) - d = 0
    /// Generated from class `MR::Plane3d`.
    /// This is the const half of the class.
    public class Const_Plane3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Plane3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Plane3d_Destroy(_Underlying *_this);
            __MR_Plane3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Plane3d() {Dispose(false);}

        public unsafe MR.Const_Vector3d N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Get_n", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_Plane3d_Get_n(_Underlying *_this);
                return new(__MR_Plane3d_Get_n(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe double D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Get_d", ExactSpelling = true)]
                extern static double *__MR_Plane3d_Get_d(_Underlying *_this);
                return *__MR_Plane3d_Get_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Plane3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Plane3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Plane3d::Plane3d`.
        public unsafe Const_Plane3d(MR.Const_Plane3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_ConstructFromAnother(MR.Plane3d._Underlying *_other);
            _UnderlyingPtr = __MR_Plane3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Plane3d::Plane3d`.
        public unsafe Const_Plane3d(MR.Const_Vector3d n, double d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Construct", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_Construct(MR.Const_Vector3d._Underlying *n, double d);
            _UnderlyingPtr = __MR_Plane3d_Construct(n._UnderlyingPtr, d);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Plane3d::Plane3d`.
        public unsafe Const_Plane3d(MR.Const_Plane3f p) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Construct_float", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_Construct_float(MR.Const_Plane3f._Underlying *p);
            _UnderlyingPtr = __MR_Plane3d_Construct_float(p._UnderlyingPtr);
        }

        /// Generated from method `MR::Plane3d::fromDirAndPt`.
        public static unsafe MR.Plane3d FromDirAndPt(MR.Const_Vector3d n, MR.Const_Vector3d p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_fromDirAndPt", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_fromDirAndPt(MR.Const_Vector3d._Underlying *n, MR.Const_Vector3d._Underlying *p);
            return new(__MR_Plane3d_fromDirAndPt(n._UnderlyingPtr, p._UnderlyingPtr), is_owning: true);
        }

        /// returns distance from given point to this plane (if n is a unit vector)
        /// Generated from method `MR::Plane3d::distance`.
        public unsafe double Distance(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_distance", ExactSpelling = true)]
            extern static double __MR_Plane3d_distance(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_Plane3d_distance(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns same plane represented with flipped direction of n-vector
        /// Generated from method `MR::Plane3d::operator-`.
        public static unsafe MR.Plane3d operator-(MR.Const_Plane3d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Plane3d", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_neg_MR_Plane3d(MR.Const_Plane3d._Underlying *_this);
            return new(__MR_neg_MR_Plane3d(_this._UnderlyingPtr), is_owning: true);
        }

        /// returns same representation
        /// Generated from method `MR::Plane3d::operator+`.
        public static unsafe MR.Const_Plane3d operator+(MR.Const_Plane3d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Plane3d", ExactSpelling = true)]
            extern static MR.Const_Plane3d._Underlying *__MR_pos_MR_Plane3d(MR.Const_Plane3d._Underlying *_this);
            return new(__MR_pos_MR_Plane3d(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns same plane represented with unit n-vector
        /// Generated from method `MR::Plane3d::normalized`.
        public unsafe MR.Plane3d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_normalized", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_normalized(_Underlying *_this);
            return new(__MR_Plane3d_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// finds the closest point on plane
        /// Generated from method `MR::Plane3d::project`.
        public unsafe MR.Vector3d Project(MR.Const_Vector3d p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_project", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Plane3d_project(_Underlying *_this, MR.Const_Vector3d._Underlying *p);
            return __MR_Plane3d_project(_UnderlyingPtr, p._UnderlyingPtr);
        }
    }

    /// 3-dimensional plane: dot(n,x) - d = 0
    /// Generated from class `MR::Plane3d`.
    /// This is the non-const half of the class.
    public class Plane3d : Const_Plane3d
    {
        internal unsafe Plane3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3d N
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_GetMutable_n", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_Plane3d_GetMutable_n(_Underlying *_this);
                return new(__MR_Plane3d_GetMutable_n(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref double D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_GetMutable_d", ExactSpelling = true)]
                extern static double *__MR_Plane3d_GetMutable_d(_Underlying *_this);
                return ref *__MR_Plane3d_GetMutable_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Plane3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Plane3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Plane3d::Plane3d`.
        public unsafe Plane3d(MR.Const_Plane3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_ConstructFromAnother(MR.Plane3d._Underlying *_other);
            _UnderlyingPtr = __MR_Plane3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Plane3d::Plane3d`.
        public unsafe Plane3d(MR.Const_Vector3d n, double d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Construct", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_Construct(MR.Const_Vector3d._Underlying *n, double d);
            _UnderlyingPtr = __MR_Plane3d_Construct(n._UnderlyingPtr, d);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Plane3d::Plane3d`.
        public unsafe Plane3d(MR.Const_Plane3f p) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_Construct_float", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_Construct_float(MR.Const_Plane3f._Underlying *p);
            _UnderlyingPtr = __MR_Plane3d_Construct_float(p._UnderlyingPtr);
        }

        /// Generated from method `MR::Plane3d::operator=`.
        public unsafe MR.Plane3d Assign(MR.Const_Plane3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Plane3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Plane3d_AssignFromAnother(_Underlying *_this, MR.Plane3d._Underlying *_other);
            return new(__MR_Plane3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Plane3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Plane3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Plane3d`/`Const_Plane3d` directly.
    public class _InOptMut_Plane3d
    {
        public Plane3d? Opt;

        public _InOptMut_Plane3d() {}
        public _InOptMut_Plane3d(Plane3d value) {Opt = value;}
        public static implicit operator _InOptMut_Plane3d(Plane3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Plane3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Plane3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Plane3d`/`Const_Plane3d` to pass it to the function.
    public class _InOptConst_Plane3d
    {
        public Const_Plane3d? Opt;

        public _InOptConst_Plane3d() {}
        public _InOptConst_Plane3d(Const_Plane3d value) {Opt = value;}
        public static implicit operator _InOptConst_Plane3d(Const_Plane3d value) {return new(value);}
    }
}
