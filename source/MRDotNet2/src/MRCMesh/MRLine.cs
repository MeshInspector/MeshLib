public static partial class MR
{
    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line2f`.
    /// This is the const half of the class.
    public class Const_Line2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Line2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_Destroy", ExactSpelling = true)]
            extern static void __MR_Line2f_Destroy(_Underlying *_this);
            __MR_Line2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Line2f() {Dispose(false);}

        public unsafe MR.Const_Vector2f P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_Get_p", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_Line2f_Get_p(_Underlying *_this);
                return new(__MR_Line2f_Get_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector2f D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_Get_d", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_Line2f_Get_d(_Underlying *_this);
                return new(__MR_Line2f_Get_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Line2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_DefaultConstruct();
            _UnderlyingPtr = __MR_Line2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line2f::Line2f`.
        public unsafe Const_Line2f(MR.Const_Line2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_ConstructFromAnother(MR.Line2f._Underlying *_other);
            _UnderlyingPtr = __MR_Line2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line2f::Line2f`.
        public unsafe Const_Line2f(MR.Const_Vector2f p, MR.Const_Vector2f d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_Construct", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_Construct(MR.Const_Vector2f._Underlying *p, MR.Const_Vector2f._Underlying *d);
            _UnderlyingPtr = __MR_Line2f_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns p and param=1 returns p+d
        /// Generated from method `MR::Line2f::operator()`.
        public unsafe MR.Vector2f Call(float param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_call", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Line2f_call(_Underlying *_this, float param);
            return __MR_Line2f_call(_UnderlyingPtr, param);
        }

        /// returns squared distance from given point to this line
        /// Generated from method `MR::Line2f::distanceSq`.
        public unsafe float DistanceSq(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_distanceSq", ExactSpelling = true)]
            extern static float __MR_Line2f_distanceSq(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_Line2f_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns same line represented with flipped direction of d-vector
        /// Generated from method `MR::Line2f::operator-`.
        public static unsafe MR.Line2f operator-(MR.Const_Line2f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Line2f", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_neg_MR_Line2f(MR.Const_Line2f._Underlying *_this);
            return new(__MR_neg_MR_Line2f(_this._UnderlyingPtr), is_owning: true);
        }

        /// returns same representation
        /// Generated from method `MR::Line2f::operator+`.
        public static unsafe MR.Const_Line2f operator+(MR.Const_Line2f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Line2f", ExactSpelling = true)]
            extern static MR.Const_Line2f._Underlying *__MR_pos_MR_Line2f(MR.Const_Line2f._Underlying *_this);
            return new(__MR_pos_MR_Line2f(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns same line represented with unit d-vector
        /// Generated from method `MR::Line2f::normalized`.
        public unsafe MR.Line2f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_normalized", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_normalized(_Underlying *_this);
            return new(__MR_Line2f_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// finds the closest point on line
        /// Generated from method `MR::Line2f::project`.
        public unsafe MR.Vector2f Project(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_project", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Line2f_project(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_Line2f_project(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line2f`.
    /// This is the non-const half of the class.
    public class Line2f : Const_Line2f
    {
        internal unsafe Line2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector2f P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_GetMutable_p", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_Line2f_GetMutable_p(_Underlying *_this);
                return new(__MR_Line2f_GetMutable_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector2f D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_GetMutable_d", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_Line2f_GetMutable_d(_Underlying *_this);
                return new(__MR_Line2f_GetMutable_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Line2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_DefaultConstruct();
            _UnderlyingPtr = __MR_Line2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line2f::Line2f`.
        public unsafe Line2f(MR.Const_Line2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_ConstructFromAnother(MR.Line2f._Underlying *_other);
            _UnderlyingPtr = __MR_Line2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line2f::Line2f`.
        public unsafe Line2f(MR.Const_Vector2f p, MR.Const_Vector2f d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_Construct", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_Construct(MR.Const_Vector2f._Underlying *p, MR.Const_Vector2f._Underlying *d);
            _UnderlyingPtr = __MR_Line2f_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// Generated from method `MR::Line2f::operator=`.
        public unsafe MR.Line2f Assign(MR.Const_Line2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Line2f._Underlying *__MR_Line2f_AssignFromAnother(_Underlying *_this, MR.Line2f._Underlying *_other);
            return new(__MR_Line2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Line2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Line2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line2f`/`Const_Line2f` directly.
    public class _InOptMut_Line2f
    {
        public Line2f? Opt;

        public _InOptMut_Line2f() {}
        public _InOptMut_Line2f(Line2f value) {Opt = value;}
        public static implicit operator _InOptMut_Line2f(Line2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Line2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Line2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line2f`/`Const_Line2f` to pass it to the function.
    public class _InOptConst_Line2f
    {
        public Const_Line2f? Opt;

        public _InOptConst_Line2f() {}
        public _InOptConst_Line2f(Const_Line2f value) {Opt = value;}
        public static implicit operator _InOptConst_Line2f(Const_Line2f value) {return new(value);}
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line2d`.
    /// This is the const half of the class.
    public class Const_Line2d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Line2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_Destroy", ExactSpelling = true)]
            extern static void __MR_Line2d_Destroy(_Underlying *_this);
            __MR_Line2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Line2d() {Dispose(false);}

        public unsafe MR.Const_Vector2d P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_Get_p", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_Line2d_Get_p(_Underlying *_this);
                return new(__MR_Line2d_Get_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector2d D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_Get_d", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_Line2d_Get_d(_Underlying *_this);
                return new(__MR_Line2d_Get_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Line2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_DefaultConstruct();
            _UnderlyingPtr = __MR_Line2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line2d::Line2d`.
        public unsafe Const_Line2d(MR.Const_Line2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_ConstructFromAnother(MR.Line2d._Underlying *_other);
            _UnderlyingPtr = __MR_Line2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line2d::Line2d`.
        public unsafe Const_Line2d(MR.Const_Vector2d p, MR.Const_Vector2d d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_Construct", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_Construct(MR.Const_Vector2d._Underlying *p, MR.Const_Vector2d._Underlying *d);
            _UnderlyingPtr = __MR_Line2d_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns p and param=1 returns p+d
        /// Generated from method `MR::Line2d::operator()`.
        public unsafe MR.Vector2d Call(double param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_call", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Line2d_call(_Underlying *_this, double param);
            return __MR_Line2d_call(_UnderlyingPtr, param);
        }

        /// returns squared distance from given point to this line
        /// Generated from method `MR::Line2d::distanceSq`.
        public unsafe double DistanceSq(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_distanceSq", ExactSpelling = true)]
            extern static double __MR_Line2d_distanceSq(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_Line2d_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns same line represented with flipped direction of d-vector
        /// Generated from method `MR::Line2d::operator-`.
        public static unsafe MR.Line2d operator-(MR.Const_Line2d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Line2d", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_neg_MR_Line2d(MR.Const_Line2d._Underlying *_this);
            return new(__MR_neg_MR_Line2d(_this._UnderlyingPtr), is_owning: true);
        }

        /// returns same representation
        /// Generated from method `MR::Line2d::operator+`.
        public static unsafe MR.Const_Line2d operator+(MR.Const_Line2d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Line2d", ExactSpelling = true)]
            extern static MR.Const_Line2d._Underlying *__MR_pos_MR_Line2d(MR.Const_Line2d._Underlying *_this);
            return new(__MR_pos_MR_Line2d(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns same line represented with unit d-vector
        /// Generated from method `MR::Line2d::normalized`.
        public unsafe MR.Line2d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_normalized", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_normalized(_Underlying *_this);
            return new(__MR_Line2d_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// finds the closest point on line
        /// Generated from method `MR::Line2d::project`.
        public unsafe MR.Vector2d Project(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_project", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Line2d_project(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_Line2d_project(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line2d`.
    /// This is the non-const half of the class.
    public class Line2d : Const_Line2d
    {
        internal unsafe Line2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector2d P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_GetMutable_p", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_Line2d_GetMutable_p(_Underlying *_this);
                return new(__MR_Line2d_GetMutable_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector2d D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_GetMutable_d", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_Line2d_GetMutable_d(_Underlying *_this);
                return new(__MR_Line2d_GetMutable_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Line2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_DefaultConstruct();
            _UnderlyingPtr = __MR_Line2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line2d::Line2d`.
        public unsafe Line2d(MR.Const_Line2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_ConstructFromAnother(MR.Line2d._Underlying *_other);
            _UnderlyingPtr = __MR_Line2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line2d::Line2d`.
        public unsafe Line2d(MR.Const_Vector2d p, MR.Const_Vector2d d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_Construct", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_Construct(MR.Const_Vector2d._Underlying *p, MR.Const_Vector2d._Underlying *d);
            _UnderlyingPtr = __MR_Line2d_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// Generated from method `MR::Line2d::operator=`.
        public unsafe MR.Line2d Assign(MR.Const_Line2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Line2d._Underlying *__MR_Line2d_AssignFromAnother(_Underlying *_this, MR.Line2d._Underlying *_other);
            return new(__MR_Line2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Line2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Line2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line2d`/`Const_Line2d` directly.
    public class _InOptMut_Line2d
    {
        public Line2d? Opt;

        public _InOptMut_Line2d() {}
        public _InOptMut_Line2d(Line2d value) {Opt = value;}
        public static implicit operator _InOptMut_Line2d(Line2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Line2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Line2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line2d`/`Const_Line2d` to pass it to the function.
    public class _InOptConst_Line2d
    {
        public Const_Line2d? Opt;

        public _InOptConst_Line2d() {}
        public _InOptConst_Line2d(Const_Line2d value) {Opt = value;}
        public static implicit operator _InOptConst_Line2d(Const_Line2d value) {return new(value);}
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line3f`.
    /// This is the const half of the class.
    public class Const_Line3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Line3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Line3f_Destroy(_Underlying *_this);
            __MR_Line3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Line3f() {Dispose(false);}

        public unsafe MR.Const_Vector3f P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Get_p", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Line3f_Get_p(_Underlying *_this);
                return new(__MR_Line3f_Get_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Get_d", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Line3f_Get_d(_Underlying *_this);
                return new(__MR_Line3f_Get_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Line3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line3f::Line3f`.
        public unsafe Const_Line3f(MR.Const_Line3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_ConstructFromAnother(MR.Line3f._Underlying *_other);
            _UnderlyingPtr = __MR_Line3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line3f::Line3f`.
        public unsafe Const_Line3f(MR.Const_Vector3f p, MR.Const_Vector3f d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Construct", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_Construct(MR.Const_Vector3f._Underlying *p, MR.Const_Vector3f._Underlying *d);
            _UnderlyingPtr = __MR_Line3f_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line3f::Line3f`.
        public unsafe Const_Line3f(MR.Const_Line3d l) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Construct_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_Construct_MR_Vector3d(MR.Const_Line3d._Underlying *l);
            _UnderlyingPtr = __MR_Line3f_Construct_MR_Vector3d(l._UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns p and param=1 returns p+d
        /// Generated from method `MR::Line3f::operator()`.
        public unsafe MR.Vector3f Call(float param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_call", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Line3f_call(_Underlying *_this, float param);
            return __MR_Line3f_call(_UnderlyingPtr, param);
        }

        /// returns squared distance from given point to this line
        /// Generated from method `MR::Line3f::distanceSq`.
        public unsafe float DistanceSq(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_distanceSq", ExactSpelling = true)]
            extern static float __MR_Line3f_distanceSq(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_Line3f_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns same line represented with flipped direction of d-vector
        /// Generated from method `MR::Line3f::operator-`.
        public static unsafe MR.Line3f operator-(MR.Const_Line3f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Line3f", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_neg_MR_Line3f(MR.Const_Line3f._Underlying *_this);
            return new(__MR_neg_MR_Line3f(_this._UnderlyingPtr), is_owning: true);
        }

        /// returns same representation
        /// Generated from method `MR::Line3f::operator+`.
        public static unsafe MR.Const_Line3f operator+(MR.Const_Line3f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Line3f", ExactSpelling = true)]
            extern static MR.Const_Line3f._Underlying *__MR_pos_MR_Line3f(MR.Const_Line3f._Underlying *_this);
            return new(__MR_pos_MR_Line3f(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns same line represented with unit d-vector
        /// Generated from method `MR::Line3f::normalized`.
        public unsafe MR.Line3f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_normalized", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_normalized(_Underlying *_this);
            return new(__MR_Line3f_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// finds the closest point on line
        /// Generated from method `MR::Line3f::project`.
        public unsafe MR.Vector3f Project(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_project", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Line3f_project(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_Line3f_project(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line3f`.
    /// This is the non-const half of the class.
    public class Line3f : Const_Line3f
    {
        internal unsafe Line3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_GetMutable_p", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Line3f_GetMutable_p(_Underlying *_this);
                return new(__MR_Line3f_GetMutable_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_GetMutable_d", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Line3f_GetMutable_d(_Underlying *_this);
                return new(__MR_Line3f_GetMutable_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Line3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line3f::Line3f`.
        public unsafe Line3f(MR.Const_Line3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_ConstructFromAnother(MR.Line3f._Underlying *_other);
            _UnderlyingPtr = __MR_Line3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line3f::Line3f`.
        public unsafe Line3f(MR.Const_Vector3f p, MR.Const_Vector3f d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Construct", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_Construct(MR.Const_Vector3f._Underlying *p, MR.Const_Vector3f._Underlying *d);
            _UnderlyingPtr = __MR_Line3f_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line3f::Line3f`.
        public unsafe Line3f(MR.Const_Line3d l) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_Construct_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_Construct_MR_Vector3d(MR.Const_Line3d._Underlying *l);
            _UnderlyingPtr = __MR_Line3f_Construct_MR_Vector3d(l._UnderlyingPtr);
        }

        /// Generated from method `MR::Line3f::operator=`.
        public unsafe MR.Line3f Assign(MR.Const_Line3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_Line3f_AssignFromAnother(_Underlying *_this, MR.Line3f._Underlying *_other);
            return new(__MR_Line3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Line3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Line3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3f`/`Const_Line3f` directly.
    public class _InOptMut_Line3f
    {
        public Line3f? Opt;

        public _InOptMut_Line3f() {}
        public _InOptMut_Line3f(Line3f value) {Opt = value;}
        public static implicit operator _InOptMut_Line3f(Line3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Line3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Line3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3f`/`Const_Line3f` to pass it to the function.
    public class _InOptConst_Line3f
    {
        public Const_Line3f? Opt;

        public _InOptConst_Line3f() {}
        public _InOptConst_Line3f(Const_Line3f value) {Opt = value;}
        public static implicit operator _InOptConst_Line3f(Const_Line3f value) {return new(value);}
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line3d`.
    /// This is the const half of the class.
    public class Const_Line3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Line3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Line3d_Destroy(_Underlying *_this);
            __MR_Line3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Line3d() {Dispose(false);}

        public unsafe MR.Const_Vector3d P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_Get_p", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_Line3d_Get_p(_Underlying *_this);
                return new(__MR_Line3d_Get_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3d D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_Get_d", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_Line3d_Get_d(_Underlying *_this);
                return new(__MR_Line3d_Get_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Line3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line3d::Line3d`.
        public unsafe Const_Line3d(MR.Const_Line3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_ConstructFromAnother(MR.Line3d._Underlying *_other);
            _UnderlyingPtr = __MR_Line3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line3d::Line3d`.
        public unsafe Const_Line3d(MR.Const_Vector3d p, MR.Const_Vector3d d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_Construct", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_Construct(MR.Const_Vector3d._Underlying *p, MR.Const_Vector3d._Underlying *d);
            _UnderlyingPtr = __MR_Line3d_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns p and param=1 returns p+d
        /// Generated from method `MR::Line3d::operator()`.
        public unsafe MR.Vector3d Call(double param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_call", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Line3d_call(_Underlying *_this, double param);
            return __MR_Line3d_call(_UnderlyingPtr, param);
        }

        /// returns squared distance from given point to this line
        /// Generated from method `MR::Line3d::distanceSq`.
        public unsafe double DistanceSq(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_distanceSq", ExactSpelling = true)]
            extern static double __MR_Line3d_distanceSq(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_Line3d_distanceSq(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// returns same line represented with flipped direction of d-vector
        /// Generated from method `MR::Line3d::operator-`.
        public static unsafe MR.Line3d operator-(MR.Const_Line3d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Line3d", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_neg_MR_Line3d(MR.Const_Line3d._Underlying *_this);
            return new(__MR_neg_MR_Line3d(_this._UnderlyingPtr), is_owning: true);
        }

        /// returns same representation
        /// Generated from method `MR::Line3d::operator+`.
        public static unsafe MR.Const_Line3d operator+(MR.Const_Line3d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Line3d", ExactSpelling = true)]
            extern static MR.Const_Line3d._Underlying *__MR_pos_MR_Line3d(MR.Const_Line3d._Underlying *_this);
            return new(__MR_pos_MR_Line3d(_this._UnderlyingPtr), is_owning: false);
        }

        /// returns same line represented with unit d-vector
        /// Generated from method `MR::Line3d::normalized`.
        public unsafe MR.Line3d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_normalized", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_normalized(_Underlying *_this);
            return new(__MR_Line3d_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// finds the closest point on line
        /// Generated from method `MR::Line3d::project`.
        public unsafe MR.Vector3d Project(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_project", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Line3d_project(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_Line3d_project(_UnderlyingPtr, x._UnderlyingPtr);
        }
    }

    /// 2- or 3-dimensional line: cross( x - p, d ) = 0
    /// Generated from class `MR::Line3d`.
    /// This is the non-const half of the class.
    public class Line3d : Const_Line3d
    {
        internal unsafe Line3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3d P
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_GetMutable_p", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_Line3d_GetMutable_p(_Underlying *_this);
                return new(__MR_Line3d_GetMutable_p(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3d D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_GetMutable_d", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_Line3d_GetMutable_d(_Underlying *_this);
                return new(__MR_Line3d_GetMutable_d(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Line3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Line3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Line3d::Line3d`.
        public unsafe Line3d(MR.Const_Line3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_ConstructFromAnother(MR.Line3d._Underlying *_other);
            _UnderlyingPtr = __MR_Line3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Line3d::Line3d`.
        public unsafe Line3d(MR.Const_Vector3d p, MR.Const_Vector3d d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_Construct", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_Construct(MR.Const_Vector3d._Underlying *p, MR.Const_Vector3d._Underlying *d);
            _UnderlyingPtr = __MR_Line3d_Construct(p._UnderlyingPtr, d._UnderlyingPtr);
        }

        /// Generated from method `MR::Line3d::operator=`.
        public unsafe MR.Line3d Assign(MR.Const_Line3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Line3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_Line3d_AssignFromAnother(_Underlying *_this, MR.Line3d._Underlying *_other);
            return new(__MR_Line3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Line3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Line3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3d`/`Const_Line3d` directly.
    public class _InOptMut_Line3d
    {
        public Line3d? Opt;

        public _InOptMut_Line3d() {}
        public _InOptMut_Line3d(Line3d value) {Opt = value;}
        public static implicit operator _InOptMut_Line3d(Line3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Line3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Line3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Line3d`/`Const_Line3d` to pass it to the function.
    public class _InOptConst_Line3d
    {
        public Const_Line3d? Opt;

        public _InOptConst_Line3d() {}
        public _InOptConst_Line3d(Const_Line3d value) {Opt = value;}
        public static implicit operator _InOptConst_Line3d(Const_Line3d value) {return new(value);}
    }
}
