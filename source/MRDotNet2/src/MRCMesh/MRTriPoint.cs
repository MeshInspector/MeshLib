public static partial class MR
{
    /// \brief encodes a point inside a triangle using barycentric coordinates
    /// \details Notations used below: v0, v1, v2 - points of the triangle
    /// Generated from class `MR::TriPointf`.
    /// This is the const half of the class.
    public class Const_TriPointf : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_TriPointf>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TriPointf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Destroy", ExactSpelling = true)]
            extern static void __MR_TriPointf_Destroy(_Underlying *_this);
            __MR_TriPointf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TriPointf() {Dispose(false);}

        public static unsafe float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Get_eps", ExactSpelling = true)]
                extern static float *__MR_TriPointf_Get_eps();
                return *__MR_TriPointf_Get_eps();
            }
        }

        ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public unsafe float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Get_a", ExactSpelling = true)]
                extern static float *__MR_TriPointf_Get_a(_Underlying *_this);
                return *__MR_TriPointf_Get_a(_UnderlyingPtr);
            }
        }

        ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public unsafe float B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Get_b", ExactSpelling = true)]
                extern static float *__MR_TriPointf_Get_b(_Underlying *_this);
                return *__MR_TriPointf_Get_b(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TriPointf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_DefaultConstruct();
            _UnderlyingPtr = __MR_TriPointf_DefaultConstruct();
        }

        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe Const_TriPointf(MR.Const_TriPointf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_ConstructFromAnother(MR.TriPointf._Underlying *_other);
            _UnderlyingPtr = __MR_TriPointf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe Const_TriPointf(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_1", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_TriPointf_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe Const_TriPointf(float a, float b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_2", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_2(float a, float b);
            _UnderlyingPtr = __MR_TriPointf_Construct_2(a, b);
        }

        /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe Const_TriPointf(MR.Const_Vector3f p, MR.Const_Vector3f v0, MR.Const_Vector3f v1, MR.Const_Vector3f v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_4", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_4(MR.Const_Vector3f._Underlying *p, MR.Const_Vector3f._Underlying *v0, MR.Const_Vector3f._Underlying *v1, MR.Const_Vector3f._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointf_Construct_4(p._UnderlyingPtr, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe Const_TriPointf(MR.Const_Vector3f p, MR.Const_Vector3f v1, MR.Const_Vector3f v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_3", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_3(MR.Const_Vector3f._Underlying *p, MR.Const_Vector3f._Underlying *v1, MR.Const_Vector3f._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointf_Construct_3(p._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// represents the same point relative to next edge in the same triangle
        /// Generated from method `MR::TriPointf::lnext`.
        public unsafe MR.TriPointf Lnext()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_lnext", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_lnext(_Underlying *_this);
            return new(__MR_TriPointf_lnext(_UnderlyingPtr), is_owning: true);
        }

        /// returns [0,2] if the point is in a vertex or -1 otherwise
        /// Generated from method `MR::TriPointf::inVertex`.
        public unsafe int InVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_inVertex", ExactSpelling = true)]
            extern static int __MR_TriPointf_inVertex(_Underlying *_this);
            return __MR_TriPointf_inVertex(_UnderlyingPtr);
        }

        /// returns [0,2] if the point is on edge or -1 otherwise:
        /// 0 means edge [v1,v2]; 1 means edge [v2,v0]; 2 means edge [v0,v1]
        /// Generated from method `MR::TriPointf::onEdge`.
        public unsafe int OnEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_onEdge", ExactSpelling = true)]
            extern static int __MR_TriPointf_onEdge(_Underlying *_this);
            return __MR_TriPointf_onEdge(_UnderlyingPtr);
        }

        /// returns true if two points have equal (a,b) representation
        /// Generated from method `MR::TriPointf::operator==`.
        public static unsafe bool operator==(MR.Const_TriPointf _this, MR.Const_TriPointf rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_TriPointf", ExactSpelling = true)]
            extern static byte __MR_equal_MR_TriPointf(MR.Const_TriPointf._Underlying *_this, MR.Const_TriPointf._Underlying *rhs);
            return __MR_equal_MR_TriPointf(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_TriPointf _this, MR.Const_TriPointf rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_TriPointf? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_TriPointf)
                return this == (MR.Const_TriPointf)other;
            return false;
        }
    }

    /// \brief encodes a point inside a triangle using barycentric coordinates
    /// \details Notations used below: v0, v1, v2 - points of the triangle
    /// Generated from class `MR::TriPointf`.
    /// This is the non-const half of the class.
    public class TriPointf : Const_TriPointf
    {
        internal unsafe TriPointf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public new unsafe ref float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_GetMutable_a", ExactSpelling = true)]
                extern static float *__MR_TriPointf_GetMutable_a(_Underlying *_this);
                return ref *__MR_TriPointf_GetMutable_a(_UnderlyingPtr);
            }
        }

        ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public new unsafe ref float B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_GetMutable_b", ExactSpelling = true)]
                extern static float *__MR_TriPointf_GetMutable_b(_Underlying *_this);
                return ref *__MR_TriPointf_GetMutable_b(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TriPointf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_DefaultConstruct();
            _UnderlyingPtr = __MR_TriPointf_DefaultConstruct();
        }

        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe TriPointf(MR.Const_TriPointf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_ConstructFromAnother(MR.TriPointf._Underlying *_other);
            _UnderlyingPtr = __MR_TriPointf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe TriPointf(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_1", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_TriPointf_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe TriPointf(float a, float b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_2", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_2(float a, float b);
            _UnderlyingPtr = __MR_TriPointf_Construct_2(a, b);
        }

        /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe TriPointf(MR.Const_Vector3f p, MR.Const_Vector3f v0, MR.Const_Vector3f v1, MR.Const_Vector3f v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_4", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_4(MR.Const_Vector3f._Underlying *p, MR.Const_Vector3f._Underlying *v0, MR.Const_Vector3f._Underlying *v1, MR.Const_Vector3f._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointf_Construct_4(p._UnderlyingPtr, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointf::TriPointf`.
        public unsafe TriPointf(MR.Const_Vector3f p, MR.Const_Vector3f v1, MR.Const_Vector3f v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_Construct_3", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_Construct_3(MR.Const_Vector3f._Underlying *p, MR.Const_Vector3f._Underlying *v1, MR.Const_Vector3f._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointf_Construct_3(p._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// Generated from method `MR::TriPointf::operator=`.
        public unsafe MR.TriPointf Assign(MR.Const_TriPointf _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TriPointf._Underlying *__MR_TriPointf_AssignFromAnother(_Underlying *_this, MR.TriPointf._Underlying *_other);
            return new(__MR_TriPointf_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TriPointf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TriPointf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriPointf`/`Const_TriPointf` directly.
    public class _InOptMut_TriPointf
    {
        public TriPointf? Opt;

        public _InOptMut_TriPointf() {}
        public _InOptMut_TriPointf(TriPointf value) {Opt = value;}
        public static implicit operator _InOptMut_TriPointf(TriPointf value) {return new(value);}
    }

    /// This is used for optional parameters of class `TriPointf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TriPointf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriPointf`/`Const_TriPointf` to pass it to the function.
    public class _InOptConst_TriPointf
    {
        public Const_TriPointf? Opt;

        public _InOptConst_TriPointf() {}
        public _InOptConst_TriPointf(Const_TriPointf value) {Opt = value;}
        public static implicit operator _InOptConst_TriPointf(Const_TriPointf value) {return new(value);}
    }

    /// \brief encodes a point inside a triangle using barycentric coordinates
    /// \details Notations used below: v0, v1, v2 - points of the triangle
    /// Generated from class `MR::TriPointd`.
    /// This is the const half of the class.
    public class Const_TriPointd : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_TriPointd>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TriPointd(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Destroy", ExactSpelling = true)]
            extern static void __MR_TriPointd_Destroy(_Underlying *_this);
            __MR_TriPointd_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TriPointd() {Dispose(false);}

        public static unsafe double Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Get_eps", ExactSpelling = true)]
                extern static double *__MR_TriPointd_Get_eps();
                return *__MR_TriPointd_Get_eps();
            }
        }

        ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public unsafe double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Get_a", ExactSpelling = true)]
                extern static double *__MR_TriPointd_Get_a(_Underlying *_this);
                return *__MR_TriPointd_Get_a(_UnderlyingPtr);
            }
        }

        ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public unsafe double B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Get_b", ExactSpelling = true)]
                extern static double *__MR_TriPointd_Get_b(_Underlying *_this);
                return *__MR_TriPointd_Get_b(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TriPointd() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_DefaultConstruct();
            _UnderlyingPtr = __MR_TriPointd_DefaultConstruct();
        }

        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe Const_TriPointd(MR.Const_TriPointd _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_ConstructFromAnother(MR.TriPointd._Underlying *_other);
            _UnderlyingPtr = __MR_TriPointd_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe Const_TriPointd(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_1", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_TriPointd_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe Const_TriPointd(double a, double b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_2", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_2(double a, double b);
            _UnderlyingPtr = __MR_TriPointd_Construct_2(a, b);
        }

        /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe Const_TriPointd(MR.Const_Vector3d p, MR.Const_Vector3d v0, MR.Const_Vector3d v1, MR.Const_Vector3d v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_4", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_4(MR.Const_Vector3d._Underlying *p, MR.Const_Vector3d._Underlying *v0, MR.Const_Vector3d._Underlying *v1, MR.Const_Vector3d._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointd_Construct_4(p._UnderlyingPtr, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe Const_TriPointd(MR.Const_Vector3d p, MR.Const_Vector3d v1, MR.Const_Vector3d v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_3", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_3(MR.Const_Vector3d._Underlying *p, MR.Const_Vector3d._Underlying *v1, MR.Const_Vector3d._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointd_Construct_3(p._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// represents the same point relative to next edge in the same triangle
        /// Generated from method `MR::TriPointd::lnext`.
        public unsafe MR.TriPointd Lnext()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_lnext", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_lnext(_Underlying *_this);
            return new(__MR_TriPointd_lnext(_UnderlyingPtr), is_owning: true);
        }

        /// returns [0,2] if the point is in a vertex or -1 otherwise
        /// Generated from method `MR::TriPointd::inVertex`.
        public unsafe int InVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_inVertex", ExactSpelling = true)]
            extern static int __MR_TriPointd_inVertex(_Underlying *_this);
            return __MR_TriPointd_inVertex(_UnderlyingPtr);
        }

        /// returns [0,2] if the point is on edge or -1 otherwise:
        /// 0 means edge [v1,v2]; 1 means edge [v2,v0]; 2 means edge [v0,v1]
        /// Generated from method `MR::TriPointd::onEdge`.
        public unsafe int OnEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_onEdge", ExactSpelling = true)]
            extern static int __MR_TriPointd_onEdge(_Underlying *_this);
            return __MR_TriPointd_onEdge(_UnderlyingPtr);
        }

        /// returns true if two points have equal (a,b) representation
        /// Generated from method `MR::TriPointd::operator==`.
        public static unsafe bool operator==(MR.Const_TriPointd _this, MR.Const_TriPointd rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_TriPointd", ExactSpelling = true)]
            extern static byte __MR_equal_MR_TriPointd(MR.Const_TriPointd._Underlying *_this, MR.Const_TriPointd._Underlying *rhs);
            return __MR_equal_MR_TriPointd(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_TriPointd _this, MR.Const_TriPointd rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_TriPointd? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_TriPointd)
                return this == (MR.Const_TriPointd)other;
            return false;
        }
    }

    /// \brief encodes a point inside a triangle using barycentric coordinates
    /// \details Notations used below: v0, v1, v2 - points of the triangle
    /// Generated from class `MR::TriPointd`.
    /// This is the non-const half of the class.
    public class TriPointd : Const_TriPointd
    {
        internal unsafe TriPointd(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public new unsafe ref double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_GetMutable_a", ExactSpelling = true)]
                extern static double *__MR_TriPointd_GetMutable_a(_Underlying *_this);
                return ref *__MR_TriPointd_GetMutable_a(_UnderlyingPtr);
            }
        }

        ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public new unsafe ref double B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_GetMutable_b", ExactSpelling = true)]
                extern static double *__MR_TriPointd_GetMutable_b(_Underlying *_this);
                return ref *__MR_TriPointd_GetMutable_b(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TriPointd() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_DefaultConstruct();
            _UnderlyingPtr = __MR_TriPointd_DefaultConstruct();
        }

        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe TriPointd(MR.Const_TriPointd _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_ConstructFromAnother(MR.TriPointd._Underlying *_other);
            _UnderlyingPtr = __MR_TriPointd_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe TriPointd(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_1", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_TriPointd_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe TriPointd(double a, double b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_2", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_2(double a, double b);
            _UnderlyingPtr = __MR_TriPointd_Construct_2(a, b);
        }

        /// given a point coordinates and triangle (v0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe TriPointd(MR.Const_Vector3d p, MR.Const_Vector3d v0, MR.Const_Vector3d v1, MR.Const_Vector3d v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_4", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_4(MR.Const_Vector3d._Underlying *p, MR.Const_Vector3d._Underlying *v0, MR.Const_Vector3d._Underlying *v1, MR.Const_Vector3d._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointd_Construct_4(p._UnderlyingPtr, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// given a point coordinates and triangle (0,v1,v2) computes barycentric coordinates of the point
        /// Generated from constructor `MR::TriPointd::TriPointd`.
        public unsafe TriPointd(MR.Const_Vector3d p, MR.Const_Vector3d v1, MR.Const_Vector3d v2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_Construct_3", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_Construct_3(MR.Const_Vector3d._Underlying *p, MR.Const_Vector3d._Underlying *v1, MR.Const_Vector3d._Underlying *v2);
            _UnderlyingPtr = __MR_TriPointd_Construct_3(p._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// Generated from method `MR::TriPointd::operator=`.
        public unsafe MR.TriPointd Assign(MR.Const_TriPointd _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriPointd_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TriPointd._Underlying *__MR_TriPointd_AssignFromAnother(_Underlying *_this, MR.TriPointd._Underlying *_other);
            return new(__MR_TriPointd_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TriPointd` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TriPointd`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriPointd`/`Const_TriPointd` directly.
    public class _InOptMut_TriPointd
    {
        public TriPointd? Opt;

        public _InOptMut_TriPointd() {}
        public _InOptMut_TriPointd(TriPointd value) {Opt = value;}
        public static implicit operator _InOptMut_TriPointd(TriPointd value) {return new(value);}
    }

    /// This is used for optional parameters of class `TriPointd` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TriPointd`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriPointd`/`Const_TriPointd` to pass it to the function.
    public class _InOptConst_TriPointd
    {
        public Const_TriPointd? Opt;

        public _InOptConst_TriPointd() {}
        public _InOptConst_TriPointd(Const_TriPointd value) {Opt = value;}
        public static implicit operator _InOptConst_TriPointd(Const_TriPointd value) {return new(value);}
    }
}
