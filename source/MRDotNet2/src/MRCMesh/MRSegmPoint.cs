public static partial class MR
{
    /// \brief encodes a point inside a line segment using relative distance in [0,1]
    /// Generated from class `MR::SegmPointf`.
    /// This is the const half of the class.
    public class Const_SegmPointf : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SegmPointf>, System.IEquatable<float>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SegmPointf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_Destroy", ExactSpelling = true)]
            extern static void __MR_SegmPointf_Destroy(_Underlying *_this);
            __MR_SegmPointf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SegmPointf() {Dispose(false);}

        public static unsafe float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_Get_eps", ExactSpelling = true)]
                extern static float *__MR_SegmPointf_Get_eps();
                return *__MR_SegmPointf_Get_eps();
            }
        }

        ///< a in [0,1], a=0 => point is in v0, a=1 => point is in v1
        public unsafe float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_Get_a", ExactSpelling = true)]
                extern static float *__MR_SegmPointf_Get_a(_Underlying *_this);
                return *__MR_SegmPointf_Get_a(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SegmPointf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_DefaultConstruct();
            _UnderlyingPtr = __MR_SegmPointf_DefaultConstruct();
        }

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public unsafe Const_SegmPointf(MR.Const_SegmPointf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_ConstructFromAnother(MR.SegmPointf._Underlying *_other);
            _UnderlyingPtr = __MR_SegmPointf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public unsafe Const_SegmPointf(float a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_Construct", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_Construct(float a);
            _UnderlyingPtr = __MR_SegmPointf_Construct(a);
        }

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public static unsafe implicit operator Const_SegmPointf(float a) {return new(a);}

        /// Generated from conversion operator `MR::SegmPointf::operator float`.
        public static unsafe implicit operator float(MR.Const_SegmPointf _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_ConvertTo_float", ExactSpelling = true)]
            extern static float __MR_SegmPointf_ConvertTo_float(MR.Const_SegmPointf._Underlying *_this);
            return __MR_SegmPointf_ConvertTo_float(_this._UnderlyingPtr);
        }

        /// returns [0,1] if the point is in a vertex or -1 otherwise
        /// Generated from method `MR::SegmPointf::inVertex`.
        public unsafe int InVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_inVertex", ExactSpelling = true)]
            extern static int __MR_SegmPointf_inVertex(_Underlying *_this);
            return __MR_SegmPointf_inVertex(_UnderlyingPtr);
        }

        /// represents the same point relative to oppositely directed segment
        /// Generated from method `MR::SegmPointf::sym`.
        public unsafe MR.SegmPointf Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_sym", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_sym(_Underlying *_this);
            return new(__MR_SegmPointf_sym(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if two points have equal (a) representation
        /// Generated from method `MR::SegmPointf::operator==`.
        public static unsafe bool operator==(MR.Const_SegmPointf _this, MR.Const_SegmPointf rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SegmPointf", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SegmPointf(MR.Const_SegmPointf._Underlying *_this, MR.Const_SegmPointf._Underlying *rhs);
            return __MR_equal_MR_SegmPointf(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SegmPointf _this, MR.Const_SegmPointf rhs)
        {
            return !(_this == rhs);
        }

        /// Generated from method `MR::SegmPointf::operator==`.
        public static unsafe bool operator==(MR.Const_SegmPointf _this, float rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SegmPointf_float", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SegmPointf_float(MR.Const_SegmPointf._Underlying *_this, float rhs);
            return __MR_equal_MR_SegmPointf_float(_this._UnderlyingPtr, rhs) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SegmPointf _this, float rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_SegmPointf? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public bool Equals(float rhs)
        {
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SegmPointf)
                return this == (MR.Const_SegmPointf)other;
            if (other is float)
                return this == (float)other;
            return false;
        }
    }

    /// \brief encodes a point inside a line segment using relative distance in [0,1]
    /// Generated from class `MR::SegmPointf`.
    /// This is the non-const half of the class.
    public class SegmPointf : Const_SegmPointf
    {
        internal unsafe SegmPointf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< a in [0,1], a=0 => point is in v0, a=1 => point is in v1
        public new unsafe ref float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_GetMutable_a", ExactSpelling = true)]
                extern static float *__MR_SegmPointf_GetMutable_a(_Underlying *_this);
                return ref *__MR_SegmPointf_GetMutable_a(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SegmPointf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_DefaultConstruct();
            _UnderlyingPtr = __MR_SegmPointf_DefaultConstruct();
        }

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public unsafe SegmPointf(MR.Const_SegmPointf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_ConstructFromAnother(MR.SegmPointf._Underlying *_other);
            _UnderlyingPtr = __MR_SegmPointf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public unsafe SegmPointf(float a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_Construct", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_Construct(float a);
            _UnderlyingPtr = __MR_SegmPointf_Construct(a);
        }

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public static unsafe implicit operator SegmPointf(float a) {return new(a);}

        /// Generated from conversion operator `MR::SegmPointf::operator float &`.
        public unsafe ref float ConvertTo_FloatRef()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_ConvertTo_float_ref", ExactSpelling = true)]
            extern static float *__MR_SegmPointf_ConvertTo_float_ref(_Underlying *_this);
            return ref *__MR_SegmPointf_ConvertTo_float_ref(_UnderlyingPtr);
        }

        /// Generated from method `MR::SegmPointf::operator=`.
        public unsafe MR.SegmPointf Assign(MR.Const_SegmPointf _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SegmPointf._Underlying *__MR_SegmPointf_AssignFromAnother(_Underlying *_this, MR.SegmPointf._Underlying *_other);
            return new(__MR_SegmPointf_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SegmPointf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SegmPointf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SegmPointf`/`Const_SegmPointf` directly.
    public class _InOptMut_SegmPointf
    {
        public SegmPointf? Opt;

        public _InOptMut_SegmPointf() {}
        public _InOptMut_SegmPointf(SegmPointf value) {Opt = value;}
        public static implicit operator _InOptMut_SegmPointf(SegmPointf value) {return new(value);}
    }

    /// This is used for optional parameters of class `SegmPointf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SegmPointf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SegmPointf`/`Const_SegmPointf` to pass it to the function.
    public class _InOptConst_SegmPointf
    {
        public Const_SegmPointf? Opt;

        public _InOptConst_SegmPointf() {}
        public _InOptConst_SegmPointf(Const_SegmPointf value) {Opt = value;}
        public static implicit operator _InOptConst_SegmPointf(Const_SegmPointf value) {return new(value);}

        /// Generated from constructor `MR::SegmPointf::SegmPointf`.
        public static unsafe implicit operator _InOptConst_SegmPointf(float a) {return new MR.SegmPointf(a);}
    }

    /// \brief encodes a point inside a line segment using relative distance in [0,1]
    /// Generated from class `MR::SegmPointd`.
    /// This is the const half of the class.
    public class Const_SegmPointd : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SegmPointd>, System.IEquatable<double>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SegmPointd(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_Destroy", ExactSpelling = true)]
            extern static void __MR_SegmPointd_Destroy(_Underlying *_this);
            __MR_SegmPointd_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SegmPointd() {Dispose(false);}

        public static unsafe double Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_Get_eps", ExactSpelling = true)]
                extern static double *__MR_SegmPointd_Get_eps();
                return *__MR_SegmPointd_Get_eps();
            }
        }

        ///< a in [0,1], a=0 => point is in v0, a=1 => point is in v1
        public unsafe double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_Get_a", ExactSpelling = true)]
                extern static double *__MR_SegmPointd_Get_a(_Underlying *_this);
                return *__MR_SegmPointd_Get_a(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SegmPointd() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_DefaultConstruct();
            _UnderlyingPtr = __MR_SegmPointd_DefaultConstruct();
        }

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public unsafe Const_SegmPointd(MR.Const_SegmPointd _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_ConstructFromAnother(MR.SegmPointd._Underlying *_other);
            _UnderlyingPtr = __MR_SegmPointd_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public unsafe Const_SegmPointd(double a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_Construct", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_Construct(double a);
            _UnderlyingPtr = __MR_SegmPointd_Construct(a);
        }

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public static unsafe implicit operator Const_SegmPointd(double a) {return new(a);}

        /// Generated from conversion operator `MR::SegmPointd::operator double`.
        public static unsafe implicit operator double(MR.Const_SegmPointd _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_ConvertTo_double", ExactSpelling = true)]
            extern static double __MR_SegmPointd_ConvertTo_double(MR.Const_SegmPointd._Underlying *_this);
            return __MR_SegmPointd_ConvertTo_double(_this._UnderlyingPtr);
        }

        /// returns [0,1] if the point is in a vertex or -1 otherwise
        /// Generated from method `MR::SegmPointd::inVertex`.
        public unsafe int InVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_inVertex", ExactSpelling = true)]
            extern static int __MR_SegmPointd_inVertex(_Underlying *_this);
            return __MR_SegmPointd_inVertex(_UnderlyingPtr);
        }

        /// represents the same point relative to oppositely directed segment
        /// Generated from method `MR::SegmPointd::sym`.
        public unsafe MR.SegmPointd Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_sym", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_sym(_Underlying *_this);
            return new(__MR_SegmPointd_sym(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if two points have equal (a) representation
        /// Generated from method `MR::SegmPointd::operator==`.
        public static unsafe bool operator==(MR.Const_SegmPointd _this, MR.Const_SegmPointd rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SegmPointd", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SegmPointd(MR.Const_SegmPointd._Underlying *_this, MR.Const_SegmPointd._Underlying *rhs);
            return __MR_equal_MR_SegmPointd(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SegmPointd _this, MR.Const_SegmPointd rhs)
        {
            return !(_this == rhs);
        }

        /// Generated from method `MR::SegmPointd::operator==`.
        public static unsafe bool operator==(MR.Const_SegmPointd _this, double rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SegmPointd_double", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SegmPointd_double(MR.Const_SegmPointd._Underlying *_this, double rhs);
            return __MR_equal_MR_SegmPointd_double(_this._UnderlyingPtr, rhs) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SegmPointd _this, double rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_SegmPointd? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public bool Equals(double rhs)
        {
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SegmPointd)
                return this == (MR.Const_SegmPointd)other;
            if (other is double)
                return this == (double)other;
            return false;
        }
    }

    /// \brief encodes a point inside a line segment using relative distance in [0,1]
    /// Generated from class `MR::SegmPointd`.
    /// This is the non-const half of the class.
    public class SegmPointd : Const_SegmPointd
    {
        internal unsafe SegmPointd(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< a in [0,1], a=0 => point is in v0, a=1 => point is in v1
        public new unsafe ref double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_GetMutable_a", ExactSpelling = true)]
                extern static double *__MR_SegmPointd_GetMutable_a(_Underlying *_this);
                return ref *__MR_SegmPointd_GetMutable_a(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SegmPointd() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_DefaultConstruct();
            _UnderlyingPtr = __MR_SegmPointd_DefaultConstruct();
        }

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public unsafe SegmPointd(MR.Const_SegmPointd _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_ConstructFromAnother(MR.SegmPointd._Underlying *_other);
            _UnderlyingPtr = __MR_SegmPointd_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public unsafe SegmPointd(double a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_Construct", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_Construct(double a);
            _UnderlyingPtr = __MR_SegmPointd_Construct(a);
        }

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public static unsafe implicit operator SegmPointd(double a) {return new(a);}

        /// Generated from conversion operator `MR::SegmPointd::operator double &`.
        public unsafe ref double ConvertTo_DoubleRef()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_ConvertTo_double_ref", ExactSpelling = true)]
            extern static double *__MR_SegmPointd_ConvertTo_double_ref(_Underlying *_this);
            return ref *__MR_SegmPointd_ConvertTo_double_ref(_UnderlyingPtr);
        }

        /// Generated from method `MR::SegmPointd::operator=`.
        public unsafe MR.SegmPointd Assign(MR.Const_SegmPointd _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegmPointd_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SegmPointd._Underlying *__MR_SegmPointd_AssignFromAnother(_Underlying *_this, MR.SegmPointd._Underlying *_other);
            return new(__MR_SegmPointd_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SegmPointd` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SegmPointd`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SegmPointd`/`Const_SegmPointd` directly.
    public class _InOptMut_SegmPointd
    {
        public SegmPointd? Opt;

        public _InOptMut_SegmPointd() {}
        public _InOptMut_SegmPointd(SegmPointd value) {Opt = value;}
        public static implicit operator _InOptMut_SegmPointd(SegmPointd value) {return new(value);}
    }

    /// This is used for optional parameters of class `SegmPointd` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SegmPointd`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SegmPointd`/`Const_SegmPointd` to pass it to the function.
    public class _InOptConst_SegmPointd
    {
        public Const_SegmPointd? Opt;

        public _InOptConst_SegmPointd() {}
        public _InOptConst_SegmPointd(Const_SegmPointd value) {Opt = value;}
        public static implicit operator _InOptConst_SegmPointd(Const_SegmPointd value) {return new(value);}

        /// Generated from constructor `MR::SegmPointd::SegmPointd`.
        public static unsafe implicit operator _InOptConst_SegmPointd(double a) {return new MR.SegmPointd(a);}
    }
}
