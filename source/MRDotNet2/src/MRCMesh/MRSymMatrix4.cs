public static partial class MR
{
    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4b`.
    /// This is the const half of the class.
    public class Const_SymMatrix4b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SymMatrix4b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix4b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix4b_Destroy(_Underlying *_this);
            __MR_SymMatrix4b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix4b() {Dispose(false);}

        /// zero matrix by default
        public unsafe bool Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_xx", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_xy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_xz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_xw", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_xw(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_yy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_yz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_yw", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_yw(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_zz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_zw", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_zw(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_Get_ww", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_Get_ww(_Underlying *_this);
                return *__MR_SymMatrix4b_Get_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix4b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4b_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4b::SymMatrix4b`.
        public unsafe Const_SymMatrix4b(MR.Const_SymMatrix4b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_ConstructFromAnother(MR.SymMatrix4b._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4b_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4b::identity`.
        public static unsafe MR.SymMatrix4b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_identity", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_identity();
            return new(__MR_SymMatrix4b_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix4b::diagonal`.
        public static unsafe MR.SymMatrix4b Diagonal(bool diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_diagonal(byte diagVal);
            return new(__MR_SymMatrix4b_diagonal(diagVal ? (byte)1 : (byte)0), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix4b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_trace", ExactSpelling = true)]
            extern static byte __MR_SymMatrix4b_trace(_Underlying *_this);
            return __MR_SymMatrix4b_trace(_UnderlyingPtr) != 0;
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
        /// Generated from method `MR::SymMatrix4b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_normSq", ExactSpelling = true)]
            extern static byte __MR_SymMatrix4b_normSq(_Underlying *_this);
            return __MR_SymMatrix4b_normSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::SymMatrix4b::operator==`.
        public static unsafe bool operator==(MR.Const_SymMatrix4b _this, MR.Const_SymMatrix4b _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SymMatrix4b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SymMatrix4b(MR.Const_SymMatrix4b._Underlying *_this, MR.Const_SymMatrix4b._Underlying *_1);
            return __MR_equal_MR_SymMatrix4b(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SymMatrix4b _this, MR.Const_SymMatrix4b _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_SymMatrix4b? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SymMatrix4b)
                return this == (MR.Const_SymMatrix4b)other;
            return false;
        }
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4b`.
    /// This is the non-const half of the class.
    public class SymMatrix4b : Const_SymMatrix4b
    {
        internal unsafe SymMatrix4b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref bool Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_xx", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_xy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_xz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_xw", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_xw(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_yy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_yz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_yw", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_yw(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_zz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_zw", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_zw(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_GetMutable_ww", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix4b_GetMutable_ww(_Underlying *_this);
                return ref *__MR_SymMatrix4b_GetMutable_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix4b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4b_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4b::SymMatrix4b`.
        public unsafe SymMatrix4b(MR.Const_SymMatrix4b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_ConstructFromAnother(MR.SymMatrix4b._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4b_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4b::operator=`.
        public unsafe MR.SymMatrix4b Assign(MR.Const_SymMatrix4b _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_AssignFromAnother(_Underlying *_this, MR.SymMatrix4b._Underlying *_other);
            return new(__MR_SymMatrix4b_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4b::operator+=`.
        public unsafe MR.SymMatrix4b AddAssign(MR.Const_SymMatrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_add_assign(_Underlying *_this, MR.Const_SymMatrix4b._Underlying *b);
            return new(__MR_SymMatrix4b_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4b::operator-=`.
        public unsafe MR.SymMatrix4b SubAssign(MR.Const_SymMatrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_sub_assign(_Underlying *_this, MR.Const_SymMatrix4b._Underlying *b);
            return new(__MR_SymMatrix4b_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4b::operator*=`.
        public unsafe MR.SymMatrix4b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_mul_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix4b_mul_assign(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4b::operator/=`.
        public unsafe MR.SymMatrix4b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4b_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4b._Underlying *__MR_SymMatrix4b_div_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix4b_div_assign(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix4b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix4b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4b`/`Const_SymMatrix4b` directly.
    public class _InOptMut_SymMatrix4b
    {
        public SymMatrix4b? Opt;

        public _InOptMut_SymMatrix4b() {}
        public _InOptMut_SymMatrix4b(SymMatrix4b value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix4b(SymMatrix4b value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix4b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix4b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4b`/`Const_SymMatrix4b` to pass it to the function.
    public class _InOptConst_SymMatrix4b
    {
        public Const_SymMatrix4b? Opt;

        public _InOptConst_SymMatrix4b() {}
        public _InOptConst_SymMatrix4b(Const_SymMatrix4b value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix4b(Const_SymMatrix4b value) {return new(value);}
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4i`.
    /// This is the const half of the class.
    public class Const_SymMatrix4i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SymMatrix4i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix4i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix4i_Destroy(_Underlying *_this);
            __MR_SymMatrix4i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix4i() {Dispose(false);}

        /// zero matrix by default
        public unsafe int Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_xx", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_xy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_xz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_xw", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_xw(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_yy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_yz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_yw", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_yw(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_zz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_zw", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_zw(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_Get_ww", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_Get_ww(_Underlying *_this);
                return *__MR_SymMatrix4i_Get_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix4i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4i_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4i::SymMatrix4i`.
        public unsafe Const_SymMatrix4i(MR.Const_SymMatrix4i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_ConstructFromAnother(MR.SymMatrix4i._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4i_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4i::identity`.
        public static unsafe MR.SymMatrix4i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_identity", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_identity();
            return new(__MR_SymMatrix4i_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix4i::diagonal`.
        public static unsafe MR.SymMatrix4i Diagonal(int diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_diagonal(int diagVal);
            return new(__MR_SymMatrix4i_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix4i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_trace", ExactSpelling = true)]
            extern static int __MR_SymMatrix4i_trace(_Underlying *_this);
            return __MR_SymMatrix4i_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
        /// Generated from method `MR::SymMatrix4i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_normSq", ExactSpelling = true)]
            extern static int __MR_SymMatrix4i_normSq(_Underlying *_this);
            return __MR_SymMatrix4i_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4i::operator==`.
        public static unsafe bool operator==(MR.Const_SymMatrix4i _this, MR.Const_SymMatrix4i _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SymMatrix4i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SymMatrix4i(MR.Const_SymMatrix4i._Underlying *_this, MR.Const_SymMatrix4i._Underlying *_1);
            return __MR_equal_MR_SymMatrix4i(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SymMatrix4i _this, MR.Const_SymMatrix4i _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_SymMatrix4i? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SymMatrix4i)
                return this == (MR.Const_SymMatrix4i)other;
            return false;
        }
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4i`.
    /// This is the non-const half of the class.
    public class SymMatrix4i : Const_SymMatrix4i
    {
        internal unsafe SymMatrix4i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref int Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_xx", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_xy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_xz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_xw", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_xw(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_yy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_yz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_yw", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_yw(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_zz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_zw", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_zw(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_GetMutable_ww", ExactSpelling = true)]
                extern static int *__MR_SymMatrix4i_GetMutable_ww(_Underlying *_this);
                return ref *__MR_SymMatrix4i_GetMutable_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix4i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4i_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4i::SymMatrix4i`.
        public unsafe SymMatrix4i(MR.Const_SymMatrix4i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_ConstructFromAnother(MR.SymMatrix4i._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4i_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4i::operator=`.
        public unsafe MR.SymMatrix4i Assign(MR.Const_SymMatrix4i _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_AssignFromAnother(_Underlying *_this, MR.SymMatrix4i._Underlying *_other);
            return new(__MR_SymMatrix4i_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i::operator+=`.
        public unsafe MR.SymMatrix4i AddAssign(MR.Const_SymMatrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_add_assign(_Underlying *_this, MR.Const_SymMatrix4i._Underlying *b);
            return new(__MR_SymMatrix4i_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i::operator-=`.
        public unsafe MR.SymMatrix4i SubAssign(MR.Const_SymMatrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_sub_assign(_Underlying *_this, MR.Const_SymMatrix4i._Underlying *b);
            return new(__MR_SymMatrix4i_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i::operator*=`.
        public unsafe MR.SymMatrix4i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_mul_assign(_Underlying *_this, int b);
            return new(__MR_SymMatrix4i_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i::operator/=`.
        public unsafe MR.SymMatrix4i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i._Underlying *__MR_SymMatrix4i_div_assign(_Underlying *_this, int b);
            return new(__MR_SymMatrix4i_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix4i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix4i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4i`/`Const_SymMatrix4i` directly.
    public class _InOptMut_SymMatrix4i
    {
        public SymMatrix4i? Opt;

        public _InOptMut_SymMatrix4i() {}
        public _InOptMut_SymMatrix4i(SymMatrix4i value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix4i(SymMatrix4i value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix4i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix4i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4i`/`Const_SymMatrix4i` to pass it to the function.
    public class _InOptConst_SymMatrix4i
    {
        public Const_SymMatrix4i? Opt;

        public _InOptConst_SymMatrix4i() {}
        public _InOptConst_SymMatrix4i(Const_SymMatrix4i value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix4i(Const_SymMatrix4i value) {return new(value);}
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4i64`.
    /// This is the const half of the class.
    public class Const_SymMatrix4i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SymMatrix4i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix4i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix4i64_Destroy(_Underlying *_this);
            __MR_SymMatrix4i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix4i64() {Dispose(false);}

        /// zero matrix by default
        public unsafe long Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_xx", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_xy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_xz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_xw", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_xw(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_yy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_yz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_yw", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_yw(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_zz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_zw", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_zw(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_Get_ww", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_Get_ww(_Underlying *_this);
                return *__MR_SymMatrix4i64_Get_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix4i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4i64::SymMatrix4i64`.
        public unsafe Const_SymMatrix4i64(MR.Const_SymMatrix4i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_ConstructFromAnother(MR.SymMatrix4i64._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4i64_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4i64::identity`.
        public static unsafe MR.SymMatrix4i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_identity", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_identity();
            return new(__MR_SymMatrix4i64_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix4i64::diagonal`.
        public static unsafe MR.SymMatrix4i64 Diagonal(long diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_diagonal(long diagVal);
            return new(__MR_SymMatrix4i64_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix4i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_trace", ExactSpelling = true)]
            extern static long __MR_SymMatrix4i64_trace(_Underlying *_this);
            return __MR_SymMatrix4i64_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
        /// Generated from method `MR::SymMatrix4i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_normSq", ExactSpelling = true)]
            extern static long __MR_SymMatrix4i64_normSq(_Underlying *_this);
            return __MR_SymMatrix4i64_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4i64::operator==`.
        public static unsafe bool operator==(MR.Const_SymMatrix4i64 _this, MR.Const_SymMatrix4i64 _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SymMatrix4i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SymMatrix4i64(MR.Const_SymMatrix4i64._Underlying *_this, MR.Const_SymMatrix4i64._Underlying *_1);
            return __MR_equal_MR_SymMatrix4i64(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SymMatrix4i64 _this, MR.Const_SymMatrix4i64 _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_SymMatrix4i64? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SymMatrix4i64)
                return this == (MR.Const_SymMatrix4i64)other;
            return false;
        }
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4i64`.
    /// This is the non-const half of the class.
    public class SymMatrix4i64 : Const_SymMatrix4i64
    {
        internal unsafe SymMatrix4i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref long Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_xx", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_xy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_xz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_xw", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_xw(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_yy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_yz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_yw", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_yw(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_zz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_zw", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_zw(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_GetMutable_ww", ExactSpelling = true)]
                extern static long *__MR_SymMatrix4i64_GetMutable_ww(_Underlying *_this);
                return ref *__MR_SymMatrix4i64_GetMutable_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix4i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4i64::SymMatrix4i64`.
        public unsafe SymMatrix4i64(MR.Const_SymMatrix4i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_ConstructFromAnother(MR.SymMatrix4i64._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4i64_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4i64::operator=`.
        public unsafe MR.SymMatrix4i64 Assign(MR.Const_SymMatrix4i64 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_AssignFromAnother(_Underlying *_this, MR.SymMatrix4i64._Underlying *_other);
            return new(__MR_SymMatrix4i64_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i64::operator+=`.
        public unsafe MR.SymMatrix4i64 AddAssign(MR.Const_SymMatrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_add_assign(_Underlying *_this, MR.Const_SymMatrix4i64._Underlying *b);
            return new(__MR_SymMatrix4i64_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i64::operator-=`.
        public unsafe MR.SymMatrix4i64 SubAssign(MR.Const_SymMatrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_sub_assign(_Underlying *_this, MR.Const_SymMatrix4i64._Underlying *b);
            return new(__MR_SymMatrix4i64_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i64::operator*=`.
        public unsafe MR.SymMatrix4i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_mul_assign(_Underlying *_this, long b);
            return new(__MR_SymMatrix4i64_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4i64::operator/=`.
        public unsafe MR.SymMatrix4i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4i64_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4i64._Underlying *__MR_SymMatrix4i64_div_assign(_Underlying *_this, long b);
            return new(__MR_SymMatrix4i64_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix4i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix4i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4i64`/`Const_SymMatrix4i64` directly.
    public class _InOptMut_SymMatrix4i64
    {
        public SymMatrix4i64? Opt;

        public _InOptMut_SymMatrix4i64() {}
        public _InOptMut_SymMatrix4i64(SymMatrix4i64 value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix4i64(SymMatrix4i64 value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix4i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix4i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4i64`/`Const_SymMatrix4i64` to pass it to the function.
    public class _InOptConst_SymMatrix4i64
    {
        public Const_SymMatrix4i64? Opt;

        public _InOptConst_SymMatrix4i64() {}
        public _InOptConst_SymMatrix4i64(Const_SymMatrix4i64 value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix4i64(Const_SymMatrix4i64 value) {return new(value);}
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4f`.
    /// This is the const half of the class.
    public class Const_SymMatrix4f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SymMatrix4f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix4f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix4f_Destroy(_Underlying *_this);
            __MR_SymMatrix4f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix4f() {Dispose(false);}

        /// zero matrix by default
        public unsafe float Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_xx", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_xy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_xz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_xw", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_xw(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_yy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_yz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_yw", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_yw(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_zz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_zw", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_zw(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_Get_ww", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_Get_ww(_Underlying *_this);
                return *__MR_SymMatrix4f_Get_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix4f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4f_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4f::SymMatrix4f`.
        public unsafe Const_SymMatrix4f(MR.Const_SymMatrix4f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_ConstructFromAnother(MR.SymMatrix4f._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4f::identity`.
        public static unsafe MR.SymMatrix4f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_identity", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_identity();
            return new(__MR_SymMatrix4f_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix4f::diagonal`.
        public static unsafe MR.SymMatrix4f Diagonal(float diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_diagonal(float diagVal);
            return new(__MR_SymMatrix4f_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix4f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_trace", ExactSpelling = true)]
            extern static float __MR_SymMatrix4f_trace(_Underlying *_this);
            return __MR_SymMatrix4f_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
        /// Generated from method `MR::SymMatrix4f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_normSq", ExactSpelling = true)]
            extern static float __MR_SymMatrix4f_normSq(_Underlying *_this);
            return __MR_SymMatrix4f_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4f::operator==`.
        public static unsafe bool operator==(MR.Const_SymMatrix4f _this, MR.Const_SymMatrix4f _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SymMatrix4f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SymMatrix4f(MR.Const_SymMatrix4f._Underlying *_this, MR.Const_SymMatrix4f._Underlying *_1);
            return __MR_equal_MR_SymMatrix4f(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SymMatrix4f _this, MR.Const_SymMatrix4f _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_SymMatrix4f? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SymMatrix4f)
                return this == (MR.Const_SymMatrix4f)other;
            return false;
        }
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4f`.
    /// This is the non-const half of the class.
    public class SymMatrix4f : Const_SymMatrix4f
    {
        internal unsafe SymMatrix4f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref float Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_xx", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_xy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_xz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_xw", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_xw(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_yy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_yz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_yw", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_yw(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_zz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_zw", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_zw(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_GetMutable_ww", ExactSpelling = true)]
                extern static float *__MR_SymMatrix4f_GetMutable_ww(_Underlying *_this);
                return ref *__MR_SymMatrix4f_GetMutable_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix4f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4f_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4f::SymMatrix4f`.
        public unsafe SymMatrix4f(MR.Const_SymMatrix4f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_ConstructFromAnother(MR.SymMatrix4f._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4f::operator=`.
        public unsafe MR.SymMatrix4f Assign(MR.Const_SymMatrix4f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_AssignFromAnother(_Underlying *_this, MR.SymMatrix4f._Underlying *_other);
            return new(__MR_SymMatrix4f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4f::operator+=`.
        public unsafe MR.SymMatrix4f AddAssign(MR.Const_SymMatrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_add_assign(_Underlying *_this, MR.Const_SymMatrix4f._Underlying *b);
            return new(__MR_SymMatrix4f_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4f::operator-=`.
        public unsafe MR.SymMatrix4f SubAssign(MR.Const_SymMatrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_sub_assign(_Underlying *_this, MR.Const_SymMatrix4f._Underlying *b);
            return new(__MR_SymMatrix4f_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4f::operator*=`.
        public unsafe MR.SymMatrix4f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_mul_assign(_Underlying *_this, float b);
            return new(__MR_SymMatrix4f_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4f::operator/=`.
        public unsafe MR.SymMatrix4f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4f_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4f._Underlying *__MR_SymMatrix4f_div_assign(_Underlying *_this, float b);
            return new(__MR_SymMatrix4f_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix4f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix4f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4f`/`Const_SymMatrix4f` directly.
    public class _InOptMut_SymMatrix4f
    {
        public SymMatrix4f? Opt;

        public _InOptMut_SymMatrix4f() {}
        public _InOptMut_SymMatrix4f(SymMatrix4f value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix4f(SymMatrix4f value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix4f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix4f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4f`/`Const_SymMatrix4f` to pass it to the function.
    public class _InOptConst_SymMatrix4f
    {
        public Const_SymMatrix4f? Opt;

        public _InOptConst_SymMatrix4f() {}
        public _InOptConst_SymMatrix4f(Const_SymMatrix4f value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix4f(Const_SymMatrix4f value) {return new(value);}
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4d`.
    /// This is the const half of the class.
    public class Const_SymMatrix4d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SymMatrix4d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix4d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix4d_Destroy(_Underlying *_this);
            __MR_SymMatrix4d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix4d() {Dispose(false);}

        /// zero matrix by default
        public unsafe double Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_xx", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_xy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_xz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_xw", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_xw(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_yy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_yz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_yw", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_yw(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_zz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_zw", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_zw(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_Get_ww", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_Get_ww(_Underlying *_this);
                return *__MR_SymMatrix4d_Get_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix4d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4d_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4d::SymMatrix4d`.
        public unsafe Const_SymMatrix4d(MR.Const_SymMatrix4d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_ConstructFromAnother(MR.SymMatrix4d._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4d::identity`.
        public static unsafe MR.SymMatrix4d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_identity", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_identity();
            return new(__MR_SymMatrix4d_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix4d::diagonal`.
        public static unsafe MR.SymMatrix4d Diagonal(double diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_diagonal(double diagVal);
            return new(__MR_SymMatrix4d_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix4d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_trace", ExactSpelling = true)]
            extern static double __MR_SymMatrix4d_trace(_Underlying *_this);
            return __MR_SymMatrix4d_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
        /// Generated from method `MR::SymMatrix4d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_normSq", ExactSpelling = true)]
            extern static double __MR_SymMatrix4d_normSq(_Underlying *_this);
            return __MR_SymMatrix4d_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4d::operator==`.
        public static unsafe bool operator==(MR.Const_SymMatrix4d _this, MR.Const_SymMatrix4d _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SymMatrix4d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SymMatrix4d(MR.Const_SymMatrix4d._Underlying *_this, MR.Const_SymMatrix4d._Underlying *_1);
            return __MR_equal_MR_SymMatrix4d(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SymMatrix4d _this, MR.Const_SymMatrix4d _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_SymMatrix4d? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SymMatrix4d)
                return this == (MR.Const_SymMatrix4d)other;
            return false;
        }
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4d`.
    /// This is the non-const half of the class.
    public class SymMatrix4d : Const_SymMatrix4d
    {
        internal unsafe SymMatrix4d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref double Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_xx", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_xy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_xz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_xw", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_xw(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_yy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_yz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_yw", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_yw(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_zz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_zw", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_zw(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_GetMutable_ww", ExactSpelling = true)]
                extern static double *__MR_SymMatrix4d_GetMutable_ww(_Underlying *_this);
                return ref *__MR_SymMatrix4d_GetMutable_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix4d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4d_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4d::SymMatrix4d`.
        public unsafe SymMatrix4d(MR.Const_SymMatrix4d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_ConstructFromAnother(MR.SymMatrix4d._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4d::operator=`.
        public unsafe MR.SymMatrix4d Assign(MR.Const_SymMatrix4d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_AssignFromAnother(_Underlying *_this, MR.SymMatrix4d._Underlying *_other);
            return new(__MR_SymMatrix4d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4d::operator+=`.
        public unsafe MR.SymMatrix4d AddAssign(MR.Const_SymMatrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_add_assign(_Underlying *_this, MR.Const_SymMatrix4d._Underlying *b);
            return new(__MR_SymMatrix4d_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4d::operator-=`.
        public unsafe MR.SymMatrix4d SubAssign(MR.Const_SymMatrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_sub_assign(_Underlying *_this, MR.Const_SymMatrix4d._Underlying *b);
            return new(__MR_SymMatrix4d_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4d::operator*=`.
        public unsafe MR.SymMatrix4d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_mul_assign(_Underlying *_this, double b);
            return new(__MR_SymMatrix4d_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4d::operator/=`.
        public unsafe MR.SymMatrix4d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4d_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4d._Underlying *__MR_SymMatrix4d_div_assign(_Underlying *_this, double b);
            return new(__MR_SymMatrix4d_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix4d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix4d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4d`/`Const_SymMatrix4d` directly.
    public class _InOptMut_SymMatrix4d
    {
        public SymMatrix4d? Opt;

        public _InOptMut_SymMatrix4d() {}
        public _InOptMut_SymMatrix4d(SymMatrix4d value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix4d(SymMatrix4d value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix4d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix4d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4d`/`Const_SymMatrix4d` to pass it to the function.
    public class _InOptConst_SymMatrix4d
    {
        public Const_SymMatrix4d? Opt;

        public _InOptConst_SymMatrix4d() {}
        public _InOptConst_SymMatrix4d(Const_SymMatrix4d value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix4d(Const_SymMatrix4d value) {return new(value);}
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4<unsigned char>`.
    /// This is the const half of the class.
    public class Const_SymMatrix4_UnsignedChar : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_SymMatrix4_UnsignedChar>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix4_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix4_unsigned_char_Destroy(_Underlying *_this);
            __MR_SymMatrix4_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix4_UnsignedChar() {Dispose(false);}

        /// zero matrix by default
        public unsafe byte Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_xx", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_xy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_xz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_xw", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_xw(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_yy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_yz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_yw", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_yw(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_zz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_zw", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_zw(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_Get_ww", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_Get_ww(_Underlying *_this);
                return *__MR_SymMatrix4_unsigned_char_Get_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix4_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4<unsigned char>::SymMatrix4`.
        public unsafe Const_SymMatrix4_UnsignedChar(MR.Const_SymMatrix4_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_ConstructFromAnother(MR.SymMatrix4_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::identity`.
        public static unsafe MR.SymMatrix4_UnsignedChar Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_identity", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_identity();
            return new(__MR_SymMatrix4_unsigned_char_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::diagonal`.
        public static unsafe MR.SymMatrix4_UnsignedChar Diagonal(byte diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_diagonal(byte diagVal);
            return new(__MR_SymMatrix4_unsigned_char_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix4<unsigned char>::trace`.
        public unsafe byte Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_trace", ExactSpelling = true)]
            extern static byte __MR_SymMatrix4_unsigned_char_trace(_Underlying *_this);
            return __MR_SymMatrix4_unsigned_char_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 16 squared elements
        /// Generated from method `MR::SymMatrix4<unsigned char>::normSq`.
        public unsafe byte NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_normSq", ExactSpelling = true)]
            extern static byte __MR_SymMatrix4_unsigned_char_normSq(_Underlying *_this);
            return __MR_SymMatrix4_unsigned_char_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::operator==`.
        public static unsafe bool operator==(MR.Const_SymMatrix4_UnsignedChar _this, MR.Const_SymMatrix4_UnsignedChar _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SymMatrix4_unsigned_char", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SymMatrix4_unsigned_char(MR.Const_SymMatrix4_UnsignedChar._Underlying *_this, MR.Const_SymMatrix4_UnsignedChar._Underlying *_1);
            return __MR_equal_MR_SymMatrix4_unsigned_char(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SymMatrix4_UnsignedChar _this, MR.Const_SymMatrix4_UnsignedChar _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_SymMatrix4_UnsignedChar? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_SymMatrix4_UnsignedChar)
                return this == (MR.Const_SymMatrix4_UnsignedChar)other;
            return false;
        }
    }

    /// symmetric 4x4 matrix
    /// Generated from class `MR::SymMatrix4<unsigned char>`.
    /// This is the non-const half of the class.
    public class SymMatrix4_UnsignedChar : Const_SymMatrix4_UnsignedChar
    {
        internal unsafe SymMatrix4_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref byte Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_xx", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_xy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_xz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Xw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_xw", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_xw(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_xw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_yy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_yz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Yw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_yw", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_yw(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_yw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_zz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Zw
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_zw", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_zw(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_zw(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Ww
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_GetMutable_ww", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix4_unsigned_char_GetMutable_ww(_Underlying *_this);
                return ref *__MR_SymMatrix4_unsigned_char_GetMutable_ww(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix4_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix4<unsigned char>::SymMatrix4`.
        public unsafe SymMatrix4_UnsignedChar(MR.Const_SymMatrix4_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_ConstructFromAnother(MR.SymMatrix4_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix4_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::operator=`.
        public unsafe MR.SymMatrix4_UnsignedChar Assign(MR.Const_SymMatrix4_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_AssignFromAnother(_Underlying *_this, MR.SymMatrix4_UnsignedChar._Underlying *_other);
            return new(__MR_SymMatrix4_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::operator+=`.
        public unsafe MR.SymMatrix4_UnsignedChar AddAssign(MR.Const_SymMatrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_add_assign(_Underlying *_this, MR.Const_SymMatrix4_UnsignedChar._Underlying *b);
            return new(__MR_SymMatrix4_unsigned_char_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::operator-=`.
        public unsafe MR.SymMatrix4_UnsignedChar SubAssign(MR.Const_SymMatrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_sub_assign(_Underlying *_this, MR.Const_SymMatrix4_UnsignedChar._Underlying *b);
            return new(__MR_SymMatrix4_unsigned_char_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::operator*=`.
        public unsafe MR.SymMatrix4_UnsignedChar MulAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_mul_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix4_unsigned_char_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix4<unsigned char>::operator/=`.
        public unsafe MR.SymMatrix4_UnsignedChar DivAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix4_unsigned_char_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix4_UnsignedChar._Underlying *__MR_SymMatrix4_unsigned_char_div_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix4_unsigned_char_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix4_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix4_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4_UnsignedChar`/`Const_SymMatrix4_UnsignedChar` directly.
    public class _InOptMut_SymMatrix4_UnsignedChar
    {
        public SymMatrix4_UnsignedChar? Opt;

        public _InOptMut_SymMatrix4_UnsignedChar() {}
        public _InOptMut_SymMatrix4_UnsignedChar(SymMatrix4_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix4_UnsignedChar(SymMatrix4_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix4_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix4_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix4_UnsignedChar`/`Const_SymMatrix4_UnsignedChar` to pass it to the function.
    public class _InOptConst_SymMatrix4_UnsignedChar
    {
        public Const_SymMatrix4_UnsignedChar? Opt;

        public _InOptConst_SymMatrix4_UnsignedChar() {}
        public _InOptConst_SymMatrix4_UnsignedChar(Const_SymMatrix4_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix4_UnsignedChar(Const_SymMatrix4_UnsignedChar value) {return new(value);}
    }
}
