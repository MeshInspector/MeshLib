public static partial class MR
{
    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3b`.
    /// This is the const half of the class.
    public class Const_SymMatrix3b : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix3b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix3b_Destroy(_Underlying *_this);
            __MR_SymMatrix3b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix3b() {Dispose(false);}

        /// zero matrix by default
        public unsafe bool Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Get_xx", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix3b_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Get_xy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix3b_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Get_xz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix3b_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Get_yy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix3b_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Get_yz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix3b_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_Get_zz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix3b_Get_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix3b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3b_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3b::SymMatrix3b`.
        public unsafe Const_SymMatrix3b(MR.Const_SymMatrix3b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_ConstructFromAnother(MR.SymMatrix3b._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3b_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3b::identity`.
        public static unsafe MR.SymMatrix3b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_identity", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_identity();
            return new(__MR_SymMatrix3b_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix3b::diagonal`.
        public static unsafe MR.SymMatrix3b Diagonal(bool diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_diagonal(byte diagVal);
            return new(__MR_SymMatrix3b_diagonal(diagVal ? (byte)1 : (byte)0), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix3b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_trace", ExactSpelling = true)]
            extern static byte __MR_SymMatrix3b_trace(_Underlying *_this);
            return __MR_SymMatrix3b_trace(_UnderlyingPtr) != 0;
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
        /// Generated from method `MR::SymMatrix3b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_normSq", ExactSpelling = true)]
            extern static byte __MR_SymMatrix3b_normSq(_Underlying *_this);
            return __MR_SymMatrix3b_normSq(_UnderlyingPtr) != 0;
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix3b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_det", ExactSpelling = true)]
            extern static byte __MR_SymMatrix3b_det(_Underlying *_this);
            return __MR_SymMatrix3b_det(_UnderlyingPtr) != 0;
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix3b::inverse`.
        public unsafe MR.SymMatrix3b Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix3b_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix3b::inverse`.
        public unsafe MR.SymMatrix3b Inverse(bool det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_inverse_1(_Underlying *_this, byte det);
            return new(__MR_SymMatrix3b_inverse_1(_UnderlyingPtr, det ? (byte)1 : (byte)0), is_owning: true);
        }
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3b`.
    /// This is the non-const half of the class.
    public class SymMatrix3b : Const_SymMatrix3b
    {
        internal unsafe SymMatrix3b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref bool Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_GetMutable_xx", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix3b_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_GetMutable_xy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix3b_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_GetMutable_xz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix3b_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_GetMutable_yy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix3b_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_GetMutable_yz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix3b_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_GetMutable_zz", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix3b_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix3b_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix3b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3b_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3b::SymMatrix3b`.
        public unsafe SymMatrix3b(MR.Const_SymMatrix3b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_ConstructFromAnother(MR.SymMatrix3b._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3b_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3b::operator=`.
        public unsafe MR.SymMatrix3b Assign(MR.Const_SymMatrix3b _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_AssignFromAnother(_Underlying *_this, MR.SymMatrix3b._Underlying *_other);
            return new(__MR_SymMatrix3b_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3b::operator+=`.
        public unsafe MR.SymMatrix3b AddAssign(MR.Const_SymMatrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_add_assign(_Underlying *_this, MR.Const_SymMatrix3b._Underlying *b);
            return new(__MR_SymMatrix3b_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3b::operator-=`.
        public unsafe MR.SymMatrix3b SubAssign(MR.Const_SymMatrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_sub_assign(_Underlying *_this, MR.Const_SymMatrix3b._Underlying *b);
            return new(__MR_SymMatrix3b_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3b::operator*=`.
        public unsafe MR.SymMatrix3b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_mul_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix3b_mul_assign(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3b::operator/=`.
        public unsafe MR.SymMatrix3b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3b_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3b._Underlying *__MR_SymMatrix3b_div_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix3b_div_assign(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix3b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix3b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3b`/`Const_SymMatrix3b` directly.
    public class _InOptMut_SymMatrix3b
    {
        public SymMatrix3b? Opt;

        public _InOptMut_SymMatrix3b() {}
        public _InOptMut_SymMatrix3b(SymMatrix3b value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix3b(SymMatrix3b value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix3b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix3b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3b`/`Const_SymMatrix3b` to pass it to the function.
    public class _InOptConst_SymMatrix3b
    {
        public Const_SymMatrix3b? Opt;

        public _InOptConst_SymMatrix3b() {}
        public _InOptConst_SymMatrix3b(Const_SymMatrix3b value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix3b(Const_SymMatrix3b value) {return new(value);}
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3i`.
    /// This is the const half of the class.
    public class Const_SymMatrix3i : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix3i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix3i_Destroy(_Underlying *_this);
            __MR_SymMatrix3i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix3i() {Dispose(false);}

        /// zero matrix by default
        public unsafe int Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Get_xx", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix3i_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Get_xy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix3i_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Get_xz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix3i_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Get_yy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix3i_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Get_yz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix3i_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_Get_zz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix3i_Get_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3i_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3i::SymMatrix3i`.
        public unsafe Const_SymMatrix3i(MR.Const_SymMatrix3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_ConstructFromAnother(MR.SymMatrix3i._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3i_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3i::identity`.
        public static unsafe MR.SymMatrix3i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_identity", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_identity();
            return new(__MR_SymMatrix3i_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix3i::diagonal`.
        public static unsafe MR.SymMatrix3i Diagonal(int diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_diagonal(int diagVal);
            return new(__MR_SymMatrix3i_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix3i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_trace", ExactSpelling = true)]
            extern static int __MR_SymMatrix3i_trace(_Underlying *_this);
            return __MR_SymMatrix3i_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
        /// Generated from method `MR::SymMatrix3i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_normSq", ExactSpelling = true)]
            extern static int __MR_SymMatrix3i_normSq(_Underlying *_this);
            return __MR_SymMatrix3i_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix3i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_det", ExactSpelling = true)]
            extern static int __MR_SymMatrix3i_det(_Underlying *_this);
            return __MR_SymMatrix3i_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix3i::inverse`.
        public unsafe MR.SymMatrix3i Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix3i_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix3i::inverse`.
        public unsafe MR.SymMatrix3i Inverse(int det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_inverse_1(_Underlying *_this, int det);
            return new(__MR_SymMatrix3i_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3i`.
    /// This is the non-const half of the class.
    public class SymMatrix3i : Const_SymMatrix3i
    {
        internal unsafe SymMatrix3i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref int Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_GetMutable_xx", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix3i_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_GetMutable_xy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix3i_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_GetMutable_xz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix3i_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_GetMutable_yy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix3i_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_GetMutable_yz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix3i_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_GetMutable_zz", ExactSpelling = true)]
                extern static int *__MR_SymMatrix3i_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix3i_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3i_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3i::SymMatrix3i`.
        public unsafe SymMatrix3i(MR.Const_SymMatrix3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_ConstructFromAnother(MR.SymMatrix3i._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3i_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3i::operator=`.
        public unsafe MR.SymMatrix3i Assign(MR.Const_SymMatrix3i _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_AssignFromAnother(_Underlying *_this, MR.SymMatrix3i._Underlying *_other);
            return new(__MR_SymMatrix3i_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i::operator+=`.
        public unsafe MR.SymMatrix3i AddAssign(MR.Const_SymMatrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_add_assign(_Underlying *_this, MR.Const_SymMatrix3i._Underlying *b);
            return new(__MR_SymMatrix3i_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i::operator-=`.
        public unsafe MR.SymMatrix3i SubAssign(MR.Const_SymMatrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_sub_assign(_Underlying *_this, MR.Const_SymMatrix3i._Underlying *b);
            return new(__MR_SymMatrix3i_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i::operator*=`.
        public unsafe MR.SymMatrix3i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_mul_assign(_Underlying *_this, int b);
            return new(__MR_SymMatrix3i_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i::operator/=`.
        public unsafe MR.SymMatrix3i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i._Underlying *__MR_SymMatrix3i_div_assign(_Underlying *_this, int b);
            return new(__MR_SymMatrix3i_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix3i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3i`/`Const_SymMatrix3i` directly.
    public class _InOptMut_SymMatrix3i
    {
        public SymMatrix3i? Opt;

        public _InOptMut_SymMatrix3i() {}
        public _InOptMut_SymMatrix3i(SymMatrix3i value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix3i(SymMatrix3i value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix3i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3i`/`Const_SymMatrix3i` to pass it to the function.
    public class _InOptConst_SymMatrix3i
    {
        public Const_SymMatrix3i? Opt;

        public _InOptConst_SymMatrix3i() {}
        public _InOptConst_SymMatrix3i(Const_SymMatrix3i value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix3i(Const_SymMatrix3i value) {return new(value);}
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3i64`.
    /// This is the const half of the class.
    public class Const_SymMatrix3i64 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix3i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix3i64_Destroy(_Underlying *_this);
            __MR_SymMatrix3i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix3i64() {Dispose(false);}

        /// zero matrix by default
        public unsafe long Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Get_xx", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix3i64_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Get_xy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix3i64_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Get_xz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix3i64_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Get_yy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix3i64_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Get_yz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix3i64_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_Get_zz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix3i64_Get_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3i64::SymMatrix3i64`.
        public unsafe Const_SymMatrix3i64(MR.Const_SymMatrix3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_ConstructFromAnother(MR.SymMatrix3i64._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3i64_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3i64::identity`.
        public static unsafe MR.SymMatrix3i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_identity", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_identity();
            return new(__MR_SymMatrix3i64_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix3i64::diagonal`.
        public static unsafe MR.SymMatrix3i64 Diagonal(long diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_diagonal(long diagVal);
            return new(__MR_SymMatrix3i64_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix3i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_trace", ExactSpelling = true)]
            extern static long __MR_SymMatrix3i64_trace(_Underlying *_this);
            return __MR_SymMatrix3i64_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
        /// Generated from method `MR::SymMatrix3i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_normSq", ExactSpelling = true)]
            extern static long __MR_SymMatrix3i64_normSq(_Underlying *_this);
            return __MR_SymMatrix3i64_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix3i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_det", ExactSpelling = true)]
            extern static long __MR_SymMatrix3i64_det(_Underlying *_this);
            return __MR_SymMatrix3i64_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix3i64::inverse`.
        public unsafe MR.SymMatrix3i64 Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix3i64_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix3i64::inverse`.
        public unsafe MR.SymMatrix3i64 Inverse(long det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_inverse_1(_Underlying *_this, long det);
            return new(__MR_SymMatrix3i64_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3i64`.
    /// This is the non-const half of the class.
    public class SymMatrix3i64 : Const_SymMatrix3i64
    {
        internal unsafe SymMatrix3i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref long Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_GetMutable_xx", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix3i64_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_GetMutable_xy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix3i64_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_GetMutable_xz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix3i64_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_GetMutable_yy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix3i64_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_GetMutable_yz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix3i64_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_GetMutable_zz", ExactSpelling = true)]
                extern static long *__MR_SymMatrix3i64_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix3i64_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3i64::SymMatrix3i64`.
        public unsafe SymMatrix3i64(MR.Const_SymMatrix3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_ConstructFromAnother(MR.SymMatrix3i64._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3i64_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3i64::operator=`.
        public unsafe MR.SymMatrix3i64 Assign(MR.Const_SymMatrix3i64 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_AssignFromAnother(_Underlying *_this, MR.SymMatrix3i64._Underlying *_other);
            return new(__MR_SymMatrix3i64_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i64::operator+=`.
        public unsafe MR.SymMatrix3i64 AddAssign(MR.Const_SymMatrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_add_assign(_Underlying *_this, MR.Const_SymMatrix3i64._Underlying *b);
            return new(__MR_SymMatrix3i64_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i64::operator-=`.
        public unsafe MR.SymMatrix3i64 SubAssign(MR.Const_SymMatrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_sub_assign(_Underlying *_this, MR.Const_SymMatrix3i64._Underlying *b);
            return new(__MR_SymMatrix3i64_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i64::operator*=`.
        public unsafe MR.SymMatrix3i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_mul_assign(_Underlying *_this, long b);
            return new(__MR_SymMatrix3i64_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3i64::operator/=`.
        public unsafe MR.SymMatrix3i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3i64_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3i64._Underlying *__MR_SymMatrix3i64_div_assign(_Underlying *_this, long b);
            return new(__MR_SymMatrix3i64_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix3i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3i64`/`Const_SymMatrix3i64` directly.
    public class _InOptMut_SymMatrix3i64
    {
        public SymMatrix3i64? Opt;

        public _InOptMut_SymMatrix3i64() {}
        public _InOptMut_SymMatrix3i64(SymMatrix3i64 value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix3i64(SymMatrix3i64 value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix3i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3i64`/`Const_SymMatrix3i64` to pass it to the function.
    public class _InOptConst_SymMatrix3i64
    {
        public Const_SymMatrix3i64? Opt;

        public _InOptConst_SymMatrix3i64() {}
        public _InOptConst_SymMatrix3i64(Const_SymMatrix3i64 value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix3i64(Const_SymMatrix3i64 value) {return new(value);}
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3f`.
    /// This is the const half of the class.
    public class Const_SymMatrix3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix3f_Destroy(_Underlying *_this);
            __MR_SymMatrix3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix3f() {Dispose(false);}

        /// zero matrix by default
        public unsafe float Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Get_xx", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix3f_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Get_xy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix3f_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Get_xz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix3f_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Get_yy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix3f_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Get_yz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix3f_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_Get_zz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix3f_Get_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3f::SymMatrix3f`.
        public unsafe Const_SymMatrix3f(MR.Const_SymMatrix3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_ConstructFromAnother(MR.SymMatrix3f._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3f::identity`.
        public static unsafe MR.SymMatrix3f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_identity", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_identity();
            return new(__MR_SymMatrix3f_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix3f::diagonal`.
        public static unsafe MR.SymMatrix3f Diagonal(float diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_diagonal(float diagVal);
            return new(__MR_SymMatrix3f_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix3f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_trace", ExactSpelling = true)]
            extern static float __MR_SymMatrix3f_trace(_Underlying *_this);
            return __MR_SymMatrix3f_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
        /// Generated from method `MR::SymMatrix3f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_normSq", ExactSpelling = true)]
            extern static float __MR_SymMatrix3f_normSq(_Underlying *_this);
            return __MR_SymMatrix3f_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix3f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_det", ExactSpelling = true)]
            extern static float __MR_SymMatrix3f_det(_Underlying *_this);
            return __MR_SymMatrix3f_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix3f::inverse`.
        public unsafe MR.SymMatrix3f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix3f_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix3f::inverse`.
        public unsafe MR.SymMatrix3f Inverse(float det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_inverse_1(_Underlying *_this, float det);
            return new(__MR_SymMatrix3f_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }

        /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
        /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
        /// M*V^T = V^T*L; M = V^T*L*V
        /// Generated from method `MR::SymMatrix3f::eigens`.
        public unsafe MR.Vector3f Eigens(MR.Mut_Matrix3f? eigenvectors = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_eigens", ExactSpelling = true)]
            extern static MR.Vector3f __MR_SymMatrix3f_eigens(_Underlying *_this, MR.Mut_Matrix3f._Underlying *eigenvectors);
            return __MR_SymMatrix3f_eigens(_UnderlyingPtr, eigenvectors is not null ? eigenvectors._UnderlyingPtr : null);
        }

        /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
        /// Generated from method `MR::SymMatrix3f::eigenvector`.
        public unsafe MR.Vector3f Eigenvector(float eigenvalue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_eigenvector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_SymMatrix3f_eigenvector(_Underlying *_this, float eigenvalue);
            return __MR_SymMatrix3f_eigenvector(_UnderlyingPtr, eigenvalue);
        }

        /// for not-degenerate matrix returns just inverse matrix, otherwise
        /// returns degenerate matrix, which performs inversion on not-kernel subspace;
        /// \param tol relative epsilon-tolerance for too small number detection
        /// \param rank optional output for this matrix rank according to given tolerance
        /// \param space rank=1: unit direction of solution line, rank=2: unit normal to solution plane, rank=3: zero vector
        /// Generated from method `MR::SymMatrix3f::pseudoinverse`.
        /// Parameter `tol` defaults to `std::numeric_limits<float>::epsilon()`.
        public unsafe MR.SymMatrix3f Pseudoinverse(float? tol = null, MR.Misc.InOut<int>? rank = null, MR.Mut_Vector3f? space = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_pseudoinverse", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_pseudoinverse(_Underlying *_this, float *tol, int *rank, MR.Mut_Vector3f._Underlying *space);
            float __deref_tol = tol.GetValueOrDefault();
            int __value_rank = rank is not null ? rank.Value : default(int);
            var __ret = __MR_SymMatrix3f_pseudoinverse(_UnderlyingPtr, tol.HasValue ? &__deref_tol : null, rank is not null ? &__value_rank : null, space is not null ? space._UnderlyingPtr : null);
            if (rank is not null) rank.Value = __value_rank;
            return new(__ret, is_owning: true);
        }
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3f`.
    /// This is the non-const half of the class.
    public class SymMatrix3f : Const_SymMatrix3f
    {
        internal unsafe SymMatrix3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref float Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_GetMutable_xx", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix3f_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_GetMutable_xy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix3f_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_GetMutable_xz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix3f_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_GetMutable_yy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix3f_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_GetMutable_yz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix3f_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_GetMutable_zz", ExactSpelling = true)]
                extern static float *__MR_SymMatrix3f_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix3f_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3f::SymMatrix3f`.
        public unsafe SymMatrix3f(MR.Const_SymMatrix3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_ConstructFromAnother(MR.SymMatrix3f._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3f::operator=`.
        public unsafe MR.SymMatrix3f Assign(MR.Const_SymMatrix3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_AssignFromAnother(_Underlying *_this, MR.SymMatrix3f._Underlying *_other);
            return new(__MR_SymMatrix3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3f::operator+=`.
        public unsafe MR.SymMatrix3f AddAssign(MR.Const_SymMatrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_add_assign(_Underlying *_this, MR.Const_SymMatrix3f._Underlying *b);
            return new(__MR_SymMatrix3f_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3f::operator-=`.
        public unsafe MR.SymMatrix3f SubAssign(MR.Const_SymMatrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_sub_assign(_Underlying *_this, MR.Const_SymMatrix3f._Underlying *b);
            return new(__MR_SymMatrix3f_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3f::operator*=`.
        public unsafe MR.SymMatrix3f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_mul_assign(_Underlying *_this, float b);
            return new(__MR_SymMatrix3f_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3f::operator/=`.
        public unsafe MR.SymMatrix3f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3f_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3f._Underlying *__MR_SymMatrix3f_div_assign(_Underlying *_this, float b);
            return new(__MR_SymMatrix3f_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3f`/`Const_SymMatrix3f` directly.
    public class _InOptMut_SymMatrix3f
    {
        public SymMatrix3f? Opt;

        public _InOptMut_SymMatrix3f() {}
        public _InOptMut_SymMatrix3f(SymMatrix3f value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix3f(SymMatrix3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3f`/`Const_SymMatrix3f` to pass it to the function.
    public class _InOptConst_SymMatrix3f
    {
        public Const_SymMatrix3f? Opt;

        public _InOptConst_SymMatrix3f() {}
        public _InOptConst_SymMatrix3f(Const_SymMatrix3f value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix3f(Const_SymMatrix3f value) {return new(value);}
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3d`.
    /// This is the const half of the class.
    public class Const_SymMatrix3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix3d_Destroy(_Underlying *_this);
            __MR_SymMatrix3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix3d() {Dispose(false);}

        /// zero matrix by default
        public unsafe double Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Get_xx", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix3d_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Get_xy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix3d_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Get_xz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix3d_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Get_yy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix3d_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Get_yz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix3d_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_Get_zz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix3d_Get_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3d::SymMatrix3d`.
        public unsafe Const_SymMatrix3d(MR.Const_SymMatrix3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_ConstructFromAnother(MR.SymMatrix3d._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3d::identity`.
        public static unsafe MR.SymMatrix3d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_identity", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_identity();
            return new(__MR_SymMatrix3d_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix3d::diagonal`.
        public static unsafe MR.SymMatrix3d Diagonal(double diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_diagonal(double diagVal);
            return new(__MR_SymMatrix3d_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix3d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_trace", ExactSpelling = true)]
            extern static double __MR_SymMatrix3d_trace(_Underlying *_this);
            return __MR_SymMatrix3d_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
        /// Generated from method `MR::SymMatrix3d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_normSq", ExactSpelling = true)]
            extern static double __MR_SymMatrix3d_normSq(_Underlying *_this);
            return __MR_SymMatrix3d_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix3d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_det", ExactSpelling = true)]
            extern static double __MR_SymMatrix3d_det(_Underlying *_this);
            return __MR_SymMatrix3d_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix3d::inverse`.
        public unsafe MR.SymMatrix3d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix3d_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix3d::inverse`.
        public unsafe MR.SymMatrix3d Inverse(double det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_inverse_1(_Underlying *_this, double det);
            return new(__MR_SymMatrix3d_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }

        /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
        /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
        /// M*V^T = V^T*L; M = V^T*L*V
        /// Generated from method `MR::SymMatrix3d::eigens`.
        public unsafe MR.Vector3d Eigens(MR.Mut_Matrix3d? eigenvectors = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_eigens", ExactSpelling = true)]
            extern static MR.Vector3d __MR_SymMatrix3d_eigens(_Underlying *_this, MR.Mut_Matrix3d._Underlying *eigenvectors);
            return __MR_SymMatrix3d_eigens(_UnderlyingPtr, eigenvectors is not null ? eigenvectors._UnderlyingPtr : null);
        }

        /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
        /// Generated from method `MR::SymMatrix3d::eigenvector`.
        public unsafe MR.Vector3d Eigenvector(double eigenvalue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_eigenvector", ExactSpelling = true)]
            extern static MR.Vector3d __MR_SymMatrix3d_eigenvector(_Underlying *_this, double eigenvalue);
            return __MR_SymMatrix3d_eigenvector(_UnderlyingPtr, eigenvalue);
        }

        /// for not-degenerate matrix returns just inverse matrix, otherwise
        /// returns degenerate matrix, which performs inversion on not-kernel subspace;
        /// \param tol relative epsilon-tolerance for too small number detection
        /// \param rank optional output for this matrix rank according to given tolerance
        /// \param space rank=1: unit direction of solution line, rank=2: unit normal to solution plane, rank=3: zero vector
        /// Generated from method `MR::SymMatrix3d::pseudoinverse`.
        /// Parameter `tol` defaults to `std::numeric_limits<double>::epsilon()`.
        public unsafe MR.SymMatrix3d Pseudoinverse(double? tol = null, MR.Misc.InOut<int>? rank = null, MR.Mut_Vector3d? space = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_pseudoinverse", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_pseudoinverse(_Underlying *_this, double *tol, int *rank, MR.Mut_Vector3d._Underlying *space);
            double __deref_tol = tol.GetValueOrDefault();
            int __value_rank = rank is not null ? rank.Value : default(int);
            var __ret = __MR_SymMatrix3d_pseudoinverse(_UnderlyingPtr, tol.HasValue ? &__deref_tol : null, rank is not null ? &__value_rank : null, space is not null ? space._UnderlyingPtr : null);
            if (rank is not null) rank.Value = __value_rank;
            return new(__ret, is_owning: true);
        }
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3d`.
    /// This is the non-const half of the class.
    public class SymMatrix3d : Const_SymMatrix3d
    {
        internal unsafe SymMatrix3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref double Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_GetMutable_xx", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix3d_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_GetMutable_xy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix3d_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_GetMutable_xz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix3d_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_GetMutable_yy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix3d_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_GetMutable_yz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix3d_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_GetMutable_zz", ExactSpelling = true)]
                extern static double *__MR_SymMatrix3d_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix3d_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3d::SymMatrix3d`.
        public unsafe SymMatrix3d(MR.Const_SymMatrix3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_ConstructFromAnother(MR.SymMatrix3d._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3d::operator=`.
        public unsafe MR.SymMatrix3d Assign(MR.Const_SymMatrix3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_AssignFromAnother(_Underlying *_this, MR.SymMatrix3d._Underlying *_other);
            return new(__MR_SymMatrix3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3d::operator+=`.
        public unsafe MR.SymMatrix3d AddAssign(MR.Const_SymMatrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_add_assign(_Underlying *_this, MR.Const_SymMatrix3d._Underlying *b);
            return new(__MR_SymMatrix3d_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3d::operator-=`.
        public unsafe MR.SymMatrix3d SubAssign(MR.Const_SymMatrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_sub_assign(_Underlying *_this, MR.Const_SymMatrix3d._Underlying *b);
            return new(__MR_SymMatrix3d_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3d::operator*=`.
        public unsafe MR.SymMatrix3d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_mul_assign(_Underlying *_this, double b);
            return new(__MR_SymMatrix3d_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3d::operator/=`.
        public unsafe MR.SymMatrix3d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3d_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3d._Underlying *__MR_SymMatrix3d_div_assign(_Underlying *_this, double b);
            return new(__MR_SymMatrix3d_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3d`/`Const_SymMatrix3d` directly.
    public class _InOptMut_SymMatrix3d
    {
        public SymMatrix3d? Opt;

        public _InOptMut_SymMatrix3d() {}
        public _InOptMut_SymMatrix3d(SymMatrix3d value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix3d(SymMatrix3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3d`/`Const_SymMatrix3d` to pass it to the function.
    public class _InOptConst_SymMatrix3d
    {
        public Const_SymMatrix3d? Opt;

        public _InOptConst_SymMatrix3d() {}
        public _InOptConst_SymMatrix3d(Const_SymMatrix3d value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix3d(Const_SymMatrix3d value) {return new(value);}
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3<unsigned char>`.
    /// This is the const half of the class.
    public class Const_SymMatrix3_UnsignedChar : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix3_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix3_unsigned_char_Destroy(_Underlying *_this);
            __MR_SymMatrix3_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix3_UnsignedChar() {Dispose(false);}

        /// zero matrix by default
        public unsafe byte Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Get_xx", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix3_unsigned_char_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Get_xy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix3_unsigned_char_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Get_xz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_Get_xz(_Underlying *_this);
                return *__MR_SymMatrix3_unsigned_char_Get_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Get_yy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix3_unsigned_char_Get_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Get_yz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_Get_yz(_Underlying *_this);
                return *__MR_SymMatrix3_unsigned_char_Get_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe byte Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_Get_zz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_Get_zz(_Underlying *_this);
                return *__MR_SymMatrix3_unsigned_char_Get_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix3_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3<unsigned char>::SymMatrix3`.
        public unsafe Const_SymMatrix3_UnsignedChar(MR.Const_SymMatrix3_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_ConstructFromAnother(MR.SymMatrix3_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::identity`.
        public static unsafe MR.SymMatrix3_UnsignedChar Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_identity", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_identity();
            return new(__MR_SymMatrix3_unsigned_char_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::diagonal`.
        public static unsafe MR.SymMatrix3_UnsignedChar Diagonal(byte diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_diagonal(byte diagVal);
            return new(__MR_SymMatrix3_unsigned_char_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix3<unsigned char>::trace`.
        public unsafe byte Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_trace", ExactSpelling = true)]
            extern static byte __MR_SymMatrix3_unsigned_char_trace(_Underlying *_this);
            return __MR_SymMatrix3_unsigned_char_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 9 squared elements
        /// Generated from method `MR::SymMatrix3<unsigned char>::normSq`.
        public unsafe byte NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_normSq", ExactSpelling = true)]
            extern static byte __MR_SymMatrix3_unsigned_char_normSq(_Underlying *_this);
            return __MR_SymMatrix3_unsigned_char_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix3<unsigned char>::det`.
        public unsafe byte Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_det", ExactSpelling = true)]
            extern static byte __MR_SymMatrix3_unsigned_char_det(_Underlying *_this);
            return __MR_SymMatrix3_unsigned_char_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix3<unsigned char>::inverse`.
        public unsafe MR.SymMatrix3_UnsignedChar Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix3_unsigned_char_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix3<unsigned char>::inverse`.
        public unsafe MR.SymMatrix3_UnsignedChar Inverse(byte det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_inverse_1(_Underlying *_this, byte det);
            return new(__MR_SymMatrix3_unsigned_char_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }
    }

    /// symmetric 3x3 matrix
    /// Generated from class `MR::SymMatrix3<unsigned char>`.
    /// This is the non-const half of the class.
    public class SymMatrix3_UnsignedChar : Const_SymMatrix3_UnsignedChar
    {
        internal unsafe SymMatrix3_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref byte Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_GetMutable_xx", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix3_unsigned_char_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_GetMutable_xy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix3_unsigned_char_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Xz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_GetMutable_xz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_GetMutable_xz(_Underlying *_this);
                return ref *__MR_SymMatrix3_unsigned_char_GetMutable_xz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_GetMutable_yy", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix3_unsigned_char_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Yz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_GetMutable_yz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_GetMutable_yz(_Underlying *_this);
                return ref *__MR_SymMatrix3_unsigned_char_GetMutable_yz(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref byte Zz
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_GetMutable_zz", ExactSpelling = true)]
                extern static byte *__MR_SymMatrix3_unsigned_char_GetMutable_zz(_Underlying *_this);
                return ref *__MR_SymMatrix3_unsigned_char_GetMutable_zz(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix3_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix3_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix3<unsigned char>::SymMatrix3`.
        public unsafe SymMatrix3_UnsignedChar(MR.Const_SymMatrix3_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_ConstructFromAnother(MR.SymMatrix3_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix3_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::operator=`.
        public unsafe MR.SymMatrix3_UnsignedChar Assign(MR.Const_SymMatrix3_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_AssignFromAnother(_Underlying *_this, MR.SymMatrix3_UnsignedChar._Underlying *_other);
            return new(__MR_SymMatrix3_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::operator+=`.
        public unsafe MR.SymMatrix3_UnsignedChar AddAssign(MR.Const_SymMatrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_add_assign(_Underlying *_this, MR.Const_SymMatrix3_UnsignedChar._Underlying *b);
            return new(__MR_SymMatrix3_unsigned_char_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::operator-=`.
        public unsafe MR.SymMatrix3_UnsignedChar SubAssign(MR.Const_SymMatrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_sub_assign(_Underlying *_this, MR.Const_SymMatrix3_UnsignedChar._Underlying *b);
            return new(__MR_SymMatrix3_unsigned_char_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::operator*=`.
        public unsafe MR.SymMatrix3_UnsignedChar MulAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_mul_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix3_unsigned_char_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix3<unsigned char>::operator/=`.
        public unsafe MR.SymMatrix3_UnsignedChar DivAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix3_unsigned_char_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix3_UnsignedChar._Underlying *__MR_SymMatrix3_unsigned_char_div_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix3_unsigned_char_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix3_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix3_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3_UnsignedChar`/`Const_SymMatrix3_UnsignedChar` directly.
    public class _InOptMut_SymMatrix3_UnsignedChar
    {
        public SymMatrix3_UnsignedChar? Opt;

        public _InOptMut_SymMatrix3_UnsignedChar() {}
        public _InOptMut_SymMatrix3_UnsignedChar(SymMatrix3_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix3_UnsignedChar(SymMatrix3_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix3_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix3_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix3_UnsignedChar`/`Const_SymMatrix3_UnsignedChar` to pass it to the function.
    public class _InOptConst_SymMatrix3_UnsignedChar
    {
        public Const_SymMatrix3_UnsignedChar? Opt;

        public _InOptConst_SymMatrix3_UnsignedChar() {}
        public _InOptConst_SymMatrix3_UnsignedChar(Const_SymMatrix3_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix3_UnsignedChar(Const_SymMatrix3_UnsignedChar value) {return new(value);}
    }
}
