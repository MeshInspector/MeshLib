public static partial class MR
{
    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2b`.
    /// This is the const half of the class.
    public class Const_SymMatrix2b : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix2b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix2b_Destroy(_Underlying *_this);
            __MR_SymMatrix2b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix2b() {Dispose(false);}

        /// zero matrix by default
        public unsafe bool Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_Get_xx", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix2b_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix2b_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_Get_xy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix2b_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix2b_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe bool Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_Get_yy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix2b_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix2b_Get_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix2b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2b_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2b::SymMatrix2b`.
        public unsafe Const_SymMatrix2b(MR.Const_SymMatrix2b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_ConstructFromAnother(MR.SymMatrix2b._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2b_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2b::identity`.
        public static unsafe MR.SymMatrix2b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_identity", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_identity();
            return new(__MR_SymMatrix2b_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix2b::diagonal`.
        public static unsafe MR.SymMatrix2b Diagonal(bool diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_diagonal(byte diagVal);
            return new(__MR_SymMatrix2b_diagonal(diagVal ? (byte)1 : (byte)0), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix2b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_trace", ExactSpelling = true)]
            extern static byte __MR_SymMatrix2b_trace(_Underlying *_this);
            return __MR_SymMatrix2b_trace(_UnderlyingPtr) != 0;
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 4 squared elements
        /// Generated from method `MR::SymMatrix2b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_normSq", ExactSpelling = true)]
            extern static byte __MR_SymMatrix2b_normSq(_Underlying *_this);
            return __MR_SymMatrix2b_normSq(_UnderlyingPtr) != 0;
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix2b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_det", ExactSpelling = true)]
            extern static byte __MR_SymMatrix2b_det(_Underlying *_this);
            return __MR_SymMatrix2b_det(_UnderlyingPtr) != 0;
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix2b::inverse`.
        public unsafe MR.SymMatrix2b Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix2b_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix2b::inverse`.
        public unsafe MR.SymMatrix2b Inverse(bool det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_inverse_1(_Underlying *_this, byte det);
            return new(__MR_SymMatrix2b_inverse_1(_UnderlyingPtr, det ? (byte)1 : (byte)0), is_owning: true);
        }
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2b`.
    /// This is the non-const half of the class.
    public class SymMatrix2b : Const_SymMatrix2b
    {
        internal unsafe SymMatrix2b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref bool Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_GetMutable_xx", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix2b_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix2b_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_GetMutable_xy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix2b_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix2b_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref bool Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_GetMutable_yy", ExactSpelling = true)]
                extern static bool *__MR_SymMatrix2b_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix2b_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix2b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2b_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2b::SymMatrix2b`.
        public unsafe SymMatrix2b(MR.Const_SymMatrix2b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_ConstructFromAnother(MR.SymMatrix2b._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2b_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2b::operator=`.
        public unsafe MR.SymMatrix2b Assign(MR.Const_SymMatrix2b _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_AssignFromAnother(_Underlying *_this, MR.SymMatrix2b._Underlying *_other);
            return new(__MR_SymMatrix2b_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2b::operator+=`.
        public unsafe MR.SymMatrix2b AddAssign(MR.Const_SymMatrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_add_assign(_Underlying *_this, MR.Const_SymMatrix2b._Underlying *b);
            return new(__MR_SymMatrix2b_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2b::operator-=`.
        public unsafe MR.SymMatrix2b SubAssign(MR.Const_SymMatrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_sub_assign(_Underlying *_this, MR.Const_SymMatrix2b._Underlying *b);
            return new(__MR_SymMatrix2b_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2b::operator*=`.
        public unsafe MR.SymMatrix2b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_mul_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix2b_mul_assign(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2b::operator/=`.
        public unsafe MR.SymMatrix2b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2b_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2b._Underlying *__MR_SymMatrix2b_div_assign(_Underlying *_this, byte b);
            return new(__MR_SymMatrix2b_div_assign(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix2b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix2b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2b`/`Const_SymMatrix2b` directly.
    public class _InOptMut_SymMatrix2b
    {
        public SymMatrix2b? Opt;

        public _InOptMut_SymMatrix2b() {}
        public _InOptMut_SymMatrix2b(SymMatrix2b value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix2b(SymMatrix2b value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix2b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix2b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2b`/`Const_SymMatrix2b` to pass it to the function.
    public class _InOptConst_SymMatrix2b
    {
        public Const_SymMatrix2b? Opt;

        public _InOptConst_SymMatrix2b() {}
        public _InOptConst_SymMatrix2b(Const_SymMatrix2b value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix2b(Const_SymMatrix2b value) {return new(value);}
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2i`.
    /// This is the const half of the class.
    public class Const_SymMatrix2i : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix2i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix2i_Destroy(_Underlying *_this);
            __MR_SymMatrix2i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix2i() {Dispose(false);}

        /// zero matrix by default
        public unsafe int Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_Get_xx", ExactSpelling = true)]
                extern static int *__MR_SymMatrix2i_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix2i_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_Get_xy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix2i_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix2i_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe int Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_Get_yy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix2i_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix2i_Get_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2i_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2i::SymMatrix2i`.
        public unsafe Const_SymMatrix2i(MR.Const_SymMatrix2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_ConstructFromAnother(MR.SymMatrix2i._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2i_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2i::identity`.
        public static unsafe MR.SymMatrix2i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_identity", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_identity();
            return new(__MR_SymMatrix2i_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix2i::diagonal`.
        public static unsafe MR.SymMatrix2i Diagonal(int diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_diagonal(int diagVal);
            return new(__MR_SymMatrix2i_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix2i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_trace", ExactSpelling = true)]
            extern static int __MR_SymMatrix2i_trace(_Underlying *_this);
            return __MR_SymMatrix2i_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 4 squared elements
        /// Generated from method `MR::SymMatrix2i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_normSq", ExactSpelling = true)]
            extern static int __MR_SymMatrix2i_normSq(_Underlying *_this);
            return __MR_SymMatrix2i_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix2i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_det", ExactSpelling = true)]
            extern static int __MR_SymMatrix2i_det(_Underlying *_this);
            return __MR_SymMatrix2i_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix2i::inverse`.
        public unsafe MR.SymMatrix2i Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix2i_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix2i::inverse`.
        public unsafe MR.SymMatrix2i Inverse(int det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_inverse_1(_Underlying *_this, int det);
            return new(__MR_SymMatrix2i_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2i`.
    /// This is the non-const half of the class.
    public class SymMatrix2i : Const_SymMatrix2i
    {
        internal unsafe SymMatrix2i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref int Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_GetMutable_xx", ExactSpelling = true)]
                extern static int *__MR_SymMatrix2i_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix2i_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_GetMutable_xy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix2i_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix2i_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref int Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_GetMutable_yy", ExactSpelling = true)]
                extern static int *__MR_SymMatrix2i_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix2i_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2i_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2i::SymMatrix2i`.
        public unsafe SymMatrix2i(MR.Const_SymMatrix2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_ConstructFromAnother(MR.SymMatrix2i._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2i_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2i::operator=`.
        public unsafe MR.SymMatrix2i Assign(MR.Const_SymMatrix2i _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_AssignFromAnother(_Underlying *_this, MR.SymMatrix2i._Underlying *_other);
            return new(__MR_SymMatrix2i_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i::operator+=`.
        public unsafe MR.SymMatrix2i AddAssign(MR.Const_SymMatrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_add_assign(_Underlying *_this, MR.Const_SymMatrix2i._Underlying *b);
            return new(__MR_SymMatrix2i_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i::operator-=`.
        public unsafe MR.SymMatrix2i SubAssign(MR.Const_SymMatrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_sub_assign(_Underlying *_this, MR.Const_SymMatrix2i._Underlying *b);
            return new(__MR_SymMatrix2i_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i::operator*=`.
        public unsafe MR.SymMatrix2i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_mul_assign(_Underlying *_this, int b);
            return new(__MR_SymMatrix2i_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i::operator/=`.
        public unsafe MR.SymMatrix2i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i._Underlying *__MR_SymMatrix2i_div_assign(_Underlying *_this, int b);
            return new(__MR_SymMatrix2i_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix2i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2i`/`Const_SymMatrix2i` directly.
    public class _InOptMut_SymMatrix2i
    {
        public SymMatrix2i? Opt;

        public _InOptMut_SymMatrix2i() {}
        public _InOptMut_SymMatrix2i(SymMatrix2i value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix2i(SymMatrix2i value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix2i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2i`/`Const_SymMatrix2i` to pass it to the function.
    public class _InOptConst_SymMatrix2i
    {
        public Const_SymMatrix2i? Opt;

        public _InOptConst_SymMatrix2i() {}
        public _InOptConst_SymMatrix2i(Const_SymMatrix2i value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix2i(Const_SymMatrix2i value) {return new(value);}
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2i64`.
    /// This is the const half of the class.
    public class Const_SymMatrix2i64 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix2i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix2i64_Destroy(_Underlying *_this);
            __MR_SymMatrix2i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix2i64() {Dispose(false);}

        /// zero matrix by default
        public unsafe long Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_Get_xx", ExactSpelling = true)]
                extern static long *__MR_SymMatrix2i64_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix2i64_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_Get_xy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix2i64_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix2i64_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe long Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_Get_yy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix2i64_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix2i64_Get_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2i64::SymMatrix2i64`.
        public unsafe Const_SymMatrix2i64(MR.Const_SymMatrix2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_ConstructFromAnother(MR.SymMatrix2i64._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2i64_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2i64::identity`.
        public static unsafe MR.SymMatrix2i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_identity", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_identity();
            return new(__MR_SymMatrix2i64_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix2i64::diagonal`.
        public static unsafe MR.SymMatrix2i64 Diagonal(long diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_diagonal(long diagVal);
            return new(__MR_SymMatrix2i64_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix2i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_trace", ExactSpelling = true)]
            extern static long __MR_SymMatrix2i64_trace(_Underlying *_this);
            return __MR_SymMatrix2i64_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 4 squared elements
        /// Generated from method `MR::SymMatrix2i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_normSq", ExactSpelling = true)]
            extern static long __MR_SymMatrix2i64_normSq(_Underlying *_this);
            return __MR_SymMatrix2i64_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix2i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_det", ExactSpelling = true)]
            extern static long __MR_SymMatrix2i64_det(_Underlying *_this);
            return __MR_SymMatrix2i64_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix2i64::inverse`.
        public unsafe MR.SymMatrix2i64 Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix2i64_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix2i64::inverse`.
        public unsafe MR.SymMatrix2i64 Inverse(long det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_inverse_1(_Underlying *_this, long det);
            return new(__MR_SymMatrix2i64_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2i64`.
    /// This is the non-const half of the class.
    public class SymMatrix2i64 : Const_SymMatrix2i64
    {
        internal unsafe SymMatrix2i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref long Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_GetMutable_xx", ExactSpelling = true)]
                extern static long *__MR_SymMatrix2i64_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix2i64_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_GetMutable_xy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix2i64_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix2i64_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref long Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_GetMutable_yy", ExactSpelling = true)]
                extern static long *__MR_SymMatrix2i64_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix2i64_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2i64::SymMatrix2i64`.
        public unsafe SymMatrix2i64(MR.Const_SymMatrix2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_ConstructFromAnother(MR.SymMatrix2i64._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2i64_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2i64::operator=`.
        public unsafe MR.SymMatrix2i64 Assign(MR.Const_SymMatrix2i64 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_AssignFromAnother(_Underlying *_this, MR.SymMatrix2i64._Underlying *_other);
            return new(__MR_SymMatrix2i64_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i64::operator+=`.
        public unsafe MR.SymMatrix2i64 AddAssign(MR.Const_SymMatrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_add_assign(_Underlying *_this, MR.Const_SymMatrix2i64._Underlying *b);
            return new(__MR_SymMatrix2i64_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i64::operator-=`.
        public unsafe MR.SymMatrix2i64 SubAssign(MR.Const_SymMatrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_sub_assign(_Underlying *_this, MR.Const_SymMatrix2i64._Underlying *b);
            return new(__MR_SymMatrix2i64_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i64::operator*=`.
        public unsafe MR.SymMatrix2i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_mul_assign(_Underlying *_this, long b);
            return new(__MR_SymMatrix2i64_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2i64::operator/=`.
        public unsafe MR.SymMatrix2i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2i64_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2i64._Underlying *__MR_SymMatrix2i64_div_assign(_Underlying *_this, long b);
            return new(__MR_SymMatrix2i64_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix2i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2i64`/`Const_SymMatrix2i64` directly.
    public class _InOptMut_SymMatrix2i64
    {
        public SymMatrix2i64? Opt;

        public _InOptMut_SymMatrix2i64() {}
        public _InOptMut_SymMatrix2i64(SymMatrix2i64 value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix2i64(SymMatrix2i64 value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix2i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2i64`/`Const_SymMatrix2i64` to pass it to the function.
    public class _InOptConst_SymMatrix2i64
    {
        public Const_SymMatrix2i64? Opt;

        public _InOptConst_SymMatrix2i64() {}
        public _InOptConst_SymMatrix2i64(Const_SymMatrix2i64 value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix2i64(Const_SymMatrix2i64 value) {return new(value);}
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2f`.
    /// This is the const half of the class.
    public class Const_SymMatrix2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix2f_Destroy(_Underlying *_this);
            __MR_SymMatrix2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix2f() {Dispose(false);}

        /// zero matrix by default
        public unsafe float Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_Get_xx", ExactSpelling = true)]
                extern static float *__MR_SymMatrix2f_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix2f_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_Get_xy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix2f_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix2f_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe float Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_Get_yy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix2f_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix2f_Get_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2f::SymMatrix2f`.
        public unsafe Const_SymMatrix2f(MR.Const_SymMatrix2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_ConstructFromAnother(MR.SymMatrix2f._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2f::identity`.
        public static unsafe MR.SymMatrix2f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_identity", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_identity();
            return new(__MR_SymMatrix2f_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix2f::diagonal`.
        public static unsafe MR.SymMatrix2f Diagonal(float diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_diagonal(float diagVal);
            return new(__MR_SymMatrix2f_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix2f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_trace", ExactSpelling = true)]
            extern static float __MR_SymMatrix2f_trace(_Underlying *_this);
            return __MR_SymMatrix2f_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 4 squared elements
        /// Generated from method `MR::SymMatrix2f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_normSq", ExactSpelling = true)]
            extern static float __MR_SymMatrix2f_normSq(_Underlying *_this);
            return __MR_SymMatrix2f_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix2f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_det", ExactSpelling = true)]
            extern static float __MR_SymMatrix2f_det(_Underlying *_this);
            return __MR_SymMatrix2f_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix2f::inverse`.
        public unsafe MR.SymMatrix2f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix2f_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix2f::inverse`.
        public unsafe MR.SymMatrix2f Inverse(float det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_inverse_1(_Underlying *_this, float det);
            return new(__MR_SymMatrix2f_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }

        /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
        /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
        /// M*V^T = V^T*L; M = V^T*L*V
        /// Generated from method `MR::SymMatrix2f::eigens`.
        public unsafe MR.Vector2f Eigens(MR.Mut_Matrix2f? eigenvectors = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_eigens", ExactSpelling = true)]
            extern static MR.Vector2f __MR_SymMatrix2f_eigens(_Underlying *_this, MR.Mut_Matrix2f._Underlying *eigenvectors);
            return __MR_SymMatrix2f_eigens(_UnderlyingPtr, eigenvectors is not null ? eigenvectors._UnderlyingPtr : null);
        }

        /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
        /// Generated from method `MR::SymMatrix2f::eigenvector`.
        public unsafe MR.Vector2f Eigenvector(float eigenvalue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_eigenvector", ExactSpelling = true)]
            extern static MR.Vector2f __MR_SymMatrix2f_eigenvector(_Underlying *_this, float eigenvalue);
            return __MR_SymMatrix2f_eigenvector(_UnderlyingPtr, eigenvalue);
        }

        /// computes not-unit eigenvector corresponding to maximum eigenvalue
        /// Generated from method `MR::SymMatrix2f::maxEigenvector`.
        public unsafe MR.Vector2f MaxEigenvector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_maxEigenvector", ExactSpelling = true)]
            extern static MR.Vector2f __MR_SymMatrix2f_maxEigenvector(_Underlying *_this);
            return __MR_SymMatrix2f_maxEigenvector(_UnderlyingPtr);
        }

        /// for not-degenerate matrix returns just inverse matrix, otherwise
        /// returns degenerate matrix, which performs inversion on not-kernel subspace;
        /// \param tol relative epsilon-tolerance for too small number detection
        /// \param rank optional output for this matrix rank according to given tolerance
        /// \param space rank=1: unit direction of solution line, rank=2: zero vector
        /// Generated from method `MR::SymMatrix2f::pseudoinverse`.
        /// Parameter `tol` defaults to `std::numeric_limits<float>::epsilon()`.
        public unsafe MR.SymMatrix2f Pseudoinverse(float? tol = null, MR.Misc.InOut<int>? rank = null, MR.Mut_Vector2f? space = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_pseudoinverse", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_pseudoinverse(_Underlying *_this, float *tol, int *rank, MR.Mut_Vector2f._Underlying *space);
            float __deref_tol = tol.GetValueOrDefault();
            int __value_rank = rank is not null ? rank.Value : default(int);
            var __ret = __MR_SymMatrix2f_pseudoinverse(_UnderlyingPtr, tol.HasValue ? &__deref_tol : null, rank is not null ? &__value_rank : null, space is not null ? space._UnderlyingPtr : null);
            if (rank is not null) rank.Value = __value_rank;
            return new(__ret, is_owning: true);
        }
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2f`.
    /// This is the non-const half of the class.
    public class SymMatrix2f : Const_SymMatrix2f
    {
        internal unsafe SymMatrix2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref float Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_GetMutable_xx", ExactSpelling = true)]
                extern static float *__MR_SymMatrix2f_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix2f_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_GetMutable_xy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix2f_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix2f_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref float Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_GetMutable_yy", ExactSpelling = true)]
                extern static float *__MR_SymMatrix2f_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix2f_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2f::SymMatrix2f`.
        public unsafe SymMatrix2f(MR.Const_SymMatrix2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_ConstructFromAnother(MR.SymMatrix2f._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2f::operator=`.
        public unsafe MR.SymMatrix2f Assign(MR.Const_SymMatrix2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_AssignFromAnother(_Underlying *_this, MR.SymMatrix2f._Underlying *_other);
            return new(__MR_SymMatrix2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2f::operator+=`.
        public unsafe MR.SymMatrix2f AddAssign(MR.Const_SymMatrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_add_assign(_Underlying *_this, MR.Const_SymMatrix2f._Underlying *b);
            return new(__MR_SymMatrix2f_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2f::operator-=`.
        public unsafe MR.SymMatrix2f SubAssign(MR.Const_SymMatrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_sub_assign(_Underlying *_this, MR.Const_SymMatrix2f._Underlying *b);
            return new(__MR_SymMatrix2f_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2f::operator*=`.
        public unsafe MR.SymMatrix2f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_mul_assign(_Underlying *_this, float b);
            return new(__MR_SymMatrix2f_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2f::operator/=`.
        public unsafe MR.SymMatrix2f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2f_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2f._Underlying *__MR_SymMatrix2f_div_assign(_Underlying *_this, float b);
            return new(__MR_SymMatrix2f_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2f`/`Const_SymMatrix2f` directly.
    public class _InOptMut_SymMatrix2f
    {
        public SymMatrix2f? Opt;

        public _InOptMut_SymMatrix2f() {}
        public _InOptMut_SymMatrix2f(SymMatrix2f value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix2f(SymMatrix2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2f`/`Const_SymMatrix2f` to pass it to the function.
    public class _InOptConst_SymMatrix2f
    {
        public Const_SymMatrix2f? Opt;

        public _InOptConst_SymMatrix2f() {}
        public _InOptConst_SymMatrix2f(Const_SymMatrix2f value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix2f(Const_SymMatrix2f value) {return new(value);}
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2d`.
    /// This is the const half of the class.
    public class Const_SymMatrix2d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymMatrix2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_Destroy", ExactSpelling = true)]
            extern static void __MR_SymMatrix2d_Destroy(_Underlying *_this);
            __MR_SymMatrix2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymMatrix2d() {Dispose(false);}

        /// zero matrix by default
        public unsafe double Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_Get_xx", ExactSpelling = true)]
                extern static double *__MR_SymMatrix2d_Get_xx(_Underlying *_this);
                return *__MR_SymMatrix2d_Get_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_Get_xy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix2d_Get_xy(_Underlying *_this);
                return *__MR_SymMatrix2d_Get_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public unsafe double Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_Get_yy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix2d_Get_yy(_Underlying *_this);
                return *__MR_SymMatrix2d_Get_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymMatrix2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2d::SymMatrix2d`.
        public unsafe Const_SymMatrix2d(MR.Const_SymMatrix2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_ConstructFromAnother(MR.SymMatrix2d._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2d::identity`.
        public static unsafe MR.SymMatrix2d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_identity", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_identity();
            return new(__MR_SymMatrix2d_identity(), is_owning: true);
        }

        /// Generated from method `MR::SymMatrix2d::diagonal`.
        public static unsafe MR.SymMatrix2d Diagonal(double diagVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_diagonal", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_diagonal(double diagVal);
            return new(__MR_SymMatrix2d_diagonal(diagVal), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::SymMatrix2d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_trace", ExactSpelling = true)]
            extern static double __MR_SymMatrix2d_trace(_Underlying *_this);
            return __MR_SymMatrix2d_trace(_UnderlyingPtr);
        }

        /// computes the squared norm of the matrix, which is equal to the sum of 4 squared elements
        /// Generated from method `MR::SymMatrix2d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_normSq", ExactSpelling = true)]
            extern static double __MR_SymMatrix2d_normSq(_Underlying *_this);
            return __MR_SymMatrix2d_normSq(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::SymMatrix2d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_det", ExactSpelling = true)]
            extern static double __MR_SymMatrix2d_det(_Underlying *_this);
            return __MR_SymMatrix2d_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::SymMatrix2d::inverse`.
        public unsafe MR.SymMatrix2d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_inverse_0", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_inverse_0(_Underlying *_this);
            return new(__MR_SymMatrix2d_inverse_0(_UnderlyingPtr), is_owning: true);
        }

        /// computes inverse matrix given determinant of this
        /// Generated from method `MR::SymMatrix2d::inverse`.
        public unsafe MR.SymMatrix2d Inverse(double det)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_inverse_1", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_inverse_1(_Underlying *_this, double det);
            return new(__MR_SymMatrix2d_inverse_1(_UnderlyingPtr, det), is_owning: true);
        }

        /// returns eigenvalues of the matrix in ascending order (diagonal matrix L), and
        /// optionally returns corresponding unit eigenvectors in the rows of orthogonal matrix V,
        /// M*V^T = V^T*L; M = V^T*L*V
        /// Generated from method `MR::SymMatrix2d::eigens`.
        public unsafe MR.Vector2d Eigens(MR.Mut_Matrix2d? eigenvectors = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_eigens", ExactSpelling = true)]
            extern static MR.Vector2d __MR_SymMatrix2d_eigens(_Underlying *_this, MR.Mut_Matrix2d._Underlying *eigenvectors);
            return __MR_SymMatrix2d_eigens(_UnderlyingPtr, eigenvectors is not null ? eigenvectors._UnderlyingPtr : null);
        }

        /// computes not-unit eigenvector corresponding to a not-repeating eigenvalue
        /// Generated from method `MR::SymMatrix2d::eigenvector`.
        public unsafe MR.Vector2d Eigenvector(double eigenvalue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_eigenvector", ExactSpelling = true)]
            extern static MR.Vector2d __MR_SymMatrix2d_eigenvector(_Underlying *_this, double eigenvalue);
            return __MR_SymMatrix2d_eigenvector(_UnderlyingPtr, eigenvalue);
        }

        /// computes not-unit eigenvector corresponding to maximum eigenvalue
        /// Generated from method `MR::SymMatrix2d::maxEigenvector`.
        public unsafe MR.Vector2d MaxEigenvector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_maxEigenvector", ExactSpelling = true)]
            extern static MR.Vector2d __MR_SymMatrix2d_maxEigenvector(_Underlying *_this);
            return __MR_SymMatrix2d_maxEigenvector(_UnderlyingPtr);
        }

        /// for not-degenerate matrix returns just inverse matrix, otherwise
        /// returns degenerate matrix, which performs inversion on not-kernel subspace;
        /// \param tol relative epsilon-tolerance for too small number detection
        /// \param rank optional output for this matrix rank according to given tolerance
        /// \param space rank=1: unit direction of solution line, rank=2: zero vector
        /// Generated from method `MR::SymMatrix2d::pseudoinverse`.
        /// Parameter `tol` defaults to `std::numeric_limits<double>::epsilon()`.
        public unsafe MR.SymMatrix2d Pseudoinverse(double? tol = null, MR.Misc.InOut<int>? rank = null, MR.Mut_Vector2d? space = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_pseudoinverse", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_pseudoinverse(_Underlying *_this, double *tol, int *rank, MR.Mut_Vector2d._Underlying *space);
            double __deref_tol = tol.GetValueOrDefault();
            int __value_rank = rank is not null ? rank.Value : default(int);
            var __ret = __MR_SymMatrix2d_pseudoinverse(_UnderlyingPtr, tol.HasValue ? &__deref_tol : null, rank is not null ? &__value_rank : null, space is not null ? space._UnderlyingPtr : null);
            if (rank is not null) rank.Value = __value_rank;
            return new(__ret, is_owning: true);
        }
    }

    /// symmetric 2x2 matrix
    /// Generated from class `MR::SymMatrix2d`.
    /// This is the non-const half of the class.
    public class SymMatrix2d : Const_SymMatrix2d
    {
        internal unsafe SymMatrix2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// zero matrix by default
        public new unsafe ref double Xx
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_GetMutable_xx", ExactSpelling = true)]
                extern static double *__MR_SymMatrix2d_GetMutable_xx(_Underlying *_this);
                return ref *__MR_SymMatrix2d_GetMutable_xx(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Xy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_GetMutable_xy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix2d_GetMutable_xy(_Underlying *_this);
                return ref *__MR_SymMatrix2d_GetMutable_xy(_UnderlyingPtr);
            }
        }

        /// zero matrix by default
        public new unsafe ref double Yy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_GetMutable_yy", ExactSpelling = true)]
                extern static double *__MR_SymMatrix2d_GetMutable_yy(_Underlying *_this);
                return ref *__MR_SymMatrix2d_GetMutable_yy(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymMatrix2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_DefaultConstruct();
            _UnderlyingPtr = __MR_SymMatrix2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::SymMatrix2d::SymMatrix2d`.
        public unsafe SymMatrix2d(MR.Const_SymMatrix2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_ConstructFromAnother(MR.SymMatrix2d._Underlying *_other);
            _UnderlyingPtr = __MR_SymMatrix2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SymMatrix2d::operator=`.
        public unsafe MR.SymMatrix2d Assign(MR.Const_SymMatrix2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_AssignFromAnother(_Underlying *_this, MR.SymMatrix2d._Underlying *_other);
            return new(__MR_SymMatrix2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2d::operator+=`.
        public unsafe MR.SymMatrix2d AddAssign(MR.Const_SymMatrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_add_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_add_assign(_Underlying *_this, MR.Const_SymMatrix2d._Underlying *b);
            return new(__MR_SymMatrix2d_add_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2d::operator-=`.
        public unsafe MR.SymMatrix2d SubAssign(MR.Const_SymMatrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_sub_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_sub_assign(_Underlying *_this, MR.Const_SymMatrix2d._Underlying *b);
            return new(__MR_SymMatrix2d_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2d::operator*=`.
        public unsafe MR.SymMatrix2d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_mul_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_mul_assign(_Underlying *_this, double b);
            return new(__MR_SymMatrix2d_mul_assign(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from method `MR::SymMatrix2d::operator/=`.
        public unsafe MR.SymMatrix2d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymMatrix2d_div_assign", ExactSpelling = true)]
            extern static MR.SymMatrix2d._Underlying *__MR_SymMatrix2d_div_assign(_Underlying *_this, double b);
            return new(__MR_SymMatrix2d_div_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SymMatrix2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymMatrix2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2d`/`Const_SymMatrix2d` directly.
    public class _InOptMut_SymMatrix2d
    {
        public SymMatrix2d? Opt;

        public _InOptMut_SymMatrix2d() {}
        public _InOptMut_SymMatrix2d(SymMatrix2d value) {Opt = value;}
        public static implicit operator _InOptMut_SymMatrix2d(SymMatrix2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymMatrix2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymMatrix2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymMatrix2d`/`Const_SymMatrix2d` to pass it to the function.
    public class _InOptConst_SymMatrix2d
    {
        public Const_SymMatrix2d? Opt;

        public _InOptConst_SymMatrix2d() {}
        public _InOptConst_SymMatrix2d(Const_SymMatrix2d value) {Opt = value;}
        public static implicit operator _InOptConst_SymMatrix2d(Const_SymMatrix2d value) {return new(value);}
    }
}
