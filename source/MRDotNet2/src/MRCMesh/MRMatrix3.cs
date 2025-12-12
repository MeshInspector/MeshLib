public static partial class MR
{
    public static partial class Matrix3_Bool
    {
        /// Generated from class `MR::Matrix3<bool>::QR`.
        /// This is the const half of the class.
        public class Const_QR : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_QR(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_Destroy", ExactSpelling = true)]
                extern static void __MR_Matrix3_bool_QR_Destroy(_Underlying *_this);
                __MR_Matrix3_bool_QR_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_QR() {Dispose(false);}

            public unsafe MR.Const_Matrix3b Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_Get_q", ExactSpelling = true)]
                    extern static MR.Const_Matrix3b._Underlying *__MR_Matrix3_bool_QR_Get_q(_Underlying *_this);
                    return new(__MR_Matrix3_bool_QR_Get_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Matrix3b R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_Get_r", ExactSpelling = true)]
                    extern static MR.Const_Matrix3b._Underlying *__MR_Matrix3_bool_QR_Get_r(_Underlying *_this);
                    return new(__MR_Matrix3_bool_QR_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_bool_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<bool>::QR` elementwise.
            public unsafe Const_QR(MR.Matrix3b q, MR.Matrix3b r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_ConstructFrom(MR.Matrix3b q, MR.Matrix3b r);
                _UnderlyingPtr = __MR_Matrix3_bool_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<bool>::QR::QR`.
            public unsafe Const_QR(MR.Matrix3_Bool.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_ConstructFromAnother(MR.Matrix3_Bool.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_bool_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Matrix3<bool>::QR`.
        /// This is the non-const half of the class.
        public class QR : Const_QR
        {
            internal unsafe QR(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Matrix3b Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_GetMutable_q", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3b._Underlying *__MR_Matrix3_bool_QR_GetMutable_q(_Underlying *_this);
                    return new(__MR_Matrix3_bool_QR_GetMutable_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Matrix3b R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3b._Underlying *__MR_Matrix3_bool_QR_GetMutable_r(_Underlying *_this);
                    return new(__MR_Matrix3_bool_QR_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_bool_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<bool>::QR` elementwise.
            public unsafe QR(MR.Matrix3b q, MR.Matrix3b r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_ConstructFrom(MR.Matrix3b q, MR.Matrix3b r);
                _UnderlyingPtr = __MR_Matrix3_bool_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<bool>::QR::QR`.
            public unsafe QR(MR.Matrix3_Bool.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_ConstructFromAnother(MR.Matrix3_Bool.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_bool_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Matrix3<bool>::QR::operator=`.
            public unsafe MR.Matrix3_Bool.QR Assign(MR.Matrix3_Bool.Const_QR _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_bool_QR_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Bool.QR._Underlying *__MR_Matrix3_bool_QR_AssignFromAnother(_Underlying *_this, MR.Matrix3_Bool.QR._Underlying *_other);
                return new(__MR_Matrix3_bool_QR_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` directly.
        public class _InOptMut_QR
        {
            public QR? Opt;

            public _InOptMut_QR() {}
            public _InOptMut_QR(QR value) {Opt = value;}
            public static implicit operator _InOptMut_QR(QR value) {return new(value);}
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` to pass it to the function.
        public class _InOptConst_QR
        {
            public Const_QR? Opt;

            public _InOptConst_QR() {}
            public _InOptConst_QR(Const_QR value) {Opt = value;}
            public static implicit operator _InOptConst_QR(Const_QR value) {return new(value);}
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3b`.
    /// This is the const reference to the struct.
    public class Const_Matrix3b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix3b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix3b UnderlyingStruct => ref *(Matrix3b *)_UnderlyingPtr;

        internal unsafe Const_Matrix3b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix3b_Destroy(_Underlying *_this);
            __MR_Matrix3b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix3b() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector3b X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector3b Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector3b Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Const_Matrix3b(Const_Matrix3b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(9);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 9);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix3b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(9);
            MR.Matrix3b _ctor_result = __MR_Matrix3b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 9);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3b::Matrix3b`.
        public unsafe Const_Matrix3b(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_Construct", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_Construct(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(9);
            MR.Matrix3b _ctor_result = __MR_Matrix3b_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 9);
        }

        /// Generated from method `MR::Matrix3b::zero`.
        public static MR.Matrix3b Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_zero", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_zero();
            return __MR_Matrix3b_zero();
        }

        /// Generated from method `MR::Matrix3b::identity`.
        public static MR.Matrix3b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_identity", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_identity();
            return __MR_Matrix3b_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3b::scale`.
        public static MR.Matrix3b Scale(bool s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_scale_1_bool", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_scale_1_bool(byte s);
            return __MR_Matrix3b_scale_1_bool(s ? (byte)1 : (byte)0);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3b::scale`.
        public static MR.Matrix3b Scale(bool sx, bool sy, bool sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_scale_3(byte sx, byte sy, byte sz);
            return __MR_Matrix3b_scale_3(sx ? (byte)1 : (byte)0, sy ? (byte)1 : (byte)0, sz ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Matrix3b::scale`.
        public static unsafe MR.Matrix3b Scale(MR.Const_Vector3b s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_scale_1_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_scale_1_MR_Vector3b(MR.Const_Vector3b._Underlying *s);
            return __MR_Matrix3b_scale_1_MR_Vector3b(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3b::fromRows`.
        public static unsafe MR.Matrix3b FromRows(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_fromRows(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            return __MR_Matrix3b_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3b::fromColumns`.
        public static unsafe MR.Matrix3b FromColumns(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_fromColumns(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            return __MR_Matrix3b_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3b::operator[]`.
        public unsafe MR.Const_Vector3b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3b._Underlying *__MR_Matrix3b_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix3b_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix3b::col`.
        public unsafe MR.Vector3b Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_col", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Matrix3b_col(_Underlying *_this, int i);
            return __MR_Matrix3b_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix3b_trace(_Underlying *_this);
            return __MR_Matrix3b_trace(_UnderlyingPtr) != 0;
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix3b_normSq(_Underlying *_this);
            return __MR_Matrix3b_normSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Matrix3b::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3b_norm(_Underlying *_this);
            return __MR_Matrix3b_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_det", ExactSpelling = true)]
            extern static byte __MR_Matrix3b_det(_Underlying *_this);
            return __MR_Matrix3b_det(_UnderlyingPtr) != 0;
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3b::transposed`.
        public unsafe MR.Matrix3b Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_transposed", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_transposed(_Underlying *_this);
            return __MR_Matrix3b_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_equal_MR_Matrix3b(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix3b a, MR.Const_Matrix3b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i operator+(MR.Const_Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_add_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_add_MR_Matrix3b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i operator-(MR.Const_Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_sub_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_sub_MR_Matrix3b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(bool a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_bool_MR_Matrix3b(byte a, MR.Const_Matrix3b._Underlying *b);
            return __MR_mul_bool_MR_Matrix3b(a ? (byte)1 : (byte)0, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Const_Matrix3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3b_bool(MR.Const_Matrix3b._Underlying *b, byte a);
            return __MR_mul_MR_Matrix3b_bool(b._UnderlyingPtr, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix3i operator/(Const_Matrix3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_div_MR_Matrix3b_bool(MR.Matrix3b b, byte a);
            return __MR_div_MR_Matrix3b_bool(b.UnderlyingStruct, a ? (byte)1 : (byte)0);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Const_Matrix3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3b_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Matrix3b_MR_Vector3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_mul_MR_Matrix3b_MR_Vector3b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Const_Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_mul_MR_Matrix3b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix3b? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix3b)
                return this == (MR.Const_Matrix3b)other;
            return false;
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3b`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix3b : Const_Matrix3b
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix3b UnderlyingStruct => ref *(Matrix3b *)_UnderlyingPtr;

        internal unsafe Mut_Matrix3b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector3b X => ref UnderlyingStruct.X;

        public new ref MR.Vector3b Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector3b Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Matrix3b(Const_Matrix3b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(9);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 9);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix3b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(9);
            MR.Matrix3b _ctor_result = __MR_Matrix3b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 9);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3b::Matrix3b`.
        public unsafe Mut_Matrix3b(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_Construct", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_Construct(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(9);
            MR.Matrix3b _ctor_result = __MR_Matrix3b_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 9);
        }

        /// Generated from method `MR::Matrix3b::operator[]`.
        public unsafe new MR.Mut_Vector3b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_Matrix3b_index(_Underlying *_this, int row);
            return new(__MR_Matrix3b_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3b AddAssign(MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_add_assign_MR_Matrix3b(_Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return new(__MR_add_assign_MR_Matrix3b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3b SubAssign(MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_sub_assign_MR_Matrix3b(_Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix3b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_mul_assign_MR_Matrix3b_bool(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Matrix3b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_div_assign_MR_Matrix3b_bool(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Matrix3b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3b`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 9)]
    public struct Matrix3b
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix3b(Const_Matrix3b other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix3b(Matrix3b other) => new(new Mut_Matrix3b((Mut_Matrix3b._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3b X;

        [System.Runtime.InteropServices.FieldOffset(3)]
        public MR.Vector3b Y;

        [System.Runtime.InteropServices.FieldOffset(6)]
        public MR.Vector3b Z;

        /// Generated copy constructor.
        public Matrix3b(Matrix3b _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix3b()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_DefaultConstruct();
            this = __MR_Matrix3b_DefaultConstruct();
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3b::Matrix3b`.
        public unsafe Matrix3b(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_Construct", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_Construct(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            this = __MR_Matrix3b_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3b::zero`.
        public static MR.Matrix3b Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_zero", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_zero();
            return __MR_Matrix3b_zero();
        }

        /// Generated from method `MR::Matrix3b::identity`.
        public static MR.Matrix3b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_identity", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_identity();
            return __MR_Matrix3b_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3b::scale`.
        public static MR.Matrix3b Scale(bool s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_scale_1_bool", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_scale_1_bool(byte s);
            return __MR_Matrix3b_scale_1_bool(s ? (byte)1 : (byte)0);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3b::scale`.
        public static MR.Matrix3b Scale(bool sx, bool sy, bool sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_scale_3(byte sx, byte sy, byte sz);
            return __MR_Matrix3b_scale_3(sx ? (byte)1 : (byte)0, sy ? (byte)1 : (byte)0, sz ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Matrix3b::scale`.
        public static unsafe MR.Matrix3b Scale(MR.Const_Vector3b s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_scale_1_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_scale_1_MR_Vector3b(MR.Const_Vector3b._Underlying *s);
            return __MR_Matrix3b_scale_1_MR_Vector3b(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3b::fromRows`.
        public static unsafe MR.Matrix3b FromRows(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_fromRows(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            return __MR_Matrix3b_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3b::fromColumns`.
        public static unsafe MR.Matrix3b FromColumns(MR.Const_Vector3b x, MR.Const_Vector3b y, MR.Const_Vector3b z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_fromColumns(MR.Const_Vector3b._Underlying *x, MR.Const_Vector3b._Underlying *y, MR.Const_Vector3b._Underlying *z);
            return __MR_Matrix3b_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3b::operator[]`.
        public unsafe MR.Const_Vector3b Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3b._Underlying *__MR_Matrix3b_index_const(MR.Matrix3b *_this, int row);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return new(__MR_Matrix3b_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix3b::operator[]`.
        public unsafe MR.Mut_Vector3b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_Matrix3b_index(MR.Matrix3b *_this, int row);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return new(__MR_Matrix3b_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix3b::col`.
        public unsafe MR.Vector3b Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_col", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Matrix3b_col(MR.Matrix3b *_this, int i);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return __MR_Matrix3b_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix3b_trace(MR.Matrix3b *_this);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return __MR_Matrix3b_trace(__ptr__this) != 0;
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix3b_normSq(MR.Matrix3b *_this);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return __MR_Matrix3b_normSq(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::Matrix3b::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3b_norm(MR.Matrix3b *_this);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return __MR_Matrix3b_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_det", ExactSpelling = true)]
            extern static byte __MR_Matrix3b_det(MR.Matrix3b *_this);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return __MR_Matrix3b_det(__ptr__this) != 0;
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3b::transposed`.
        public unsafe MR.Matrix3b Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3b_transposed", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix3b_transposed(MR.Matrix3b *_this);
            fixed (MR.Matrix3b *__ptr__this = &this)
            {
                return __MR_Matrix3b_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix3b a, MR.Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_equal_MR_Matrix3b((MR.Mut_Matrix3b._Underlying *)&a, (MR.Mut_Matrix3b._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix3b a, MR.Matrix3b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i operator+(MR.Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_add_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_add_MR_Matrix3b((MR.Mut_Matrix3b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i operator-(MR.Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_sub_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_sub_MR_Matrix3b((MR.Mut_Matrix3b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(bool a, MR.Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_bool_MR_Matrix3b(byte a, MR.Const_Matrix3b._Underlying *b);
            return __MR_mul_bool_MR_Matrix3b(a ? (byte)1 : (byte)0, (MR.Mut_Matrix3b._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Matrix3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3b_bool(MR.Const_Matrix3b._Underlying *b, byte a);
            return __MR_mul_MR_Matrix3b_bool((MR.Mut_Matrix3b._Underlying *)&b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix3i operator/(MR.Matrix3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_div_MR_Matrix3b_bool(MR.Matrix3b b, byte a);
            return __MR_div_MR_Matrix3b_bool(b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3b AddAssign(MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_add_assign_MR_Matrix3b(MR.Matrix3b *a, MR.Const_Matrix3b._Underlying *b);
            fixed (MR.Matrix3b *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix3b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3b SubAssign(MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_sub_assign_MR_Matrix3b(MR.Matrix3b *a, MR.Const_Matrix3b._Underlying *b);
            fixed (MR.Matrix3b *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix3b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_mul_assign_MR_Matrix3b_bool(MR.Matrix3b *a, byte b);
            fixed (MR.Matrix3b *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix3b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix3b._Underlying *__MR_div_assign_MR_Matrix3b_bool(MR.Matrix3b *a, byte b);
            fixed (MR.Matrix3b *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix3b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Matrix3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3b_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Matrix3b_MR_Vector3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_mul_MR_Matrix3b_MR_Vector3b((MR.Mut_Matrix3b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Matrix3b a, MR.Const_Matrix3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3b", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3b(MR.Const_Matrix3b._Underlying *a, MR.Const_Matrix3b._Underlying *b);
            return __MR_mul_MR_Matrix3b((MR.Mut_Matrix3b._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix3b b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix3b)
                return this == (MR.Matrix3b)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix3b` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix3b`/`Const_Matrix3b` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix3b
    {
        public readonly bool HasValue;
        internal readonly Matrix3b Object;
        public Matrix3b Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix3b() {HasValue = false;}
        public _InOpt_Matrix3b(Matrix3b new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix3b(Matrix3b new_value) {return new(new_value);}
        public _InOpt_Matrix3b(Const_Matrix3b new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix3b(Const_Matrix3b new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix3b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix3b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3b`/`Const_Matrix3b` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3b`.
    public class _InOptMut_Matrix3b
    {
        public Mut_Matrix3b? Opt;

        public _InOptMut_Matrix3b() {}
        public _InOptMut_Matrix3b(Mut_Matrix3b value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix3b(Mut_Matrix3b value) {return new(value);}
        public unsafe _InOptMut_Matrix3b(ref Matrix3b value)
        {
            fixed (Matrix3b *value_ptr = &value)
            {
                Opt = new((Const_Matrix3b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix3b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix3b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3b`/`Const_Matrix3b` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3b`.
    public class _InOptConst_Matrix3b
    {
        public Const_Matrix3b? Opt;

        public _InOptConst_Matrix3b() {}
        public _InOptConst_Matrix3b(Const_Matrix3b value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix3b(Const_Matrix3b value) {return new(value);}
        public unsafe _InOptConst_Matrix3b(ref readonly Matrix3b value)
        {
            fixed (Matrix3b *value_ptr = &value)
            {
                Opt = new((Const_Matrix3b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    public static partial class Matrix3_Int
    {
        /// Generated from class `MR::Matrix3<int>::QR`.
        /// This is the const half of the class.
        public class Const_QR : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_QR(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_Destroy", ExactSpelling = true)]
                extern static void __MR_Matrix3_int_QR_Destroy(_Underlying *_this);
                __MR_Matrix3_int_QR_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_QR() {Dispose(false);}

            public unsafe MR.Const_Matrix3i Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_Get_q", ExactSpelling = true)]
                    extern static MR.Const_Matrix3i._Underlying *__MR_Matrix3_int_QR_Get_q(_Underlying *_this);
                    return new(__MR_Matrix3_int_QR_Get_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Matrix3i R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_Get_r", ExactSpelling = true)]
                    extern static MR.Const_Matrix3i._Underlying *__MR_Matrix3_int_QR_Get_r(_Underlying *_this);
                    return new(__MR_Matrix3_int_QR_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_int_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<int>::QR` elementwise.
            public unsafe Const_QR(MR.Matrix3i q, MR.Matrix3i r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_ConstructFrom(MR.Matrix3i q, MR.Matrix3i r);
                _UnderlyingPtr = __MR_Matrix3_int_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<int>::QR::QR`.
            public unsafe Const_QR(MR.Matrix3_Int.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_ConstructFromAnother(MR.Matrix3_Int.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_int_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Matrix3<int>::QR`.
        /// This is the non-const half of the class.
        public class QR : Const_QR
        {
            internal unsafe QR(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Matrix3i Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_GetMutable_q", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3i._Underlying *__MR_Matrix3_int_QR_GetMutable_q(_Underlying *_this);
                    return new(__MR_Matrix3_int_QR_GetMutable_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Matrix3i R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3i._Underlying *__MR_Matrix3_int_QR_GetMutable_r(_Underlying *_this);
                    return new(__MR_Matrix3_int_QR_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_int_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<int>::QR` elementwise.
            public unsafe QR(MR.Matrix3i q, MR.Matrix3i r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_ConstructFrom(MR.Matrix3i q, MR.Matrix3i r);
                _UnderlyingPtr = __MR_Matrix3_int_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<int>::QR::QR`.
            public unsafe QR(MR.Matrix3_Int.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_ConstructFromAnother(MR.Matrix3_Int.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_int_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Matrix3<int>::QR::operator=`.
            public unsafe MR.Matrix3_Int.QR Assign(MR.Matrix3_Int.Const_QR _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int_QR_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Int.QR._Underlying *__MR_Matrix3_int_QR_AssignFromAnother(_Underlying *_this, MR.Matrix3_Int.QR._Underlying *_other);
                return new(__MR_Matrix3_int_QR_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` directly.
        public class _InOptMut_QR
        {
            public QR? Opt;

            public _InOptMut_QR() {}
            public _InOptMut_QR(QR value) {Opt = value;}
            public static implicit operator _InOptMut_QR(QR value) {return new(value);}
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` to pass it to the function.
        public class _InOptConst_QR
        {
            public Const_QR? Opt;

            public _InOptConst_QR() {}
            public _InOptConst_QR(Const_QR value) {Opt = value;}
            public static implicit operator _InOptConst_QR(Const_QR value) {return new(value);}
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3i`.
    /// This is the const reference to the struct.
    public class Const_Matrix3i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix3i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix3i UnderlyingStruct => ref *(Matrix3i *)_UnderlyingPtr;

        internal unsafe Const_Matrix3i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix3i_Destroy(_Underlying *_this);
            __MR_Matrix3i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix3i() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector3i X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector3i Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector3i Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Const_Matrix3i(Const_Matrix3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 36);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3i _ctor_result = __MR_Matrix3i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3i::Matrix3i`.
        public unsafe Const_Matrix3i(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_Construct", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_Construct(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3i _ctor_result = __MR_Matrix3i_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// Generated from method `MR::Matrix3i::zero`.
        public static MR.Matrix3i Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_zero", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_zero();
            return __MR_Matrix3i_zero();
        }

        /// Generated from method `MR::Matrix3i::identity`.
        public static MR.Matrix3i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_identity", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_identity();
            return __MR_Matrix3i_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3i::scale`.
        public static MR.Matrix3i Scale(int s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_scale_1_int", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_scale_1_int(int s);
            return __MR_Matrix3i_scale_1_int(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3i::scale`.
        public static MR.Matrix3i Scale(int sx, int sy, int sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_scale_3(int sx, int sy, int sz);
            return __MR_Matrix3i_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3i::scale`.
        public static unsafe MR.Matrix3i Scale(MR.Const_Vector3i s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_scale_1_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_scale_1_MR_Vector3i(MR.Const_Vector3i._Underlying *s);
            return __MR_Matrix3i_scale_1_MR_Vector3i(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3i::fromRows`.
        public static unsafe MR.Matrix3i FromRows(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_fromRows(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            return __MR_Matrix3i_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3i::fromColumns`.
        public static unsafe MR.Matrix3i FromColumns(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_fromColumns(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            return __MR_Matrix3i_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3i::operator[]`.
        public unsafe MR.Const_Vector3i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_Matrix3i_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix3i_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix3i::col`.
        public unsafe MR.Vector3i Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_col", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Matrix3i_col(_Underlying *_this, int i);
            return __MR_Matrix3i_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_trace", ExactSpelling = true)]
            extern static int __MR_Matrix3i_trace(_Underlying *_this);
            return __MR_Matrix3i_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_normSq", ExactSpelling = true)]
            extern static int __MR_Matrix3i_normSq(_Underlying *_this);
            return __MR_Matrix3i_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3i::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3i_norm(_Underlying *_this);
            return __MR_Matrix3i_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_det", ExactSpelling = true)]
            extern static int __MR_Matrix3i_det(_Underlying *_this);
            return __MR_Matrix3i_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3i::transposed`.
        public unsafe MR.Matrix3i Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_transposed", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_transposed(_Underlying *_this);
            return __MR_Matrix3i_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_equal_MR_Matrix3i(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix3i a, MR.Const_Matrix3i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i operator+(MR.Const_Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_add_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_add_MR_Matrix3i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i operator-(MR.Const_Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_sub_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_sub_MR_Matrix3i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(int a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_int_MR_Matrix3i(int a, MR.Const_Matrix3i._Underlying *b);
            return __MR_mul_int_MR_Matrix3i(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Const_Matrix3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3i_int(MR.Const_Matrix3i._Underlying *b, int a);
            return __MR_mul_MR_Matrix3i_int(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix3i operator/(Const_Matrix3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_div_MR_Matrix3i_int(MR.Matrix3i b, int a);
            return __MR_div_MR_Matrix3i_int(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Const_Matrix3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Matrix3i_MR_Vector3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_mul_MR_Matrix3i_MR_Vector3i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Const_Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_mul_MR_Matrix3i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix3i? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix3i)
                return this == (MR.Const_Matrix3i)other;
            return false;
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3i`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix3i : Const_Matrix3i
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix3i UnderlyingStruct => ref *(Matrix3i *)_UnderlyingPtr;

        internal unsafe Mut_Matrix3i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector3i X => ref UnderlyingStruct.X;

        public new ref MR.Vector3i Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector3i Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Matrix3i(Const_Matrix3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 36);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3i _ctor_result = __MR_Matrix3i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3i::Matrix3i`.
        public unsafe Mut_Matrix3i(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_Construct", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_Construct(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3i _ctor_result = __MR_Matrix3i_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// Generated from method `MR::Matrix3i::operator[]`.
        public unsafe new MR.Mut_Vector3i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_Matrix3i_index(_Underlying *_this, int row);
            return new(__MR_Matrix3i_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3i AddAssign(MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_add_assign_MR_Matrix3i(_Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return new(__MR_add_assign_MR_Matrix3i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3i SubAssign(MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_sub_assign_MR_Matrix3i(_Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix3i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_mul_assign_MR_Matrix3i_int(_Underlying *a, int b);
            return new(__MR_mul_assign_MR_Matrix3i_int(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_div_assign_MR_Matrix3i_int(_Underlying *a, int b);
            return new(__MR_div_assign_MR_Matrix3i_int(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 36)]
    public struct Matrix3i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix3i(Const_Matrix3i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix3i(Matrix3i other) => new(new Mut_Matrix3i((Mut_Matrix3i._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3i X;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public MR.Vector3i Y;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public MR.Vector3i Z;

        /// Generated copy constructor.
        public Matrix3i(Matrix3i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix3i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_DefaultConstruct();
            this = __MR_Matrix3i_DefaultConstruct();
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3i::Matrix3i`.
        public unsafe Matrix3i(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_Construct", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_Construct(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            this = __MR_Matrix3i_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3i::zero`.
        public static MR.Matrix3i Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_zero", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_zero();
            return __MR_Matrix3i_zero();
        }

        /// Generated from method `MR::Matrix3i::identity`.
        public static MR.Matrix3i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_identity", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_identity();
            return __MR_Matrix3i_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3i::scale`.
        public static MR.Matrix3i Scale(int s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_scale_1_int", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_scale_1_int(int s);
            return __MR_Matrix3i_scale_1_int(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3i::scale`.
        public static MR.Matrix3i Scale(int sx, int sy, int sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_scale_3(int sx, int sy, int sz);
            return __MR_Matrix3i_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3i::scale`.
        public static unsafe MR.Matrix3i Scale(MR.Const_Vector3i s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_scale_1_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_scale_1_MR_Vector3i(MR.Const_Vector3i._Underlying *s);
            return __MR_Matrix3i_scale_1_MR_Vector3i(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3i::fromRows`.
        public static unsafe MR.Matrix3i FromRows(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_fromRows(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            return __MR_Matrix3i_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3i::fromColumns`.
        public static unsafe MR.Matrix3i FromColumns(MR.Const_Vector3i x, MR.Const_Vector3i y, MR.Const_Vector3i z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_fromColumns(MR.Const_Vector3i._Underlying *x, MR.Const_Vector3i._Underlying *y, MR.Const_Vector3i._Underlying *z);
            return __MR_Matrix3i_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3i::operator[]`.
        public unsafe MR.Const_Vector3i Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_Matrix3i_index_const(MR.Matrix3i *_this, int row);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return new(__MR_Matrix3i_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix3i::operator[]`.
        public unsafe MR.Mut_Vector3i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_Matrix3i_index(MR.Matrix3i *_this, int row);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return new(__MR_Matrix3i_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix3i::col`.
        public unsafe MR.Vector3i Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_col", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Matrix3i_col(MR.Matrix3i *_this, int i);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return __MR_Matrix3i_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_trace", ExactSpelling = true)]
            extern static int __MR_Matrix3i_trace(MR.Matrix3i *_this);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return __MR_Matrix3i_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_normSq", ExactSpelling = true)]
            extern static int __MR_Matrix3i_normSq(MR.Matrix3i *_this);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return __MR_Matrix3i_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix3i::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3i_norm(MR.Matrix3i *_this);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return __MR_Matrix3i_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_det", ExactSpelling = true)]
            extern static int __MR_Matrix3i_det(MR.Matrix3i *_this);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return __MR_Matrix3i_det(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3i::transposed`.
        public unsafe MR.Matrix3i Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i_transposed", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix3i_transposed(MR.Matrix3i *_this);
            fixed (MR.Matrix3i *__ptr__this = &this)
            {
                return __MR_Matrix3i_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix3i a, MR.Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_equal_MR_Matrix3i((MR.Mut_Matrix3i._Underlying *)&a, (MR.Mut_Matrix3i._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix3i a, MR.Matrix3i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i operator+(MR.Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_add_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_add_MR_Matrix3i((MR.Mut_Matrix3i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i operator-(MR.Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_sub_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_sub_MR_Matrix3i((MR.Mut_Matrix3i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(int a, MR.Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_int_MR_Matrix3i(int a, MR.Const_Matrix3i._Underlying *b);
            return __MR_mul_int_MR_Matrix3i(a, (MR.Mut_Matrix3i._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Matrix3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3i_int(MR.Const_Matrix3i._Underlying *b, int a);
            return __MR_mul_MR_Matrix3i_int((MR.Mut_Matrix3i._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix3i operator/(MR.Matrix3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_div_MR_Matrix3i_int(MR.Matrix3i b, int a);
            return __MR_div_MR_Matrix3i_int(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3i AddAssign(MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_add_assign_MR_Matrix3i(MR.Matrix3i *a, MR.Const_Matrix3i._Underlying *b);
            fixed (MR.Matrix3i *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix3i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3i SubAssign(MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_sub_assign_MR_Matrix3i(MR.Matrix3i *a, MR.Const_Matrix3i._Underlying *b);
            fixed (MR.Matrix3i *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix3i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_mul_assign_MR_Matrix3i_int(MR.Matrix3i *a, int b);
            fixed (MR.Matrix3i *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix3i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i._Underlying *__MR_div_assign_MR_Matrix3i_int(MR.Matrix3i *a, int b);
            fixed (MR.Matrix3i *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix3i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Matrix3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Matrix3i_MR_Vector3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_mul_MR_Matrix3i_MR_Vector3i((MR.Mut_Matrix3i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Matrix3i a, MR.Const_Matrix3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3i(MR.Const_Matrix3i._Underlying *a, MR.Const_Matrix3i._Underlying *b);
            return __MR_mul_MR_Matrix3i((MR.Mut_Matrix3i._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix3i b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix3i)
                return this == (MR.Matrix3i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix3i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix3i`/`Const_Matrix3i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix3i
    {
        public readonly bool HasValue;
        internal readonly Matrix3i Object;
        public Matrix3i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix3i() {HasValue = false;}
        public _InOpt_Matrix3i(Matrix3i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix3i(Matrix3i new_value) {return new(new_value);}
        public _InOpt_Matrix3i(Const_Matrix3i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix3i(Const_Matrix3i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix3i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3i`/`Const_Matrix3i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3i`.
    public class _InOptMut_Matrix3i
    {
        public Mut_Matrix3i? Opt;

        public _InOptMut_Matrix3i() {}
        public _InOptMut_Matrix3i(Mut_Matrix3i value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix3i(Mut_Matrix3i value) {return new(value);}
        public unsafe _InOptMut_Matrix3i(ref Matrix3i value)
        {
            fixed (Matrix3i *value_ptr = &value)
            {
                Opt = new((Const_Matrix3i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix3i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3i`/`Const_Matrix3i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3i`.
    public class _InOptConst_Matrix3i
    {
        public Const_Matrix3i? Opt;

        public _InOptConst_Matrix3i() {}
        public _InOptConst_Matrix3i(Const_Matrix3i value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix3i(Const_Matrix3i value) {return new(value);}
        public unsafe _InOptConst_Matrix3i(ref readonly Matrix3i value)
        {
            fixed (Matrix3i *value_ptr = &value)
            {
                Opt = new((Const_Matrix3i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    public static partial class Matrix3_MRInt64T
    {
        /// Generated from class `MR::Matrix3<MR_int64_t>::QR`.
        /// This is the const half of the class.
        public class Const_QR : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_QR(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_Destroy", ExactSpelling = true)]
                extern static void __MR_Matrix3_int64_t_QR_Destroy(_Underlying *_this);
                __MR_Matrix3_int64_t_QR_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_QR() {Dispose(false);}

            public unsafe MR.Const_Matrix3i64 Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_Get_q", ExactSpelling = true)]
                    extern static MR.Const_Matrix3i64._Underlying *__MR_Matrix3_int64_t_QR_Get_q(_Underlying *_this);
                    return new(__MR_Matrix3_int64_t_QR_Get_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Matrix3i64 R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_Get_r", ExactSpelling = true)]
                    extern static MR.Const_Matrix3i64._Underlying *__MR_Matrix3_int64_t_QR_Get_r(_Underlying *_this);
                    return new(__MR_Matrix3_int64_t_QR_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_int64_t_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<MR_int64_t>::QR` elementwise.
            public unsafe Const_QR(MR.Matrix3i64 q, MR.Matrix3i64 r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_ConstructFrom(MR.Matrix3i64 q, MR.Matrix3i64 r);
                _UnderlyingPtr = __MR_Matrix3_int64_t_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<MR_int64_t>::QR::QR`.
            public unsafe Const_QR(MR.Matrix3_MRInt64T.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_ConstructFromAnother(MR.Matrix3_MRInt64T.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_int64_t_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Matrix3<MR_int64_t>::QR`.
        /// This is the non-const half of the class.
        public class QR : Const_QR
        {
            internal unsafe QR(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Matrix3i64 Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_GetMutable_q", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3i64._Underlying *__MR_Matrix3_int64_t_QR_GetMutable_q(_Underlying *_this);
                    return new(__MR_Matrix3_int64_t_QR_GetMutable_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Matrix3i64 R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3i64._Underlying *__MR_Matrix3_int64_t_QR_GetMutable_r(_Underlying *_this);
                    return new(__MR_Matrix3_int64_t_QR_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_int64_t_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<MR_int64_t>::QR` elementwise.
            public unsafe QR(MR.Matrix3i64 q, MR.Matrix3i64 r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_ConstructFrom(MR.Matrix3i64 q, MR.Matrix3i64 r);
                _UnderlyingPtr = __MR_Matrix3_int64_t_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<MR_int64_t>::QR::QR`.
            public unsafe QR(MR.Matrix3_MRInt64T.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_ConstructFromAnother(MR.Matrix3_MRInt64T.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_int64_t_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Matrix3<MR_int64_t>::QR::operator=`.
            public unsafe MR.Matrix3_MRInt64T.QR Assign(MR.Matrix3_MRInt64T.Const_QR _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_int64_t_QR_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_MRInt64T.QR._Underlying *__MR_Matrix3_int64_t_QR_AssignFromAnother(_Underlying *_this, MR.Matrix3_MRInt64T.QR._Underlying *_other);
                return new(__MR_Matrix3_int64_t_QR_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` directly.
        public class _InOptMut_QR
        {
            public QR? Opt;

            public _InOptMut_QR() {}
            public _InOptMut_QR(QR value) {Opt = value;}
            public static implicit operator _InOptMut_QR(QR value) {return new(value);}
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` to pass it to the function.
        public class _InOptConst_QR
        {
            public Const_QR? Opt;

            public _InOptConst_QR() {}
            public _InOptConst_QR(Const_QR value) {Opt = value;}
            public static implicit operator _InOptConst_QR(Const_QR value) {return new(value);}
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3i64`.
    /// This is the const reference to the struct.
    public class Const_Matrix3i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix3i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix3i64 UnderlyingStruct => ref *(Matrix3i64 *)_UnderlyingPtr;

        internal unsafe Const_Matrix3i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix3i64_Destroy(_Underlying *_this);
            __MR_Matrix3i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix3i64() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector3i64 X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector3i64 Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector3i64 Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Const_Matrix3i64(Const_Matrix3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 72);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3i64 _ctor_result = __MR_Matrix3i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3i64::Matrix3i64`.
        public unsafe Const_Matrix3i64(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_Construct", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_Construct(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3i64 _ctor_result = __MR_Matrix3i64_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// Generated from method `MR::Matrix3i64::zero`.
        public static MR.Matrix3i64 Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_zero", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_zero();
            return __MR_Matrix3i64_zero();
        }

        /// Generated from method `MR::Matrix3i64::identity`.
        public static MR.Matrix3i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_identity", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_identity();
            return __MR_Matrix3i64_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3i64::scale`.
        public static MR.Matrix3i64 Scale(long s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_scale_1_int64_t", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_scale_1_int64_t(long s);
            return __MR_Matrix3i64_scale_1_int64_t(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3i64::scale`.
        public static MR.Matrix3i64 Scale(long sx, long sy, long sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_scale_3(long sx, long sy, long sz);
            return __MR_Matrix3i64_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3i64::scale`.
        public static unsafe MR.Matrix3i64 Scale(MR.Const_Vector3i64 s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_scale_1_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_scale_1_MR_Vector3i64(MR.Const_Vector3i64._Underlying *s);
            return __MR_Matrix3i64_scale_1_MR_Vector3i64(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3i64::fromRows`.
        public static unsafe MR.Matrix3i64 FromRows(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_fromRows(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            return __MR_Matrix3i64_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3i64::fromColumns`.
        public static unsafe MR.Matrix3i64 FromColumns(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_fromColumns(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            return __MR_Matrix3i64_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3i64::operator[]`.
        public unsafe MR.Const_Vector3i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i64._Underlying *__MR_Matrix3i64_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix3i64_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix3i64::col`.
        public unsafe MR.Vector3i64 Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_col", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Matrix3i64_col(_Underlying *_this, int i);
            return __MR_Matrix3i64_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_trace", ExactSpelling = true)]
            extern static long __MR_Matrix3i64_trace(_Underlying *_this);
            return __MR_Matrix3i64_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_normSq", ExactSpelling = true)]
            extern static long __MR_Matrix3i64_normSq(_Underlying *_this);
            return __MR_Matrix3i64_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3i64::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3i64_norm(_Underlying *_this);
            return __MR_Matrix3i64_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_det", ExactSpelling = true)]
            extern static long __MR_Matrix3i64_det(_Underlying *_this);
            return __MR_Matrix3i64_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3i64::transposed`.
        public unsafe MR.Matrix3i64 Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_transposed", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_transposed(_Underlying *_this);
            return __MR_Matrix3i64_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_equal_MR_Matrix3i64(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i64 operator+(MR.Const_Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_add_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_add_MR_Matrix3i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i64 operator-(MR.Const_Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_sub_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_sub_MR_Matrix3i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i64 operator*(long a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_mul_int64_t_MR_Matrix3i64(long a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_mul_int64_t_MR_Matrix3i64(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i64 operator*(MR.Const_Matrix3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_mul_MR_Matrix3i64_int64_t(MR.Const_Matrix3i64._Underlying *b, long a);
            return __MR_mul_MR_Matrix3i64_int64_t(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix3i64 operator/(Const_Matrix3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_div_MR_Matrix3i64_int64_t(MR.Matrix3i64 b, long a);
            return __MR_div_MR_Matrix3i64_int64_t(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i64 operator*(MR.Const_Matrix3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i64_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_mul_MR_Matrix3i64_MR_Vector3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_mul_MR_Matrix3i64_MR_Vector3i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i64 operator*(MR.Const_Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_mul_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_mul_MR_Matrix3i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix3i64? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix3i64)
                return this == (MR.Const_Matrix3i64)other;
            return false;
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix3i64 : Const_Matrix3i64
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix3i64 UnderlyingStruct => ref *(Matrix3i64 *)_UnderlyingPtr;

        internal unsafe Mut_Matrix3i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector3i64 X => ref UnderlyingStruct.X;

        public new ref MR.Vector3i64 Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector3i64 Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Matrix3i64(Const_Matrix3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 72);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3i64 _ctor_result = __MR_Matrix3i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3i64::Matrix3i64`.
        public unsafe Mut_Matrix3i64(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_Construct", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_Construct(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3i64 _ctor_result = __MR_Matrix3i64_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// Generated from method `MR::Matrix3i64::operator[]`.
        public unsafe new MR.Mut_Vector3i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_Matrix3i64_index(_Underlying *_this, int row);
            return new(__MR_Matrix3i64_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3i64 AddAssign(MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_add_assign_MR_Matrix3i64(_Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return new(__MR_add_assign_MR_Matrix3i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3i64 SubAssign(MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_sub_assign_MR_Matrix3i64(_Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix3i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_mul_assign_MR_Matrix3i64_int64_t(_Underlying *a, long b);
            return new(__MR_mul_assign_MR_Matrix3i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_div_assign_MR_Matrix3i64_int64_t(_Underlying *a, long b);
            return new(__MR_div_assign_MR_Matrix3i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 72)]
    public struct Matrix3i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix3i64(Const_Matrix3i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix3i64(Matrix3i64 other) => new(new Mut_Matrix3i64((Mut_Matrix3i64._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3i64 X;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public MR.Vector3i64 Y;

        [System.Runtime.InteropServices.FieldOffset(48)]
        public MR.Vector3i64 Z;

        /// Generated copy constructor.
        public Matrix3i64(Matrix3i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix3i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_DefaultConstruct();
            this = __MR_Matrix3i64_DefaultConstruct();
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3i64::Matrix3i64`.
        public unsafe Matrix3i64(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_Construct", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_Construct(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            this = __MR_Matrix3i64_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3i64::zero`.
        public static MR.Matrix3i64 Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_zero", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_zero();
            return __MR_Matrix3i64_zero();
        }

        /// Generated from method `MR::Matrix3i64::identity`.
        public static MR.Matrix3i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_identity", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_identity();
            return __MR_Matrix3i64_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3i64::scale`.
        public static MR.Matrix3i64 Scale(long s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_scale_1_int64_t", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_scale_1_int64_t(long s);
            return __MR_Matrix3i64_scale_1_int64_t(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3i64::scale`.
        public static MR.Matrix3i64 Scale(long sx, long sy, long sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_scale_3(long sx, long sy, long sz);
            return __MR_Matrix3i64_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3i64::scale`.
        public static unsafe MR.Matrix3i64 Scale(MR.Const_Vector3i64 s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_scale_1_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_scale_1_MR_Vector3i64(MR.Const_Vector3i64._Underlying *s);
            return __MR_Matrix3i64_scale_1_MR_Vector3i64(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3i64::fromRows`.
        public static unsafe MR.Matrix3i64 FromRows(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_fromRows(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            return __MR_Matrix3i64_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3i64::fromColumns`.
        public static unsafe MR.Matrix3i64 FromColumns(MR.Const_Vector3i64 x, MR.Const_Vector3i64 y, MR.Const_Vector3i64 z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_fromColumns(MR.Const_Vector3i64._Underlying *x, MR.Const_Vector3i64._Underlying *y, MR.Const_Vector3i64._Underlying *z);
            return __MR_Matrix3i64_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3i64::operator[]`.
        public unsafe MR.Const_Vector3i64 Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i64._Underlying *__MR_Matrix3i64_index_const(MR.Matrix3i64 *_this, int row);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return new(__MR_Matrix3i64_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix3i64::operator[]`.
        public unsafe MR.Mut_Vector3i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_Matrix3i64_index(MR.Matrix3i64 *_this, int row);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return new(__MR_Matrix3i64_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix3i64::col`.
        public unsafe MR.Vector3i64 Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_col", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Matrix3i64_col(MR.Matrix3i64 *_this, int i);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return __MR_Matrix3i64_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_trace", ExactSpelling = true)]
            extern static long __MR_Matrix3i64_trace(MR.Matrix3i64 *_this);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return __MR_Matrix3i64_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_normSq", ExactSpelling = true)]
            extern static long __MR_Matrix3i64_normSq(MR.Matrix3i64 *_this);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return __MR_Matrix3i64_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix3i64::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3i64_norm(MR.Matrix3i64 *_this);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return __MR_Matrix3i64_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_det", ExactSpelling = true)]
            extern static long __MR_Matrix3i64_det(MR.Matrix3i64 *_this);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return __MR_Matrix3i64_det(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3i64::transposed`.
        public unsafe MR.Matrix3i64 Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3i64_transposed", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix3i64_transposed(MR.Matrix3i64 *_this);
            fixed (MR.Matrix3i64 *__ptr__this = &this)
            {
                return __MR_Matrix3i64_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix3i64 a, MR.Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_equal_MR_Matrix3i64((MR.Mut_Matrix3i64._Underlying *)&a, (MR.Mut_Matrix3i64._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix3i64 a, MR.Matrix3i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i64 operator+(MR.Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_add_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_add_MR_Matrix3i64((MR.Mut_Matrix3i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i64 operator-(MR.Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_sub_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_sub_MR_Matrix3i64((MR.Mut_Matrix3i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i64 operator*(long a, MR.Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_mul_int64_t_MR_Matrix3i64(long a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_mul_int64_t_MR_Matrix3i64(a, (MR.Mut_Matrix3i64._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i64 operator*(MR.Matrix3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_mul_MR_Matrix3i64_int64_t(MR.Const_Matrix3i64._Underlying *b, long a);
            return __MR_mul_MR_Matrix3i64_int64_t((MR.Mut_Matrix3i64._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix3i64 operator/(MR.Matrix3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_div_MR_Matrix3i64_int64_t(MR.Matrix3i64 b, long a);
            return __MR_div_MR_Matrix3i64_int64_t(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3i64 AddAssign(MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_add_assign_MR_Matrix3i64(MR.Matrix3i64 *a, MR.Const_Matrix3i64._Underlying *b);
            fixed (MR.Matrix3i64 *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix3i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3i64 SubAssign(MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_sub_assign_MR_Matrix3i64(MR.Matrix3i64 *a, MR.Const_Matrix3i64._Underlying *b);
            fixed (MR.Matrix3i64 *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix3i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_mul_assign_MR_Matrix3i64_int64_t(MR.Matrix3i64 *a, long b);
            fixed (MR.Matrix3i64 *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix3i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix3i64._Underlying *__MR_div_assign_MR_Matrix3i64_int64_t(MR.Matrix3i64 *a, long b);
            fixed (MR.Matrix3i64 *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix3i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i64 operator*(MR.Matrix3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i64_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_mul_MR_Matrix3i64_MR_Vector3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_mul_MR_Matrix3i64_MR_Vector3i64((MR.Mut_Matrix3i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i64 operator*(MR.Matrix3i64 a, MR.Const_Matrix3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3i64", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_mul_MR_Matrix3i64(MR.Const_Matrix3i64._Underlying *a, MR.Const_Matrix3i64._Underlying *b);
            return __MR_mul_MR_Matrix3i64((MR.Mut_Matrix3i64._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix3i64 b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix3i64)
                return this == (MR.Matrix3i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix3i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix3i64`/`Const_Matrix3i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix3i64
    {
        public readonly bool HasValue;
        internal readonly Matrix3i64 Object;
        public Matrix3i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix3i64() {HasValue = false;}
        public _InOpt_Matrix3i64(Matrix3i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix3i64(Matrix3i64 new_value) {return new(new_value);}
        public _InOpt_Matrix3i64(Const_Matrix3i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix3i64(Const_Matrix3i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix3i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3i64`/`Const_Matrix3i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3i64`.
    public class _InOptMut_Matrix3i64
    {
        public Mut_Matrix3i64? Opt;

        public _InOptMut_Matrix3i64() {}
        public _InOptMut_Matrix3i64(Mut_Matrix3i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix3i64(Mut_Matrix3i64 value) {return new(value);}
        public unsafe _InOptMut_Matrix3i64(ref Matrix3i64 value)
        {
            fixed (Matrix3i64 *value_ptr = &value)
            {
                Opt = new((Const_Matrix3i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix3i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3i64`/`Const_Matrix3i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3i64`.
    public class _InOptConst_Matrix3i64
    {
        public Const_Matrix3i64? Opt;

        public _InOptConst_Matrix3i64() {}
        public _InOptConst_Matrix3i64(Const_Matrix3i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix3i64(Const_Matrix3i64 value) {return new(value);}
        public unsafe _InOptConst_Matrix3i64(ref readonly Matrix3i64 value)
        {
            fixed (Matrix3i64 *value_ptr = &value)
            {
                Opt = new((Const_Matrix3i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    public static partial class Matrix3_Float
    {
        /// Generated from class `MR::Matrix3<float>::QR`.
        /// This is the const half of the class.
        public class Const_QR : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_QR(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_Destroy", ExactSpelling = true)]
                extern static void __MR_Matrix3_float_QR_Destroy(_Underlying *_this);
                __MR_Matrix3_float_QR_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_QR() {Dispose(false);}

            public unsafe MR.Const_Matrix3f Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_Get_q", ExactSpelling = true)]
                    extern static MR.Const_Matrix3f._Underlying *__MR_Matrix3_float_QR_Get_q(_Underlying *_this);
                    return new(__MR_Matrix3_float_QR_Get_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Matrix3f R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_Get_r", ExactSpelling = true)]
                    extern static MR.Const_Matrix3f._Underlying *__MR_Matrix3_float_QR_Get_r(_Underlying *_this);
                    return new(__MR_Matrix3_float_QR_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_float_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<float>::QR` elementwise.
            public unsafe Const_QR(MR.Matrix3f q, MR.Matrix3f r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_ConstructFrom(MR.Matrix3f q, MR.Matrix3f r);
                _UnderlyingPtr = __MR_Matrix3_float_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<float>::QR::QR`.
            public unsafe Const_QR(MR.Matrix3_Float.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_ConstructFromAnother(MR.Matrix3_Float.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_float_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Matrix3<float>::QR`.
        /// This is the non-const half of the class.
        public class QR : Const_QR
        {
            internal unsafe QR(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Matrix3f Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_GetMutable_q", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3f._Underlying *__MR_Matrix3_float_QR_GetMutable_q(_Underlying *_this);
                    return new(__MR_Matrix3_float_QR_GetMutable_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Matrix3f R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3f._Underlying *__MR_Matrix3_float_QR_GetMutable_r(_Underlying *_this);
                    return new(__MR_Matrix3_float_QR_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_float_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<float>::QR` elementwise.
            public unsafe QR(MR.Matrix3f q, MR.Matrix3f r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_ConstructFrom(MR.Matrix3f q, MR.Matrix3f r);
                _UnderlyingPtr = __MR_Matrix3_float_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<float>::QR::QR`.
            public unsafe QR(MR.Matrix3_Float.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_ConstructFromAnother(MR.Matrix3_Float.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_float_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Matrix3<float>::QR::operator=`.
            public unsafe MR.Matrix3_Float.QR Assign(MR.Matrix3_Float.Const_QR _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_float_QR_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3_float_QR_AssignFromAnother(_Underlying *_this, MR.Matrix3_Float.QR._Underlying *_other);
                return new(__MR_Matrix3_float_QR_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` directly.
        public class _InOptMut_QR
        {
            public QR? Opt;

            public _InOptMut_QR() {}
            public _InOptMut_QR(QR value) {Opt = value;}
            public static implicit operator _InOptMut_QR(QR value) {return new(value);}
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` to pass it to the function.
        public class _InOptConst_QR
        {
            public Const_QR? Opt;

            public _InOptConst_QR() {}
            public _InOptConst_QR(Const_QR value) {Opt = value;}
            public static implicit operator _InOptConst_QR(Const_QR value) {return new(value);}
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3f`.
    /// This is the const reference to the struct.
    public class Const_Matrix3f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix3f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix3f UnderlyingStruct => ref *(Matrix3f *)_UnderlyingPtr;

        internal unsafe Const_Matrix3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix3f_Destroy(_Underlying *_this);
            __MR_Matrix3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix3f() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector3f X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector3f Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector3f Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Const_Matrix3f(Const_Matrix3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 36);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3f _ctor_result = __MR_Matrix3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3f::Matrix3f`.
        public unsafe Const_Matrix3f(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Construct", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_Construct(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3f _ctor_result = __MR_Matrix3f_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Matrix3f::Matrix3f`.
        public unsafe Const_Matrix3f(MR.Const_Matrix3d m) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Construct_double", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_Construct_double(MR.Const_Matrix3d._Underlying *m);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3f _ctor_result = __MR_Matrix3f_Construct_double(m._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// Generated from method `MR::Matrix3f::zero`.
        public static MR.Matrix3f Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_zero", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_zero();
            return __MR_Matrix3f_zero();
        }

        /// Generated from method `MR::Matrix3f::identity`.
        public static MR.Matrix3f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_identity", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_identity();
            return __MR_Matrix3f_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3f::scale`.
        public static MR.Matrix3f Scale(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_scale_1_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_scale_1_float(float s);
            return __MR_Matrix3f_scale_1_float(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3f::scale`.
        public static MR.Matrix3f Scale(float sx, float sy, float sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_scale_3(float sx, float sy, float sz);
            return __MR_Matrix3f_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3f::scale`.
        public static unsafe MR.Matrix3f Scale(MR.Const_Vector3f s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_scale_1_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_scale_1_MR_Vector3f(MR.Const_Vector3f._Underlying *s);
            return __MR_Matrix3f_scale_1_MR_Vector3f(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around given axis on given angle
        /// Generated from method `MR::Matrix3f::rotation`.
        public static unsafe MR.Matrix3f Rotation(MR.Const_Vector3f axis, float angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_rotation_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_rotation_float(MR.Const_Vector3f._Underlying *axis, float angle);
            return __MR_Matrix3f_rotation_float(axis._UnderlyingPtr, angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix3f::rotation`.
        public static unsafe MR.Matrix3f Rotation(MR.Const_Vector3f from, MR.Const_Vector3f to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_rotation_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_rotation_MR_Vector3f(MR.Const_Vector3f._Underlying *from, MR.Const_Vector3f._Underlying *to);
            return __MR_Matrix3f_rotation_MR_Vector3f(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)
        /// see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
        /// Generated from method `MR::Matrix3f::rotationFromEuler`.
        public static unsafe MR.Matrix3f RotationFromEuler(MR.Const_Vector3f eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_rotationFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_rotationFromEuler(MR.Const_Vector3f._Underlying *eulerAngles);
            return __MR_Matrix3f_rotationFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// returns linear by angles approximation of the rotation matrix, which is close to true rotation matrix for small angles
        /// Generated from method `MR::Matrix3f::approximateLinearRotationMatrixFromEuler`.
        public static unsafe MR.Matrix3f ApproximateLinearRotationMatrixFromEuler(MR.Const_Vector3f eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_approximateLinearRotationMatrixFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_approximateLinearRotationMatrixFromEuler(MR.Const_Vector3f._Underlying *eulerAngles);
            return __MR_Matrix3f_approximateLinearRotationMatrixFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3f::fromRows`.
        public static unsafe MR.Matrix3f FromRows(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_fromRows(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            return __MR_Matrix3f_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3f::fromColumns`.
        public static unsafe MR.Matrix3f FromColumns(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_fromColumns(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            return __MR_Matrix3f_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3f::operator[]`.
        public unsafe MR.Const_Vector3f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Matrix3f_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix3f_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix3f::col`.
        public unsafe MR.Vector3f Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_col", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix3f_col(_Underlying *_this, int i);
            return __MR_Matrix3f_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_trace", ExactSpelling = true)]
            extern static float __MR_Matrix3f_trace(_Underlying *_this);
            return __MR_Matrix3f_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_normSq", ExactSpelling = true)]
            extern static float __MR_Matrix3f_normSq(_Underlying *_this);
            return __MR_Matrix3f_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3f::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_norm", ExactSpelling = true)]
            extern static float __MR_Matrix3f_norm(_Underlying *_this);
            return __MR_Matrix3f_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_det", ExactSpelling = true)]
            extern static float __MR_Matrix3f_det(_Underlying *_this);
            return __MR_Matrix3f_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix3f::inverse`.
        public unsafe MR.Matrix3f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_inverse", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_inverse(_Underlying *_this);
            return __MR_Matrix3f_inverse(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3f::transposed`.
        public unsafe MR.Matrix3f Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_transposed", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_transposed(_Underlying *_this);
            return __MR_Matrix3f_transposed(_UnderlyingPtr);
        }

        /// returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)
        /// Generated from method `MR::Matrix3f::toEulerAngles`.
        public unsafe MR.Vector3f ToEulerAngles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_toEulerAngles", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix3f_toEulerAngles(_Underlying *_this);
            return __MR_Matrix3f_toEulerAngles(_UnderlyingPtr);
        }

        /// decompose this matrix on the product Q*R, where Q is orthogonal and R is upper triangular
        /// Generated from method `MR::Matrix3f::qr`.
        public unsafe MR.Matrix3_Float.QR Qr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_qr", ExactSpelling = true)]
            extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3f_qr(_Underlying *_this);
            return new(__MR_Matrix3f_qr(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_equal_MR_Matrix3f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix3f a, MR.Const_Matrix3f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3f operator+(MR.Const_Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_add_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_add_MR_Matrix3f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3f operator-(MR.Const_Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_sub_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_sub_MR_Matrix3f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3f operator*(float a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_mul_float_MR_Matrix3f(float a, MR.Const_Matrix3f._Underlying *b);
            return __MR_mul_float_MR_Matrix3f(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3f operator*(MR.Const_Matrix3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_mul_MR_Matrix3f_float(MR.Const_Matrix3f._Underlying *b, float a);
            return __MR_mul_MR_Matrix3f_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix3f operator/(Const_Matrix3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_div_MR_Matrix3f_float(MR.Matrix3f b, float a);
            return __MR_div_MR_Matrix3f_float(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3f operator*(MR.Const_Matrix3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3f_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_mul_MR_Matrix3f_MR_Vector3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_mul_MR_Matrix3f_MR_Vector3f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3f operator*(MR.Const_Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_mul_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_mul_MR_Matrix3f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix3f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix3f)
                return this == (MR.Const_Matrix3f)other;
            return false;
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3f`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix3f : Const_Matrix3f
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix3f UnderlyingStruct => ref *(Matrix3f *)_UnderlyingPtr;

        internal unsafe Mut_Matrix3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector3f X => ref UnderlyingStruct.X;

        public new ref MR.Vector3f Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector3f Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Matrix3f(Const_Matrix3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 36);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3f _ctor_result = __MR_Matrix3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3f::Matrix3f`.
        public unsafe Mut_Matrix3f(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Construct", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_Construct(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3f _ctor_result = __MR_Matrix3f_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Matrix3f::Matrix3f`.
        public unsafe Mut_Matrix3f(MR.Const_Matrix3d m) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Construct_double", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_Construct_double(MR.Const_Matrix3d._Underlying *m);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(36);
            MR.Matrix3f _ctor_result = __MR_Matrix3f_Construct_double(m._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 36);
        }

        /// Generated from method `MR::Matrix3f::operator[]`.
        public unsafe new MR.Mut_Vector3f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Matrix3f_index(_Underlying *_this, int row);
            return new(__MR_Matrix3f_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3f AddAssign(MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_add_assign_MR_Matrix3f(_Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return new(__MR_add_assign_MR_Matrix3f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3f SubAssign(MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_sub_assign_MR_Matrix3f(_Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix3f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_mul_assign_MR_Matrix3f_float(_Underlying *a, float b);
            return new(__MR_mul_assign_MR_Matrix3f_float(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_div_assign_MR_Matrix3f_float(_Underlying *a, float b);
            return new(__MR_div_assign_MR_Matrix3f_float(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 36)]
    public struct Matrix3f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix3f(Const_Matrix3f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix3f(Matrix3f other) => new(new Mut_Matrix3f((Mut_Matrix3f._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3f X;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public MR.Vector3f Y;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public MR.Vector3f Z;

        /// Generated copy constructor.
        public Matrix3f(Matrix3f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix3f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_DefaultConstruct();
            this = __MR_Matrix3f_DefaultConstruct();
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3f::Matrix3f`.
        public unsafe Matrix3f(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Construct", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_Construct(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            this = __MR_Matrix3f_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::Matrix3f::Matrix3f`.
        public unsafe Matrix3f(MR.Const_Matrix3d m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_Construct_double", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_Construct_double(MR.Const_Matrix3d._Underlying *m);
            this = __MR_Matrix3f_Construct_double(m._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3f::zero`.
        public static MR.Matrix3f Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_zero", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_zero();
            return __MR_Matrix3f_zero();
        }

        /// Generated from method `MR::Matrix3f::identity`.
        public static MR.Matrix3f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_identity", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_identity();
            return __MR_Matrix3f_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3f::scale`.
        public static MR.Matrix3f Scale(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_scale_1_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_scale_1_float(float s);
            return __MR_Matrix3f_scale_1_float(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3f::scale`.
        public static MR.Matrix3f Scale(float sx, float sy, float sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_scale_3(float sx, float sy, float sz);
            return __MR_Matrix3f_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3f::scale`.
        public static unsafe MR.Matrix3f Scale(MR.Const_Vector3f s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_scale_1_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_scale_1_MR_Vector3f(MR.Const_Vector3f._Underlying *s);
            return __MR_Matrix3f_scale_1_MR_Vector3f(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around given axis on given angle
        /// Generated from method `MR::Matrix3f::rotation`.
        public static unsafe MR.Matrix3f Rotation(MR.Const_Vector3f axis, float angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_rotation_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_rotation_float(MR.Const_Vector3f._Underlying *axis, float angle);
            return __MR_Matrix3f_rotation_float(axis._UnderlyingPtr, angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix3f::rotation`.
        public static unsafe MR.Matrix3f Rotation(MR.Const_Vector3f from, MR.Const_Vector3f to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_rotation_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_rotation_MR_Vector3f(MR.Const_Vector3f._Underlying *from, MR.Const_Vector3f._Underlying *to);
            return __MR_Matrix3f_rotation_MR_Vector3f(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)
        /// see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
        /// Generated from method `MR::Matrix3f::rotationFromEuler`.
        public static unsafe MR.Matrix3f RotationFromEuler(MR.Const_Vector3f eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_rotationFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_rotationFromEuler(MR.Const_Vector3f._Underlying *eulerAngles);
            return __MR_Matrix3f_rotationFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// returns linear by angles approximation of the rotation matrix, which is close to true rotation matrix for small angles
        /// Generated from method `MR::Matrix3f::approximateLinearRotationMatrixFromEuler`.
        public static unsafe MR.Matrix3f ApproximateLinearRotationMatrixFromEuler(MR.Const_Vector3f eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_approximateLinearRotationMatrixFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_approximateLinearRotationMatrixFromEuler(MR.Const_Vector3f._Underlying *eulerAngles);
            return __MR_Matrix3f_approximateLinearRotationMatrixFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3f::fromRows`.
        public static unsafe MR.Matrix3f FromRows(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_fromRows(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            return __MR_Matrix3f_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3f::fromColumns`.
        public static unsafe MR.Matrix3f FromColumns(MR.Const_Vector3f x, MR.Const_Vector3f y, MR.Const_Vector3f z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_fromColumns(MR.Const_Vector3f._Underlying *x, MR.Const_Vector3f._Underlying *y, MR.Const_Vector3f._Underlying *z);
            return __MR_Matrix3f_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3f::operator[]`.
        public unsafe MR.Const_Vector3f Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Matrix3f_index_const(MR.Matrix3f *_this, int row);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return new(__MR_Matrix3f_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix3f::operator[]`.
        public unsafe MR.Mut_Vector3f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Matrix3f_index(MR.Matrix3f *_this, int row);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return new(__MR_Matrix3f_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix3f::col`.
        public unsafe MR.Vector3f Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_col", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix3f_col(MR.Matrix3f *_this, int i);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_trace", ExactSpelling = true)]
            extern static float __MR_Matrix3f_trace(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_normSq", ExactSpelling = true)]
            extern static float __MR_Matrix3f_normSq(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix3f::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_norm", ExactSpelling = true)]
            extern static float __MR_Matrix3f_norm(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_det", ExactSpelling = true)]
            extern static float __MR_Matrix3f_det(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_det(__ptr__this);
            }
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix3f::inverse`.
        public unsafe MR.Matrix3f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_inverse", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_inverse(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_inverse(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3f::transposed`.
        public unsafe MR.Matrix3f Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_transposed", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix3f_transposed(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_transposed(__ptr__this);
            }
        }

        /// returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)
        /// Generated from method `MR::Matrix3f::toEulerAngles`.
        public unsafe MR.Vector3f ToEulerAngles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_toEulerAngles", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix3f_toEulerAngles(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return __MR_Matrix3f_toEulerAngles(__ptr__this);
            }
        }

        /// decompose this matrix on the product Q*R, where Q is orthogonal and R is upper triangular
        /// Generated from method `MR::Matrix3f::qr`.
        public unsafe MR.Matrix3_Float.QR Qr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3f_qr", ExactSpelling = true)]
            extern static MR.Matrix3_Float.QR._Underlying *__MR_Matrix3f_qr(MR.Matrix3f *_this);
            fixed (MR.Matrix3f *__ptr__this = &this)
            {
                return new(__MR_Matrix3f_qr(__ptr__this), is_owning: true);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix3f a, MR.Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_equal_MR_Matrix3f((MR.Mut_Matrix3f._Underlying *)&a, (MR.Mut_Matrix3f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix3f a, MR.Matrix3f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3f operator+(MR.Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_add_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_add_MR_Matrix3f((MR.Mut_Matrix3f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3f operator-(MR.Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_sub_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_sub_MR_Matrix3f((MR.Mut_Matrix3f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3f operator*(float a, MR.Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_mul_float_MR_Matrix3f(float a, MR.Const_Matrix3f._Underlying *b);
            return __MR_mul_float_MR_Matrix3f(a, (MR.Mut_Matrix3f._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3f operator*(MR.Matrix3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_mul_MR_Matrix3f_float(MR.Const_Matrix3f._Underlying *b, float a);
            return __MR_mul_MR_Matrix3f_float((MR.Mut_Matrix3f._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix3f operator/(MR.Matrix3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_div_MR_Matrix3f_float(MR.Matrix3f b, float a);
            return __MR_div_MR_Matrix3f_float(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3f AddAssign(MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_add_assign_MR_Matrix3f(MR.Matrix3f *a, MR.Const_Matrix3f._Underlying *b);
            fixed (MR.Matrix3f *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix3f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3f SubAssign(MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_sub_assign_MR_Matrix3f(MR.Matrix3f *a, MR.Const_Matrix3f._Underlying *b);
            fixed (MR.Matrix3f *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix3f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_mul_assign_MR_Matrix3f_float(MR.Matrix3f *a, float b);
            fixed (MR.Matrix3f *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix3f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix3f._Underlying *__MR_div_assign_MR_Matrix3f_float(MR.Matrix3f *a, float b);
            fixed (MR.Matrix3f *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix3f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3f operator*(MR.Matrix3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3f_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_mul_MR_Matrix3f_MR_Vector3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_mul_MR_Matrix3f_MR_Vector3f((MR.Mut_Matrix3f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3f operator*(MR.Matrix3f a, MR.Const_Matrix3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_mul_MR_Matrix3f(MR.Const_Matrix3f._Underlying *a, MR.Const_Matrix3f._Underlying *b);
            return __MR_mul_MR_Matrix3f((MR.Mut_Matrix3f._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix3f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix3f)
                return this == (MR.Matrix3f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix3f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix3f`/`Const_Matrix3f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix3f
    {
        public readonly bool HasValue;
        internal readonly Matrix3f Object;
        public Matrix3f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix3f() {HasValue = false;}
        public _InOpt_Matrix3f(Matrix3f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix3f(Matrix3f new_value) {return new(new_value);}
        public _InOpt_Matrix3f(Const_Matrix3f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix3f(Const_Matrix3f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3f`/`Const_Matrix3f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3f`.
    public class _InOptMut_Matrix3f
    {
        public Mut_Matrix3f? Opt;

        public _InOptMut_Matrix3f() {}
        public _InOptMut_Matrix3f(Mut_Matrix3f value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix3f(Mut_Matrix3f value) {return new(value);}
        public unsafe _InOptMut_Matrix3f(ref Matrix3f value)
        {
            fixed (Matrix3f *value_ptr = &value)
            {
                Opt = new((Const_Matrix3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3f`/`Const_Matrix3f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3f`.
    public class _InOptConst_Matrix3f
    {
        public Const_Matrix3f? Opt;

        public _InOptConst_Matrix3f() {}
        public _InOptConst_Matrix3f(Const_Matrix3f value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix3f(Const_Matrix3f value) {return new(value);}
        public unsafe _InOptConst_Matrix3f(ref readonly Matrix3f value)
        {
            fixed (Matrix3f *value_ptr = &value)
            {
                Opt = new((Const_Matrix3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    public static partial class Matrix3_Double
    {
        /// Generated from class `MR::Matrix3<double>::QR`.
        /// This is the const half of the class.
        public class Const_QR : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_QR(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_Destroy", ExactSpelling = true)]
                extern static void __MR_Matrix3_double_QR_Destroy(_Underlying *_this);
                __MR_Matrix3_double_QR_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_QR() {Dispose(false);}

            public unsafe MR.Const_Matrix3d Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_Get_q", ExactSpelling = true)]
                    extern static MR.Const_Matrix3d._Underlying *__MR_Matrix3_double_QR_Get_q(_Underlying *_this);
                    return new(__MR_Matrix3_double_QR_Get_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Matrix3d R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_Get_r", ExactSpelling = true)]
                    extern static MR.Const_Matrix3d._Underlying *__MR_Matrix3_double_QR_Get_r(_Underlying *_this);
                    return new(__MR_Matrix3_double_QR_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_double_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<double>::QR` elementwise.
            public unsafe Const_QR(MR.Matrix3d q, MR.Matrix3d r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_ConstructFrom(MR.Matrix3d q, MR.Matrix3d r);
                _UnderlyingPtr = __MR_Matrix3_double_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<double>::QR::QR`.
            public unsafe Const_QR(MR.Matrix3_Double.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_ConstructFromAnother(MR.Matrix3_Double.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_double_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Matrix3<double>::QR`.
        /// This is the non-const half of the class.
        public class QR : Const_QR
        {
            internal unsafe QR(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Matrix3d Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_GetMutable_q", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3d._Underlying *__MR_Matrix3_double_QR_GetMutable_q(_Underlying *_this);
                    return new(__MR_Matrix3_double_QR_GetMutable_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Matrix3d R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Mut_Matrix3d._Underlying *__MR_Matrix3_double_QR_GetMutable_r(_Underlying *_this);
                    return new(__MR_Matrix3_double_QR_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_double_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<double>::QR` elementwise.
            public unsafe QR(MR.Matrix3d q, MR.Matrix3d r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_ConstructFrom(MR.Matrix3d q, MR.Matrix3d r);
                _UnderlyingPtr = __MR_Matrix3_double_QR_ConstructFrom(q, r);
            }

            /// Generated from constructor `MR::Matrix3<double>::QR::QR`.
            public unsafe QR(MR.Matrix3_Double.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_ConstructFromAnother(MR.Matrix3_Double.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_double_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Matrix3<double>::QR::operator=`.
            public unsafe MR.Matrix3_Double.QR Assign(MR.Matrix3_Double.Const_QR _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_double_QR_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3_double_QR_AssignFromAnother(_Underlying *_this, MR.Matrix3_Double.QR._Underlying *_other);
                return new(__MR_Matrix3_double_QR_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` directly.
        public class _InOptMut_QR
        {
            public QR? Opt;

            public _InOptMut_QR() {}
            public _InOptMut_QR(QR value) {Opt = value;}
            public static implicit operator _InOptMut_QR(QR value) {return new(value);}
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` to pass it to the function.
        public class _InOptConst_QR
        {
            public Const_QR? Opt;

            public _InOptConst_QR() {}
            public _InOptConst_QR(Const_QR value) {Opt = value;}
            public static implicit operator _InOptConst_QR(Const_QR value) {return new(value);}
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3d`.
    /// This is the const reference to the struct.
    public class Const_Matrix3d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix3d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix3d UnderlyingStruct => ref *(Matrix3d *)_UnderlyingPtr;

        internal unsafe Const_Matrix3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix3d_Destroy(_Underlying *_this);
            __MR_Matrix3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix3d() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector3d X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector3d Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector3d Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Const_Matrix3d(Const_Matrix3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 72);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3d _ctor_result = __MR_Matrix3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3d::Matrix3d`.
        public unsafe Const_Matrix3d(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_Construct", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_Construct(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3d _ctor_result = __MR_Matrix3d_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// Generated from method `MR::Matrix3d::zero`.
        public static MR.Matrix3d Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_zero", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_zero();
            return __MR_Matrix3d_zero();
        }

        /// Generated from method `MR::Matrix3d::identity`.
        public static MR.Matrix3d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_identity", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_identity();
            return __MR_Matrix3d_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3d::scale`.
        public static MR.Matrix3d Scale(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_scale_1_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_scale_1_double(double s);
            return __MR_Matrix3d_scale_1_double(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3d::scale`.
        public static MR.Matrix3d Scale(double sx, double sy, double sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_scale_3(double sx, double sy, double sz);
            return __MR_Matrix3d_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3d::scale`.
        public static unsafe MR.Matrix3d Scale(MR.Const_Vector3d s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_scale_1_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_scale_1_MR_Vector3d(MR.Const_Vector3d._Underlying *s);
            return __MR_Matrix3d_scale_1_MR_Vector3d(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around given axis on given angle
        /// Generated from method `MR::Matrix3d::rotation`.
        public static unsafe MR.Matrix3d Rotation(MR.Const_Vector3d axis, double angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_rotation_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_rotation_double(MR.Const_Vector3d._Underlying *axis, double angle);
            return __MR_Matrix3d_rotation_double(axis._UnderlyingPtr, angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix3d::rotation`.
        public static unsafe MR.Matrix3d Rotation(MR.Const_Vector3d from, MR.Const_Vector3d to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_rotation_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_rotation_MR_Vector3d(MR.Const_Vector3d._Underlying *from, MR.Const_Vector3d._Underlying *to);
            return __MR_Matrix3d_rotation_MR_Vector3d(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)
        /// see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
        /// Generated from method `MR::Matrix3d::rotationFromEuler`.
        public static unsafe MR.Matrix3d RotationFromEuler(MR.Const_Vector3d eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_rotationFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_rotationFromEuler(MR.Const_Vector3d._Underlying *eulerAngles);
            return __MR_Matrix3d_rotationFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// returns linear by angles approximation of the rotation matrix, which is close to true rotation matrix for small angles
        /// Generated from method `MR::Matrix3d::approximateLinearRotationMatrixFromEuler`.
        public static unsafe MR.Matrix3d ApproximateLinearRotationMatrixFromEuler(MR.Const_Vector3d eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_approximateLinearRotationMatrixFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_approximateLinearRotationMatrixFromEuler(MR.Const_Vector3d._Underlying *eulerAngles);
            return __MR_Matrix3d_approximateLinearRotationMatrixFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3d::fromRows`.
        public static unsafe MR.Matrix3d FromRows(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_fromRows(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            return __MR_Matrix3d_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3d::fromColumns`.
        public static unsafe MR.Matrix3d FromColumns(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_fromColumns(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            return __MR_Matrix3d_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3d::operator[]`.
        public unsafe MR.Const_Vector3d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Matrix3d_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix3d_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix3d::col`.
        public unsafe MR.Vector3d Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_col", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix3d_col(_Underlying *_this, int i);
            return __MR_Matrix3d_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_trace", ExactSpelling = true)]
            extern static double __MR_Matrix3d_trace(_Underlying *_this);
            return __MR_Matrix3d_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_normSq", ExactSpelling = true)]
            extern static double __MR_Matrix3d_normSq(_Underlying *_this);
            return __MR_Matrix3d_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3d::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3d_norm(_Underlying *_this);
            return __MR_Matrix3d_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_det", ExactSpelling = true)]
            extern static double __MR_Matrix3d_det(_Underlying *_this);
            return __MR_Matrix3d_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix3d::inverse`.
        public unsafe MR.Matrix3d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_inverse", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_inverse(_Underlying *_this);
            return __MR_Matrix3d_inverse(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3d::transposed`.
        public unsafe MR.Matrix3d Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_transposed", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_transposed(_Underlying *_this);
            return __MR_Matrix3d_transposed(_UnderlyingPtr);
        }

        /// returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)
        /// Generated from method `MR::Matrix3d::toEulerAngles`.
        public unsafe MR.Vector3d ToEulerAngles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_toEulerAngles", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix3d_toEulerAngles(_Underlying *_this);
            return __MR_Matrix3d_toEulerAngles(_UnderlyingPtr);
        }

        /// decompose this matrix on the product Q*R, where Q is orthogonal and R is upper triangular
        /// Generated from method `MR::Matrix3d::qr`.
        public unsafe MR.Matrix3_Double.QR Qr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_qr", ExactSpelling = true)]
            extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3d_qr(_Underlying *_this);
            return new(__MR_Matrix3d_qr(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_equal_MR_Matrix3d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix3d a, MR.Const_Matrix3d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3d operator+(MR.Const_Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_add_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_add_MR_Matrix3d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3d operator-(MR.Const_Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_sub_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_sub_MR_Matrix3d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3d operator*(double a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_mul_double_MR_Matrix3d(double a, MR.Const_Matrix3d._Underlying *b);
            return __MR_mul_double_MR_Matrix3d(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3d operator*(MR.Const_Matrix3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_mul_MR_Matrix3d_double(MR.Const_Matrix3d._Underlying *b, double a);
            return __MR_mul_MR_Matrix3d_double(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix3d operator/(Const_Matrix3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_div_MR_Matrix3d_double(MR.Matrix3d b, double a);
            return __MR_div_MR_Matrix3d_double(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3d operator*(MR.Const_Matrix3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3d_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_mul_MR_Matrix3d_MR_Vector3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_mul_MR_Matrix3d_MR_Vector3d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3d operator*(MR.Const_Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_mul_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_mul_MR_Matrix3d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix3d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix3d)
                return this == (MR.Const_Matrix3d)other;
            return false;
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3d`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix3d : Const_Matrix3d
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix3d UnderlyingStruct => ref *(Matrix3d *)_UnderlyingPtr;

        internal unsafe Mut_Matrix3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector3d X => ref UnderlyingStruct.X;

        public new ref MR.Vector3d Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector3d Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Matrix3d(Const_Matrix3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 72);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3d _ctor_result = __MR_Matrix3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3d::Matrix3d`.
        public unsafe Mut_Matrix3d(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_Construct", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_Construct(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(72);
            MR.Matrix3d _ctor_result = __MR_Matrix3d_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 72);
        }

        /// Generated from method `MR::Matrix3d::operator[]`.
        public unsafe new MR.Mut_Vector3d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Matrix3d_index(_Underlying *_this, int row);
            return new(__MR_Matrix3d_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3d AddAssign(MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_add_assign_MR_Matrix3d(_Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return new(__MR_add_assign_MR_Matrix3d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3d SubAssign(MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_sub_assign_MR_Matrix3d(_Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix3d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_mul_assign_MR_Matrix3d_double(_Underlying *a, double b);
            return new(__MR_mul_assign_MR_Matrix3d_double(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_div_assign_MR_Matrix3d_double(_Underlying *a, double b);
            return new(__MR_div_assign_MR_Matrix3d_double(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 72)]
    public struct Matrix3d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix3d(Const_Matrix3d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix3d(Matrix3d other) => new(new Mut_Matrix3d((Mut_Matrix3d._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3d X;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public MR.Vector3d Y;

        [System.Runtime.InteropServices.FieldOffset(48)]
        public MR.Vector3d Z;

        /// Generated copy constructor.
        public Matrix3d(Matrix3d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_DefaultConstruct();
            this = __MR_Matrix3d_DefaultConstruct();
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3d::Matrix3d`.
        public unsafe Matrix3d(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_Construct", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_Construct(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            this = __MR_Matrix3d_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3d::zero`.
        public static MR.Matrix3d Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_zero", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_zero();
            return __MR_Matrix3d_zero();
        }

        /// Generated from method `MR::Matrix3d::identity`.
        public static MR.Matrix3d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_identity", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_identity();
            return __MR_Matrix3d_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3d::scale`.
        public static MR.Matrix3d Scale(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_scale_1_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_scale_1_double(double s);
            return __MR_Matrix3d_scale_1_double(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3d::scale`.
        public static MR.Matrix3d Scale(double sx, double sy, double sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_scale_3(double sx, double sy, double sz);
            return __MR_Matrix3d_scale_3(sx, sy, sz);
        }

        /// Generated from method `MR::Matrix3d::scale`.
        public static unsafe MR.Matrix3d Scale(MR.Const_Vector3d s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_scale_1_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_scale_1_MR_Vector3d(MR.Const_Vector3d._Underlying *s);
            return __MR_Matrix3d_scale_1_MR_Vector3d(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around given axis on given angle
        /// Generated from method `MR::Matrix3d::rotation`.
        public static unsafe MR.Matrix3d Rotation(MR.Const_Vector3d axis, double angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_rotation_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_rotation_double(MR.Const_Vector3d._Underlying *axis, double angle);
            return __MR_Matrix3d_rotation_double(axis._UnderlyingPtr, angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix3d::rotation`.
        public static unsafe MR.Matrix3d Rotation(MR.Const_Vector3d from, MR.Const_Vector3d to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_rotation_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_rotation_MR_Vector3d(MR.Const_Vector3d._Underlying *from, MR.Const_Vector3d._Underlying *to);
            return __MR_Matrix3d_rotation_MR_Vector3d(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// creates matrix representing rotation from 3 Euler angles: R=R(z)*R(y)*R(x)
        /// see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
        /// Generated from method `MR::Matrix3d::rotationFromEuler`.
        public static unsafe MR.Matrix3d RotationFromEuler(MR.Const_Vector3d eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_rotationFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_rotationFromEuler(MR.Const_Vector3d._Underlying *eulerAngles);
            return __MR_Matrix3d_rotationFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// returns linear by angles approximation of the rotation matrix, which is close to true rotation matrix for small angles
        /// Generated from method `MR::Matrix3d::approximateLinearRotationMatrixFromEuler`.
        public static unsafe MR.Matrix3d ApproximateLinearRotationMatrixFromEuler(MR.Const_Vector3d eulerAngles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_approximateLinearRotationMatrixFromEuler", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_approximateLinearRotationMatrixFromEuler(MR.Const_Vector3d._Underlying *eulerAngles);
            return __MR_Matrix3d_approximateLinearRotationMatrixFromEuler(eulerAngles._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3d::fromRows`.
        public static unsafe MR.Matrix3d FromRows(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_fromRows(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            return __MR_Matrix3d_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3d::fromColumns`.
        public static unsafe MR.Matrix3d FromColumns(MR.Const_Vector3d x, MR.Const_Vector3d y, MR.Const_Vector3d z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_fromColumns(MR.Const_Vector3d._Underlying *x, MR.Const_Vector3d._Underlying *y, MR.Const_Vector3d._Underlying *z);
            return __MR_Matrix3d_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix3d::operator[]`.
        public unsafe MR.Const_Vector3d Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Matrix3d_index_const(MR.Matrix3d *_this, int row);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return new(__MR_Matrix3d_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix3d::operator[]`.
        public unsafe MR.Mut_Vector3d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Matrix3d_index(MR.Matrix3d *_this, int row);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return new(__MR_Matrix3d_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix3d::col`.
        public unsafe MR.Vector3d Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_col", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix3d_col(MR.Matrix3d *_this, int i);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_trace", ExactSpelling = true)]
            extern static double __MR_Matrix3d_trace(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_normSq", ExactSpelling = true)]
            extern static double __MR_Matrix3d_normSq(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix3d::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3d_norm(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_det", ExactSpelling = true)]
            extern static double __MR_Matrix3d_det(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_det(__ptr__this);
            }
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix3d::inverse`.
        public unsafe MR.Matrix3d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_inverse", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_inverse(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_inverse(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3d::transposed`.
        public unsafe MR.Matrix3d Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_transposed", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix3d_transposed(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_transposed(__ptr__this);
            }
        }

        /// returns 3 Euler angles, assuming this is a rotation matrix composed as follows: R=R(z)*R(y)*R(x)
        /// Generated from method `MR::Matrix3d::toEulerAngles`.
        public unsafe MR.Vector3d ToEulerAngles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_toEulerAngles", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix3d_toEulerAngles(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return __MR_Matrix3d_toEulerAngles(__ptr__this);
            }
        }

        /// decompose this matrix on the product Q*R, where Q is orthogonal and R is upper triangular
        /// Generated from method `MR::Matrix3d::qr`.
        public unsafe MR.Matrix3_Double.QR Qr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3d_qr", ExactSpelling = true)]
            extern static MR.Matrix3_Double.QR._Underlying *__MR_Matrix3d_qr(MR.Matrix3d *_this);
            fixed (MR.Matrix3d *__ptr__this = &this)
            {
                return new(__MR_Matrix3d_qr(__ptr__this), is_owning: true);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix3d a, MR.Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_equal_MR_Matrix3d((MR.Mut_Matrix3d._Underlying *)&a, (MR.Mut_Matrix3d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix3d a, MR.Matrix3d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3d operator+(MR.Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_add_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_add_MR_Matrix3d((MR.Mut_Matrix3d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3d operator-(MR.Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_sub_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_sub_MR_Matrix3d((MR.Mut_Matrix3d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3d operator*(double a, MR.Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_mul_double_MR_Matrix3d(double a, MR.Const_Matrix3d._Underlying *b);
            return __MR_mul_double_MR_Matrix3d(a, (MR.Mut_Matrix3d._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3d operator*(MR.Matrix3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_mul_MR_Matrix3d_double(MR.Const_Matrix3d._Underlying *b, double a);
            return __MR_mul_MR_Matrix3d_double((MR.Mut_Matrix3d._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix3d operator/(MR.Matrix3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_div_MR_Matrix3d_double(MR.Matrix3d b, double a);
            return __MR_div_MR_Matrix3d_double(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix3d AddAssign(MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_add_assign_MR_Matrix3d(MR.Matrix3d *a, MR.Const_Matrix3d._Underlying *b);
            fixed (MR.Matrix3d *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix3d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix3d SubAssign(MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_sub_assign_MR_Matrix3d(MR.Matrix3d *a, MR.Const_Matrix3d._Underlying *b);
            fixed (MR.Matrix3d *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix3d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix3d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_mul_assign_MR_Matrix3d_double(MR.Matrix3d *a, double b);
            fixed (MR.Matrix3d *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix3d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix3d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix3d._Underlying *__MR_div_assign_MR_Matrix3d_double(MR.Matrix3d *a, double b);
            fixed (MR.Matrix3d *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix3d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3d operator*(MR.Matrix3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3d_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_mul_MR_Matrix3d_MR_Vector3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_mul_MR_Matrix3d_MR_Vector3d((MR.Mut_Matrix3d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3d operator*(MR.Matrix3d a, MR.Const_Matrix3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_mul_MR_Matrix3d(MR.Const_Matrix3d._Underlying *a, MR.Const_Matrix3d._Underlying *b);
            return __MR_mul_MR_Matrix3d((MR.Mut_Matrix3d._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix3d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix3d)
                return this == (MR.Matrix3d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix3d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix3d`/`Const_Matrix3d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix3d
    {
        public readonly bool HasValue;
        internal readonly Matrix3d Object;
        public Matrix3d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix3d() {HasValue = false;}
        public _InOpt_Matrix3d(Matrix3d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix3d(Matrix3d new_value) {return new(new_value);}
        public _InOpt_Matrix3d(Const_Matrix3d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix3d(Const_Matrix3d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3d`/`Const_Matrix3d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3d`.
    public class _InOptMut_Matrix3d
    {
        public Mut_Matrix3d? Opt;

        public _InOptMut_Matrix3d() {}
        public _InOptMut_Matrix3d(Mut_Matrix3d value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix3d(Mut_Matrix3d value) {return new(value);}
        public unsafe _InOptMut_Matrix3d(ref Matrix3d value)
        {
            fixed (Matrix3d *value_ptr = &value)
            {
                Opt = new((Const_Matrix3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix3d`/`Const_Matrix3d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix3d`.
    public class _InOptConst_Matrix3d
    {
        public Const_Matrix3d? Opt;

        public _InOptConst_Matrix3d() {}
        public _InOptConst_Matrix3d(Const_Matrix3d value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix3d(Const_Matrix3d value) {return new(value);}
        public unsafe _InOptConst_Matrix3d(ref readonly Matrix3d value)
        {
            fixed (Matrix3d *value_ptr = &value)
            {
                Opt = new((Const_Matrix3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3<unsigned char>`.
    /// This is the const half of the class.
    public class Const_Matrix3_UnsignedChar : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix3_UnsignedChar>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Matrix3_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix3_unsigned_char_Destroy(_Underlying *_this);
            __MR_Matrix3_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix3_UnsignedChar() {Dispose(false);}

        /// rows, identity matrix by default
        public unsafe MR.Const_Vector3_UnsignedChar X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_Get_x", ExactSpelling = true)]
                extern static MR.Const_Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_Get_x(_Underlying *_this);
                return new(__MR_Matrix3_unsigned_char_Get_x(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3_UnsignedChar Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_Get_y", ExactSpelling = true)]
                extern static MR.Const_Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_Get_y(_Underlying *_this);
                return new(__MR_Matrix3_unsigned_char_Get_y(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3_UnsignedChar Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_Get_z", ExactSpelling = true)]
                extern static MR.Const_Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_Get_z(_Underlying *_this);
                return new(__MR_Matrix3_unsigned_char_Get_z(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix3_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Matrix3_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Matrix3<unsigned char>::Matrix3`.
        public unsafe Const_Matrix3_UnsignedChar(MR.Const_Matrix3_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_ConstructFromAnother(MR.Matrix3_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Matrix3_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3<unsigned char>::Matrix3`.
        public unsafe Const_Matrix3_UnsignedChar(MR.Const_Vector3_UnsignedChar x, MR.Const_Vector3_UnsignedChar y, MR.Const_Vector3_UnsignedChar z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_Construct(MR.Const_Vector3_UnsignedChar._Underlying *x, MR.Const_Vector3_UnsignedChar._Underlying *y, MR.Const_Vector3_UnsignedChar._Underlying *z);
            _UnderlyingPtr = __MR_Matrix3_unsigned_char_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3<unsigned char>::zero`.
        public static unsafe MR.Matrix3_UnsignedChar Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_zero", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_zero();
            return new(__MR_Matrix3_unsigned_char_zero(), is_owning: true);
        }

        /// Generated from method `MR::Matrix3<unsigned char>::identity`.
        public static unsafe MR.Matrix3_UnsignedChar Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_identity", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_identity();
            return new(__MR_Matrix3_unsigned_char_identity(), is_owning: true);
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix3<unsigned char>::scale`.
        public static unsafe MR.Matrix3_UnsignedChar Scale(byte s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_scale_1_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_scale_1_unsigned_char(byte s);
            return new(__MR_Matrix3_unsigned_char_scale_1_unsigned_char(s), is_owning: true);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix3<unsigned char>::scale`.
        public static unsafe MR.Matrix3_UnsignedChar Scale(byte sx, byte sy, byte sz)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_scale_3", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_scale_3(byte sx, byte sy, byte sz);
            return new(__MR_Matrix3_unsigned_char_scale_3(sx, sy, sz), is_owning: true);
        }

        /// Generated from method `MR::Matrix3<unsigned char>::scale`.
        public static unsafe MR.Matrix3_UnsignedChar Scale(MR.Const_Vector3_UnsignedChar s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_scale_1_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_scale_1_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *s);
            return new(__MR_Matrix3_unsigned_char_scale_1_MR_Vector3_unsigned_char(s._UnderlyingPtr), is_owning: true);
        }

        /// constructs a matrix from its 3 rows
        /// Generated from method `MR::Matrix3<unsigned char>::fromRows`.
        public static unsafe MR.Matrix3_UnsignedChar FromRows(MR.Const_Vector3_UnsignedChar x, MR.Const_Vector3_UnsignedChar y, MR.Const_Vector3_UnsignedChar z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_fromRows", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_fromRows(MR.Const_Vector3_UnsignedChar._Underlying *x, MR.Const_Vector3_UnsignedChar._Underlying *y, MR.Const_Vector3_UnsignedChar._Underlying *z);
            return new(__MR_Matrix3_unsigned_char_fromRows(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr), is_owning: true);
        }

        /// constructs a matrix from its 3 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY, plusZ ) into vectors ( x, y, z ) respectively
        /// Generated from method `MR::Matrix3<unsigned char>::fromColumns`.
        public static unsafe MR.Matrix3_UnsignedChar FromColumns(MR.Const_Vector3_UnsignedChar x, MR.Const_Vector3_UnsignedChar y, MR.Const_Vector3_UnsignedChar z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_fromColumns(MR.Const_Vector3_UnsignedChar._Underlying *x, MR.Const_Vector3_UnsignedChar._Underlying *y, MR.Const_Vector3_UnsignedChar._Underlying *z);
            return new(__MR_Matrix3_unsigned_char_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr), is_owning: true);
        }

        /// row access
        /// Generated from method `MR::Matrix3<unsigned char>::operator[]`.
        public unsafe MR.Const_Vector3_UnsignedChar Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix3_unsigned_char_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix3<unsigned char>::col`.
        public unsafe MR.Vector3_UnsignedChar Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_col", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_col(_Underlying *_this, int i);
            return new(__MR_Matrix3_unsigned_char_col(_UnderlyingPtr, i), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix3<unsigned char>::trace`.
        public unsafe byte Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix3_unsigned_char_trace(_Underlying *_this);
            return __MR_Matrix3_unsigned_char_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix3<unsigned char>::normSq`.
        public unsafe byte NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix3_unsigned_char_normSq(_Underlying *_this);
            return __MR_Matrix3_unsigned_char_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3<unsigned char>::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_norm", ExactSpelling = true)]
            extern static double __MR_Matrix3_unsigned_char_norm(_Underlying *_this);
            return __MR_Matrix3_unsigned_char_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix3<unsigned char>::det`.
        public unsafe byte Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_det", ExactSpelling = true)]
            extern static byte __MR_Matrix3_unsigned_char_det(_Underlying *_this);
            return __MR_Matrix3_unsigned_char_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix3<unsigned char>::transposed`.
        public unsafe MR.Matrix3_UnsignedChar Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_transposed", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_transposed(_Underlying *_this);
            return new(__MR_Matrix3_unsigned_char_transposed(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix3_UnsignedChar a, MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix3_unsigned_char(MR.Const_Matrix3_UnsignedChar._Underlying *a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return __MR_equal_MR_Matrix3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix3_UnsignedChar a, MR.Const_Matrix3_UnsignedChar b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix3i operator+(MR.Const_Matrix3_UnsignedChar a, MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_add_MR_Matrix3_unsigned_char(MR.Const_Matrix3_UnsignedChar._Underlying *a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return __MR_add_MR_Matrix3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix3i operator-(MR.Const_Matrix3_UnsignedChar a, MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_sub_MR_Matrix3_unsigned_char(MR.Const_Matrix3_UnsignedChar._Underlying *a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return __MR_sub_MR_Matrix3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(byte a, MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_unsigned_char_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_unsigned_char_MR_Matrix3_unsigned_char(byte a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return __MR_mul_unsigned_char_MR_Matrix3_unsigned_char(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Const_Matrix3_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3_unsigned_char_unsigned_char(MR.Const_Matrix3_UnsignedChar._Underlying *b, byte a);
            return __MR_mul_MR_Matrix3_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix3i operator/(Const_Matrix3_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_div_MR_Matrix3_unsigned_char_unsigned_char(MR.Matrix3_UnsignedChar._Underlying *b, byte a);
            return __MR_div_MR_Matrix3_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Const_Matrix3_UnsignedChar a, MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3_unsigned_char_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Matrix3_unsigned_char_MR_Vector3_unsigned_char(MR.Const_Matrix3_UnsignedChar._Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return __MR_mul_MR_Matrix3_unsigned_char_MR_Vector3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix3i operator*(MR.Const_Matrix3_UnsignedChar a, MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_mul_MR_Matrix3_unsigned_char(MR.Const_Matrix3_UnsignedChar._Underlying *a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return __MR_mul_MR_Matrix3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from class `MR::Matrix3<unsigned char>::QR`.
        /// This is the const half of the class.
        public class Const_QR : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_QR(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_Destroy", ExactSpelling = true)]
                extern static void __MR_Matrix3_unsigned_char_QR_Destroy(_Underlying *_this);
                __MR_Matrix3_unsigned_char_QR_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_QR() {Dispose(false);}

            public unsafe MR.Const_Matrix3_UnsignedChar Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_Get_q", ExactSpelling = true)]
                    extern static MR.Const_Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_QR_Get_q(_Underlying *_this);
                    return new(__MR_Matrix3_unsigned_char_QR_Get_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Matrix3_UnsignedChar R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_Get_r", ExactSpelling = true)]
                    extern static MR.Const_Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_QR_Get_r(_Underlying *_this);
                    return new(__MR_Matrix3_unsigned_char_QR_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_unsigned_char_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<unsigned char>::QR` elementwise.
            public unsafe Const_QR(MR.Const_Matrix3_UnsignedChar q, MR.Const_Matrix3_UnsignedChar r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_ConstructFrom(MR.Matrix3_UnsignedChar._Underlying *q, MR.Matrix3_UnsignedChar._Underlying *r);
                _UnderlyingPtr = __MR_Matrix3_unsigned_char_QR_ConstructFrom(q._UnderlyingPtr, r._UnderlyingPtr);
            }

            /// Generated from constructor `MR::Matrix3<unsigned char>::QR::QR`.
            public unsafe Const_QR(MR.Matrix3_UnsignedChar.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_ConstructFromAnother(MR.Matrix3_UnsignedChar.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_unsigned_char_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Matrix3<unsigned char>::QR`.
        /// This is the non-const half of the class.
        public class QR : Const_QR
        {
            internal unsafe QR(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Matrix3_UnsignedChar Q
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_GetMutable_q", ExactSpelling = true)]
                    extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_QR_GetMutable_q(_Underlying *_this);
                    return new(__MR_Matrix3_unsigned_char_QR_GetMutable_q(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Matrix3_UnsignedChar R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_QR_GetMutable_r(_Underlying *_this);
                    return new(__MR_Matrix3_unsigned_char_QR_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe QR() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_DefaultConstruct();
                _UnderlyingPtr = __MR_Matrix3_unsigned_char_QR_DefaultConstruct();
            }

            /// Constructs `MR::Matrix3<unsigned char>::QR` elementwise.
            public unsafe QR(MR.Const_Matrix3_UnsignedChar q, MR.Const_Matrix3_UnsignedChar r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_ConstructFrom", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_ConstructFrom(MR.Matrix3_UnsignedChar._Underlying *q, MR.Matrix3_UnsignedChar._Underlying *r);
                _UnderlyingPtr = __MR_Matrix3_unsigned_char_QR_ConstructFrom(q._UnderlyingPtr, r._UnderlyingPtr);
            }

            /// Generated from constructor `MR::Matrix3<unsigned char>::QR::QR`.
            public unsafe QR(MR.Matrix3_UnsignedChar.Const_QR _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_ConstructFromAnother(MR.Matrix3_UnsignedChar.QR._Underlying *_other);
                _UnderlyingPtr = __MR_Matrix3_unsigned_char_QR_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Matrix3<unsigned char>::QR::operator=`.
            public unsafe MR.Matrix3_UnsignedChar.QR Assign(MR.Matrix3_UnsignedChar.Const_QR _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_QR_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Matrix3_UnsignedChar.QR._Underlying *__MR_Matrix3_unsigned_char_QR_AssignFromAnother(_Underlying *_this, MR.Matrix3_UnsignedChar.QR._Underlying *_other);
                return new(__MR_Matrix3_unsigned_char_QR_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` directly.
        public class _InOptMut_QR
        {
            public QR? Opt;

            public _InOptMut_QR() {}
            public _InOptMut_QR(QR value) {Opt = value;}
            public static implicit operator _InOptMut_QR(QR value) {return new(value);}
        }

        /// This is used for optional parameters of class `QR` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_QR`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `QR`/`Const_QR` to pass it to the function.
        public class _InOptConst_QR
        {
            public Const_QR? Opt;

            public _InOptConst_QR() {}
            public _InOptConst_QR(Const_QR value) {Opt = value;}
            public static implicit operator _InOptConst_QR(Const_QR value) {return new(value);}
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix3_UnsignedChar? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix3_UnsignedChar)
                return this == (MR.Const_Matrix3_UnsignedChar)other;
            return false;
        }
    }

    /// arbitrary 3x3 matrix
    /// Generated from class `MR::Matrix3<unsigned char>`.
    /// This is the non-const half of the class.
    public class Matrix3_UnsignedChar : Const_Matrix3_UnsignedChar
    {
        internal unsafe Matrix3_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new unsafe MR.Vector3_UnsignedChar X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_GetMutable_x", ExactSpelling = true)]
                extern static MR.Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_GetMutable_x(_Underlying *_this);
                return new(__MR_Matrix3_unsigned_char_GetMutable_x(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Vector3_UnsignedChar Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_GetMutable_y", ExactSpelling = true)]
                extern static MR.Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_GetMutable_y(_Underlying *_this);
                return new(__MR_Matrix3_unsigned_char_GetMutable_y(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Vector3_UnsignedChar Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_GetMutable_z", ExactSpelling = true)]
                extern static MR.Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_GetMutable_z(_Underlying *_this);
                return new(__MR_Matrix3_unsigned_char_GetMutable_z(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix3_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Matrix3_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Matrix3<unsigned char>::Matrix3`.
        public unsafe Matrix3_UnsignedChar(MR.Const_Matrix3_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_ConstructFromAnother(MR.Matrix3_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Matrix3_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// initializes matrix from its 3 rows
        /// Generated from constructor `MR::Matrix3<unsigned char>::Matrix3`.
        public unsafe Matrix3_UnsignedChar(MR.Const_Vector3_UnsignedChar x, MR.Const_Vector3_UnsignedChar y, MR.Const_Vector3_UnsignedChar z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_Construct(MR.Const_Vector3_UnsignedChar._Underlying *x, MR.Const_Vector3_UnsignedChar._Underlying *y, MR.Const_Vector3_UnsignedChar._Underlying *z);
            _UnderlyingPtr = __MR_Matrix3_unsigned_char_Construct(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix3<unsigned char>::operator=`.
        public unsafe MR.Matrix3_UnsignedChar Assign(MR.Const_Matrix3_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Matrix3_UnsignedChar._Underlying *_other);
            return new(__MR_Matrix3_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Matrix3<unsigned char>::operator[]`.
        public unsafe new MR.Vector3_UnsignedChar Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix3_unsigned_char_index", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Matrix3_unsigned_char_index(_Underlying *_this, int row);
            return new(__MR_Matrix3_unsigned_char_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Matrix3_UnsignedChar AddAssign(MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_add_assign_MR_Matrix3_unsigned_char(_Underlying *a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return new(__MR_add_assign_MR_Matrix3_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Matrix3_UnsignedChar SubAssign(MR.Const_Matrix3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix3_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_sub_assign_MR_Matrix3_unsigned_char(_Underlying *a, MR.Const_Matrix3_UnsignedChar._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix3_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Matrix3_UnsignedChar MulAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_mul_assign_MR_Matrix3_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Matrix3_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Matrix3_UnsignedChar DivAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_div_assign_MR_Matrix3_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Matrix3_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Matrix3_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix3_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Matrix3_UnsignedChar`/`Const_Matrix3_UnsignedChar` directly.
    public class _InOptMut_Matrix3_UnsignedChar
    {
        public Matrix3_UnsignedChar? Opt;

        public _InOptMut_Matrix3_UnsignedChar() {}
        public _InOptMut_Matrix3_UnsignedChar(Matrix3_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix3_UnsignedChar(Matrix3_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `Matrix3_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix3_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Matrix3_UnsignedChar`/`Const_Matrix3_UnsignedChar` to pass it to the function.
    public class _InOptConst_Matrix3_UnsignedChar
    {
        public Const_Matrix3_UnsignedChar? Opt;

        public _InOptConst_Matrix3_UnsignedChar() {}
        public _InOptConst_Matrix3_UnsignedChar(Const_Matrix3_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix3_UnsignedChar(Const_Matrix3_UnsignedChar value) {return new(value);}
    }
}
