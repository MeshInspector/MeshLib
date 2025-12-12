public static partial class MR
{
    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2b`.
    /// This is the const reference to the struct.
    public class Const_Matrix2b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix2b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix2b UnderlyingStruct => ref *(Matrix2b *)_UnderlyingPtr;

        internal unsafe Const_Matrix2b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix2b_Destroy(_Underlying *_this);
            __MR_Matrix2b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix2b() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector2b X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector2b Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Const_Matrix2b(Const_Matrix2b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix2b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Matrix2b _ctor_result = __MR_Matrix2b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2b::Matrix2b`.
        public unsafe Const_Matrix2b(MR.Const_Vector2b x, MR.Const_Vector2b y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_Construct", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_Construct(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Matrix2b _ctor_result = __MR_Matrix2b_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::Matrix2b::zero`.
        public static MR.Matrix2b Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_zero", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_zero();
            return __MR_Matrix2b_zero();
        }

        /// Generated from method `MR::Matrix2b::identity`.
        public static MR.Matrix2b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_identity", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_identity();
            return __MR_Matrix2b_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2b::scale`.
        public static MR.Matrix2b Scale(bool s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_scale_1_bool", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_scale_1_bool(byte s);
            return __MR_Matrix2b_scale_1_bool(s ? (byte)1 : (byte)0);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2b::scale`.
        public static MR.Matrix2b Scale(bool sx, bool sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_scale_2(byte sx, byte sy);
            return __MR_Matrix2b_scale_2(sx ? (byte)1 : (byte)0, sy ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Matrix2b::scale`.
        public static unsafe MR.Matrix2b Scale(MR.Const_Vector2b s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_scale_1_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_scale_1_MR_Vector2b(MR.Const_Vector2b._Underlying *s);
            return __MR_Matrix2b_scale_1_MR_Vector2b(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2b::fromRows`.
        public static unsafe MR.Matrix2b FromRows(MR.Const_Vector2b x, MR.Const_Vector2b y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_fromRows(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            return __MR_Matrix2b_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2b::fromColumns`.
        public static unsafe MR.Matrix2b FromColumns(MR.Const_Vector2b x, MR.Const_Vector2b y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_fromColumns(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            return __MR_Matrix2b_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2b::operator[]`.
        public unsafe MR.Const_Vector2b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2b._Underlying *__MR_Matrix2b_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix2b_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix2b::col`.
        public unsafe MR.Vector2b Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_col", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Matrix2b_col(_Underlying *_this, int i);
            return __MR_Matrix2b_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix2b_trace(_Underlying *_this);
            return __MR_Matrix2b_trace(_UnderlyingPtr) != 0;
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix2b_normSq(_Underlying *_this);
            return __MR_Matrix2b_normSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Matrix2b::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2b_norm(_Underlying *_this);
            return __MR_Matrix2b_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_det", ExactSpelling = true)]
            extern static byte __MR_Matrix2b_det(_Underlying *_this);
            return __MR_Matrix2b_det(_UnderlyingPtr) != 0;
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2b::transposed`.
        public unsafe MR.Matrix2b Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_transposed", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_transposed(_Underlying *_this);
            return __MR_Matrix2b_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_equal_MR_Matrix2b(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix2b a, MR.Const_Matrix2b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2i operator+(MR.Const_Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_add_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_add_MR_Matrix2b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2i operator-(MR.Const_Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_sub_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_sub_MR_Matrix2b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(bool a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_bool_MR_Matrix2b(byte a, MR.Const_Matrix2b._Underlying *b);
            return __MR_mul_bool_MR_Matrix2b(a ? (byte)1 : (byte)0, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Const_Matrix2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2b_bool(MR.Const_Matrix2b._Underlying *b, byte a);
            return __MR_mul_MR_Matrix2b_bool(b._UnderlyingPtr, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix2i operator/(Const_Matrix2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_div_MR_Matrix2b_bool(MR.Matrix2b b, byte a);
            return __MR_div_MR_Matrix2b_bool(b.UnderlyingStruct, a ? (byte)1 : (byte)0);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Const_Matrix2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2b_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Matrix2b_MR_Vector2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_mul_MR_Matrix2b_MR_Vector2b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Const_Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_mul_MR_Matrix2b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix2b? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix2b)
                return this == (MR.Const_Matrix2b)other;
            return false;
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2b`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix2b : Const_Matrix2b
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix2b UnderlyingStruct => ref *(Matrix2b *)_UnderlyingPtr;

        internal unsafe Mut_Matrix2b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector2b X => ref UnderlyingStruct.X;

        public new ref MR.Vector2b Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Matrix2b(Const_Matrix2b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix2b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Matrix2b _ctor_result = __MR_Matrix2b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2b::Matrix2b`.
        public unsafe Mut_Matrix2b(MR.Const_Vector2b x, MR.Const_Vector2b y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_Construct", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_Construct(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Matrix2b _ctor_result = __MR_Matrix2b_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::Matrix2b::operator[]`.
        public unsafe new MR.Mut_Vector2b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_Matrix2b_index(_Underlying *_this, int row);
            return new(__MR_Matrix2b_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2b AddAssign(MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_add_assign_MR_Matrix2b(_Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return new(__MR_add_assign_MR_Matrix2b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2b SubAssign(MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_sub_assign_MR_Matrix2b(_Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix2b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_mul_assign_MR_Matrix2b_bool(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Matrix2b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_div_assign_MR_Matrix2b_bool(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Matrix2b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2b`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct Matrix2b
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix2b(Const_Matrix2b other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix2b(Matrix2b other) => new(new Mut_Matrix2b((Mut_Matrix2b._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2b X;

        [System.Runtime.InteropServices.FieldOffset(2)]
        public MR.Vector2b Y;

        /// Generated copy constructor.
        public Matrix2b(Matrix2b _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix2b()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_DefaultConstruct();
            this = __MR_Matrix2b_DefaultConstruct();
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2b::Matrix2b`.
        public unsafe Matrix2b(MR.Const_Vector2b x, MR.Const_Vector2b y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_Construct", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_Construct(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            this = __MR_Matrix2b_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2b::zero`.
        public static MR.Matrix2b Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_zero", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_zero();
            return __MR_Matrix2b_zero();
        }

        /// Generated from method `MR::Matrix2b::identity`.
        public static MR.Matrix2b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_identity", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_identity();
            return __MR_Matrix2b_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2b::scale`.
        public static MR.Matrix2b Scale(bool s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_scale_1_bool", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_scale_1_bool(byte s);
            return __MR_Matrix2b_scale_1_bool(s ? (byte)1 : (byte)0);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2b::scale`.
        public static MR.Matrix2b Scale(bool sx, bool sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_scale_2(byte sx, byte sy);
            return __MR_Matrix2b_scale_2(sx ? (byte)1 : (byte)0, sy ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Matrix2b::scale`.
        public static unsafe MR.Matrix2b Scale(MR.Const_Vector2b s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_scale_1_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_scale_1_MR_Vector2b(MR.Const_Vector2b._Underlying *s);
            return __MR_Matrix2b_scale_1_MR_Vector2b(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2b::fromRows`.
        public static unsafe MR.Matrix2b FromRows(MR.Const_Vector2b x, MR.Const_Vector2b y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_fromRows(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            return __MR_Matrix2b_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2b::fromColumns`.
        public static unsafe MR.Matrix2b FromColumns(MR.Const_Vector2b x, MR.Const_Vector2b y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_fromColumns(MR.Const_Vector2b._Underlying *x, MR.Const_Vector2b._Underlying *y);
            return __MR_Matrix2b_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2b::operator[]`.
        public unsafe MR.Const_Vector2b Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2b._Underlying *__MR_Matrix2b_index_const(MR.Matrix2b *_this, int row);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return new(__MR_Matrix2b_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix2b::operator[]`.
        public unsafe MR.Mut_Vector2b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_Matrix2b_index(MR.Matrix2b *_this, int row);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return new(__MR_Matrix2b_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix2b::col`.
        public unsafe MR.Vector2b Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_col", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Matrix2b_col(MR.Matrix2b *_this, int i);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return __MR_Matrix2b_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix2b_trace(MR.Matrix2b *_this);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return __MR_Matrix2b_trace(__ptr__this) != 0;
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix2b_normSq(MR.Matrix2b *_this);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return __MR_Matrix2b_normSq(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::Matrix2b::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2b_norm(MR.Matrix2b *_this);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return __MR_Matrix2b_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_det", ExactSpelling = true)]
            extern static byte __MR_Matrix2b_det(MR.Matrix2b *_this);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return __MR_Matrix2b_det(__ptr__this) != 0;
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2b::transposed`.
        public unsafe MR.Matrix2b Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2b_transposed", ExactSpelling = true)]
            extern static MR.Matrix2b __MR_Matrix2b_transposed(MR.Matrix2b *_this);
            fixed (MR.Matrix2b *__ptr__this = &this)
            {
                return __MR_Matrix2b_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix2b a, MR.Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_equal_MR_Matrix2b((MR.Mut_Matrix2b._Underlying *)&a, (MR.Mut_Matrix2b._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix2b a, MR.Matrix2b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2i operator+(MR.Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_add_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_add_MR_Matrix2b((MR.Mut_Matrix2b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2i operator-(MR.Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_sub_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_sub_MR_Matrix2b((MR.Mut_Matrix2b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(bool a, MR.Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_bool_MR_Matrix2b(byte a, MR.Const_Matrix2b._Underlying *b);
            return __MR_mul_bool_MR_Matrix2b(a ? (byte)1 : (byte)0, (MR.Mut_Matrix2b._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Matrix2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2b_bool(MR.Const_Matrix2b._Underlying *b, byte a);
            return __MR_mul_MR_Matrix2b_bool((MR.Mut_Matrix2b._Underlying *)&b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix2i operator/(MR.Matrix2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_div_MR_Matrix2b_bool(MR.Matrix2b b, byte a);
            return __MR_div_MR_Matrix2b_bool(b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2b AddAssign(MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_add_assign_MR_Matrix2b(MR.Matrix2b *a, MR.Const_Matrix2b._Underlying *b);
            fixed (MR.Matrix2b *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix2b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2b SubAssign(MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_sub_assign_MR_Matrix2b(MR.Matrix2b *a, MR.Const_Matrix2b._Underlying *b);
            fixed (MR.Matrix2b *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix2b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_mul_assign_MR_Matrix2b_bool(MR.Matrix2b *a, byte b);
            fixed (MR.Matrix2b *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix2b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix2b._Underlying *__MR_div_assign_MR_Matrix2b_bool(MR.Matrix2b *a, byte b);
            fixed (MR.Matrix2b *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix2b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Matrix2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2b_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Matrix2b_MR_Vector2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_mul_MR_Matrix2b_MR_Vector2b((MR.Mut_Matrix2b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Matrix2b a, MR.Const_Matrix2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2b", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2b(MR.Const_Matrix2b._Underlying *a, MR.Const_Matrix2b._Underlying *b);
            return __MR_mul_MR_Matrix2b((MR.Mut_Matrix2b._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix2b b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix2b)
                return this == (MR.Matrix2b)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix2b` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix2b`/`Const_Matrix2b` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix2b
    {
        public readonly bool HasValue;
        internal readonly Matrix2b Object;
        public Matrix2b Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix2b() {HasValue = false;}
        public _InOpt_Matrix2b(Matrix2b new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix2b(Matrix2b new_value) {return new(new_value);}
        public _InOpt_Matrix2b(Const_Matrix2b new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix2b(Const_Matrix2b new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix2b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix2b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2b`/`Const_Matrix2b` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2b`.
    public class _InOptMut_Matrix2b
    {
        public Mut_Matrix2b? Opt;

        public _InOptMut_Matrix2b() {}
        public _InOptMut_Matrix2b(Mut_Matrix2b value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix2b(Mut_Matrix2b value) {return new(value);}
        public unsafe _InOptMut_Matrix2b(ref Matrix2b value)
        {
            fixed (Matrix2b *value_ptr = &value)
            {
                Opt = new((Const_Matrix2b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix2b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix2b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2b`/`Const_Matrix2b` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2b`.
    public class _InOptConst_Matrix2b
    {
        public Const_Matrix2b? Opt;

        public _InOptConst_Matrix2b() {}
        public _InOptConst_Matrix2b(Const_Matrix2b value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix2b(Const_Matrix2b value) {return new(value);}
        public unsafe _InOptConst_Matrix2b(ref readonly Matrix2b value)
        {
            fixed (Matrix2b *value_ptr = &value)
            {
                Opt = new((Const_Matrix2b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2i`.
    /// This is the const reference to the struct.
    public class Const_Matrix2i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix2i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix2i UnderlyingStruct => ref *(Matrix2i *)_UnderlyingPtr;

        internal unsafe Const_Matrix2i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix2i_Destroy(_Underlying *_this);
            __MR_Matrix2i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix2i() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector2i X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector2i Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Const_Matrix2i(Const_Matrix2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2i _ctor_result = __MR_Matrix2i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2i::Matrix2i`.
        public unsafe Const_Matrix2i(MR.Const_Vector2i x, MR.Const_Vector2i y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_Construct", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_Construct(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2i _ctor_result = __MR_Matrix2i_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Matrix2i::zero`.
        public static MR.Matrix2i Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_zero", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_zero();
            return __MR_Matrix2i_zero();
        }

        /// Generated from method `MR::Matrix2i::identity`.
        public static MR.Matrix2i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_identity", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_identity();
            return __MR_Matrix2i_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2i::scale`.
        public static MR.Matrix2i Scale(int s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_scale_1_int", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_scale_1_int(int s);
            return __MR_Matrix2i_scale_1_int(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2i::scale`.
        public static MR.Matrix2i Scale(int sx, int sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_scale_2(int sx, int sy);
            return __MR_Matrix2i_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2i::scale`.
        public static unsafe MR.Matrix2i Scale(MR.Const_Vector2i s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_scale_1_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_scale_1_MR_Vector2i(MR.Const_Vector2i._Underlying *s);
            return __MR_Matrix2i_scale_1_MR_Vector2i(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2i::fromRows`.
        public static unsafe MR.Matrix2i FromRows(MR.Const_Vector2i x, MR.Const_Vector2i y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_fromRows(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            return __MR_Matrix2i_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2i::fromColumns`.
        public static unsafe MR.Matrix2i FromColumns(MR.Const_Vector2i x, MR.Const_Vector2i y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_fromColumns(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            return __MR_Matrix2i_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2i::operator[]`.
        public unsafe MR.Const_Vector2i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_Matrix2i_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix2i_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix2i::col`.
        public unsafe MR.Vector2i Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_col", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Matrix2i_col(_Underlying *_this, int i);
            return __MR_Matrix2i_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_trace", ExactSpelling = true)]
            extern static int __MR_Matrix2i_trace(_Underlying *_this);
            return __MR_Matrix2i_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_normSq", ExactSpelling = true)]
            extern static int __MR_Matrix2i_normSq(_Underlying *_this);
            return __MR_Matrix2i_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2i::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2i_norm(_Underlying *_this);
            return __MR_Matrix2i_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_det", ExactSpelling = true)]
            extern static int __MR_Matrix2i_det(_Underlying *_this);
            return __MR_Matrix2i_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2i::transposed`.
        public unsafe MR.Matrix2i Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_transposed", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_transposed(_Underlying *_this);
            return __MR_Matrix2i_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_equal_MR_Matrix2i(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix2i a, MR.Const_Matrix2i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2i operator+(MR.Const_Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_add_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_add_MR_Matrix2i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2i operator-(MR.Const_Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_sub_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_sub_MR_Matrix2i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(int a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_int_MR_Matrix2i(int a, MR.Const_Matrix2i._Underlying *b);
            return __MR_mul_int_MR_Matrix2i(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Const_Matrix2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2i_int(MR.Const_Matrix2i._Underlying *b, int a);
            return __MR_mul_MR_Matrix2i_int(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix2i operator/(Const_Matrix2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_div_MR_Matrix2i_int(MR.Matrix2i b, int a);
            return __MR_div_MR_Matrix2i_int(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Const_Matrix2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Matrix2i_MR_Vector2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_mul_MR_Matrix2i_MR_Vector2i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Const_Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_mul_MR_Matrix2i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix2i? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix2i)
                return this == (MR.Const_Matrix2i)other;
            return false;
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2i`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix2i : Const_Matrix2i
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix2i UnderlyingStruct => ref *(Matrix2i *)_UnderlyingPtr;

        internal unsafe Mut_Matrix2i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector2i X => ref UnderlyingStruct.X;

        public new ref MR.Vector2i Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Matrix2i(Const_Matrix2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2i _ctor_result = __MR_Matrix2i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2i::Matrix2i`.
        public unsafe Mut_Matrix2i(MR.Const_Vector2i x, MR.Const_Vector2i y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_Construct", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_Construct(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2i _ctor_result = __MR_Matrix2i_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Matrix2i::operator[]`.
        public unsafe new MR.Mut_Vector2i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_Matrix2i_index(_Underlying *_this, int row);
            return new(__MR_Matrix2i_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2i AddAssign(MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_add_assign_MR_Matrix2i(_Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return new(__MR_add_assign_MR_Matrix2i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2i SubAssign(MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_sub_assign_MR_Matrix2i(_Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix2i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_mul_assign_MR_Matrix2i_int(_Underlying *a, int b);
            return new(__MR_mul_assign_MR_Matrix2i_int(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_div_assign_MR_Matrix2i_int(_Underlying *a, int b);
            return new(__MR_div_assign_MR_Matrix2i_int(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Matrix2i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix2i(Const_Matrix2i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix2i(Matrix2i other) => new(new Mut_Matrix2i((Mut_Matrix2i._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2i X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public MR.Vector2i Y;

        /// Generated copy constructor.
        public Matrix2i(Matrix2i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix2i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_DefaultConstruct();
            this = __MR_Matrix2i_DefaultConstruct();
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2i::Matrix2i`.
        public unsafe Matrix2i(MR.Const_Vector2i x, MR.Const_Vector2i y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_Construct", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_Construct(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            this = __MR_Matrix2i_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2i::zero`.
        public static MR.Matrix2i Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_zero", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_zero();
            return __MR_Matrix2i_zero();
        }

        /// Generated from method `MR::Matrix2i::identity`.
        public static MR.Matrix2i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_identity", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_identity();
            return __MR_Matrix2i_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2i::scale`.
        public static MR.Matrix2i Scale(int s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_scale_1_int", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_scale_1_int(int s);
            return __MR_Matrix2i_scale_1_int(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2i::scale`.
        public static MR.Matrix2i Scale(int sx, int sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_scale_2(int sx, int sy);
            return __MR_Matrix2i_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2i::scale`.
        public static unsafe MR.Matrix2i Scale(MR.Const_Vector2i s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_scale_1_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_scale_1_MR_Vector2i(MR.Const_Vector2i._Underlying *s);
            return __MR_Matrix2i_scale_1_MR_Vector2i(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2i::fromRows`.
        public static unsafe MR.Matrix2i FromRows(MR.Const_Vector2i x, MR.Const_Vector2i y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_fromRows(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            return __MR_Matrix2i_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2i::fromColumns`.
        public static unsafe MR.Matrix2i FromColumns(MR.Const_Vector2i x, MR.Const_Vector2i y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_fromColumns(MR.Const_Vector2i._Underlying *x, MR.Const_Vector2i._Underlying *y);
            return __MR_Matrix2i_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2i::operator[]`.
        public unsafe MR.Const_Vector2i Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_Matrix2i_index_const(MR.Matrix2i *_this, int row);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return new(__MR_Matrix2i_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix2i::operator[]`.
        public unsafe MR.Mut_Vector2i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_Matrix2i_index(MR.Matrix2i *_this, int row);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return new(__MR_Matrix2i_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix2i::col`.
        public unsafe MR.Vector2i Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_col", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Matrix2i_col(MR.Matrix2i *_this, int i);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return __MR_Matrix2i_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_trace", ExactSpelling = true)]
            extern static int __MR_Matrix2i_trace(MR.Matrix2i *_this);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return __MR_Matrix2i_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_normSq", ExactSpelling = true)]
            extern static int __MR_Matrix2i_normSq(MR.Matrix2i *_this);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return __MR_Matrix2i_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix2i::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2i_norm(MR.Matrix2i *_this);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return __MR_Matrix2i_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_det", ExactSpelling = true)]
            extern static int __MR_Matrix2i_det(MR.Matrix2i *_this);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return __MR_Matrix2i_det(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2i::transposed`.
        public unsafe MR.Matrix2i Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i_transposed", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_Matrix2i_transposed(MR.Matrix2i *_this);
            fixed (MR.Matrix2i *__ptr__this = &this)
            {
                return __MR_Matrix2i_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix2i a, MR.Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_equal_MR_Matrix2i((MR.Mut_Matrix2i._Underlying *)&a, (MR.Mut_Matrix2i._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix2i a, MR.Matrix2i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2i operator+(MR.Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_add_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_add_MR_Matrix2i((MR.Mut_Matrix2i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2i operator-(MR.Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_sub_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_sub_MR_Matrix2i((MR.Mut_Matrix2i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(int a, MR.Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_int_MR_Matrix2i(int a, MR.Const_Matrix2i._Underlying *b);
            return __MR_mul_int_MR_Matrix2i(a, (MR.Mut_Matrix2i._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Matrix2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2i_int(MR.Const_Matrix2i._Underlying *b, int a);
            return __MR_mul_MR_Matrix2i_int((MR.Mut_Matrix2i._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix2i operator/(MR.Matrix2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_div_MR_Matrix2i_int(MR.Matrix2i b, int a);
            return __MR_div_MR_Matrix2i_int(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2i AddAssign(MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_add_assign_MR_Matrix2i(MR.Matrix2i *a, MR.Const_Matrix2i._Underlying *b);
            fixed (MR.Matrix2i *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix2i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2i SubAssign(MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_sub_assign_MR_Matrix2i(MR.Matrix2i *a, MR.Const_Matrix2i._Underlying *b);
            fixed (MR.Matrix2i *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix2i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_mul_assign_MR_Matrix2i_int(MR.Matrix2i *a, int b);
            fixed (MR.Matrix2i *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix2i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i._Underlying *__MR_div_assign_MR_Matrix2i_int(MR.Matrix2i *a, int b);
            fixed (MR.Matrix2i *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix2i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Matrix2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Matrix2i_MR_Vector2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_mul_MR_Matrix2i_MR_Vector2i((MR.Mut_Matrix2i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i operator*(MR.Matrix2i a, MR.Const_Matrix2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i", ExactSpelling = true)]
            extern static MR.Matrix2i __MR_mul_MR_Matrix2i(MR.Const_Matrix2i._Underlying *a, MR.Const_Matrix2i._Underlying *b);
            return __MR_mul_MR_Matrix2i((MR.Mut_Matrix2i._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix2i b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix2i)
                return this == (MR.Matrix2i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix2i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix2i`/`Const_Matrix2i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix2i
    {
        public readonly bool HasValue;
        internal readonly Matrix2i Object;
        public Matrix2i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix2i() {HasValue = false;}
        public _InOpt_Matrix2i(Matrix2i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix2i(Matrix2i new_value) {return new(new_value);}
        public _InOpt_Matrix2i(Const_Matrix2i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix2i(Const_Matrix2i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix2i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2i`/`Const_Matrix2i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2i`.
    public class _InOptMut_Matrix2i
    {
        public Mut_Matrix2i? Opt;

        public _InOptMut_Matrix2i() {}
        public _InOptMut_Matrix2i(Mut_Matrix2i value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix2i(Mut_Matrix2i value) {return new(value);}
        public unsafe _InOptMut_Matrix2i(ref Matrix2i value)
        {
            fixed (Matrix2i *value_ptr = &value)
            {
                Opt = new((Const_Matrix2i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix2i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2i`/`Const_Matrix2i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2i`.
    public class _InOptConst_Matrix2i
    {
        public Const_Matrix2i? Opt;

        public _InOptConst_Matrix2i() {}
        public _InOptConst_Matrix2i(Const_Matrix2i value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix2i(Const_Matrix2i value) {return new(value);}
        public unsafe _InOptConst_Matrix2i(ref readonly Matrix2i value)
        {
            fixed (Matrix2i *value_ptr = &value)
            {
                Opt = new((Const_Matrix2i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2i64`.
    /// This is the const reference to the struct.
    public class Const_Matrix2i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix2i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix2i64 UnderlyingStruct => ref *(Matrix2i64 *)_UnderlyingPtr;

        internal unsafe Const_Matrix2i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix2i64_Destroy(_Underlying *_this);
            __MR_Matrix2i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix2i64() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector2i64 X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector2i64 Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Const_Matrix2i64(Const_Matrix2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2i64 _ctor_result = __MR_Matrix2i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2i64::Matrix2i64`.
        public unsafe Const_Matrix2i64(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_Construct", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_Construct(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2i64 _ctor_result = __MR_Matrix2i64_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Matrix2i64::zero`.
        public static MR.Matrix2i64 Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_zero", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_zero();
            return __MR_Matrix2i64_zero();
        }

        /// Generated from method `MR::Matrix2i64::identity`.
        public static MR.Matrix2i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_identity", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_identity();
            return __MR_Matrix2i64_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2i64::scale`.
        public static MR.Matrix2i64 Scale(long s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_scale_1_int64_t", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_scale_1_int64_t(long s);
            return __MR_Matrix2i64_scale_1_int64_t(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2i64::scale`.
        public static MR.Matrix2i64 Scale(long sx, long sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_scale_2(long sx, long sy);
            return __MR_Matrix2i64_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2i64::scale`.
        public static unsafe MR.Matrix2i64 Scale(MR.Const_Vector2i64 s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_scale_1_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_scale_1_MR_Vector2i64(MR.Const_Vector2i64._Underlying *s);
            return __MR_Matrix2i64_scale_1_MR_Vector2i64(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2i64::fromRows`.
        public static unsafe MR.Matrix2i64 FromRows(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_fromRows(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            return __MR_Matrix2i64_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2i64::fromColumns`.
        public static unsafe MR.Matrix2i64 FromColumns(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_fromColumns(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            return __MR_Matrix2i64_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2i64::operator[]`.
        public unsafe MR.Const_Vector2i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i64._Underlying *__MR_Matrix2i64_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix2i64_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix2i64::col`.
        public unsafe MR.Vector2i64 Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_col", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Matrix2i64_col(_Underlying *_this, int i);
            return __MR_Matrix2i64_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_trace", ExactSpelling = true)]
            extern static long __MR_Matrix2i64_trace(_Underlying *_this);
            return __MR_Matrix2i64_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_normSq", ExactSpelling = true)]
            extern static long __MR_Matrix2i64_normSq(_Underlying *_this);
            return __MR_Matrix2i64_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2i64::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2i64_norm(_Underlying *_this);
            return __MR_Matrix2i64_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_det", ExactSpelling = true)]
            extern static long __MR_Matrix2i64_det(_Underlying *_this);
            return __MR_Matrix2i64_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2i64::transposed`.
        public unsafe MR.Matrix2i64 Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_transposed", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_transposed(_Underlying *_this);
            return __MR_Matrix2i64_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_equal_MR_Matrix2i64(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2i64 operator+(MR.Const_Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_add_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_add_MR_Matrix2i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2i64 operator-(MR.Const_Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_sub_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_sub_MR_Matrix2i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i64 operator*(long a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_mul_int64_t_MR_Matrix2i64(long a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_mul_int64_t_MR_Matrix2i64(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i64 operator*(MR.Const_Matrix2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_mul_MR_Matrix2i64_int64_t(MR.Const_Matrix2i64._Underlying *b, long a);
            return __MR_mul_MR_Matrix2i64_int64_t(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix2i64 operator/(Const_Matrix2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_div_MR_Matrix2i64_int64_t(MR.Matrix2i64 b, long a);
            return __MR_div_MR_Matrix2i64_int64_t(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i64 operator*(MR.Const_Matrix2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i64_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_mul_MR_Matrix2i64_MR_Vector2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_mul_MR_Matrix2i64_MR_Vector2i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i64 operator*(MR.Const_Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_mul_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_mul_MR_Matrix2i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix2i64? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix2i64)
                return this == (MR.Const_Matrix2i64)other;
            return false;
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix2i64 : Const_Matrix2i64
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix2i64 UnderlyingStruct => ref *(Matrix2i64 *)_UnderlyingPtr;

        internal unsafe Mut_Matrix2i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector2i64 X => ref UnderlyingStruct.X;

        public new ref MR.Vector2i64 Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Matrix2i64(Const_Matrix2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2i64 _ctor_result = __MR_Matrix2i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2i64::Matrix2i64`.
        public unsafe Mut_Matrix2i64(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_Construct", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_Construct(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2i64 _ctor_result = __MR_Matrix2i64_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Matrix2i64::operator[]`.
        public unsafe new MR.Mut_Vector2i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_Matrix2i64_index(_Underlying *_this, int row);
            return new(__MR_Matrix2i64_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2i64 AddAssign(MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_add_assign_MR_Matrix2i64(_Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return new(__MR_add_assign_MR_Matrix2i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2i64 SubAssign(MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_sub_assign_MR_Matrix2i64(_Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix2i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_mul_assign_MR_Matrix2i64_int64_t(_Underlying *a, long b);
            return new(__MR_mul_assign_MR_Matrix2i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_div_assign_MR_Matrix2i64_int64_t(_Underlying *a, long b);
            return new(__MR_div_assign_MR_Matrix2i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
    public struct Matrix2i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix2i64(Const_Matrix2i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix2i64(Matrix2i64 other) => new(new Mut_Matrix2i64((Mut_Matrix2i64._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2i64 X;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector2i64 Y;

        /// Generated copy constructor.
        public Matrix2i64(Matrix2i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix2i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_DefaultConstruct();
            this = __MR_Matrix2i64_DefaultConstruct();
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2i64::Matrix2i64`.
        public unsafe Matrix2i64(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_Construct", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_Construct(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            this = __MR_Matrix2i64_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2i64::zero`.
        public static MR.Matrix2i64 Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_zero", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_zero();
            return __MR_Matrix2i64_zero();
        }

        /// Generated from method `MR::Matrix2i64::identity`.
        public static MR.Matrix2i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_identity", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_identity();
            return __MR_Matrix2i64_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2i64::scale`.
        public static MR.Matrix2i64 Scale(long s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_scale_1_int64_t", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_scale_1_int64_t(long s);
            return __MR_Matrix2i64_scale_1_int64_t(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2i64::scale`.
        public static MR.Matrix2i64 Scale(long sx, long sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_scale_2(long sx, long sy);
            return __MR_Matrix2i64_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2i64::scale`.
        public static unsafe MR.Matrix2i64 Scale(MR.Const_Vector2i64 s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_scale_1_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_scale_1_MR_Vector2i64(MR.Const_Vector2i64._Underlying *s);
            return __MR_Matrix2i64_scale_1_MR_Vector2i64(s._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2i64::fromRows`.
        public static unsafe MR.Matrix2i64 FromRows(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_fromRows(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            return __MR_Matrix2i64_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2i64::fromColumns`.
        public static unsafe MR.Matrix2i64 FromColumns(MR.Const_Vector2i64 x, MR.Const_Vector2i64 y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_fromColumns(MR.Const_Vector2i64._Underlying *x, MR.Const_Vector2i64._Underlying *y);
            return __MR_Matrix2i64_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2i64::operator[]`.
        public unsafe MR.Const_Vector2i64 Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i64._Underlying *__MR_Matrix2i64_index_const(MR.Matrix2i64 *_this, int row);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return new(__MR_Matrix2i64_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix2i64::operator[]`.
        public unsafe MR.Mut_Vector2i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_Matrix2i64_index(MR.Matrix2i64 *_this, int row);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return new(__MR_Matrix2i64_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix2i64::col`.
        public unsafe MR.Vector2i64 Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_col", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Matrix2i64_col(MR.Matrix2i64 *_this, int i);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return __MR_Matrix2i64_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_trace", ExactSpelling = true)]
            extern static long __MR_Matrix2i64_trace(MR.Matrix2i64 *_this);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return __MR_Matrix2i64_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_normSq", ExactSpelling = true)]
            extern static long __MR_Matrix2i64_normSq(MR.Matrix2i64 *_this);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return __MR_Matrix2i64_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix2i64::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2i64_norm(MR.Matrix2i64 *_this);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return __MR_Matrix2i64_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_det", ExactSpelling = true)]
            extern static long __MR_Matrix2i64_det(MR.Matrix2i64 *_this);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return __MR_Matrix2i64_det(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2i64::transposed`.
        public unsafe MR.Matrix2i64 Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2i64_transposed", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_Matrix2i64_transposed(MR.Matrix2i64 *_this);
            fixed (MR.Matrix2i64 *__ptr__this = &this)
            {
                return __MR_Matrix2i64_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix2i64 a, MR.Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_equal_MR_Matrix2i64((MR.Mut_Matrix2i64._Underlying *)&a, (MR.Mut_Matrix2i64._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix2i64 a, MR.Matrix2i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2i64 operator+(MR.Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_add_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_add_MR_Matrix2i64((MR.Mut_Matrix2i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2i64 operator-(MR.Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_sub_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_sub_MR_Matrix2i64((MR.Mut_Matrix2i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i64 operator*(long a, MR.Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_mul_int64_t_MR_Matrix2i64(long a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_mul_int64_t_MR_Matrix2i64(a, (MR.Mut_Matrix2i64._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i64 operator*(MR.Matrix2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_mul_MR_Matrix2i64_int64_t(MR.Const_Matrix2i64._Underlying *b, long a);
            return __MR_mul_MR_Matrix2i64_int64_t((MR.Mut_Matrix2i64._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix2i64 operator/(MR.Matrix2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_div_MR_Matrix2i64_int64_t(MR.Matrix2i64 b, long a);
            return __MR_div_MR_Matrix2i64_int64_t(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2i64 AddAssign(MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_add_assign_MR_Matrix2i64(MR.Matrix2i64 *a, MR.Const_Matrix2i64._Underlying *b);
            fixed (MR.Matrix2i64 *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix2i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2i64 SubAssign(MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_sub_assign_MR_Matrix2i64(MR.Matrix2i64 *a, MR.Const_Matrix2i64._Underlying *b);
            fixed (MR.Matrix2i64 *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix2i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_mul_assign_MR_Matrix2i64_int64_t(MR.Matrix2i64 *a, long b);
            fixed (MR.Matrix2i64 *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix2i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix2i64._Underlying *__MR_div_assign_MR_Matrix2i64_int64_t(MR.Matrix2i64 *a, long b);
            fixed (MR.Matrix2i64 *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix2i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i64 operator*(MR.Matrix2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i64_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_mul_MR_Matrix2i64_MR_Vector2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_mul_MR_Matrix2i64_MR_Vector2i64((MR.Mut_Matrix2i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2i64 operator*(MR.Matrix2i64 a, MR.Const_Matrix2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2i64", ExactSpelling = true)]
            extern static MR.Matrix2i64 __MR_mul_MR_Matrix2i64(MR.Const_Matrix2i64._Underlying *a, MR.Const_Matrix2i64._Underlying *b);
            return __MR_mul_MR_Matrix2i64((MR.Mut_Matrix2i64._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix2i64 b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix2i64)
                return this == (MR.Matrix2i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix2i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix2i64`/`Const_Matrix2i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix2i64
    {
        public readonly bool HasValue;
        internal readonly Matrix2i64 Object;
        public Matrix2i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix2i64() {HasValue = false;}
        public _InOpt_Matrix2i64(Matrix2i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix2i64(Matrix2i64 new_value) {return new(new_value);}
        public _InOpt_Matrix2i64(Const_Matrix2i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix2i64(Const_Matrix2i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix2i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2i64`/`Const_Matrix2i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2i64`.
    public class _InOptMut_Matrix2i64
    {
        public Mut_Matrix2i64? Opt;

        public _InOptMut_Matrix2i64() {}
        public _InOptMut_Matrix2i64(Mut_Matrix2i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix2i64(Mut_Matrix2i64 value) {return new(value);}
        public unsafe _InOptMut_Matrix2i64(ref Matrix2i64 value)
        {
            fixed (Matrix2i64 *value_ptr = &value)
            {
                Opt = new((Const_Matrix2i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix2i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2i64`/`Const_Matrix2i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2i64`.
    public class _InOptConst_Matrix2i64
    {
        public Const_Matrix2i64? Opt;

        public _InOptConst_Matrix2i64() {}
        public _InOptConst_Matrix2i64(Const_Matrix2i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix2i64(Const_Matrix2i64 value) {return new(value);}
        public unsafe _InOptConst_Matrix2i64(ref readonly Matrix2i64 value)
        {
            fixed (Matrix2i64 *value_ptr = &value)
            {
                Opt = new((Const_Matrix2i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2f`.
    /// This is the const reference to the struct.
    public class Const_Matrix2f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix2f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix2f UnderlyingStruct => ref *(Matrix2f *)_UnderlyingPtr;

        internal unsafe Const_Matrix2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix2f_Destroy(_Underlying *_this);
            __MR_Matrix2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix2f() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector2f X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector2f Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Const_Matrix2f(Const_Matrix2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2f _ctor_result = __MR_Matrix2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2f::Matrix2f`.
        public unsafe Const_Matrix2f(MR.Const_Vector2f x, MR.Const_Vector2f y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_Construct", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_Construct(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2f _ctor_result = __MR_Matrix2f_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Matrix2f::zero`.
        public static MR.Matrix2f Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_zero", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_zero();
            return __MR_Matrix2f_zero();
        }

        /// Generated from method `MR::Matrix2f::identity`.
        public static MR.Matrix2f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_identity", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_identity();
            return __MR_Matrix2f_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2f::scale`.
        public static MR.Matrix2f Scale(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_scale_1_float", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_scale_1_float(float s);
            return __MR_Matrix2f_scale_1_float(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2f::scale`.
        public static MR.Matrix2f Scale(float sx, float sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_scale_2(float sx, float sy);
            return __MR_Matrix2f_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2f::scale`.
        public static unsafe MR.Matrix2f Scale(MR.Const_Vector2f s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_scale_1_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_scale_1_MR_Vector2f(MR.Const_Vector2f._Underlying *s);
            return __MR_Matrix2f_scale_1_MR_Vector2f(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around origin on given angle
        /// Generated from method `MR::Matrix2f::rotation`.
        public static MR.Matrix2f Rotation(float angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_rotation_1", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_rotation_1(float angle);
            return __MR_Matrix2f_rotation_1(angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix2f::rotation`.
        public static unsafe MR.Matrix2f Rotation(MR.Const_Vector2f from, MR.Const_Vector2f to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_rotation_2", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_rotation_2(MR.Const_Vector2f._Underlying *from, MR.Const_Vector2f._Underlying *to);
            return __MR_Matrix2f_rotation_2(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2f::fromRows`.
        public static unsafe MR.Matrix2f FromRows(MR.Const_Vector2f x, MR.Const_Vector2f y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_fromRows(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            return __MR_Matrix2f_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2f::fromColumns`.
        public static unsafe MR.Matrix2f FromColumns(MR.Const_Vector2f x, MR.Const_Vector2f y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_fromColumns(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            return __MR_Matrix2f_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2f::operator[]`.
        public unsafe MR.Const_Vector2f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2f._Underlying *__MR_Matrix2f_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix2f_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix2f::col`.
        public unsafe MR.Vector2f Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_col", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Matrix2f_col(_Underlying *_this, int i);
            return __MR_Matrix2f_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_trace", ExactSpelling = true)]
            extern static float __MR_Matrix2f_trace(_Underlying *_this);
            return __MR_Matrix2f_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_normSq", ExactSpelling = true)]
            extern static float __MR_Matrix2f_normSq(_Underlying *_this);
            return __MR_Matrix2f_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2f::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_norm", ExactSpelling = true)]
            extern static float __MR_Matrix2f_norm(_Underlying *_this);
            return __MR_Matrix2f_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_det", ExactSpelling = true)]
            extern static float __MR_Matrix2f_det(_Underlying *_this);
            return __MR_Matrix2f_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix2f::inverse`.
        public unsafe MR.Matrix2f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_inverse", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_inverse(_Underlying *_this);
            return __MR_Matrix2f_inverse(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2f::transposed`.
        public unsafe MR.Matrix2f Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_transposed", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_transposed(_Underlying *_this);
            return __MR_Matrix2f_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_equal_MR_Matrix2f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix2f a, MR.Const_Matrix2f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2f operator+(MR.Const_Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_add_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_add_MR_Matrix2f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2f operator-(MR.Const_Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_sub_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_sub_MR_Matrix2f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2f operator*(float a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_mul_float_MR_Matrix2f(float a, MR.Const_Matrix2f._Underlying *b);
            return __MR_mul_float_MR_Matrix2f(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2f operator*(MR.Const_Matrix2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_mul_MR_Matrix2f_float(MR.Const_Matrix2f._Underlying *b, float a);
            return __MR_mul_MR_Matrix2f_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix2f operator/(Const_Matrix2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_div_MR_Matrix2f_float(MR.Matrix2f b, float a);
            return __MR_div_MR_Matrix2f_float(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2f operator*(MR.Const_Matrix2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2f_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_mul_MR_Matrix2f_MR_Vector2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_mul_MR_Matrix2f_MR_Vector2f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2f operator*(MR.Const_Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_mul_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_mul_MR_Matrix2f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix2f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix2f)
                return this == (MR.Const_Matrix2f)other;
            return false;
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2f`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix2f : Const_Matrix2f
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix2f UnderlyingStruct => ref *(Matrix2f *)_UnderlyingPtr;

        internal unsafe Mut_Matrix2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector2f X => ref UnderlyingStruct.X;

        public new ref MR.Vector2f Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Matrix2f(Const_Matrix2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2f _ctor_result = __MR_Matrix2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2f::Matrix2f`.
        public unsafe Mut_Matrix2f(MR.Const_Vector2f x, MR.Const_Vector2f y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_Construct", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_Construct(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix2f _ctor_result = __MR_Matrix2f_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Matrix2f::operator[]`.
        public unsafe new MR.Mut_Vector2f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_Matrix2f_index(_Underlying *_this, int row);
            return new(__MR_Matrix2f_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2f AddAssign(MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_add_assign_MR_Matrix2f(_Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return new(__MR_add_assign_MR_Matrix2f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2f SubAssign(MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_sub_assign_MR_Matrix2f(_Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix2f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_mul_assign_MR_Matrix2f_float(_Underlying *a, float b);
            return new(__MR_mul_assign_MR_Matrix2f_float(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_div_assign_MR_Matrix2f_float(_Underlying *a, float b);
            return new(__MR_div_assign_MR_Matrix2f_float(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Matrix2f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix2f(Const_Matrix2f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix2f(Matrix2f other) => new(new Mut_Matrix2f((Mut_Matrix2f._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2f X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public MR.Vector2f Y;

        /// Generated copy constructor.
        public Matrix2f(Matrix2f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix2f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_DefaultConstruct();
            this = __MR_Matrix2f_DefaultConstruct();
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2f::Matrix2f`.
        public unsafe Matrix2f(MR.Const_Vector2f x, MR.Const_Vector2f y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_Construct", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_Construct(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            this = __MR_Matrix2f_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2f::zero`.
        public static MR.Matrix2f Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_zero", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_zero();
            return __MR_Matrix2f_zero();
        }

        /// Generated from method `MR::Matrix2f::identity`.
        public static MR.Matrix2f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_identity", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_identity();
            return __MR_Matrix2f_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2f::scale`.
        public static MR.Matrix2f Scale(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_scale_1_float", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_scale_1_float(float s);
            return __MR_Matrix2f_scale_1_float(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2f::scale`.
        public static MR.Matrix2f Scale(float sx, float sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_scale_2(float sx, float sy);
            return __MR_Matrix2f_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2f::scale`.
        public static unsafe MR.Matrix2f Scale(MR.Const_Vector2f s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_scale_1_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_scale_1_MR_Vector2f(MR.Const_Vector2f._Underlying *s);
            return __MR_Matrix2f_scale_1_MR_Vector2f(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around origin on given angle
        /// Generated from method `MR::Matrix2f::rotation`.
        public static MR.Matrix2f Rotation(float angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_rotation_1", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_rotation_1(float angle);
            return __MR_Matrix2f_rotation_1(angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix2f::rotation`.
        public static unsafe MR.Matrix2f Rotation(MR.Const_Vector2f from, MR.Const_Vector2f to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_rotation_2", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_rotation_2(MR.Const_Vector2f._Underlying *from, MR.Const_Vector2f._Underlying *to);
            return __MR_Matrix2f_rotation_2(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2f::fromRows`.
        public static unsafe MR.Matrix2f FromRows(MR.Const_Vector2f x, MR.Const_Vector2f y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_fromRows(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            return __MR_Matrix2f_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2f::fromColumns`.
        public static unsafe MR.Matrix2f FromColumns(MR.Const_Vector2f x, MR.Const_Vector2f y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_fromColumns(MR.Const_Vector2f._Underlying *x, MR.Const_Vector2f._Underlying *y);
            return __MR_Matrix2f_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2f::operator[]`.
        public unsafe MR.Const_Vector2f Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2f._Underlying *__MR_Matrix2f_index_const(MR.Matrix2f *_this, int row);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return new(__MR_Matrix2f_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix2f::operator[]`.
        public unsafe MR.Mut_Vector2f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_Matrix2f_index(MR.Matrix2f *_this, int row);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return new(__MR_Matrix2f_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix2f::col`.
        public unsafe MR.Vector2f Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_col", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Matrix2f_col(MR.Matrix2f *_this, int i);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_trace", ExactSpelling = true)]
            extern static float __MR_Matrix2f_trace(MR.Matrix2f *_this);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_normSq", ExactSpelling = true)]
            extern static float __MR_Matrix2f_normSq(MR.Matrix2f *_this);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix2f::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_norm", ExactSpelling = true)]
            extern static float __MR_Matrix2f_norm(MR.Matrix2f *_this);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_det", ExactSpelling = true)]
            extern static float __MR_Matrix2f_det(MR.Matrix2f *_this);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_det(__ptr__this);
            }
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix2f::inverse`.
        public unsafe MR.Matrix2f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_inverse", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_inverse(MR.Matrix2f *_this);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_inverse(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2f::transposed`.
        public unsafe MR.Matrix2f Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2f_transposed", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_Matrix2f_transposed(MR.Matrix2f *_this);
            fixed (MR.Matrix2f *__ptr__this = &this)
            {
                return __MR_Matrix2f_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix2f a, MR.Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_equal_MR_Matrix2f((MR.Mut_Matrix2f._Underlying *)&a, (MR.Mut_Matrix2f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix2f a, MR.Matrix2f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2f operator+(MR.Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_add_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_add_MR_Matrix2f((MR.Mut_Matrix2f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2f operator-(MR.Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_sub_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_sub_MR_Matrix2f((MR.Mut_Matrix2f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2f operator*(float a, MR.Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_mul_float_MR_Matrix2f(float a, MR.Const_Matrix2f._Underlying *b);
            return __MR_mul_float_MR_Matrix2f(a, (MR.Mut_Matrix2f._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2f operator*(MR.Matrix2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_mul_MR_Matrix2f_float(MR.Const_Matrix2f._Underlying *b, float a);
            return __MR_mul_MR_Matrix2f_float((MR.Mut_Matrix2f._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix2f operator/(MR.Matrix2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_div_MR_Matrix2f_float(MR.Matrix2f b, float a);
            return __MR_div_MR_Matrix2f_float(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2f AddAssign(MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_add_assign_MR_Matrix2f(MR.Matrix2f *a, MR.Const_Matrix2f._Underlying *b);
            fixed (MR.Matrix2f *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix2f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2f SubAssign(MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_sub_assign_MR_Matrix2f(MR.Matrix2f *a, MR.Const_Matrix2f._Underlying *b);
            fixed (MR.Matrix2f *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix2f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_mul_assign_MR_Matrix2f_float(MR.Matrix2f *a, float b);
            fixed (MR.Matrix2f *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix2f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix2f._Underlying *__MR_div_assign_MR_Matrix2f_float(MR.Matrix2f *a, float b);
            fixed (MR.Matrix2f *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix2f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2f operator*(MR.Matrix2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2f_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_mul_MR_Matrix2f_MR_Vector2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_mul_MR_Matrix2f_MR_Vector2f((MR.Mut_Matrix2f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2f operator*(MR.Matrix2f a, MR.Const_Matrix2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2f", ExactSpelling = true)]
            extern static MR.Matrix2f __MR_mul_MR_Matrix2f(MR.Const_Matrix2f._Underlying *a, MR.Const_Matrix2f._Underlying *b);
            return __MR_mul_MR_Matrix2f((MR.Mut_Matrix2f._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix2f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix2f)
                return this == (MR.Matrix2f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix2f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix2f`/`Const_Matrix2f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix2f
    {
        public readonly bool HasValue;
        internal readonly Matrix2f Object;
        public Matrix2f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix2f() {HasValue = false;}
        public _InOpt_Matrix2f(Matrix2f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix2f(Matrix2f new_value) {return new(new_value);}
        public _InOpt_Matrix2f(Const_Matrix2f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix2f(Const_Matrix2f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2f`/`Const_Matrix2f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2f`.
    public class _InOptMut_Matrix2f
    {
        public Mut_Matrix2f? Opt;

        public _InOptMut_Matrix2f() {}
        public _InOptMut_Matrix2f(Mut_Matrix2f value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix2f(Mut_Matrix2f value) {return new(value);}
        public unsafe _InOptMut_Matrix2f(ref Matrix2f value)
        {
            fixed (Matrix2f *value_ptr = &value)
            {
                Opt = new((Const_Matrix2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2f`/`Const_Matrix2f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2f`.
    public class _InOptConst_Matrix2f
    {
        public Const_Matrix2f? Opt;

        public _InOptConst_Matrix2f() {}
        public _InOptConst_Matrix2f(Const_Matrix2f value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix2f(Const_Matrix2f value) {return new(value);}
        public unsafe _InOptConst_Matrix2f(ref readonly Matrix2f value)
        {
            fixed (Matrix2f *value_ptr = &value)
            {
                Opt = new((Const_Matrix2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2d`.
    /// This is the const reference to the struct.
    public class Const_Matrix2d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix2d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix2d UnderlyingStruct => ref *(Matrix2d *)_UnderlyingPtr;

        internal unsafe Const_Matrix2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix2d_Destroy(_Underlying *_this);
            __MR_Matrix2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix2d() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector2d X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector2d Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Const_Matrix2d(Const_Matrix2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2d _ctor_result = __MR_Matrix2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2d::Matrix2d`.
        public unsafe Const_Matrix2d(MR.Const_Vector2d x, MR.Const_Vector2d y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_Construct", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_Construct(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2d _ctor_result = __MR_Matrix2d_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Matrix2d::zero`.
        public static MR.Matrix2d Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_zero", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_zero();
            return __MR_Matrix2d_zero();
        }

        /// Generated from method `MR::Matrix2d::identity`.
        public static MR.Matrix2d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_identity", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_identity();
            return __MR_Matrix2d_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2d::scale`.
        public static MR.Matrix2d Scale(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_scale_1_double", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_scale_1_double(double s);
            return __MR_Matrix2d_scale_1_double(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2d::scale`.
        public static MR.Matrix2d Scale(double sx, double sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_scale_2(double sx, double sy);
            return __MR_Matrix2d_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2d::scale`.
        public static unsafe MR.Matrix2d Scale(MR.Const_Vector2d s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_scale_1_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_scale_1_MR_Vector2d(MR.Const_Vector2d._Underlying *s);
            return __MR_Matrix2d_scale_1_MR_Vector2d(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around origin on given angle
        /// Generated from method `MR::Matrix2d::rotation`.
        public static MR.Matrix2d Rotation(double angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_rotation_1", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_rotation_1(double angle);
            return __MR_Matrix2d_rotation_1(angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix2d::rotation`.
        public static unsafe MR.Matrix2d Rotation(MR.Const_Vector2d from, MR.Const_Vector2d to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_rotation_2", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_rotation_2(MR.Const_Vector2d._Underlying *from, MR.Const_Vector2d._Underlying *to);
            return __MR_Matrix2d_rotation_2(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2d::fromRows`.
        public static unsafe MR.Matrix2d FromRows(MR.Const_Vector2d x, MR.Const_Vector2d y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_fromRows(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            return __MR_Matrix2d_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2d::fromColumns`.
        public static unsafe MR.Matrix2d FromColumns(MR.Const_Vector2d x, MR.Const_Vector2d y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_fromColumns(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            return __MR_Matrix2d_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2d::operator[]`.
        public unsafe MR.Const_Vector2d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2d._Underlying *__MR_Matrix2d_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix2d_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix2d::col`.
        public unsafe MR.Vector2d Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_col", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Matrix2d_col(_Underlying *_this, int i);
            return __MR_Matrix2d_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_trace", ExactSpelling = true)]
            extern static double __MR_Matrix2d_trace(_Underlying *_this);
            return __MR_Matrix2d_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_normSq", ExactSpelling = true)]
            extern static double __MR_Matrix2d_normSq(_Underlying *_this);
            return __MR_Matrix2d_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2d::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2d_norm(_Underlying *_this);
            return __MR_Matrix2d_norm(_UnderlyingPtr);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_det", ExactSpelling = true)]
            extern static double __MR_Matrix2d_det(_Underlying *_this);
            return __MR_Matrix2d_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix2d::inverse`.
        public unsafe MR.Matrix2d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_inverse", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_inverse(_Underlying *_this);
            return __MR_Matrix2d_inverse(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2d::transposed`.
        public unsafe MR.Matrix2d Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_transposed", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_transposed(_Underlying *_this);
            return __MR_Matrix2d_transposed(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_equal_MR_Matrix2d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix2d a, MR.Const_Matrix2d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2d operator+(MR.Const_Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_add_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_add_MR_Matrix2d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2d operator-(MR.Const_Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_sub_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_sub_MR_Matrix2d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2d operator*(double a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_mul_double_MR_Matrix2d(double a, MR.Const_Matrix2d._Underlying *b);
            return __MR_mul_double_MR_Matrix2d(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2d operator*(MR.Const_Matrix2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_mul_MR_Matrix2d_double(MR.Const_Matrix2d._Underlying *b, double a);
            return __MR_mul_MR_Matrix2d_double(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix2d operator/(Const_Matrix2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_div_MR_Matrix2d_double(MR.Matrix2d b, double a);
            return __MR_div_MR_Matrix2d_double(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2d operator*(MR.Const_Matrix2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2d_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_mul_MR_Matrix2d_MR_Vector2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_mul_MR_Matrix2d_MR_Vector2d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2d operator*(MR.Const_Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_mul_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_mul_MR_Matrix2d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix2d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix2d)
                return this == (MR.Const_Matrix2d)other;
            return false;
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2d`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix2d : Const_Matrix2d
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix2d UnderlyingStruct => ref *(Matrix2d *)_UnderlyingPtr;

        internal unsafe Mut_Matrix2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector2d X => ref UnderlyingStruct.X;

        public new ref MR.Vector2d Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Matrix2d(Const_Matrix2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2d _ctor_result = __MR_Matrix2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2d::Matrix2d`.
        public unsafe Mut_Matrix2d(MR.Const_Vector2d x, MR.Const_Vector2d y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_Construct", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_Construct(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Matrix2d _ctor_result = __MR_Matrix2d_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Matrix2d::operator[]`.
        public unsafe new MR.Mut_Vector2d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_Matrix2d_index(_Underlying *_this, int row);
            return new(__MR_Matrix2d_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2d AddAssign(MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_add_assign_MR_Matrix2d(_Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return new(__MR_add_assign_MR_Matrix2d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2d SubAssign(MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_sub_assign_MR_Matrix2d(_Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix2d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_mul_assign_MR_Matrix2d_double(_Underlying *a, double b);
            return new(__MR_mul_assign_MR_Matrix2d_double(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_div_assign_MR_Matrix2d_double(_Underlying *a, double b);
            return new(__MR_div_assign_MR_Matrix2d_double(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 2x2 matrix
    /// Generated from class `MR::Matrix2d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
    public struct Matrix2d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix2d(Const_Matrix2d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix2d(Matrix2d other) => new(new Mut_Matrix2d((Mut_Matrix2d._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2d X;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector2d Y;

        /// Generated copy constructor.
        public Matrix2d(Matrix2d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix2d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_DefaultConstruct();
            this = __MR_Matrix2d_DefaultConstruct();
        }

        /// initializes matrix from its 2 rows
        /// Generated from constructor `MR::Matrix2d::Matrix2d`.
        public unsafe Matrix2d(MR.Const_Vector2d x, MR.Const_Vector2d y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_Construct", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_Construct(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            this = __MR_Matrix2d_Construct(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix2d::zero`.
        public static MR.Matrix2d Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_zero", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_zero();
            return __MR_Matrix2d_zero();
        }

        /// Generated from method `MR::Matrix2d::identity`.
        public static MR.Matrix2d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_identity", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_identity();
            return __MR_Matrix2d_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix2d::scale`.
        public static MR.Matrix2d Scale(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_scale_1_double", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_scale_1_double(double s);
            return __MR_Matrix2d_scale_1_double(s);
        }

        /// returns a matrix that has its own scale along each axis
        /// Generated from method `MR::Matrix2d::scale`.
        public static MR.Matrix2d Scale(double sx, double sy)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_scale_2", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_scale_2(double sx, double sy);
            return __MR_Matrix2d_scale_2(sx, sy);
        }

        /// Generated from method `MR::Matrix2d::scale`.
        public static unsafe MR.Matrix2d Scale(MR.Const_Vector2d s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_scale_1_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_scale_1_MR_Vector2d(MR.Const_Vector2d._Underlying *s);
            return __MR_Matrix2d_scale_1_MR_Vector2d(s._UnderlyingPtr);
        }

        /// creates matrix representing rotation around origin on given angle
        /// Generated from method `MR::Matrix2d::rotation`.
        public static MR.Matrix2d Rotation(double angle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_rotation_1", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_rotation_1(double angle);
            return __MR_Matrix2d_rotation_1(angle);
        }

        /// creates matrix representing rotation that after application to (from) makes (to) vector
        /// Generated from method `MR::Matrix2d::rotation`.
        public static unsafe MR.Matrix2d Rotation(MR.Const_Vector2d from, MR.Const_Vector2d to)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_rotation_2", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_rotation_2(MR.Const_Vector2d._Underlying *from, MR.Const_Vector2d._Underlying *to);
            return __MR_Matrix2d_rotation_2(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 rows
        /// Generated from method `MR::Matrix2d::fromRows`.
        public static unsafe MR.Matrix2d FromRows(MR.Const_Vector2d x, MR.Const_Vector2d y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_fromRows", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_fromRows(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            return __MR_Matrix2d_fromRows(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// constructs a matrix from its 2 columns;
        /// use this method to get the matrix that transforms basis vectors ( plusX, plusY ) into vectors ( x, y ) respectively
        /// Generated from method `MR::Matrix2d::fromColumns`.
        public static unsafe MR.Matrix2d FromColumns(MR.Const_Vector2d x, MR.Const_Vector2d y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_fromColumns", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_fromColumns(MR.Const_Vector2d._Underlying *x, MR.Const_Vector2d._Underlying *y);
            return __MR_Matrix2d_fromColumns(x._UnderlyingPtr, y._UnderlyingPtr);
        }

        /// row access
        /// Generated from method `MR::Matrix2d::operator[]`.
        public unsafe MR.Const_Vector2d Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2d._Underlying *__MR_Matrix2d_index_const(MR.Matrix2d *_this, int row);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return new(__MR_Matrix2d_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix2d::operator[]`.
        public unsafe MR.Mut_Vector2d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_Matrix2d_index(MR.Matrix2d *_this, int row);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return new(__MR_Matrix2d_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix2d::col`.
        public unsafe MR.Vector2d Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_col", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Matrix2d_col(MR.Matrix2d *_this, int i);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix2d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_trace", ExactSpelling = true)]
            extern static double __MR_Matrix2d_trace(MR.Matrix2d *_this);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix2d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_normSq", ExactSpelling = true)]
            extern static double __MR_Matrix2d_normSq(MR.Matrix2d *_this);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix2d::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_norm", ExactSpelling = true)]
            extern static double __MR_Matrix2d_norm(MR.Matrix2d *_this);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_norm(__ptr__this);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix2d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_det", ExactSpelling = true)]
            extern static double __MR_Matrix2d_det(MR.Matrix2d *_this);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_det(__ptr__this);
            }
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix2d::inverse`.
        public unsafe MR.Matrix2d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_inverse", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_inverse(MR.Matrix2d *_this);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_inverse(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix2d::transposed`.
        public unsafe MR.Matrix2d Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix2d_transposed", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_Matrix2d_transposed(MR.Matrix2d *_this);
            fixed (MR.Matrix2d *__ptr__this = &this)
            {
                return __MR_Matrix2d_transposed(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix2d a, MR.Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_equal_MR_Matrix2d((MR.Mut_Matrix2d._Underlying *)&a, (MR.Mut_Matrix2d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix2d a, MR.Matrix2d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix2d operator+(MR.Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_add_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_add_MR_Matrix2d((MR.Mut_Matrix2d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix2d operator-(MR.Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_sub_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_sub_MR_Matrix2d((MR.Mut_Matrix2d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2d operator*(double a, MR.Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_mul_double_MR_Matrix2d(double a, MR.Const_Matrix2d._Underlying *b);
            return __MR_mul_double_MR_Matrix2d(a, (MR.Mut_Matrix2d._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2d operator*(MR.Matrix2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_mul_MR_Matrix2d_double(MR.Const_Matrix2d._Underlying *b, double a);
            return __MR_mul_MR_Matrix2d_double((MR.Mut_Matrix2d._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix2d operator/(MR.Matrix2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_div_MR_Matrix2d_double(MR.Matrix2d b, double a);
            return __MR_div_MR_Matrix2d_double(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix2d AddAssign(MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_add_assign_MR_Matrix2d(MR.Matrix2d *a, MR.Const_Matrix2d._Underlying *b);
            fixed (MR.Matrix2d *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix2d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix2d SubAssign(MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_sub_assign_MR_Matrix2d(MR.Matrix2d *a, MR.Const_Matrix2d._Underlying *b);
            fixed (MR.Matrix2d *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix2d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix2d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_mul_assign_MR_Matrix2d_double(MR.Matrix2d *a, double b);
            fixed (MR.Matrix2d *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix2d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix2d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix2d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix2d._Underlying *__MR_div_assign_MR_Matrix2d_double(MR.Matrix2d *a, double b);
            fixed (MR.Matrix2d *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix2d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2d operator*(MR.Matrix2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2d_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_mul_MR_Matrix2d_MR_Vector2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_mul_MR_Matrix2d_MR_Vector2d((MR.Mut_Matrix2d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix2d operator*(MR.Matrix2d a, MR.Const_Matrix2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix2d", ExactSpelling = true)]
            extern static MR.Matrix2d __MR_mul_MR_Matrix2d(MR.Const_Matrix2d._Underlying *a, MR.Const_Matrix2d._Underlying *b);
            return __MR_mul_MR_Matrix2d((MR.Mut_Matrix2d._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix2d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix2d)
                return this == (MR.Matrix2d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix2d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix2d`/`Const_Matrix2d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix2d
    {
        public readonly bool HasValue;
        internal readonly Matrix2d Object;
        public Matrix2d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix2d() {HasValue = false;}
        public _InOpt_Matrix2d(Matrix2d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix2d(Matrix2d new_value) {return new(new_value);}
        public _InOpt_Matrix2d(Const_Matrix2d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix2d(Const_Matrix2d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2d`/`Const_Matrix2d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2d`.
    public class _InOptMut_Matrix2d
    {
        public Mut_Matrix2d? Opt;

        public _InOptMut_Matrix2d() {}
        public _InOptMut_Matrix2d(Mut_Matrix2d value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix2d(Mut_Matrix2d value) {return new(value);}
        public unsafe _InOptMut_Matrix2d(ref Matrix2d value)
        {
            fixed (Matrix2d *value_ptr = &value)
            {
                Opt = new((Const_Matrix2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix2d`/`Const_Matrix2d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix2d`.
    public class _InOptConst_Matrix2d
    {
        public Const_Matrix2d? Opt;

        public _InOptConst_Matrix2d() {}
        public _InOptConst_Matrix2d(Const_Matrix2d value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix2d(Const_Matrix2d value) {return new(value);}
        public unsafe _InOptConst_Matrix2d(ref readonly Matrix2d value)
        {
            fixed (Matrix2d *value_ptr = &value)
            {
                Opt = new((Const_Matrix2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }
}
