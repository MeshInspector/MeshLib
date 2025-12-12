public static partial class MR
{
    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4b`.
    /// This is the const reference to the struct.
    public class Const_Matrix4b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix4b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix4b UnderlyingStruct => ref *(Matrix4b *)_UnderlyingPtr;

        internal unsafe Const_Matrix4b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix4b_Destroy(_Underlying *_this);
            __MR_Matrix4b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix4b() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector4b X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector4b Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector4b Z => ref UnderlyingStruct.Z;

        public ref readonly MR.Vector4b W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Const_Matrix4b(Const_Matrix4b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix4b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix4b _ctor_result = __MR_Matrix4b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4b::Matrix4b`.
        public unsafe Const_Matrix4b(MR.Const_Vector4b x, MR.Const_Vector4b y, MR.Const_Vector4b z, MR.Const_Vector4b w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_Construct_4(MR.Const_Vector4b._Underlying *x, MR.Const_Vector4b._Underlying *y, MR.Const_Vector4b._Underlying *z, MR.Const_Vector4b._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix4b _ctor_result = __MR_Matrix4b_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4b::Matrix4b`.
        public unsafe Const_Matrix4b(MR.Const_Matrix3b r, MR.Const_Vector3b t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_Construct_2(MR.Const_Matrix3b._Underlying *r, MR.Const_Vector3b._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix4b _ctor_result = __MR_Matrix4b_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Matrix4b::zero`.
        public static MR.Matrix4b Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_zero", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_zero();
            return __MR_Matrix4b_zero();
        }

        /// Generated from method `MR::Matrix4b::identity`.
        public static MR.Matrix4b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_identity", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_identity();
            return __MR_Matrix4b_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4b::scale`.
        public static MR.Matrix4b Scale(bool s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_scale", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_scale(byte s);
            return __MR_Matrix4b_scale(s ? (byte)1 : (byte)0);
        }

        /// element access
        /// Generated from method `MR::Matrix4b::operator()`.
        public unsafe bool Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_call_const", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_call_const(_Underlying *_this, int row, int col);
            return *__MR_Matrix4b_call_const(_UnderlyingPtr, row, col);
        }

        /// row access
        /// Generated from method `MR::Matrix4b::operator[]`.
        public unsafe MR.Const_Vector4b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4b._Underlying *__MR_Matrix4b_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix4b_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix4b::col`.
        public unsafe MR.Vector4b Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_col", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Matrix4b_col(_Underlying *_this, int i);
            return __MR_Matrix4b_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix4b_trace(_Underlying *_this);
            return __MR_Matrix4b_trace(_UnderlyingPtr) != 0;
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix4b_normSq(_Underlying *_this);
            return __MR_Matrix4b_normSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Matrix4b::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4b_norm(_Underlying *_this);
            return __MR_Matrix4b_norm(_UnderlyingPtr);
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4b::submatrix3`.
        public unsafe MR.Matrix3b Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix4b_submatrix3(_Underlying *_this, int i, int j);
            return __MR_Matrix4b_submatrix3(_UnderlyingPtr, i, j);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_det", ExactSpelling = true)]
            extern static byte __MR_Matrix4b_det(_Underlying *_this);
            return __MR_Matrix4b_det(_UnderlyingPtr) != 0;
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4b::transposed`.
        public unsafe MR.Matrix4b Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_transposed", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_transposed(_Underlying *_this);
            return __MR_Matrix4b_transposed(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4b::getRotation`.
        public unsafe MR.Matrix3b GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix4b_getRotation(_Underlying *_this);
            return __MR_Matrix4b_getRotation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4b::getTranslation`.
        public unsafe MR.Vector3b GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Matrix4b_getTranslation(_Underlying *_this);
            return __MR_Matrix4b_getTranslation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4b::data`.
        public unsafe bool? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_data_const", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_data_const(_Underlying *_this);
            var __ret = __MR_Matrix4b_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_equal_MR_Matrix4b(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix4b a, MR.Const_Matrix4b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i operator+(MR.Const_Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_add_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_add_MR_Matrix4b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i operator-(MR.Const_Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_sub_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_sub_MR_Matrix4b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(bool a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_bool_MR_Matrix4b(byte a, MR.Const_Matrix4b._Underlying *b);
            return __MR_mul_bool_MR_Matrix4b(a ? (byte)1 : (byte)0, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Const_Matrix4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4b_bool(MR.Const_Matrix4b._Underlying *b, byte a);
            return __MR_mul_MR_Matrix4b_bool(b._UnderlyingPtr, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix4i operator/(Const_Matrix4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_div_MR_Matrix4b_bool(MR.Matrix4b b, byte a);
            return __MR_div_MR_Matrix4b_bool(b.UnderlyingStruct, a ? (byte)1 : (byte)0);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Const_Matrix4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4b_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Matrix4b_MR_Vector4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_mul_MR_Matrix4b_MR_Vector4b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Const_Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_mul_MR_Matrix4b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix4b? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix4b)
                return this == (MR.Const_Matrix4b)other;
            return false;
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4b`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix4b : Const_Matrix4b
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix4b UnderlyingStruct => ref *(Matrix4b *)_UnderlyingPtr;

        internal unsafe Mut_Matrix4b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector4b X => ref UnderlyingStruct.X;

        public new ref MR.Vector4b Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector4b Z => ref UnderlyingStruct.Z;

        public new ref MR.Vector4b W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Matrix4b(Const_Matrix4b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix4b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix4b _ctor_result = __MR_Matrix4b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4b::Matrix4b`.
        public unsafe Mut_Matrix4b(MR.Const_Vector4b x, MR.Const_Vector4b y, MR.Const_Vector4b z, MR.Const_Vector4b w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_Construct_4(MR.Const_Vector4b._Underlying *x, MR.Const_Vector4b._Underlying *y, MR.Const_Vector4b._Underlying *z, MR.Const_Vector4b._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix4b _ctor_result = __MR_Matrix4b_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4b::Matrix4b`.
        public unsafe Mut_Matrix4b(MR.Const_Matrix3b r, MR.Const_Vector3b t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_Construct_2(MR.Const_Matrix3b._Underlying *r, MR.Const_Vector3b._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Matrix4b _ctor_result = __MR_Matrix4b_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Matrix4b::operator()`.
        public unsafe new ref bool Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_call", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_call(_Underlying *_this, int row, int col);
            return ref *__MR_Matrix4b_call(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix4b::operator[]`.
        public unsafe new MR.Mut_Vector4b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_Matrix4b_index(_Underlying *_this, int row);
            return new(__MR_Matrix4b_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from method `MR::Matrix4b::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3b rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4b_setRotation(_Underlying *_this, MR.Const_Matrix3b._Underlying *rot);
            __MR_Matrix4b_setRotation(_UnderlyingPtr, rot._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4b::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3b t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4b_setTranslation(_Underlying *_this, MR.Const_Vector3b._Underlying *t);
            __MR_Matrix4b_setTranslation(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4b::data`.
        public unsafe new MR.Misc.Ref<bool>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_data", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_data(_Underlying *_this);
            var __ret = __MR_Matrix4b_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<bool>(__ret) : null;
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4b AddAssign(MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_add_assign_MR_Matrix4b(_Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return new(__MR_add_assign_MR_Matrix4b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4b SubAssign(MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_sub_assign_MR_Matrix4b(_Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix4b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_mul_assign_MR_Matrix4b_bool(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Matrix4b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_div_assign_MR_Matrix4b_bool(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Matrix4b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4b`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Matrix4b
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix4b(Const_Matrix4b other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix4b(Matrix4b other) => new(new Mut_Matrix4b((Mut_Matrix4b._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector4b X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public MR.Vector4b Y;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public MR.Vector4b Z;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public MR.Vector4b W;

        /// Generated copy constructor.
        public Matrix4b(Matrix4b _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix4b()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_DefaultConstruct();
            this = __MR_Matrix4b_DefaultConstruct();
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4b::Matrix4b`.
        public unsafe Matrix4b(MR.Const_Vector4b x, MR.Const_Vector4b y, MR.Const_Vector4b z, MR.Const_Vector4b w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_Construct_4(MR.Const_Vector4b._Underlying *x, MR.Const_Vector4b._Underlying *y, MR.Const_Vector4b._Underlying *z, MR.Const_Vector4b._Underlying *w);
            this = __MR_Matrix4b_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4b::Matrix4b`.
        public unsafe Matrix4b(MR.Const_Matrix3b r, MR.Const_Vector3b t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_Construct_2(MR.Const_Matrix3b._Underlying *r, MR.Const_Vector3b._Underlying *t);
            this = __MR_Matrix4b_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4b::zero`.
        public static MR.Matrix4b Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_zero", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_zero();
            return __MR_Matrix4b_zero();
        }

        /// Generated from method `MR::Matrix4b::identity`.
        public static MR.Matrix4b Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_identity", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_identity();
            return __MR_Matrix4b_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4b::scale`.
        public static MR.Matrix4b Scale(bool s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_scale", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_scale(byte s);
            return __MR_Matrix4b_scale(s ? (byte)1 : (byte)0);
        }

        /// element access
        /// Generated from method `MR::Matrix4b::operator()`.
        public unsafe bool Call_Const(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_call_const", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_call_const(MR.Matrix4b *_this, int row, int col);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return *__MR_Matrix4b_call_const(__ptr__this, row, col);
            }
        }

        /// Generated from method `MR::Matrix4b::operator()`.
        public unsafe ref bool Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_call", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_call(MR.Matrix4b *_this, int row, int col);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return ref *__MR_Matrix4b_call(__ptr__this, row, col);
            }
        }

        /// row access
        /// Generated from method `MR::Matrix4b::operator[]`.
        public unsafe MR.Const_Vector4b Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4b._Underlying *__MR_Matrix4b_index_const(MR.Matrix4b *_this, int row);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return new(__MR_Matrix4b_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix4b::operator[]`.
        public unsafe MR.Mut_Vector4b Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_Matrix4b_index(MR.Matrix4b *_this, int row);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return new(__MR_Matrix4b_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix4b::col`.
        public unsafe MR.Vector4b Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_col", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Matrix4b_col(MR.Matrix4b *_this, int i);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4b::trace`.
        public unsafe bool Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix4b_trace(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_trace(__ptr__this) != 0;
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4b::normSq`.
        public unsafe bool NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix4b_normSq(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_normSq(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::Matrix4b::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4b_norm(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_norm(__ptr__this);
            }
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4b::submatrix3`.
        public unsafe MR.Matrix3b Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix4b_submatrix3(MR.Matrix4b *_this, int i, int j);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_submatrix3(__ptr__this, i, j);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4b::det`.
        public unsafe bool Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_det", ExactSpelling = true)]
            extern static byte __MR_Matrix4b_det(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_det(__ptr__this) != 0;
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4b::transposed`.
        public unsafe MR.Matrix4b Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_transposed", ExactSpelling = true)]
            extern static MR.Matrix4b __MR_Matrix4b_transposed(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_transposed(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4b::getRotation`.
        public unsafe MR.Matrix3b GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3b __MR_Matrix4b_getRotation(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_getRotation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4b::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3b rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4b_setRotation(MR.Matrix4b *_this, MR.Const_Matrix3b._Underlying *rot);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                __MR_Matrix4b_setRotation(__ptr__this, rot._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4b::getTranslation`.
        public unsafe MR.Vector3b GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Matrix4b_getTranslation(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                return __MR_Matrix4b_getTranslation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4b::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3b t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4b_setTranslation(MR.Matrix4b *_this, MR.Const_Vector3b._Underlying *t);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                __MR_Matrix4b_setTranslation(__ptr__this, t._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4b::data`.
        public unsafe MR.Misc.Ref<bool>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_data", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_data(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4b_data(__ptr__this);
                return __ret is not null ? new MR.Misc.Ref<bool>(__ret) : null;
            }
        }

        /// Generated from method `MR::Matrix4b::data`.
        public unsafe bool? Data_Const()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4b_data_const", ExactSpelling = true)]
            extern static bool *__MR_Matrix4b_data_const(MR.Matrix4b *_this);
            fixed (MR.Matrix4b *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4b_data_const(__ptr__this);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix4b a, MR.Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_equal_MR_Matrix4b((MR.Mut_Matrix4b._Underlying *)&a, (MR.Mut_Matrix4b._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix4b a, MR.Matrix4b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i operator+(MR.Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_add_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_add_MR_Matrix4b((MR.Mut_Matrix4b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i operator-(MR.Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_sub_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_sub_MR_Matrix4b((MR.Mut_Matrix4b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(bool a, MR.Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_bool_MR_Matrix4b(byte a, MR.Const_Matrix4b._Underlying *b);
            return __MR_mul_bool_MR_Matrix4b(a ? (byte)1 : (byte)0, (MR.Mut_Matrix4b._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Matrix4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4b_bool(MR.Const_Matrix4b._Underlying *b, byte a);
            return __MR_mul_MR_Matrix4b_bool((MR.Mut_Matrix4b._Underlying *)&b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix4i operator/(MR.Matrix4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_div_MR_Matrix4b_bool(MR.Matrix4b b, byte a);
            return __MR_div_MR_Matrix4b_bool(b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4b AddAssign(MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_add_assign_MR_Matrix4b(MR.Matrix4b *a, MR.Const_Matrix4b._Underlying *b);
            fixed (MR.Matrix4b *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix4b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4b SubAssign(MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_sub_assign_MR_Matrix4b(MR.Matrix4b *a, MR.Const_Matrix4b._Underlying *b);
            fixed (MR.Matrix4b *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix4b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_mul_assign_MR_Matrix4b_bool(MR.Matrix4b *a, byte b);
            fixed (MR.Matrix4b *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix4b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Matrix4b._Underlying *__MR_div_assign_MR_Matrix4b_bool(MR.Matrix4b *a, byte b);
            fixed (MR.Matrix4b *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix4b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Matrix4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4b_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Matrix4b_MR_Vector4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_mul_MR_Matrix4b_MR_Vector4b((MR.Mut_Matrix4b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Matrix4b a, MR.Const_Matrix4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4b", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4b(MR.Const_Matrix4b._Underlying *a, MR.Const_Matrix4b._Underlying *b);
            return __MR_mul_MR_Matrix4b((MR.Mut_Matrix4b._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix4b b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix4b)
                return this == (MR.Matrix4b)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix4b` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix4b`/`Const_Matrix4b` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix4b
    {
        public readonly bool HasValue;
        internal readonly Matrix4b Object;
        public Matrix4b Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix4b() {HasValue = false;}
        public _InOpt_Matrix4b(Matrix4b new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix4b(Matrix4b new_value) {return new(new_value);}
        public _InOpt_Matrix4b(Const_Matrix4b new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix4b(Const_Matrix4b new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix4b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix4b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4b`/`Const_Matrix4b` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4b`.
    public class _InOptMut_Matrix4b
    {
        public Mut_Matrix4b? Opt;

        public _InOptMut_Matrix4b() {}
        public _InOptMut_Matrix4b(Mut_Matrix4b value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix4b(Mut_Matrix4b value) {return new(value);}
        public unsafe _InOptMut_Matrix4b(ref Matrix4b value)
        {
            fixed (Matrix4b *value_ptr = &value)
            {
                Opt = new((Const_Matrix4b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix4b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix4b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4b`/`Const_Matrix4b` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4b`.
    public class _InOptConst_Matrix4b
    {
        public Const_Matrix4b? Opt;

        public _InOptConst_Matrix4b() {}
        public _InOptConst_Matrix4b(Const_Matrix4b value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix4b(Const_Matrix4b value) {return new(value);}
        public unsafe _InOptConst_Matrix4b(ref readonly Matrix4b value)
        {
            fixed (Matrix4b *value_ptr = &value)
            {
                Opt = new((Const_Matrix4b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4i`.
    /// This is the const reference to the struct.
    public class Const_Matrix4i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix4i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix4i UnderlyingStruct => ref *(Matrix4i *)_UnderlyingPtr;

        internal unsafe Const_Matrix4i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix4i_Destroy(_Underlying *_this);
            __MR_Matrix4i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix4i() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector4i X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector4i Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector4i Z => ref UnderlyingStruct.Z;

        public ref readonly MR.Vector4i W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Const_Matrix4i(Const_Matrix4i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 64);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix4i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4i _ctor_result = __MR_Matrix4i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4i::Matrix4i`.
        public unsafe Const_Matrix4i(MR.Const_Vector4i x, MR.Const_Vector4i y, MR.Const_Vector4i z, MR.Const_Vector4i w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_Construct_4(MR.Const_Vector4i._Underlying *x, MR.Const_Vector4i._Underlying *y, MR.Const_Vector4i._Underlying *z, MR.Const_Vector4i._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4i _ctor_result = __MR_Matrix4i_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4i::Matrix4i`.
        public unsafe Const_Matrix4i(MR.Const_Matrix3i r, MR.Const_Vector3i t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_Construct_2(MR.Const_Matrix3i._Underlying *r, MR.Const_Vector3i._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4i _ctor_result = __MR_Matrix4i_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// Generated from method `MR::Matrix4i::zero`.
        public static MR.Matrix4i Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_zero", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_zero();
            return __MR_Matrix4i_zero();
        }

        /// Generated from method `MR::Matrix4i::identity`.
        public static MR.Matrix4i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_identity", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_identity();
            return __MR_Matrix4i_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4i::scale`.
        public static MR.Matrix4i Scale(int s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_scale", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_scale(int s);
            return __MR_Matrix4i_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4i::operator()`.
        public unsafe int Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_call_const", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_call_const(_Underlying *_this, int row, int col);
            return *__MR_Matrix4i_call_const(_UnderlyingPtr, row, col);
        }

        /// row access
        /// Generated from method `MR::Matrix4i::operator[]`.
        public unsafe MR.Const_Vector4i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4i._Underlying *__MR_Matrix4i_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix4i_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix4i::col`.
        public unsafe MR.Vector4i Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_col", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Matrix4i_col(_Underlying *_this, int i);
            return __MR_Matrix4i_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_trace", ExactSpelling = true)]
            extern static int __MR_Matrix4i_trace(_Underlying *_this);
            return __MR_Matrix4i_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_normSq", ExactSpelling = true)]
            extern static int __MR_Matrix4i_normSq(_Underlying *_this);
            return __MR_Matrix4i_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4i_norm(_Underlying *_this);
            return __MR_Matrix4i_norm(_UnderlyingPtr);
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4i::submatrix3`.
        public unsafe MR.Matrix3i Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix4i_submatrix3(_Underlying *_this, int i, int j);
            return __MR_Matrix4i_submatrix3(_UnderlyingPtr, i, j);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_det", ExactSpelling = true)]
            extern static int __MR_Matrix4i_det(_Underlying *_this);
            return __MR_Matrix4i_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4i::transposed`.
        public unsafe MR.Matrix4i Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_transposed", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_transposed(_Underlying *_this);
            return __MR_Matrix4i_transposed(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::getRotation`.
        public unsafe MR.Matrix3i GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix4i_getRotation(_Underlying *_this);
            return __MR_Matrix4i_getRotation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::getTranslation`.
        public unsafe MR.Vector3i GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Matrix4i_getTranslation(_Underlying *_this);
            return __MR_Matrix4i_getTranslation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::data`.
        public unsafe int? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_data_const", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_data_const(_Underlying *_this);
            var __ret = __MR_Matrix4i_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_equal_MR_Matrix4i(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix4i a, MR.Const_Matrix4i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i operator+(MR.Const_Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_add_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_add_MR_Matrix4i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i operator-(MR.Const_Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_sub_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_sub_MR_Matrix4i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(int a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_int_MR_Matrix4i(int a, MR.Const_Matrix4i._Underlying *b);
            return __MR_mul_int_MR_Matrix4i(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Const_Matrix4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4i_int(MR.Const_Matrix4i._Underlying *b, int a);
            return __MR_mul_MR_Matrix4i_int(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix4i operator/(Const_Matrix4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_div_MR_Matrix4i_int(MR.Matrix4i b, int a);
            return __MR_div_MR_Matrix4i_int(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Const_Matrix4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Matrix4i_MR_Vector4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_mul_MR_Matrix4i_MR_Vector4i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Const_Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_mul_MR_Matrix4i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix4i? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix4i)
                return this == (MR.Const_Matrix4i)other;
            return false;
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4i`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix4i : Const_Matrix4i
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix4i UnderlyingStruct => ref *(Matrix4i *)_UnderlyingPtr;

        internal unsafe Mut_Matrix4i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector4i X => ref UnderlyingStruct.X;

        public new ref MR.Vector4i Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector4i Z => ref UnderlyingStruct.Z;

        public new ref MR.Vector4i W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Matrix4i(Const_Matrix4i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 64);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix4i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4i _ctor_result = __MR_Matrix4i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4i::Matrix4i`.
        public unsafe Mut_Matrix4i(MR.Const_Vector4i x, MR.Const_Vector4i y, MR.Const_Vector4i z, MR.Const_Vector4i w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_Construct_4(MR.Const_Vector4i._Underlying *x, MR.Const_Vector4i._Underlying *y, MR.Const_Vector4i._Underlying *z, MR.Const_Vector4i._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4i _ctor_result = __MR_Matrix4i_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4i::Matrix4i`.
        public unsafe Mut_Matrix4i(MR.Const_Matrix3i r, MR.Const_Vector3i t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_Construct_2(MR.Const_Matrix3i._Underlying *r, MR.Const_Vector3i._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4i _ctor_result = __MR_Matrix4i_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// Generated from method `MR::Matrix4i::operator()`.
        public unsafe new ref int Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_call", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_call(_Underlying *_this, int row, int col);
            return ref *__MR_Matrix4i_call(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix4i::operator[]`.
        public unsafe new MR.Mut_Vector4i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_Matrix4i_index(_Underlying *_this, int row);
            return new(__MR_Matrix4i_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from method `MR::Matrix4i::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3i rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4i_setRotation(_Underlying *_this, MR.Const_Matrix3i._Underlying *rot);
            __MR_Matrix4i_setRotation(_UnderlyingPtr, rot._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3i t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4i_setTranslation(_Underlying *_this, MR.Const_Vector3i._Underlying *t);
            __MR_Matrix4i_setTranslation(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::data`.
        public unsafe new MR.Misc.Ref<int>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_data", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_data(_Underlying *_this);
            var __ret = __MR_Matrix4i_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<int>(__ret) : null;
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4i AddAssign(MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_add_assign_MR_Matrix4i(_Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return new(__MR_add_assign_MR_Matrix4i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4i SubAssign(MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_sub_assign_MR_Matrix4i(_Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix4i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_mul_assign_MR_Matrix4i_int(_Underlying *a, int b);
            return new(__MR_mul_assign_MR_Matrix4i_int(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_div_assign_MR_Matrix4i_int(_Underlying *a, int b);
            return new(__MR_div_assign_MR_Matrix4i_int(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 64)]
    public struct Matrix4i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix4i(Const_Matrix4i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix4i(Matrix4i other) => new(new Mut_Matrix4i((Mut_Matrix4i._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector4i X;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector4i Y;

        [System.Runtime.InteropServices.FieldOffset(32)]
        public MR.Vector4i Z;

        [System.Runtime.InteropServices.FieldOffset(48)]
        public MR.Vector4i W;

        /// Generated copy constructor.
        public Matrix4i(Matrix4i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix4i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_DefaultConstruct();
            this = __MR_Matrix4i_DefaultConstruct();
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4i::Matrix4i`.
        public unsafe Matrix4i(MR.Const_Vector4i x, MR.Const_Vector4i y, MR.Const_Vector4i z, MR.Const_Vector4i w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_Construct_4(MR.Const_Vector4i._Underlying *x, MR.Const_Vector4i._Underlying *y, MR.Const_Vector4i._Underlying *z, MR.Const_Vector4i._Underlying *w);
            this = __MR_Matrix4i_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4i::Matrix4i`.
        public unsafe Matrix4i(MR.Const_Matrix3i r, MR.Const_Vector3i t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_Construct_2(MR.Const_Matrix3i._Underlying *r, MR.Const_Vector3i._Underlying *t);
            this = __MR_Matrix4i_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i::zero`.
        public static MR.Matrix4i Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_zero", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_zero();
            return __MR_Matrix4i_zero();
        }

        /// Generated from method `MR::Matrix4i::identity`.
        public static MR.Matrix4i Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_identity", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_identity();
            return __MR_Matrix4i_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4i::scale`.
        public static MR.Matrix4i Scale(int s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_scale", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_scale(int s);
            return __MR_Matrix4i_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4i::operator()`.
        public unsafe int Call_Const(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_call_const", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_call_const(MR.Matrix4i *_this, int row, int col);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return *__MR_Matrix4i_call_const(__ptr__this, row, col);
            }
        }

        /// Generated from method `MR::Matrix4i::operator()`.
        public unsafe ref int Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_call", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_call(MR.Matrix4i *_this, int row, int col);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return ref *__MR_Matrix4i_call(__ptr__this, row, col);
            }
        }

        /// row access
        /// Generated from method `MR::Matrix4i::operator[]`.
        public unsafe MR.Const_Vector4i Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4i._Underlying *__MR_Matrix4i_index_const(MR.Matrix4i *_this, int row);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return new(__MR_Matrix4i_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix4i::operator[]`.
        public unsafe MR.Mut_Vector4i Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_Matrix4i_index(MR.Matrix4i *_this, int row);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return new(__MR_Matrix4i_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix4i::col`.
        public unsafe MR.Vector4i Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_col", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Matrix4i_col(MR.Matrix4i *_this, int i);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4i::trace`.
        public unsafe int Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_trace", ExactSpelling = true)]
            extern static int __MR_Matrix4i_trace(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4i::normSq`.
        public unsafe int NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_normSq", ExactSpelling = true)]
            extern static int __MR_Matrix4i_normSq(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4i_norm(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_norm(__ptr__this);
            }
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4i::submatrix3`.
        public unsafe MR.Matrix3i Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix4i_submatrix3(MR.Matrix4i *_this, int i, int j);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_submatrix3(__ptr__this, i, j);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4i::det`.
        public unsafe int Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_det", ExactSpelling = true)]
            extern static int __MR_Matrix4i_det(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_det(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4i::transposed`.
        public unsafe MR.Matrix4i Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_transposed", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_Matrix4i_transposed(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_transposed(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i::getRotation`.
        public unsafe MR.Matrix3i GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3i __MR_Matrix4i_getRotation(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_getRotation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3i rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4i_setRotation(MR.Matrix4i *_this, MR.Const_Matrix3i._Underlying *rot);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                __MR_Matrix4i_setRotation(__ptr__this, rot._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4i::getTranslation`.
        public unsafe MR.Vector3i GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Matrix4i_getTranslation(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                return __MR_Matrix4i_getTranslation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3i t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4i_setTranslation(MR.Matrix4i *_this, MR.Const_Vector3i._Underlying *t);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                __MR_Matrix4i_setTranslation(__ptr__this, t._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4i::data`.
        public unsafe MR.Misc.Ref<int>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_data", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_data(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4i_data(__ptr__this);
                return __ret is not null ? new MR.Misc.Ref<int>(__ret) : null;
            }
        }

        /// Generated from method `MR::Matrix4i::data`.
        public unsafe int? Data_Const()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i_data_const", ExactSpelling = true)]
            extern static int *__MR_Matrix4i_data_const(MR.Matrix4i *_this);
            fixed (MR.Matrix4i *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4i_data_const(__ptr__this);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix4i a, MR.Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_equal_MR_Matrix4i((MR.Mut_Matrix4i._Underlying *)&a, (MR.Mut_Matrix4i._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix4i a, MR.Matrix4i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i operator+(MR.Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_add_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_add_MR_Matrix4i((MR.Mut_Matrix4i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i operator-(MR.Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_sub_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_sub_MR_Matrix4i((MR.Mut_Matrix4i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(int a, MR.Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_int_MR_Matrix4i(int a, MR.Const_Matrix4i._Underlying *b);
            return __MR_mul_int_MR_Matrix4i(a, (MR.Mut_Matrix4i._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Matrix4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4i_int(MR.Const_Matrix4i._Underlying *b, int a);
            return __MR_mul_MR_Matrix4i_int((MR.Mut_Matrix4i._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix4i operator/(MR.Matrix4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_div_MR_Matrix4i_int(MR.Matrix4i b, int a);
            return __MR_div_MR_Matrix4i_int(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4i AddAssign(MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_add_assign_MR_Matrix4i(MR.Matrix4i *a, MR.Const_Matrix4i._Underlying *b);
            fixed (MR.Matrix4i *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix4i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4i SubAssign(MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_sub_assign_MR_Matrix4i(MR.Matrix4i *a, MR.Const_Matrix4i._Underlying *b);
            fixed (MR.Matrix4i *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix4i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_mul_assign_MR_Matrix4i_int(MR.Matrix4i *a, int b);
            fixed (MR.Matrix4i *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix4i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4i_int", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i._Underlying *__MR_div_assign_MR_Matrix4i_int(MR.Matrix4i *a, int b);
            fixed (MR.Matrix4i *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix4i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Matrix4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Matrix4i_MR_Vector4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_mul_MR_Matrix4i_MR_Vector4i((MR.Mut_Matrix4i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Matrix4i a, MR.Const_Matrix4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4i(MR.Const_Matrix4i._Underlying *a, MR.Const_Matrix4i._Underlying *b);
            return __MR_mul_MR_Matrix4i((MR.Mut_Matrix4i._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix4i b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix4i)
                return this == (MR.Matrix4i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix4i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix4i`/`Const_Matrix4i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix4i
    {
        public readonly bool HasValue;
        internal readonly Matrix4i Object;
        public Matrix4i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix4i() {HasValue = false;}
        public _InOpt_Matrix4i(Matrix4i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix4i(Matrix4i new_value) {return new(new_value);}
        public _InOpt_Matrix4i(Const_Matrix4i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix4i(Const_Matrix4i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix4i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix4i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4i`/`Const_Matrix4i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4i`.
    public class _InOptMut_Matrix4i
    {
        public Mut_Matrix4i? Opt;

        public _InOptMut_Matrix4i() {}
        public _InOptMut_Matrix4i(Mut_Matrix4i value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix4i(Mut_Matrix4i value) {return new(value);}
        public unsafe _InOptMut_Matrix4i(ref Matrix4i value)
        {
            fixed (Matrix4i *value_ptr = &value)
            {
                Opt = new((Const_Matrix4i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix4i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix4i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4i`/`Const_Matrix4i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4i`.
    public class _InOptConst_Matrix4i
    {
        public Const_Matrix4i? Opt;

        public _InOptConst_Matrix4i() {}
        public _InOptConst_Matrix4i(Const_Matrix4i value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix4i(Const_Matrix4i value) {return new(value);}
        public unsafe _InOptConst_Matrix4i(ref readonly Matrix4i value)
        {
            fixed (Matrix4i *value_ptr = &value)
            {
                Opt = new((Const_Matrix4i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4i64`.
    /// This is the const reference to the struct.
    public class Const_Matrix4i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix4i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix4i64 UnderlyingStruct => ref *(Matrix4i64 *)_UnderlyingPtr;

        internal unsafe Const_Matrix4i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix4i64_Destroy(_Underlying *_this);
            __MR_Matrix4i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix4i64() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector4i64 X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector4i64 Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector4i64 Z => ref UnderlyingStruct.Z;

        public ref readonly MR.Vector4i64 W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Const_Matrix4i64(Const_Matrix4i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 128);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix4i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4i64 _ctor_result = __MR_Matrix4i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4i64::Matrix4i64`.
        public unsafe Const_Matrix4i64(MR.Const_Vector4i64 x, MR.Const_Vector4i64 y, MR.Const_Vector4i64 z, MR.Const_Vector4i64 w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_Construct_4(MR.Const_Vector4i64._Underlying *x, MR.Const_Vector4i64._Underlying *y, MR.Const_Vector4i64._Underlying *z, MR.Const_Vector4i64._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4i64 _ctor_result = __MR_Matrix4i64_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4i64::Matrix4i64`.
        public unsafe Const_Matrix4i64(MR.Const_Matrix3i64 r, MR.Const_Vector3i64 t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_Construct_2(MR.Const_Matrix3i64._Underlying *r, MR.Const_Vector3i64._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4i64 _ctor_result = __MR_Matrix4i64_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// Generated from method `MR::Matrix4i64::zero`.
        public static MR.Matrix4i64 Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_zero", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_zero();
            return __MR_Matrix4i64_zero();
        }

        /// Generated from method `MR::Matrix4i64::identity`.
        public static MR.Matrix4i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_identity", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_identity();
            return __MR_Matrix4i64_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4i64::scale`.
        public static MR.Matrix4i64 Scale(long s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_scale", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_scale(long s);
            return __MR_Matrix4i64_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4i64::operator()`.
        public unsafe long Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_call_const", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_call_const(_Underlying *_this, int row, int col);
            return *__MR_Matrix4i64_call_const(_UnderlyingPtr, row, col);
        }

        /// row access
        /// Generated from method `MR::Matrix4i64::operator[]`.
        public unsafe MR.Const_Vector4i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4i64._Underlying *__MR_Matrix4i64_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix4i64_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix4i64::col`.
        public unsafe MR.Vector4i64 Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_col", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Matrix4i64_col(_Underlying *_this, int i);
            return __MR_Matrix4i64_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_trace", ExactSpelling = true)]
            extern static long __MR_Matrix4i64_trace(_Underlying *_this);
            return __MR_Matrix4i64_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_normSq", ExactSpelling = true)]
            extern static long __MR_Matrix4i64_normSq(_Underlying *_this);
            return __MR_Matrix4i64_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4i64_norm(_Underlying *_this);
            return __MR_Matrix4i64_norm(_UnderlyingPtr);
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4i64::submatrix3`.
        public unsafe MR.Matrix3i64 Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix4i64_submatrix3(_Underlying *_this, int i, int j);
            return __MR_Matrix4i64_submatrix3(_UnderlyingPtr, i, j);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_det", ExactSpelling = true)]
            extern static long __MR_Matrix4i64_det(_Underlying *_this);
            return __MR_Matrix4i64_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4i64::transposed`.
        public unsafe MR.Matrix4i64 Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_transposed", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_transposed(_Underlying *_this);
            return __MR_Matrix4i64_transposed(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::getRotation`.
        public unsafe MR.Matrix3i64 GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix4i64_getRotation(_Underlying *_this);
            return __MR_Matrix4i64_getRotation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::getTranslation`.
        public unsafe MR.Vector3i64 GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Matrix4i64_getTranslation(_Underlying *_this);
            return __MR_Matrix4i64_getTranslation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::data`.
        public unsafe long? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_data_const", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_data_const(_Underlying *_this);
            var __ret = __MR_Matrix4i64_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_equal_MR_Matrix4i64(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i64 operator+(MR.Const_Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_add_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_add_MR_Matrix4i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i64 operator-(MR.Const_Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_sub_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_sub_MR_Matrix4i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i64 operator*(long a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_mul_int64_t_MR_Matrix4i64(long a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_mul_int64_t_MR_Matrix4i64(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i64 operator*(MR.Const_Matrix4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_mul_MR_Matrix4i64_int64_t(MR.Const_Matrix4i64._Underlying *b, long a);
            return __MR_mul_MR_Matrix4i64_int64_t(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix4i64 operator/(Const_Matrix4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_div_MR_Matrix4i64_int64_t(MR.Matrix4i64 b, long a);
            return __MR_div_MR_Matrix4i64_int64_t(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i64 operator*(MR.Const_Matrix4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i64_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_mul_MR_Matrix4i64_MR_Vector4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_mul_MR_Matrix4i64_MR_Vector4i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i64 operator*(MR.Const_Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_mul_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_mul_MR_Matrix4i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix4i64? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix4i64)
                return this == (MR.Const_Matrix4i64)other;
            return false;
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix4i64 : Const_Matrix4i64
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix4i64 UnderlyingStruct => ref *(Matrix4i64 *)_UnderlyingPtr;

        internal unsafe Mut_Matrix4i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector4i64 X => ref UnderlyingStruct.X;

        public new ref MR.Vector4i64 Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector4i64 Z => ref UnderlyingStruct.Z;

        public new ref MR.Vector4i64 W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Matrix4i64(Const_Matrix4i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 128);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix4i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4i64 _ctor_result = __MR_Matrix4i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4i64::Matrix4i64`.
        public unsafe Mut_Matrix4i64(MR.Const_Vector4i64 x, MR.Const_Vector4i64 y, MR.Const_Vector4i64 z, MR.Const_Vector4i64 w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_Construct_4(MR.Const_Vector4i64._Underlying *x, MR.Const_Vector4i64._Underlying *y, MR.Const_Vector4i64._Underlying *z, MR.Const_Vector4i64._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4i64 _ctor_result = __MR_Matrix4i64_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4i64::Matrix4i64`.
        public unsafe Mut_Matrix4i64(MR.Const_Matrix3i64 r, MR.Const_Vector3i64 t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_Construct_2(MR.Const_Matrix3i64._Underlying *r, MR.Const_Vector3i64._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4i64 _ctor_result = __MR_Matrix4i64_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// Generated from method `MR::Matrix4i64::operator()`.
        public unsafe new ref long Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_call", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_call(_Underlying *_this, int row, int col);
            return ref *__MR_Matrix4i64_call(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix4i64::operator[]`.
        public unsafe new MR.Mut_Vector4i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_Matrix4i64_index(_Underlying *_this, int row);
            return new(__MR_Matrix4i64_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from method `MR::Matrix4i64::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3i64 rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4i64_setRotation(_Underlying *_this, MR.Const_Matrix3i64._Underlying *rot);
            __MR_Matrix4i64_setRotation(_UnderlyingPtr, rot._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3i64 t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4i64_setTranslation(_Underlying *_this, MR.Const_Vector3i64._Underlying *t);
            __MR_Matrix4i64_setTranslation(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::data`.
        public unsafe new MR.Misc.Ref<long>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_data", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_data(_Underlying *_this);
            var __ret = __MR_Matrix4i64_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<long>(__ret) : null;
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4i64 AddAssign(MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_add_assign_MR_Matrix4i64(_Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return new(__MR_add_assign_MR_Matrix4i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4i64 SubAssign(MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_sub_assign_MR_Matrix4i64(_Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix4i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_mul_assign_MR_Matrix4i64_int64_t(_Underlying *a, long b);
            return new(__MR_mul_assign_MR_Matrix4i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_div_assign_MR_Matrix4i64_int64_t(_Underlying *a, long b);
            return new(__MR_div_assign_MR_Matrix4i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 128)]
    public struct Matrix4i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix4i64(Const_Matrix4i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix4i64(Matrix4i64 other) => new(new Mut_Matrix4i64((Mut_Matrix4i64._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector4i64 X;

        [System.Runtime.InteropServices.FieldOffset(32)]
        public MR.Vector4i64 Y;

        [System.Runtime.InteropServices.FieldOffset(64)]
        public MR.Vector4i64 Z;

        [System.Runtime.InteropServices.FieldOffset(96)]
        public MR.Vector4i64 W;

        /// Generated copy constructor.
        public Matrix4i64(Matrix4i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix4i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_DefaultConstruct();
            this = __MR_Matrix4i64_DefaultConstruct();
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4i64::Matrix4i64`.
        public unsafe Matrix4i64(MR.Const_Vector4i64 x, MR.Const_Vector4i64 y, MR.Const_Vector4i64 z, MR.Const_Vector4i64 w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_Construct_4(MR.Const_Vector4i64._Underlying *x, MR.Const_Vector4i64._Underlying *y, MR.Const_Vector4i64._Underlying *z, MR.Const_Vector4i64._Underlying *w);
            this = __MR_Matrix4i64_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4i64::Matrix4i64`.
        public unsafe Matrix4i64(MR.Const_Matrix3i64 r, MR.Const_Vector3i64 t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_Construct_2(MR.Const_Matrix3i64._Underlying *r, MR.Const_Vector3i64._Underlying *t);
            this = __MR_Matrix4i64_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4i64::zero`.
        public static MR.Matrix4i64 Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_zero", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_zero();
            return __MR_Matrix4i64_zero();
        }

        /// Generated from method `MR::Matrix4i64::identity`.
        public static MR.Matrix4i64 Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_identity", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_identity();
            return __MR_Matrix4i64_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4i64::scale`.
        public static MR.Matrix4i64 Scale(long s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_scale", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_scale(long s);
            return __MR_Matrix4i64_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4i64::operator()`.
        public unsafe long Call_Const(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_call_const", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_call_const(MR.Matrix4i64 *_this, int row, int col);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return *__MR_Matrix4i64_call_const(__ptr__this, row, col);
            }
        }

        /// Generated from method `MR::Matrix4i64::operator()`.
        public unsafe ref long Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_call", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_call(MR.Matrix4i64 *_this, int row, int col);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return ref *__MR_Matrix4i64_call(__ptr__this, row, col);
            }
        }

        /// row access
        /// Generated from method `MR::Matrix4i64::operator[]`.
        public unsafe MR.Const_Vector4i64 Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4i64._Underlying *__MR_Matrix4i64_index_const(MR.Matrix4i64 *_this, int row);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return new(__MR_Matrix4i64_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix4i64::operator[]`.
        public unsafe MR.Mut_Vector4i64 Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_Matrix4i64_index(MR.Matrix4i64 *_this, int row);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return new(__MR_Matrix4i64_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix4i64::col`.
        public unsafe MR.Vector4i64 Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_col", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Matrix4i64_col(MR.Matrix4i64 *_this, int i);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4i64::trace`.
        public unsafe long Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_trace", ExactSpelling = true)]
            extern static long __MR_Matrix4i64_trace(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4i64::normSq`.
        public unsafe long NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_normSq", ExactSpelling = true)]
            extern static long __MR_Matrix4i64_normSq(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i64::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4i64_norm(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_norm(__ptr__this);
            }
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4i64::submatrix3`.
        public unsafe MR.Matrix3i64 Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix4i64_submatrix3(MR.Matrix4i64 *_this, int i, int j);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_submatrix3(__ptr__this, i, j);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4i64::det`.
        public unsafe long Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_det", ExactSpelling = true)]
            extern static long __MR_Matrix4i64_det(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_det(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4i64::transposed`.
        public unsafe MR.Matrix4i64 Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_transposed", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_Matrix4i64_transposed(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_transposed(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i64::getRotation`.
        public unsafe MR.Matrix3i64 GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3i64 __MR_Matrix4i64_getRotation(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_getRotation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i64::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3i64 rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4i64_setRotation(MR.Matrix4i64 *_this, MR.Const_Matrix3i64._Underlying *rot);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                __MR_Matrix4i64_setRotation(__ptr__this, rot._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4i64::getTranslation`.
        public unsafe MR.Vector3i64 GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Matrix4i64_getTranslation(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                return __MR_Matrix4i64_getTranslation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4i64::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3i64 t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4i64_setTranslation(MR.Matrix4i64 *_this, MR.Const_Vector3i64._Underlying *t);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                __MR_Matrix4i64_setTranslation(__ptr__this, t._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4i64::data`.
        public unsafe MR.Misc.Ref<long>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_data", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_data(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4i64_data(__ptr__this);
                return __ret is not null ? new MR.Misc.Ref<long>(__ret) : null;
            }
        }

        /// Generated from method `MR::Matrix4i64::data`.
        public unsafe long? Data_Const()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4i64_data_const", ExactSpelling = true)]
            extern static long *__MR_Matrix4i64_data_const(MR.Matrix4i64 *_this);
            fixed (MR.Matrix4i64 *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4i64_data_const(__ptr__this);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix4i64 a, MR.Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_equal_MR_Matrix4i64((MR.Mut_Matrix4i64._Underlying *)&a, (MR.Mut_Matrix4i64._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix4i64 a, MR.Matrix4i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i64 operator+(MR.Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_add_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_add_MR_Matrix4i64((MR.Mut_Matrix4i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i64 operator-(MR.Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_sub_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_sub_MR_Matrix4i64((MR.Mut_Matrix4i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i64 operator*(long a, MR.Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_mul_int64_t_MR_Matrix4i64(long a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_mul_int64_t_MR_Matrix4i64(a, (MR.Mut_Matrix4i64._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i64 operator*(MR.Matrix4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_mul_MR_Matrix4i64_int64_t(MR.Const_Matrix4i64._Underlying *b, long a);
            return __MR_mul_MR_Matrix4i64_int64_t((MR.Mut_Matrix4i64._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix4i64 operator/(MR.Matrix4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_div_MR_Matrix4i64_int64_t(MR.Matrix4i64 b, long a);
            return __MR_div_MR_Matrix4i64_int64_t(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4i64 AddAssign(MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_add_assign_MR_Matrix4i64(MR.Matrix4i64 *a, MR.Const_Matrix4i64._Underlying *b);
            fixed (MR.Matrix4i64 *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix4i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4i64 SubAssign(MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_sub_assign_MR_Matrix4i64(MR.Matrix4i64 *a, MR.Const_Matrix4i64._Underlying *b);
            fixed (MR.Matrix4i64 *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix4i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_mul_assign_MR_Matrix4i64_int64_t(MR.Matrix4i64 *a, long b);
            fixed (MR.Matrix4i64 *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix4i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Matrix4i64._Underlying *__MR_div_assign_MR_Matrix4i64_int64_t(MR.Matrix4i64 *a, long b);
            fixed (MR.Matrix4i64 *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix4i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i64 operator*(MR.Matrix4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i64_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_mul_MR_Matrix4i64_MR_Vector4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_mul_MR_Matrix4i64_MR_Vector4i64((MR.Mut_Matrix4i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i64 operator*(MR.Matrix4i64 a, MR.Const_Matrix4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4i64", ExactSpelling = true)]
            extern static MR.Matrix4i64 __MR_mul_MR_Matrix4i64(MR.Const_Matrix4i64._Underlying *a, MR.Const_Matrix4i64._Underlying *b);
            return __MR_mul_MR_Matrix4i64((MR.Mut_Matrix4i64._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix4i64 b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix4i64)
                return this == (MR.Matrix4i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix4i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix4i64`/`Const_Matrix4i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix4i64
    {
        public readonly bool HasValue;
        internal readonly Matrix4i64 Object;
        public Matrix4i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix4i64() {HasValue = false;}
        public _InOpt_Matrix4i64(Matrix4i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix4i64(Matrix4i64 new_value) {return new(new_value);}
        public _InOpt_Matrix4i64(Const_Matrix4i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix4i64(Const_Matrix4i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Matrix4i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix4i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4i64`/`Const_Matrix4i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4i64`.
    public class _InOptMut_Matrix4i64
    {
        public Mut_Matrix4i64? Opt;

        public _InOptMut_Matrix4i64() {}
        public _InOptMut_Matrix4i64(Mut_Matrix4i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix4i64(Mut_Matrix4i64 value) {return new(value);}
        public unsafe _InOptMut_Matrix4i64(ref Matrix4i64 value)
        {
            fixed (Matrix4i64 *value_ptr = &value)
            {
                Opt = new((Const_Matrix4i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix4i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix4i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4i64`/`Const_Matrix4i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4i64`.
    public class _InOptConst_Matrix4i64
    {
        public Const_Matrix4i64? Opt;

        public _InOptConst_Matrix4i64() {}
        public _InOptConst_Matrix4i64(Const_Matrix4i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix4i64(Const_Matrix4i64 value) {return new(value);}
        public unsafe _InOptConst_Matrix4i64(ref readonly Matrix4i64 value)
        {
            fixed (Matrix4i64 *value_ptr = &value)
            {
                Opt = new((Const_Matrix4i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4f`.
    /// This is the const reference to the struct.
    public class Const_Matrix4f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix4f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix4f UnderlyingStruct => ref *(Matrix4f *)_UnderlyingPtr;

        internal unsafe Const_Matrix4f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix4f_Destroy(_Underlying *_this);
            __MR_Matrix4f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix4f() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector4f X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector4f Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector4f Z => ref UnderlyingStruct.Z;

        public ref readonly MR.Vector4f W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Const_Matrix4f(Const_Matrix4f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 64);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix4f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Const_Matrix4f(MR.Const_Vector4f x, MR.Const_Vector4f y, MR.Const_Vector4f z, MR.Const_Vector4f w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_4(MR.Const_Vector4f._Underlying *x, MR.Const_Vector4f._Underlying *y, MR.Const_Vector4f._Underlying *z, MR.Const_Vector4f._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Const_Matrix4f(MR.Const_Matrix3f r, MR.Const_Vector3f t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_2(MR.Const_Matrix3f._Underlying *r, MR.Const_Vector3f._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Const_Matrix4f(MR.Const_AffineXf3f xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_float(MR.Const_AffineXf3f._Underlying *xf);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_Construct_float(xf._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public static unsafe implicit operator Const_Matrix4f(MR.Const_AffineXf3f xf) {return new(xf);}

        /// Generated from conversion operator `MR::Matrix4f::operator MR::AffineXf3f`.
        public static unsafe implicit operator MR.AffineXf3f(MR.Const_Matrix4f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_ConvertTo_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_Matrix4f_ConvertTo_MR_AffineXf3f(MR.Const_Matrix4f._Underlying *_this);
            return __MR_Matrix4f_ConvertTo_MR_AffineXf3f(_this._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::zero`.
        public static MR.Matrix4f Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_zero", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_zero();
            return __MR_Matrix4f_zero();
        }

        /// Generated from method `MR::Matrix4f::identity`.
        public static MR.Matrix4f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_identity", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_identity();
            return __MR_Matrix4f_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4f::scale`.
        public static MR.Matrix4f Scale(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_scale", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_scale(float s);
            return __MR_Matrix4f_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4f::operator()`.
        public unsafe float Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_call_const_2", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_call_const_2(_Underlying *_this, int row, int col);
            return *__MR_Matrix4f_call_const_2(_UnderlyingPtr, row, col);
        }

        /// row access
        /// Generated from method `MR::Matrix4f::operator[]`.
        public unsafe MR.Const_Vector4f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4f._Underlying *__MR_Matrix4f_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix4f_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix4f::col`.
        public unsafe MR.Vector4f Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_col", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Matrix4f_col(_Underlying *_this, int i);
            return __MR_Matrix4f_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_trace", ExactSpelling = true)]
            extern static float __MR_Matrix4f_trace(_Underlying *_this);
            return __MR_Matrix4f_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_normSq", ExactSpelling = true)]
            extern static float __MR_Matrix4f_normSq(_Underlying *_this);
            return __MR_Matrix4f_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_norm", ExactSpelling = true)]
            extern static float __MR_Matrix4f_norm(_Underlying *_this);
            return __MR_Matrix4f_norm(_UnderlyingPtr);
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4f::submatrix3`.
        public unsafe MR.Matrix3f Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix4f_submatrix3(_Underlying *_this, int i, int j);
            return __MR_Matrix4f_submatrix3(_UnderlyingPtr, i, j);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_det", ExactSpelling = true)]
            extern static float __MR_Matrix4f_det(_Underlying *_this);
            return __MR_Matrix4f_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix4f::inverse`.
        public unsafe MR.Matrix4f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_inverse", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_inverse(_Underlying *_this);
            return __MR_Matrix4f_inverse(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4f::transposed`.
        public unsafe MR.Matrix4f Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_transposed", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_transposed(_Underlying *_this);
            return __MR_Matrix4f_transposed(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::getRotation`.
        public unsafe MR.Matrix3f GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix4f_getRotation(_Underlying *_this);
            return __MR_Matrix4f_getRotation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::getTranslation`.
        public unsafe MR.Vector3f GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix4f_getTranslation(_Underlying *_this);
            return __MR_Matrix4f_getTranslation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::data`.
        public unsafe float? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_data_const", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_data_const(_Underlying *_this);
            var __ret = __MR_Matrix4f_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// converts 3d-vector b in 4d-vector (b,1), multiplies matrix on it,
        /// and assuming the result is in homogeneous coordinates returns it as 3d-vector
        /// Generated from method `MR::Matrix4f::operator()`.
        public unsafe MR.Vector3f Call(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_call_const_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix4f_call_const_1(_Underlying *_this, MR.Const_Vector3f._Underlying *b);
            return __MR_Matrix4f_call_const_1(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_equal_MR_Matrix4f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix4f a, MR.Const_Matrix4f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4f operator+(MR.Const_Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_add_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_add_MR_Matrix4f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4f operator-(MR.Const_Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_sub_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_sub_MR_Matrix4f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4f operator*(float a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_mul_float_MR_Matrix4f(float a, MR.Const_Matrix4f._Underlying *b);
            return __MR_mul_float_MR_Matrix4f(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4f operator*(MR.Const_Matrix4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_mul_MR_Matrix4f_float(MR.Const_Matrix4f._Underlying *b, float a);
            return __MR_mul_MR_Matrix4f_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix4f operator/(Const_Matrix4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_div_MR_Matrix4f_float(MR.Matrix4f b, float a);
            return __MR_div_MR_Matrix4f_float(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4f operator*(MR.Const_Matrix4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4f_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_mul_MR_Matrix4f_MR_Vector4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_mul_MR_Matrix4f_MR_Vector4f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4f operator*(MR.Const_Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_mul_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_mul_MR_Matrix4f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix4f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix4f)
                return this == (MR.Const_Matrix4f)other;
            return false;
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4f`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix4f : Const_Matrix4f
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix4f UnderlyingStruct => ref *(Matrix4f *)_UnderlyingPtr;

        internal unsafe Mut_Matrix4f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector4f X => ref UnderlyingStruct.X;

        public new ref MR.Vector4f Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector4f Z => ref UnderlyingStruct.Z;

        public new ref MR.Vector4f W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Matrix4f(Const_Matrix4f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 64);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix4f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Mut_Matrix4f(MR.Const_Vector4f x, MR.Const_Vector4f y, MR.Const_Vector4f z, MR.Const_Vector4f w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_4(MR.Const_Vector4f._Underlying *x, MR.Const_Vector4f._Underlying *y, MR.Const_Vector4f._Underlying *z, MR.Const_Vector4f._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Mut_Matrix4f(MR.Const_Matrix3f r, MR.Const_Vector3f t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_2(MR.Const_Matrix3f._Underlying *r, MR.Const_Vector3f._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Mut_Matrix4f(MR.Const_AffineXf3f xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_float(MR.Const_AffineXf3f._Underlying *xf);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(64);
            MR.Matrix4f _ctor_result = __MR_Matrix4f_Construct_float(xf._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 64);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public static unsafe implicit operator Mut_Matrix4f(MR.Const_AffineXf3f xf) {return new(xf);}

        /// Generated from method `MR::Matrix4f::operator()`.
        public unsafe new ref float Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_call", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_call(_Underlying *_this, int row, int col);
            return ref *__MR_Matrix4f_call(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix4f::operator[]`.
        public unsafe new MR.Mut_Vector4f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_Matrix4f_index(_Underlying *_this, int row);
            return new(__MR_Matrix4f_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from method `MR::Matrix4f::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3f rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4f_setRotation(_Underlying *_this, MR.Const_Matrix3f._Underlying *rot);
            __MR_Matrix4f_setRotation(_UnderlyingPtr, rot._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3f t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4f_setTranslation(_Underlying *_this, MR.Const_Vector3f._Underlying *t);
            __MR_Matrix4f_setTranslation(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4f::data`.
        public unsafe new MR.Misc.Ref<float>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_data", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_data(_Underlying *_this);
            var __ret = __MR_Matrix4f_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<float>(__ret) : null;
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4f AddAssign(MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_add_assign_MR_Matrix4f(_Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return new(__MR_add_assign_MR_Matrix4f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4f SubAssign(MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_sub_assign_MR_Matrix4f(_Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix4f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_mul_assign_MR_Matrix4f_float(_Underlying *a, float b);
            return new(__MR_mul_assign_MR_Matrix4f_float(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_div_assign_MR_Matrix4f_float(_Underlying *a, float b);
            return new(__MR_div_assign_MR_Matrix4f_float(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 64)]
    public struct Matrix4f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix4f(Const_Matrix4f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix4f(Matrix4f other) => new(new Mut_Matrix4f((Mut_Matrix4f._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector4f X;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector4f Y;

        [System.Runtime.InteropServices.FieldOffset(32)]
        public MR.Vector4f Z;

        [System.Runtime.InteropServices.FieldOffset(48)]
        public MR.Vector4f W;

        /// Generated copy constructor.
        public Matrix4f(Matrix4f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix4f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_DefaultConstruct();
            this = __MR_Matrix4f_DefaultConstruct();
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Matrix4f(MR.Const_Vector4f x, MR.Const_Vector4f y, MR.Const_Vector4f z, MR.Const_Vector4f w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_4(MR.Const_Vector4f._Underlying *x, MR.Const_Vector4f._Underlying *y, MR.Const_Vector4f._Underlying *z, MR.Const_Vector4f._Underlying *w);
            this = __MR_Matrix4f_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Matrix4f(MR.Const_Matrix3f r, MR.Const_Vector3f t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_2(MR.Const_Matrix3f._Underlying *r, MR.Const_Vector3f._Underlying *t);
            this = __MR_Matrix4f_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public unsafe Matrix4f(MR.Const_AffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_Construct_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_Construct_float(MR.Const_AffineXf3f._Underlying *xf);
            this = __MR_Matrix4f_Construct_float(xf._UnderlyingPtr);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public static unsafe implicit operator Matrix4f(MR.Const_AffineXf3f xf) {return new(xf);}

        /// Generated from conversion operator `MR::Matrix4f::operator MR::AffineXf3f`.
        public static unsafe implicit operator MR.AffineXf3f(MR.Matrix4f _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_ConvertTo_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_Matrix4f_ConvertTo_MR_AffineXf3f(MR.Const_Matrix4f._Underlying *_this);
            return __MR_Matrix4f_ConvertTo_MR_AffineXf3f((MR.Mut_Matrix4f._Underlying *)&_this);
        }

        /// Generated from method `MR::Matrix4f::zero`.
        public static MR.Matrix4f Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_zero", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_zero();
            return __MR_Matrix4f_zero();
        }

        /// Generated from method `MR::Matrix4f::identity`.
        public static MR.Matrix4f Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_identity", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_identity();
            return __MR_Matrix4f_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4f::scale`.
        public static MR.Matrix4f Scale(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_scale", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_scale(float s);
            return __MR_Matrix4f_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4f::operator()`.
        public unsafe float Call_Const(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_call_const_2", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_call_const_2(MR.Matrix4f *_this, int row, int col);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return *__MR_Matrix4f_call_const_2(__ptr__this, row, col);
            }
        }

        /// Generated from method `MR::Matrix4f::operator()`.
        public unsafe ref float Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_call", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_call(MR.Matrix4f *_this, int row, int col);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return ref *__MR_Matrix4f_call(__ptr__this, row, col);
            }
        }

        /// row access
        /// Generated from method `MR::Matrix4f::operator[]`.
        public unsafe MR.Const_Vector4f Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4f._Underlying *__MR_Matrix4f_index_const(MR.Matrix4f *_this, int row);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return new(__MR_Matrix4f_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix4f::operator[]`.
        public unsafe MR.Mut_Vector4f Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_Matrix4f_index(MR.Matrix4f *_this, int row);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return new(__MR_Matrix4f_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix4f::col`.
        public unsafe MR.Vector4f Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_col", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Matrix4f_col(MR.Matrix4f *_this, int i);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4f::trace`.
        public unsafe float Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_trace", ExactSpelling = true)]
            extern static float __MR_Matrix4f_trace(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4f::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_normSq", ExactSpelling = true)]
            extern static float __MR_Matrix4f_normSq(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4f::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_norm", ExactSpelling = true)]
            extern static float __MR_Matrix4f_norm(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_norm(__ptr__this);
            }
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4f::submatrix3`.
        public unsafe MR.Matrix3f Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix4f_submatrix3(MR.Matrix4f *_this, int i, int j);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_submatrix3(__ptr__this, i, j);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4f::det`.
        public unsafe float Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_det", ExactSpelling = true)]
            extern static float __MR_Matrix4f_det(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_det(__ptr__this);
            }
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix4f::inverse`.
        public unsafe MR.Matrix4f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_inverse", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_inverse(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_inverse(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4f::transposed`.
        public unsafe MR.Matrix4f Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_transposed", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_Matrix4f_transposed(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_transposed(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4f::getRotation`.
        public unsafe MR.Matrix3f GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Matrix4f_getRotation(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_getRotation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4f::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3f rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4f_setRotation(MR.Matrix4f *_this, MR.Const_Matrix3f._Underlying *rot);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                __MR_Matrix4f_setRotation(__ptr__this, rot._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4f::getTranslation`.
        public unsafe MR.Vector3f GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix4f_getTranslation(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_getTranslation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4f::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3f t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4f_setTranslation(MR.Matrix4f *_this, MR.Const_Vector3f._Underlying *t);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                __MR_Matrix4f_setTranslation(__ptr__this, t._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4f::data`.
        public unsafe MR.Misc.Ref<float>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_data", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_data(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4f_data(__ptr__this);
                return __ret is not null ? new MR.Misc.Ref<float>(__ret) : null;
            }
        }

        /// Generated from method `MR::Matrix4f::data`.
        public unsafe float? Data_Const()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_data_const", ExactSpelling = true)]
            extern static float *__MR_Matrix4f_data_const(MR.Matrix4f *_this);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4f_data_const(__ptr__this);
                return __ret is not null ? *__ret : null;
            }
        }

        /// converts 3d-vector b in 4d-vector (b,1), multiplies matrix on it,
        /// and assuming the result is in homogeneous coordinates returns it as 3d-vector
        /// Generated from method `MR::Matrix4f::operator()`.
        public unsafe MR.Vector3f Call(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4f_call_const_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Matrix4f_call_const_1(MR.Matrix4f *_this, MR.Const_Vector3f._Underlying *b);
            fixed (MR.Matrix4f *__ptr__this = &this)
            {
                return __MR_Matrix4f_call_const_1(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix4f a, MR.Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_equal_MR_Matrix4f((MR.Mut_Matrix4f._Underlying *)&a, (MR.Mut_Matrix4f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix4f a, MR.Matrix4f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4f operator+(MR.Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_add_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_add_MR_Matrix4f((MR.Mut_Matrix4f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4f operator-(MR.Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_sub_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_sub_MR_Matrix4f((MR.Mut_Matrix4f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4f operator*(float a, MR.Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_mul_float_MR_Matrix4f(float a, MR.Const_Matrix4f._Underlying *b);
            return __MR_mul_float_MR_Matrix4f(a, (MR.Mut_Matrix4f._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4f operator*(MR.Matrix4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_mul_MR_Matrix4f_float(MR.Const_Matrix4f._Underlying *b, float a);
            return __MR_mul_MR_Matrix4f_float((MR.Mut_Matrix4f._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix4f operator/(MR.Matrix4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_div_MR_Matrix4f_float(MR.Matrix4f b, float a);
            return __MR_div_MR_Matrix4f_float(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4f AddAssign(MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_add_assign_MR_Matrix4f(MR.Matrix4f *a, MR.Const_Matrix4f._Underlying *b);
            fixed (MR.Matrix4f *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix4f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4f SubAssign(MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_sub_assign_MR_Matrix4f(MR.Matrix4f *a, MR.Const_Matrix4f._Underlying *b);
            fixed (MR.Matrix4f *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix4f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_mul_assign_MR_Matrix4f_float(MR.Matrix4f *a, float b);
            fixed (MR.Matrix4f *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix4f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4f_float", ExactSpelling = true)]
            extern static MR.Mut_Matrix4f._Underlying *__MR_div_assign_MR_Matrix4f_float(MR.Matrix4f *a, float b);
            fixed (MR.Matrix4f *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix4f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4f operator*(MR.Matrix4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4f_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_mul_MR_Matrix4f_MR_Vector4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_mul_MR_Matrix4f_MR_Vector4f((MR.Mut_Matrix4f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4f operator*(MR.Matrix4f a, MR.Const_Matrix4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4f", ExactSpelling = true)]
            extern static MR.Matrix4f __MR_mul_MR_Matrix4f(MR.Const_Matrix4f._Underlying *a, MR.Const_Matrix4f._Underlying *b);
            return __MR_mul_MR_Matrix4f((MR.Mut_Matrix4f._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix4f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix4f)
                return this == (MR.Matrix4f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix4f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix4f`/`Const_Matrix4f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix4f
    {
        public readonly bool HasValue;
        internal readonly Matrix4f Object;
        public Matrix4f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix4f() {HasValue = false;}
        public _InOpt_Matrix4f(Matrix4f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix4f(Matrix4f new_value) {return new(new_value);}
        public _InOpt_Matrix4f(Const_Matrix4f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix4f(Const_Matrix4f new_value) {return new(new_value);}

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public static unsafe implicit operator _InOpt_Matrix4f(MR.Const_AffineXf3f xf) {return new MR.Matrix4f(xf);}
    }

    /// This is used for optional parameters of class `Mut_Matrix4f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix4f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4f`/`Const_Matrix4f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4f`.
    public class _InOptMut_Matrix4f
    {
        public Mut_Matrix4f? Opt;

        public _InOptMut_Matrix4f() {}
        public _InOptMut_Matrix4f(Mut_Matrix4f value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix4f(Mut_Matrix4f value) {return new(value);}
        public unsafe _InOptMut_Matrix4f(ref Matrix4f value)
        {
            fixed (Matrix4f *value_ptr = &value)
            {
                Opt = new((Const_Matrix4f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix4f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix4f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4f`/`Const_Matrix4f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4f`.
    public class _InOptConst_Matrix4f
    {
        public Const_Matrix4f? Opt;

        public _InOptConst_Matrix4f() {}
        public _InOptConst_Matrix4f(Const_Matrix4f value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix4f(Const_Matrix4f value) {return new(value);}
        public unsafe _InOptConst_Matrix4f(ref readonly Matrix4f value)
        {
            fixed (Matrix4f *value_ptr = &value)
            {
                Opt = new((Const_Matrix4f._Underlying *)value_ptr, is_owning: false);
            }
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4f::Matrix4f`.
        public static unsafe implicit operator _InOptConst_Matrix4f(MR.Const_AffineXf3f xf) {return new MR.Matrix4f(xf);}
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4d`.
    /// This is the const reference to the struct.
    public class Const_Matrix4d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix4d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Matrix4d UnderlyingStruct => ref *(Matrix4d *)_UnderlyingPtr;

        internal unsafe Const_Matrix4d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix4d_Destroy(_Underlying *_this);
            __MR_Matrix4d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix4d() {Dispose(false);}

        /// rows, identity matrix by default
        public ref readonly MR.Vector4d X => ref UnderlyingStruct.X;

        public ref readonly MR.Vector4d Y => ref UnderlyingStruct.Y;

        public ref readonly MR.Vector4d Z => ref UnderlyingStruct.Z;

        public ref readonly MR.Vector4d W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Const_Matrix4d(Const_Matrix4d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 128);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix4d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Const_Matrix4d(MR.Const_Vector4d x, MR.Const_Vector4d y, MR.Const_Vector4d z, MR.Const_Vector4d w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_4(MR.Const_Vector4d._Underlying *x, MR.Const_Vector4d._Underlying *y, MR.Const_Vector4d._Underlying *z, MR.Const_Vector4d._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Const_Matrix4d(MR.Const_Matrix3d r, MR.Const_Vector3d t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_2(MR.Const_Matrix3d._Underlying *r, MR.Const_Vector3d._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Const_Matrix4d(MR.Const_AffineXf3d xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_double(MR.Const_AffineXf3d._Underlying *xf);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_Construct_double(xf._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public static unsafe implicit operator Const_Matrix4d(MR.Const_AffineXf3d xf) {return new(xf);}

        /// Generated from conversion operator `MR::Matrix4d::operator MR::AffineXf3d`.
        public static unsafe implicit operator MR.AffineXf3d(MR.Const_Matrix4d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_ConvertTo_MR_AffineXf3d", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_Matrix4d_ConvertTo_MR_AffineXf3d(MR.Const_Matrix4d._Underlying *_this);
            return __MR_Matrix4d_ConvertTo_MR_AffineXf3d(_this._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::zero`.
        public static MR.Matrix4d Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_zero", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_zero();
            return __MR_Matrix4d_zero();
        }

        /// Generated from method `MR::Matrix4d::identity`.
        public static MR.Matrix4d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_identity", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_identity();
            return __MR_Matrix4d_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4d::scale`.
        public static MR.Matrix4d Scale(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_scale", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_scale(double s);
            return __MR_Matrix4d_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4d::operator()`.
        public unsafe double Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_call_const_2", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_call_const_2(_Underlying *_this, int row, int col);
            return *__MR_Matrix4d_call_const_2(_UnderlyingPtr, row, col);
        }

        /// row access
        /// Generated from method `MR::Matrix4d::operator[]`.
        public unsafe MR.Const_Vector4d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4d._Underlying *__MR_Matrix4d_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix4d_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix4d::col`.
        public unsafe MR.Vector4d Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_col", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Matrix4d_col(_Underlying *_this, int i);
            return __MR_Matrix4d_col(_UnderlyingPtr, i);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_trace", ExactSpelling = true)]
            extern static double __MR_Matrix4d_trace(_Underlying *_this);
            return __MR_Matrix4d_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_normSq", ExactSpelling = true)]
            extern static double __MR_Matrix4d_normSq(_Underlying *_this);
            return __MR_Matrix4d_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4d_norm(_Underlying *_this);
            return __MR_Matrix4d_norm(_UnderlyingPtr);
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4d::submatrix3`.
        public unsafe MR.Matrix3d Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix4d_submatrix3(_Underlying *_this, int i, int j);
            return __MR_Matrix4d_submatrix3(_UnderlyingPtr, i, j);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_det", ExactSpelling = true)]
            extern static double __MR_Matrix4d_det(_Underlying *_this);
            return __MR_Matrix4d_det(_UnderlyingPtr);
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix4d::inverse`.
        public unsafe MR.Matrix4d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_inverse", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_inverse(_Underlying *_this);
            return __MR_Matrix4d_inverse(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4d::transposed`.
        public unsafe MR.Matrix4d Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_transposed", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_transposed(_Underlying *_this);
            return __MR_Matrix4d_transposed(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::getRotation`.
        public unsafe MR.Matrix3d GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix4d_getRotation(_Underlying *_this);
            return __MR_Matrix4d_getRotation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::getTranslation`.
        public unsafe MR.Vector3d GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix4d_getTranslation(_Underlying *_this);
            return __MR_Matrix4d_getTranslation(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::data`.
        public unsafe double? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_data_const", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_data_const(_Underlying *_this);
            var __ret = __MR_Matrix4d_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// converts 3d-vector b in 4d-vector (b,1), multiplies matrix on it,
        /// and assuming the result is in homogeneous coordinates returns it as 3d-vector
        /// Generated from method `MR::Matrix4d::operator()`.
        public unsafe MR.Vector3d Call(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_call_const_1", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix4d_call_const_1(_Underlying *_this, MR.Const_Vector3d._Underlying *b);
            return __MR_Matrix4d_call_const_1(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_equal_MR_Matrix4d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix4d a, MR.Const_Matrix4d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4d operator+(MR.Const_Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_add_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_add_MR_Matrix4d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4d operator-(MR.Const_Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_sub_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_sub_MR_Matrix4d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4d operator*(double a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_mul_double_MR_Matrix4d(double a, MR.Const_Matrix4d._Underlying *b);
            return __MR_mul_double_MR_Matrix4d(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4d operator*(MR.Const_Matrix4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_mul_MR_Matrix4d_double(MR.Const_Matrix4d._Underlying *b, double a);
            return __MR_mul_MR_Matrix4d_double(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix4d operator/(Const_Matrix4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_div_MR_Matrix4d_double(MR.Matrix4d b, double a);
            return __MR_div_MR_Matrix4d_double(b.UnderlyingStruct, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4d operator*(MR.Const_Matrix4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4d_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_mul_MR_Matrix4d_MR_Vector4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_mul_MR_Matrix4d_MR_Vector4d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4d operator*(MR.Const_Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_mul_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_mul_MR_Matrix4d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix4d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix4d)
                return this == (MR.Const_Matrix4d)other;
            return false;
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4d`.
    /// This is the non-const reference to the struct.
    public class Mut_Matrix4d : Const_Matrix4d
    {
        /// Get the underlying struct.
        public unsafe new ref Matrix4d UnderlyingStruct => ref *(Matrix4d *)_UnderlyingPtr;

        internal unsafe Mut_Matrix4d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new ref MR.Vector4d X => ref UnderlyingStruct.X;

        public new ref MR.Vector4d Y => ref UnderlyingStruct.Y;

        public new ref MR.Vector4d Z => ref UnderlyingStruct.Z;

        public new ref MR.Vector4d W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Matrix4d(Const_Matrix4d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 128);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Matrix4d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Mut_Matrix4d(MR.Const_Vector4d x, MR.Const_Vector4d y, MR.Const_Vector4d z, MR.Const_Vector4d w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_4(MR.Const_Vector4d._Underlying *x, MR.Const_Vector4d._Underlying *y, MR.Const_Vector4d._Underlying *z, MR.Const_Vector4d._Underlying *w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Mut_Matrix4d(MR.Const_Matrix3d r, MR.Const_Vector3d t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_2(MR.Const_Matrix3d._Underlying *r, MR.Const_Vector3d._Underlying *t);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Mut_Matrix4d(MR.Const_AffineXf3d xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_double(MR.Const_AffineXf3d._Underlying *xf);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(128);
            MR.Matrix4d _ctor_result = __MR_Matrix4d_Construct_double(xf._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 128);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public static unsafe implicit operator Mut_Matrix4d(MR.Const_AffineXf3d xf) {return new(xf);}

        /// Generated from method `MR::Matrix4d::operator()`.
        public unsafe new ref double Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_call", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_call(_Underlying *_this, int row, int col);
            return ref *__MR_Matrix4d_call(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix4d::operator[]`.
        public unsafe new MR.Mut_Vector4d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_Matrix4d_index(_Underlying *_this, int row);
            return new(__MR_Matrix4d_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from method `MR::Matrix4d::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3d rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4d_setRotation(_Underlying *_this, MR.Const_Matrix3d._Underlying *rot);
            __MR_Matrix4d_setRotation(_UnderlyingPtr, rot._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3d t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4d_setTranslation(_Underlying *_this, MR.Const_Vector3d._Underlying *t);
            __MR_Matrix4d_setTranslation(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4d::data`.
        public unsafe new MR.Misc.Ref<double>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_data", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_data(_Underlying *_this);
            var __ret = __MR_Matrix4d_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<double>(__ret) : null;
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4d AddAssign(MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_add_assign_MR_Matrix4d(_Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return new(__MR_add_assign_MR_Matrix4d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4d SubAssign(MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_sub_assign_MR_Matrix4d(_Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix4d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_mul_assign_MR_Matrix4d_double(_Underlying *a, double b);
            return new(__MR_mul_assign_MR_Matrix4d_double(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_div_assign_MR_Matrix4d_double(_Underlying *a, double b);
            return new(__MR_div_assign_MR_Matrix4d_double(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 128)]
    public struct Matrix4d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Matrix4d(Const_Matrix4d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Matrix4d(Matrix4d other) => new(new Mut_Matrix4d((Mut_Matrix4d._Underlying *)&other, is_owning: false));

        /// rows, identity matrix by default
        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector4d X;

        [System.Runtime.InteropServices.FieldOffset(32)]
        public MR.Vector4d Y;

        [System.Runtime.InteropServices.FieldOffset(64)]
        public MR.Vector4d Z;

        [System.Runtime.InteropServices.FieldOffset(96)]
        public MR.Vector4d W;

        /// Generated copy constructor.
        public Matrix4d(Matrix4d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix4d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_DefaultConstruct();
            this = __MR_Matrix4d_DefaultConstruct();
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Matrix4d(MR.Const_Vector4d x, MR.Const_Vector4d y, MR.Const_Vector4d z, MR.Const_Vector4d w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_4(MR.Const_Vector4d._Underlying *x, MR.Const_Vector4d._Underlying *y, MR.Const_Vector4d._Underlying *z, MR.Const_Vector4d._Underlying *w);
            this = __MR_Matrix4d_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Matrix4d(MR.Const_Matrix3d r, MR.Const_Vector3d t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_2(MR.Const_Matrix3d._Underlying *r, MR.Const_Vector3d._Underlying *t);
            this = __MR_Matrix4d_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public unsafe Matrix4d(MR.Const_AffineXf3d xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_Construct_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_Construct_double(MR.Const_AffineXf3d._Underlying *xf);
            this = __MR_Matrix4d_Construct_double(xf._UnderlyingPtr);
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public static unsafe implicit operator Matrix4d(MR.Const_AffineXf3d xf) {return new(xf);}

        /// Generated from conversion operator `MR::Matrix4d::operator MR::AffineXf3d`.
        public static unsafe implicit operator MR.AffineXf3d(MR.Matrix4d _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_ConvertTo_MR_AffineXf3d", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_Matrix4d_ConvertTo_MR_AffineXf3d(MR.Const_Matrix4d._Underlying *_this);
            return __MR_Matrix4d_ConvertTo_MR_AffineXf3d((MR.Mut_Matrix4d._Underlying *)&_this);
        }

        /// Generated from method `MR::Matrix4d::zero`.
        public static MR.Matrix4d Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_zero", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_zero();
            return __MR_Matrix4d_zero();
        }

        /// Generated from method `MR::Matrix4d::identity`.
        public static MR.Matrix4d Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_identity", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_identity();
            return __MR_Matrix4d_identity();
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4d::scale`.
        public static MR.Matrix4d Scale(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_scale", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_scale(double s);
            return __MR_Matrix4d_scale(s);
        }

        /// element access
        /// Generated from method `MR::Matrix4d::operator()`.
        public unsafe double Call_Const(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_call_const_2", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_call_const_2(MR.Matrix4d *_this, int row, int col);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return *__MR_Matrix4d_call_const_2(__ptr__this, row, col);
            }
        }

        /// Generated from method `MR::Matrix4d::operator()`.
        public unsafe ref double Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_call", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_call(MR.Matrix4d *_this, int row, int col);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return ref *__MR_Matrix4d_call(__ptr__this, row, col);
            }
        }

        /// row access
        /// Generated from method `MR::Matrix4d::operator[]`.
        public unsafe MR.Const_Vector4d Index_Const(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4d._Underlying *__MR_Matrix4d_index_const(MR.Matrix4d *_this, int row);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return new(__MR_Matrix4d_index_const(__ptr__this, row), is_owning: false);
            }
        }

        /// Generated from method `MR::Matrix4d::operator[]`.
        public unsafe MR.Mut_Vector4d Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_Matrix4d_index(MR.Matrix4d *_this, int row);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return new(__MR_Matrix4d_index(__ptr__this, row), is_owning: false);
            }
        }

        /// column access
        /// Generated from method `MR::Matrix4d::col`.
        public unsafe MR.Vector4d Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_col", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Matrix4d_col(MR.Matrix4d *_this, int i);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_col(__ptr__this, i);
            }
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4d::trace`.
        public unsafe double Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_trace", ExactSpelling = true)]
            extern static double __MR_Matrix4d_trace(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_trace(__ptr__this);
            }
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4d::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_normSq", ExactSpelling = true)]
            extern static double __MR_Matrix4d_normSq(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_normSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4d::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4d_norm(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_norm(__ptr__this);
            }
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4d::submatrix3`.
        public unsafe MR.Matrix3d Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix4d_submatrix3(MR.Matrix4d *_this, int i, int j);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_submatrix3(__ptr__this, i, j);
            }
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4d::det`.
        public unsafe double Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_det", ExactSpelling = true)]
            extern static double __MR_Matrix4d_det(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_det(__ptr__this);
            }
        }

        /// computes inverse matrix
        /// Generated from method `MR::Matrix4d::inverse`.
        public unsafe MR.Matrix4d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_inverse", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_inverse(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_inverse(__ptr__this);
            }
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4d::transposed`.
        public unsafe MR.Matrix4d Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_transposed", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_Matrix4d_transposed(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_transposed(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4d::getRotation`.
        public unsafe MR.Matrix3d GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Matrix4d_getRotation(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_getRotation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4d::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3d rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4d_setRotation(MR.Matrix4d *_this, MR.Const_Matrix3d._Underlying *rot);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                __MR_Matrix4d_setRotation(__ptr__this, rot._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4d::getTranslation`.
        public unsafe MR.Vector3d GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix4d_getTranslation(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_getTranslation(__ptr__this);
            }
        }

        /// Generated from method `MR::Matrix4d::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3d t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4d_setTranslation(MR.Matrix4d *_this, MR.Const_Vector3d._Underlying *t);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                __MR_Matrix4d_setTranslation(__ptr__this, t._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Matrix4d::data`.
        public unsafe MR.Misc.Ref<double>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_data", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_data(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4d_data(__ptr__this);
                return __ret is not null ? new MR.Misc.Ref<double>(__ret) : null;
            }
        }

        /// Generated from method `MR::Matrix4d::data`.
        public unsafe double? Data_Const()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_data_const", ExactSpelling = true)]
            extern static double *__MR_Matrix4d_data_const(MR.Matrix4d *_this);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                var __ret = __MR_Matrix4d_data_const(__ptr__this);
                return __ret is not null ? *__ret : null;
            }
        }

        /// converts 3d-vector b in 4d-vector (b,1), multiplies matrix on it,
        /// and assuming the result is in homogeneous coordinates returns it as 3d-vector
        /// Generated from method `MR::Matrix4d::operator()`.
        public unsafe MR.Vector3d Call(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4d_call_const_1", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Matrix4d_call_const_1(MR.Matrix4d *_this, MR.Const_Vector3d._Underlying *b);
            fixed (MR.Matrix4d *__ptr__this = &this)
            {
                return __MR_Matrix4d_call_const_1(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Matrix4d a, MR.Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_equal_MR_Matrix4d((MR.Mut_Matrix4d._Underlying *)&a, (MR.Mut_Matrix4d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Matrix4d a, MR.Matrix4d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4d operator+(MR.Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_add_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_add_MR_Matrix4d((MR.Mut_Matrix4d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4d operator-(MR.Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_sub_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_sub_MR_Matrix4d((MR.Mut_Matrix4d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4d operator*(double a, MR.Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_mul_double_MR_Matrix4d(double a, MR.Const_Matrix4d._Underlying *b);
            return __MR_mul_double_MR_Matrix4d(a, (MR.Mut_Matrix4d._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4d operator*(MR.Matrix4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_mul_MR_Matrix4d_double(MR.Const_Matrix4d._Underlying *b, double a);
            return __MR_mul_MR_Matrix4d_double((MR.Mut_Matrix4d._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Matrix4d operator/(MR.Matrix4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_div_MR_Matrix4d_double(MR.Matrix4d b, double a);
            return __MR_div_MR_Matrix4d_double(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Matrix4d AddAssign(MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_add_assign_MR_Matrix4d(MR.Matrix4d *a, MR.Const_Matrix4d._Underlying *b);
            fixed (MR.Matrix4d *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Matrix4d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Matrix4d SubAssign(MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_sub_assign_MR_Matrix4d(MR.Matrix4d *a, MR.Const_Matrix4d._Underlying *b);
            fixed (MR.Matrix4d *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Matrix4d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Matrix4d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_mul_assign_MR_Matrix4d_double(MR.Matrix4d *a, double b);
            fixed (MR.Matrix4d *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Matrix4d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Matrix4d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4d_double", ExactSpelling = true)]
            extern static MR.Mut_Matrix4d._Underlying *__MR_div_assign_MR_Matrix4d_double(MR.Matrix4d *a, double b);
            fixed (MR.Matrix4d *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Matrix4d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4d operator*(MR.Matrix4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4d_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_mul_MR_Matrix4d_MR_Vector4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_mul_MR_Matrix4d_MR_Vector4d((MR.Mut_Matrix4d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4d operator*(MR.Matrix4d a, MR.Const_Matrix4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4d", ExactSpelling = true)]
            extern static MR.Matrix4d __MR_mul_MR_Matrix4d(MR.Const_Matrix4d._Underlying *a, MR.Const_Matrix4d._Underlying *b);
            return __MR_mul_MR_Matrix4d((MR.Mut_Matrix4d._Underlying *)&a, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Matrix4d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Matrix4d)
                return this == (MR.Matrix4d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Matrix4d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Matrix4d`/`Const_Matrix4d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Matrix4d
    {
        public readonly bool HasValue;
        internal readonly Matrix4d Object;
        public Matrix4d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Matrix4d() {HasValue = false;}
        public _InOpt_Matrix4d(Matrix4d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Matrix4d(Matrix4d new_value) {return new(new_value);}
        public _InOpt_Matrix4d(Const_Matrix4d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Matrix4d(Const_Matrix4d new_value) {return new(new_value);}

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public static unsafe implicit operator _InOpt_Matrix4d(MR.Const_AffineXf3d xf) {return new MR.Matrix4d(xf);}
    }

    /// This is used for optional parameters of class `Mut_Matrix4d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix4d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4d`/`Const_Matrix4d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4d`.
    public class _InOptMut_Matrix4d
    {
        public Mut_Matrix4d? Opt;

        public _InOptMut_Matrix4d() {}
        public _InOptMut_Matrix4d(Mut_Matrix4d value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix4d(Mut_Matrix4d value) {return new(value);}
        public unsafe _InOptMut_Matrix4d(ref Matrix4d value)
        {
            fixed (Matrix4d *value_ptr = &value)
            {
                Opt = new((Const_Matrix4d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Matrix4d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix4d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Matrix4d`/`Const_Matrix4d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Matrix4d`.
    public class _InOptConst_Matrix4d
    {
        public Const_Matrix4d? Opt;

        public _InOptConst_Matrix4d() {}
        public _InOptConst_Matrix4d(Const_Matrix4d value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix4d(Const_Matrix4d value) {return new(value);}
        public unsafe _InOptConst_Matrix4d(ref readonly Matrix4d value)
        {
            fixed (Matrix4d *value_ptr = &value)
            {
                Opt = new((Const_Matrix4d._Underlying *)value_ptr, is_owning: false);
            }
        }

        // Currently `AffineXf3<long long>` doesn't seem to compile, so we disable this constructor for `Matrix4<long long>`, because otherwise
        // mrbind instantiates the entire `AffineXf3<long long>` and chokes on it.
        /// Generated from constructor `MR::Matrix4d::Matrix4d`.
        public static unsafe implicit operator _InOptConst_Matrix4d(MR.Const_AffineXf3d xf) {return new MR.Matrix4d(xf);}
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4<unsigned char>`.
    /// This is the const half of the class.
    public class Const_Matrix4_UnsignedChar : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Matrix4_UnsignedChar>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Matrix4_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_Matrix4_unsigned_char_Destroy(_Underlying *_this);
            __MR_Matrix4_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Matrix4_UnsignedChar() {Dispose(false);}

        /// rows, identity matrix by default
        public unsafe MR.Const_Vector4_UnsignedChar X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Get_x", ExactSpelling = true)]
                extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Get_x(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_Get_x(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector4_UnsignedChar Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Get_y", ExactSpelling = true)]
                extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Get_y(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_Get_y(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector4_UnsignedChar Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Get_z", ExactSpelling = true)]
                extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Get_z(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_Get_z(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector4_UnsignedChar W
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Get_w", ExactSpelling = true)]
                extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Get_w(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_Get_w(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Matrix4_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Matrix4<unsigned char>::Matrix4`.
        public unsafe Const_Matrix4_UnsignedChar(MR.Const_Matrix4_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_ConstructFromAnother(MR.Matrix4_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4<unsigned char>::Matrix4`.
        public unsafe Const_Matrix4_UnsignedChar(MR.Const_Vector4_UnsignedChar x, MR.Const_Vector4_UnsignedChar y, MR.Const_Vector4_UnsignedChar z, MR.Const_Vector4_UnsignedChar w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Construct_4(MR.Const_Vector4_UnsignedChar._Underlying *x, MR.Const_Vector4_UnsignedChar._Underlying *y, MR.Const_Vector4_UnsignedChar._Underlying *z, MR.Const_Vector4_UnsignedChar._Underlying *w);
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4<unsigned char>::Matrix4`.
        public unsafe Const_Matrix4_UnsignedChar(MR.Const_Matrix3_UnsignedChar r, MR.Const_Vector3_UnsignedChar t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Construct_2(MR.Const_Matrix3_UnsignedChar._Underlying *r, MR.Const_Vector3_UnsignedChar._Underlying *t);
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::zero`.
        public static unsafe MR.Matrix4_UnsignedChar Zero()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_zero", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_zero();
            return new(__MR_Matrix4_unsigned_char_zero(), is_owning: true);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::identity`.
        public static unsafe MR.Matrix4_UnsignedChar Identity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_identity", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_identity();
            return new(__MR_Matrix4_unsigned_char_identity(), is_owning: true);
        }

        /// returns a matrix that scales uniformly
        /// Generated from method `MR::Matrix4<unsigned char>::scale`.
        public static unsafe MR.Matrix4_UnsignedChar Scale(byte s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_scale", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_scale(byte s);
            return new(__MR_Matrix4_unsigned_char_scale(s), is_owning: true);
        }

        /// element access
        /// Generated from method `MR::Matrix4<unsigned char>::operator()`.
        public unsafe byte Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_call_const", ExactSpelling = true)]
            extern static byte *__MR_Matrix4_unsigned_char_call_const(_Underlying *_this, int row, int col);
            return *__MR_Matrix4_unsigned_char_call_const(_UnderlyingPtr, row, col);
        }

        /// row access
        /// Generated from method `MR::Matrix4<unsigned char>::operator[]`.
        public unsafe MR.Const_Vector4_UnsignedChar Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_index_const(_Underlying *_this, int row);
            return new(__MR_Matrix4_unsigned_char_index_const(_UnderlyingPtr, row), is_owning: false);
        }

        /// column access
        /// Generated from method `MR::Matrix4<unsigned char>::col`.
        public unsafe MR.Vector4_UnsignedChar Col(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_col", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_col(_Underlying *_this, int i);
            return new(__MR_Matrix4_unsigned_char_col(_UnderlyingPtr, i), is_owning: true);
        }

        /// computes trace of the matrix
        /// Generated from method `MR::Matrix4<unsigned char>::trace`.
        public unsafe byte Trace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_trace", ExactSpelling = true)]
            extern static byte __MR_Matrix4_unsigned_char_trace(_Underlying *_this);
            return __MR_Matrix4_unsigned_char_trace(_UnderlyingPtr);
        }

        /// compute sum of squared matrix elements
        /// Generated from method `MR::Matrix4<unsigned char>::normSq`.
        public unsafe byte NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_normSq", ExactSpelling = true)]
            extern static byte __MR_Matrix4_unsigned_char_normSq(_Underlying *_this);
            return __MR_Matrix4_unsigned_char_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_norm", ExactSpelling = true)]
            extern static double __MR_Matrix4_unsigned_char_norm(_Underlying *_this);
            return __MR_Matrix4_unsigned_char_norm(_UnderlyingPtr);
        }

        /// computes submatrix of the matrix with excluded i-th row and j-th column
        /// Generated from method `MR::Matrix4<unsigned char>::submatrix3`.
        public unsafe MR.Matrix3_UnsignedChar Submatrix3(int i, int j)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_submatrix3", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_submatrix3(_Underlying *_this, int i, int j);
            return new(__MR_Matrix4_unsigned_char_submatrix3(_UnderlyingPtr, i, j), is_owning: true);
        }

        /// computes determinant of the matrix
        /// Generated from method `MR::Matrix4<unsigned char>::det`.
        public unsafe byte Det()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_det", ExactSpelling = true)]
            extern static byte __MR_Matrix4_unsigned_char_det(_Underlying *_this);
            return __MR_Matrix4_unsigned_char_det(_UnderlyingPtr);
        }

        /// computes transposed matrix
        /// Generated from method `MR::Matrix4<unsigned char>::transposed`.
        public unsafe MR.Matrix4_UnsignedChar Transposed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_transposed", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_transposed(_Underlying *_this);
            return new(__MR_Matrix4_unsigned_char_transposed(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::getRotation`.
        public unsafe MR.Matrix3_UnsignedChar GetRotation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_getRotation", ExactSpelling = true)]
            extern static MR.Matrix3_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_getRotation(_Underlying *_this);
            return new(__MR_Matrix4_unsigned_char_getRotation(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::getTranslation`.
        public unsafe MR.Vector3_UnsignedChar GetTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_getTranslation", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_getTranslation(_Underlying *_this);
            return new(__MR_Matrix4_unsigned_char_getTranslation(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::data`.
        public unsafe byte? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_data_const", ExactSpelling = true)]
            extern static byte *__MR_Matrix4_unsigned_char_data_const(_Underlying *_this);
            var __ret = __MR_Matrix4_unsigned_char_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Matrix4_UnsignedChar a, MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Matrix4_unsigned_char(MR.Const_Matrix4_UnsignedChar._Underlying *a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return __MR_equal_MR_Matrix4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Matrix4_UnsignedChar a, MR.Const_Matrix4_UnsignedChar b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Matrix4i operator+(MR.Const_Matrix4_UnsignedChar a, MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_add_MR_Matrix4_unsigned_char(MR.Const_Matrix4_UnsignedChar._Underlying *a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return __MR_add_MR_Matrix4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Matrix4i operator-(MR.Const_Matrix4_UnsignedChar a, MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_sub_MR_Matrix4_unsigned_char(MR.Const_Matrix4_UnsignedChar._Underlying *a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return __MR_sub_MR_Matrix4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(byte a, MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_unsigned_char_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_unsigned_char_MR_Matrix4_unsigned_char(byte a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return __MR_mul_unsigned_char_MR_Matrix4_unsigned_char(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Const_Matrix4_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4_unsigned_char_unsigned_char(MR.Const_Matrix4_UnsignedChar._Underlying *b, byte a);
            return __MR_mul_MR_Matrix4_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Matrix4i operator/(Const_Matrix4_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Matrix4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_div_MR_Matrix4_unsigned_char_unsigned_char(MR.Matrix4_UnsignedChar._Underlying *b, byte a);
            return __MR_div_MR_Matrix4_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        /// x = a * b
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Const_Matrix4_UnsignedChar a, MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4_unsigned_char_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Matrix4_unsigned_char_MR_Vector4_unsigned_char(MR.Const_Matrix4_UnsignedChar._Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return __MR_mul_MR_Matrix4_unsigned_char_MR_Vector4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// product of two matrices
        /// Generated from function `MR::operator*`.
        public static unsafe MR.Matrix4i operator*(MR.Const_Matrix4_UnsignedChar a, MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4i __MR_mul_MR_Matrix4_unsigned_char(MR.Const_Matrix4_UnsignedChar._Underlying *a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return __MR_mul_MR_Matrix4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Matrix4_UnsignedChar? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Matrix4_UnsignedChar)
                return this == (MR.Const_Matrix4_UnsignedChar)other;
            return false;
        }
    }

    /// arbitrary 4x4 matrix
    /// Generated from class `MR::Matrix4<unsigned char>`.
    /// This is the non-const half of the class.
    public class Matrix4_UnsignedChar : Const_Matrix4_UnsignedChar
    {
        internal unsafe Matrix4_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// rows, identity matrix by default
        public new unsafe MR.Vector4_UnsignedChar X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_GetMutable_x", ExactSpelling = true)]
                extern static MR.Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_GetMutable_x(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_GetMutable_x(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Vector4_UnsignedChar Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_GetMutable_y", ExactSpelling = true)]
                extern static MR.Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_GetMutable_y(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_GetMutable_y(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Vector4_UnsignedChar Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_GetMutable_z", ExactSpelling = true)]
                extern static MR.Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_GetMutable_z(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_GetMutable_z(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Vector4_UnsignedChar W
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_GetMutable_w", ExactSpelling = true)]
                extern static MR.Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_GetMutable_w(_Underlying *_this);
                return new(__MR_Matrix4_unsigned_char_GetMutable_w(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Matrix4_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Matrix4<unsigned char>::Matrix4`.
        public unsafe Matrix4_UnsignedChar(MR.Const_Matrix4_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_ConstructFromAnother(MR.Matrix4_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// initializes matrix from 4 row-vectors
        /// Generated from constructor `MR::Matrix4<unsigned char>::Matrix4`.
        public unsafe Matrix4_UnsignedChar(MR.Const_Vector4_UnsignedChar x, MR.Const_Vector4_UnsignedChar y, MR.Const_Vector4_UnsignedChar z, MR.Const_Vector4_UnsignedChar w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Construct_4", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Construct_4(MR.Const_Vector4_UnsignedChar._Underlying *x, MR.Const_Vector4_UnsignedChar._Underlying *y, MR.Const_Vector4_UnsignedChar._Underlying *z, MR.Const_Vector4_UnsignedChar._Underlying *w);
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_Construct_4(x._UnderlyingPtr, y._UnderlyingPtr, z._UnderlyingPtr, w._UnderlyingPtr);
        }

        /// construct from rotation matrix and translation vector
        /// Generated from constructor `MR::Matrix4<unsigned char>::Matrix4`.
        public unsafe Matrix4_UnsignedChar(MR.Const_Matrix3_UnsignedChar r, MR.Const_Vector3_UnsignedChar t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_Construct_2", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_Construct_2(MR.Const_Matrix3_UnsignedChar._Underlying *r, MR.Const_Vector3_UnsignedChar._Underlying *t);
            _UnderlyingPtr = __MR_Matrix4_unsigned_char_Construct_2(r._UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::operator=`.
        public unsafe MR.Matrix4_UnsignedChar Assign(MR.Const_Matrix4_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Matrix4_UnsignedChar._Underlying *_other);
            return new(__MR_Matrix4_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::operator()`.
        public unsafe new ref byte Call(int row, int col)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_call", ExactSpelling = true)]
            extern static byte *__MR_Matrix4_unsigned_char_call(_Underlying *_this, int row, int col);
            return ref *__MR_Matrix4_unsigned_char_call(_UnderlyingPtr, row, col);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::operator[]`.
        public unsafe new MR.Vector4_UnsignedChar Index(int row)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_index", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Matrix4_unsigned_char_index(_Underlying *_this, int row);
            return new(__MR_Matrix4_unsigned_char_index(_UnderlyingPtr, row), is_owning: false);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::setRotation`.
        public unsafe void SetRotation(MR.Const_Matrix3_UnsignedChar rot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_setRotation", ExactSpelling = true)]
            extern static void __MR_Matrix4_unsigned_char_setRotation(_Underlying *_this, MR.Const_Matrix3_UnsignedChar._Underlying *rot);
            __MR_Matrix4_unsigned_char_setRotation(_UnderlyingPtr, rot._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::setTranslation`.
        public unsafe void SetTranslation(MR.Const_Vector3_UnsignedChar t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_setTranslation", ExactSpelling = true)]
            extern static void __MR_Matrix4_unsigned_char_setTranslation(_Underlying *_this, MR.Const_Vector3_UnsignedChar._Underlying *t);
            __MR_Matrix4_unsigned_char_setTranslation(_UnderlyingPtr, t._UnderlyingPtr);
        }

        /// Generated from method `MR::Matrix4<unsigned char>::data`.
        public unsafe new MR.Misc.Ref<byte>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Matrix4_unsigned_char_data", ExactSpelling = true)]
            extern static byte *__MR_Matrix4_unsigned_char_data(_Underlying *_this);
            var __ret = __MR_Matrix4_unsigned_char_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<byte>(__ret) : null;
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Matrix4_UnsignedChar AddAssign(MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_add_assign_MR_Matrix4_unsigned_char(_Underlying *a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return new(__MR_add_assign_MR_Matrix4_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Matrix4_UnsignedChar SubAssign(MR.Const_Matrix4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Matrix4_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_sub_assign_MR_Matrix4_unsigned_char(_Underlying *a, MR.Const_Matrix4_UnsignedChar._Underlying *b);
            return new(__MR_sub_assign_MR_Matrix4_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Matrix4_UnsignedChar MulAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Matrix4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_mul_assign_MR_Matrix4_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Matrix4_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Matrix4_UnsignedChar DivAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Matrix4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Matrix4_UnsignedChar._Underlying *__MR_div_assign_MR_Matrix4_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Matrix4_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Matrix4_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Matrix4_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Matrix4_UnsignedChar`/`Const_Matrix4_UnsignedChar` directly.
    public class _InOptMut_Matrix4_UnsignedChar
    {
        public Matrix4_UnsignedChar? Opt;

        public _InOptMut_Matrix4_UnsignedChar() {}
        public _InOptMut_Matrix4_UnsignedChar(Matrix4_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_Matrix4_UnsignedChar(Matrix4_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `Matrix4_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Matrix4_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Matrix4_UnsignedChar`/`Const_Matrix4_UnsignedChar` to pass it to the function.
    public class _InOptConst_Matrix4_UnsignedChar
    {
        public Const_Matrix4_UnsignedChar? Opt;

        public _InOptConst_Matrix4_UnsignedChar() {}
        public _InOptConst_Matrix4_UnsignedChar(Const_Matrix4_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_Matrix4_UnsignedChar(Const_Matrix4_UnsignedChar value) {return new(value);}
    }
}
