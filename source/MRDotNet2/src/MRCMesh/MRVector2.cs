public static partial class MR
{
    /// two-dimensional vector
    /// Generated from class `MR::Vector2b`.
    /// This is the const reference to the struct.
    public class Const_Vector2b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector2b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector2b UnderlyingStruct => ref *(Vector2b *)_UnderlyingPtr;

        internal unsafe Const_Vector2b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector2b_Destroy(_Underlying *_this);
            __MR_Vector2b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector2b() {Dispose(false);}

        public bool X => UnderlyingStruct.X;

        public bool Y => UnderlyingStruct.Y;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector2b_Get_elements();
                return *__MR_Vector2b_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector2b(Const_Vector2b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 2);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector2b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            MR.Vector2b _ctor_result = __MR_Vector2b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 2);
        }

        /// Generated from constructor `MR::Vector2b::Vector2b`.
        public unsafe Const_Vector2b(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            MR.Vector2b _ctor_result = __MR_Vector2b_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 2);
        }

        /// Generated from constructor `MR::Vector2b::Vector2b`.
        public unsafe Const_Vector2b(bool x, bool y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_Construct_2(byte x, byte y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            MR.Vector2b _ctor_result = __MR_Vector2b_Construct_2(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 2);
        }

        /// Generated from method `MR::Vector2b::diagonal`.
        public static MR.Vector2b Diagonal(bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_diagonal", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_diagonal(byte a);
            return __MR_Vector2b_diagonal(a ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector2b::plusX`.
        public static MR.Vector2b PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_plusX", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_plusX();
            return __MR_Vector2b_plusX();
        }

        /// Generated from method `MR::Vector2b::plusY`.
        public static MR.Vector2b PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_plusY", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_plusY();
            return __MR_Vector2b_plusY();
        }

        /// Generated from method `MR::Vector2b::operator[]`.
        public unsafe bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_index_const", ExactSpelling = true)]
            extern static bool *__MR_Vector2b_index_const(_Underlying *_this, int e);
            return *__MR_Vector2b_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector2b::lengthSq`.
        public unsafe bool LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector2b_lengthSq(_Underlying *_this);
            return __MR_Vector2b_lengthSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Vector2b::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_length", ExactSpelling = true)]
            extern static double __MR_Vector2b_length(_Underlying *_this);
            return __MR_Vector2b_length(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_equal_MR_Vector2b(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector2b a, MR.Const_Vector2b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2b operator+(MR.Const_Vector2b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Const_Vector2b._Underlying *__MR_pos_MR_Vector2b(MR.Const_Vector2b._Underlying *a);
            return new(__MR_pos_MR_Vector2b(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Const_Vector2b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_neg_MR_Vector2b(MR.Const_Vector2b._Underlying *a);
            return __MR_neg_MR_Vector2b(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2i operator+(MR.Const_Vector2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_add_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_add_MR_Vector2b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Const_Vector2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_sub_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_sub_MR_Vector2b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(bool a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_bool_MR_Vector2b(byte a, MR.Const_Vector2b._Underlying *b);
            return __MR_mul_bool_MR_Vector2b(a ? (byte)1 : (byte)0, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Const_Vector2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Vector2b_bool(MR.Const_Vector2b._Underlying *b, byte a);
            return __MR_mul_MR_Vector2b_bool(b._UnderlyingPtr, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector2i operator/(Const_Vector2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Vector2i __MR_div_MR_Vector2b_bool(MR.Vector2b b, byte a);
            return __MR_div_MR_Vector2b_bool(b.UnderlyingStruct, a ? (byte)1 : (byte)0);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector2b? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector2b)
                return this == (MR.Const_Vector2b)other;
            return false;
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2b`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector2b : Const_Vector2b
    {
        /// Get the underlying struct.
        public unsafe new ref Vector2b UnderlyingStruct => ref *(Vector2b *)_UnderlyingPtr;

        internal unsafe Mut_Vector2b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new bool X {get => UnderlyingStruct.X; set => UnderlyingStruct.X = value;}

        public new bool Y {get => UnderlyingStruct.Y; set => UnderlyingStruct.Y = value;}

        /// Generated copy constructor.
        public unsafe Mut_Vector2b(Const_Vector2b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 2);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector2b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            MR.Vector2b _ctor_result = __MR_Vector2b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 2);
        }

        /// Generated from constructor `MR::Vector2b::Vector2b`.
        public unsafe Mut_Vector2b(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            MR.Vector2b _ctor_result = __MR_Vector2b_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 2);
        }

        /// Generated from constructor `MR::Vector2b::Vector2b`.
        public unsafe Mut_Vector2b(bool x, bool y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_Construct_2(byte x, byte y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(2);
            MR.Vector2b _ctor_result = __MR_Vector2b_Construct_2(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 2);
        }

        /// Generated from method `MR::Vector2b::operator[]`.
        public unsafe new ref bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_index", ExactSpelling = true)]
            extern static bool *__MR_Vector2b_index(_Underlying *_this, int e);
            return ref *__MR_Vector2b_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2b AddAssign(MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_add_assign_MR_Vector2b(_Underlying *a, MR.Const_Vector2b._Underlying *b);
            return new(__MR_add_assign_MR_Vector2b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2b SubAssign(MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_sub_assign_MR_Vector2b(_Underlying *a, MR.Const_Vector2b._Underlying *b);
            return new(__MR_sub_assign_MR_Vector2b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_mul_assign_MR_Vector2b_bool(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Vector2b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_div_assign_MR_Vector2b_bool(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Vector2b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2b`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 2)]
    public struct Vector2b
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector2b(Const_Vector2b other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector2b(Vector2b other) => new(new Mut_Vector2b((Mut_Vector2b._Underlying *)&other, is_owning: false));

        public bool X {get => __storage_X != 0; set => __storage_X = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(0)]
        byte __storage_X;

        public bool Y {get => __storage_Y != 0; set => __storage_Y = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(1)]
        byte __storage_Y;

        /// Generated copy constructor.
        public Vector2b(Vector2b _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector2b()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_DefaultConstruct();
            this = __MR_Vector2b_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector2b::Vector2b`.
        public unsafe Vector2b(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector2b_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector2b::Vector2b`.
        public unsafe Vector2b(bool x, bool y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_Construct_2(byte x, byte y);
            this = __MR_Vector2b_Construct_2(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector2b::diagonal`.
        public static MR.Vector2b Diagonal(bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_diagonal", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_diagonal(byte a);
            return __MR_Vector2b_diagonal(a ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector2b::plusX`.
        public static MR.Vector2b PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_plusX", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_plusX();
            return __MR_Vector2b_plusX();
        }

        /// Generated from method `MR::Vector2b::plusY`.
        public static MR.Vector2b PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_plusY", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Vector2b_plusY();
            return __MR_Vector2b_plusY();
        }

        /// Generated from method `MR::Vector2b::operator[]`.
        public unsafe bool Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_index_const", ExactSpelling = true)]
            extern static bool *__MR_Vector2b_index_const(MR.Vector2b *_this, int e);
            fixed (MR.Vector2b *__ptr__this = &this)
            {
                return *__MR_Vector2b_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2b::operator[]`.
        public unsafe ref bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_index", ExactSpelling = true)]
            extern static bool *__MR_Vector2b_index(MR.Vector2b *_this, int e);
            fixed (MR.Vector2b *__ptr__this = &this)
            {
                return ref *__MR_Vector2b_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2b::lengthSq`.
        public unsafe bool LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector2b_lengthSq(MR.Vector2b *_this);
            fixed (MR.Vector2b *__ptr__this = &this)
            {
                return __MR_Vector2b_lengthSq(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::Vector2b::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2b_length", ExactSpelling = true)]
            extern static double __MR_Vector2b_length(MR.Vector2b *_this);
            fixed (MR.Vector2b *__ptr__this = &this)
            {
                return __MR_Vector2b_length(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector2b a, MR.Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_equal_MR_Vector2b((MR.Mut_Vector2b._Underlying *)&a, (MR.Mut_Vector2b._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector2b a, MR.Vector2b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2b operator+(MR.Vector2b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Const_Vector2b._Underlying *__MR_pos_MR_Vector2b(MR.Const_Vector2b._Underlying *a);
            return new(__MR_pos_MR_Vector2b((MR.Mut_Vector2b._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Vector2b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_neg_MR_Vector2b(MR.Const_Vector2b._Underlying *a);
            return __MR_neg_MR_Vector2b((MR.Mut_Vector2b._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2i operator+(MR.Vector2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_add_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_add_MR_Vector2b((MR.Mut_Vector2b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Vector2b a, MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_sub_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
            return __MR_sub_MR_Vector2b((MR.Mut_Vector2b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(bool a, MR.Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_bool_MR_Vector2b(byte a, MR.Const_Vector2b._Underlying *b);
            return __MR_mul_bool_MR_Vector2b(a ? (byte)1 : (byte)0, (MR.Mut_Vector2b._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Vector2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Vector2b_bool(MR.Const_Vector2b._Underlying *b, byte a);
            return __MR_mul_MR_Vector2b_bool((MR.Mut_Vector2b._Underlying *)&b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector2i operator/(MR.Vector2b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Vector2i __MR_div_MR_Vector2b_bool(MR.Vector2b b, byte a);
            return __MR_div_MR_Vector2b_bool(b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2b AddAssign(MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_add_assign_MR_Vector2b(MR.Vector2b *a, MR.Const_Vector2b._Underlying *b);
            fixed (MR.Vector2b *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector2b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2b SubAssign(MR.Const_Vector2b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2b", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_sub_assign_MR_Vector2b(MR.Vector2b *a, MR.Const_Vector2b._Underlying *b);
            fixed (MR.Vector2b *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector2b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_mul_assign_MR_Vector2b_bool(MR.Vector2b *a, byte b);
            fixed (MR.Vector2b *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector2b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector2b._Underlying *__MR_div_assign_MR_Vector2b_bool(MR.Vector2b *a, byte b);
            fixed (MR.Vector2b *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector2b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector2b b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector2b)
                return this == (MR.Vector2b)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector2b` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector2b`/`Const_Vector2b` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector2b
    {
        public readonly bool HasValue;
        internal readonly Vector2b Object;
        public Vector2b Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector2b() {HasValue = false;}
        public _InOpt_Vector2b(Vector2b new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector2b(Vector2b new_value) {return new(new_value);}
        public _InOpt_Vector2b(Const_Vector2b new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector2b(Const_Vector2b new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector2b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector2b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2b`/`Const_Vector2b` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2b`.
    public class _InOptMut_Vector2b
    {
        public Mut_Vector2b? Opt;

        public _InOptMut_Vector2b() {}
        public _InOptMut_Vector2b(Mut_Vector2b value) {Opt = value;}
        public static implicit operator _InOptMut_Vector2b(Mut_Vector2b value) {return new(value);}
        public unsafe _InOptMut_Vector2b(ref Vector2b value)
        {
            fixed (Vector2b *value_ptr = &value)
            {
                Opt = new((Const_Vector2b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector2b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector2b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2b`/`Const_Vector2b` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2b`.
    public class _InOptConst_Vector2b
    {
        public Const_Vector2b? Opt;

        public _InOptConst_Vector2b() {}
        public _InOptConst_Vector2b(Const_Vector2b value) {Opt = value;}
        public static implicit operator _InOptConst_Vector2b(Const_Vector2b value) {return new(value);}
        public unsafe _InOptConst_Vector2b(ref readonly Vector2b value)
        {
            fixed (Vector2b *value_ptr = &value)
            {
                Opt = new((Const_Vector2b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2i`.
    /// This is the const reference to the struct.
    public class Const_Vector2i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector2i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector2i UnderlyingStruct => ref *(Vector2i *)_UnderlyingPtr;

        internal unsafe Const_Vector2i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector2i_Destroy(_Underlying *_this);
            __MR_Vector2i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector2i() {Dispose(false);}

        public ref readonly int X => ref UnderlyingStruct.X;

        public ref readonly int Y => ref UnderlyingStruct.Y;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector2i_Get_elements();
                return *__MR_Vector2i_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector2i(Const_Vector2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2i _ctor_result = __MR_Vector2i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2i::Vector2i`.
        public unsafe Const_Vector2i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2i _ctor_result = __MR_Vector2i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2i::Vector2i`.
        public unsafe Const_Vector2i(int x, int y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_Construct_2(int x, int y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2i _ctor_result = __MR_Vector2i_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::Vector2i::diagonal`.
        public static MR.Vector2i Diagonal(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_diagonal", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_diagonal(int a);
            return __MR_Vector2i_diagonal(a);
        }

        /// Generated from method `MR::Vector2i::plusX`.
        public static MR.Vector2i PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_plusX", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_plusX();
            return __MR_Vector2i_plusX();
        }

        /// Generated from method `MR::Vector2i::plusY`.
        public static MR.Vector2i PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_plusY", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_plusY();
            return __MR_Vector2i_plusY();
        }

        /// Generated from method `MR::Vector2i::minusX`.
        public static MR.Vector2i MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_minusX", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_minusX();
            return __MR_Vector2i_minusX();
        }

        /// Generated from method `MR::Vector2i::minusY`.
        public static MR.Vector2i MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_minusY", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_minusY();
            return __MR_Vector2i_minusY();
        }

        /// Generated from method `MR::Vector2i::operator[]`.
        public unsafe int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_index_const", ExactSpelling = true)]
            extern static int *__MR_Vector2i_index_const(_Underlying *_this, int e);
            return *__MR_Vector2i_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector2i::lengthSq`.
        public unsafe int LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_lengthSq", ExactSpelling = true)]
            extern static int __MR_Vector2i_lengthSq(_Underlying *_this);
            return __MR_Vector2i_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2i::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_length", ExactSpelling = true)]
            extern static double __MR_Vector2i_length(_Underlying *_this);
            return __MR_Vector2i_length(_UnderlyingPtr);
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2i::furthestBasisVector`.
        public unsafe MR.Vector2i FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_furthestBasisVector(_Underlying *_this);
            return __MR_Vector2i_furthestBasisVector(_UnderlyingPtr);
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2i::perpendicular`.
        public unsafe MR.Vector2i Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_perpendicular(_Underlying *_this);
            return __MR_Vector2i_perpendicular(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_equal_MR_Vector2i(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector2i a, MR.Const_Vector2i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2i operator+(MR.Const_Vector2i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_pos_MR_Vector2i(MR.Const_Vector2i._Underlying *a);
            return new(__MR_pos_MR_Vector2i(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Const_Vector2i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_neg_MR_Vector2i(MR.Const_Vector2i._Underlying *a);
            return __MR_neg_MR_Vector2i(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2i operator+(MR.Const_Vector2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_add_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_add_MR_Vector2i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Const_Vector2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_sub_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_sub_MR_Vector2i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(int a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_int_MR_Vector2i(int a, MR.Const_Vector2i._Underlying *b);
            return __MR_mul_int_MR_Vector2i(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Const_Vector2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Vector2i_int(MR.Const_Vector2i._Underlying *b, int a);
            return __MR_mul_MR_Vector2i_int(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector2i operator/(Const_Vector2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Vector2i __MR_div_MR_Vector2i_int(MR.Vector2i b, int a);
            return __MR_div_MR_Vector2i_int(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector2i? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector2i)
                return this == (MR.Const_Vector2i)other;
            return false;
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2i`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector2i : Const_Vector2i
    {
        /// Get the underlying struct.
        public unsafe new ref Vector2i UnderlyingStruct => ref *(Vector2i *)_UnderlyingPtr;

        internal unsafe Mut_Vector2i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int X => ref UnderlyingStruct.X;

        public new ref int Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Vector2i(Const_Vector2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2i _ctor_result = __MR_Vector2i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2i::Vector2i`.
        public unsafe Mut_Vector2i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2i _ctor_result = __MR_Vector2i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2i::Vector2i`.
        public unsafe Mut_Vector2i(int x, int y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_Construct_2(int x, int y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2i _ctor_result = __MR_Vector2i_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::Vector2i::operator[]`.
        public unsafe new ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_index", ExactSpelling = true)]
            extern static int *__MR_Vector2i_index(_Underlying *_this, int e);
            return ref *__MR_Vector2i_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2i AddAssign(MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_add_assign_MR_Vector2i(_Underlying *a, MR.Const_Vector2i._Underlying *b);
            return new(__MR_add_assign_MR_Vector2i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2i SubAssign(MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_sub_assign_MR_Vector2i(_Underlying *a, MR.Const_Vector2i._Underlying *b);
            return new(__MR_sub_assign_MR_Vector2i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_mul_assign_MR_Vector2i_int(_Underlying *a, int b);
            return new(__MR_mul_assign_MR_Vector2i_int(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_div_assign_MR_Vector2i_int(_Underlying *a, int b);
            return new(__MR_div_assign_MR_Vector2i_int(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
    public struct Vector2i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector2i(Const_Vector2i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector2i(Vector2i other) => new(new Mut_Vector2i((Mut_Vector2i._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public int Y;

        /// Generated copy constructor.
        public Vector2i(Vector2i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector2i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_DefaultConstruct();
            this = __MR_Vector2i_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector2i::Vector2i`.
        public unsafe Vector2i(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector2i_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector2i::Vector2i`.
        public unsafe Vector2i(int x, int y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_Construct_2(int x, int y);
            this = __MR_Vector2i_Construct_2(x, y);
        }

        /// Generated from method `MR::Vector2i::diagonal`.
        public static MR.Vector2i Diagonal(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_diagonal", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_diagonal(int a);
            return __MR_Vector2i_diagonal(a);
        }

        /// Generated from method `MR::Vector2i::plusX`.
        public static MR.Vector2i PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_plusX", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_plusX();
            return __MR_Vector2i_plusX();
        }

        /// Generated from method `MR::Vector2i::plusY`.
        public static MR.Vector2i PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_plusY", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_plusY();
            return __MR_Vector2i_plusY();
        }

        /// Generated from method `MR::Vector2i::minusX`.
        public static MR.Vector2i MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_minusX", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_minusX();
            return __MR_Vector2i_minusX();
        }

        /// Generated from method `MR::Vector2i::minusY`.
        public static MR.Vector2i MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_minusY", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_minusY();
            return __MR_Vector2i_minusY();
        }

        /// Generated from method `MR::Vector2i::operator[]`.
        public unsafe int Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_index_const", ExactSpelling = true)]
            extern static int *__MR_Vector2i_index_const(MR.Vector2i *_this, int e);
            fixed (MR.Vector2i *__ptr__this = &this)
            {
                return *__MR_Vector2i_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2i::operator[]`.
        public unsafe ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_index", ExactSpelling = true)]
            extern static int *__MR_Vector2i_index(MR.Vector2i *_this, int e);
            fixed (MR.Vector2i *__ptr__this = &this)
            {
                return ref *__MR_Vector2i_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2i::lengthSq`.
        public unsafe int LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_lengthSq", ExactSpelling = true)]
            extern static int __MR_Vector2i_lengthSq(MR.Vector2i *_this);
            fixed (MR.Vector2i *__ptr__this = &this)
            {
                return __MR_Vector2i_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2i::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_length", ExactSpelling = true)]
            extern static double __MR_Vector2i_length(MR.Vector2i *_this);
            fixed (MR.Vector2i *__ptr__this = &this)
            {
                return __MR_Vector2i_length(__ptr__this);
            }
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2i::furthestBasisVector`.
        public unsafe MR.Vector2i FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_furthestBasisVector(MR.Vector2i *_this);
            fixed (MR.Vector2i *__ptr__this = &this)
            {
                return __MR_Vector2i_furthestBasisVector(__ptr__this);
            }
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2i::perpendicular`.
        public unsafe MR.Vector2i Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Vector2i_perpendicular(MR.Vector2i *_this);
            fixed (MR.Vector2i *__ptr__this = &this)
            {
                return __MR_Vector2i_perpendicular(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector2i a, MR.Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_equal_MR_Vector2i((MR.Mut_Vector2i._Underlying *)&a, (MR.Mut_Vector2i._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector2i a, MR.Vector2i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2i operator+(MR.Vector2i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_pos_MR_Vector2i(MR.Const_Vector2i._Underlying *a);
            return new(__MR_pos_MR_Vector2i((MR.Mut_Vector2i._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Vector2i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_neg_MR_Vector2i(MR.Const_Vector2i._Underlying *a);
            return __MR_neg_MR_Vector2i((MR.Mut_Vector2i._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2i operator+(MR.Vector2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_add_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_add_MR_Vector2i((MR.Mut_Vector2i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i operator-(MR.Vector2i a, MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_sub_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
            return __MR_sub_MR_Vector2i((MR.Mut_Vector2i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(int a, MR.Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_int_MR_Vector2i(int a, MR.Const_Vector2i._Underlying *b);
            return __MR_mul_int_MR_Vector2i(a, (MR.Mut_Vector2i._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i operator*(MR.Vector2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Vector2i __MR_mul_MR_Vector2i_int(MR.Const_Vector2i._Underlying *b, int a);
            return __MR_mul_MR_Vector2i_int((MR.Mut_Vector2i._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector2i operator/(MR.Vector2i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Vector2i __MR_div_MR_Vector2i_int(MR.Vector2i b, int a);
            return __MR_div_MR_Vector2i_int(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2i AddAssign(MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_add_assign_MR_Vector2i(MR.Vector2i *a, MR.Const_Vector2i._Underlying *b);
            fixed (MR.Vector2i *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector2i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2i SubAssign(MR.Const_Vector2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2i", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_sub_assign_MR_Vector2i(MR.Vector2i *a, MR.Const_Vector2i._Underlying *b);
            fixed (MR.Vector2i *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector2i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_mul_assign_MR_Vector2i_int(MR.Vector2i *a, int b);
            fixed (MR.Vector2i *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector2i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_div_assign_MR_Vector2i_int(MR.Vector2i *a, int b);
            fixed (MR.Vector2i *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector2i_int(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector2i b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector2i)
                return this == (MR.Vector2i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector2i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector2i`/`Const_Vector2i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector2i
    {
        public readonly bool HasValue;
        internal readonly Vector2i Object;
        public Vector2i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector2i() {HasValue = false;}
        public _InOpt_Vector2i(Vector2i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector2i(Vector2i new_value) {return new(new_value);}
        public _InOpt_Vector2i(Const_Vector2i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector2i(Const_Vector2i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector2i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2i`/`Const_Vector2i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2i`.
    public class _InOptMut_Vector2i
    {
        public Mut_Vector2i? Opt;

        public _InOptMut_Vector2i() {}
        public _InOptMut_Vector2i(Mut_Vector2i value) {Opt = value;}
        public static implicit operator _InOptMut_Vector2i(Mut_Vector2i value) {return new(value);}
        public unsafe _InOptMut_Vector2i(ref Vector2i value)
        {
            fixed (Vector2i *value_ptr = &value)
            {
                Opt = new((Const_Vector2i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector2i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2i`/`Const_Vector2i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2i`.
    public class _InOptConst_Vector2i
    {
        public Const_Vector2i? Opt;

        public _InOptConst_Vector2i() {}
        public _InOptConst_Vector2i(Const_Vector2i value) {Opt = value;}
        public static implicit operator _InOptConst_Vector2i(Const_Vector2i value) {return new(value);}
        public unsafe _InOptConst_Vector2i(ref readonly Vector2i value)
        {
            fixed (Vector2i *value_ptr = &value)
            {
                Opt = new((Const_Vector2i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2i64`.
    /// This is the const reference to the struct.
    public class Const_Vector2i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector2i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector2i64 UnderlyingStruct => ref *(Vector2i64 *)_UnderlyingPtr;

        internal unsafe Const_Vector2i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector2i64_Destroy(_Underlying *_this);
            __MR_Vector2i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector2i64() {Dispose(false);}

        public ref readonly long X => ref UnderlyingStruct.X;

        public ref readonly long Y => ref UnderlyingStruct.Y;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector2i64_Get_elements();
                return *__MR_Vector2i64_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector2i64(Const_Vector2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2i64 _ctor_result = __MR_Vector2i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2i64::Vector2i64`.
        public unsafe Const_Vector2i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2i64 _ctor_result = __MR_Vector2i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2i64::Vector2i64`.
        public unsafe Const_Vector2i64(long x, long y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_Construct_2(long x, long y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2i64 _ctor_result = __MR_Vector2i64_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector2i64::diagonal`.
        public static MR.Vector2i64 Diagonal(long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_diagonal", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_diagonal(long a);
            return __MR_Vector2i64_diagonal(a);
        }

        /// Generated from method `MR::Vector2i64::plusX`.
        public static MR.Vector2i64 PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_plusX", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_plusX();
            return __MR_Vector2i64_plusX();
        }

        /// Generated from method `MR::Vector2i64::plusY`.
        public static MR.Vector2i64 PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_plusY", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_plusY();
            return __MR_Vector2i64_plusY();
        }

        /// Generated from method `MR::Vector2i64::minusX`.
        public static MR.Vector2i64 MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_minusX", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_minusX();
            return __MR_Vector2i64_minusX();
        }

        /// Generated from method `MR::Vector2i64::minusY`.
        public static MR.Vector2i64 MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_minusY", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_minusY();
            return __MR_Vector2i64_minusY();
        }

        /// Generated from method `MR::Vector2i64::operator[]`.
        public unsafe long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Vector2i64_index_const(_Underlying *_this, int e);
            return *__MR_Vector2i64_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector2i64::lengthSq`.
        public unsafe long LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_lengthSq", ExactSpelling = true)]
            extern static long __MR_Vector2i64_lengthSq(_Underlying *_this);
            return __MR_Vector2i64_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2i64::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_length", ExactSpelling = true)]
            extern static double __MR_Vector2i64_length(_Underlying *_this);
            return __MR_Vector2i64_length(_UnderlyingPtr);
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2i64::furthestBasisVector`.
        public unsafe MR.Vector2i64 FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_furthestBasisVector(_Underlying *_this);
            return __MR_Vector2i64_furthestBasisVector(_UnderlyingPtr);
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2i64::perpendicular`.
        public unsafe MR.Vector2i64 Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_perpendicular(_Underlying *_this);
            return __MR_Vector2i64_perpendicular(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_equal_MR_Vector2i64(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector2i64 a, MR.Const_Vector2i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2i64 operator+(MR.Const_Vector2i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Const_Vector2i64._Underlying *__MR_pos_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a);
            return new(__MR_pos_MR_Vector2i64(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i64 operator-(MR.Const_Vector2i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_neg_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a);
            return __MR_neg_MR_Vector2i64(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2i64 operator+(MR.Const_Vector2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_add_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_add_MR_Vector2i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i64 operator-(MR.Const_Vector2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_sub_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_sub_MR_Vector2i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i64 operator*(long a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_mul_int64_t_MR_Vector2i64(long a, MR.Const_Vector2i64._Underlying *b);
            return __MR_mul_int64_t_MR_Vector2i64(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i64 operator*(MR.Const_Vector2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_mul_MR_Vector2i64_int64_t(MR.Const_Vector2i64._Underlying *b, long a);
            return __MR_mul_MR_Vector2i64_int64_t(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector2i64 operator/(Const_Vector2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_div_MR_Vector2i64_int64_t(MR.Vector2i64 b, long a);
            return __MR_div_MR_Vector2i64_int64_t(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector2i64? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector2i64)
                return this == (MR.Const_Vector2i64)other;
            return false;
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector2i64 : Const_Vector2i64
    {
        /// Get the underlying struct.
        public unsafe new ref Vector2i64 UnderlyingStruct => ref *(Vector2i64 *)_UnderlyingPtr;

        internal unsafe Mut_Vector2i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref long X => ref UnderlyingStruct.X;

        public new ref long Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Vector2i64(Const_Vector2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2i64 _ctor_result = __MR_Vector2i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2i64::Vector2i64`.
        public unsafe Mut_Vector2i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2i64 _ctor_result = __MR_Vector2i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2i64::Vector2i64`.
        public unsafe Mut_Vector2i64(long x, long y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_Construct_2(long x, long y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2i64 _ctor_result = __MR_Vector2i64_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector2i64::operator[]`.
        public unsafe new ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_index", ExactSpelling = true)]
            extern static long *__MR_Vector2i64_index(_Underlying *_this, int e);
            return ref *__MR_Vector2i64_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2i64 AddAssign(MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_add_assign_MR_Vector2i64(_Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return new(__MR_add_assign_MR_Vector2i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2i64 SubAssign(MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_sub_assign_MR_Vector2i64(_Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return new(__MR_sub_assign_MR_Vector2i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_mul_assign_MR_Vector2i64_int64_t(_Underlying *a, long b);
            return new(__MR_mul_assign_MR_Vector2i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_div_assign_MR_Vector2i64_int64_t(_Underlying *a, long b);
            return new(__MR_div_assign_MR_Vector2i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Vector2i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector2i64(Const_Vector2i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector2i64(Vector2i64 other) => new(new Mut_Vector2i64((Mut_Vector2i64._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public long X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public long Y;

        /// Generated copy constructor.
        public Vector2i64(Vector2i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector2i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_DefaultConstruct();
            this = __MR_Vector2i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector2i64::Vector2i64`.
        public unsafe Vector2i64(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector2i64_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector2i64::Vector2i64`.
        public unsafe Vector2i64(long x, long y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_Construct_2(long x, long y);
            this = __MR_Vector2i64_Construct_2(x, y);
        }

        /// Generated from method `MR::Vector2i64::diagonal`.
        public static MR.Vector2i64 Diagonal(long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_diagonal", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_diagonal(long a);
            return __MR_Vector2i64_diagonal(a);
        }

        /// Generated from method `MR::Vector2i64::plusX`.
        public static MR.Vector2i64 PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_plusX", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_plusX();
            return __MR_Vector2i64_plusX();
        }

        /// Generated from method `MR::Vector2i64::plusY`.
        public static MR.Vector2i64 PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_plusY", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_plusY();
            return __MR_Vector2i64_plusY();
        }

        /// Generated from method `MR::Vector2i64::minusX`.
        public static MR.Vector2i64 MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_minusX", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_minusX();
            return __MR_Vector2i64_minusX();
        }

        /// Generated from method `MR::Vector2i64::minusY`.
        public static MR.Vector2i64 MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_minusY", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_minusY();
            return __MR_Vector2i64_minusY();
        }

        /// Generated from method `MR::Vector2i64::operator[]`.
        public unsafe long Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Vector2i64_index_const(MR.Vector2i64 *_this, int e);
            fixed (MR.Vector2i64 *__ptr__this = &this)
            {
                return *__MR_Vector2i64_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2i64::operator[]`.
        public unsafe ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_index", ExactSpelling = true)]
            extern static long *__MR_Vector2i64_index(MR.Vector2i64 *_this, int e);
            fixed (MR.Vector2i64 *__ptr__this = &this)
            {
                return ref *__MR_Vector2i64_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2i64::lengthSq`.
        public unsafe long LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_lengthSq", ExactSpelling = true)]
            extern static long __MR_Vector2i64_lengthSq(MR.Vector2i64 *_this);
            fixed (MR.Vector2i64 *__ptr__this = &this)
            {
                return __MR_Vector2i64_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2i64::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_length", ExactSpelling = true)]
            extern static double __MR_Vector2i64_length(MR.Vector2i64 *_this);
            fixed (MR.Vector2i64 *__ptr__this = &this)
            {
                return __MR_Vector2i64_length(__ptr__this);
            }
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2i64::furthestBasisVector`.
        public unsafe MR.Vector2i64 FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_furthestBasisVector(MR.Vector2i64 *_this);
            fixed (MR.Vector2i64 *__ptr__this = &this)
            {
                return __MR_Vector2i64_furthestBasisVector(__ptr__this);
            }
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2i64::perpendicular`.
        public unsafe MR.Vector2i64 Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2i64_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Vector2i64_perpendicular(MR.Vector2i64 *_this);
            fixed (MR.Vector2i64 *__ptr__this = &this)
            {
                return __MR_Vector2i64_perpendicular(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector2i64 a, MR.Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_equal_MR_Vector2i64((MR.Mut_Vector2i64._Underlying *)&a, (MR.Mut_Vector2i64._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector2i64 a, MR.Vector2i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2i64 operator+(MR.Vector2i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Const_Vector2i64._Underlying *__MR_pos_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a);
            return new(__MR_pos_MR_Vector2i64((MR.Mut_Vector2i64._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i64 operator-(MR.Vector2i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_neg_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a);
            return __MR_neg_MR_Vector2i64((MR.Mut_Vector2i64._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2i64 operator+(MR.Vector2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_add_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_add_MR_Vector2i64((MR.Mut_Vector2i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2i64 operator-(MR.Vector2i64 a, MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_sub_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
            return __MR_sub_MR_Vector2i64((MR.Mut_Vector2i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i64 operator*(long a, MR.Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_mul_int64_t_MR_Vector2i64(long a, MR.Const_Vector2i64._Underlying *b);
            return __MR_mul_int64_t_MR_Vector2i64(a, (MR.Mut_Vector2i64._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2i64 operator*(MR.Vector2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_mul_MR_Vector2i64_int64_t(MR.Const_Vector2i64._Underlying *b, long a);
            return __MR_mul_MR_Vector2i64_int64_t((MR.Mut_Vector2i64._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector2i64 operator/(MR.Vector2i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_div_MR_Vector2i64_int64_t(MR.Vector2i64 b, long a);
            return __MR_div_MR_Vector2i64_int64_t(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2i64 AddAssign(MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_add_assign_MR_Vector2i64(MR.Vector2i64 *a, MR.Const_Vector2i64._Underlying *b);
            fixed (MR.Vector2i64 *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector2i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2i64 SubAssign(MR.Const_Vector2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2i64", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_sub_assign_MR_Vector2i64(MR.Vector2i64 *a, MR.Const_Vector2i64._Underlying *b);
            fixed (MR.Vector2i64 *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector2i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_mul_assign_MR_Vector2i64_int64_t(MR.Vector2i64 *a, long b);
            fixed (MR.Vector2i64 *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector2i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_div_assign_MR_Vector2i64_int64_t(MR.Vector2i64 *a, long b);
            fixed (MR.Vector2i64 *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector2i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector2i64 b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector2i64)
                return this == (MR.Vector2i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector2i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector2i64`/`Const_Vector2i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector2i64
    {
        public readonly bool HasValue;
        internal readonly Vector2i64 Object;
        public Vector2i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector2i64() {HasValue = false;}
        public _InOpt_Vector2i64(Vector2i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector2i64(Vector2i64 new_value) {return new(new_value);}
        public _InOpt_Vector2i64(Const_Vector2i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector2i64(Const_Vector2i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector2i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2i64`/`Const_Vector2i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2i64`.
    public class _InOptMut_Vector2i64
    {
        public Mut_Vector2i64? Opt;

        public _InOptMut_Vector2i64() {}
        public _InOptMut_Vector2i64(Mut_Vector2i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Vector2i64(Mut_Vector2i64 value) {return new(value);}
        public unsafe _InOptMut_Vector2i64(ref Vector2i64 value)
        {
            fixed (Vector2i64 *value_ptr = &value)
            {
                Opt = new((Const_Vector2i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector2i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2i64`/`Const_Vector2i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2i64`.
    public class _InOptConst_Vector2i64
    {
        public Const_Vector2i64? Opt;

        public _InOptConst_Vector2i64() {}
        public _InOptConst_Vector2i64(Const_Vector2i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Vector2i64(Const_Vector2i64 value) {return new(value);}
        public unsafe _InOptConst_Vector2i64(ref readonly Vector2i64 value)
        {
            fixed (Vector2i64 *value_ptr = &value)
            {
                Opt = new((Const_Vector2i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2f`.
    /// This is the const reference to the struct.
    public class Const_Vector2f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector2f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector2f UnderlyingStruct => ref *(Vector2f *)_UnderlyingPtr;

        internal unsafe Const_Vector2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector2f_Destroy(_Underlying *_this);
            __MR_Vector2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector2f() {Dispose(false);}

        public ref readonly float X => ref UnderlyingStruct.X;

        public ref readonly float Y => ref UnderlyingStruct.Y;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector2f_Get_elements();
                return *__MR_Vector2f_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector2f(Const_Vector2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2f _ctor_result = __MR_Vector2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2f::Vector2f`.
        public unsafe Const_Vector2f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2f _ctor_result = __MR_Vector2f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2f::Vector2f`.
        public unsafe Const_Vector2f(float x, float y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_Construct_2(float x, float y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2f _ctor_result = __MR_Vector2f_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::Vector2f::diagonal`.
        public static MR.Vector2f Diagonal(float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_diagonal", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_diagonal(float a);
            return __MR_Vector2f_diagonal(a);
        }

        /// Generated from method `MR::Vector2f::plusX`.
        public static MR.Vector2f PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_plusX", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_plusX();
            return __MR_Vector2f_plusX();
        }

        /// Generated from method `MR::Vector2f::plusY`.
        public static MR.Vector2f PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_plusY", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_plusY();
            return __MR_Vector2f_plusY();
        }

        /// Generated from method `MR::Vector2f::minusX`.
        public static MR.Vector2f MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_minusX", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_minusX();
            return __MR_Vector2f_minusX();
        }

        /// Generated from method `MR::Vector2f::minusY`.
        public static MR.Vector2f MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_minusY", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_minusY();
            return __MR_Vector2f_minusY();
        }

        /// Generated from method `MR::Vector2f::operator[]`.
        public unsafe float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_index_const", ExactSpelling = true)]
            extern static float *__MR_Vector2f_index_const(_Underlying *_this, int e);
            return *__MR_Vector2f_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector2f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_lengthSq", ExactSpelling = true)]
            extern static float __MR_Vector2f_lengthSq(_Underlying *_this);
            return __MR_Vector2f_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_length", ExactSpelling = true)]
            extern static float __MR_Vector2f_length(_Underlying *_this);
            return __MR_Vector2f_length(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2f::normalized`.
        public unsafe MR.Vector2f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_normalized", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_normalized(_Underlying *_this);
            return __MR_Vector2f_normalized(_UnderlyingPtr);
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2f::furthestBasisVector`.
        public unsafe MR.Vector2f FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_furthestBasisVector(_Underlying *_this);
            return __MR_Vector2f_furthestBasisVector(_UnderlyingPtr);
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2f::perpendicular`.
        public unsafe MR.Vector2f Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_perpendicular(_Underlying *_this);
            return __MR_Vector2f_perpendicular(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2f::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector2f_isFinite(_Underlying *_this);
            return __MR_Vector2f_isFinite(_UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_equal_MR_Vector2f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector2f a, MR.Const_Vector2f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2f operator+(MR.Const_Vector2f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Const_Vector2f._Underlying *__MR_pos_MR_Vector2f(MR.Const_Vector2f._Underlying *a);
            return new(__MR_pos_MR_Vector2f(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2f operator-(MR.Const_Vector2f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_neg_MR_Vector2f(MR.Const_Vector2f._Underlying *a);
            return __MR_neg_MR_Vector2f(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2f operator+(MR.Const_Vector2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_add_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_add_MR_Vector2f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2f operator-(MR.Const_Vector2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_sub_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_sub_MR_Vector2f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2f operator*(float a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_mul_float_MR_Vector2f(float a, MR.Const_Vector2f._Underlying *b);
            return __MR_mul_float_MR_Vector2f(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2f operator*(MR.Const_Vector2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Vector2f __MR_mul_MR_Vector2f_float(MR.Const_Vector2f._Underlying *b, float a);
            return __MR_mul_MR_Vector2f_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector2f operator/(Const_Vector2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Vector2f __MR_div_MR_Vector2f_float(MR.Vector2f b, float a);
            return __MR_div_MR_Vector2f_float(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector2f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector2f)
                return this == (MR.Const_Vector2f)other;
            return false;
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2f`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector2f : Const_Vector2f
    {
        /// Get the underlying struct.
        public unsafe new ref Vector2f UnderlyingStruct => ref *(Vector2f *)_UnderlyingPtr;

        internal unsafe Mut_Vector2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref float X => ref UnderlyingStruct.X;

        public new ref float Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Vector2f(Const_Vector2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2f _ctor_result = __MR_Vector2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2f::Vector2f`.
        public unsafe Mut_Vector2f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2f _ctor_result = __MR_Vector2f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Vector2f::Vector2f`.
        public unsafe Mut_Vector2f(float x, float y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_Construct_2(float x, float y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Vector2f _ctor_result = __MR_Vector2f_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::Vector2f::operator[]`.
        public unsafe new ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_index", ExactSpelling = true)]
            extern static float *__MR_Vector2f_index(_Underlying *_this, int e);
            return ref *__MR_Vector2f_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2f AddAssign(MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_add_assign_MR_Vector2f(_Underlying *a, MR.Const_Vector2f._Underlying *b);
            return new(__MR_add_assign_MR_Vector2f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2f SubAssign(MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_sub_assign_MR_Vector2f(_Underlying *a, MR.Const_Vector2f._Underlying *b);
            return new(__MR_sub_assign_MR_Vector2f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_mul_assign_MR_Vector2f_float(_Underlying *a, float b);
            return new(__MR_mul_assign_MR_Vector2f_float(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_div_assign_MR_Vector2f_float(_Underlying *a, float b);
            return new(__MR_div_assign_MR_Vector2f_float(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
    public struct Vector2f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector2f(Const_Vector2f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector2f(Vector2f other) => new(new Mut_Vector2f((Mut_Vector2f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public float X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public float Y;

        /// Generated copy constructor.
        public Vector2f(Vector2f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector2f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_DefaultConstruct();
            this = __MR_Vector2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector2f::Vector2f`.
        public unsafe Vector2f(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector2f_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector2f::Vector2f`.
        public unsafe Vector2f(float x, float y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_Construct_2(float x, float y);
            this = __MR_Vector2f_Construct_2(x, y);
        }

        /// Generated from method `MR::Vector2f::diagonal`.
        public static MR.Vector2f Diagonal(float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_diagonal", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_diagonal(float a);
            return __MR_Vector2f_diagonal(a);
        }

        /// Generated from method `MR::Vector2f::plusX`.
        public static MR.Vector2f PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_plusX", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_plusX();
            return __MR_Vector2f_plusX();
        }

        /// Generated from method `MR::Vector2f::plusY`.
        public static MR.Vector2f PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_plusY", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_plusY();
            return __MR_Vector2f_plusY();
        }

        /// Generated from method `MR::Vector2f::minusX`.
        public static MR.Vector2f MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_minusX", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_minusX();
            return __MR_Vector2f_minusX();
        }

        /// Generated from method `MR::Vector2f::minusY`.
        public static MR.Vector2f MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_minusY", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_minusY();
            return __MR_Vector2f_minusY();
        }

        /// Generated from method `MR::Vector2f::operator[]`.
        public unsafe float Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_index_const", ExactSpelling = true)]
            extern static float *__MR_Vector2f_index_const(MR.Vector2f *_this, int e);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return *__MR_Vector2f_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2f::operator[]`.
        public unsafe ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_index", ExactSpelling = true)]
            extern static float *__MR_Vector2f_index(MR.Vector2f *_this, int e);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return ref *__MR_Vector2f_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_lengthSq", ExactSpelling = true)]
            extern static float __MR_Vector2f_lengthSq(MR.Vector2f *_this);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return __MR_Vector2f_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_length", ExactSpelling = true)]
            extern static float __MR_Vector2f_length(MR.Vector2f *_this);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return __MR_Vector2f_length(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2f::normalized`.
        public unsafe MR.Vector2f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_normalized", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_normalized(MR.Vector2f *_this);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return __MR_Vector2f_normalized(__ptr__this);
            }
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2f::furthestBasisVector`.
        public unsafe MR.Vector2f FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_furthestBasisVector(MR.Vector2f *_this);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return __MR_Vector2f_furthestBasisVector(__ptr__this);
            }
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2f::perpendicular`.
        public unsafe MR.Vector2f Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Vector2f_perpendicular(MR.Vector2f *_this);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return __MR_Vector2f_perpendicular(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2f::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2f_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector2f_isFinite(MR.Vector2f *_this);
            fixed (MR.Vector2f *__ptr__this = &this)
            {
                return __MR_Vector2f_isFinite(__ptr__this) != 0;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector2f a, MR.Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_equal_MR_Vector2f((MR.Mut_Vector2f._Underlying *)&a, (MR.Mut_Vector2f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector2f a, MR.Vector2f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2f operator+(MR.Vector2f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Const_Vector2f._Underlying *__MR_pos_MR_Vector2f(MR.Const_Vector2f._Underlying *a);
            return new(__MR_pos_MR_Vector2f((MR.Mut_Vector2f._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2f operator-(MR.Vector2f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_neg_MR_Vector2f(MR.Const_Vector2f._Underlying *a);
            return __MR_neg_MR_Vector2f((MR.Mut_Vector2f._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2f operator+(MR.Vector2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_add_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_add_MR_Vector2f((MR.Mut_Vector2f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2f operator-(MR.Vector2f a, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_sub_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            return __MR_sub_MR_Vector2f((MR.Mut_Vector2f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2f operator*(float a, MR.Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Vector2f __MR_mul_float_MR_Vector2f(float a, MR.Const_Vector2f._Underlying *b);
            return __MR_mul_float_MR_Vector2f(a, (MR.Mut_Vector2f._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2f operator*(MR.Vector2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Vector2f __MR_mul_MR_Vector2f_float(MR.Const_Vector2f._Underlying *b, float a);
            return __MR_mul_MR_Vector2f_float((MR.Mut_Vector2f._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector2f operator/(MR.Vector2f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Vector2f __MR_div_MR_Vector2f_float(MR.Vector2f b, float a);
            return __MR_div_MR_Vector2f_float(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2f AddAssign(MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_add_assign_MR_Vector2f(MR.Vector2f *a, MR.Const_Vector2f._Underlying *b);
            fixed (MR.Vector2f *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector2f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2f SubAssign(MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_sub_assign_MR_Vector2f(MR.Vector2f *a, MR.Const_Vector2f._Underlying *b);
            fixed (MR.Vector2f *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector2f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_mul_assign_MR_Vector2f_float(MR.Vector2f *a, float b);
            fixed (MR.Vector2f *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector2f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_div_assign_MR_Vector2f_float(MR.Vector2f *a, float b);
            fixed (MR.Vector2f *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector2f_float(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector2f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector2f)
                return this == (MR.Vector2f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector2f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector2f`/`Const_Vector2f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector2f
    {
        public readonly bool HasValue;
        internal readonly Vector2f Object;
        public Vector2f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector2f() {HasValue = false;}
        public _InOpt_Vector2f(Vector2f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector2f(Vector2f new_value) {return new(new_value);}
        public _InOpt_Vector2f(Const_Vector2f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector2f(Const_Vector2f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2f`/`Const_Vector2f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2f`.
    public class _InOptMut_Vector2f
    {
        public Mut_Vector2f? Opt;

        public _InOptMut_Vector2f() {}
        public _InOptMut_Vector2f(Mut_Vector2f value) {Opt = value;}
        public static implicit operator _InOptMut_Vector2f(Mut_Vector2f value) {return new(value);}
        public unsafe _InOptMut_Vector2f(ref Vector2f value)
        {
            fixed (Vector2f *value_ptr = &value)
            {
                Opt = new((Const_Vector2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2f`/`Const_Vector2f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2f`.
    public class _InOptConst_Vector2f
    {
        public Const_Vector2f? Opt;

        public _InOptConst_Vector2f() {}
        public _InOptConst_Vector2f(Const_Vector2f value) {Opt = value;}
        public static implicit operator _InOptConst_Vector2f(Const_Vector2f value) {return new(value);}
        public unsafe _InOptConst_Vector2f(ref readonly Vector2f value)
        {
            fixed (Vector2f *value_ptr = &value)
            {
                Opt = new((Const_Vector2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2d`.
    /// This is the const reference to the struct.
    public class Const_Vector2d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector2d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector2d UnderlyingStruct => ref *(Vector2d *)_UnderlyingPtr;

        internal unsafe Const_Vector2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector2d_Destroy(_Underlying *_this);
            __MR_Vector2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector2d() {Dispose(false);}

        public ref readonly double X => ref UnderlyingStruct.X;

        public ref readonly double Y => ref UnderlyingStruct.Y;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector2d_Get_elements();
                return *__MR_Vector2d_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector2d(Const_Vector2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2d _ctor_result = __MR_Vector2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2d::Vector2d`.
        public unsafe Const_Vector2d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2d _ctor_result = __MR_Vector2d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2d::Vector2d`.
        public unsafe Const_Vector2d(double x, double y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_Construct_2(double x, double y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2d _ctor_result = __MR_Vector2d_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector2d::diagonal`.
        public static MR.Vector2d Diagonal(double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_diagonal", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_diagonal(double a);
            return __MR_Vector2d_diagonal(a);
        }

        /// Generated from method `MR::Vector2d::plusX`.
        public static MR.Vector2d PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_plusX", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_plusX();
            return __MR_Vector2d_plusX();
        }

        /// Generated from method `MR::Vector2d::plusY`.
        public static MR.Vector2d PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_plusY", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_plusY();
            return __MR_Vector2d_plusY();
        }

        /// Generated from method `MR::Vector2d::minusX`.
        public static MR.Vector2d MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_minusX", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_minusX();
            return __MR_Vector2d_minusX();
        }

        /// Generated from method `MR::Vector2d::minusY`.
        public static MR.Vector2d MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_minusY", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_minusY();
            return __MR_Vector2d_minusY();
        }

        /// Generated from method `MR::Vector2d::operator[]`.
        public unsafe double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_index_const", ExactSpelling = true)]
            extern static double *__MR_Vector2d_index_const(_Underlying *_this, int e);
            return *__MR_Vector2d_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector2d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_lengthSq", ExactSpelling = true)]
            extern static double __MR_Vector2d_lengthSq(_Underlying *_this);
            return __MR_Vector2d_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_length", ExactSpelling = true)]
            extern static double __MR_Vector2d_length(_Underlying *_this);
            return __MR_Vector2d_length(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2d::normalized`.
        public unsafe MR.Vector2d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_normalized", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_normalized(_Underlying *_this);
            return __MR_Vector2d_normalized(_UnderlyingPtr);
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2d::furthestBasisVector`.
        public unsafe MR.Vector2d FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_furthestBasisVector(_Underlying *_this);
            return __MR_Vector2d_furthestBasisVector(_UnderlyingPtr);
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2d::perpendicular`.
        public unsafe MR.Vector2d Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_perpendicular(_Underlying *_this);
            return __MR_Vector2d_perpendicular(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector2d::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector2d_isFinite(_Underlying *_this);
            return __MR_Vector2d_isFinite(_UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_equal_MR_Vector2d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector2d a, MR.Const_Vector2d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2d operator+(MR.Const_Vector2d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Const_Vector2d._Underlying *__MR_pos_MR_Vector2d(MR.Const_Vector2d._Underlying *a);
            return new(__MR_pos_MR_Vector2d(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2d operator-(MR.Const_Vector2d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_neg_MR_Vector2d(MR.Const_Vector2d._Underlying *a);
            return __MR_neg_MR_Vector2d(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2d operator+(MR.Const_Vector2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_add_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_add_MR_Vector2d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2d operator-(MR.Const_Vector2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_sub_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_sub_MR_Vector2d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2d operator*(double a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_mul_double_MR_Vector2d(double a, MR.Const_Vector2d._Underlying *b);
            return __MR_mul_double_MR_Vector2d(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2d operator*(MR.Const_Vector2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Vector2d __MR_mul_MR_Vector2d_double(MR.Const_Vector2d._Underlying *b, double a);
            return __MR_mul_MR_Vector2d_double(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector2d operator/(Const_Vector2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Vector2d __MR_div_MR_Vector2d_double(MR.Vector2d b, double a);
            return __MR_div_MR_Vector2d_double(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector2d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector2d)
                return this == (MR.Const_Vector2d)other;
            return false;
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2d`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector2d : Const_Vector2d
    {
        /// Get the underlying struct.
        public unsafe new ref Vector2d UnderlyingStruct => ref *(Vector2d *)_UnderlyingPtr;

        internal unsafe Mut_Vector2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref double X => ref UnderlyingStruct.X;

        public new ref double Y => ref UnderlyingStruct.Y;

        /// Generated copy constructor.
        public unsafe Mut_Vector2d(Const_Vector2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2d _ctor_result = __MR_Vector2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2d::Vector2d`.
        public unsafe Mut_Vector2d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2d _ctor_result = __MR_Vector2d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector2d::Vector2d`.
        public unsafe Mut_Vector2d(double x, double y) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_Construct_2(double x, double y);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector2d _ctor_result = __MR_Vector2d_Construct_2(x, y);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector2d::operator[]`.
        public unsafe new ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_index", ExactSpelling = true)]
            extern static double *__MR_Vector2d_index(_Underlying *_this, int e);
            return ref *__MR_Vector2d_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2d AddAssign(MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_add_assign_MR_Vector2d(_Underlying *a, MR.Const_Vector2d._Underlying *b);
            return new(__MR_add_assign_MR_Vector2d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2d SubAssign(MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_sub_assign_MR_Vector2d(_Underlying *a, MR.Const_Vector2d._Underlying *b);
            return new(__MR_sub_assign_MR_Vector2d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_mul_assign_MR_Vector2d_double(_Underlying *a, double b);
            return new(__MR_mul_assign_MR_Vector2d_double(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_div_assign_MR_Vector2d_double(_Underlying *a, double b);
            return new(__MR_div_assign_MR_Vector2d_double(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// two-dimensional vector
    /// Generated from class `MR::Vector2d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Vector2d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector2d(Const_Vector2d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector2d(Vector2d other) => new(new Mut_Vector2d((Mut_Vector2d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public double X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public double Y;

        /// Generated copy constructor.
        public Vector2d(Vector2d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector2d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_DefaultConstruct();
            this = __MR_Vector2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector2d::Vector2d`.
        public unsafe Vector2d(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector2d_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector2d::Vector2d`.
        public unsafe Vector2d(double x, double y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_Construct_2", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_Construct_2(double x, double y);
            this = __MR_Vector2d_Construct_2(x, y);
        }

        /// Generated from method `MR::Vector2d::diagonal`.
        public static MR.Vector2d Diagonal(double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_diagonal", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_diagonal(double a);
            return __MR_Vector2d_diagonal(a);
        }

        /// Generated from method `MR::Vector2d::plusX`.
        public static MR.Vector2d PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_plusX", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_plusX();
            return __MR_Vector2d_plusX();
        }

        /// Generated from method `MR::Vector2d::plusY`.
        public static MR.Vector2d PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_plusY", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_plusY();
            return __MR_Vector2d_plusY();
        }

        /// Generated from method `MR::Vector2d::minusX`.
        public static MR.Vector2d MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_minusX", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_minusX();
            return __MR_Vector2d_minusX();
        }

        /// Generated from method `MR::Vector2d::minusY`.
        public static MR.Vector2d MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_minusY", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_minusY();
            return __MR_Vector2d_minusY();
        }

        /// Generated from method `MR::Vector2d::operator[]`.
        public unsafe double Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_index_const", ExactSpelling = true)]
            extern static double *__MR_Vector2d_index_const(MR.Vector2d *_this, int e);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return *__MR_Vector2d_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2d::operator[]`.
        public unsafe ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_index", ExactSpelling = true)]
            extern static double *__MR_Vector2d_index(MR.Vector2d *_this, int e);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return ref *__MR_Vector2d_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector2d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_lengthSq", ExactSpelling = true)]
            extern static double __MR_Vector2d_lengthSq(MR.Vector2d *_this);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return __MR_Vector2d_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_length", ExactSpelling = true)]
            extern static double __MR_Vector2d_length(MR.Vector2d *_this);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return __MR_Vector2d_length(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2d::normalized`.
        public unsafe MR.Vector2d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_normalized", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_normalized(MR.Vector2d *_this);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return __MR_Vector2d_normalized(__ptr__this);
            }
        }

        /// returns one of 2 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector2d::furthestBasisVector`.
        public unsafe MR.Vector2d FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_furthestBasisVector(MR.Vector2d *_this);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return __MR_Vector2d_furthestBasisVector(__ptr__this);
            }
        }

        /// returns same length vector orthogonal to this (rotated 90 degrees counter-clockwise)
        /// Generated from method `MR::Vector2d::perpendicular`.
        public unsafe MR.Vector2d Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_perpendicular", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Vector2d_perpendicular(MR.Vector2d *_this);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return __MR_Vector2d_perpendicular(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector2d::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector2d_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector2d_isFinite(MR.Vector2d *_this);
            fixed (MR.Vector2d *__ptr__this = &this)
            {
                return __MR_Vector2d_isFinite(__ptr__this) != 0;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector2d a, MR.Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_equal_MR_Vector2d((MR.Mut_Vector2d._Underlying *)&a, (MR.Mut_Vector2d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector2d a, MR.Vector2d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector2d operator+(MR.Vector2d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Const_Vector2d._Underlying *__MR_pos_MR_Vector2d(MR.Const_Vector2d._Underlying *a);
            return new(__MR_pos_MR_Vector2d((MR.Mut_Vector2d._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2d operator-(MR.Vector2d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_neg_MR_Vector2d(MR.Const_Vector2d._Underlying *a);
            return __MR_neg_MR_Vector2d((MR.Mut_Vector2d._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector2d operator+(MR.Vector2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_add_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_add_MR_Vector2d((MR.Mut_Vector2d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector2d operator-(MR.Vector2d a, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_sub_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            return __MR_sub_MR_Vector2d((MR.Mut_Vector2d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2d operator*(double a, MR.Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Vector2d __MR_mul_double_MR_Vector2d(double a, MR.Const_Vector2d._Underlying *b);
            return __MR_mul_double_MR_Vector2d(a, (MR.Mut_Vector2d._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector2d operator*(MR.Vector2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Vector2d __MR_mul_MR_Vector2d_double(MR.Const_Vector2d._Underlying *b, double a);
            return __MR_mul_MR_Vector2d_double((MR.Mut_Vector2d._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector2d operator/(MR.Vector2d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Vector2d __MR_div_MR_Vector2d_double(MR.Vector2d b, double a);
            return __MR_div_MR_Vector2d_double(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector2d AddAssign(MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_add_assign_MR_Vector2d(MR.Vector2d *a, MR.Const_Vector2d._Underlying *b);
            fixed (MR.Vector2d *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector2d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector2d SubAssign(MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_sub_assign_MR_Vector2d(MR.Vector2d *a, MR.Const_Vector2d._Underlying *b);
            fixed (MR.Vector2d *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector2d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector2d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_mul_assign_MR_Vector2d_double(MR.Vector2d *a, double b);
            fixed (MR.Vector2d *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector2d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector2d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector2d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_div_assign_MR_Vector2d_double(MR.Vector2d *a, double b);
            fixed (MR.Vector2d *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector2d_double(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector2d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector2d)
                return this == (MR.Vector2d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector2d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector2d`/`Const_Vector2d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector2d
    {
        public readonly bool HasValue;
        internal readonly Vector2d Object;
        public Vector2d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector2d() {HasValue = false;}
        public _InOpt_Vector2d(Vector2d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector2d(Vector2d new_value) {return new(new_value);}
        public _InOpt_Vector2d(Const_Vector2d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector2d(Const_Vector2d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2d`/`Const_Vector2d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2d`.
    public class _InOptMut_Vector2d
    {
        public Mut_Vector2d? Opt;

        public _InOptMut_Vector2d() {}
        public _InOptMut_Vector2d(Mut_Vector2d value) {Opt = value;}
        public static implicit operator _InOptMut_Vector2d(Mut_Vector2d value) {return new(value);}
        public unsafe _InOptMut_Vector2d(ref Vector2d value)
        {
            fixed (Vector2d *value_ptr = &value)
            {
                Opt = new((Const_Vector2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector2d`/`Const_Vector2d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector2d`.
    public class _InOptConst_Vector2d
    {
        public Const_Vector2d? Opt;

        public _InOptConst_Vector2d() {}
        public _InOptConst_Vector2d(Const_Vector2d value) {Opt = value;}
        public static implicit operator _InOptConst_Vector2d(Const_Vector2d value) {return new(value);}
        public unsafe _InOptConst_Vector2d(ref readonly Vector2d value)
        {
            fixed (Vector2d *value_ptr = &value)
            {
                Opt = new((Const_Vector2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// cross product
    /// Generated from function `MR::cross<float>`.
    public static unsafe float Cross(MR.Const_Vector2f a, MR.Const_Vector2f b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cross", ExactSpelling = true)]
        extern static float __MR_cross(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
        return __MR_cross(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<bool>`.
    public static unsafe int Dot(MR.Const_Vector2b a, MR.Const_Vector2b b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_bool_MR_Vector2b", ExactSpelling = true)]
        extern static int __MR_dot_bool_MR_Vector2b(MR.Const_Vector2b._Underlying *a, MR.Const_Vector2b._Underlying *b);
        return __MR_dot_bool_MR_Vector2b(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<int>`.
    public static unsafe int Dot(MR.Const_Vector2i a, MR.Const_Vector2i b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_int_MR_Vector2i", ExactSpelling = true)]
        extern static int __MR_dot_int_MR_Vector2i(MR.Const_Vector2i._Underlying *a, MR.Const_Vector2i._Underlying *b);
        return __MR_dot_int_MR_Vector2i(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<MR_int64_t>`.
    public static unsafe long Dot(MR.Const_Vector2i64 a, MR.Const_Vector2i64 b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_int64_t_MR_Vector2i64", ExactSpelling = true)]
        extern static long __MR_dot_int64_t_MR_Vector2i64(MR.Const_Vector2i64._Underlying *a, MR.Const_Vector2i64._Underlying *b);
        return __MR_dot_int64_t_MR_Vector2i64(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<float>`.
    public static unsafe float Dot(MR.Const_Vector2f a, MR.Const_Vector2f b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_float_MR_Vector2f", ExactSpelling = true)]
        extern static float __MR_dot_float_MR_Vector2f(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
        return __MR_dot_float_MR_Vector2f(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<double>`.
    public static unsafe double Dot(MR.Const_Vector2d a, MR.Const_Vector2d b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_double_MR_Vector2d", ExactSpelling = true)]
        extern static double __MR_dot_double_MR_Vector2d(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
        return __MR_dot_double_MR_Vector2d(a._UnderlyingPtr, b._UnderlyingPtr);
    }
}
