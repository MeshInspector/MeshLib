public static partial class MR
{
    /// three-dimensional vector
    /// Generated from class `MR::Vector3b`.
    /// This is the const reference to the struct.
    public class Const_Vector3b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector3b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector3b UnderlyingStruct => ref *(Vector3b *)_UnderlyingPtr;

        internal unsafe Const_Vector3b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector3b_Destroy(_Underlying *_this);
            __MR_Vector3b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector3b() {Dispose(false);}

        public bool X => UnderlyingStruct.X;

        public bool Y => UnderlyingStruct.Y;

        public bool Z => UnderlyingStruct.Z;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector3b_Get_elements();
                return *__MR_Vector3b_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector3b(Const_Vector3b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 3);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector3b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            MR.Vector3b _ctor_result = __MR_Vector3b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 3);
        }

        /// Generated from constructor `MR::Vector3b::Vector3b`.
        public unsafe Const_Vector3b(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            MR.Vector3b _ctor_result = __MR_Vector3b_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 3);
        }

        /// Generated from constructor `MR::Vector3b::Vector3b`.
        public unsafe Const_Vector3b(bool x, bool y, bool z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_Construct_3(byte x, byte y, byte z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            MR.Vector3b _ctor_result = __MR_Vector3b_Construct_3(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0, z ? (byte)1 : (byte)0);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 3);
        }

        /// Generated from method `MR::Vector3b::diagonal`.
        public static MR.Vector3b Diagonal(bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_diagonal", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_diagonal(byte a);
            return __MR_Vector3b_diagonal(a ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector3b::plusX`.
        public static MR.Vector3b PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_plusX", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_plusX();
            return __MR_Vector3b_plusX();
        }

        /// Generated from method `MR::Vector3b::plusY`.
        public static MR.Vector3b PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_plusY", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_plusY();
            return __MR_Vector3b_plusY();
        }

        /// Generated from method `MR::Vector3b::plusZ`.
        public static MR.Vector3b PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_plusZ", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_plusZ();
            return __MR_Vector3b_plusZ();
        }

        /// Generated from method `MR::Vector3b::operator[]`.
        public unsafe bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_index_const", ExactSpelling = true)]
            extern static bool *__MR_Vector3b_index_const(_Underlying *_this, int e);
            return *__MR_Vector3b_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector3b::lengthSq`.
        public unsafe bool LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector3b_lengthSq(_Underlying *_this);
            return __MR_Vector3b_lengthSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Vector3b::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_length", ExactSpelling = true)]
            extern static double __MR_Vector3b_length(_Underlying *_this);
            return __MR_Vector3b_length(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_equal_MR_Vector3b(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector3b a, MR.Const_Vector3b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3b operator+(MR.Const_Vector3b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Const_Vector3b._Underlying *__MR_pos_MR_Vector3b(MR.Const_Vector3b._Underlying *a);
            return new(__MR_pos_MR_Vector3b(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Const_Vector3b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_neg_MR_Vector3b(MR.Const_Vector3b._Underlying *a);
            return __MR_neg_MR_Vector3b(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i operator+(MR.Const_Vector3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_add_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_add_MR_Vector3b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Const_Vector3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_sub_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_sub_MR_Vector3b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(bool a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_bool_MR_Vector3b(byte a, MR.Const_Vector3b._Underlying *b);
            return __MR_mul_bool_MR_Vector3b(a ? (byte)1 : (byte)0, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Const_Vector3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Vector3b_bool(MR.Const_Vector3b._Underlying *b, byte a);
            return __MR_mul_MR_Vector3b_bool(b._UnderlyingPtr, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector3i operator/(Const_Vector3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Vector3i __MR_div_MR_Vector3b_bool(MR.Vector3b b, byte a);
            return __MR_div_MR_Vector3b_bool(b.UnderlyingStruct, a ? (byte)1 : (byte)0);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector3b? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector3b)
                return this == (MR.Const_Vector3b)other;
            return false;
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3b`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector3b : Const_Vector3b
    {
        /// Get the underlying struct.
        public unsafe new ref Vector3b UnderlyingStruct => ref *(Vector3b *)_UnderlyingPtr;

        internal unsafe Mut_Vector3b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new bool X {get => UnderlyingStruct.X; set => UnderlyingStruct.X = value;}

        public new bool Y {get => UnderlyingStruct.Y; set => UnderlyingStruct.Y = value;}

        public new bool Z {get => UnderlyingStruct.Z; set => UnderlyingStruct.Z = value;}

        /// Generated copy constructor.
        public unsafe Mut_Vector3b(Const_Vector3b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 3);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector3b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            MR.Vector3b _ctor_result = __MR_Vector3b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 3);
        }

        /// Generated from constructor `MR::Vector3b::Vector3b`.
        public unsafe Mut_Vector3b(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            MR.Vector3b _ctor_result = __MR_Vector3b_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 3);
        }

        /// Generated from constructor `MR::Vector3b::Vector3b`.
        public unsafe Mut_Vector3b(bool x, bool y, bool z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_Construct_3(byte x, byte y, byte z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(3);
            MR.Vector3b _ctor_result = __MR_Vector3b_Construct_3(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0, z ? (byte)1 : (byte)0);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 3);
        }

        /// Generated from method `MR::Vector3b::operator[]`.
        public unsafe new ref bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_index", ExactSpelling = true)]
            extern static bool *__MR_Vector3b_index(_Underlying *_this, int e);
            return ref *__MR_Vector3b_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3b AddAssign(MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_add_assign_MR_Vector3b(_Underlying *a, MR.Const_Vector3b._Underlying *b);
            return new(__MR_add_assign_MR_Vector3b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3b SubAssign(MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_sub_assign_MR_Vector3b(_Underlying *a, MR.Const_Vector3b._Underlying *b);
            return new(__MR_sub_assign_MR_Vector3b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_mul_assign_MR_Vector3b_bool(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Vector3b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_div_assign_MR_Vector3b_bool(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Vector3b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3b`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 3)]
    public struct Vector3b
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector3b(Const_Vector3b other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector3b(Vector3b other) => new(new Mut_Vector3b((Mut_Vector3b._Underlying *)&other, is_owning: false));

        public bool X {get => __storage_X != 0; set => __storage_X = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(0)]
        byte __storage_X;

        public bool Y {get => __storage_Y != 0; set => __storage_Y = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(1)]
        byte __storage_Y;

        public bool Z {get => __storage_Z != 0; set => __storage_Z = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(2)]
        byte __storage_Z;

        /// Generated copy constructor.
        public Vector3b(Vector3b _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector3b()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_DefaultConstruct();
            this = __MR_Vector3b_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3b::Vector3b`.
        public unsafe Vector3b(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector3b_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3b::Vector3b`.
        public unsafe Vector3b(bool x, bool y, bool z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_Construct_3(byte x, byte y, byte z);
            this = __MR_Vector3b_Construct_3(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0, z ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector3b::diagonal`.
        public static MR.Vector3b Diagonal(bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_diagonal", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_diagonal(byte a);
            return __MR_Vector3b_diagonal(a ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector3b::plusX`.
        public static MR.Vector3b PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_plusX", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_plusX();
            return __MR_Vector3b_plusX();
        }

        /// Generated from method `MR::Vector3b::plusY`.
        public static MR.Vector3b PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_plusY", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_plusY();
            return __MR_Vector3b_plusY();
        }

        /// Generated from method `MR::Vector3b::plusZ`.
        public static MR.Vector3b PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_plusZ", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Vector3b_plusZ();
            return __MR_Vector3b_plusZ();
        }

        /// Generated from method `MR::Vector3b::operator[]`.
        public unsafe bool Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_index_const", ExactSpelling = true)]
            extern static bool *__MR_Vector3b_index_const(MR.Vector3b *_this, int e);
            fixed (MR.Vector3b *__ptr__this = &this)
            {
                return *__MR_Vector3b_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3b::operator[]`.
        public unsafe ref bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_index", ExactSpelling = true)]
            extern static bool *__MR_Vector3b_index(MR.Vector3b *_this, int e);
            fixed (MR.Vector3b *__ptr__this = &this)
            {
                return ref *__MR_Vector3b_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3b::lengthSq`.
        public unsafe bool LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector3b_lengthSq(MR.Vector3b *_this);
            fixed (MR.Vector3b *__ptr__this = &this)
            {
                return __MR_Vector3b_lengthSq(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::Vector3b::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3b_length", ExactSpelling = true)]
            extern static double __MR_Vector3b_length(MR.Vector3b *_this);
            fixed (MR.Vector3b *__ptr__this = &this)
            {
                return __MR_Vector3b_length(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector3b a, MR.Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_equal_MR_Vector3b((MR.Mut_Vector3b._Underlying *)&a, (MR.Mut_Vector3b._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector3b a, MR.Vector3b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3b operator+(MR.Vector3b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Const_Vector3b._Underlying *__MR_pos_MR_Vector3b(MR.Const_Vector3b._Underlying *a);
            return new(__MR_pos_MR_Vector3b((MR.Mut_Vector3b._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Vector3b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_neg_MR_Vector3b(MR.Const_Vector3b._Underlying *a);
            return __MR_neg_MR_Vector3b((MR.Mut_Vector3b._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i operator+(MR.Vector3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_add_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_add_MR_Vector3b((MR.Mut_Vector3b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Vector3b a, MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_sub_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
            return __MR_sub_MR_Vector3b((MR.Mut_Vector3b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(bool a, MR.Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_bool_MR_Vector3b(byte a, MR.Const_Vector3b._Underlying *b);
            return __MR_mul_bool_MR_Vector3b(a ? (byte)1 : (byte)0, (MR.Mut_Vector3b._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Vector3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Vector3b_bool(MR.Const_Vector3b._Underlying *b, byte a);
            return __MR_mul_MR_Vector3b_bool((MR.Mut_Vector3b._Underlying *)&b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector3i operator/(MR.Vector3b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Vector3i __MR_div_MR_Vector3b_bool(MR.Vector3b b, byte a);
            return __MR_div_MR_Vector3b_bool(b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3b AddAssign(MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_add_assign_MR_Vector3b(MR.Vector3b *a, MR.Const_Vector3b._Underlying *b);
            fixed (MR.Vector3b *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector3b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3b SubAssign(MR.Const_Vector3b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3b", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_sub_assign_MR_Vector3b(MR.Vector3b *a, MR.Const_Vector3b._Underlying *b);
            fixed (MR.Vector3b *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector3b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_mul_assign_MR_Vector3b_bool(MR.Vector3b *a, byte b);
            fixed (MR.Vector3b *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector3b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector3b._Underlying *__MR_div_assign_MR_Vector3b_bool(MR.Vector3b *a, byte b);
            fixed (MR.Vector3b *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector3b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector3b b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector3b)
                return this == (MR.Vector3b)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector3b` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector3b`/`Const_Vector3b` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector3b
    {
        public readonly bool HasValue;
        internal readonly Vector3b Object;
        public Vector3b Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector3b() {HasValue = false;}
        public _InOpt_Vector3b(Vector3b new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector3b(Vector3b new_value) {return new(new_value);}
        public _InOpt_Vector3b(Const_Vector3b new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector3b(Const_Vector3b new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector3b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector3b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3b`/`Const_Vector3b` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3b`.
    public class _InOptMut_Vector3b
    {
        public Mut_Vector3b? Opt;

        public _InOptMut_Vector3b() {}
        public _InOptMut_Vector3b(Mut_Vector3b value) {Opt = value;}
        public static implicit operator _InOptMut_Vector3b(Mut_Vector3b value) {return new(value);}
        public unsafe _InOptMut_Vector3b(ref Vector3b value)
        {
            fixed (Vector3b *value_ptr = &value)
            {
                Opt = new((Const_Vector3b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector3b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector3b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3b`/`Const_Vector3b` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3b`.
    public class _InOptConst_Vector3b
    {
        public Const_Vector3b? Opt;

        public _InOptConst_Vector3b() {}
        public _InOptConst_Vector3b(Const_Vector3b value) {Opt = value;}
        public static implicit operator _InOptConst_Vector3b(Const_Vector3b value) {return new(value);}
        public unsafe _InOptConst_Vector3b(ref readonly Vector3b value)
        {
            fixed (Vector3b *value_ptr = &value)
            {
                Opt = new((Const_Vector3b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3i`.
    /// This is the const reference to the struct.
    public class Const_Vector3i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector3i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector3i UnderlyingStruct => ref *(Vector3i *)_UnderlyingPtr;

        internal unsafe Const_Vector3i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector3i_Destroy(_Underlying *_this);
            __MR_Vector3i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector3i() {Dispose(false);}

        public ref readonly int X => ref UnderlyingStruct.X;

        public ref readonly int Y => ref UnderlyingStruct.Y;

        public ref readonly int Z => ref UnderlyingStruct.Z;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector3i_Get_elements();
                return *__MR_Vector3i_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector3i(Const_Vector3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 12);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Const_Vector3i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Const_Vector3i(int x, int y, int z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_3(int x, int y, int z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Const_Vector3i(MR.Const_Vector3f v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_float", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_float(MR.Const_Vector3f._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_Construct_float(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from method `MR::Vector3i::diagonal`.
        public static MR.Vector3i Diagonal(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_diagonal", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_diagonal(int a);
            return __MR_Vector3i_diagonal(a);
        }

        /// Generated from method `MR::Vector3i::plusX`.
        public static MR.Vector3i PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_plusX", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_plusX();
            return __MR_Vector3i_plusX();
        }

        /// Generated from method `MR::Vector3i::plusY`.
        public static MR.Vector3i PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_plusY", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_plusY();
            return __MR_Vector3i_plusY();
        }

        /// Generated from method `MR::Vector3i::plusZ`.
        public static MR.Vector3i PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_plusZ", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_plusZ();
            return __MR_Vector3i_plusZ();
        }

        /// Generated from method `MR::Vector3i::minusX`.
        public static MR.Vector3i MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_minusX", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_minusX();
            return __MR_Vector3i_minusX();
        }

        /// Generated from method `MR::Vector3i::minusY`.
        public static MR.Vector3i MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_minusY", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_minusY();
            return __MR_Vector3i_minusY();
        }

        /// Generated from method `MR::Vector3i::minusZ`.
        public static MR.Vector3i MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_minusZ", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_minusZ();
            return __MR_Vector3i_minusZ();
        }

        /// Generated from method `MR::Vector3i::operator[]`.
        public unsafe int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_index_const", ExactSpelling = true)]
            extern static int *__MR_Vector3i_index_const(_Underlying *_this, int e);
            return *__MR_Vector3i_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector3i::lengthSq`.
        public unsafe int LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_lengthSq", ExactSpelling = true)]
            extern static int __MR_Vector3i_lengthSq(_Underlying *_this);
            return __MR_Vector3i_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3i::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_length", ExactSpelling = true)]
            extern static double __MR_Vector3i_length(_Underlying *_this);
            return __MR_Vector3i_length(_UnderlyingPtr);
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3i::furthestBasisVector`.
        public unsafe MR.Vector3i FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_furthestBasisVector(_Underlying *_this);
            return __MR_Vector3i_furthestBasisVector(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_equal_MR_Vector3i(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector3i a, MR.Const_Vector3i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3i operator+(MR.Const_Vector3i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_pos_MR_Vector3i(MR.Const_Vector3i._Underlying *a);
            return new(__MR_pos_MR_Vector3i(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Const_Vector3i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_neg_MR_Vector3i(MR.Const_Vector3i._Underlying *a);
            return __MR_neg_MR_Vector3i(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i operator+(MR.Const_Vector3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_add_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_add_MR_Vector3i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Const_Vector3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_sub_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_sub_MR_Vector3i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(int a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_int_MR_Vector3i(int a, MR.Const_Vector3i._Underlying *b);
            return __MR_mul_int_MR_Vector3i(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Const_Vector3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Vector3i_int(MR.Const_Vector3i._Underlying *b, int a);
            return __MR_mul_MR_Vector3i_int(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector3i operator/(Const_Vector3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Vector3i __MR_div_MR_Vector3i_int(MR.Vector3i b, int a);
            return __MR_div_MR_Vector3i_int(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector3i? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector3i)
                return this == (MR.Const_Vector3i)other;
            return false;
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3i`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector3i : Const_Vector3i
    {
        /// Get the underlying struct.
        public unsafe new ref Vector3i UnderlyingStruct => ref *(Vector3i *)_UnderlyingPtr;

        internal unsafe Mut_Vector3i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int X => ref UnderlyingStruct.X;

        public new ref int Y => ref UnderlyingStruct.Y;

        public new ref int Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Vector3i(Const_Vector3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 12);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Mut_Vector3i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Mut_Vector3i(int x, int y, int z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_3(int x, int y, int z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Mut_Vector3i(MR.Const_Vector3f v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_float", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_float(MR.Const_Vector3f._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3i _ctor_result = __MR_Vector3i_Construct_float(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from method `MR::Vector3i::operator[]`.
        public unsafe new ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_index", ExactSpelling = true)]
            extern static int *__MR_Vector3i_index(_Underlying *_this, int e);
            return ref *__MR_Vector3i_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3i AddAssign(MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_add_assign_MR_Vector3i(_Underlying *a, MR.Const_Vector3i._Underlying *b);
            return new(__MR_add_assign_MR_Vector3i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3i SubAssign(MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_sub_assign_MR_Vector3i(_Underlying *a, MR.Const_Vector3i._Underlying *b);
            return new(__MR_sub_assign_MR_Vector3i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_mul_assign_MR_Vector3i_int(_Underlying *a, int b);
            return new(__MR_mul_assign_MR_Vector3i_int(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_div_assign_MR_Vector3i_int(_Underlying *a, int b);
            return new(__MR_div_assign_MR_Vector3i_int(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 12)]
    public struct Vector3i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector3i(Const_Vector3i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector3i(Vector3i other) => new(new Mut_Vector3i((Mut_Vector3i._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public int Y;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public int Z;

        /// Generated copy constructor.
        public Vector3i(Vector3i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector3i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_DefaultConstruct();
            this = __MR_Vector3i_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Vector3i(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector3i_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Vector3i(int x, int y, int z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_3(int x, int y, int z);
            this = __MR_Vector3i_Construct_3(x, y, z);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3i::Vector3i`.
        public unsafe Vector3i(MR.Const_Vector3f v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_Construct_float", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_Construct_float(MR.Const_Vector3f._Underlying *v);
            this = __MR_Vector3i_Construct_float(v._UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3i::diagonal`.
        public static MR.Vector3i Diagonal(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_diagonal", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_diagonal(int a);
            return __MR_Vector3i_diagonal(a);
        }

        /// Generated from method `MR::Vector3i::plusX`.
        public static MR.Vector3i PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_plusX", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_plusX();
            return __MR_Vector3i_plusX();
        }

        /// Generated from method `MR::Vector3i::plusY`.
        public static MR.Vector3i PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_plusY", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_plusY();
            return __MR_Vector3i_plusY();
        }

        /// Generated from method `MR::Vector3i::plusZ`.
        public static MR.Vector3i PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_plusZ", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_plusZ();
            return __MR_Vector3i_plusZ();
        }

        /// Generated from method `MR::Vector3i::minusX`.
        public static MR.Vector3i MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_minusX", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_minusX();
            return __MR_Vector3i_minusX();
        }

        /// Generated from method `MR::Vector3i::minusY`.
        public static MR.Vector3i MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_minusY", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_minusY();
            return __MR_Vector3i_minusY();
        }

        /// Generated from method `MR::Vector3i::minusZ`.
        public static MR.Vector3i MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_minusZ", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_minusZ();
            return __MR_Vector3i_minusZ();
        }

        /// Generated from method `MR::Vector3i::operator[]`.
        public unsafe int Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_index_const", ExactSpelling = true)]
            extern static int *__MR_Vector3i_index_const(MR.Vector3i *_this, int e);
            fixed (MR.Vector3i *__ptr__this = &this)
            {
                return *__MR_Vector3i_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3i::operator[]`.
        public unsafe ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_index", ExactSpelling = true)]
            extern static int *__MR_Vector3i_index(MR.Vector3i *_this, int e);
            fixed (MR.Vector3i *__ptr__this = &this)
            {
                return ref *__MR_Vector3i_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3i::lengthSq`.
        public unsafe int LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_lengthSq", ExactSpelling = true)]
            extern static int __MR_Vector3i_lengthSq(MR.Vector3i *_this);
            fixed (MR.Vector3i *__ptr__this = &this)
            {
                return __MR_Vector3i_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3i::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_length", ExactSpelling = true)]
            extern static double __MR_Vector3i_length(MR.Vector3i *_this);
            fixed (MR.Vector3i *__ptr__this = &this)
            {
                return __MR_Vector3i_length(__ptr__this);
            }
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3i::furthestBasisVector`.
        public unsafe MR.Vector3i FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Vector3i_furthestBasisVector(MR.Vector3i *_this);
            fixed (MR.Vector3i *__ptr__this = &this)
            {
                return __MR_Vector3i_furthestBasisVector(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector3i a, MR.Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_equal_MR_Vector3i((MR.Mut_Vector3i._Underlying *)&a, (MR.Mut_Vector3i._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector3i a, MR.Vector3i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3i operator+(MR.Vector3i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_pos_MR_Vector3i(MR.Const_Vector3i._Underlying *a);
            return new(__MR_pos_MR_Vector3i((MR.Mut_Vector3i._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Vector3i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_neg_MR_Vector3i(MR.Const_Vector3i._Underlying *a);
            return __MR_neg_MR_Vector3i((MR.Mut_Vector3i._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i operator+(MR.Vector3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_add_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_add_MR_Vector3i((MR.Mut_Vector3i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Vector3i a, MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_sub_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
            return __MR_sub_MR_Vector3i((MR.Mut_Vector3i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(int a, MR.Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_int_MR_Vector3i(int a, MR.Const_Vector3i._Underlying *b);
            return __MR_mul_int_MR_Vector3i(a, (MR.Mut_Vector3i._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Vector3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Vector3i_int(MR.Const_Vector3i._Underlying *b, int a);
            return __MR_mul_MR_Vector3i_int((MR.Mut_Vector3i._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector3i operator/(MR.Vector3i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Vector3i __MR_div_MR_Vector3i_int(MR.Vector3i b, int a);
            return __MR_div_MR_Vector3i_int(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3i AddAssign(MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_add_assign_MR_Vector3i(MR.Vector3i *a, MR.Const_Vector3i._Underlying *b);
            fixed (MR.Vector3i *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector3i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3i SubAssign(MR.Const_Vector3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3i", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_sub_assign_MR_Vector3i(MR.Vector3i *a, MR.Const_Vector3i._Underlying *b);
            fixed (MR.Vector3i *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector3i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_mul_assign_MR_Vector3i_int(MR.Vector3i *a, int b);
            fixed (MR.Vector3i *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector3i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_div_assign_MR_Vector3i_int(MR.Vector3i *a, int b);
            fixed (MR.Vector3i *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector3i_int(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector3i b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector3i)
                return this == (MR.Vector3i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector3i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector3i`/`Const_Vector3i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector3i
    {
        public readonly bool HasValue;
        internal readonly Vector3i Object;
        public Vector3i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector3i() {HasValue = false;}
        public _InOpt_Vector3i(Vector3i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector3i(Vector3i new_value) {return new(new_value);}
        public _InOpt_Vector3i(Const_Vector3i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector3i(Const_Vector3i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector3i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3i`/`Const_Vector3i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3i`.
    public class _InOptMut_Vector3i
    {
        public Mut_Vector3i? Opt;

        public _InOptMut_Vector3i() {}
        public _InOptMut_Vector3i(Mut_Vector3i value) {Opt = value;}
        public static implicit operator _InOptMut_Vector3i(Mut_Vector3i value) {return new(value);}
        public unsafe _InOptMut_Vector3i(ref Vector3i value)
        {
            fixed (Vector3i *value_ptr = &value)
            {
                Opt = new((Const_Vector3i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector3i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3i`/`Const_Vector3i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3i`.
    public class _InOptConst_Vector3i
    {
        public Const_Vector3i? Opt;

        public _InOptConst_Vector3i() {}
        public _InOptConst_Vector3i(Const_Vector3i value) {Opt = value;}
        public static implicit operator _InOptConst_Vector3i(Const_Vector3i value) {return new(value);}
        public unsafe _InOptConst_Vector3i(ref readonly Vector3i value)
        {
            fixed (Vector3i *value_ptr = &value)
            {
                Opt = new((Const_Vector3i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3i64`.
    /// This is the const reference to the struct.
    public class Const_Vector3i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector3i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector3i64 UnderlyingStruct => ref *(Vector3i64 *)_UnderlyingPtr;

        internal unsafe Const_Vector3i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector3i64_Destroy(_Underlying *_this);
            __MR_Vector3i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector3i64() {Dispose(false);}

        public ref readonly long X => ref UnderlyingStruct.X;

        public ref readonly long Y => ref UnderlyingStruct.Y;

        public ref readonly long Z => ref UnderlyingStruct.Z;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector3i64_Get_elements();
                return *__MR_Vector3i64_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector3i64(Const_Vector3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3i64 _ctor_result = __MR_Vector3i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3i64::Vector3i64`.
        public unsafe Const_Vector3i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3i64 _ctor_result = __MR_Vector3i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3i64::Vector3i64`.
        public unsafe Const_Vector3i64(long x, long y, long z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_Construct_3(long x, long y, long z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3i64 _ctor_result = __MR_Vector3i64_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from method `MR::Vector3i64::diagonal`.
        public static MR.Vector3i64 Diagonal(long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_diagonal", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_diagonal(long a);
            return __MR_Vector3i64_diagonal(a);
        }

        /// Generated from method `MR::Vector3i64::plusX`.
        public static MR.Vector3i64 PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_plusX", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_plusX();
            return __MR_Vector3i64_plusX();
        }

        /// Generated from method `MR::Vector3i64::plusY`.
        public static MR.Vector3i64 PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_plusY", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_plusY();
            return __MR_Vector3i64_plusY();
        }

        /// Generated from method `MR::Vector3i64::plusZ`.
        public static MR.Vector3i64 PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_plusZ", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_plusZ();
            return __MR_Vector3i64_plusZ();
        }

        /// Generated from method `MR::Vector3i64::minusX`.
        public static MR.Vector3i64 MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_minusX", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_minusX();
            return __MR_Vector3i64_minusX();
        }

        /// Generated from method `MR::Vector3i64::minusY`.
        public static MR.Vector3i64 MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_minusY", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_minusY();
            return __MR_Vector3i64_minusY();
        }

        /// Generated from method `MR::Vector3i64::minusZ`.
        public static MR.Vector3i64 MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_minusZ", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_minusZ();
            return __MR_Vector3i64_minusZ();
        }

        /// Generated from method `MR::Vector3i64::operator[]`.
        public unsafe long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Vector3i64_index_const(_Underlying *_this, int e);
            return *__MR_Vector3i64_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector3i64::lengthSq`.
        public unsafe long LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_lengthSq", ExactSpelling = true)]
            extern static long __MR_Vector3i64_lengthSq(_Underlying *_this);
            return __MR_Vector3i64_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3i64::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_length", ExactSpelling = true)]
            extern static double __MR_Vector3i64_length(_Underlying *_this);
            return __MR_Vector3i64_length(_UnderlyingPtr);
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3i64::furthestBasisVector`.
        public unsafe MR.Vector3i64 FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_furthestBasisVector(_Underlying *_this);
            return __MR_Vector3i64_furthestBasisVector(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_equal_MR_Vector3i64(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector3i64 a, MR.Const_Vector3i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3i64 operator+(MR.Const_Vector3i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Const_Vector3i64._Underlying *__MR_pos_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a);
            return new(__MR_pos_MR_Vector3i64(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i64 operator-(MR.Const_Vector3i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_neg_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a);
            return __MR_neg_MR_Vector3i64(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i64 operator+(MR.Const_Vector3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_add_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_add_MR_Vector3i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i64 operator-(MR.Const_Vector3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_sub_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_sub_MR_Vector3i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i64 operator*(long a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_mul_int64_t_MR_Vector3i64(long a, MR.Const_Vector3i64._Underlying *b);
            return __MR_mul_int64_t_MR_Vector3i64(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i64 operator*(MR.Const_Vector3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_mul_MR_Vector3i64_int64_t(MR.Const_Vector3i64._Underlying *b, long a);
            return __MR_mul_MR_Vector3i64_int64_t(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector3i64 operator/(Const_Vector3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_div_MR_Vector3i64_int64_t(MR.Vector3i64 b, long a);
            return __MR_div_MR_Vector3i64_int64_t(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector3i64? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector3i64)
                return this == (MR.Const_Vector3i64)other;
            return false;
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector3i64 : Const_Vector3i64
    {
        /// Get the underlying struct.
        public unsafe new ref Vector3i64 UnderlyingStruct => ref *(Vector3i64 *)_UnderlyingPtr;

        internal unsafe Mut_Vector3i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref long X => ref UnderlyingStruct.X;

        public new ref long Y => ref UnderlyingStruct.Y;

        public new ref long Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Vector3i64(Const_Vector3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3i64 _ctor_result = __MR_Vector3i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3i64::Vector3i64`.
        public unsafe Mut_Vector3i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3i64 _ctor_result = __MR_Vector3i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3i64::Vector3i64`.
        public unsafe Mut_Vector3i64(long x, long y, long z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_Construct_3(long x, long y, long z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3i64 _ctor_result = __MR_Vector3i64_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from method `MR::Vector3i64::operator[]`.
        public unsafe new ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_index", ExactSpelling = true)]
            extern static long *__MR_Vector3i64_index(_Underlying *_this, int e);
            return ref *__MR_Vector3i64_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3i64 AddAssign(MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_add_assign_MR_Vector3i64(_Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return new(__MR_add_assign_MR_Vector3i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3i64 SubAssign(MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_sub_assign_MR_Vector3i64(_Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return new(__MR_sub_assign_MR_Vector3i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_mul_assign_MR_Vector3i64_int64_t(_Underlying *a, long b);
            return new(__MR_mul_assign_MR_Vector3i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_div_assign_MR_Vector3i64_int64_t(_Underlying *a, long b);
            return new(__MR_div_assign_MR_Vector3i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 24)]
    public struct Vector3i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector3i64(Const_Vector3i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector3i64(Vector3i64 other) => new(new Mut_Vector3i64((Mut_Vector3i64._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public long X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public long Y;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public long Z;

        /// Generated copy constructor.
        public Vector3i64(Vector3i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector3i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_DefaultConstruct();
            this = __MR_Vector3i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3i64::Vector3i64`.
        public unsafe Vector3i64(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector3i64_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3i64::Vector3i64`.
        public unsafe Vector3i64(long x, long y, long z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_Construct_3(long x, long y, long z);
            this = __MR_Vector3i64_Construct_3(x, y, z);
        }

        /// Generated from method `MR::Vector3i64::diagonal`.
        public static MR.Vector3i64 Diagonal(long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_diagonal", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_diagonal(long a);
            return __MR_Vector3i64_diagonal(a);
        }

        /// Generated from method `MR::Vector3i64::plusX`.
        public static MR.Vector3i64 PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_plusX", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_plusX();
            return __MR_Vector3i64_plusX();
        }

        /// Generated from method `MR::Vector3i64::plusY`.
        public static MR.Vector3i64 PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_plusY", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_plusY();
            return __MR_Vector3i64_plusY();
        }

        /// Generated from method `MR::Vector3i64::plusZ`.
        public static MR.Vector3i64 PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_plusZ", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_plusZ();
            return __MR_Vector3i64_plusZ();
        }

        /// Generated from method `MR::Vector3i64::minusX`.
        public static MR.Vector3i64 MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_minusX", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_minusX();
            return __MR_Vector3i64_minusX();
        }

        /// Generated from method `MR::Vector3i64::minusY`.
        public static MR.Vector3i64 MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_minusY", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_minusY();
            return __MR_Vector3i64_minusY();
        }

        /// Generated from method `MR::Vector3i64::minusZ`.
        public static MR.Vector3i64 MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_minusZ", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_minusZ();
            return __MR_Vector3i64_minusZ();
        }

        /// Generated from method `MR::Vector3i64::operator[]`.
        public unsafe long Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Vector3i64_index_const(MR.Vector3i64 *_this, int e);
            fixed (MR.Vector3i64 *__ptr__this = &this)
            {
                return *__MR_Vector3i64_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3i64::operator[]`.
        public unsafe ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_index", ExactSpelling = true)]
            extern static long *__MR_Vector3i64_index(MR.Vector3i64 *_this, int e);
            fixed (MR.Vector3i64 *__ptr__this = &this)
            {
                return ref *__MR_Vector3i64_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3i64::lengthSq`.
        public unsafe long LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_lengthSq", ExactSpelling = true)]
            extern static long __MR_Vector3i64_lengthSq(MR.Vector3i64 *_this);
            fixed (MR.Vector3i64 *__ptr__this = &this)
            {
                return __MR_Vector3i64_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3i64::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_length", ExactSpelling = true)]
            extern static double __MR_Vector3i64_length(MR.Vector3i64 *_this);
            fixed (MR.Vector3i64 *__ptr__this = &this)
            {
                return __MR_Vector3i64_length(__ptr__this);
            }
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3i64::furthestBasisVector`.
        public unsafe MR.Vector3i64 FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3i64_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Vector3i64_furthestBasisVector(MR.Vector3i64 *_this);
            fixed (MR.Vector3i64 *__ptr__this = &this)
            {
                return __MR_Vector3i64_furthestBasisVector(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector3i64 a, MR.Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_equal_MR_Vector3i64((MR.Mut_Vector3i64._Underlying *)&a, (MR.Mut_Vector3i64._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector3i64 a, MR.Vector3i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3i64 operator+(MR.Vector3i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Const_Vector3i64._Underlying *__MR_pos_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a);
            return new(__MR_pos_MR_Vector3i64((MR.Mut_Vector3i64._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i64 operator-(MR.Vector3i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_neg_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a);
            return __MR_neg_MR_Vector3i64((MR.Mut_Vector3i64._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i64 operator+(MR.Vector3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_add_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_add_MR_Vector3i64((MR.Mut_Vector3i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i64 operator-(MR.Vector3i64 a, MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_sub_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
            return __MR_sub_MR_Vector3i64((MR.Mut_Vector3i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i64 operator*(long a, MR.Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_mul_int64_t_MR_Vector3i64(long a, MR.Const_Vector3i64._Underlying *b);
            return __MR_mul_int64_t_MR_Vector3i64(a, (MR.Mut_Vector3i64._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i64 operator*(MR.Vector3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_mul_MR_Vector3i64_int64_t(MR.Const_Vector3i64._Underlying *b, long a);
            return __MR_mul_MR_Vector3i64_int64_t((MR.Mut_Vector3i64._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector3i64 operator/(MR.Vector3i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_div_MR_Vector3i64_int64_t(MR.Vector3i64 b, long a);
            return __MR_div_MR_Vector3i64_int64_t(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3i64 AddAssign(MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_add_assign_MR_Vector3i64(MR.Vector3i64 *a, MR.Const_Vector3i64._Underlying *b);
            fixed (MR.Vector3i64 *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector3i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3i64 SubAssign(MR.Const_Vector3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3i64", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_sub_assign_MR_Vector3i64(MR.Vector3i64 *a, MR.Const_Vector3i64._Underlying *b);
            fixed (MR.Vector3i64 *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector3i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_mul_assign_MR_Vector3i64_int64_t(MR.Vector3i64 *a, long b);
            fixed (MR.Vector3i64 *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector3i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_div_assign_MR_Vector3i64_int64_t(MR.Vector3i64 *a, long b);
            fixed (MR.Vector3i64 *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector3i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector3i64 b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector3i64)
                return this == (MR.Vector3i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector3i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector3i64`/`Const_Vector3i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector3i64
    {
        public readonly bool HasValue;
        internal readonly Vector3i64 Object;
        public Vector3i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector3i64() {HasValue = false;}
        public _InOpt_Vector3i64(Vector3i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector3i64(Vector3i64 new_value) {return new(new_value);}
        public _InOpt_Vector3i64(Const_Vector3i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector3i64(Const_Vector3i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector3i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3i64`/`Const_Vector3i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3i64`.
    public class _InOptMut_Vector3i64
    {
        public Mut_Vector3i64? Opt;

        public _InOptMut_Vector3i64() {}
        public _InOptMut_Vector3i64(Mut_Vector3i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Vector3i64(Mut_Vector3i64 value) {return new(value);}
        public unsafe _InOptMut_Vector3i64(ref Vector3i64 value)
        {
            fixed (Vector3i64 *value_ptr = &value)
            {
                Opt = new((Const_Vector3i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector3i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3i64`/`Const_Vector3i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3i64`.
    public class _InOptConst_Vector3i64
    {
        public Const_Vector3i64? Opt;

        public _InOptConst_Vector3i64() {}
        public _InOptConst_Vector3i64(Const_Vector3i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Vector3i64(Const_Vector3i64 value) {return new(value);}
        public unsafe _InOptConst_Vector3i64(ref readonly Vector3i64 value)
        {
            fixed (Vector3i64 *value_ptr = &value)
            {
                Opt = new((Const_Vector3i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3f`.
    /// This is the const reference to the struct.
    public class Const_Vector3f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector3f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector3f UnderlyingStruct => ref *(Vector3f *)_UnderlyingPtr;

        internal unsafe Const_Vector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector3f_Destroy(_Underlying *_this);
            __MR_Vector3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector3f() {Dispose(false);}

        public ref readonly float X => ref UnderlyingStruct.X;

        public ref readonly float Y => ref UnderlyingStruct.Y;

        public ref readonly float Z => ref UnderlyingStruct.Z;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector3f_Get_elements();
                return *__MR_Vector3f_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector3f(Const_Vector3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 12);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Const_Vector3f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Const_Vector3f(float x, float y, float z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_3(float x, float y, float z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Const_Vector3f(MR.Const_Vector3d v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_double", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_double(MR.Const_Vector3d._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_double(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Const_Vector3f(MR.Const_Vector3i v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_int", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_int(MR.Const_Vector3i._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_int(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from method `MR::Vector3f::diagonal`.
        public static MR.Vector3f Diagonal(float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_diagonal", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_diagonal(float a);
            return __MR_Vector3f_diagonal(a);
        }

        /// Generated from method `MR::Vector3f::plusX`.
        public static MR.Vector3f PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_plusX", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_plusX();
            return __MR_Vector3f_plusX();
        }

        /// Generated from method `MR::Vector3f::plusY`.
        public static MR.Vector3f PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_plusY", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_plusY();
            return __MR_Vector3f_plusY();
        }

        /// Generated from method `MR::Vector3f::plusZ`.
        public static MR.Vector3f PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_plusZ", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_plusZ();
            return __MR_Vector3f_plusZ();
        }

        /// Generated from method `MR::Vector3f::minusX`.
        public static MR.Vector3f MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_minusX", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_minusX();
            return __MR_Vector3f_minusX();
        }

        /// Generated from method `MR::Vector3f::minusY`.
        public static MR.Vector3f MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_minusY", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_minusY();
            return __MR_Vector3f_minusY();
        }

        /// Generated from method `MR::Vector3f::minusZ`.
        public static MR.Vector3f MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_minusZ", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_minusZ();
            return __MR_Vector3f_minusZ();
        }

        /// Generated from method `MR::Vector3f::operator[]`.
        public unsafe float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_index_const", ExactSpelling = true)]
            extern static float *__MR_Vector3f_index_const(_Underlying *_this, int e);
            return *__MR_Vector3f_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector3f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_lengthSq", ExactSpelling = true)]
            extern static float __MR_Vector3f_lengthSq(_Underlying *_this);
            return __MR_Vector3f_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_length", ExactSpelling = true)]
            extern static float __MR_Vector3f_length(_Underlying *_this);
            return __MR_Vector3f_length(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3f::normalized`.
        public unsafe MR.Vector3f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_normalized", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_normalized(_Underlying *_this);
            return __MR_Vector3f_normalized(_UnderlyingPtr);
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3f::furthestBasisVector`.
        public unsafe MR.Vector3f FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_furthestBasisVector(_Underlying *_this);
            return __MR_Vector3f_furthestBasisVector(_UnderlyingPtr);
        }

        /// returns 2 unit vector, which together with this vector make an orthogonal basis
        /// Currently not implemented for integral vectors.
        /// Generated from method `MR::Vector3f::perpendicular`.
        public unsafe MR.Std.Pair_MRVector3f_MRVector3f Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_perpendicular", ExactSpelling = true)]
            extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_Vector3f_perpendicular(_Underlying *_this);
            return new(__MR_Vector3f_perpendicular(_UnderlyingPtr), is_owning: true);
        }

        /// returns this vector transformed by xf if it is
        /// Generated from method `MR::Vector3f::transformed<float>`.
        public unsafe MR.Vector3f Transformed(MR.Const_AffineXf3f? xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_transformed", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_transformed(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf);
            return __MR_Vector3f_transformed(_UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Vector3f::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector3f_isFinite(_Underlying *_this);
            return __MR_Vector3f_isFinite(_UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_equal_MR_Vector3f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector3f a, MR.Const_Vector3f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3f operator+(MR.Const_Vector3f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_pos_MR_Vector3f(MR.Const_Vector3f._Underlying *a);
            return new(__MR_pos_MR_Vector3f(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3f operator-(MR.Const_Vector3f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_neg_MR_Vector3f(MR.Const_Vector3f._Underlying *a);
            return __MR_neg_MR_Vector3f(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3f operator+(MR.Const_Vector3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_add_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_add_MR_Vector3f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3f operator-(MR.Const_Vector3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_sub_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_sub_MR_Vector3f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3f operator*(float a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_mul_float_MR_Vector3f(float a, MR.Const_Vector3f._Underlying *b);
            return __MR_mul_float_MR_Vector3f(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3f operator*(MR.Const_Vector3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Vector3f __MR_mul_MR_Vector3f_float(MR.Const_Vector3f._Underlying *b, float a);
            return __MR_mul_MR_Vector3f_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector3f operator/(Const_Vector3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Vector3f __MR_div_MR_Vector3f_float(MR.Vector3f b, float a);
            return __MR_div_MR_Vector3f_float(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector3f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector3f)
                return this == (MR.Const_Vector3f)other;
            return false;
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3f`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector3f : Const_Vector3f
    {
        /// Get the underlying struct.
        public unsafe new ref Vector3f UnderlyingStruct => ref *(Vector3f *)_UnderlyingPtr;

        internal unsafe Mut_Vector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref float X => ref UnderlyingStruct.X;

        public new ref float Y => ref UnderlyingStruct.Y;

        public new ref float Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Vector3f(Const_Vector3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 12);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Mut_Vector3f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Mut_Vector3f(float x, float y, float z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_3(float x, float y, float z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Mut_Vector3f(MR.Const_Vector3d v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_double", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_double(MR.Const_Vector3d._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_double(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Mut_Vector3f(MR.Const_Vector3i v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_int", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_int(MR.Const_Vector3i._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(12);
            MR.Vector3f _ctor_result = __MR_Vector3f_Construct_int(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 12);
        }

        /// Generated from method `MR::Vector3f::operator[]`.
        public unsafe new ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_index", ExactSpelling = true)]
            extern static float *__MR_Vector3f_index(_Underlying *_this, int e);
            return ref *__MR_Vector3f_index(_UnderlyingPtr, e);
        }

        /// get rid of signed zero values to be sure that equal vectors have identical binary representation
        /// Generated from method `MR::Vector3f::unsignZeroValues`.
        public unsafe void UnsignZeroValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_unsignZeroValues", ExactSpelling = true)]
            extern static void __MR_Vector3f_unsignZeroValues(_Underlying *_this);
            __MR_Vector3f_unsignZeroValues(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3f AddAssign(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_add_assign_MR_Vector3f(_Underlying *a, MR.Const_Vector3f._Underlying *b);
            return new(__MR_add_assign_MR_Vector3f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3f SubAssign(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_sub_assign_MR_Vector3f(_Underlying *a, MR.Const_Vector3f._Underlying *b);
            return new(__MR_sub_assign_MR_Vector3f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_mul_assign_MR_Vector3f_float(_Underlying *a, float b);
            return new(__MR_mul_assign_MR_Vector3f_float(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_div_assign_MR_Vector3f_float(_Underlying *a, float b);
            return new(__MR_div_assign_MR_Vector3f_float(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 12)]
    public struct Vector3f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector3f(Const_Vector3f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector3f(Vector3f other) => new(new Mut_Vector3f((Mut_Vector3f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public float X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public float Y;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public float Z;

        /// Generated copy constructor.
        public Vector3f(Vector3f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector3f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_DefaultConstruct();
            this = __MR_Vector3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Vector3f(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector3f_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Vector3f(float x, float y, float z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_3(float x, float y, float z);
            this = __MR_Vector3f_Construct_3(x, y, z);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Vector3f(MR.Const_Vector3d v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_double", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_double(MR.Const_Vector3d._Underlying *v);
            this = __MR_Vector3f_Construct_double(v._UnderlyingPtr);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3f::Vector3f`.
        public unsafe Vector3f(MR.Const_Vector3i v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_Construct_int", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_Construct_int(MR.Const_Vector3i._Underlying *v);
            this = __MR_Vector3f_Construct_int(v._UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3f::diagonal`.
        public static MR.Vector3f Diagonal(float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_diagonal", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_diagonal(float a);
            return __MR_Vector3f_diagonal(a);
        }

        /// Generated from method `MR::Vector3f::plusX`.
        public static MR.Vector3f PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_plusX", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_plusX();
            return __MR_Vector3f_plusX();
        }

        /// Generated from method `MR::Vector3f::plusY`.
        public static MR.Vector3f PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_plusY", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_plusY();
            return __MR_Vector3f_plusY();
        }

        /// Generated from method `MR::Vector3f::plusZ`.
        public static MR.Vector3f PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_plusZ", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_plusZ();
            return __MR_Vector3f_plusZ();
        }

        /// Generated from method `MR::Vector3f::minusX`.
        public static MR.Vector3f MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_minusX", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_minusX();
            return __MR_Vector3f_minusX();
        }

        /// Generated from method `MR::Vector3f::minusY`.
        public static MR.Vector3f MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_minusY", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_minusY();
            return __MR_Vector3f_minusY();
        }

        /// Generated from method `MR::Vector3f::minusZ`.
        public static MR.Vector3f MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_minusZ", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_minusZ();
            return __MR_Vector3f_minusZ();
        }

        /// Generated from method `MR::Vector3f::operator[]`.
        public unsafe float Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_index_const", ExactSpelling = true)]
            extern static float *__MR_Vector3f_index_const(MR.Vector3f *_this, int e);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return *__MR_Vector3f_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3f::operator[]`.
        public unsafe ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_index", ExactSpelling = true)]
            extern static float *__MR_Vector3f_index(MR.Vector3f *_this, int e);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return ref *__MR_Vector3f_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_lengthSq", ExactSpelling = true)]
            extern static float __MR_Vector3f_lengthSq(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return __MR_Vector3f_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_length", ExactSpelling = true)]
            extern static float __MR_Vector3f_length(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return __MR_Vector3f_length(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3f::normalized`.
        public unsafe MR.Vector3f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_normalized", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_normalized(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return __MR_Vector3f_normalized(__ptr__this);
            }
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3f::furthestBasisVector`.
        public unsafe MR.Vector3f FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_furthestBasisVector(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return __MR_Vector3f_furthestBasisVector(__ptr__this);
            }
        }

        /// returns 2 unit vector, which together with this vector make an orthogonal basis
        /// Currently not implemented for integral vectors.
        /// Generated from method `MR::Vector3f::perpendicular`.
        public unsafe MR.Std.Pair_MRVector3f_MRVector3f Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_perpendicular", ExactSpelling = true)]
            extern static MR.Std.Pair_MRVector3f_MRVector3f._Underlying *__MR_Vector3f_perpendicular(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return new(__MR_Vector3f_perpendicular(__ptr__this), is_owning: true);
            }
        }

        /// returns this vector transformed by xf if it is
        /// Generated from method `MR::Vector3f::transformed<float>`.
        public unsafe MR.Vector3f Transformed(MR.Const_AffineXf3f? xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_transformed", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector3f_transformed(MR.Vector3f *_this, MR.Const_AffineXf3f._Underlying *xf);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return __MR_Vector3f_transformed(__ptr__this, xf is not null ? xf._UnderlyingPtr : null);
            }
        }

        /// get rid of signed zero values to be sure that equal vectors have identical binary representation
        /// Generated from method `MR::Vector3f::unsignZeroValues`.
        public unsafe void UnsignZeroValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_unsignZeroValues", ExactSpelling = true)]
            extern static void __MR_Vector3f_unsignZeroValues(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                __MR_Vector3f_unsignZeroValues(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3f::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3f_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector3f_isFinite(MR.Vector3f *_this);
            fixed (MR.Vector3f *__ptr__this = &this)
            {
                return __MR_Vector3f_isFinite(__ptr__this) != 0;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector3f a, MR.Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_equal_MR_Vector3f((MR.Mut_Vector3f._Underlying *)&a, (MR.Mut_Vector3f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector3f a, MR.Vector3f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3f operator+(MR.Vector3f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_pos_MR_Vector3f(MR.Const_Vector3f._Underlying *a);
            return new(__MR_pos_MR_Vector3f((MR.Mut_Vector3f._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3f operator-(MR.Vector3f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_neg_MR_Vector3f(MR.Const_Vector3f._Underlying *a);
            return __MR_neg_MR_Vector3f((MR.Mut_Vector3f._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3f operator+(MR.Vector3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_add_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_add_MR_Vector3f((MR.Mut_Vector3f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3f operator-(MR.Vector3f a, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_sub_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            return __MR_sub_MR_Vector3f((MR.Mut_Vector3f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3f operator*(float a, MR.Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_mul_float_MR_Vector3f(float a, MR.Const_Vector3f._Underlying *b);
            return __MR_mul_float_MR_Vector3f(a, (MR.Mut_Vector3f._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3f operator*(MR.Vector3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Vector3f __MR_mul_MR_Vector3f_float(MR.Const_Vector3f._Underlying *b, float a);
            return __MR_mul_MR_Vector3f_float((MR.Mut_Vector3f._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector3f operator/(MR.Vector3f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Vector3f __MR_div_MR_Vector3f_float(MR.Vector3f b, float a);
            return __MR_div_MR_Vector3f_float(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3f AddAssign(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_add_assign_MR_Vector3f(MR.Vector3f *a, MR.Const_Vector3f._Underlying *b);
            fixed (MR.Vector3f *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector3f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3f SubAssign(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_sub_assign_MR_Vector3f(MR.Vector3f *a, MR.Const_Vector3f._Underlying *b);
            fixed (MR.Vector3f *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector3f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_mul_assign_MR_Vector3f_float(MR.Vector3f *a, float b);
            fixed (MR.Vector3f *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector3f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_div_assign_MR_Vector3f_float(MR.Vector3f *a, float b);
            fixed (MR.Vector3f *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector3f_float(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector3f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector3f)
                return this == (MR.Vector3f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector3f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector3f`/`Const_Vector3f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector3f
    {
        public readonly bool HasValue;
        internal readonly Vector3f Object;
        public Vector3f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector3f() {HasValue = false;}
        public _InOpt_Vector3f(Vector3f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector3f(Vector3f new_value) {return new(new_value);}
        public _InOpt_Vector3f(Const_Vector3f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector3f(Const_Vector3f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3f`/`Const_Vector3f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3f`.
    public class _InOptMut_Vector3f
    {
        public Mut_Vector3f? Opt;

        public _InOptMut_Vector3f() {}
        public _InOptMut_Vector3f(Mut_Vector3f value) {Opt = value;}
        public static implicit operator _InOptMut_Vector3f(Mut_Vector3f value) {return new(value);}
        public unsafe _InOptMut_Vector3f(ref Vector3f value)
        {
            fixed (Vector3f *value_ptr = &value)
            {
                Opt = new((Const_Vector3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3f`/`Const_Vector3f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3f`.
    public class _InOptConst_Vector3f
    {
        public Const_Vector3f? Opt;

        public _InOptConst_Vector3f() {}
        public _InOptConst_Vector3f(Const_Vector3f value) {Opt = value;}
        public static implicit operator _InOptConst_Vector3f(Const_Vector3f value) {return new(value);}
        public unsafe _InOptConst_Vector3f(ref readonly Vector3f value)
        {
            fixed (Vector3f *value_ptr = &value)
            {
                Opt = new((Const_Vector3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3d`.
    /// This is the const reference to the struct.
    public class Const_Vector3d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector3d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector3d UnderlyingStruct => ref *(Vector3d *)_UnderlyingPtr;

        internal unsafe Const_Vector3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector3d_Destroy(_Underlying *_this);
            __MR_Vector3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector3d() {Dispose(false);}

        public ref readonly double X => ref UnderlyingStruct.X;

        public ref readonly double Y => ref UnderlyingStruct.Y;

        public ref readonly double Z => ref UnderlyingStruct.Z;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector3d_Get_elements();
                return *__MR_Vector3d_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector3d(Const_Vector3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Const_Vector3d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Const_Vector3d(double x, double y, double z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_3(double x, double y, double z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Const_Vector3d(MR.Const_Vector3f v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_float", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_float(MR.Const_Vector3f._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_Construct_float(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from method `MR::Vector3d::diagonal`.
        public static MR.Vector3d Diagonal(double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_diagonal", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_diagonal(double a);
            return __MR_Vector3d_diagonal(a);
        }

        /// Generated from method `MR::Vector3d::plusX`.
        public static MR.Vector3d PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_plusX", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_plusX();
            return __MR_Vector3d_plusX();
        }

        /// Generated from method `MR::Vector3d::plusY`.
        public static MR.Vector3d PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_plusY", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_plusY();
            return __MR_Vector3d_plusY();
        }

        /// Generated from method `MR::Vector3d::plusZ`.
        public static MR.Vector3d PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_plusZ", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_plusZ();
            return __MR_Vector3d_plusZ();
        }

        /// Generated from method `MR::Vector3d::minusX`.
        public static MR.Vector3d MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_minusX", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_minusX();
            return __MR_Vector3d_minusX();
        }

        /// Generated from method `MR::Vector3d::minusY`.
        public static MR.Vector3d MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_minusY", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_minusY();
            return __MR_Vector3d_minusY();
        }

        /// Generated from method `MR::Vector3d::minusZ`.
        public static MR.Vector3d MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_minusZ", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_minusZ();
            return __MR_Vector3d_minusZ();
        }

        /// Generated from method `MR::Vector3d::operator[]`.
        public unsafe double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_index_const", ExactSpelling = true)]
            extern static double *__MR_Vector3d_index_const(_Underlying *_this, int e);
            return *__MR_Vector3d_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector3d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_lengthSq", ExactSpelling = true)]
            extern static double __MR_Vector3d_lengthSq(_Underlying *_this);
            return __MR_Vector3d_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_length", ExactSpelling = true)]
            extern static double __MR_Vector3d_length(_Underlying *_this);
            return __MR_Vector3d_length(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3d::normalized`.
        public unsafe MR.Vector3d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_normalized", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_normalized(_Underlying *_this);
            return __MR_Vector3d_normalized(_UnderlyingPtr);
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3d::furthestBasisVector`.
        public unsafe MR.Vector3d FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_furthestBasisVector(_Underlying *_this);
            return __MR_Vector3d_furthestBasisVector(_UnderlyingPtr);
        }

        /// returns 2 unit vector, which together with this vector make an orthogonal basis
        /// Currently not implemented for integral vectors.
        /// Generated from method `MR::Vector3d::perpendicular`.
        public unsafe MR.Std.Pair_MRVector3d_MRVector3d Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_perpendicular", ExactSpelling = true)]
            extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_Vector3d_perpendicular(_Underlying *_this);
            return new(__MR_Vector3d_perpendicular(_UnderlyingPtr), is_owning: true);
        }

        /// returns this vector transformed by xf if it is
        /// Generated from method `MR::Vector3d::transformed<double>`.
        public unsafe MR.Vector3d Transformed(MR.Const_AffineXf3d? xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_transformed", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_transformed(_Underlying *_this, MR.Const_AffineXf3d._Underlying *xf);
            return __MR_Vector3d_transformed(_UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Vector3d::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector3d_isFinite(_Underlying *_this);
            return __MR_Vector3d_isFinite(_UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_equal_MR_Vector3d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector3d a, MR.Const_Vector3d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3d operator+(MR.Const_Vector3d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_pos_MR_Vector3d(MR.Const_Vector3d._Underlying *a);
            return new(__MR_pos_MR_Vector3d(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3d operator-(MR.Const_Vector3d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_neg_MR_Vector3d(MR.Const_Vector3d._Underlying *a);
            return __MR_neg_MR_Vector3d(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3d operator+(MR.Const_Vector3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_add_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_add_MR_Vector3d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3d operator-(MR.Const_Vector3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_sub_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_sub_MR_Vector3d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3d operator*(double a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_mul_double_MR_Vector3d(double a, MR.Const_Vector3d._Underlying *b);
            return __MR_mul_double_MR_Vector3d(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3d operator*(MR.Const_Vector3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Vector3d __MR_mul_MR_Vector3d_double(MR.Const_Vector3d._Underlying *b, double a);
            return __MR_mul_MR_Vector3d_double(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector3d operator/(Const_Vector3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Vector3d __MR_div_MR_Vector3d_double(MR.Vector3d b, double a);
            return __MR_div_MR_Vector3d_double(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector3d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector3d)
                return this == (MR.Const_Vector3d)other;
            return false;
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3d`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector3d : Const_Vector3d
    {
        /// Get the underlying struct.
        public unsafe new ref Vector3d UnderlyingStruct => ref *(Vector3d *)_UnderlyingPtr;

        internal unsafe Mut_Vector3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref double X => ref UnderlyingStruct.X;

        public new ref double Y => ref UnderlyingStruct.Y;

        public new ref double Z => ref UnderlyingStruct.Z;

        /// Generated copy constructor.
        public unsafe Mut_Vector3d(Const_Vector3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Mut_Vector3d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Mut_Vector3d(double x, double y, double z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_3(double x, double y, double z);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_Construct_3(x, y, z);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Mut_Vector3d(MR.Const_Vector3f v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_float", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_float(MR.Const_Vector3f._Underlying *v);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Vector3d _ctor_result = __MR_Vector3d_Construct_float(v._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from method `MR::Vector3d::operator[]`.
        public unsafe new ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_index", ExactSpelling = true)]
            extern static double *__MR_Vector3d_index(_Underlying *_this, int e);
            return ref *__MR_Vector3d_index(_UnderlyingPtr, e);
        }

        /// get rid of signed zero values to be sure that equal vectors have identical binary representation
        /// Generated from method `MR::Vector3d::unsignZeroValues`.
        public unsafe void UnsignZeroValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_unsignZeroValues", ExactSpelling = true)]
            extern static void __MR_Vector3d_unsignZeroValues(_Underlying *_this);
            __MR_Vector3d_unsignZeroValues(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3d AddAssign(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_add_assign_MR_Vector3d(_Underlying *a, MR.Const_Vector3d._Underlying *b);
            return new(__MR_add_assign_MR_Vector3d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3d SubAssign(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_sub_assign_MR_Vector3d(_Underlying *a, MR.Const_Vector3d._Underlying *b);
            return new(__MR_sub_assign_MR_Vector3d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_mul_assign_MR_Vector3d_double(_Underlying *a, double b);
            return new(__MR_mul_assign_MR_Vector3d_double(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_div_assign_MR_Vector3d_double(_Underlying *a, double b);
            return new(__MR_div_assign_MR_Vector3d_double(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 24)]
    public struct Vector3d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector3d(Const_Vector3d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector3d(Vector3d other) => new(new Mut_Vector3d((Mut_Vector3d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public double X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public double Y;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public double Z;

        /// Generated copy constructor.
        public Vector3d(Vector3d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_DefaultConstruct();
            this = __MR_Vector3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Vector3d(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector3d_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Vector3d(double x, double y, double z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_3(double x, double y, double z);
            this = __MR_Vector3d_Construct_3(x, y, z);
        }

        // Here `T == U` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and looks out of place there.
        /// Generated from constructor `MR::Vector3d::Vector3d`.
        public unsafe Vector3d(MR.Const_Vector3f v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_Construct_float", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_Construct_float(MR.Const_Vector3f._Underlying *v);
            this = __MR_Vector3d_Construct_float(v._UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3d::diagonal`.
        public static MR.Vector3d Diagonal(double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_diagonal", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_diagonal(double a);
            return __MR_Vector3d_diagonal(a);
        }

        /// Generated from method `MR::Vector3d::plusX`.
        public static MR.Vector3d PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_plusX", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_plusX();
            return __MR_Vector3d_plusX();
        }

        /// Generated from method `MR::Vector3d::plusY`.
        public static MR.Vector3d PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_plusY", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_plusY();
            return __MR_Vector3d_plusY();
        }

        /// Generated from method `MR::Vector3d::plusZ`.
        public static MR.Vector3d PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_plusZ", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_plusZ();
            return __MR_Vector3d_plusZ();
        }

        /// Generated from method `MR::Vector3d::minusX`.
        public static MR.Vector3d MinusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_minusX", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_minusX();
            return __MR_Vector3d_minusX();
        }

        /// Generated from method `MR::Vector3d::minusY`.
        public static MR.Vector3d MinusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_minusY", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_minusY();
            return __MR_Vector3d_minusY();
        }

        /// Generated from method `MR::Vector3d::minusZ`.
        public static MR.Vector3d MinusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_minusZ", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_minusZ();
            return __MR_Vector3d_minusZ();
        }

        /// Generated from method `MR::Vector3d::operator[]`.
        public unsafe double Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_index_const", ExactSpelling = true)]
            extern static double *__MR_Vector3d_index_const(MR.Vector3d *_this, int e);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return *__MR_Vector3d_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3d::operator[]`.
        public unsafe ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_index", ExactSpelling = true)]
            extern static double *__MR_Vector3d_index(MR.Vector3d *_this, int e);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return ref *__MR_Vector3d_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector3d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_lengthSq", ExactSpelling = true)]
            extern static double __MR_Vector3d_lengthSq(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return __MR_Vector3d_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_length", ExactSpelling = true)]
            extern static double __MR_Vector3d_length(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return __MR_Vector3d_length(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3d::normalized`.
        public unsafe MR.Vector3d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_normalized", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_normalized(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return __MR_Vector3d_normalized(__ptr__this);
            }
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3d::furthestBasisVector`.
        public unsafe MR.Vector3d FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_furthestBasisVector(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return __MR_Vector3d_furthestBasisVector(__ptr__this);
            }
        }

        /// returns 2 unit vector, which together with this vector make an orthogonal basis
        /// Currently not implemented for integral vectors.
        /// Generated from method `MR::Vector3d::perpendicular`.
        public unsafe MR.Std.Pair_MRVector3d_MRVector3d Perpendicular()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_perpendicular", ExactSpelling = true)]
            extern static MR.Std.Pair_MRVector3d_MRVector3d._Underlying *__MR_Vector3d_perpendicular(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return new(__MR_Vector3d_perpendicular(__ptr__this), is_owning: true);
            }
        }

        /// returns this vector transformed by xf if it is
        /// Generated from method `MR::Vector3d::transformed<double>`.
        public unsafe MR.Vector3d Transformed(MR.Const_AffineXf3d? xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_transformed", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector3d_transformed(MR.Vector3d *_this, MR.Const_AffineXf3d._Underlying *xf);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return __MR_Vector3d_transformed(__ptr__this, xf is not null ? xf._UnderlyingPtr : null);
            }
        }

        /// get rid of signed zero values to be sure that equal vectors have identical binary representation
        /// Generated from method `MR::Vector3d::unsignZeroValues`.
        public unsafe void UnsignZeroValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_unsignZeroValues", ExactSpelling = true)]
            extern static void __MR_Vector3d_unsignZeroValues(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                __MR_Vector3d_unsignZeroValues(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector3d::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3d_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector3d_isFinite(MR.Vector3d *_this);
            fixed (MR.Vector3d *__ptr__this = &this)
            {
                return __MR_Vector3d_isFinite(__ptr__this) != 0;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector3d a, MR.Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_equal_MR_Vector3d((MR.Mut_Vector3d._Underlying *)&a, (MR.Mut_Vector3d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector3d a, MR.Vector3d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3d operator+(MR.Vector3d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_pos_MR_Vector3d(MR.Const_Vector3d._Underlying *a);
            return new(__MR_pos_MR_Vector3d((MR.Mut_Vector3d._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3d operator-(MR.Vector3d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_neg_MR_Vector3d(MR.Const_Vector3d._Underlying *a);
            return __MR_neg_MR_Vector3d((MR.Mut_Vector3d._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3d operator+(MR.Vector3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_add_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_add_MR_Vector3d((MR.Mut_Vector3d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3d operator-(MR.Vector3d a, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_sub_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            return __MR_sub_MR_Vector3d((MR.Mut_Vector3d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3d operator*(double a, MR.Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_mul_double_MR_Vector3d(double a, MR.Const_Vector3d._Underlying *b);
            return __MR_mul_double_MR_Vector3d(a, (MR.Mut_Vector3d._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3d operator*(MR.Vector3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Vector3d __MR_mul_MR_Vector3d_double(MR.Const_Vector3d._Underlying *b, double a);
            return __MR_mul_MR_Vector3d_double((MR.Mut_Vector3d._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector3d operator/(MR.Vector3d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Vector3d __MR_div_MR_Vector3d_double(MR.Vector3d b, double a);
            return __MR_div_MR_Vector3d_double(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector3d AddAssign(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_add_assign_MR_Vector3d(MR.Vector3d *a, MR.Const_Vector3d._Underlying *b);
            fixed (MR.Vector3d *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector3d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector3d SubAssign(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_sub_assign_MR_Vector3d(MR.Vector3d *a, MR.Const_Vector3d._Underlying *b);
            fixed (MR.Vector3d *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector3d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector3d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_mul_assign_MR_Vector3d_double(MR.Vector3d *a, double b);
            fixed (MR.Vector3d *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector3d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector3d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_div_assign_MR_Vector3d_double(MR.Vector3d *a, double b);
            fixed (MR.Vector3d *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector3d_double(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector3d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector3d)
                return this == (MR.Vector3d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector3d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector3d`/`Const_Vector3d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector3d
    {
        public readonly bool HasValue;
        internal readonly Vector3d Object;
        public Vector3d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector3d() {HasValue = false;}
        public _InOpt_Vector3d(Vector3d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector3d(Vector3d new_value) {return new(new_value);}
        public _InOpt_Vector3d(Const_Vector3d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector3d(Const_Vector3d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3d`/`Const_Vector3d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3d`.
    public class _InOptMut_Vector3d
    {
        public Mut_Vector3d? Opt;

        public _InOptMut_Vector3d() {}
        public _InOptMut_Vector3d(Mut_Vector3d value) {Opt = value;}
        public static implicit operator _InOptMut_Vector3d(Mut_Vector3d value) {return new(value);}
        public unsafe _InOptMut_Vector3d(ref Vector3d value)
        {
            fixed (Vector3d *value_ptr = &value)
            {
                Opt = new((Const_Vector3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector3d`/`Const_Vector3d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector3d`.
    public class _InOptConst_Vector3d
    {
        public Const_Vector3d? Opt;

        public _InOptConst_Vector3d() {}
        public _InOptConst_Vector3d(Const_Vector3d value) {Opt = value;}
        public static implicit operator _InOptConst_Vector3d(Const_Vector3d value) {return new(value);}
        public unsafe _InOptConst_Vector3d(ref readonly Vector3d value)
        {
            fixed (Vector3d *value_ptr = &value)
            {
                Opt = new((Const_Vector3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3<unsigned char>`.
    /// This is the const half of the class.
    public class Const_Vector3_UnsignedChar : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector3_UnsignedChar>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Vector3_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector3_unsigned_char_Destroy(_Underlying *_this);
            __MR_Vector3_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector3_UnsignedChar() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector3_unsigned_char_Get_elements();
                return *__MR_Vector3_unsigned_char_Get_elements();
            }
        }

        public unsafe byte X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Get_x", ExactSpelling = true)]
                extern static byte *__MR_Vector3_unsigned_char_Get_x(_Underlying *_this);
                return *__MR_Vector3_unsigned_char_Get_x(_UnderlyingPtr);
            }
        }

        public unsafe byte Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Get_y", ExactSpelling = true)]
                extern static byte *__MR_Vector3_unsigned_char_Get_y(_Underlying *_this);
                return *__MR_Vector3_unsigned_char_Get_y(_UnderlyingPtr);
            }
        }

        public unsafe byte Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Get_z", ExactSpelling = true)]
                extern static byte *__MR_Vector3_unsigned_char_Get_z(_Underlying *_this);
                return *__MR_Vector3_unsigned_char_Get_z(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector3_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Vector3_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3<unsigned char>::Vector3`.
        public unsafe Const_Vector3_UnsignedChar(MR.Const_Vector3_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_ConstructFromAnother(MR.Vector3_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Vector3_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3<unsigned char>::Vector3`.
        public unsafe Const_Vector3_UnsignedChar(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Vector3_unsigned_char_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3<unsigned char>::Vector3`.
        public unsafe Const_Vector3_UnsignedChar(byte x, byte y, byte z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_Construct_3(byte x, byte y, byte z);
            _UnderlyingPtr = __MR_Vector3_unsigned_char_Construct_3(x, y, z);
        }

        /// Generated from method `MR::Vector3<unsigned char>::diagonal`.
        public static unsafe MR.Vector3_UnsignedChar Diagonal(byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_diagonal", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_diagonal(byte a);
            return new(__MR_Vector3_unsigned_char_diagonal(a), is_owning: true);
        }

        /// Generated from method `MR::Vector3<unsigned char>::plusX`.
        public static unsafe MR.Vector3_UnsignedChar PlusX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_plusX", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_plusX();
            return new(__MR_Vector3_unsigned_char_plusX(), is_owning: true);
        }

        /// Generated from method `MR::Vector3<unsigned char>::plusY`.
        public static unsafe MR.Vector3_UnsignedChar PlusY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_plusY", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_plusY();
            return new(__MR_Vector3_unsigned_char_plusY(), is_owning: true);
        }

        /// Generated from method `MR::Vector3<unsigned char>::plusZ`.
        public static unsafe MR.Vector3_UnsignedChar PlusZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_plusZ", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_plusZ();
            return new(__MR_Vector3_unsigned_char_plusZ(), is_owning: true);
        }

        /// Generated from method `MR::Vector3<unsigned char>::operator[]`.
        public unsafe byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_index_const", ExactSpelling = true)]
            extern static byte *__MR_Vector3_unsigned_char_index_const(_Underlying *_this, int e);
            return *__MR_Vector3_unsigned_char_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector3<unsigned char>::lengthSq`.
        public unsafe byte LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector3_unsigned_char_lengthSq(_Underlying *_this);
            return __MR_Vector3_unsigned_char_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector3<unsigned char>::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_length", ExactSpelling = true)]
            extern static double __MR_Vector3_unsigned_char_length(_Underlying *_this);
            return __MR_Vector3_unsigned_char_length(_UnderlyingPtr);
        }

        /// returns one of 3 basis unit vector that makes the biggest angle with the direction specified by this
        /// Generated from method `MR::Vector3<unsigned char>::furthestBasisVector`.
        public unsafe MR.Vector3_UnsignedChar FurthestBasisVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_furthestBasisVector", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_furthestBasisVector(_Underlying *_this);
            return new(__MR_Vector3_unsigned_char_furthestBasisVector(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector3_UnsignedChar a, MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return __MR_equal_MR_Vector3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector3_UnsignedChar a, MR.Const_Vector3_UnsignedChar b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector3_UnsignedChar operator+(MR.Const_Vector3_UnsignedChar a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Const_Vector3_UnsignedChar._Underlying *__MR_pos_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *a);
            return new(__MR_pos_MR_Vector3_unsigned_char(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Const_Vector3_UnsignedChar a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_neg_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *a);
            return __MR_neg_MR_Vector3_unsigned_char(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector3i operator+(MR.Const_Vector3_UnsignedChar a, MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_add_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return __MR_add_MR_Vector3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector3i operator-(MR.Const_Vector3_UnsignedChar a, MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_sub_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return __MR_sub_MR_Vector3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(byte a, MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_unsigned_char_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_unsigned_char_MR_Vector3_unsigned_char(byte a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return __MR_mul_unsigned_char_MR_Vector3_unsigned_char(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector3i operator*(MR.Const_Vector3_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_mul_MR_Vector3_unsigned_char_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *b, byte a);
            return __MR_mul_MR_Vector3_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector3i operator/(Const_Vector3_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3i __MR_div_MR_Vector3_unsigned_char_unsigned_char(MR.Vector3_UnsignedChar._Underlying *b, byte a);
            return __MR_div_MR_Vector3_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector3_UnsignedChar? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector3_UnsignedChar)
                return this == (MR.Const_Vector3_UnsignedChar)other;
            return false;
        }
    }

    /// three-dimensional vector
    /// Generated from class `MR::Vector3<unsigned char>`.
    /// This is the non-const half of the class.
    public class Vector3_UnsignedChar : Const_Vector3_UnsignedChar
    {
        internal unsafe Vector3_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref byte X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_GetMutable_x", ExactSpelling = true)]
                extern static byte *__MR_Vector3_unsigned_char_GetMutable_x(_Underlying *_this);
                return ref *__MR_Vector3_unsigned_char_GetMutable_x(_UnderlyingPtr);
            }
        }

        public new unsafe ref byte Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_GetMutable_y", ExactSpelling = true)]
                extern static byte *__MR_Vector3_unsigned_char_GetMutable_y(_Underlying *_this);
                return ref *__MR_Vector3_unsigned_char_GetMutable_y(_UnderlyingPtr);
            }
        }

        public new unsafe ref byte Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_GetMutable_z", ExactSpelling = true)]
                extern static byte *__MR_Vector3_unsigned_char_GetMutable_z(_Underlying *_this);
                return ref *__MR_Vector3_unsigned_char_GetMutable_z(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector3_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Vector3_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector3<unsigned char>::Vector3`.
        public unsafe Vector3_UnsignedChar(MR.Const_Vector3_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_ConstructFromAnother(MR.Vector3_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Vector3_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3<unsigned char>::Vector3`.
        public unsafe Vector3_UnsignedChar(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Construct_1", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Vector3_unsigned_char_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector3<unsigned char>::Vector3`.
        public unsafe Vector3_UnsignedChar(byte x, byte y, byte z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_Construct_3", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_Construct_3(byte x, byte y, byte z);
            _UnderlyingPtr = __MR_Vector3_unsigned_char_Construct_3(x, y, z);
        }

        /// Generated from method `MR::Vector3<unsigned char>::operator=`.
        public unsafe MR.Vector3_UnsignedChar Assign(MR.Const_Vector3_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_Vector3_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Vector3_UnsignedChar._Underlying *_other);
            return new(__MR_Vector3_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Vector3<unsigned char>::operator[]`.
        public unsafe new ref byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector3_unsigned_char_index", ExactSpelling = true)]
            extern static byte *__MR_Vector3_unsigned_char_index(_Underlying *_this, int e);
            return ref *__MR_Vector3_unsigned_char_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Vector3_UnsignedChar AddAssign(MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_add_assign_MR_Vector3_unsigned_char(_Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return new(__MR_add_assign_MR_Vector3_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Vector3_UnsignedChar SubAssign(MR.Const_Vector3_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector3_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_sub_assign_MR_Vector3_unsigned_char(_Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
            return new(__MR_sub_assign_MR_Vector3_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Vector3_UnsignedChar MulAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_mul_assign_MR_Vector3_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Vector3_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Vector3_UnsignedChar DivAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector3_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector3_UnsignedChar._Underlying *__MR_div_assign_MR_Vector3_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Vector3_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Vector3_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector3_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Vector3_UnsignedChar`/`Const_Vector3_UnsignedChar` directly.
    public class _InOptMut_Vector3_UnsignedChar
    {
        public Vector3_UnsignedChar? Opt;

        public _InOptMut_Vector3_UnsignedChar() {}
        public _InOptMut_Vector3_UnsignedChar(Vector3_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_Vector3_UnsignedChar(Vector3_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `Vector3_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector3_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Vector3_UnsignedChar`/`Const_Vector3_UnsignedChar` to pass it to the function.
    public class _InOptConst_Vector3_UnsignedChar
    {
        public Const_Vector3_UnsignedChar? Opt;

        public _InOptConst_Vector3_UnsignedChar() {}
        public _InOptConst_Vector3_UnsignedChar(Const_Vector3_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_Vector3_UnsignedChar(Const_Vector3_UnsignedChar value) {return new(value);}
    }

    /// dot product
    /// Generated from function `MR::dot<float>`.
    public static unsafe float Dot(MR.Const_Vector3f a, MR.Const_Vector3f b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_float_MR_Vector3f", ExactSpelling = true)]
        extern static float __MR_dot_float_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
        return __MR_dot_float_MR_Vector3f(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<double>`.
    public static unsafe double Dot(MR.Const_Vector3d a, MR.Const_Vector3d b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_double_MR_Vector3d", ExactSpelling = true)]
        extern static double __MR_dot_double_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
        return __MR_dot_double_MR_Vector3d(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<bool>`.
    public static unsafe int Dot(MR.Const_Vector3b a, MR.Const_Vector3b b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_bool_MR_Vector3b", ExactSpelling = true)]
        extern static int __MR_dot_bool_MR_Vector3b(MR.Const_Vector3b._Underlying *a, MR.Const_Vector3b._Underlying *b);
        return __MR_dot_bool_MR_Vector3b(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<int>`.
    public static unsafe int Dot(MR.Const_Vector3i a, MR.Const_Vector3i b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_int_MR_Vector3i", ExactSpelling = true)]
        extern static int __MR_dot_int_MR_Vector3i(MR.Const_Vector3i._Underlying *a, MR.Const_Vector3i._Underlying *b);
        return __MR_dot_int_MR_Vector3i(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<MR_int64_t>`.
    public static unsafe long Dot(MR.Const_Vector3i64 a, MR.Const_Vector3i64 b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_int64_t_MR_Vector3i64", ExactSpelling = true)]
        extern static long __MR_dot_int64_t_MR_Vector3i64(MR.Const_Vector3i64._Underlying *a, MR.Const_Vector3i64._Underlying *b);
        return __MR_dot_int64_t_MR_Vector3i64(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<unsigned char>`.
    public static unsafe int Dot(MR.Const_Vector3_UnsignedChar a, MR.Const_Vector3_UnsignedChar b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_unsigned_char_MR_Vector3_unsigned_char", ExactSpelling = true)]
        extern static int __MR_dot_unsigned_char_MR_Vector3_unsigned_char(MR.Const_Vector3_UnsignedChar._Underlying *a, MR.Const_Vector3_UnsignedChar._Underlying *b);
        return __MR_dot_unsigned_char_MR_Vector3_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
    }
}
