public static partial class MR
{
    /// four-dimensional vector
    /// Generated from class `MR::Vector4b`.
    /// This is the const reference to the struct.
    public class Const_Vector4b : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector4b>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector4b UnderlyingStruct => ref *(Vector4b *)_UnderlyingPtr;

        internal unsafe Const_Vector4b(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector4b_Destroy(_Underlying *_this);
            __MR_Vector4b_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector4b() {Dispose(false);}

        public bool X => UnderlyingStruct.X;

        public bool Y => UnderlyingStruct.Y;

        public bool Z => UnderlyingStruct.Z;

        public bool W => UnderlyingStruct.W;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector4b_Get_elements();
                return *__MR_Vector4b_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector4b(Const_Vector4b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector4b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Vector4b _ctor_result = __MR_Vector4b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Vector4b::Vector4b`.
        public unsafe Const_Vector4b(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Vector4b _ctor_result = __MR_Vector4b_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Vector4b::Vector4b`.
        public unsafe Const_Vector4b(bool x, bool y, bool z, bool w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_Construct_4(byte x, byte y, byte z, byte w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Vector4b _ctor_result = __MR_Vector4b_Construct_4(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0, z ? (byte)1 : (byte)0, w ? (byte)1 : (byte)0);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::Vector4b::diagonal`.
        public static MR.Vector4b Diagonal(bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_diagonal", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_diagonal(byte a);
            return __MR_Vector4b_diagonal(a ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector4b::operator[]`.
        public unsafe bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_index_const", ExactSpelling = true)]
            extern static bool *__MR_Vector4b_index_const(_Underlying *_this, int e);
            return *__MR_Vector4b_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector4b::lengthSq`.
        public unsafe bool LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector4b_lengthSq(_Underlying *_this);
            return __MR_Vector4b_lengthSq(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Vector4b::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_length", ExactSpelling = true)]
            extern static double __MR_Vector4b_length(_Underlying *_this);
            return __MR_Vector4b_length(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_equal_MR_Vector4b(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector4b a, MR.Const_Vector4b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4b operator+(MR.Const_Vector4b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Const_Vector4b._Underlying *__MR_pos_MR_Vector4b(MR.Const_Vector4b._Underlying *a);
            return new(__MR_pos_MR_Vector4b(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Const_Vector4b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_neg_MR_Vector4b(MR.Const_Vector4b._Underlying *a);
            return __MR_neg_MR_Vector4b(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i operator+(MR.Const_Vector4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_add_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_add_MR_Vector4b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Const_Vector4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_sub_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_sub_MR_Vector4b(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(bool a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_bool_MR_Vector4b(byte a, MR.Const_Vector4b._Underlying *b);
            return __MR_mul_bool_MR_Vector4b(a ? (byte)1 : (byte)0, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Const_Vector4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Vector4b_bool(MR.Const_Vector4b._Underlying *b, byte a);
            return __MR_mul_MR_Vector4b_bool(b._UnderlyingPtr, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector4i operator/(Const_Vector4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Vector4i __MR_div_MR_Vector4b_bool(MR.Vector4b b, byte a);
            return __MR_div_MR_Vector4b_bool(b.UnderlyingStruct, a ? (byte)1 : (byte)0);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector4b? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector4b)
                return this == (MR.Const_Vector4b)other;
            return false;
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4b`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector4b : Const_Vector4b
    {
        /// Get the underlying struct.
        public unsafe new ref Vector4b UnderlyingStruct => ref *(Vector4b *)_UnderlyingPtr;

        internal unsafe Mut_Vector4b(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new bool X {get => UnderlyingStruct.X; set => UnderlyingStruct.X = value;}

        public new bool Y {get => UnderlyingStruct.Y; set => UnderlyingStruct.Y = value;}

        public new bool Z {get => UnderlyingStruct.Z; set => UnderlyingStruct.Z = value;}

        public new bool W {get => UnderlyingStruct.W; set => UnderlyingStruct.W = value;}

        /// Generated copy constructor.
        public unsafe Mut_Vector4b(Const_Vector4b _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector4b() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Vector4b _ctor_result = __MR_Vector4b_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Vector4b::Vector4b`.
        public unsafe Mut_Vector4b(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Vector4b _ctor_result = __MR_Vector4b_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Vector4b::Vector4b`.
        public unsafe Mut_Vector4b(bool x, bool y, bool z, bool w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_Construct_4(byte x, byte y, byte z, byte w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Vector4b _ctor_result = __MR_Vector4b_Construct_4(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0, z ? (byte)1 : (byte)0, w ? (byte)1 : (byte)0);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::Vector4b::operator[]`.
        public unsafe new ref bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_index", ExactSpelling = true)]
            extern static bool *__MR_Vector4b_index(_Underlying *_this, int e);
            return ref *__MR_Vector4b_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4b AddAssign(MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_add_assign_MR_Vector4b(_Underlying *a, MR.Const_Vector4b._Underlying *b);
            return new(__MR_add_assign_MR_Vector4b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4b SubAssign(MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_sub_assign_MR_Vector4b(_Underlying *a, MR.Const_Vector4b._Underlying *b);
            return new(__MR_sub_assign_MR_Vector4b(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_mul_assign_MR_Vector4b_bool(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Vector4b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_div_assign_MR_Vector4b_bool(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Vector4b_bool(_UnderlyingPtr, b ? (byte)1 : (byte)0), is_owning: false);
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4b`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct Vector4b
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector4b(Const_Vector4b other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector4b(Vector4b other) => new(new Mut_Vector4b((Mut_Vector4b._Underlying *)&other, is_owning: false));

        public bool X {get => __storage_X != 0; set => __storage_X = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(0)]
        byte __storage_X;

        public bool Y {get => __storage_Y != 0; set => __storage_Y = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(1)]
        byte __storage_Y;

        public bool Z {get => __storage_Z != 0; set => __storage_Z = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(2)]
        byte __storage_Z;

        public bool W {get => __storage_W != 0; set => __storage_W = value ? (byte)1 : (byte)0;}
        [System.Runtime.InteropServices.FieldOffset(3)]
        byte __storage_W;

        /// Generated copy constructor.
        public Vector4b(Vector4b _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector4b()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_DefaultConstruct();
            this = __MR_Vector4b_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4b::Vector4b`.
        public unsafe Vector4b(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector4b_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4b::Vector4b`.
        public unsafe Vector4b(bool x, bool y, bool z, bool w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_Construct_4(byte x, byte y, byte z, byte w);
            this = __MR_Vector4b_Construct_4(x ? (byte)1 : (byte)0, y ? (byte)1 : (byte)0, z ? (byte)1 : (byte)0, w ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector4b::diagonal`.
        public static MR.Vector4b Diagonal(bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_diagonal", ExactSpelling = true)]
            extern static MR.Vector4b __MR_Vector4b_diagonal(byte a);
            return __MR_Vector4b_diagonal(a ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Vector4b::operator[]`.
        public unsafe bool Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_index_const", ExactSpelling = true)]
            extern static bool *__MR_Vector4b_index_const(MR.Vector4b *_this, int e);
            fixed (MR.Vector4b *__ptr__this = &this)
            {
                return *__MR_Vector4b_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4b::operator[]`.
        public unsafe ref bool Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_index", ExactSpelling = true)]
            extern static bool *__MR_Vector4b_index(MR.Vector4b *_this, int e);
            fixed (MR.Vector4b *__ptr__this = &this)
            {
                return ref *__MR_Vector4b_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4b::lengthSq`.
        public unsafe bool LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector4b_lengthSq(MR.Vector4b *_this);
            fixed (MR.Vector4b *__ptr__this = &this)
            {
                return __MR_Vector4b_lengthSq(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::Vector4b::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4b_length", ExactSpelling = true)]
            extern static double __MR_Vector4b_length(MR.Vector4b *_this);
            fixed (MR.Vector4b *__ptr__this = &this)
            {
                return __MR_Vector4b_length(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector4b a, MR.Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4b", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_equal_MR_Vector4b((MR.Mut_Vector4b._Underlying *)&a, (MR.Mut_Vector4b._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector4b a, MR.Vector4b b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4b operator+(MR.Vector4b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Const_Vector4b._Underlying *__MR_pos_MR_Vector4b(MR.Const_Vector4b._Underlying *a);
            return new(__MR_pos_MR_Vector4b((MR.Mut_Vector4b._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Vector4b a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_neg_MR_Vector4b(MR.Const_Vector4b._Underlying *a);
            return __MR_neg_MR_Vector4b((MR.Mut_Vector4b._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i operator+(MR.Vector4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_add_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_add_MR_Vector4b((MR.Mut_Vector4b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Vector4b a, MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_sub_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
            return __MR_sub_MR_Vector4b((MR.Mut_Vector4b._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(bool a, MR.Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_bool_MR_Vector4b(byte a, MR.Const_Vector4b._Underlying *b);
            return __MR_mul_bool_MR_Vector4b(a ? (byte)1 : (byte)0, (MR.Mut_Vector4b._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Vector4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Vector4b_bool(MR.Const_Vector4b._Underlying *b, byte a);
            return __MR_mul_MR_Vector4b_bool((MR.Mut_Vector4b._Underlying *)&b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector4i operator/(MR.Vector4b b, bool a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Vector4i __MR_div_MR_Vector4b_bool(MR.Vector4b b, byte a);
            return __MR_div_MR_Vector4b_bool(b, a ? (byte)1 : (byte)0);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4b AddAssign(MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_add_assign_MR_Vector4b(MR.Vector4b *a, MR.Const_Vector4b._Underlying *b);
            fixed (MR.Vector4b *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector4b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4b SubAssign(MR.Const_Vector4b b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4b", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_sub_assign_MR_Vector4b(MR.Vector4b *a, MR.Const_Vector4b._Underlying *b);
            fixed (MR.Vector4b *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector4b(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4b MulAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_mul_assign_MR_Vector4b_bool(MR.Vector4b *a, byte b);
            fixed (MR.Vector4b *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector4b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4b DivAssign(bool b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4b_bool", ExactSpelling = true)]
            extern static MR.Mut_Vector4b._Underlying *__MR_div_assign_MR_Vector4b_bool(MR.Vector4b *a, byte b);
            fixed (MR.Vector4b *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector4b_bool(__ptr_a, b ? (byte)1 : (byte)0), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector4b b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector4b)
                return this == (MR.Vector4b)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector4b` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector4b`/`Const_Vector4b` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector4b
    {
        public readonly bool HasValue;
        internal readonly Vector4b Object;
        public Vector4b Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector4b() {HasValue = false;}
        public _InOpt_Vector4b(Vector4b new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector4b(Vector4b new_value) {return new(new_value);}
        public _InOpt_Vector4b(Const_Vector4b new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector4b(Const_Vector4b new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector4b` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector4b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4b`/`Const_Vector4b` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4b`.
    public class _InOptMut_Vector4b
    {
        public Mut_Vector4b? Opt;

        public _InOptMut_Vector4b() {}
        public _InOptMut_Vector4b(Mut_Vector4b value) {Opt = value;}
        public static implicit operator _InOptMut_Vector4b(Mut_Vector4b value) {return new(value);}
        public unsafe _InOptMut_Vector4b(ref Vector4b value)
        {
            fixed (Vector4b *value_ptr = &value)
            {
                Opt = new((Const_Vector4b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector4b` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector4b`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4b`/`Const_Vector4b` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4b`.
    public class _InOptConst_Vector4b
    {
        public Const_Vector4b? Opt;

        public _InOptConst_Vector4b() {}
        public _InOptConst_Vector4b(Const_Vector4b value) {Opt = value;}
        public static implicit operator _InOptConst_Vector4b(Const_Vector4b value) {return new(value);}
        public unsafe _InOptConst_Vector4b(ref readonly Vector4b value)
        {
            fixed (Vector4b *value_ptr = &value)
            {
                Opt = new((Const_Vector4b._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4i`.
    /// This is the const reference to the struct.
    public class Const_Vector4i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector4i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector4i UnderlyingStruct => ref *(Vector4i *)_UnderlyingPtr;

        internal unsafe Const_Vector4i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector4i_Destroy(_Underlying *_this);
            __MR_Vector4i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector4i() {Dispose(false);}

        public ref readonly int X => ref UnderlyingStruct.X;

        public ref readonly int Y => ref UnderlyingStruct.Y;

        public ref readonly int Z => ref UnderlyingStruct.Z;

        public ref readonly int W => ref UnderlyingStruct.W;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector4i_Get_elements();
                return *__MR_Vector4i_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector4i(Const_Vector4i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector4i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4i _ctor_result = __MR_Vector4i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4i::Vector4i`.
        public unsafe Const_Vector4i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4i _ctor_result = __MR_Vector4i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4i::Vector4i`.
        public unsafe Const_Vector4i(int x, int y, int z, int w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_Construct_4(int x, int y, int z, int w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4i _ctor_result = __MR_Vector4i_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector4i::diagonal`.
        public static MR.Vector4i Diagonal(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_diagonal", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_diagonal(int a);
            return __MR_Vector4i_diagonal(a);
        }

        /// Generated from method `MR::Vector4i::operator[]`.
        public unsafe int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_index_const", ExactSpelling = true)]
            extern static int *__MR_Vector4i_index_const(_Underlying *_this, int e);
            return *__MR_Vector4i_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector4i::lengthSq`.
        public unsafe int LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_lengthSq", ExactSpelling = true)]
            extern static int __MR_Vector4i_lengthSq(_Underlying *_this);
            return __MR_Vector4i_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4i::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_length", ExactSpelling = true)]
            extern static double __MR_Vector4i_length(_Underlying *_this);
            return __MR_Vector4i_length(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_equal_MR_Vector4i(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector4i a, MR.Const_Vector4i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4i operator+(MR.Const_Vector4i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Const_Vector4i._Underlying *__MR_pos_MR_Vector4i(MR.Const_Vector4i._Underlying *a);
            return new(__MR_pos_MR_Vector4i(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Const_Vector4i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_neg_MR_Vector4i(MR.Const_Vector4i._Underlying *a);
            return __MR_neg_MR_Vector4i(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i operator+(MR.Const_Vector4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_add_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_add_MR_Vector4i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Const_Vector4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_sub_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_sub_MR_Vector4i(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(int a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_int_MR_Vector4i(int a, MR.Const_Vector4i._Underlying *b);
            return __MR_mul_int_MR_Vector4i(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Const_Vector4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Vector4i_int(MR.Const_Vector4i._Underlying *b, int a);
            return __MR_mul_MR_Vector4i_int(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector4i operator/(Const_Vector4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Vector4i __MR_div_MR_Vector4i_int(MR.Vector4i b, int a);
            return __MR_div_MR_Vector4i_int(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector4i? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector4i)
                return this == (MR.Const_Vector4i)other;
            return false;
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4i`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector4i : Const_Vector4i
    {
        /// Get the underlying struct.
        public unsafe new ref Vector4i UnderlyingStruct => ref *(Vector4i *)_UnderlyingPtr;

        internal unsafe Mut_Vector4i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int X => ref UnderlyingStruct.X;

        public new ref int Y => ref UnderlyingStruct.Y;

        public new ref int Z => ref UnderlyingStruct.Z;

        public new ref int W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Vector4i(Const_Vector4i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector4i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4i _ctor_result = __MR_Vector4i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4i::Vector4i`.
        public unsafe Mut_Vector4i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4i _ctor_result = __MR_Vector4i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4i::Vector4i`.
        public unsafe Mut_Vector4i(int x, int y, int z, int w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_Construct_4(int x, int y, int z, int w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4i _ctor_result = __MR_Vector4i_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector4i::operator[]`.
        public unsafe new ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_index", ExactSpelling = true)]
            extern static int *__MR_Vector4i_index(_Underlying *_this, int e);
            return ref *__MR_Vector4i_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4i AddAssign(MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_add_assign_MR_Vector4i(_Underlying *a, MR.Const_Vector4i._Underlying *b);
            return new(__MR_add_assign_MR_Vector4i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4i SubAssign(MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_sub_assign_MR_Vector4i(_Underlying *a, MR.Const_Vector4i._Underlying *b);
            return new(__MR_sub_assign_MR_Vector4i(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_mul_assign_MR_Vector4i_int(_Underlying *a, int b);
            return new(__MR_mul_assign_MR_Vector4i_int(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_div_assign_MR_Vector4i_int(_Underlying *a, int b);
            return new(__MR_div_assign_MR_Vector4i_int(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Vector4i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector4i(Const_Vector4i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector4i(Vector4i other) => new(new Mut_Vector4i((Mut_Vector4i._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public int Y;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public int Z;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public int W;

        /// Generated copy constructor.
        public Vector4i(Vector4i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector4i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_DefaultConstruct();
            this = __MR_Vector4i_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4i::Vector4i`.
        public unsafe Vector4i(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector4i_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4i::Vector4i`.
        public unsafe Vector4i(int x, int y, int z, int w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_Construct_4(int x, int y, int z, int w);
            this = __MR_Vector4i_Construct_4(x, y, z, w);
        }

        /// Generated from method `MR::Vector4i::diagonal`.
        public static MR.Vector4i Diagonal(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_diagonal", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Vector4i_diagonal(int a);
            return __MR_Vector4i_diagonal(a);
        }

        /// Generated from method `MR::Vector4i::operator[]`.
        public unsafe int Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_index_const", ExactSpelling = true)]
            extern static int *__MR_Vector4i_index_const(MR.Vector4i *_this, int e);
            fixed (MR.Vector4i *__ptr__this = &this)
            {
                return *__MR_Vector4i_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4i::operator[]`.
        public unsafe ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_index", ExactSpelling = true)]
            extern static int *__MR_Vector4i_index(MR.Vector4i *_this, int e);
            fixed (MR.Vector4i *__ptr__this = &this)
            {
                return ref *__MR_Vector4i_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4i::lengthSq`.
        public unsafe int LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_lengthSq", ExactSpelling = true)]
            extern static int __MR_Vector4i_lengthSq(MR.Vector4i *_this);
            fixed (MR.Vector4i *__ptr__this = &this)
            {
                return __MR_Vector4i_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4i::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i_length", ExactSpelling = true)]
            extern static double __MR_Vector4i_length(MR.Vector4i *_this);
            fixed (MR.Vector4i *__ptr__this = &this)
            {
                return __MR_Vector4i_length(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector4i a, MR.Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_equal_MR_Vector4i((MR.Mut_Vector4i._Underlying *)&a, (MR.Mut_Vector4i._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector4i a, MR.Vector4i b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4i operator+(MR.Vector4i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Const_Vector4i._Underlying *__MR_pos_MR_Vector4i(MR.Const_Vector4i._Underlying *a);
            return new(__MR_pos_MR_Vector4i((MR.Mut_Vector4i._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Vector4i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_neg_MR_Vector4i(MR.Const_Vector4i._Underlying *a);
            return __MR_neg_MR_Vector4i((MR.Mut_Vector4i._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i operator+(MR.Vector4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_add_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_add_MR_Vector4i((MR.Mut_Vector4i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Vector4i a, MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_sub_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
            return __MR_sub_MR_Vector4i((MR.Mut_Vector4i._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(int a, MR.Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_int_MR_Vector4i(int a, MR.Const_Vector4i._Underlying *b);
            return __MR_mul_int_MR_Vector4i(a, (MR.Mut_Vector4i._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Vector4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Vector4i_int(MR.Const_Vector4i._Underlying *b, int a);
            return __MR_mul_MR_Vector4i_int((MR.Mut_Vector4i._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector4i operator/(MR.Vector4i b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Vector4i __MR_div_MR_Vector4i_int(MR.Vector4i b, int a);
            return __MR_div_MR_Vector4i_int(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4i AddAssign(MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_add_assign_MR_Vector4i(MR.Vector4i *a, MR.Const_Vector4i._Underlying *b);
            fixed (MR.Vector4i *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector4i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4i SubAssign(MR.Const_Vector4i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_sub_assign_MR_Vector4i(MR.Vector4i *a, MR.Const_Vector4i._Underlying *b);
            fixed (MR.Vector4i *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector4i(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4i MulAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_mul_assign_MR_Vector4i_int(MR.Vector4i *a, int b);
            fixed (MR.Vector4i *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector4i_int(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4i DivAssign(int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4i_int", ExactSpelling = true)]
            extern static MR.Mut_Vector4i._Underlying *__MR_div_assign_MR_Vector4i_int(MR.Vector4i *a, int b);
            fixed (MR.Vector4i *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector4i_int(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector4i b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector4i)
                return this == (MR.Vector4i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector4i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector4i`/`Const_Vector4i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector4i
    {
        public readonly bool HasValue;
        internal readonly Vector4i Object;
        public Vector4i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector4i() {HasValue = false;}
        public _InOpt_Vector4i(Vector4i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector4i(Vector4i new_value) {return new(new_value);}
        public _InOpt_Vector4i(Const_Vector4i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector4i(Const_Vector4i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector4i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector4i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4i`/`Const_Vector4i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4i`.
    public class _InOptMut_Vector4i
    {
        public Mut_Vector4i? Opt;

        public _InOptMut_Vector4i() {}
        public _InOptMut_Vector4i(Mut_Vector4i value) {Opt = value;}
        public static implicit operator _InOptMut_Vector4i(Mut_Vector4i value) {return new(value);}
        public unsafe _InOptMut_Vector4i(ref Vector4i value)
        {
            fixed (Vector4i *value_ptr = &value)
            {
                Opt = new((Const_Vector4i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector4i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector4i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4i`/`Const_Vector4i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4i`.
    public class _InOptConst_Vector4i
    {
        public Const_Vector4i? Opt;

        public _InOptConst_Vector4i() {}
        public _InOptConst_Vector4i(Const_Vector4i value) {Opt = value;}
        public static implicit operator _InOptConst_Vector4i(Const_Vector4i value) {return new(value);}
        public unsafe _InOptConst_Vector4i(ref readonly Vector4i value)
        {
            fixed (Vector4i *value_ptr = &value)
            {
                Opt = new((Const_Vector4i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4i64`.
    /// This is the const reference to the struct.
    public class Const_Vector4i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector4i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector4i64 UnderlyingStruct => ref *(Vector4i64 *)_UnderlyingPtr;

        internal unsafe Const_Vector4i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector4i64_Destroy(_Underlying *_this);
            __MR_Vector4i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector4i64() {Dispose(false);}

        public ref readonly long X => ref UnderlyingStruct.X;

        public ref readonly long Y => ref UnderlyingStruct.Y;

        public ref readonly long Z => ref UnderlyingStruct.Z;

        public ref readonly long W => ref UnderlyingStruct.W;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector4i64_Get_elements();
                return *__MR_Vector4i64_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector4i64(Const_Vector4i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector4i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4i64 _ctor_result = __MR_Vector4i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4i64::Vector4i64`.
        public unsafe Const_Vector4i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4i64 _ctor_result = __MR_Vector4i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4i64::Vector4i64`.
        public unsafe Const_Vector4i64(long x, long y, long z, long w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_Construct_4(long x, long y, long z, long w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4i64 _ctor_result = __MR_Vector4i64_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Vector4i64::diagonal`.
        public static MR.Vector4i64 Diagonal(long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_diagonal", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_diagonal(long a);
            return __MR_Vector4i64_diagonal(a);
        }

        /// Generated from method `MR::Vector4i64::operator[]`.
        public unsafe long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Vector4i64_index_const(_Underlying *_this, int e);
            return *__MR_Vector4i64_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector4i64::lengthSq`.
        public unsafe long LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_lengthSq", ExactSpelling = true)]
            extern static long __MR_Vector4i64_lengthSq(_Underlying *_this);
            return __MR_Vector4i64_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4i64::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_length", ExactSpelling = true)]
            extern static double __MR_Vector4i64_length(_Underlying *_this);
            return __MR_Vector4i64_length(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_equal_MR_Vector4i64(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector4i64 a, MR.Const_Vector4i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4i64 operator+(MR.Const_Vector4i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Const_Vector4i64._Underlying *__MR_pos_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a);
            return new(__MR_pos_MR_Vector4i64(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i64 operator-(MR.Const_Vector4i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_neg_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a);
            return __MR_neg_MR_Vector4i64(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i64 operator+(MR.Const_Vector4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_add_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_add_MR_Vector4i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i64 operator-(MR.Const_Vector4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_sub_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_sub_MR_Vector4i64(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i64 operator*(long a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_mul_int64_t_MR_Vector4i64(long a, MR.Const_Vector4i64._Underlying *b);
            return __MR_mul_int64_t_MR_Vector4i64(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i64 operator*(MR.Const_Vector4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_mul_MR_Vector4i64_int64_t(MR.Const_Vector4i64._Underlying *b, long a);
            return __MR_mul_MR_Vector4i64_int64_t(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector4i64 operator/(Const_Vector4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_div_MR_Vector4i64_int64_t(MR.Vector4i64 b, long a);
            return __MR_div_MR_Vector4i64_int64_t(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector4i64? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector4i64)
                return this == (MR.Const_Vector4i64)other;
            return false;
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector4i64 : Const_Vector4i64
    {
        /// Get the underlying struct.
        public unsafe new ref Vector4i64 UnderlyingStruct => ref *(Vector4i64 *)_UnderlyingPtr;

        internal unsafe Mut_Vector4i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref long X => ref UnderlyingStruct.X;

        public new ref long Y => ref UnderlyingStruct.Y;

        public new ref long Z => ref UnderlyingStruct.Z;

        public new ref long W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Vector4i64(Const_Vector4i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector4i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4i64 _ctor_result = __MR_Vector4i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4i64::Vector4i64`.
        public unsafe Mut_Vector4i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4i64 _ctor_result = __MR_Vector4i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4i64::Vector4i64`.
        public unsafe Mut_Vector4i64(long x, long y, long z, long w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_Construct_4(long x, long y, long z, long w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4i64 _ctor_result = __MR_Vector4i64_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Vector4i64::operator[]`.
        public unsafe new ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_index", ExactSpelling = true)]
            extern static long *__MR_Vector4i64_index(_Underlying *_this, int e);
            return ref *__MR_Vector4i64_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4i64 AddAssign(MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_add_assign_MR_Vector4i64(_Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return new(__MR_add_assign_MR_Vector4i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4i64 SubAssign(MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_sub_assign_MR_Vector4i64(_Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return new(__MR_sub_assign_MR_Vector4i64(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_mul_assign_MR_Vector4i64_int64_t(_Underlying *a, long b);
            return new(__MR_mul_assign_MR_Vector4i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_div_assign_MR_Vector4i64_int64_t(_Underlying *a, long b);
            return new(__MR_div_assign_MR_Vector4i64_int64_t(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
    public struct Vector4i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector4i64(Const_Vector4i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector4i64(Vector4i64 other) => new(new Mut_Vector4i64((Mut_Vector4i64._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public long X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public long Y;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public long Z;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public long W;

        /// Generated copy constructor.
        public Vector4i64(Vector4i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector4i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_DefaultConstruct();
            this = __MR_Vector4i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4i64::Vector4i64`.
        public unsafe Vector4i64(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector4i64_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4i64::Vector4i64`.
        public unsafe Vector4i64(long x, long y, long z, long w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_Construct_4(long x, long y, long z, long w);
            this = __MR_Vector4i64_Construct_4(x, y, z, w);
        }

        /// Generated from method `MR::Vector4i64::diagonal`.
        public static MR.Vector4i64 Diagonal(long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_diagonal", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_Vector4i64_diagonal(long a);
            return __MR_Vector4i64_diagonal(a);
        }

        /// Generated from method `MR::Vector4i64::operator[]`.
        public unsafe long Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Vector4i64_index_const(MR.Vector4i64 *_this, int e);
            fixed (MR.Vector4i64 *__ptr__this = &this)
            {
                return *__MR_Vector4i64_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4i64::operator[]`.
        public unsafe ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_index", ExactSpelling = true)]
            extern static long *__MR_Vector4i64_index(MR.Vector4i64 *_this, int e);
            fixed (MR.Vector4i64 *__ptr__this = &this)
            {
                return ref *__MR_Vector4i64_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4i64::lengthSq`.
        public unsafe long LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_lengthSq", ExactSpelling = true)]
            extern static long __MR_Vector4i64_lengthSq(MR.Vector4i64 *_this);
            fixed (MR.Vector4i64 *__ptr__this = &this)
            {
                return __MR_Vector4i64_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4i64::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4i64_length", ExactSpelling = true)]
            extern static double __MR_Vector4i64_length(MR.Vector4i64 *_this);
            fixed (MR.Vector4i64 *__ptr__this = &this)
            {
                return __MR_Vector4i64_length(__ptr__this);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector4i64 a, MR.Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_equal_MR_Vector4i64((MR.Mut_Vector4i64._Underlying *)&a, (MR.Mut_Vector4i64._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector4i64 a, MR.Vector4i64 b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4i64 operator+(MR.Vector4i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Const_Vector4i64._Underlying *__MR_pos_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a);
            return new(__MR_pos_MR_Vector4i64((MR.Mut_Vector4i64._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i64 operator-(MR.Vector4i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_neg_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a);
            return __MR_neg_MR_Vector4i64((MR.Mut_Vector4i64._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i64 operator+(MR.Vector4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_add_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_add_MR_Vector4i64((MR.Mut_Vector4i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i64 operator-(MR.Vector4i64 a, MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_sub_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
            return __MR_sub_MR_Vector4i64((MR.Mut_Vector4i64._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i64 operator*(long a, MR.Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_int64_t_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_mul_int64_t_MR_Vector4i64(long a, MR.Const_Vector4i64._Underlying *b);
            return __MR_mul_int64_t_MR_Vector4i64(a, (MR.Mut_Vector4i64._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i64 operator*(MR.Vector4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_mul_MR_Vector4i64_int64_t(MR.Const_Vector4i64._Underlying *b, long a);
            return __MR_mul_MR_Vector4i64_int64_t((MR.Mut_Vector4i64._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector4i64 operator/(MR.Vector4i64 b, long a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Vector4i64 __MR_div_MR_Vector4i64_int64_t(MR.Vector4i64 b, long a);
            return __MR_div_MR_Vector4i64_int64_t(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4i64 AddAssign(MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_add_assign_MR_Vector4i64(MR.Vector4i64 *a, MR.Const_Vector4i64._Underlying *b);
            fixed (MR.Vector4i64 *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector4i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4i64 SubAssign(MR.Const_Vector4i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4i64", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_sub_assign_MR_Vector4i64(MR.Vector4i64 *a, MR.Const_Vector4i64._Underlying *b);
            fixed (MR.Vector4i64 *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector4i64(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4i64 MulAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_mul_assign_MR_Vector4i64_int64_t(MR.Vector4i64 *a, long b);
            fixed (MR.Vector4i64 *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector4i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4i64 DivAssign(long b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4i64_int64_t", ExactSpelling = true)]
            extern static MR.Mut_Vector4i64._Underlying *__MR_div_assign_MR_Vector4i64_int64_t(MR.Vector4i64 *a, long b);
            fixed (MR.Vector4i64 *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector4i64_int64_t(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector4i64 b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector4i64)
                return this == (MR.Vector4i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector4i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector4i64`/`Const_Vector4i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector4i64
    {
        public readonly bool HasValue;
        internal readonly Vector4i64 Object;
        public Vector4i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector4i64() {HasValue = false;}
        public _InOpt_Vector4i64(Vector4i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector4i64(Vector4i64 new_value) {return new(new_value);}
        public _InOpt_Vector4i64(Const_Vector4i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector4i64(Const_Vector4i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector4i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector4i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4i64`/`Const_Vector4i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4i64`.
    public class _InOptMut_Vector4i64
    {
        public Mut_Vector4i64? Opt;

        public _InOptMut_Vector4i64() {}
        public _InOptMut_Vector4i64(Mut_Vector4i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Vector4i64(Mut_Vector4i64 value) {return new(value);}
        public unsafe _InOptMut_Vector4i64(ref Vector4i64 value)
        {
            fixed (Vector4i64 *value_ptr = &value)
            {
                Opt = new((Const_Vector4i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector4i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector4i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4i64`/`Const_Vector4i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4i64`.
    public class _InOptConst_Vector4i64
    {
        public Const_Vector4i64? Opt;

        public _InOptConst_Vector4i64() {}
        public _InOptConst_Vector4i64(Const_Vector4i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Vector4i64(Const_Vector4i64 value) {return new(value);}
        public unsafe _InOptConst_Vector4i64(ref readonly Vector4i64 value)
        {
            fixed (Vector4i64 *value_ptr = &value)
            {
                Opt = new((Const_Vector4i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4f`.
    /// This is the const reference to the struct.
    public class Const_Vector4f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector4f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector4f UnderlyingStruct => ref *(Vector4f *)_UnderlyingPtr;

        internal unsafe Const_Vector4f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector4f_Destroy(_Underlying *_this);
            __MR_Vector4f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector4f() {Dispose(false);}

        public ref readonly float X => ref UnderlyingStruct.X;

        public ref readonly float Y => ref UnderlyingStruct.Y;

        public ref readonly float Z => ref UnderlyingStruct.Z;

        public ref readonly float W => ref UnderlyingStruct.W;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector4f_Get_elements();
                return *__MR_Vector4f_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector4f(Const_Vector4f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector4f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4f _ctor_result = __MR_Vector4f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4f::Vector4f`.
        public unsafe Const_Vector4f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4f _ctor_result = __MR_Vector4f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4f::Vector4f`.
        public unsafe Const_Vector4f(float x, float y, float z, float w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_Construct_4(float x, float y, float z, float w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4f _ctor_result = __MR_Vector4f_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector4f::diagonal`.
        public static MR.Vector4f Diagonal(float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_diagonal", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_diagonal(float a);
            return __MR_Vector4f_diagonal(a);
        }

        /// Generated from method `MR::Vector4f::operator[]`.
        public unsafe float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_index_const", ExactSpelling = true)]
            extern static float *__MR_Vector4f_index_const(_Underlying *_this, int e);
            return *__MR_Vector4f_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector4f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_lengthSq", ExactSpelling = true)]
            extern static float __MR_Vector4f_lengthSq(_Underlying *_this);
            return __MR_Vector4f_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_length", ExactSpelling = true)]
            extern static float __MR_Vector4f_length(_Underlying *_this);
            return __MR_Vector4f_length(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4f::normalized`.
        public unsafe MR.Vector4f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_normalized", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_normalized(_Underlying *_this);
            return __MR_Vector4f_normalized(_UnderlyingPtr);
        }

        /// assuming this is a point represented in homogeneous 4D coordinates, returns the point as 3D-vector
        /// Generated from method `MR::Vector4f::proj3d`.
        public unsafe MR.Vector3f Proj3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_proj3d", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector4f_proj3d(_Underlying *_this);
            return __MR_Vector4f_proj3d(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4f::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector4f_isFinite(_Underlying *_this);
            return __MR_Vector4f_isFinite(_UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_equal_MR_Vector4f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector4f a, MR.Const_Vector4f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4f operator+(MR.Const_Vector4f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Const_Vector4f._Underlying *__MR_pos_MR_Vector4f(MR.Const_Vector4f._Underlying *a);
            return new(__MR_pos_MR_Vector4f(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4f operator-(MR.Const_Vector4f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_neg_MR_Vector4f(MR.Const_Vector4f._Underlying *a);
            return __MR_neg_MR_Vector4f(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4f operator+(MR.Const_Vector4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_add_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_add_MR_Vector4f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4f operator-(MR.Const_Vector4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_sub_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_sub_MR_Vector4f(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4f operator*(float a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_mul_float_MR_Vector4f(float a, MR.Const_Vector4f._Underlying *b);
            return __MR_mul_float_MR_Vector4f(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4f operator*(MR.Const_Vector4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Vector4f __MR_mul_MR_Vector4f_float(MR.Const_Vector4f._Underlying *b, float a);
            return __MR_mul_MR_Vector4f_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector4f operator/(Const_Vector4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Vector4f __MR_div_MR_Vector4f_float(MR.Vector4f b, float a);
            return __MR_div_MR_Vector4f_float(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector4f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector4f)
                return this == (MR.Const_Vector4f)other;
            return false;
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4f`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector4f : Const_Vector4f
    {
        /// Get the underlying struct.
        public unsafe new ref Vector4f UnderlyingStruct => ref *(Vector4f *)_UnderlyingPtr;

        internal unsafe Mut_Vector4f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref float X => ref UnderlyingStruct.X;

        public new ref float Y => ref UnderlyingStruct.Y;

        public new ref float Z => ref UnderlyingStruct.Z;

        public new ref float W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Vector4f(Const_Vector4f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector4f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4f _ctor_result = __MR_Vector4f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4f::Vector4f`.
        public unsafe Mut_Vector4f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4f _ctor_result = __MR_Vector4f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Vector4f::Vector4f`.
        public unsafe Mut_Vector4f(float x, float y, float z, float w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_Construct_4(float x, float y, float z, float w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Vector4f _ctor_result = __MR_Vector4f_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Vector4f::operator[]`.
        public unsafe new ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_index", ExactSpelling = true)]
            extern static float *__MR_Vector4f_index(_Underlying *_this, int e);
            return ref *__MR_Vector4f_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4f AddAssign(MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_add_assign_MR_Vector4f(_Underlying *a, MR.Const_Vector4f._Underlying *b);
            return new(__MR_add_assign_MR_Vector4f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4f SubAssign(MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_sub_assign_MR_Vector4f(_Underlying *a, MR.Const_Vector4f._Underlying *b);
            return new(__MR_sub_assign_MR_Vector4f(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_mul_assign_MR_Vector4f_float(_Underlying *a, float b);
            return new(__MR_mul_assign_MR_Vector4f_float(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_div_assign_MR_Vector4f_float(_Underlying *a, float b);
            return new(__MR_div_assign_MR_Vector4f_float(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Vector4f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector4f(Const_Vector4f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector4f(Vector4f other) => new(new Mut_Vector4f((Mut_Vector4f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public float X;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public float Y;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public float Z;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public float W;

        /// Generated copy constructor.
        public Vector4f(Vector4f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector4f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_DefaultConstruct();
            this = __MR_Vector4f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4f::Vector4f`.
        public unsafe Vector4f(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector4f_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4f::Vector4f`.
        public unsafe Vector4f(float x, float y, float z, float w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_Construct_4(float x, float y, float z, float w);
            this = __MR_Vector4f_Construct_4(x, y, z, w);
        }

        /// Generated from method `MR::Vector4f::diagonal`.
        public static MR.Vector4f Diagonal(float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_diagonal", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_diagonal(float a);
            return __MR_Vector4f_diagonal(a);
        }

        /// Generated from method `MR::Vector4f::operator[]`.
        public unsafe float Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_index_const", ExactSpelling = true)]
            extern static float *__MR_Vector4f_index_const(MR.Vector4f *_this, int e);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return *__MR_Vector4f_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4f::operator[]`.
        public unsafe ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_index", ExactSpelling = true)]
            extern static float *__MR_Vector4f_index(MR.Vector4f *_this, int e);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return ref *__MR_Vector4f_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_lengthSq", ExactSpelling = true)]
            extern static float __MR_Vector4f_lengthSq(MR.Vector4f *_this);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return __MR_Vector4f_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_length", ExactSpelling = true)]
            extern static float __MR_Vector4f_length(MR.Vector4f *_this);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return __MR_Vector4f_length(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4f::normalized`.
        public unsafe MR.Vector4f Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_normalized", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Vector4f_normalized(MR.Vector4f *_this);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return __MR_Vector4f_normalized(__ptr__this);
            }
        }

        /// assuming this is a point represented in homogeneous 4D coordinates, returns the point as 3D-vector
        /// Generated from method `MR::Vector4f::proj3d`.
        public unsafe MR.Vector3f Proj3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_proj3d", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Vector4f_proj3d(MR.Vector4f *_this);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return __MR_Vector4f_proj3d(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4f::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4f_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector4f_isFinite(MR.Vector4f *_this);
            fixed (MR.Vector4f *__ptr__this = &this)
            {
                return __MR_Vector4f_isFinite(__ptr__this) != 0;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector4f a, MR.Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_equal_MR_Vector4f((MR.Mut_Vector4f._Underlying *)&a, (MR.Mut_Vector4f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector4f a, MR.Vector4f b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4f operator+(MR.Vector4f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Const_Vector4f._Underlying *__MR_pos_MR_Vector4f(MR.Const_Vector4f._Underlying *a);
            return new(__MR_pos_MR_Vector4f((MR.Mut_Vector4f._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4f operator-(MR.Vector4f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_neg_MR_Vector4f(MR.Const_Vector4f._Underlying *a);
            return __MR_neg_MR_Vector4f((MR.Mut_Vector4f._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4f operator+(MR.Vector4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_add_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_add_MR_Vector4f((MR.Mut_Vector4f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4f operator-(MR.Vector4f a, MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_sub_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
            return __MR_sub_MR_Vector4f((MR.Mut_Vector4f._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4f operator*(float a, MR.Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_mul_float_MR_Vector4f(float a, MR.Const_Vector4f._Underlying *b);
            return __MR_mul_float_MR_Vector4f(a, (MR.Mut_Vector4f._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4f operator*(MR.Vector4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Vector4f __MR_mul_MR_Vector4f_float(MR.Const_Vector4f._Underlying *b, float a);
            return __MR_mul_MR_Vector4f_float((MR.Mut_Vector4f._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector4f operator/(MR.Vector4f b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Vector4f __MR_div_MR_Vector4f_float(MR.Vector4f b, float a);
            return __MR_div_MR_Vector4f_float(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4f AddAssign(MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_add_assign_MR_Vector4f(MR.Vector4f *a, MR.Const_Vector4f._Underlying *b);
            fixed (MR.Vector4f *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector4f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4f SubAssign(MR.Const_Vector4f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_sub_assign_MR_Vector4f(MR.Vector4f *a, MR.Const_Vector4f._Underlying *b);
            fixed (MR.Vector4f *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector4f(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4f MulAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_mul_assign_MR_Vector4f_float(MR.Vector4f *a, float b);
            fixed (MR.Vector4f *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector4f_float(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4f DivAssign(float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4f_float", ExactSpelling = true)]
            extern static MR.Mut_Vector4f._Underlying *__MR_div_assign_MR_Vector4f_float(MR.Vector4f *a, float b);
            fixed (MR.Vector4f *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector4f_float(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector4f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector4f)
                return this == (MR.Vector4f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector4f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector4f`/`Const_Vector4f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector4f
    {
        public readonly bool HasValue;
        internal readonly Vector4f Object;
        public Vector4f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector4f() {HasValue = false;}
        public _InOpt_Vector4f(Vector4f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector4f(Vector4f new_value) {return new(new_value);}
        public _InOpt_Vector4f(Const_Vector4f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector4f(Const_Vector4f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector4f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector4f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4f`/`Const_Vector4f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4f`.
    public class _InOptMut_Vector4f
    {
        public Mut_Vector4f? Opt;

        public _InOptMut_Vector4f() {}
        public _InOptMut_Vector4f(Mut_Vector4f value) {Opt = value;}
        public static implicit operator _InOptMut_Vector4f(Mut_Vector4f value) {return new(value);}
        public unsafe _InOptMut_Vector4f(ref Vector4f value)
        {
            fixed (Vector4f *value_ptr = &value)
            {
                Opt = new((Const_Vector4f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector4f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector4f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4f`/`Const_Vector4f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4f`.
    public class _InOptConst_Vector4f
    {
        public Const_Vector4f? Opt;

        public _InOptConst_Vector4f() {}
        public _InOptConst_Vector4f(Const_Vector4f value) {Opt = value;}
        public static implicit operator _InOptConst_Vector4f(Const_Vector4f value) {return new(value);}
        public unsafe _InOptConst_Vector4f(ref readonly Vector4f value)
        {
            fixed (Vector4f *value_ptr = &value)
            {
                Opt = new((Const_Vector4f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4d`.
    /// This is the const reference to the struct.
    public class Const_Vector4d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector4d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Vector4d UnderlyingStruct => ref *(Vector4d *)_UnderlyingPtr;

        internal unsafe Const_Vector4d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector4d_Destroy(_Underlying *_this);
            __MR_Vector4d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector4d() {Dispose(false);}

        public ref readonly double X => ref UnderlyingStruct.X;

        public ref readonly double Y => ref UnderlyingStruct.Y;

        public ref readonly double Z => ref UnderlyingStruct.Z;

        public ref readonly double W => ref UnderlyingStruct.W;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector4d_Get_elements();
                return *__MR_Vector4d_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Vector4d(Const_Vector4d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector4d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4d _ctor_result = __MR_Vector4d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4d::Vector4d`.
        public unsafe Const_Vector4d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4d _ctor_result = __MR_Vector4d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4d::Vector4d`.
        public unsafe Const_Vector4d(double x, double y, double z, double w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_Construct_4(double x, double y, double z, double w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4d _ctor_result = __MR_Vector4d_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Vector4d::diagonal`.
        public static MR.Vector4d Diagonal(double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_diagonal", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_diagonal(double a);
            return __MR_Vector4d_diagonal(a);
        }

        /// Generated from method `MR::Vector4d::operator[]`.
        public unsafe double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_index_const", ExactSpelling = true)]
            extern static double *__MR_Vector4d_index_const(_Underlying *_this, int e);
            return *__MR_Vector4d_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector4d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_lengthSq", ExactSpelling = true)]
            extern static double __MR_Vector4d_lengthSq(_Underlying *_this);
            return __MR_Vector4d_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_length", ExactSpelling = true)]
            extern static double __MR_Vector4d_length(_Underlying *_this);
            return __MR_Vector4d_length(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4d::normalized`.
        public unsafe MR.Vector4d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_normalized", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_normalized(_Underlying *_this);
            return __MR_Vector4d_normalized(_UnderlyingPtr);
        }

        /// assuming this is a point represented in homogeneous 4D coordinates, returns the point as 3D-vector
        /// Generated from method `MR::Vector4d::proj3d`.
        public unsafe MR.Vector3d Proj3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_proj3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector4d_proj3d(_Underlying *_this);
            return __MR_Vector4d_proj3d(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4d::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector4d_isFinite(_Underlying *_this);
            return __MR_Vector4d_isFinite(_UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_equal_MR_Vector4d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector4d a, MR.Const_Vector4d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4d operator+(MR.Const_Vector4d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Const_Vector4d._Underlying *__MR_pos_MR_Vector4d(MR.Const_Vector4d._Underlying *a);
            return new(__MR_pos_MR_Vector4d(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4d operator-(MR.Const_Vector4d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_neg_MR_Vector4d(MR.Const_Vector4d._Underlying *a);
            return __MR_neg_MR_Vector4d(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4d operator+(MR.Const_Vector4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_add_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_add_MR_Vector4d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4d operator-(MR.Const_Vector4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_sub_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_sub_MR_Vector4d(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4d operator*(double a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_mul_double_MR_Vector4d(double a, MR.Const_Vector4d._Underlying *b);
            return __MR_mul_double_MR_Vector4d(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4d operator*(MR.Const_Vector4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Vector4d __MR_mul_MR_Vector4d_double(MR.Const_Vector4d._Underlying *b, double a);
            return __MR_mul_MR_Vector4d_double(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector4d operator/(Const_Vector4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Vector4d __MR_div_MR_Vector4d_double(MR.Vector4d b, double a);
            return __MR_div_MR_Vector4d_double(b.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector4d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector4d)
                return this == (MR.Const_Vector4d)other;
            return false;
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4d`.
    /// This is the non-const reference to the struct.
    public class Mut_Vector4d : Const_Vector4d
    {
        /// Get the underlying struct.
        public unsafe new ref Vector4d UnderlyingStruct => ref *(Vector4d *)_UnderlyingPtr;

        internal unsafe Mut_Vector4d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref double X => ref UnderlyingStruct.X;

        public new ref double Y => ref UnderlyingStruct.Y;

        public new ref double Z => ref UnderlyingStruct.Z;

        public new ref double W => ref UnderlyingStruct.W;

        /// Generated copy constructor.
        public unsafe Mut_Vector4d(Const_Vector4d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Vector4d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4d _ctor_result = __MR_Vector4d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4d::Vector4d`.
        public unsafe Mut_Vector4d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4d _ctor_result = __MR_Vector4d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Vector4d::Vector4d`.
        public unsafe Mut_Vector4d(double x, double y, double z, double w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_Construct_4(double x, double y, double z, double w);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Vector4d _ctor_result = __MR_Vector4d_Construct_4(x, y, z, w);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Vector4d::operator[]`.
        public unsafe new ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_index", ExactSpelling = true)]
            extern static double *__MR_Vector4d_index(_Underlying *_this, int e);
            return ref *__MR_Vector4d_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4d AddAssign(MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_add_assign_MR_Vector4d(_Underlying *a, MR.Const_Vector4d._Underlying *b);
            return new(__MR_add_assign_MR_Vector4d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4d SubAssign(MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_sub_assign_MR_Vector4d(_Underlying *a, MR.Const_Vector4d._Underlying *b);
            return new(__MR_sub_assign_MR_Vector4d(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_mul_assign_MR_Vector4d_double(_Underlying *a, double b);
            return new(__MR_mul_assign_MR_Vector4d_double(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_div_assign_MR_Vector4d_double(_Underlying *a, double b);
            return new(__MR_div_assign_MR_Vector4d_double(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
    public struct Vector4d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Vector4d(Const_Vector4d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Vector4d(Vector4d other) => new(new Mut_Vector4d((Mut_Vector4d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public double X;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public double Y;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public double Z;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public double W;

        /// Generated copy constructor.
        public Vector4d(Vector4d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector4d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_DefaultConstruct();
            this = __MR_Vector4d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4d::Vector4d`.
        public unsafe Vector4d(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Vector4d_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4d::Vector4d`.
        public unsafe Vector4d(double x, double y, double z, double w)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_Construct_4(double x, double y, double z, double w);
            this = __MR_Vector4d_Construct_4(x, y, z, w);
        }

        /// Generated from method `MR::Vector4d::diagonal`.
        public static MR.Vector4d Diagonal(double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_diagonal", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_diagonal(double a);
            return __MR_Vector4d_diagonal(a);
        }

        /// Generated from method `MR::Vector4d::operator[]`.
        public unsafe double Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_index_const", ExactSpelling = true)]
            extern static double *__MR_Vector4d_index_const(MR.Vector4d *_this, int e);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return *__MR_Vector4d_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4d::operator[]`.
        public unsafe ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_index", ExactSpelling = true)]
            extern static double *__MR_Vector4d_index(MR.Vector4d *_this, int e);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return ref *__MR_Vector4d_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Vector4d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_lengthSq", ExactSpelling = true)]
            extern static double __MR_Vector4d_lengthSq(MR.Vector4d *_this);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return __MR_Vector4d_lengthSq(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_length", ExactSpelling = true)]
            extern static double __MR_Vector4d_length(MR.Vector4d *_this);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return __MR_Vector4d_length(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4d::normalized`.
        public unsafe MR.Vector4d Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_normalized", ExactSpelling = true)]
            extern static MR.Vector4d __MR_Vector4d_normalized(MR.Vector4d *_this);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return __MR_Vector4d_normalized(__ptr__this);
            }
        }

        /// assuming this is a point represented in homogeneous 4D coordinates, returns the point as 3D-vector
        /// Generated from method `MR::Vector4d::proj3d`.
        public unsafe MR.Vector3d Proj3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_proj3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Vector4d_proj3d(MR.Vector4d *_this);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return __MR_Vector4d_proj3d(__ptr__this);
            }
        }

        /// Generated from method `MR::Vector4d::isFinite`.
        public unsafe bool IsFinite()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4d_isFinite", ExactSpelling = true)]
            extern static byte __MR_Vector4d_isFinite(MR.Vector4d *_this);
            fixed (MR.Vector4d *__ptr__this = &this)
            {
                return __MR_Vector4d_isFinite(__ptr__this) != 0;
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Vector4d a, MR.Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_equal_MR_Vector4d((MR.Mut_Vector4d._Underlying *)&a, (MR.Mut_Vector4d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Vector4d a, MR.Vector4d b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4d operator+(MR.Vector4d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Const_Vector4d._Underlying *__MR_pos_MR_Vector4d(MR.Const_Vector4d._Underlying *a);
            return new(__MR_pos_MR_Vector4d((MR.Mut_Vector4d._Underlying *)&a), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4d operator-(MR.Vector4d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_neg_MR_Vector4d(MR.Const_Vector4d._Underlying *a);
            return __MR_neg_MR_Vector4d((MR.Mut_Vector4d._Underlying *)&a);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4d operator+(MR.Vector4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_add_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_add_MR_Vector4d((MR.Mut_Vector4d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4d operator-(MR.Vector4d a, MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_sub_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
            return __MR_sub_MR_Vector4d((MR.Mut_Vector4d._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4d operator*(double a, MR.Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_double_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Vector4d __MR_mul_double_MR_Vector4d(double a, MR.Const_Vector4d._Underlying *b);
            return __MR_mul_double_MR_Vector4d(a, (MR.Mut_Vector4d._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4d operator*(MR.Vector4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Vector4d __MR_mul_MR_Vector4d_double(MR.Const_Vector4d._Underlying *b, double a);
            return __MR_mul_MR_Vector4d_double((MR.Mut_Vector4d._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static MR.Vector4d operator/(MR.Vector4d b, double a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Vector4d __MR_div_MR_Vector4d_double(MR.Vector4d b, double a);
            return __MR_div_MR_Vector4d_double(b, a);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Mut_Vector4d AddAssign(MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_add_assign_MR_Vector4d(MR.Vector4d *a, MR.Const_Vector4d._Underlying *b);
            fixed (MR.Vector4d *__ptr_a = &this)
            {
                return new(__MR_add_assign_MR_Vector4d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Mut_Vector4d SubAssign(MR.Const_Vector4d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4d", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_sub_assign_MR_Vector4d(MR.Vector4d *a, MR.Const_Vector4d._Underlying *b);
            fixed (MR.Vector4d *__ptr_a = &this)
            {
                return new(__MR_sub_assign_MR_Vector4d(__ptr_a, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Mut_Vector4d MulAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_mul_assign_MR_Vector4d_double(MR.Vector4d *a, double b);
            fixed (MR.Vector4d *__ptr_a = &this)
            {
                return new(__MR_mul_assign_MR_Vector4d_double(__ptr_a, b), is_owning: false);
            }
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Mut_Vector4d DivAssign(double b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4d_double", ExactSpelling = true)]
            extern static MR.Mut_Vector4d._Underlying *__MR_div_assign_MR_Vector4d_double(MR.Vector4d *a, double b);
            fixed (MR.Vector4d *__ptr_a = &this)
            {
                return new(__MR_div_assign_MR_Vector4d_double(__ptr_a, b), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.Vector4d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Vector4d)
                return this == (MR.Vector4d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Vector4d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Vector4d`/`Const_Vector4d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Vector4d
    {
        public readonly bool HasValue;
        internal readonly Vector4d Object;
        public Vector4d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Vector4d() {HasValue = false;}
        public _InOpt_Vector4d(Vector4d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Vector4d(Vector4d new_value) {return new(new_value);}
        public _InOpt_Vector4d(Const_Vector4d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Vector4d(Const_Vector4d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Vector4d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector4d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4d`/`Const_Vector4d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4d`.
    public class _InOptMut_Vector4d
    {
        public Mut_Vector4d? Opt;

        public _InOptMut_Vector4d() {}
        public _InOptMut_Vector4d(Mut_Vector4d value) {Opt = value;}
        public static implicit operator _InOptMut_Vector4d(Mut_Vector4d value) {return new(value);}
        public unsafe _InOptMut_Vector4d(ref Vector4d value)
        {
            fixed (Vector4d *value_ptr = &value)
            {
                Opt = new((Const_Vector4d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Vector4d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector4d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Vector4d`/`Const_Vector4d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Vector4d`.
    public class _InOptConst_Vector4d
    {
        public Const_Vector4d? Opt;

        public _InOptConst_Vector4d() {}
        public _InOptConst_Vector4d(Const_Vector4d value) {Opt = value;}
        public static implicit operator _InOptConst_Vector4d(Const_Vector4d value) {return new(value);}
        public unsafe _InOptConst_Vector4d(ref readonly Vector4d value)
        {
            fixed (Vector4d *value_ptr = &value)
            {
                Opt = new((Const_Vector4d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4<unsigned char>`.
    /// This is the const half of the class.
    public class Const_Vector4_UnsignedChar : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Vector4_UnsignedChar>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Vector4_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_Vector4_unsigned_char_Destroy(_Underlying *_this);
            __MR_Vector4_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Vector4_UnsignedChar() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Vector4_unsigned_char_Get_elements();
                return *__MR_Vector4_unsigned_char_Get_elements();
            }
        }

        public unsafe byte X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Get_x", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_Get_x(_Underlying *_this);
                return *__MR_Vector4_unsigned_char_Get_x(_UnderlyingPtr);
            }
        }

        public unsafe byte Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Get_y", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_Get_y(_Underlying *_this);
                return *__MR_Vector4_unsigned_char_Get_y(_UnderlyingPtr);
            }
        }

        public unsafe byte Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Get_z", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_Get_z(_Underlying *_this);
                return *__MR_Vector4_unsigned_char_Get_z(_UnderlyingPtr);
            }
        }

        public unsafe byte W
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Get_w", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_Get_w(_Underlying *_this);
                return *__MR_Vector4_unsigned_char_Get_w(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Vector4_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Vector4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4<unsigned char>::Vector4`.
        public unsafe Const_Vector4_UnsignedChar(MR.Const_Vector4_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_ConstructFromAnother(MR.Vector4_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Vector4_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4<unsigned char>::Vector4`.
        public unsafe Const_Vector4_UnsignedChar(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Vector4_unsigned_char_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4<unsigned char>::Vector4`.
        public unsafe Const_Vector4_UnsignedChar(byte x, byte y, byte z, byte w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_Construct_4(byte x, byte y, byte z, byte w);
            _UnderlyingPtr = __MR_Vector4_unsigned_char_Construct_4(x, y, z, w);
        }

        /// Generated from method `MR::Vector4<unsigned char>::diagonal`.
        public static unsafe MR.Vector4_UnsignedChar Diagonal(byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_diagonal", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_diagonal(byte a);
            return new(__MR_Vector4_unsigned_char_diagonal(a), is_owning: true);
        }

        /// Generated from method `MR::Vector4<unsigned char>::operator[]`.
        public unsafe byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_index_const", ExactSpelling = true)]
            extern static byte *__MR_Vector4_unsigned_char_index_const(_Underlying *_this, int e);
            return *__MR_Vector4_unsigned_char_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Vector4<unsigned char>::lengthSq`.
        public unsafe byte LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_lengthSq", ExactSpelling = true)]
            extern static byte __MR_Vector4_unsigned_char_lengthSq(_Underlying *_this);
            return __MR_Vector4_unsigned_char_lengthSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Vector4<unsigned char>::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_length", ExactSpelling = true)]
            extern static double __MR_Vector4_unsigned_char_length(_Underlying *_this);
            return __MR_Vector4_unsigned_char_length(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Vector4_UnsignedChar a, MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Vector4_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return __MR_equal_MR_Vector4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Vector4_UnsignedChar a, MR.Const_Vector4_UnsignedChar b)
        {
            return !(a == b);
        }

        // NOTE: We use `std::declval()` in the operators below because libclang 18 in our binding generator is bugged and chokes on decltyping `a.x` and such. TODO fix this when we update libclang.
        /// Generated from function `MR::operator+`.
        public static unsafe MR.Const_Vector4_UnsignedChar operator+(MR.Const_Vector4_UnsignedChar a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pos_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_pos_MR_Vector4_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *a);
            return new(__MR_pos_MR_Vector4_unsigned_char(a._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Const_Vector4_UnsignedChar a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_neg_MR_Vector4_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *a);
            return __MR_neg_MR_Vector4_unsigned_char(a._UnderlyingPtr);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Vector4i operator+(MR.Const_Vector4_UnsignedChar a, MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_add_MR_Vector4_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return __MR_add_MR_Vector4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Vector4i operator-(MR.Const_Vector4_UnsignedChar a, MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_sub_MR_Vector4_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return __MR_sub_MR_Vector4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(byte a, MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_unsigned_char_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_unsigned_char_MR_Vector4_unsigned_char(byte a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return __MR_mul_unsigned_char_MR_Vector4_unsigned_char(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Vector4i operator*(MR.Const_Vector4_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Vector4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_mul_MR_Vector4_unsigned_char_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *b, byte a);
            return __MR_mul_MR_Vector4_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Vector4i operator/(Const_Vector4_UnsignedChar b, byte a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Vector4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4i __MR_div_MR_Vector4_unsigned_char_unsigned_char(MR.Vector4_UnsignedChar._Underlying *b, byte a);
            return __MR_div_MR_Vector4_unsigned_char_unsigned_char(b._UnderlyingPtr, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Vector4_UnsignedChar? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Vector4_UnsignedChar)
                return this == (MR.Const_Vector4_UnsignedChar)other;
            return false;
        }
    }

    /// four-dimensional vector
    /// Generated from class `MR::Vector4<unsigned char>`.
    /// This is the non-const half of the class.
    public class Vector4_UnsignedChar : Const_Vector4_UnsignedChar
    {
        internal unsafe Vector4_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref byte X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_GetMutable_x", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_GetMutable_x(_Underlying *_this);
                return ref *__MR_Vector4_unsigned_char_GetMutable_x(_UnderlyingPtr);
            }
        }

        public new unsafe ref byte Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_GetMutable_y", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_GetMutable_y(_Underlying *_this);
                return ref *__MR_Vector4_unsigned_char_GetMutable_y(_UnderlyingPtr);
            }
        }

        public new unsafe ref byte Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_GetMutable_z", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_GetMutable_z(_Underlying *_this);
                return ref *__MR_Vector4_unsigned_char_GetMutable_z(_UnderlyingPtr);
            }
        }

        public new unsafe ref byte W
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_GetMutable_w", ExactSpelling = true)]
                extern static byte *__MR_Vector4_unsigned_char_GetMutable_w(_Underlying *_this);
                return ref *__MR_Vector4_unsigned_char_GetMutable_w(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Vector4_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Vector4_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Vector4<unsigned char>::Vector4`.
        public unsafe Vector4_UnsignedChar(MR.Const_Vector4_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_ConstructFromAnother(MR.Vector4_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Vector4_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4<unsigned char>::Vector4`.
        public unsafe Vector4_UnsignedChar(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Construct_1", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Vector4_unsigned_char_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Vector4<unsigned char>::Vector4`.
        public unsafe Vector4_UnsignedChar(byte x, byte y, byte z, byte w) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_Construct_4", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_Construct_4(byte x, byte y, byte z, byte w);
            _UnderlyingPtr = __MR_Vector4_unsigned_char_Construct_4(x, y, z, w);
        }

        /// Generated from method `MR::Vector4<unsigned char>::operator=`.
        public unsafe MR.Vector4_UnsignedChar Assign(MR.Const_Vector4_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_Vector4_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Vector4_UnsignedChar._Underlying *_other);
            return new(__MR_Vector4_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Vector4<unsigned char>::operator[]`.
        public unsafe new ref byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Vector4_unsigned_char_index", ExactSpelling = true)]
            extern static byte *__MR_Vector4_unsigned_char_index(_Underlying *_this, int e);
            return ref *__MR_Vector4_unsigned_char_index(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator+=`.
        public unsafe MR.Vector4_UnsignedChar AddAssign(MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_add_assign_MR_Vector4_unsigned_char(_Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return new(__MR_add_assign_MR_Vector4_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator-=`.
        public unsafe MR.Vector4_UnsignedChar SubAssign(MR.Const_Vector4_UnsignedChar b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_Vector4_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_sub_assign_MR_Vector4_unsigned_char(_Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
            return new(__MR_sub_assign_MR_Vector4_unsigned_char(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from function `MR::operator*=`.
        public unsafe MR.Vector4_UnsignedChar MulAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_Vector4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_mul_assign_MR_Vector4_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_mul_assign_MR_Vector4_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }

        /// Generated from function `MR::operator/=`.
        public unsafe MR.Vector4_UnsignedChar DivAssign(byte b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_assign_MR_Vector4_unsigned_char_unsigned_char", ExactSpelling = true)]
            extern static MR.Vector4_UnsignedChar._Underlying *__MR_div_assign_MR_Vector4_unsigned_char_unsigned_char(_Underlying *a, byte b);
            return new(__MR_div_assign_MR_Vector4_unsigned_char_unsigned_char(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Vector4_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Vector4_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Vector4_UnsignedChar`/`Const_Vector4_UnsignedChar` directly.
    public class _InOptMut_Vector4_UnsignedChar
    {
        public Vector4_UnsignedChar? Opt;

        public _InOptMut_Vector4_UnsignedChar() {}
        public _InOptMut_Vector4_UnsignedChar(Vector4_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_Vector4_UnsignedChar(Vector4_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `Vector4_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Vector4_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Vector4_UnsignedChar`/`Const_Vector4_UnsignedChar` to pass it to the function.
    public class _InOptConst_Vector4_UnsignedChar
    {
        public Const_Vector4_UnsignedChar? Opt;

        public _InOptConst_Vector4_UnsignedChar() {}
        public _InOptConst_Vector4_UnsignedChar(Const_Vector4_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_Vector4_UnsignedChar(Const_Vector4_UnsignedChar value) {return new(value);}
    }

    /// dot product
    /// Generated from function `MR::dot<bool>`.
    public static unsafe int Dot(MR.Const_Vector4b a, MR.Const_Vector4b b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_bool_MR_Vector4b", ExactSpelling = true)]
        extern static int __MR_dot_bool_MR_Vector4b(MR.Const_Vector4b._Underlying *a, MR.Const_Vector4b._Underlying *b);
        return __MR_dot_bool_MR_Vector4b(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<int>`.
    public static unsafe int Dot(MR.Const_Vector4i a, MR.Const_Vector4i b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_int_MR_Vector4i", ExactSpelling = true)]
        extern static int __MR_dot_int_MR_Vector4i(MR.Const_Vector4i._Underlying *a, MR.Const_Vector4i._Underlying *b);
        return __MR_dot_int_MR_Vector4i(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<MR_int64_t>`.
    public static unsafe long Dot(MR.Const_Vector4i64 a, MR.Const_Vector4i64 b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_int64_t_MR_Vector4i64", ExactSpelling = true)]
        extern static long __MR_dot_int64_t_MR_Vector4i64(MR.Const_Vector4i64._Underlying *a, MR.Const_Vector4i64._Underlying *b);
        return __MR_dot_int64_t_MR_Vector4i64(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<float>`.
    public static unsafe float Dot(MR.Const_Vector4f a, MR.Const_Vector4f b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_float_MR_Vector4f", ExactSpelling = true)]
        extern static float __MR_dot_float_MR_Vector4f(MR.Const_Vector4f._Underlying *a, MR.Const_Vector4f._Underlying *b);
        return __MR_dot_float_MR_Vector4f(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<double>`.
    public static unsafe double Dot(MR.Const_Vector4d a, MR.Const_Vector4d b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_double_MR_Vector4d", ExactSpelling = true)]
        extern static double __MR_dot_double_MR_Vector4d(MR.Const_Vector4d._Underlying *a, MR.Const_Vector4d._Underlying *b);
        return __MR_dot_double_MR_Vector4d(a._UnderlyingPtr, b._UnderlyingPtr);
    }

    /// dot product
    /// Generated from function `MR::dot<unsigned char>`.
    public static unsafe int Dot(MR.Const_Vector4_UnsignedChar a, MR.Const_Vector4_UnsignedChar b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dot_unsigned_char_MR_Vector4_unsigned_char", ExactSpelling = true)]
        extern static int __MR_dot_unsigned_char_MR_Vector4_unsigned_char(MR.Const_Vector4_UnsignedChar._Underlying *a, MR.Const_Vector4_UnsignedChar._Underlying *b);
        return __MR_dot_unsigned_char_MR_Vector4_unsigned_char(a._UnderlyingPtr, b._UnderlyingPtr);
    }
}
