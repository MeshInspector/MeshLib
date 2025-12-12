public static partial class MR
{
    /// Generated from class `MR::Color`.
    /// This is the const reference to the struct.
    public class Const_Color : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Color>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Color UnderlyingStruct => ref *(Color *)_UnderlyingPtr;

        internal unsafe Const_Color(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Destroy", ExactSpelling = true)]
            extern static void __MR_Color_Destroy(_Underlying *_this);
            __MR_Color_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Color() {Dispose(false);}

        public ref readonly byte R => ref UnderlyingStruct.R;

        public ref readonly byte G => ref UnderlyingStruct.G;

        public ref readonly byte B => ref UnderlyingStruct.B;

        public ref readonly byte A => ref UnderlyingStruct.A;

        /// Generated copy constructor.
        public unsafe Const_Color(Const_Color _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Color() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Color __MR_Color_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_1", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(int r, int g, int b, int a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_4_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_4_int(int r, int g, int b, int a);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_4_int(r, g, b, a);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(int r, int g, int b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_3_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_3_int(int r, int g, int b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_3_int(r, g, b);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(float r, float g, float b, float a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_4_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_4_float(float r, float g, float b, float a);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_4_float(r, g, b, a);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(float r, float g, float b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_3_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_3_float(float r, float g, float b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_3_float(r, g, b);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(MR.Const_Vector4i vec) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_int(MR.Const_Vector4i._Underlying *vec);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_int(vec._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Const_Color(MR.Const_Vector4f vec) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_float(MR.Const_Vector4f._Underlying *vec);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_float(vec._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::Color::operator MR::Vector4i`.
        public static unsafe explicit operator MR.Vector4i(MR.Const_Color _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_ConvertTo_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Color_ConvertTo_MR_Vector4i(MR.Const_Color._Underlying *_this);
            return __MR_Color_ConvertTo_MR_Vector4i(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::Color::operator MR::Vector4f`.
        public static unsafe explicit operator MR.Vector4f(MR.Const_Color _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_ConvertTo_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Color_ConvertTo_MR_Vector4f(MR.Const_Color._Underlying *_this);
            return __MR_Color_ConvertTo_MR_Vector4f(_this._UnderlyingPtr);
        }

        /// Generated from method `MR::Color::getUInt32`.
        public unsafe uint GetUInt32()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_getUInt32", ExactSpelling = true)]
            extern static uint __MR_Color_getUInt32(_Underlying *_this);
            return __MR_Color_getUInt32(_UnderlyingPtr);
        }

        /// Generated from method `MR::Color::white`.
        public static MR.Color White()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_white", ExactSpelling = true)]
            extern static MR.Color __MR_Color_white();
            return __MR_Color_white();
        }

        /// Generated from method `MR::Color::black`.
        public static MR.Color Black()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_black", ExactSpelling = true)]
            extern static MR.Color __MR_Color_black();
            return __MR_Color_black();
        }

        /// Generated from method `MR::Color::gray`.
        public static MR.Color Gray()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_gray", ExactSpelling = true)]
            extern static MR.Color __MR_Color_gray();
            return __MR_Color_gray();
        }

        /// Generated from method `MR::Color::red`.
        public static MR.Color Red()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_red", ExactSpelling = true)]
            extern static MR.Color __MR_Color_red();
            return __MR_Color_red();
        }

        /// Generated from method `MR::Color::green`.
        public static MR.Color Green()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_green", ExactSpelling = true)]
            extern static MR.Color __MR_Color_green();
            return __MR_Color_green();
        }

        /// Generated from method `MR::Color::blue`.
        public static MR.Color Blue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_blue", ExactSpelling = true)]
            extern static MR.Color __MR_Color_blue();
            return __MR_Color_blue();
        }

        /// Generated from method `MR::Color::yellow`.
        public static MR.Color Yellow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_yellow", ExactSpelling = true)]
            extern static MR.Color __MR_Color_yellow();
            return __MR_Color_yellow();
        }

        /// Generated from method `MR::Color::brown`.
        public static MR.Color Brown()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_brown", ExactSpelling = true)]
            extern static MR.Color __MR_Color_brown();
            return __MR_Color_brown();
        }

        /// Generated from method `MR::Color::purple`.
        public static MR.Color Purple()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_purple", ExactSpelling = true)]
            extern static MR.Color __MR_Color_purple();
            return __MR_Color_purple();
        }

        /// Generated from method `MR::Color::transparent`.
        public static MR.Color Transparent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_transparent", ExactSpelling = true)]
            extern static MR.Color __MR_Color_transparent();
            return __MR_Color_transparent();
        }

        /// Generated from method `MR::Color::valToUint8<int>`.
        public static byte ValToUint8(int val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_valToUint8_int", ExactSpelling = true)]
            extern static byte __MR_Color_valToUint8_int(int val);
            return __MR_Color_valToUint8_int(val);
        }

        /// Generated from method `MR::Color::valToUint8<float>`.
        public static byte ValToUint8(float val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_valToUint8_float", ExactSpelling = true)]
            extern static byte __MR_Color_valToUint8_float(float val);
            return __MR_Color_valToUint8_float(val);
        }

        /// Generated from method `MR::Color::operator[]`.
        public unsafe byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_index_const", ExactSpelling = true)]
            extern static byte *__MR_Color_index_const(_Underlying *_this, int e);
            return *__MR_Color_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Color::scaledAlpha`.
        public unsafe MR.Color ScaledAlpha(float m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_scaledAlpha", ExactSpelling = true)]
            extern static MR.Color __MR_Color_scaledAlpha(_Underlying *_this, float m);
            return __MR_Color_scaledAlpha(_UnderlyingPtr, m);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_Color a, MR.Const_Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Color", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Color(MR.Const_Color._Underlying *a, MR.Const_Color._Underlying *b);
            return __MR_equal_MR_Color(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Color a, MR.Const_Color b)
        {
            return !(a == b);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Color operator+(MR.Const_Color a, MR.Const_Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Color", ExactSpelling = true)]
            extern static MR.Color __MR_add_MR_Color(MR.Const_Color._Underlying *a, MR.Const_Color._Underlying *b);
            return __MR_add_MR_Color(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Color operator-(MR.Const_Color a, MR.Const_Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Color", ExactSpelling = true)]
            extern static MR.Color __MR_sub_MR_Color(MR.Const_Color._Underlying *a, MR.Const_Color._Underlying *b);
            return __MR_sub_MR_Color(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Color operator*(float a, MR.Const_Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Color", ExactSpelling = true)]
            extern static MR.Color __MR_mul_float_MR_Color(float a, MR.Const_Color._Underlying *b);
            return __MR_mul_float_MR_Color(a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Color operator*(MR.Const_Color b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Color_float", ExactSpelling = true)]
            extern static MR.Color __MR_mul_MR_Color_float(MR.Const_Color._Underlying *b, float a);
            return __MR_mul_MR_Color_float(b._UnderlyingPtr, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Color operator/(MR.Const_Color b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Color_float", ExactSpelling = true)]
            extern static MR.Color __MR_div_MR_Color_float(MR.Const_Color._Underlying *b, float a);
            return __MR_div_MR_Color_float(b._UnderlyingPtr, a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Color? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Color)
                return this == (MR.Const_Color)other;
            return false;
        }
    }

    /// Generated from class `MR::Color`.
    /// This is the non-const reference to the struct.
    public class Mut_Color : Const_Color
    {
        /// Get the underlying struct.
        public unsafe new ref Color UnderlyingStruct => ref *(Color *)_UnderlyingPtr;

        internal unsafe Mut_Color(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref byte R => ref UnderlyingStruct.R;

        public new ref byte G => ref UnderlyingStruct.G;

        public new ref byte B => ref UnderlyingStruct.B;

        public new ref byte A => ref UnderlyingStruct.A;

        /// Generated copy constructor.
        public unsafe Mut_Color(Const_Color _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Color() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Color __MR_Color_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_1", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(int r, int g, int b, int a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_4_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_4_int(int r, int g, int b, int a);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_4_int(r, g, b, a);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(int r, int g, int b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_3_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_3_int(int r, int g, int b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_3_int(r, g, b);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(float r, float g, float b, float a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_4_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_4_float(float r, float g, float b, float a);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_4_float(r, g, b, a);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(float r, float g, float b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_3_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_3_float(float r, float g, float b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_3_float(r, g, b);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(MR.Const_Vector4i vec) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_int(MR.Const_Vector4i._Underlying *vec);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_int(vec._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Mut_Color(MR.Const_Vector4f vec) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_float(MR.Const_Vector4f._Underlying *vec);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.Color _ctor_result = __MR_Color_Construct_float(vec._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::Color::operator[]`.
        public unsafe new ref byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_index", ExactSpelling = true)]
            extern static byte *__MR_Color_index(_Underlying *_this, int e);
            return ref *__MR_Color_index(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Color::operator+=`.
        public unsafe MR.Mut_Color AddAssign(MR.Const_Color other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_add_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_add_assign(_Underlying *_this, MR.Const_Color._Underlying *other);
            return new(__MR_Color_add_assign(_UnderlyingPtr, other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Color::operator-=`.
        public unsafe MR.Mut_Color SubAssign(MR.Const_Color other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_sub_assign(_Underlying *_this, MR.Const_Color._Underlying *other);
            return new(__MR_Color_sub_assign(_UnderlyingPtr, other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Color::operator*=`.
        public unsafe MR.Mut_Color MulAssign(float m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_mul_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_mul_assign(_Underlying *_this, float m);
            return new(__MR_Color_mul_assign(_UnderlyingPtr, m), is_owning: false);
        }

        /// Generated from method `MR::Color::operator/=`.
        public unsafe MR.Mut_Color DivAssign(float m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_div_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_div_assign(_Underlying *_this, float m);
            return new(__MR_Color_div_assign(_UnderlyingPtr, m), is_owning: false);
        }
    }

    /// Generated from class `MR::Color`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct Color
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Color(Const_Color other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Color(Color other) => new(new Mut_Color((Mut_Color._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public byte R;

        [System.Runtime.InteropServices.FieldOffset(1)]
        public byte G;

        [System.Runtime.InteropServices.FieldOffset(2)]
        public byte B;

        [System.Runtime.InteropServices.FieldOffset(3)]
        public byte A;

        /// Generated copy constructor.
        public Color(Color _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Color()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Color __MR_Color_DefaultConstruct();
            this = __MR_Color_DefaultConstruct();
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_1", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Color_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(int r, int g, int b, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_4_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_4_int(int r, int g, int b, int a);
            this = __MR_Color_Construct_4_int(r, g, b, a);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(int r, int g, int b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_3_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_3_int(int r, int g, int b);
            this = __MR_Color_Construct_3_int(r, g, b);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(float r, float g, float b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_4_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_4_float(float r, float g, float b, float a);
            this = __MR_Color_Construct_4_float(r, g, b, a);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(float r, float g, float b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_3_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_3_float(float r, float g, float b);
            this = __MR_Color_Construct_3_float(r, g, b);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(MR.Const_Vector4i vec)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_int", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_int(MR.Const_Vector4i._Underlying *vec);
            this = __MR_Color_Construct_int(vec._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Color::Color`.
        public unsafe Color(MR.Const_Vector4f vec)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_Construct_float", ExactSpelling = true)]
            extern static MR.Color __MR_Color_Construct_float(MR.Const_Vector4f._Underlying *vec);
            this = __MR_Color_Construct_float(vec._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::Color::operator MR::Vector4i`.
        public static unsafe explicit operator MR.Vector4i(MR.Color _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_ConvertTo_MR_Vector4i", ExactSpelling = true)]
            extern static MR.Vector4i __MR_Color_ConvertTo_MR_Vector4i(MR.Const_Color._Underlying *_this);
            return __MR_Color_ConvertTo_MR_Vector4i((MR.Mut_Color._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::Color::operator MR::Vector4f`.
        public static unsafe explicit operator MR.Vector4f(MR.Color _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_ConvertTo_MR_Vector4f", ExactSpelling = true)]
            extern static MR.Vector4f __MR_Color_ConvertTo_MR_Vector4f(MR.Const_Color._Underlying *_this);
            return __MR_Color_ConvertTo_MR_Vector4f((MR.Mut_Color._Underlying *)&_this);
        }

        /// Generated from method `MR::Color::getUInt32`.
        public unsafe uint GetUInt32()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_getUInt32", ExactSpelling = true)]
            extern static uint __MR_Color_getUInt32(MR.Color *_this);
            fixed (MR.Color *__ptr__this = &this)
            {
                return __MR_Color_getUInt32(__ptr__this);
            }
        }

        /// Generated from method `MR::Color::white`.
        public static MR.Color White()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_white", ExactSpelling = true)]
            extern static MR.Color __MR_Color_white();
            return __MR_Color_white();
        }

        /// Generated from method `MR::Color::black`.
        public static MR.Color Black()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_black", ExactSpelling = true)]
            extern static MR.Color __MR_Color_black();
            return __MR_Color_black();
        }

        /// Generated from method `MR::Color::gray`.
        public static MR.Color Gray()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_gray", ExactSpelling = true)]
            extern static MR.Color __MR_Color_gray();
            return __MR_Color_gray();
        }

        /// Generated from method `MR::Color::red`.
        public static MR.Color Red()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_red", ExactSpelling = true)]
            extern static MR.Color __MR_Color_red();
            return __MR_Color_red();
        }

        /// Generated from method `MR::Color::green`.
        public static MR.Color Green()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_green", ExactSpelling = true)]
            extern static MR.Color __MR_Color_green();
            return __MR_Color_green();
        }

        /// Generated from method `MR::Color::blue`.
        public static MR.Color Blue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_blue", ExactSpelling = true)]
            extern static MR.Color __MR_Color_blue();
            return __MR_Color_blue();
        }

        /// Generated from method `MR::Color::yellow`.
        public static MR.Color Yellow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_yellow", ExactSpelling = true)]
            extern static MR.Color __MR_Color_yellow();
            return __MR_Color_yellow();
        }

        /// Generated from method `MR::Color::brown`.
        public static MR.Color Brown()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_brown", ExactSpelling = true)]
            extern static MR.Color __MR_Color_brown();
            return __MR_Color_brown();
        }

        /// Generated from method `MR::Color::purple`.
        public static MR.Color Purple()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_purple", ExactSpelling = true)]
            extern static MR.Color __MR_Color_purple();
            return __MR_Color_purple();
        }

        /// Generated from method `MR::Color::transparent`.
        public static MR.Color Transparent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_transparent", ExactSpelling = true)]
            extern static MR.Color __MR_Color_transparent();
            return __MR_Color_transparent();
        }

        /// Generated from method `MR::Color::valToUint8<int>`.
        public static byte ValToUint8(int val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_valToUint8_int", ExactSpelling = true)]
            extern static byte __MR_Color_valToUint8_int(int val);
            return __MR_Color_valToUint8_int(val);
        }

        /// Generated from method `MR::Color::valToUint8<float>`.
        public static byte ValToUint8(float val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_valToUint8_float", ExactSpelling = true)]
            extern static byte __MR_Color_valToUint8_float(float val);
            return __MR_Color_valToUint8_float(val);
        }

        /// Generated from method `MR::Color::operator[]`.
        public unsafe byte Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_index_const", ExactSpelling = true)]
            extern static byte *__MR_Color_index_const(MR.Color *_this, int e);
            fixed (MR.Color *__ptr__this = &this)
            {
                return *__MR_Color_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Color::operator[]`.
        public unsafe ref byte Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_index", ExactSpelling = true)]
            extern static byte *__MR_Color_index(MR.Color *_this, int e);
            fixed (MR.Color *__ptr__this = &this)
            {
                return ref *__MR_Color_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Color::operator+=`.
        public unsafe MR.Mut_Color AddAssign(MR.Const_Color other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_add_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_add_assign(MR.Color *_this, MR.Const_Color._Underlying *other);
            fixed (MR.Color *__ptr__this = &this)
            {
                return new(__MR_Color_add_assign(__ptr__this, other._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from method `MR::Color::operator-=`.
        public unsafe MR.Mut_Color SubAssign(MR.Const_Color other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_sub_assign(MR.Color *_this, MR.Const_Color._Underlying *other);
            fixed (MR.Color *__ptr__this = &this)
            {
                return new(__MR_Color_sub_assign(__ptr__this, other._UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from method `MR::Color::operator*=`.
        public unsafe MR.Mut_Color MulAssign(float m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_mul_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_mul_assign(MR.Color *_this, float m);
            fixed (MR.Color *__ptr__this = &this)
            {
                return new(__MR_Color_mul_assign(__ptr__this, m), is_owning: false);
            }
        }

        /// Generated from method `MR::Color::operator/=`.
        public unsafe MR.Mut_Color DivAssign(float m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_div_assign", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Color_div_assign(MR.Color *_this, float m);
            fixed (MR.Color *__ptr__this = &this)
            {
                return new(__MR_Color_div_assign(__ptr__this, m), is_owning: false);
            }
        }

        /// Generated from method `MR::Color::scaledAlpha`.
        public unsafe MR.Color ScaledAlpha(float m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Color_scaledAlpha", ExactSpelling = true)]
            extern static MR.Color __MR_Color_scaledAlpha(MR.Color *_this, float m);
            fixed (MR.Color *__ptr__this = &this)
            {
                return __MR_Color_scaledAlpha(__ptr__this, m);
            }
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Color a, MR.Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Color", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Color(MR.Const_Color._Underlying *a, MR.Const_Color._Underlying *b);
            return __MR_equal_MR_Color((MR.Mut_Color._Underlying *)&a, (MR.Mut_Color._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.Color a, MR.Color b)
        {
            return !(a == b);
        }

        /// Generated from function `MR::operator+`.
        public static unsafe MR.Color operator+(MR.Color a, MR.Const_Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_Color", ExactSpelling = true)]
            extern static MR.Color __MR_add_MR_Color(MR.Const_Color._Underlying *a, MR.Const_Color._Underlying *b);
            return __MR_add_MR_Color((MR.Mut_Color._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Color operator-(MR.Color a, MR.Const_Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_Color", ExactSpelling = true)]
            extern static MR.Color __MR_sub_MR_Color(MR.Const_Color._Underlying *a, MR.Const_Color._Underlying *b);
            return __MR_sub_MR_Color((MR.Mut_Color._Underlying *)&a, b._UnderlyingPtr);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Color operator*(float a, MR.Color b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_float_MR_Color", ExactSpelling = true)]
            extern static MR.Color __MR_mul_float_MR_Color(float a, MR.Const_Color._Underlying *b);
            return __MR_mul_float_MR_Color(a, (MR.Mut_Color._Underlying *)&b);
        }

        /// Generated from function `MR::operator*`.
        public static unsafe MR.Color operator*(MR.Color b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_Color_float", ExactSpelling = true)]
            extern static MR.Color __MR_mul_MR_Color_float(MR.Const_Color._Underlying *b, float a);
            return __MR_mul_MR_Color_float((MR.Mut_Color._Underlying *)&b, a);
        }

        /// Generated from function `MR::operator/`.
        public static unsafe MR.Color operator/(MR.Color b, float a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_div_MR_Color_float", ExactSpelling = true)]
            extern static MR.Color __MR_div_MR_Color_float(MR.Const_Color._Underlying *b, float a);
            return __MR_div_MR_Color_float((MR.Mut_Color._Underlying *)&b, a);
        }

        // IEquatable:

        public bool Equals(MR.Color b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Color)
                return this == (MR.Color)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Color` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Color`/`Const_Color` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Color
    {
        public readonly bool HasValue;
        internal readonly Color Object;
        public Color Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Color() {HasValue = false;}
        public _InOpt_Color(Color new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Color(Color new_value) {return new(new_value);}
        public _InOpt_Color(Const_Color new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Color(Const_Color new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Color` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Color`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Color`/`Const_Color` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Color`.
    public class _InOptMut_Color
    {
        public Mut_Color? Opt;

        public _InOptMut_Color() {}
        public _InOptMut_Color(Mut_Color value) {Opt = value;}
        public static implicit operator _InOptMut_Color(Mut_Color value) {return new(value);}
        public unsafe _InOptMut_Color(ref Color value)
        {
            fixed (Color *value_ptr = &value)
            {
                Opt = new((Const_Color._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Color` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Color`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Color`/`Const_Color` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Color`.
    public class _InOptConst_Color
    {
        public Const_Color? Opt;

        public _InOptConst_Color() {}
        public _InOptConst_Color(Const_Color value) {Opt = value;}
        public static implicit operator _InOptConst_Color(Const_Color value) {return new(value);}
        public unsafe _InOptConst_Color(ref readonly Color value)
        {
            fixed (Color *value_ptr = &value)
            {
                Opt = new((Const_Color._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Blend two colors together
    /// \note This operation is not commutative
    /// Generated from function `MR::blend`.
    public static unsafe MR.Color Blend(MR.Const_Color front, MR.Const_Color back)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_blend", ExactSpelling = true)]
        extern static MR.Color __MR_blend(MR.Const_Color._Underlying *front, MR.Const_Color._Underlying *back);
        return __MR_blend(front._UnderlyingPtr, back._UnderlyingPtr);
    }
}
