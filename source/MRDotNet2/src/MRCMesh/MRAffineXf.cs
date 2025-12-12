public static partial class MR
{
    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf2f`.
    /// This is the const reference to the struct.
    public class Const_AffineXf2f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_AffineXf2f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly AffineXf2f UnderlyingStruct => ref *(AffineXf2f *)_UnderlyingPtr;

        internal unsafe Const_AffineXf2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_Destroy", ExactSpelling = true)]
            extern static void __MR_AffineXf2f_Destroy(_Underlying *_this);
            __MR_AffineXf2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AffineXf2f() {Dispose(false);}

        public ref readonly MR.Matrix2f A => ref UnderlyingStruct.A;

        public ref readonly MR.Vector2f B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Const_AffineXf2f(Const_AffineXf2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AffineXf2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.AffineXf2f _ctor_result = __MR_AffineXf2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::AffineXf2f::AffineXf2f`.
        public unsafe Const_AffineXf2f(MR.Const_Matrix2f A, MR.Const_Vector2f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_Construct", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_Construct(MR.Const_Matrix2f._Underlying *A, MR.Const_Vector2f._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.AffineXf2f _ctor_result = __MR_AffineXf2f_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf2f::translation`.
        public static unsafe MR.AffineXf2f Translation(MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_translation", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_translation(MR.Const_Vector2f._Underlying *b);
            return __MR_AffineXf2f_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf2f::linear`.
        public static unsafe MR.AffineXf2f Linear(MR.Const_Matrix2f A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_linear", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_linear(MR.Const_Matrix2f._Underlying *A);
            return __MR_AffineXf2f_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf2f::xfAround`.
        public static unsafe MR.AffineXf2f XfAround(MR.Const_Matrix2f A, MR.Const_Vector2f stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_xfAround(MR.Const_Matrix2f._Underlying *A, MR.Const_Vector2f._Underlying *stable);
            return __MR_AffineXf2f_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf2f::operator()`.
        public unsafe MR.Vector2f Call(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_call", ExactSpelling = true)]
            extern static MR.Vector2f __MR_AffineXf2f_call(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_AffineXf2f_call(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf2f::linearOnly`.
        public unsafe MR.Vector2f LinearOnly(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_linearOnly", ExactSpelling = true)]
            extern static MR.Vector2f __MR_AffineXf2f_linearOnly(_Underlying *_this, MR.Const_Vector2f._Underlying *x);
            return __MR_AffineXf2f_linearOnly(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf2f::inverse`.
        public unsafe MR.AffineXf2f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_inverse", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_inverse(_Underlying *_this);
            return __MR_AffineXf2f_inverse(_UnderlyingPtr);
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf2f operator*(MR.Const_AffineXf2f u, MR.Const_AffineXf2f v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf2f", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_mul_MR_AffineXf2f(MR.Const_AffineXf2f._Underlying *u, MR.Const_AffineXf2f._Underlying *v);
            return __MR_mul_MR_AffineXf2f(u._UnderlyingPtr, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_AffineXf2f a, MR.Const_AffineXf2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf2f(MR.Const_AffineXf2f._Underlying *a, MR.Const_AffineXf2f._Underlying *b);
            return __MR_equal_MR_AffineXf2f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_AffineXf2f a, MR.Const_AffineXf2f b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_AffineXf2f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_AffineXf2f)
                return this == (MR.Const_AffineXf2f)other;
            return false;
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf2f`.
    /// This is the non-const reference to the struct.
    public class Mut_AffineXf2f : Const_AffineXf2f
    {
        /// Get the underlying struct.
        public unsafe new ref AffineXf2f UnderlyingStruct => ref *(AffineXf2f *)_UnderlyingPtr;

        internal unsafe Mut_AffineXf2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Matrix2f A => ref UnderlyingStruct.A;

        public new ref MR.Vector2f B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Mut_AffineXf2f(Const_AffineXf2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_AffineXf2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.AffineXf2f _ctor_result = __MR_AffineXf2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::AffineXf2f::AffineXf2f`.
        public unsafe Mut_AffineXf2f(MR.Const_Matrix2f A, MR.Const_Vector2f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_Construct", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_Construct(MR.Const_Matrix2f._Underlying *A, MR.Const_Vector2f._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.AffineXf2f _ctor_result = __MR_AffineXf2f_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf2f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 24)]
    public struct AffineXf2f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator AffineXf2f(Const_AffineXf2f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_AffineXf2f(AffineXf2f other) => new(new Mut_AffineXf2f((Mut_AffineXf2f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Matrix2f A;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector2f B;

        /// Generated copy constructor.
        public AffineXf2f(AffineXf2f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AffineXf2f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_DefaultConstruct();
            this = __MR_AffineXf2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AffineXf2f::AffineXf2f`.
        public unsafe AffineXf2f(MR.Const_Matrix2f A, MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_Construct", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_Construct(MR.Const_Matrix2f._Underlying *A, MR.Const_Vector2f._Underlying *b);
            this = __MR_AffineXf2f_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf2f::translation`.
        public static unsafe MR.AffineXf2f Translation(MR.Const_Vector2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_translation", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_translation(MR.Const_Vector2f._Underlying *b);
            return __MR_AffineXf2f_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf2f::linear`.
        public static unsafe MR.AffineXf2f Linear(MR.Const_Matrix2f A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_linear", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_linear(MR.Const_Matrix2f._Underlying *A);
            return __MR_AffineXf2f_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf2f::xfAround`.
        public static unsafe MR.AffineXf2f XfAround(MR.Const_Matrix2f A, MR.Const_Vector2f stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_xfAround(MR.Const_Matrix2f._Underlying *A, MR.Const_Vector2f._Underlying *stable);
            return __MR_AffineXf2f_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf2f::operator()`.
        public unsafe MR.Vector2f Call(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_call", ExactSpelling = true)]
            extern static MR.Vector2f __MR_AffineXf2f_call(MR.AffineXf2f *_this, MR.Const_Vector2f._Underlying *x);
            fixed (MR.AffineXf2f *__ptr__this = &this)
            {
                return __MR_AffineXf2f_call(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf2f::linearOnly`.
        public unsafe MR.Vector2f LinearOnly(MR.Const_Vector2f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_linearOnly", ExactSpelling = true)]
            extern static MR.Vector2f __MR_AffineXf2f_linearOnly(MR.AffineXf2f *_this, MR.Const_Vector2f._Underlying *x);
            fixed (MR.AffineXf2f *__ptr__this = &this)
            {
                return __MR_AffineXf2f_linearOnly(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf2f::inverse`.
        public unsafe MR.AffineXf2f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2f_inverse", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_AffineXf2f_inverse(MR.AffineXf2f *_this);
            fixed (MR.AffineXf2f *__ptr__this = &this)
            {
                return __MR_AffineXf2f_inverse(__ptr__this);
            }
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf2f operator*(MR.AffineXf2f u, MR.Const_AffineXf2f v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf2f", ExactSpelling = true)]
            extern static MR.AffineXf2f __MR_mul_MR_AffineXf2f(MR.Const_AffineXf2f._Underlying *u, MR.Const_AffineXf2f._Underlying *v);
            return __MR_mul_MR_AffineXf2f((MR.Mut_AffineXf2f._Underlying *)&u, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.AffineXf2f a, MR.AffineXf2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf2f(MR.Const_AffineXf2f._Underlying *a, MR.Const_AffineXf2f._Underlying *b);
            return __MR_equal_MR_AffineXf2f((MR.Mut_AffineXf2f._Underlying *)&a, (MR.Mut_AffineXf2f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.AffineXf2f a, MR.AffineXf2f b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.AffineXf2f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.AffineXf2f)
                return this == (MR.AffineXf2f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_AffineXf2f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_AffineXf2f`/`Const_AffineXf2f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_AffineXf2f
    {
        public readonly bool HasValue;
        internal readonly AffineXf2f Object;
        public AffineXf2f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_AffineXf2f() {HasValue = false;}
        public _InOpt_AffineXf2f(AffineXf2f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_AffineXf2f(AffineXf2f new_value) {return new(new_value);}
        public _InOpt_AffineXf2f(Const_AffineXf2f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_AffineXf2f(Const_AffineXf2f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_AffineXf2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AffineXf2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf2f`/`Const_AffineXf2f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf2f`.
    public class _InOptMut_AffineXf2f
    {
        public Mut_AffineXf2f? Opt;

        public _InOptMut_AffineXf2f() {}
        public _InOptMut_AffineXf2f(Mut_AffineXf2f value) {Opt = value;}
        public static implicit operator _InOptMut_AffineXf2f(Mut_AffineXf2f value) {return new(value);}
        public unsafe _InOptMut_AffineXf2f(ref AffineXf2f value)
        {
            fixed (AffineXf2f *value_ptr = &value)
            {
                Opt = new((Const_AffineXf2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_AffineXf2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AffineXf2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf2f`/`Const_AffineXf2f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf2f`.
    public class _InOptConst_AffineXf2f
    {
        public Const_AffineXf2f? Opt;

        public _InOptConst_AffineXf2f() {}
        public _InOptConst_AffineXf2f(Const_AffineXf2f value) {Opt = value;}
        public static implicit operator _InOptConst_AffineXf2f(Const_AffineXf2f value) {return new(value);}
        public unsafe _InOptConst_AffineXf2f(ref readonly AffineXf2f value)
        {
            fixed (AffineXf2f *value_ptr = &value)
            {
                Opt = new((Const_AffineXf2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf2d`.
    /// This is the const reference to the struct.
    public class Const_AffineXf2d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_AffineXf2d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly AffineXf2d UnderlyingStruct => ref *(AffineXf2d *)_UnderlyingPtr;

        internal unsafe Const_AffineXf2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_Destroy", ExactSpelling = true)]
            extern static void __MR_AffineXf2d_Destroy(_Underlying *_this);
            __MR_AffineXf2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AffineXf2d() {Dispose(false);}

        public ref readonly MR.Matrix2d A => ref UnderlyingStruct.A;

        public ref readonly MR.Vector2d B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Const_AffineXf2d(Const_AffineXf2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AffineXf2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf2d _ctor_result = __MR_AffineXf2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::AffineXf2d::AffineXf2d`.
        public unsafe Const_AffineXf2d(MR.Const_Matrix2d A, MR.Const_Vector2d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_Construct", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_Construct(MR.Const_Matrix2d._Underlying *A, MR.Const_Vector2d._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf2d _ctor_result = __MR_AffineXf2d_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf2d::translation`.
        public static unsafe MR.AffineXf2d Translation(MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_translation", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_translation(MR.Const_Vector2d._Underlying *b);
            return __MR_AffineXf2d_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf2d::linear`.
        public static unsafe MR.AffineXf2d Linear(MR.Const_Matrix2d A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_linear", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_linear(MR.Const_Matrix2d._Underlying *A);
            return __MR_AffineXf2d_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf2d::xfAround`.
        public static unsafe MR.AffineXf2d XfAround(MR.Const_Matrix2d A, MR.Const_Vector2d stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_xfAround(MR.Const_Matrix2d._Underlying *A, MR.Const_Vector2d._Underlying *stable);
            return __MR_AffineXf2d_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf2d::operator()`.
        public unsafe MR.Vector2d Call(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_call", ExactSpelling = true)]
            extern static MR.Vector2d __MR_AffineXf2d_call(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_AffineXf2d_call(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf2d::linearOnly`.
        public unsafe MR.Vector2d LinearOnly(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_linearOnly", ExactSpelling = true)]
            extern static MR.Vector2d __MR_AffineXf2d_linearOnly(_Underlying *_this, MR.Const_Vector2d._Underlying *x);
            return __MR_AffineXf2d_linearOnly(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf2d::inverse`.
        public unsafe MR.AffineXf2d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_inverse", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_inverse(_Underlying *_this);
            return __MR_AffineXf2d_inverse(_UnderlyingPtr);
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf2d operator*(MR.Const_AffineXf2d u, MR.Const_AffineXf2d v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf2d", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_mul_MR_AffineXf2d(MR.Const_AffineXf2d._Underlying *u, MR.Const_AffineXf2d._Underlying *v);
            return __MR_mul_MR_AffineXf2d(u._UnderlyingPtr, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_AffineXf2d a, MR.Const_AffineXf2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf2d(MR.Const_AffineXf2d._Underlying *a, MR.Const_AffineXf2d._Underlying *b);
            return __MR_equal_MR_AffineXf2d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_AffineXf2d a, MR.Const_AffineXf2d b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_AffineXf2d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_AffineXf2d)
                return this == (MR.Const_AffineXf2d)other;
            return false;
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf2d`.
    /// This is the non-const reference to the struct.
    public class Mut_AffineXf2d : Const_AffineXf2d
    {
        /// Get the underlying struct.
        public unsafe new ref AffineXf2d UnderlyingStruct => ref *(AffineXf2d *)_UnderlyingPtr;

        internal unsafe Mut_AffineXf2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Matrix2d A => ref UnderlyingStruct.A;

        public new ref MR.Vector2d B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Mut_AffineXf2d(Const_AffineXf2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_AffineXf2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf2d _ctor_result = __MR_AffineXf2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::AffineXf2d::AffineXf2d`.
        public unsafe Mut_AffineXf2d(MR.Const_Matrix2d A, MR.Const_Vector2d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_Construct", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_Construct(MR.Const_Matrix2d._Underlying *A, MR.Const_Vector2d._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf2d _ctor_result = __MR_AffineXf2d_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf2d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 48)]
    public struct AffineXf2d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator AffineXf2d(Const_AffineXf2d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_AffineXf2d(AffineXf2d other) => new(new Mut_AffineXf2d((Mut_AffineXf2d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Matrix2d A;

        [System.Runtime.InteropServices.FieldOffset(32)]
        public MR.Vector2d B;

        /// Generated copy constructor.
        public AffineXf2d(AffineXf2d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AffineXf2d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_DefaultConstruct();
            this = __MR_AffineXf2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::AffineXf2d::AffineXf2d`.
        public unsafe AffineXf2d(MR.Const_Matrix2d A, MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_Construct", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_Construct(MR.Const_Matrix2d._Underlying *A, MR.Const_Vector2d._Underlying *b);
            this = __MR_AffineXf2d_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf2d::translation`.
        public static unsafe MR.AffineXf2d Translation(MR.Const_Vector2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_translation", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_translation(MR.Const_Vector2d._Underlying *b);
            return __MR_AffineXf2d_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf2d::linear`.
        public static unsafe MR.AffineXf2d Linear(MR.Const_Matrix2d A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_linear", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_linear(MR.Const_Matrix2d._Underlying *A);
            return __MR_AffineXf2d_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf2d::xfAround`.
        public static unsafe MR.AffineXf2d XfAround(MR.Const_Matrix2d A, MR.Const_Vector2d stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_xfAround(MR.Const_Matrix2d._Underlying *A, MR.Const_Vector2d._Underlying *stable);
            return __MR_AffineXf2d_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf2d::operator()`.
        public unsafe MR.Vector2d Call(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_call", ExactSpelling = true)]
            extern static MR.Vector2d __MR_AffineXf2d_call(MR.AffineXf2d *_this, MR.Const_Vector2d._Underlying *x);
            fixed (MR.AffineXf2d *__ptr__this = &this)
            {
                return __MR_AffineXf2d_call(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf2d::linearOnly`.
        public unsafe MR.Vector2d LinearOnly(MR.Const_Vector2d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_linearOnly", ExactSpelling = true)]
            extern static MR.Vector2d __MR_AffineXf2d_linearOnly(MR.AffineXf2d *_this, MR.Const_Vector2d._Underlying *x);
            fixed (MR.AffineXf2d *__ptr__this = &this)
            {
                return __MR_AffineXf2d_linearOnly(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf2d::inverse`.
        public unsafe MR.AffineXf2d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf2d_inverse", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_AffineXf2d_inverse(MR.AffineXf2d *_this);
            fixed (MR.AffineXf2d *__ptr__this = &this)
            {
                return __MR_AffineXf2d_inverse(__ptr__this);
            }
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf2d operator*(MR.AffineXf2d u, MR.Const_AffineXf2d v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf2d", ExactSpelling = true)]
            extern static MR.AffineXf2d __MR_mul_MR_AffineXf2d(MR.Const_AffineXf2d._Underlying *u, MR.Const_AffineXf2d._Underlying *v);
            return __MR_mul_MR_AffineXf2d((MR.Mut_AffineXf2d._Underlying *)&u, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.AffineXf2d a, MR.AffineXf2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf2d(MR.Const_AffineXf2d._Underlying *a, MR.Const_AffineXf2d._Underlying *b);
            return __MR_equal_MR_AffineXf2d((MR.Mut_AffineXf2d._Underlying *)&a, (MR.Mut_AffineXf2d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.AffineXf2d a, MR.AffineXf2d b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.AffineXf2d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.AffineXf2d)
                return this == (MR.AffineXf2d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_AffineXf2d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_AffineXf2d`/`Const_AffineXf2d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_AffineXf2d
    {
        public readonly bool HasValue;
        internal readonly AffineXf2d Object;
        public AffineXf2d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_AffineXf2d() {HasValue = false;}
        public _InOpt_AffineXf2d(AffineXf2d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_AffineXf2d(AffineXf2d new_value) {return new(new_value);}
        public _InOpt_AffineXf2d(Const_AffineXf2d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_AffineXf2d(Const_AffineXf2d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_AffineXf2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AffineXf2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf2d`/`Const_AffineXf2d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf2d`.
    public class _InOptMut_AffineXf2d
    {
        public Mut_AffineXf2d? Opt;

        public _InOptMut_AffineXf2d() {}
        public _InOptMut_AffineXf2d(Mut_AffineXf2d value) {Opt = value;}
        public static implicit operator _InOptMut_AffineXf2d(Mut_AffineXf2d value) {return new(value);}
        public unsafe _InOptMut_AffineXf2d(ref AffineXf2d value)
        {
            fixed (AffineXf2d *value_ptr = &value)
            {
                Opt = new((Const_AffineXf2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_AffineXf2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AffineXf2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf2d`/`Const_AffineXf2d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf2d`.
    public class _InOptConst_AffineXf2d
    {
        public Const_AffineXf2d? Opt;

        public _InOptConst_AffineXf2d() {}
        public _InOptConst_AffineXf2d(Const_AffineXf2d value) {Opt = value;}
        public static implicit operator _InOptConst_AffineXf2d(Const_AffineXf2d value) {return new(value);}
        public unsafe _InOptConst_AffineXf2d(ref readonly AffineXf2d value)
        {
            fixed (AffineXf2d *value_ptr = &value)
            {
                Opt = new((Const_AffineXf2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf3f`.
    /// This is the const reference to the struct.
    public class Const_AffineXf3f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_AffineXf3f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly AffineXf3f UnderlyingStruct => ref *(AffineXf3f *)_UnderlyingPtr;

        internal unsafe Const_AffineXf3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AffineXf3f_Destroy(_Underlying *_this);
            __MR_AffineXf3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AffineXf3f() {Dispose(false);}

        public ref readonly MR.Matrix3f A => ref UnderlyingStruct.A;

        public ref readonly MR.Vector3f B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Const_AffineXf3f(Const_AffineXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AffineXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf3f _ctor_result = __MR_AffineXf3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::AffineXf3f::AffineXf3f`.
        public unsafe Const_AffineXf3f(MR.Const_Matrix3f A, MR.Const_Vector3f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Construct", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_Construct(MR.Const_Matrix3f._Underlying *A, MR.Const_Vector3f._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf3f _ctor_result = __MR_AffineXf3f_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        // Here `U == V` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::AffineXf3f::AffineXf3f`.
        public unsafe Const_AffineXf3f(MR.Const_AffineXf3d xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Construct_MR_Vector3d", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_Construct_MR_Vector3d(MR.Const_AffineXf3d._Underlying *xf);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf3f _ctor_result = __MR_AffineXf3f_Construct_MR_Vector3d(xf._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf3f::translation`.
        public static unsafe MR.AffineXf3f Translation(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_translation", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_translation(MR.Const_Vector3f._Underlying *b);
            return __MR_AffineXf3f_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf3f::linear`.
        public static unsafe MR.AffineXf3f Linear(MR.Const_Matrix3f A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_linear", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_linear(MR.Const_Matrix3f._Underlying *A);
            return __MR_AffineXf3f_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf3f::xfAround`.
        public static unsafe MR.AffineXf3f XfAround(MR.Const_Matrix3f A, MR.Const_Vector3f stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_xfAround(MR.Const_Matrix3f._Underlying *A, MR.Const_Vector3f._Underlying *stable);
            return __MR_AffineXf3f_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf3f::operator()`.
        public unsafe MR.Vector3f Call(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_call", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AffineXf3f_call(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_AffineXf3f_call(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf3f::linearOnly`.
        public unsafe MR.Vector3f LinearOnly(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_linearOnly", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AffineXf3f_linearOnly(_Underlying *_this, MR.Const_Vector3f._Underlying *x);
            return __MR_AffineXf3f_linearOnly(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf3f::inverse`.
        public unsafe MR.AffineXf3f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_inverse", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_inverse(_Underlying *_this);
            return __MR_AffineXf3f_inverse(_UnderlyingPtr);
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf3f operator*(MR.Const_AffineXf3f u, MR.Const_AffineXf3f v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_mul_MR_AffineXf3f(MR.Const_AffineXf3f._Underlying *u, MR.Const_AffineXf3f._Underlying *v);
            return __MR_mul_MR_AffineXf3f(u._UnderlyingPtr, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_AffineXf3f a, MR.Const_AffineXf3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf3f(MR.Const_AffineXf3f._Underlying *a, MR.Const_AffineXf3f._Underlying *b);
            return __MR_equal_MR_AffineXf3f(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_AffineXf3f a, MR.Const_AffineXf3f b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_AffineXf3f? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_AffineXf3f)
                return this == (MR.Const_AffineXf3f)other;
            return false;
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf3f`.
    /// This is the non-const reference to the struct.
    public class Mut_AffineXf3f : Const_AffineXf3f
    {
        /// Get the underlying struct.
        public unsafe new ref AffineXf3f UnderlyingStruct => ref *(AffineXf3f *)_UnderlyingPtr;

        internal unsafe Mut_AffineXf3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Matrix3f A => ref UnderlyingStruct.A;

        public new ref MR.Vector3f B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Mut_AffineXf3f(Const_AffineXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_AffineXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf3f _ctor_result = __MR_AffineXf3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::AffineXf3f::AffineXf3f`.
        public unsafe Mut_AffineXf3f(MR.Const_Matrix3f A, MR.Const_Vector3f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Construct", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_Construct(MR.Const_Matrix3f._Underlying *A, MR.Const_Vector3f._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf3f _ctor_result = __MR_AffineXf3f_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        // Here `U == V` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::AffineXf3f::AffineXf3f`.
        public unsafe Mut_AffineXf3f(MR.Const_AffineXf3d xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Construct_MR_Vector3d", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_Construct_MR_Vector3d(MR.Const_AffineXf3d._Underlying *xf);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.AffineXf3f _ctor_result = __MR_AffineXf3f_Construct_MR_Vector3d(xf._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf3f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 48)]
    public struct AffineXf3f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator AffineXf3f(Const_AffineXf3f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_AffineXf3f(AffineXf3f other) => new(new Mut_AffineXf3f((Mut_AffineXf3f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Matrix3f A;

        [System.Runtime.InteropServices.FieldOffset(36)]
        public MR.Vector3f B;

        /// Generated copy constructor.
        public AffineXf3f(AffineXf3f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AffineXf3f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_DefaultConstruct();
            this = __MR_AffineXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AffineXf3f::AffineXf3f`.
        public unsafe AffineXf3f(MR.Const_Matrix3f A, MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Construct", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_Construct(MR.Const_Matrix3f._Underlying *A, MR.Const_Vector3f._Underlying *b);
            this = __MR_AffineXf3f_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
        }

        // Here `U == V` doesn't seem to cause any issues in the C++ code, but we're still disabling it because it somehow gets emitted
        //   when generating the bindings, and results in duplicate functions in C#.
        /// Generated from constructor `MR::AffineXf3f::AffineXf3f`.
        public unsafe AffineXf3f(MR.Const_AffineXf3d xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_Construct_MR_Vector3d", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_Construct_MR_Vector3d(MR.Const_AffineXf3d._Underlying *xf);
            this = __MR_AffineXf3f_Construct_MR_Vector3d(xf._UnderlyingPtr);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf3f::translation`.
        public static unsafe MR.AffineXf3f Translation(MR.Const_Vector3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_translation", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_translation(MR.Const_Vector3f._Underlying *b);
            return __MR_AffineXf3f_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf3f::linear`.
        public static unsafe MR.AffineXf3f Linear(MR.Const_Matrix3f A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_linear", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_linear(MR.Const_Matrix3f._Underlying *A);
            return __MR_AffineXf3f_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf3f::xfAround`.
        public static unsafe MR.AffineXf3f XfAround(MR.Const_Matrix3f A, MR.Const_Vector3f stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_xfAround(MR.Const_Matrix3f._Underlying *A, MR.Const_Vector3f._Underlying *stable);
            return __MR_AffineXf3f_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf3f::operator()`.
        public unsafe MR.Vector3f Call(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_call", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AffineXf3f_call(MR.AffineXf3f *_this, MR.Const_Vector3f._Underlying *x);
            fixed (MR.AffineXf3f *__ptr__this = &this)
            {
                return __MR_AffineXf3f_call(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf3f::linearOnly`.
        public unsafe MR.Vector3f LinearOnly(MR.Const_Vector3f x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_linearOnly", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AffineXf3f_linearOnly(MR.AffineXf3f *_this, MR.Const_Vector3f._Underlying *x);
            fixed (MR.AffineXf3f *__ptr__this = &this)
            {
                return __MR_AffineXf3f_linearOnly(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf3f::inverse`.
        public unsafe MR.AffineXf3f Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3f_inverse", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AffineXf3f_inverse(MR.AffineXf3f *_this);
            fixed (MR.AffineXf3f *__ptr__this = &this)
            {
                return __MR_AffineXf3f_inverse(__ptr__this);
            }
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf3f operator*(MR.AffineXf3f u, MR.Const_AffineXf3f v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_mul_MR_AffineXf3f(MR.Const_AffineXf3f._Underlying *u, MR.Const_AffineXf3f._Underlying *v);
            return __MR_mul_MR_AffineXf3f((MR.Mut_AffineXf3f._Underlying *)&u, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.AffineXf3f a, MR.AffineXf3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf3f(MR.Const_AffineXf3f._Underlying *a, MR.Const_AffineXf3f._Underlying *b);
            return __MR_equal_MR_AffineXf3f((MR.Mut_AffineXf3f._Underlying *)&a, (MR.Mut_AffineXf3f._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.AffineXf3f a, MR.AffineXf3f b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.AffineXf3f b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.AffineXf3f)
                return this == (MR.AffineXf3f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_AffineXf3f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_AffineXf3f`/`Const_AffineXf3f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_AffineXf3f
    {
        public readonly bool HasValue;
        internal readonly AffineXf3f Object;
        public AffineXf3f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_AffineXf3f() {HasValue = false;}
        public _InOpt_AffineXf3f(AffineXf3f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_AffineXf3f(AffineXf3f new_value) {return new(new_value);}
        public _InOpt_AffineXf3f(Const_AffineXf3f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_AffineXf3f(Const_AffineXf3f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_AffineXf3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AffineXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf3f`/`Const_AffineXf3f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf3f`.
    public class _InOptMut_AffineXf3f
    {
        public Mut_AffineXf3f? Opt;

        public _InOptMut_AffineXf3f() {}
        public _InOptMut_AffineXf3f(Mut_AffineXf3f value) {Opt = value;}
        public static implicit operator _InOptMut_AffineXf3f(Mut_AffineXf3f value) {return new(value);}
        public unsafe _InOptMut_AffineXf3f(ref AffineXf3f value)
        {
            fixed (AffineXf3f *value_ptr = &value)
            {
                Opt = new((Const_AffineXf3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_AffineXf3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AffineXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf3f`/`Const_AffineXf3f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf3f`.
    public class _InOptConst_AffineXf3f
    {
        public Const_AffineXf3f? Opt;

        public _InOptConst_AffineXf3f() {}
        public _InOptConst_AffineXf3f(Const_AffineXf3f value) {Opt = value;}
        public static implicit operator _InOptConst_AffineXf3f(Const_AffineXf3f value) {return new(value);}
        public unsafe _InOptConst_AffineXf3f(ref readonly AffineXf3f value)
        {
            fixed (AffineXf3f *value_ptr = &value)
            {
                Opt = new((Const_AffineXf3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf3d`.
    /// This is the const reference to the struct.
    public class Const_AffineXf3d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_AffineXf3d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly AffineXf3d UnderlyingStruct => ref *(AffineXf3d *)_UnderlyingPtr;

        internal unsafe Const_AffineXf3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_Destroy", ExactSpelling = true)]
            extern static void __MR_AffineXf3d_Destroy(_Underlying *_this);
            __MR_AffineXf3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AffineXf3d() {Dispose(false);}

        public ref readonly MR.Matrix3d A => ref UnderlyingStruct.A;

        public ref readonly MR.Vector3d B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Const_AffineXf3d(Const_AffineXf3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(96);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 96);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AffineXf3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(96);
            MR.AffineXf3d _ctor_result = __MR_AffineXf3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 96);
        }

        /// Generated from constructor `MR::AffineXf3d::AffineXf3d`.
        public unsafe Const_AffineXf3d(MR.Const_Matrix3d A, MR.Const_Vector3d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_Construct", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_Construct(MR.Const_Matrix3d._Underlying *A, MR.Const_Vector3d._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(96);
            MR.AffineXf3d _ctor_result = __MR_AffineXf3d_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 96);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf3d::translation`.
        public static unsafe MR.AffineXf3d Translation(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_translation", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_translation(MR.Const_Vector3d._Underlying *b);
            return __MR_AffineXf3d_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf3d::linear`.
        public static unsafe MR.AffineXf3d Linear(MR.Const_Matrix3d A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_linear", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_linear(MR.Const_Matrix3d._Underlying *A);
            return __MR_AffineXf3d_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf3d::xfAround`.
        public static unsafe MR.AffineXf3d XfAround(MR.Const_Matrix3d A, MR.Const_Vector3d stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_xfAround(MR.Const_Matrix3d._Underlying *A, MR.Const_Vector3d._Underlying *stable);
            return __MR_AffineXf3d_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf3d::operator()`.
        public unsafe MR.Vector3d Call(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_call", ExactSpelling = true)]
            extern static MR.Vector3d __MR_AffineXf3d_call(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_AffineXf3d_call(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf3d::linearOnly`.
        public unsafe MR.Vector3d LinearOnly(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_linearOnly", ExactSpelling = true)]
            extern static MR.Vector3d __MR_AffineXf3d_linearOnly(_Underlying *_this, MR.Const_Vector3d._Underlying *x);
            return __MR_AffineXf3d_linearOnly(_UnderlyingPtr, x._UnderlyingPtr);
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf3d::inverse`.
        public unsafe MR.AffineXf3d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_inverse", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_inverse(_Underlying *_this);
            return __MR_AffineXf3d_inverse(_UnderlyingPtr);
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf3d operator*(MR.Const_AffineXf3d u, MR.Const_AffineXf3d v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf3d", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_mul_MR_AffineXf3d(MR.Const_AffineXf3d._Underlying *u, MR.Const_AffineXf3d._Underlying *v);
            return __MR_mul_MR_AffineXf3d(u._UnderlyingPtr, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_AffineXf3d a, MR.Const_AffineXf3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf3d(MR.Const_AffineXf3d._Underlying *a, MR.Const_AffineXf3d._Underlying *b);
            return __MR_equal_MR_AffineXf3d(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_AffineXf3d a, MR.Const_AffineXf3d b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_AffineXf3d? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_AffineXf3d)
                return this == (MR.Const_AffineXf3d)other;
            return false;
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf3d`.
    /// This is the non-const reference to the struct.
    public class Mut_AffineXf3d : Const_AffineXf3d
    {
        /// Get the underlying struct.
        public unsafe new ref AffineXf3d UnderlyingStruct => ref *(AffineXf3d *)_UnderlyingPtr;

        internal unsafe Mut_AffineXf3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Matrix3d A => ref UnderlyingStruct.A;

        public new ref MR.Vector3d B => ref UnderlyingStruct.B;

        /// Generated copy constructor.
        public unsafe Mut_AffineXf3d(Const_AffineXf3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(96);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 96);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_AffineXf3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(96);
            MR.AffineXf3d _ctor_result = __MR_AffineXf3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 96);
        }

        /// Generated from constructor `MR::AffineXf3d::AffineXf3d`.
        public unsafe Mut_AffineXf3d(MR.Const_Matrix3d A, MR.Const_Vector3d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_Construct", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_Construct(MR.Const_Matrix3d._Underlying *A, MR.Const_Vector3d._Underlying *b);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(96);
            MR.AffineXf3d _ctor_result = __MR_AffineXf3d_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 96);
        }
    }

    /// affine transformation: y = A*x + b, where A in VxV, and b in V
    /// Generated from class `MR::AffineXf3d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 96)]
    public struct AffineXf3d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator AffineXf3d(Const_AffineXf3d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_AffineXf3d(AffineXf3d other) => new(new Mut_AffineXf3d((Mut_AffineXf3d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Matrix3d A;

        [System.Runtime.InteropServices.FieldOffset(72)]
        public MR.Vector3d B;

        /// Generated copy constructor.
        public AffineXf3d(AffineXf3d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AffineXf3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_DefaultConstruct();
            this = __MR_AffineXf3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::AffineXf3d::AffineXf3d`.
        public unsafe AffineXf3d(MR.Const_Matrix3d A, MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_Construct", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_Construct(MR.Const_Matrix3d._Underlying *A, MR.Const_Vector3d._Underlying *b);
            this = __MR_AffineXf3d_Construct(A._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// creates translation-only transformation (with identity linear component)
        /// Generated from method `MR::AffineXf3d::translation`.
        public static unsafe MR.AffineXf3d Translation(MR.Const_Vector3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_translation", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_translation(MR.Const_Vector3d._Underlying *b);
            return __MR_AffineXf3d_translation(b._UnderlyingPtr);
        }

        /// creates linear-only transformation (without translation)
        /// Generated from method `MR::AffineXf3d::linear`.
        public static unsafe MR.AffineXf3d Linear(MR.Const_Matrix3d A)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_linear", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_linear(MR.Const_Matrix3d._Underlying *A);
            return __MR_AffineXf3d_linear(A._UnderlyingPtr);
        }

        /// creates transformation with given linear part with given stable point
        /// Generated from method `MR::AffineXf3d::xfAround`.
        public static unsafe MR.AffineXf3d XfAround(MR.Const_Matrix3d A, MR.Const_Vector3d stable)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_xfAround", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_xfAround(MR.Const_Matrix3d._Underlying *A, MR.Const_Vector3d._Underlying *stable);
            return __MR_AffineXf3d_xfAround(A._UnderlyingPtr, stable._UnderlyingPtr);
        }

        /// application of the transformation to a point
        /// Generated from method `MR::AffineXf3d::operator()`.
        public unsafe MR.Vector3d Call(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_call", ExactSpelling = true)]
            extern static MR.Vector3d __MR_AffineXf3d_call(MR.AffineXf3d *_this, MR.Const_Vector3d._Underlying *x);
            fixed (MR.AffineXf3d *__ptr__this = &this)
            {
                return __MR_AffineXf3d_call(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
        /// for example if this is a rigid transformation, then only rotates input vector
        /// Generated from method `MR::AffineXf3d::linearOnly`.
        public unsafe MR.Vector3d LinearOnly(MR.Const_Vector3d x)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_linearOnly", ExactSpelling = true)]
            extern static MR.Vector3d __MR_AffineXf3d_linearOnly(MR.AffineXf3d *_this, MR.Const_Vector3d._Underlying *x);
            fixed (MR.AffineXf3d *__ptr__this = &this)
            {
                return __MR_AffineXf3d_linearOnly(__ptr__this, x._UnderlyingPtr);
            }
        }

        /// computes inverse transformation
        /// Generated from method `MR::AffineXf3d::inverse`.
        public unsafe MR.AffineXf3d Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AffineXf3d_inverse", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_AffineXf3d_inverse(MR.AffineXf3d *_this);
            fixed (MR.AffineXf3d *__ptr__this = &this)
            {
                return __MR_AffineXf3d_inverse(__ptr__this);
            }
        }

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        /// Generated from function `MR::operator*`.
        public static unsafe MR.AffineXf3d operator*(MR.AffineXf3d u, MR.Const_AffineXf3d v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_AffineXf3d", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_mul_MR_AffineXf3d(MR.Const_AffineXf3d._Underlying *u, MR.Const_AffineXf3d._Underlying *v);
            return __MR_mul_MR_AffineXf3d((MR.Mut_AffineXf3d._Underlying *)&u, v._UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.AffineXf3d a, MR.AffineXf3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_AffineXf3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_AffineXf3d(MR.Const_AffineXf3d._Underlying *a, MR.Const_AffineXf3d._Underlying *b);
            return __MR_equal_MR_AffineXf3d((MR.Mut_AffineXf3d._Underlying *)&a, (MR.Mut_AffineXf3d._Underlying *)&b) != 0;
        }

        public static unsafe bool operator!=(MR.AffineXf3d a, MR.AffineXf3d b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.AffineXf3d b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.AffineXf3d)
                return this == (MR.AffineXf3d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_AffineXf3d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_AffineXf3d`/`Const_AffineXf3d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_AffineXf3d
    {
        public readonly bool HasValue;
        internal readonly AffineXf3d Object;
        public AffineXf3d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_AffineXf3d() {HasValue = false;}
        public _InOpt_AffineXf3d(AffineXf3d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_AffineXf3d(AffineXf3d new_value) {return new(new_value);}
        public _InOpt_AffineXf3d(Const_AffineXf3d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_AffineXf3d(Const_AffineXf3d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_AffineXf3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AffineXf3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf3d`/`Const_AffineXf3d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf3d`.
    public class _InOptMut_AffineXf3d
    {
        public Mut_AffineXf3d? Opt;

        public _InOptMut_AffineXf3d() {}
        public _InOptMut_AffineXf3d(Mut_AffineXf3d value) {Opt = value;}
        public static implicit operator _InOptMut_AffineXf3d(Mut_AffineXf3d value) {return new(value);}
        public unsafe _InOptMut_AffineXf3d(ref AffineXf3d value)
        {
            fixed (AffineXf3d *value_ptr = &value)
            {
                Opt = new((Const_AffineXf3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_AffineXf3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AffineXf3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_AffineXf3d`/`Const_AffineXf3d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `AffineXf3d`.
    public class _InOptConst_AffineXf3d
    {
        public Const_AffineXf3d? Opt;

        public _InOptConst_AffineXf3d() {}
        public _InOptConst_AffineXf3d(Const_AffineXf3d value) {Opt = value;}
        public static implicit operator _InOptConst_AffineXf3d(Const_AffineXf3d value) {return new(value);}
        public unsafe _InOptConst_AffineXf3d(ref readonly AffineXf3d value)
        {
            fixed (AffineXf3d *value_ptr = &value)
            {
                Opt = new((Const_AffineXf3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }
}
