public static partial class MR
{
    public static partial class Std
    {
        /// A fixed-size array of `MR::AffineXf3d` of size 4.
        /// This is the const reference to the struct.
        public class Const_Array_MRAffineXf3d_4 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            /// Get the underlying struct.
            public unsafe ref readonly Array_MRAffineXf3d_4 UnderlyingStruct => ref *(Array_MRAffineXf3d_4 *)_UnderlyingPtr;

            internal unsafe Const_Array_MRAffineXf3d_4(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_AffineXf3d_4_Destroy", ExactSpelling = true)]
                extern static void __MR_std_array_MR_AffineXf3d_4_Destroy(_Underlying *_this);
                __MR_std_array_MR_AffineXf3d_4_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Array_MRAffineXf3d_4() {Dispose(false);}

            public ref readonly MR.ArrayAffineXf3d4 Elems => ref UnderlyingStruct.Elems;

            /// Generated copy constructor.
            public unsafe Const_Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
                extern static _Underlying *__MR_Alloc(nuint size);
                _UnderlyingPtr = __MR_Alloc(384);
                System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 384);
            }
        }

        /// A fixed-size array of `MR::AffineXf3d` of size 4.
        /// This is the non-const reference to the struct.
        public class Mut_Array_MRAffineXf3d_4 : Const_Array_MRAffineXf3d_4
        {
            /// Get the underlying struct.
            public unsafe new ref Array_MRAffineXf3d_4 UnderlyingStruct => ref *(Array_MRAffineXf3d_4 *)_UnderlyingPtr;

            internal unsafe Mut_Array_MRAffineXf3d_4(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new ref MR.ArrayAffineXf3d4 Elems => ref UnderlyingStruct.Elems;

            /// Generated copy constructor.
            public unsafe Mut_Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
                extern static _Underlying *__MR_Alloc(nuint size);
                _UnderlyingPtr = __MR_Alloc(384);
                System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 384);
            }
        }

        /// A fixed-size array of `MR::AffineXf3d` of size 4.
        /// This is the by-value version of the struct.
        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 384)]
        public struct Array_MRAffineXf3d_4
        {
            /// Copy contents from a wrapper class to this struct.
            public static implicit operator Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 other) => other.UnderlyingStruct;
            /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
            public unsafe static implicit operator Mut_Array_MRAffineXf3d_4(Array_MRAffineXf3d_4 other) => new(new Mut_Array_MRAffineXf3d_4((Mut_Array_MRAffineXf3d_4._Underlying *)&other, is_owning: false));

            [System.Runtime.InteropServices.FieldOffset(0)]
            public MR.ArrayAffineXf3d4 Elems;

            /// Generated copy constructor.
            public Array_MRAffineXf3d_4(Array_MRAffineXf3d_4 _other) {this = _other;}
        }

        /// This is used as a function parameter when passing `Mut_Array_MRAffineXf3d_4` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
        /// Usage:
        /// * Pass an instance of `Mut_Array_MRAffineXf3d_4`/`Const_Array_MRAffineXf3d_4` to copy it into the function.
        /// * Pass `null` to use the default argument
        public readonly ref struct _InOpt_Array_MRAffineXf3d_4
        {
            public readonly bool HasValue;
            internal readonly Array_MRAffineXf3d_4 Object;
            public Array_MRAffineXf3d_4 Value{
                get
                {
                    System.Diagnostics.Trace.Assert(HasValue);
                    return Object;
                }
            }

            public _InOpt_Array_MRAffineXf3d_4() {HasValue = false;}
            public _InOpt_Array_MRAffineXf3d_4(Array_MRAffineXf3d_4 new_value) {HasValue = true; Object = new_value;}
            public static implicit operator _InOpt_Array_MRAffineXf3d_4(Array_MRAffineXf3d_4 new_value) {return new(new_value);}
            public _InOpt_Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
            public static implicit operator _InOpt_Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 new_value) {return new(new_value);}
        }

        /// This is used for optional parameters of class `Mut_Array_MRAffineXf3d_4` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Array_MRAffineXf3d_4`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Mut_Array_MRAffineXf3d_4`/`Const_Array_MRAffineXf3d_4` directly.
        /// * Pass `new(ref ...)` to pass a reference to `Array_MRAffineXf3d_4`.
        public class _InOptMut_Array_MRAffineXf3d_4
        {
            public Mut_Array_MRAffineXf3d_4? Opt;

            public _InOptMut_Array_MRAffineXf3d_4() {}
            public _InOptMut_Array_MRAffineXf3d_4(Mut_Array_MRAffineXf3d_4 value) {Opt = value;}
            public static implicit operator _InOptMut_Array_MRAffineXf3d_4(Mut_Array_MRAffineXf3d_4 value) {return new(value);}
            public unsafe _InOptMut_Array_MRAffineXf3d_4(ref Array_MRAffineXf3d_4 value)
            {
                fixed (Array_MRAffineXf3d_4 *value_ptr = &value)
                {
                    Opt = new((Const_Array_MRAffineXf3d_4._Underlying *)value_ptr, is_owning: false);
                }
            }
        }

        /// This is used for optional parameters of class `Mut_Array_MRAffineXf3d_4` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Array_MRAffineXf3d_4`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Mut_Array_MRAffineXf3d_4`/`Const_Array_MRAffineXf3d_4` to pass it to the function.
        /// * Pass `new(ref ...)` to pass a reference to `Array_MRAffineXf3d_4`.
        public class _InOptConst_Array_MRAffineXf3d_4
        {
            public Const_Array_MRAffineXf3d_4? Opt;

            public _InOptConst_Array_MRAffineXf3d_4() {}
            public _InOptConst_Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 value) {Opt = value;}
            public static implicit operator _InOptConst_Array_MRAffineXf3d_4(Const_Array_MRAffineXf3d_4 value) {return new(value);}
            public unsafe _InOptConst_Array_MRAffineXf3d_4(ref readonly Array_MRAffineXf3d_4 value)
            {
                fixed (Array_MRAffineXf3d_4 *value_ptr = &value)
                {
                    Opt = new((Const_Array_MRAffineXf3d_4._Underlying *)value_ptr, is_owning: false);
                }
            }
        }
    }
}
