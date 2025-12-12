public static partial class MR
{
    public static partial class Std
    {
        /// A fixed-size array of `MR::PreciseVertCoords2` of size 3.
        /// This is the const half of the class.
        public class Const_Array_MRPreciseVertCoords2_3 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Array_MRPreciseVertCoords2_3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_Destroy", ExactSpelling = true)]
                extern static void __MR_std_array_MR_PreciseVertCoords2_3_Destroy(_Underlying *_this);
                __MR_std_array_MR_PreciseVertCoords2_3_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Array_MRPreciseVertCoords2_3() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Array_MRPreciseVertCoords2_3() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords2_3._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords2_3_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Array_MRPreciseVertCoords2_3(MR.Std.Const_Array_MRPreciseVertCoords2_3 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords2_3._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_ConstructFromAnother(MR.Std.Array_MRPreciseVertCoords2_3._Underlying *other);
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords2_3_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Const_PreciseVertCoords2 At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_At", ExactSpelling = true)]
                extern static MR.Const_PreciseVertCoords2._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_At(_Underlying *_this, ulong i);
                return new(__MR_std_array_MR_PreciseVertCoords2_3_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, read-only.
            public unsafe MR.Const_PreciseVertCoords2? Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_Data", ExactSpelling = true)]
                extern static MR.Const_PreciseVertCoords2._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_Data(_Underlying *_this);
                var __ret = __MR_std_array_MR_PreciseVertCoords2_3_Data(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_PreciseVertCoords2(__ret, is_owning: false) : null;
            }
        }

        /// A fixed-size array of `MR::PreciseVertCoords2` of size 3.
        /// This is the non-const half of the class.
        public class Array_MRPreciseVertCoords2_3 : Const_Array_MRPreciseVertCoords2_3
        {
            internal unsafe Array_MRPreciseVertCoords2_3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Array_MRPreciseVertCoords2_3() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords2_3._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords2_3_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Array_MRPreciseVertCoords2_3(MR.Std.Const_Array_MRPreciseVertCoords2_3 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords2_3._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_ConstructFromAnother(MR.Std.Array_MRPreciseVertCoords2_3._Underlying *other);
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords2_3_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Array_MRPreciseVertCoords2_3 other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_array_MR_PreciseVertCoords2_3_AssignFromAnother(_Underlying *_this, MR.Std.Array_MRPreciseVertCoords2_3._Underlying *other);
                __MR_std_array_MR_PreciseVertCoords2_3_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.PreciseVertCoords2 MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_MutableAt", ExactSpelling = true)]
                extern static MR.PreciseVertCoords2._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_array_MR_PreciseVertCoords2_3_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, mutable.
            public unsafe MR.PreciseVertCoords2? MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords2_3_MutableData", ExactSpelling = true)]
                extern static MR.PreciseVertCoords2._Underlying *__MR_std_array_MR_PreciseVertCoords2_3_MutableData(_Underlying *_this);
                var __ret = __MR_std_array_MR_PreciseVertCoords2_3_MutableData(_UnderlyingPtr);
                return __ret is not null ? new MR.PreciseVertCoords2(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Array_MRPreciseVertCoords2_3` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Array_MRPreciseVertCoords2_3`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_MRPreciseVertCoords2_3`/`Const_Array_MRPreciseVertCoords2_3` directly.
        public class _InOptMut_Array_MRPreciseVertCoords2_3
        {
            public Array_MRPreciseVertCoords2_3? Opt;

            public _InOptMut_Array_MRPreciseVertCoords2_3() {}
            public _InOptMut_Array_MRPreciseVertCoords2_3(Array_MRPreciseVertCoords2_3 value) {Opt = value;}
            public static implicit operator _InOptMut_Array_MRPreciseVertCoords2_3(Array_MRPreciseVertCoords2_3 value) {return new(value);}
        }

        /// This is used for optional parameters of class `Array_MRPreciseVertCoords2_3` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Array_MRPreciseVertCoords2_3`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_MRPreciseVertCoords2_3`/`Const_Array_MRPreciseVertCoords2_3` to pass it to the function.
        public class _InOptConst_Array_MRPreciseVertCoords2_3
        {
            public Const_Array_MRPreciseVertCoords2_3? Opt;

            public _InOptConst_Array_MRPreciseVertCoords2_3() {}
            public _InOptConst_Array_MRPreciseVertCoords2_3(Const_Array_MRPreciseVertCoords2_3 value) {Opt = value;}
            public static implicit operator _InOptConst_Array_MRPreciseVertCoords2_3(Const_Array_MRPreciseVertCoords2_3 value) {return new(value);}
        }
    }
}
