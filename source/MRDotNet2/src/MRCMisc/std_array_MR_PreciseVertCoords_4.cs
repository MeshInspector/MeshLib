public static partial class MR
{
    public static partial class Std
    {
        /// A fixed-size array of `MR::PreciseVertCoords` of size 4.
        /// This is the const half of the class.
        public class Const_Array_MRPreciseVertCoords_4 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Array_MRPreciseVertCoords_4(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_Destroy", ExactSpelling = true)]
                extern static void __MR_std_array_MR_PreciseVertCoords_4_Destroy(_Underlying *_this);
                __MR_std_array_MR_PreciseVertCoords_4_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Array_MRPreciseVertCoords_4() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Array_MRPreciseVertCoords_4() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords_4._Underlying *__MR_std_array_MR_PreciseVertCoords_4_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords_4_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Array_MRPreciseVertCoords_4(MR.Std.Const_Array_MRPreciseVertCoords_4 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords_4._Underlying *__MR_std_array_MR_PreciseVertCoords_4_ConstructFromAnother(MR.Std.Array_MRPreciseVertCoords_4._Underlying *other);
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords_4_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Const_PreciseVertCoords At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_At", ExactSpelling = true)]
                extern static MR.Const_PreciseVertCoords._Underlying *__MR_std_array_MR_PreciseVertCoords_4_At(_Underlying *_this, ulong i);
                return new(__MR_std_array_MR_PreciseVertCoords_4_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, read-only.
            public unsafe MR.Const_PreciseVertCoords? Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_Data", ExactSpelling = true)]
                extern static MR.Const_PreciseVertCoords._Underlying *__MR_std_array_MR_PreciseVertCoords_4_Data(_Underlying *_this);
                var __ret = __MR_std_array_MR_PreciseVertCoords_4_Data(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_PreciseVertCoords(__ret, is_owning: false) : null;
            }
        }

        /// A fixed-size array of `MR::PreciseVertCoords` of size 4.
        /// This is the non-const half of the class.
        public class Array_MRPreciseVertCoords_4 : Const_Array_MRPreciseVertCoords_4
        {
            internal unsafe Array_MRPreciseVertCoords_4(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Array_MRPreciseVertCoords_4() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords_4._Underlying *__MR_std_array_MR_PreciseVertCoords_4_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords_4_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Array_MRPreciseVertCoords_4(MR.Std.Const_Array_MRPreciseVertCoords_4 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_MRPreciseVertCoords_4._Underlying *__MR_std_array_MR_PreciseVertCoords_4_ConstructFromAnother(MR.Std.Array_MRPreciseVertCoords_4._Underlying *other);
                _UnderlyingPtr = __MR_std_array_MR_PreciseVertCoords_4_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Array_MRPreciseVertCoords_4 other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_array_MR_PreciseVertCoords_4_AssignFromAnother(_Underlying *_this, MR.Std.Array_MRPreciseVertCoords_4._Underlying *other);
                __MR_std_array_MR_PreciseVertCoords_4_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.PreciseVertCoords MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_MutableAt", ExactSpelling = true)]
                extern static MR.PreciseVertCoords._Underlying *__MR_std_array_MR_PreciseVertCoords_4_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_array_MR_PreciseVertCoords_4_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, mutable.
            public unsafe MR.PreciseVertCoords? MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_PreciseVertCoords_4_MutableData", ExactSpelling = true)]
                extern static MR.PreciseVertCoords._Underlying *__MR_std_array_MR_PreciseVertCoords_4_MutableData(_Underlying *_this);
                var __ret = __MR_std_array_MR_PreciseVertCoords_4_MutableData(_UnderlyingPtr);
                return __ret is not null ? new MR.PreciseVertCoords(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Array_MRPreciseVertCoords_4` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Array_MRPreciseVertCoords_4`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_MRPreciseVertCoords_4`/`Const_Array_MRPreciseVertCoords_4` directly.
        public class _InOptMut_Array_MRPreciseVertCoords_4
        {
            public Array_MRPreciseVertCoords_4? Opt;

            public _InOptMut_Array_MRPreciseVertCoords_4() {}
            public _InOptMut_Array_MRPreciseVertCoords_4(Array_MRPreciseVertCoords_4 value) {Opt = value;}
            public static implicit operator _InOptMut_Array_MRPreciseVertCoords_4(Array_MRPreciseVertCoords_4 value) {return new(value);}
        }

        /// This is used for optional parameters of class `Array_MRPreciseVertCoords_4` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Array_MRPreciseVertCoords_4`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_MRPreciseVertCoords_4`/`Const_Array_MRPreciseVertCoords_4` to pass it to the function.
        public class _InOptConst_Array_MRPreciseVertCoords_4
        {
            public Const_Array_MRPreciseVertCoords_4? Opt;

            public _InOptConst_Array_MRPreciseVertCoords_4() {}
            public _InOptConst_Array_MRPreciseVertCoords_4(Const_Array_MRPreciseVertCoords_4 value) {Opt = value;}
            public static implicit operator _InOptConst_Array_MRPreciseVertCoords_4(Const_Array_MRPreciseVertCoords_4 value) {return new(value);}
        }
    }
}
