public static partial class MR
{
    public static partial class Std
    {
        /// A fixed-size array of `MR::SimpleVolumeMinMax` of size 3.
        /// This is the const half of the class.
        public class Const_Array_MRSimpleVolumeMinMax_3 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Array_MRSimpleVolumeMinMax_3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_Destroy", ExactSpelling = true)]
                extern static void __MR_std_array_MR_SimpleVolumeMinMax_3_Destroy(_Underlying *_this);
                __MR_std_array_MR_SimpleVolumeMinMax_3_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Array_MRSimpleVolumeMinMax_3() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Array_MRSimpleVolumeMinMax_3() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_MR_SimpleVolumeMinMax_3_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Array_MRSimpleVolumeMinMax_3(MR.Std._ByValue_Array_MRSimpleVolumeMinMax_3 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *other);
                _UnderlyingPtr = __MR_std_array_MR_SimpleVolumeMinMax_3_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The element at a specific index, read-only.
            public unsafe MR.Const_SimpleVolumeMinMax At(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_At", ExactSpelling = true)]
                extern static MR.Const_SimpleVolumeMinMax._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_At(_Underlying *_this, ulong i);
                return new(__MR_std_array_MR_SimpleVolumeMinMax_3_At(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, read-only.
            public unsafe MR.Const_SimpleVolumeMinMax? Data()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_Data", ExactSpelling = true)]
                extern static MR.Const_SimpleVolumeMinMax._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_Data(_Underlying *_this);
                var __ret = __MR_std_array_MR_SimpleVolumeMinMax_3_Data(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_SimpleVolumeMinMax(__ret, is_owning: false) : null;
            }
        }

        /// A fixed-size array of `MR::SimpleVolumeMinMax` of size 3.
        /// This is the non-const half of the class.
        public class Array_MRSimpleVolumeMinMax_3 : Const_Array_MRSimpleVolumeMinMax_3
        {
            internal unsafe Array_MRSimpleVolumeMinMax_3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Array_MRSimpleVolumeMinMax_3() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_DefaultConstruct();
                _UnderlyingPtr = __MR_std_array_MR_SimpleVolumeMinMax_3_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Array_MRSimpleVolumeMinMax_3(MR.Std._ByValue_Array_MRSimpleVolumeMinMax_3 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *other);
                _UnderlyingPtr = __MR_std_array_MR_SimpleVolumeMinMax_3_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Array_MRSimpleVolumeMinMax_3 other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_array_MR_SimpleVolumeMinMax_3_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *other);
                __MR_std_array_MR_SimpleVolumeMinMax_3_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The element at a specific index, mutable.
            public unsafe MR.SimpleVolumeMinMax MutableAt(ulong i)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_MutableAt", ExactSpelling = true)]
                extern static MR.SimpleVolumeMinMax._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_MutableAt(_Underlying *_this, ulong i);
                return new(__MR_std_array_MR_SimpleVolumeMinMax_3_MutableAt(_UnderlyingPtr, i), is_owning: false);
            }

            /// Returns a pointer to the continuous storage that holds all elements, mutable.
            public unsafe MR.SimpleVolumeMinMax? MutableData()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_array_MR_SimpleVolumeMinMax_3_MutableData", ExactSpelling = true)]
                extern static MR.SimpleVolumeMinMax._Underlying *__MR_std_array_MR_SimpleVolumeMinMax_3_MutableData(_Underlying *_this);
                var __ret = __MR_std_array_MR_SimpleVolumeMinMax_3_MutableData(_UnderlyingPtr);
                return __ret is not null ? new MR.SimpleVolumeMinMax(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Array_MRSimpleVolumeMinMax_3` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Array_MRSimpleVolumeMinMax_3`/`Const_Array_MRSimpleVolumeMinMax_3` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Array_MRSimpleVolumeMinMax_3
        {
            internal readonly Const_Array_MRSimpleVolumeMinMax_3? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Array_MRSimpleVolumeMinMax_3() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Array_MRSimpleVolumeMinMax_3(Const_Array_MRSimpleVolumeMinMax_3 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Array_MRSimpleVolumeMinMax_3(Const_Array_MRSimpleVolumeMinMax_3 arg) {return new(arg);}
            public _ByValue_Array_MRSimpleVolumeMinMax_3(MR.Misc._Moved<Array_MRSimpleVolumeMinMax_3> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Array_MRSimpleVolumeMinMax_3(MR.Misc._Moved<Array_MRSimpleVolumeMinMax_3> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Array_MRSimpleVolumeMinMax_3` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Array_MRSimpleVolumeMinMax_3`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_MRSimpleVolumeMinMax_3`/`Const_Array_MRSimpleVolumeMinMax_3` directly.
        public class _InOptMut_Array_MRSimpleVolumeMinMax_3
        {
            public Array_MRSimpleVolumeMinMax_3? Opt;

            public _InOptMut_Array_MRSimpleVolumeMinMax_3() {}
            public _InOptMut_Array_MRSimpleVolumeMinMax_3(Array_MRSimpleVolumeMinMax_3 value) {Opt = value;}
            public static implicit operator _InOptMut_Array_MRSimpleVolumeMinMax_3(Array_MRSimpleVolumeMinMax_3 value) {return new(value);}
        }

        /// This is used for optional parameters of class `Array_MRSimpleVolumeMinMax_3` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Array_MRSimpleVolumeMinMax_3`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Array_MRSimpleVolumeMinMax_3`/`Const_Array_MRSimpleVolumeMinMax_3` to pass it to the function.
        public class _InOptConst_Array_MRSimpleVolumeMinMax_3
        {
            public Const_Array_MRSimpleVolumeMinMax_3? Opt;

            public _InOptConst_Array_MRSimpleVolumeMinMax_3() {}
            public _InOptConst_Array_MRSimpleVolumeMinMax_3(Const_Array_MRSimpleVolumeMinMax_3 value) {Opt = value;}
            public static implicit operator _InOptConst_Array_MRSimpleVolumeMinMax_3(Const_Array_MRSimpleVolumeMinMax_3 value) {return new(value);}
        }
    }
}
