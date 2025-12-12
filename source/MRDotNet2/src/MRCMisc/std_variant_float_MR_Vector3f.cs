public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 2 objects: `float`, `MR::Vector3f`.
        /// This is the const half of the class.
        public class Const_Variant_Float_MRVector3f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_Float_MRVector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_float_MR_Vector3f_Destroy(_Underlying *_this);
                __MR_std_variant_float_MR_Vector3f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_Float_MRVector3f() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_Float_MRVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_Float_MRVector3f(MR.Std.Const_Variant_Float_MRVector3f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_ConstructFromAnother(MR.Std.Variant_Float_MRVector3f._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_float_MR_Vector3f_Index(_Underlying *_this);
                return __MR_std_variant_float_MR_Vector3f_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `float`.
            public unsafe Const_Variant_Float_MRVector3f(float value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_ConstructAs_float", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_ConstructAs_float(float value);
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_ConstructAs_float(value);
            }

            /// Constructs the variant storing the element 1, of type `MR::Vector3f`.
            public unsafe Const_Variant_Float_MRVector3f(MR.Vector3f value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_ConstructAs_MR_Vector3f", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_ConstructAs_MR_Vector3f(MR.Vector3f value);
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_ConstructAs_MR_Vector3f(value);
            }

            /// Returns the element 0, of type `float`, read-only. If it's not the active element, returns null.
            public unsafe float? GetFloat()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_Get_float", ExactSpelling = true)]
                extern static float *__MR_std_variant_float_MR_Vector3f_Get_float(_Underlying *_this);
                var __ret = __MR_std_variant_float_MR_Vector3f_Get_float(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// Returns the element 1, of type `MR::Vector3f`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Vector3f? GetMRVector3f()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_Get_MR_Vector3f", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_std_variant_float_MR_Vector3f_Get_MR_Vector3f(_Underlying *_this);
                var __ret = __MR_std_variant_float_MR_Vector3f_Get_MR_Vector3f(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Vector3f(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 2 objects: `float`, `MR::Vector3f`.
        /// This is the non-const half of the class.
        public class Variant_Float_MRVector3f : Const_Variant_Float_MRVector3f
        {
            internal unsafe Variant_Float_MRVector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_Float_MRVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_Float_MRVector3f(MR.Std.Const_Variant_Float_MRVector3f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_ConstructFromAnother(MR.Std.Variant_Float_MRVector3f._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Variant_Float_MRVector3f other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_float_MR_Vector3f_AssignFromAnother(_Underlying *_this, MR.Std.Variant_Float_MRVector3f._Underlying *other);
                __MR_std_variant_float_MR_Vector3f_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `float`.
            public unsafe Variant_Float_MRVector3f(float value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_ConstructAs_float", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_ConstructAs_float(float value);
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_ConstructAs_float(value);
            }

            /// Constructs the variant storing the element 1, of type `MR::Vector3f`.
            public unsafe Variant_Float_MRVector3f(MR.Vector3f value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_ConstructAs_MR_Vector3f", ExactSpelling = true)]
                extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_std_variant_float_MR_Vector3f_ConstructAs_MR_Vector3f(MR.Vector3f value);
                _UnderlyingPtr = __MR_std_variant_float_MR_Vector3f_ConstructAs_MR_Vector3f(value);
            }

            /// Assigns to the variant, making it store the element 0, of type `float`.
            public unsafe void AssignAsFloat(float value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_AssignAs_float", ExactSpelling = true)]
                extern static void __MR_std_variant_float_MR_Vector3f_AssignAs_float(_Underlying *_this, float value);
                __MR_std_variant_float_MR_Vector3f_AssignAs_float(_UnderlyingPtr, value);
            }

            /// Assigns to the variant, making it store the element 1, of type `MR::Vector3f`.
            public unsafe void AssignAsMRVector3f(MR.Vector3f value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_AssignAs_MR_Vector3f", ExactSpelling = true)]
                extern static void __MR_std_variant_float_MR_Vector3f_AssignAs_MR_Vector3f(_Underlying *_this, MR.Vector3f value);
                __MR_std_variant_float_MR_Vector3f_AssignAs_MR_Vector3f(_UnderlyingPtr, value);
            }

            /// Returns the element 0, of type `float`, mutable. If it's not the active element, returns null.
            public unsafe MR.Misc.Ref<float>? GetMutableFloat()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_GetMutable_float", ExactSpelling = true)]
                extern static float *__MR_std_variant_float_MR_Vector3f_GetMutable_float(_Underlying *_this);
                var __ret = __MR_std_variant_float_MR_Vector3f_GetMutable_float(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<float>(__ret) : null;
            }

            /// Returns the element 1, of type `MR::Vector3f`, mutable. If it's not the active element, returns null.
            public unsafe MR.Mut_Vector3f? GetMutableMRVector3f()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_float_MR_Vector3f_GetMutable_MR_Vector3f", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_std_variant_float_MR_Vector3f_GetMutable_MR_Vector3f(_Underlying *_this);
                var __ret = __MR_std_variant_float_MR_Vector3f_GetMutable_MR_Vector3f(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Vector3f(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Variant_Float_MRVector3f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_Float_MRVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_Float_MRVector3f`/`Const_Variant_Float_MRVector3f` directly.
        public class _InOptMut_Variant_Float_MRVector3f
        {
            public Variant_Float_MRVector3f? Opt;

            public _InOptMut_Variant_Float_MRVector3f() {}
            public _InOptMut_Variant_Float_MRVector3f(Variant_Float_MRVector3f value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_Float_MRVector3f(Variant_Float_MRVector3f value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_Float_MRVector3f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_Float_MRVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_Float_MRVector3f`/`Const_Variant_Float_MRVector3f` to pass it to the function.
        public class _InOptConst_Variant_Float_MRVector3f
        {
            public Const_Variant_Float_MRVector3f? Opt;

            public _InOptConst_Variant_Float_MRVector3f() {}
            public _InOptConst_Variant_Float_MRVector3f(Const_Variant_Float_MRVector3f value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_Float_MRVector3f(Const_Variant_Float_MRVector3f value) {return new(value);}
        }
    }
}
