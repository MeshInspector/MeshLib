public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 7 objects: `MR::Polynomial<float, 0>`, `MR::Polynomial<float, 1>`, `MR::Polynomial<float, 2>`, `MR::Polynomial<float, 3>`, `MR::Polynomial<float, 4>`, `MR::Polynomial<float, 5>`, `MR::Polynomial<float, 6>`.
        /// This is the const half of the class.
        public class Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Destroy(_Underlying *_this);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR.Std._ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Index(_Underlying *_this);
                return __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::Polynomial<float, 0>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_0 value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_0", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_0(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_0._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_0(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `MR::Polynomial<float, 1>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_1 value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_1", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_1(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_1._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_1(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 2, of type `MR::Polynomial<float, 2>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_2 value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_2", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_2(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_2._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_2(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 3, of type `MR::Polynomial<float, 3>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_3 value, MR.Std.VariantIndex_3 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_3", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_3(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_3._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_3(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 4, of type `MR::Polynomial<float, 4>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_4 value, MR.Std.VariantIndex_4 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_4", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_4(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_4._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_4(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 5, of type `MR::Polynomial<float, 5>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_5 value, MR.Std.VariantIndex_5 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_5", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_5(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_5._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_5(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 6, of type `MR::Polynomial<float, 6>`.
            public unsafe Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_6 value, MR.Std.VariantIndex_6 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_6", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_6(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_6._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_6(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::Polynomial<float, 0>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_0? GetMRPolynomialFloat0()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_0", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_0._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_0(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_0(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_0(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `MR::Polynomial<float, 1>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_1? GetMRPolynomialFloat1()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_1", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_1._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_1(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_1(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_1(__ret, is_owning: false) : null;
            }

            /// Returns the element 2, of type `MR::Polynomial<float, 2>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_2? GetMRPolynomialFloat2()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_2", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_2._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_2(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_2(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_2(__ret, is_owning: false) : null;
            }

            /// Returns the element 3, of type `MR::Polynomial<float, 3>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_3? GetMRPolynomialFloat3()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_3", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_3._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_3(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_3(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_3(__ret, is_owning: false) : null;
            }

            /// Returns the element 4, of type `MR::Polynomial<float, 4>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_4? GetMRPolynomialFloat4()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_4", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_4._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_4(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_4(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_4(__ret, is_owning: false) : null;
            }

            /// Returns the element 5, of type `MR::Polynomial<float, 5>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_5? GetMRPolynomialFloat5()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_5", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_5._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_5(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_5(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_5(__ret, is_owning: false) : null;
            }

            /// Returns the element 6, of type `MR::Polynomial<float, 6>`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Polynomial_Float_6? GetMRPolynomialFloat6()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_6", ExactSpelling = true)]
                extern static MR.Const_Polynomial_Float_6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_6(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_Get_MR_Polynomial_float_6(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Polynomial_Float_6(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 7 objects: `MR::Polynomial<float, 0>`, `MR::Polynomial<float, 1>`, `MR::Polynomial<float, 2>`, `MR::Polynomial<float, 3>`, `MR::Polynomial<float, 4>`, `MR::Polynomial<float, 5>`, `MR::Polynomial<float, 6>`.
        /// This is the non-const half of the class.
        public class Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 : Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6
        {
            internal unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR.Std._ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *other);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `MR::Polynomial<float, 0>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_0 value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_0", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_0(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_0._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_0(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 1, of type `MR::Polynomial<float, 1>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_1 value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_1", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_1(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_1._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_1(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 2, of type `MR::Polynomial<float, 2>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_2 value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_2", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_2(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_2._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_2(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 3, of type `MR::Polynomial<float, 3>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_3 value, MR.Std.VariantIndex_3 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_3", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_3(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_3._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_3(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 4, of type `MR::Polynomial<float, 4>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_4 value, MR.Std.VariantIndex_4 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_4", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_4(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_4._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_4(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 5, of type `MR::Polynomial<float, 5>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_5 value, MR.Std.VariantIndex_5 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_5", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_5(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_5._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_5(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 6, of type `MR::Polynomial<float, 6>`.
            public unsafe Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR._ByValue_Polynomial_Float_6 value, MR.Std.VariantIndex_6 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_6", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_6(MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_6._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_ConstructAs_MR_Polynomial_float_6(value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::Polynomial<float, 0>`.
            public unsafe void AssignAsMRPolynomialFloat0(MR._ByValue_Polynomial_Float_0 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_0", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_0(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_0._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_0(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 1, of type `MR::Polynomial<float, 1>`.
            public unsafe void AssignAsMRPolynomialFloat1(MR._ByValue_Polynomial_Float_1 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_1", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_1(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_1._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_1(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 2, of type `MR::Polynomial<float, 2>`.
            public unsafe void AssignAsMRPolynomialFloat2(MR._ByValue_Polynomial_Float_2 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_2", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_2(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_2._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_2(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 3, of type `MR::Polynomial<float, 3>`.
            public unsafe void AssignAsMRPolynomialFloat3(MR._ByValue_Polynomial_Float_3 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_3", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_3(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_3._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_3(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 4, of type `MR::Polynomial<float, 4>`.
            public unsafe void AssignAsMRPolynomialFloat4(MR._ByValue_Polynomial_Float_4 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_4", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_4(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_4._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_4(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 5, of type `MR::Polynomial<float, 5>`.
            public unsafe void AssignAsMRPolynomialFloat5(MR._ByValue_Polynomial_Float_5 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_5", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_5(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_5._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_5(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Assigns to the variant, making it store the element 6, of type `MR::Polynomial<float, 6>`.
            public unsafe void AssignAsMRPolynomialFloat6(MR._ByValue_Polynomial_Float_6 value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_6", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_6(_Underlying *_this, MR.Misc._PassBy value_pass_by, MR.Polynomial_Float_6._Underlying *value);
                __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_AssignAs_MR_Polynomial_float_6(_UnderlyingPtr, value.PassByMode, value.Value is not null ? value.Value._UnderlyingPtr : null);
            }

            /// Returns the element 0, of type `MR::Polynomial<float, 0>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_0? GetMutableMRPolynomialFloat0()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_0", ExactSpelling = true)]
                extern static MR.Polynomial_Float_0._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_0(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_0(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_0(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `MR::Polynomial<float, 1>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_1? GetMutableMRPolynomialFloat1()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_1", ExactSpelling = true)]
                extern static MR.Polynomial_Float_1._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_1(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_1(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_1(__ret, is_owning: false) : null;
            }

            /// Returns the element 2, of type `MR::Polynomial<float, 2>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_2? GetMutableMRPolynomialFloat2()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_2", ExactSpelling = true)]
                extern static MR.Polynomial_Float_2._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_2(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_2(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_2(__ret, is_owning: false) : null;
            }

            /// Returns the element 3, of type `MR::Polynomial<float, 3>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_3? GetMutableMRPolynomialFloat3()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_3", ExactSpelling = true)]
                extern static MR.Polynomial_Float_3._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_3(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_3(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_3(__ret, is_owning: false) : null;
            }

            /// Returns the element 4, of type `MR::Polynomial<float, 4>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_4? GetMutableMRPolynomialFloat4()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_4", ExactSpelling = true)]
                extern static MR.Polynomial_Float_4._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_4(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_4(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_4(__ret, is_owning: false) : null;
            }

            /// Returns the element 5, of type `MR::Polynomial<float, 5>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_5? GetMutableMRPolynomialFloat5()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_5", ExactSpelling = true)]
                extern static MR.Polynomial_Float_5._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_5(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_5(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_5(__ret, is_owning: false) : null;
            }

            /// Returns the element 6, of type `MR::Polynomial<float, 6>`, mutable. If it's not the active element, returns null.
            public unsafe MR.Polynomial_Float_6? GetMutableMRPolynomialFloat6()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_6", ExactSpelling = true)]
                extern static MR.Polynomial_Float_6._Underlying *__MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_6(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Polynomial_float_0_MR_Polynomial_float_1_MR_Polynomial_float_2_MR_Polynomial_float_3_MR_Polynomial_float_4_MR_Polynomial_float_5_MR_Polynomial_float_6_GetMutable_MR_Polynomial_float_6(_UnderlyingPtr);
                return __ret is not null ? new MR.Polynomial_Float_6(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6`/`Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6
        {
            internal readonly Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 arg) {return new(arg);}
            public _ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR.Misc._Moved<Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(MR.Misc._Moved<Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6`/`Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6` directly.
        public class _InOptMut_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6
        {
            public Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6? Opt;

            public _InOptMut_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6() {}
            public _InOptMut_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6`/`Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6` to pass it to the function.
        public class _InOptConst_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6
        {
            public Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6? Opt;

            public _InOptConst_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6() {}
            public _InOptConst_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6(Const_Variant_MRPolynomialFloat0_MRPolynomialFloat1_MRPolynomialFloat2_MRPolynomialFloat3_MRPolynomialFloat4_MRPolynomialFloat5_MRPolynomialFloat6 value) {return new(value);}
        }
    }
}
