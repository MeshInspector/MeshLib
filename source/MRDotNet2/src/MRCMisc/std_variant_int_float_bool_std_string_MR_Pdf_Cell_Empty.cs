public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 5 objects: `int`, `float`, `bool`, `std::string`, `MR::Pdf::Cell::Empty`.
        /// This is the const half of the class.
        public class Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Destroy(_Underlying *_this);
                __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(MR.Std._ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Index(_Underlying *_this);
                return __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `int`.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(int value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_int", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_int(int value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_int(value);
            }

            /// Constructs the variant storing the element 1, of type `float`.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(float value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_float", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_float(float value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_float(value);
            }

            /// Constructs the variant storing the element 2, of type `bool`.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(bool value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_bool", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_bool(byte value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_bool(value ? (byte)1 : (byte)0);
            }

            /// Constructs the variant storing the element 3, of type `std::string`.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(ReadOnlySpan<char> value, MR.Std.VariantIndex_3 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_std_string", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_std_string(byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_std_string(__ptr_value, __ptr_value + __len_value);
                }
            }

            /// Constructs the variant storing the element 4, of type `MR::Pdf::Cell::Empty`.
            public unsafe Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(MR.Pdf.Cell.Const_Empty value, MR.Std.VariantIndex_4 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_MR_Pdf_Cell_Empty", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_MR_Pdf_Cell_Empty(MR.Pdf.Cell.Empty._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_MR_Pdf_Cell_Empty(value._UnderlyingPtr);
            }

            /// Returns the element 0, of type `int`, read-only. If it's not the active element, returns null.
            public unsafe int? GetInt()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_int", ExactSpelling = true)]
                extern static int *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_int(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_int(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// Returns the element 1, of type `float`, read-only. If it's not the active element, returns null.
            public unsafe float? GetFloat()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_float", ExactSpelling = true)]
                extern static float *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_float(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_float(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// Returns the element 2, of type `bool`, read-only. If it's not the active element, returns null.
            public unsafe bool? GetBool()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_bool", ExactSpelling = true)]
                extern static bool *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_bool(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_bool(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }

            /// Returns the element 3, of type `std::string`, read-only. If it's not the active element, returns null.
            public unsafe MR.Std.Const_String? GetStdString()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_std_string", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_std_string(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_std_string(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_String(__ret, is_owning: false) : null;
            }

            /// Returns the element 4, of type `MR::Pdf::Cell::Empty`, read-only. If it's not the active element, returns null.
            public unsafe MR.Pdf.Cell.Const_Empty? GetMRPdfCellEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_MR_Pdf_Cell_Empty", ExactSpelling = true)]
                extern static MR.Pdf.Cell.Const_Empty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_MR_Pdf_Cell_Empty(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_Get_MR_Pdf_Cell_Empty(_UnderlyingPtr);
                return __ret is not null ? new MR.Pdf.Cell.Const_Empty(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 5 objects: `int`, `float`, `bool`, `std::string`, `MR::Pdf::Cell::Empty`.
        /// This is the non-const half of the class.
        public class Variant_Int_Float_Bool_StdString_MRPdfCellEmpty : Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty
        {
            internal unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(MR.Std._ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *other);
                __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs the variant storing the element 0, of type `int`.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(int value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_int", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_int(int value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_int(value);
            }

            /// Constructs the variant storing the element 1, of type `float`.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(float value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_float", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_float(float value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_float(value);
            }

            /// Constructs the variant storing the element 2, of type `bool`.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(bool value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_bool", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_bool(byte value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_bool(value ? (byte)1 : (byte)0);
            }

            /// Constructs the variant storing the element 3, of type `std::string`.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(ReadOnlySpan<char> value, MR.Std.VariantIndex_3 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_std_string", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_std_string(byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_std_string(__ptr_value, __ptr_value + __len_value);
                }
            }

            /// Constructs the variant storing the element 4, of type `MR::Pdf::Cell::Empty`.
            public unsafe Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(MR.Pdf.Cell.Const_Empty value, MR.Std.VariantIndex_4 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_MR_Pdf_Cell_Empty", ExactSpelling = true)]
                extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_MR_Pdf_Cell_Empty(MR.Pdf.Cell.Empty._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_ConstructAs_MR_Pdf_Cell_Empty(value._UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 0, of type `int`.
            public unsafe void AssignAsInt(int value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_int", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_int(_Underlying *_this, int value);
                __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_int(_UnderlyingPtr, value);
            }

            /// Assigns to the variant, making it store the element 1, of type `float`.
            public unsafe void AssignAsFloat(float value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_float", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_float(_Underlying *_this, float value);
                __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_float(_UnderlyingPtr, value);
            }

            /// Assigns to the variant, making it store the element 2, of type `bool`.
            public unsafe void AssignAsBool(bool value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_bool", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_bool(_Underlying *_this, byte value);
                __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_bool(_UnderlyingPtr, value ? (byte)1 : (byte)0);
            }

            /// Assigns to the variant, making it store the element 3, of type `std::string`.
            public unsafe void AssignAsStdString(ReadOnlySpan<char> value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_std_string", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_std_string(_Underlying *_this, byte *value, byte *value_end);
                byte[] __bytes_value = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(value.Length)];
                int __len_value = System.Text.Encoding.UTF8.GetBytes(value, __bytes_value);
                fixed (byte *__ptr_value = __bytes_value)
                {
                    __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_std_string(_UnderlyingPtr, __ptr_value, __ptr_value + __len_value);
                }
            }

            /// Assigns to the variant, making it store the element 4, of type `MR::Pdf::Cell::Empty`.
            public unsafe void AssignAsMRPdfCellEmpty(MR.Pdf.Cell.Const_Empty value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_MR_Pdf_Cell_Empty", ExactSpelling = true)]
                extern static void __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_MR_Pdf_Cell_Empty(_Underlying *_this, MR.Pdf.Cell.Empty._Underlying *value);
                __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_AssignAs_MR_Pdf_Cell_Empty(_UnderlyingPtr, value._UnderlyingPtr);
            }

            /// Returns the element 0, of type `int`, mutable. If it's not the active element, returns null.
            public unsafe MR.Misc.Ref<int>? GetMutableInt()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_int", ExactSpelling = true)]
                extern static int *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_int(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_int(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<int>(__ret) : null;
            }

            /// Returns the element 1, of type `float`, mutable. If it's not the active element, returns null.
            public unsafe MR.Misc.Ref<float>? GetMutableFloat()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_float", ExactSpelling = true)]
                extern static float *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_float(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_float(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<float>(__ret) : null;
            }

            /// Returns the element 2, of type `bool`, mutable. If it's not the active element, returns null.
            public unsafe MR.Misc.Ref<bool>? GetMutableBool()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_bool", ExactSpelling = true)]
                extern static bool *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_bool(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_bool(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<bool>(__ret) : null;
            }

            /// Returns the element 3, of type `std::string`, mutable. If it's not the active element, returns null.
            public unsafe MR.Std.String? GetMutableStdString()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_std_string", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_std_string(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_std_string(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.String(__ret, is_owning: false) : null;
            }

            /// Returns the element 4, of type `MR::Pdf::Cell::Empty`, mutable. If it's not the active element, returns null.
            public unsafe MR.Pdf.Cell.Empty? GetMutableMRPdfCellEmpty()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_MR_Pdf_Cell_Empty", ExactSpelling = true)]
                extern static MR.Pdf.Cell.Empty._Underlying *__MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_MR_Pdf_Cell_Empty(_Underlying *_this);
                var __ret = __MR_std_variant_int_float_bool_std_string_MR_Pdf_Cell_Empty_GetMutable_MR_Pdf_Cell_Empty(_UnderlyingPtr);
                return __ret is not null ? new MR.Pdf.Cell.Empty(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Variant_Int_Float_Bool_StdString_MRPdfCellEmpty` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Variant_Int_Float_Bool_StdString_MRPdfCellEmpty`/`Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty
        {
            internal readonly Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty arg) {return new(arg);}
            public _ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(MR.Misc._Moved<Variant_Int_Float_Bool_StdString_MRPdfCellEmpty> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(MR.Misc._Moved<Variant_Int_Float_Bool_StdString_MRPdfCellEmpty> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Variant_Int_Float_Bool_StdString_MRPdfCellEmpty` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_Int_Float_Bool_StdString_MRPdfCellEmpty`/`Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty` directly.
        public class _InOptMut_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty
        {
            public Variant_Int_Float_Bool_StdString_MRPdfCellEmpty? Opt;

            public _InOptMut_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty() {}
            public _InOptMut_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(Variant_Int_Float_Bool_StdString_MRPdfCellEmpty value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(Variant_Int_Float_Bool_StdString_MRPdfCellEmpty value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_Int_Float_Bool_StdString_MRPdfCellEmpty` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_Int_Float_Bool_StdString_MRPdfCellEmpty`/`Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty` to pass it to the function.
        public class _InOptConst_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty
        {
            public Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty? Opt;

            public _InOptConst_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty() {}
            public _InOptConst_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty(Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty value) {return new(value);}
        }
    }
}
