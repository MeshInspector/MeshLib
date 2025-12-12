public static partial class MR
{
    public static partial class Features
    {
        // Describes a single feature produced by `forEachSubfeature()`.
        /// Generated from class `MR::Features::SubfeatureInfo`.
        /// This is the const half of the class.
        public class Const_SubfeatureInfo : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_SubfeatureInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_Destroy", ExactSpelling = true)]
                extern static void __MR_Features_SubfeatureInfo_Destroy(_Underlying *_this);
                __MR_Features_SubfeatureInfo_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_SubfeatureInfo() {Dispose(false);}

            // A user-friendly name.
            public unsafe MR.Std.Const_StringView Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_Get_name", ExactSpelling = true)]
                    extern static MR.Std.Const_StringView._Underlying *__MR_Features_SubfeatureInfo_Get_name(_Underlying *_this);
                    return new(__MR_Features_SubfeatureInfo_Get_name(_UnderlyingPtr), is_owning: false);
                }
            }

            // Whether the feature has infinite length.
            public unsafe bool IsInfinite
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_Get_isInfinite", ExactSpelling = true)]
                    extern static bool *__MR_Features_SubfeatureInfo_Get_isInfinite(_Underlying *_this);
                    return *__MR_Features_SubfeatureInfo_Get_isInfinite(_UnderlyingPtr);
                }
            }

            // Call this to create this subfeature.
            // Pass the same feature you passed to `forEachSubfeature()`, or a DIFFERENT one to create the same kind of subfeature relative to that.
            public unsafe MR.Std.Const_Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef Create
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_Get_create", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef._Underlying *__MR_Features_SubfeatureInfo_Get_create(_Underlying *_this);
                    return new(__MR_Features_SubfeatureInfo_Get_create(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_SubfeatureInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_Features_SubfeatureInfo_DefaultConstruct();
            }

            /// Constructs `MR::Features::SubfeatureInfo` elementwise.
            public unsafe Const_SubfeatureInfo(ReadOnlySpan<char> name, bool isInfinite, MR.Std._ByValue_Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef create) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_ConstructFrom", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_ConstructFrom(byte *name, byte *name_end, byte isInfinite, MR.Misc._PassBy create_pass_by, MR.Std.Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef._Underlying *create);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_Features_SubfeatureInfo_ConstructFrom(__ptr_name, __ptr_name + __len_name, isInfinite ? (byte)1 : (byte)0, create.PassByMode, create.Value is not null ? create.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from constructor `MR::Features::SubfeatureInfo::SubfeatureInfo`.
            public unsafe Const_SubfeatureInfo(MR.Features._ByValue_SubfeatureInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Features.SubfeatureInfo._Underlying *_other);
                _UnderlyingPtr = __MR_Features_SubfeatureInfo_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        // Describes a single feature produced by `forEachSubfeature()`.
        /// Generated from class `MR::Features::SubfeatureInfo`.
        /// This is the non-const half of the class.
        public class SubfeatureInfo : Const_SubfeatureInfo
        {
            internal unsafe SubfeatureInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // A user-friendly name.
            public new unsafe MR.Std.StringView Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_GetMutable_name", ExactSpelling = true)]
                    extern static MR.Std.StringView._Underlying *__MR_Features_SubfeatureInfo_GetMutable_name(_Underlying *_this);
                    return new(__MR_Features_SubfeatureInfo_GetMutable_name(_UnderlyingPtr), is_owning: false);
                }
            }

            // Whether the feature has infinite length.
            public new unsafe ref bool IsInfinite
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_GetMutable_isInfinite", ExactSpelling = true)]
                    extern static bool *__MR_Features_SubfeatureInfo_GetMutable_isInfinite(_Underlying *_this);
                    return ref *__MR_Features_SubfeatureInfo_GetMutable_isInfinite(_UnderlyingPtr);
                }
            }

            // Call this to create this subfeature.
            // Pass the same feature you passed to `forEachSubfeature()`, or a DIFFERENT one to create the same kind of subfeature relative to that.
            public new unsafe MR.Std.Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef Create
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_GetMutable_create", ExactSpelling = true)]
                    extern static MR.Std.Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef._Underlying *__MR_Features_SubfeatureInfo_GetMutable_create(_Underlying *_this);
                    return new(__MR_Features_SubfeatureInfo_GetMutable_create(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe SubfeatureInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_Features_SubfeatureInfo_DefaultConstruct();
            }

            /// Constructs `MR::Features::SubfeatureInfo` elementwise.
            public unsafe SubfeatureInfo(ReadOnlySpan<char> name, bool isInfinite, MR.Std._ByValue_Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef create) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_ConstructFrom", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_ConstructFrom(byte *name, byte *name_end, byte isInfinite, MR.Misc._PassBy create_pass_by, MR.Std.Function_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneFuncFromConstStdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlaneRef._Underlying *create);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_Features_SubfeatureInfo_ConstructFrom(__ptr_name, __ptr_name + __len_name, isInfinite ? (byte)1 : (byte)0, create.PassByMode, create.Value is not null ? create.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from constructor `MR::Features::SubfeatureInfo::SubfeatureInfo`.
            public unsafe SubfeatureInfo(MR.Features._ByValue_SubfeatureInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Features.SubfeatureInfo._Underlying *_other);
                _UnderlyingPtr = __MR_Features_SubfeatureInfo_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Features::SubfeatureInfo::operator=`.
            public unsafe MR.Features.SubfeatureInfo Assign(MR.Features._ByValue_SubfeatureInfo _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_SubfeatureInfo_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Features.SubfeatureInfo._Underlying *__MR_Features_SubfeatureInfo_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Features.SubfeatureInfo._Underlying *_other);
                return new(__MR_Features_SubfeatureInfo_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `SubfeatureInfo` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `SubfeatureInfo`/`Const_SubfeatureInfo` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_SubfeatureInfo
        {
            internal readonly Const_SubfeatureInfo? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_SubfeatureInfo() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_SubfeatureInfo(Const_SubfeatureInfo new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_SubfeatureInfo(Const_SubfeatureInfo arg) {return new(arg);}
            public _ByValue_SubfeatureInfo(MR.Misc._Moved<SubfeatureInfo> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_SubfeatureInfo(MR.Misc._Moved<SubfeatureInfo> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `SubfeatureInfo` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_SubfeatureInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SubfeatureInfo`/`Const_SubfeatureInfo` directly.
        public class _InOptMut_SubfeatureInfo
        {
            public SubfeatureInfo? Opt;

            public _InOptMut_SubfeatureInfo() {}
            public _InOptMut_SubfeatureInfo(SubfeatureInfo value) {Opt = value;}
            public static implicit operator _InOptMut_SubfeatureInfo(SubfeatureInfo value) {return new(value);}
        }

        /// This is used for optional parameters of class `SubfeatureInfo` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_SubfeatureInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SubfeatureInfo`/`Const_SubfeatureInfo` to pass it to the function.
        public class _InOptConst_SubfeatureInfo
        {
            public Const_SubfeatureInfo? Opt;

            public _InOptConst_SubfeatureInfo() {}
            public _InOptConst_SubfeatureInfo(Const_SubfeatureInfo value) {Opt = value;}
            public static implicit operator _InOptConst_SubfeatureInfo(Const_SubfeatureInfo value) {return new(value);}
        }

        // Decomposes a feature to its subfeatures, by calling `func()` on each subfeature.
        // This only returns the direct subfeatures. You can call this recursively to obtain all features,
        //   but beware of duplicates (there's no easy way to filter them).
        /// Generated from function `MR::Features::forEachSubfeature`.
        public static unsafe void ForEachSubfeature(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane feature, MR.Std.Const_Function_VoidFuncFromConstMRFeaturesSubfeatureInfoRef func)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_forEachSubfeature", ExactSpelling = true)]
            extern static void __MR_Features_forEachSubfeature(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *feature, MR.Std.Const_Function_VoidFuncFromConstMRFeaturesSubfeatureInfoRef._Underlying *func);
            __MR_Features_forEachSubfeature(feature._UnderlyingPtr, func._UnderlyingPtr);
        }
    }
}
