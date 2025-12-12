public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `std::variant<MR::Sphere3f, MR::Features::Primitives::ConeSegment, MR::Features::Primitives::Plane>` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Destroy(_Underlying *_this);
                __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFrom(MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Value", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Value(_Underlying *_this);
                var __ret = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `std::variant<MR::Sphere3f, MR::Features::Primitives::ConeSegment, MR::Features::Primitives::Plane>` or nothing.
        /// This is the non-const half of the class.
        public class Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane : Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane
        {
            internal unsafe Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFrom(MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFromAnother(_Underlying *_this, MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *other);
                __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFrom(_Underlying *_this, MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *other);
                __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFrom(_UnderlyingPtr, other is not null ? other._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_MutableValue", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane`/`Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane` directly.
        public class _InOptMut_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane
        {
            public Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane? Opt;

            public _InOptMut_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane() {}
            public _InOptMut_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane`/`Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane` to pass it to the function.
        public class _InOptConst_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane
        {
            public Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane? Opt;

            public _InOptConst_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane() {}
            public _InOptConst_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(Const_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? other) {return new MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane(other);}
        }
    }
}
