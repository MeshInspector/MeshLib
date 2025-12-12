public static partial class MR
{
    public static partial class Std
    {
        /// Stores one of 3 objects: `MR::Sphere3f`, `MR::Features::Primitives::ConeSegment`, `MR::Features::Primitives::Plane`.
        /// This is the const half of the class.
        public class Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Destroy", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Destroy(_Underlying *_this);
                __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Returns the index of the stored element type. In rare cases may return -1 if this variant is "valueless by exception".
            public unsafe ulong Index()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Index", ExactSpelling = true)]
                extern static ulong __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Index(_Underlying *_this);
                return __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Index(_UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::Sphere3f`.
            public unsafe Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Const_Sphere3f value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Sphere3f", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Sphere3f(MR.Sphere3f._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Sphere3f(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 1, of type `MR::Features::Primitives::ConeSegment`.
            public unsafe Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Features.Primitives.Const_ConeSegment value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_ConeSegment(MR.Features.Primitives.ConeSegment._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_ConeSegment(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 2, of type `MR::Features::Primitives::Plane`.
            public unsafe Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Features.Primitives.Const_Plane value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_Plane", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_Plane(MR.Features.Primitives.Plane._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_Plane(value._UnderlyingPtr);
            }

            /// Returns the element 0, of type `MR::Sphere3f`, read-only. If it's not the active element, returns null.
            public unsafe MR.Const_Sphere3f? GetMRSphere3f()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Sphere3f", ExactSpelling = true)]
                extern static MR.Const_Sphere3f._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Sphere3f(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Sphere3f(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Sphere3f(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `MR::Features::Primitives::ConeSegment`, read-only. If it's not the active element, returns null.
            public unsafe MR.Features.Primitives.Const_ConeSegment? GetMRFeaturesPrimitivesConeSegment()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
                extern static MR.Features.Primitives.Const_ConeSegment._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Features_Primitives_ConeSegment(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Features_Primitives_ConeSegment(_UnderlyingPtr);
                return __ret is not null ? new MR.Features.Primitives.Const_ConeSegment(__ret, is_owning: false) : null;
            }

            /// Returns the element 2, of type `MR::Features::Primitives::Plane`, read-only. If it's not the active element, returns null.
            public unsafe MR.Features.Primitives.Const_Plane? GetMRFeaturesPrimitivesPlane()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Features_Primitives_Plane", ExactSpelling = true)]
                extern static MR.Features.Primitives.Const_Plane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Features_Primitives_Plane(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_Get_MR_Features_Primitives_Plane(_UnderlyingPtr);
                return __ret is not null ? new MR.Features.Primitives.Const_Plane(__ret, is_owning: false) : null;
            }
        }

        /// Stores one of 3 objects: `MR::Sphere3f`, `MR::Features::Primitives::ConeSegment`, `MR::Features::Primitives::Plane`.
        /// This is the non-const half of the class.
        public class Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane : Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane
        {
            internal unsafe Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *other);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFromAnother(_Underlying *_this, MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *other);
                __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 0, of type `MR::Sphere3f`.
            public unsafe Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Const_Sphere3f value, MR.Std.VariantIndex_0 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Sphere3f", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Sphere3f(MR.Sphere3f._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Sphere3f(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 1, of type `MR::Features::Primitives::ConeSegment`.
            public unsafe Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Features.Primitives.Const_ConeSegment value, MR.Std.VariantIndex_1 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_ConeSegment(MR.Features.Primitives.ConeSegment._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_ConeSegment(value._UnderlyingPtr);
            }

            /// Constructs the variant storing the element 2, of type `MR::Features::Primitives::Plane`.
            public unsafe Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(MR.Features.Primitives.Const_Plane value, MR.Std.VariantIndex_2 tag = default) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_Plane", ExactSpelling = true)]
                extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_Plane(MR.Features.Primitives.Plane._Underlying *value);
                _UnderlyingPtr = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_ConstructAs_MR_Features_Primitives_Plane(value._UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 0, of type `MR::Sphere3f`.
            public unsafe void AssignAsMRSphere3f(MR.Const_Sphere3f value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Sphere3f", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Sphere3f(_Underlying *_this, MR.Sphere3f._Underlying *value);
                __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Sphere3f(_UnderlyingPtr, value._UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 1, of type `MR::Features::Primitives::ConeSegment`.
            public unsafe void AssignAsMRFeaturesPrimitivesConeSegment(MR.Features.Primitives.Const_ConeSegment value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Features_Primitives_ConeSegment(_Underlying *_this, MR.Features.Primitives.ConeSegment._Underlying *value);
                __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Features_Primitives_ConeSegment(_UnderlyingPtr, value._UnderlyingPtr);
            }

            /// Assigns to the variant, making it store the element 2, of type `MR::Features::Primitives::Plane`.
            public unsafe void AssignAsMRFeaturesPrimitivesPlane(MR.Features.Primitives.Const_Plane value)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Features_Primitives_Plane", ExactSpelling = true)]
                extern static void __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Features_Primitives_Plane(_Underlying *_this, MR.Features.Primitives.Plane._Underlying *value);
                __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_AssignAs_MR_Features_Primitives_Plane(_UnderlyingPtr, value._UnderlyingPtr);
            }

            /// Returns the element 0, of type `MR::Sphere3f`, mutable. If it's not the active element, returns null.
            public unsafe MR.Sphere3f? GetMutableMRSphere3f()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Sphere3f", ExactSpelling = true)]
                extern static MR.Sphere3f._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Sphere3f(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Sphere3f(_UnderlyingPtr);
                return __ret is not null ? new MR.Sphere3f(__ret, is_owning: false) : null;
            }

            /// Returns the element 1, of type `MR::Features::Primitives::ConeSegment`, mutable. If it's not the active element, returns null.
            public unsafe MR.Features.Primitives.ConeSegment? GetMutableMRFeaturesPrimitivesConeSegment()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
                extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Features_Primitives_ConeSegment(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Features_Primitives_ConeSegment(_UnderlyingPtr);
                return __ret is not null ? new MR.Features.Primitives.ConeSegment(__ret, is_owning: false) : null;
            }

            /// Returns the element 2, of type `MR::Features::Primitives::Plane`, mutable. If it's not the active element, returns null.
            public unsafe MR.Features.Primitives.Plane? GetMutableMRFeaturesPrimitivesPlane()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Features_Primitives_Plane", ExactSpelling = true)]
                extern static MR.Features.Primitives.Plane._Underlying *__MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Features_Primitives_Plane(_Underlying *_this);
                var __ret = __MR_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane_GetMutable_MR_Features_Primitives_Plane(_UnderlyingPtr);
                return __ret is not null ? new MR.Features.Primitives.Plane(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane`/`Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane` directly.
        public class _InOptMut_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane
        {
            public Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? Opt;

            public _InOptMut_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane() {}
            public _InOptMut_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane value) {Opt = value;}
            public static implicit operator _InOptMut_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane value) {return new(value);}
        }

        /// This is used for optional parameters of class `Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane`/`Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane` to pass it to the function.
        public class _InOptConst_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane
        {
            public Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane? Opt;

            public _InOptConst_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane() {}
            public _InOptConst_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane value) {Opt = value;}
            public static implicit operator _InOptConst_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane(Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane value) {return new(value);}
        }
    }
}
