public static partial class MR
{
    public static partial class Std
    {
        /// Wraps a pointer to a single shared reference-counted heap-allocated `void`.
        /// This is the const half of the class.
        public class Const_SharedPtr_Void : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_SharedPtr_Void(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_Destroy", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_Destroy(_Underlying *_this);
                __MR_std_shared_ptr_void_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_SharedPtr_Void() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_SharedPtr_Void() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_DefaultConstruct();
                _UnderlyingPtr = __MR_std_shared_ptr_void_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_SharedPtr_Void(MR.Std._ByValue_SharedPtr_Void other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.SharedPtr_Void._Underlying *other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Returns the stored pointer, possibly null.
            /// Returns a mutable pointer.
            public unsafe void *Get()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_Get", ExactSpelling = true)]
                extern static void *__MR_std_shared_ptr_void_Get(_Underlying *_this);
                return __MR_std_shared_ptr_void_Get(_UnderlyingPtr);
            }

            /// How many shared pointers share the managed object. Zero if no object is being managed.
            /// This being zero usually conincides with `MR_std_shared_ptr_void_Get()` returning null, but is ultimately orthogonal.
            /// Note that in multithreaded environments, the only safe way to use this number is comparing it with zero. Positive values might change by the time you get to use them.
            public unsafe int UseCount()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_void_UseCount(_Underlying *_this);
                return __MR_std_shared_ptr_void_UseCount(_UnderlyingPtr);
            }

            /// Create a new instance, storing a non-owning pointer.
            /// Parameter `ptr` is a mutable pointer.
            public unsafe Const_SharedPtr_Void(void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructNonOwning", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructNonOwning(void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructNonOwning(ptr);
            }

            /// The aliasing constructor. Create a new instance, copying ownership from an existing shared pointer and storing an arbitrary raw pointer.
            /// The input pointer can be reinterpreted from any other `std::shared_ptr<T>` to avoid constructing a new `std::shared_ptr<void>`.
            /// Parameter `ptr` is a mutable pointer.
            public unsafe Const_SharedPtr_Void(MR.Std._ByValue_SharedPtr_ConstVoid ownership, void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructAliasing", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *ownership, void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructAliasing(ownership.PassByMode, ownership.Value is not null ? ownership.Value._UnderlyingPtr : null, ptr);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_CircleObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CircleObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CircleObject(MR.Misc._PassBy _other_pass_by, MR.CircleObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CircleObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_CircleObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_SphereObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SphereObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SphereObject(MR.Misc._PassBy _other_pass_by, MR.SphereObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SphereObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_SphereObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ConeObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ConeObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ConeObject(MR.Misc._PassBy _other_pass_by, MR.ConeObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ConeObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ConeObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_CylinderObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CylinderObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CylinderObject(MR.Misc._PassBy _other_pass_by, MR.CylinderObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CylinderObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_CylinderObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_LineObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_LineObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_LineObject(MR.Misc._PassBy _other_pass_by, MR.LineObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_LineObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_LineObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PlaneObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PlaneObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PlaneObject(MR.Misc._PassBy _other_pass_by, MR.PlaneObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PlaneObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PlaneObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PointObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointObject(MR.Misc._PassBy _other_pass_by, MR.PointObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PointObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_FeatureObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FeatureObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FeatureObject(MR.Misc._PassBy _other_pass_by, MR.FeatureObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FeatureObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_FeatureObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PointMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.PointMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PointMeasurementObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectComparableWithReference _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference(MR.Misc._PassBy _other_pass_by, MR.ObjectComparableWithReference._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectComparableWithReference _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_DistanceMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.DistanceMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_DistanceMeasurementObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_RadiusMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.RadiusMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_RadiusMeasurementObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_MeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_MeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_MeasurementObject(MR.Misc._PassBy _other_pass_by, MR.MeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_MeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_MeasurementObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_AngleMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AngleMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AngleMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.AngleMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AngleMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_AngleMeasurementObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectMesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMesh", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMesh(MR.Misc._PassBy _other_pass_by, MR.ObjectMesh._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMesh(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectMesh _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectVoxels _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectVoxels", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectVoxels(MR.Misc._PassBy _other_pass_by, MR.ObjectVoxels._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectVoxels(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectVoxels _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectMeshHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMeshHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMeshHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectMeshHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMeshHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectMeshHolder _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectDistanceMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectDistanceMap", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectDistanceMap(MR.Misc._PassBy _other_pass_by, MR.ObjectDistanceMap._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectDistanceMap(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectDistanceMap _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectLines _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLines", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLines(MR.Misc._PassBy _other_pass_by, MR.ObjectLines._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLines(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectLines _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectLinesHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLinesHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLinesHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectLinesHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLinesHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectLinesHolder _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectGcode _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectGcode", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectGcode(MR.Misc._PassBy _other_pass_by, MR.ObjectGcode._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectGcode(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectGcode _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectLabel _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLabel", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLabel(MR.Misc._PassBy _other_pass_by, MR.ObjectLabel._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLabel(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectLabel _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectPointsHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPointsHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPointsHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectPointsHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPointsHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectPointsHolder _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectPoints _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPoints(MR.Misc._PassBy _other_pass_by, MR.ObjectPoints._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPoints(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectPoints _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_VisualObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_VisualObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_VisualObject(MR.Misc._PassBy _other_pass_by, MR.VisualObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_VisualObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_VisualObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_SceneRootObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SceneRootObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SceneRootObject(MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SceneRootObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_SceneRootObject _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ObjectChildrenHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectChildrenHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ObjectChildrenHolder _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_Object _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Object", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Object(MR.Misc._PassBy _other_pass_by, MR.Object._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Object(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_Object _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_Mesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Mesh", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Mesh(MR.Misc._PassBy _other_pass_by, MR.Mesh._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Mesh(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_Mesh _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PointCloud _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointCloud", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointCloud(MR.Misc._PassBy _other_pass_by, MR.PointCloud._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointCloud(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PointCloud _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_Polyline3 _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Polyline3", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Polyline3(MR.Misc._PassBy _other_pass_by, MR.Polyline3._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Polyline3(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_Polyline3 _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_BasicUiRenderTask _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_BasicUiRenderTask", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_BasicUiRenderTask(MR.Misc._PassBy _other_pass_by, MR.BasicUiRenderTask._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_BasicUiRenderTask(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_BasicUiRenderTask _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangVoxelSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangVoxelSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangVoxelSelectionAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeActiveBoxAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction(MR.Misc._PassBy _other_pass_by, MR.ChangeActiveBoxAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeActiveBoxAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeColoringType _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeColoringType", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeColoringType(MR.Misc._PassBy _other_pass_by, MR.ChangeColoringType._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeColoringType(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeColoringType _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeDualMarchingCubesAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction(MR.Misc._PassBy _other_pass_by, MR.ChangeDualMarchingCubesAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeDualMarchingCubesAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeFacesColorMapAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction(MR.Misc._PassBy _other_pass_by, MR.ChangeFacesColorMapAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeFacesColorMapAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeGridAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeGridAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeGridAction(MR.Misc._PassBy _other_pass_by, MR.ChangeGridAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeGridAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeGridAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeIsoAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeIsoAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeIsoAction(MR.Misc._PassBy _other_pass_by, MR.ChangeIsoAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeIsoAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeIsoAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeLabelAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLabelAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLabelAction(MR.Misc._PassBy _other_pass_by, MR.ChangeLabelAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLabelAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeLabelAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeLinesColorMapAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction(MR.Misc._PassBy _other_pass_by, MR.ChangeLinesColorMapAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeLinesColorMapAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshCreasesAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshCreasesAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshCreasesAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshDataAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshDataAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshDataAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshEdgeSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshEdgeSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshEdgeSelectionAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshFaceSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshFaceSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshFaceSelectionAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshPointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshPointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshPointsAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshTexturePerFaceAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTexturePerFaceAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshTexturePerFaceAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshTopologyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTopologyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshTopologyAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeMeshUVCoordsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshUVCoordsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeMeshUVCoordsAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeNameAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeNameAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeNameAction(MR.Misc._PassBy _other_pass_by, MR.ChangeNameAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeNameAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeNameAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeObjectAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeObjectAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeObjectColorAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectColorAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeObjectColorAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeObjectSelectedAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectSelectedAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeObjectSelectedAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeObjectVisibilityAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectVisibilityAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeObjectVisibilityAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeOneNormalInCloudAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction(MR.Misc._PassBy _other_pass_by, MR.ChangeOneNormalInCloudAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeOneNormalInCloudAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeOnePointInCloudAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInCloudAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeOnePointInCloudAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeOnePointInPolylineAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInPolylineAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeOnePointInPolylineAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePointCloudAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePointCloudAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePointCloudNormalsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudNormalsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePointCloudNormalsAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePointCloudPointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudPointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePointCloudPointsAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePointPointSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointPointSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePointPointSelectionAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePolylineAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineAction(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePolylineAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePolylinePointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction(MR.Misc._PassBy _other_pass_by, MR.ChangePolylinePointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePolylinePointsAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangePolylineTopologyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineTopologyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangePolylineTopologyAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeScaleAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeScaleAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeScaleAction(MR.Misc._PassBy _other_pass_by, MR.ChangeScaleAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeScaleAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeScaleAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeSceneAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneAction(MR.Misc._PassBy _other_pass_by, MR.ChangeSceneAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeSceneAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeSceneObjectsOrder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder(MR.Misc._PassBy _other_pass_by, MR.ChangeSceneObjectsOrder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeSceneObjectsOrder _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeSurfaceAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction(MR.Misc._PassBy _other_pass_by, MR.ChangeSurfaceAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeSurfaceAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeTextureAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeTextureAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeTextureAction(MR.Misc._PassBy _other_pass_by, MR.ChangeTextureAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeTextureAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeTextureAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeVisualizePropertyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction(MR.Misc._PassBy _other_pass_by, MR.ChangeVisualizePropertyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeVisualizePropertyAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_ChangeXfAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeXfAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeXfAction(MR.Misc._PassBy _other_pass_by, MR.ChangeXfAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeXfAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_ChangeXfAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_CombinedHistoryAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CombinedHistoryAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CombinedHistoryAction(MR.Misc._PassBy _other_pass_by, MR.CombinedHistoryAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CombinedHistoryAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_CombinedHistoryAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshDataAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshDataAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshDataAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshPointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshPointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshPointsAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshTopologyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshTopologyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PartialChangeMeshTopologyAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_HistoryAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_HistoryAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_HistoryAction(MR.Misc._PassBy _other_pass_by, MR.HistoryAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_HistoryAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_HistoryAction _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_Matrix_Float _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Matrix_float", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Matrix_float(MR.Misc._PassBy _other_pass_by, MR.Matrix_Float._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Matrix_float(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_Matrix_Float _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_RectIndexer _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RectIndexer", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RectIndexer(MR.Misc._PassBy _other_pass_by, MR.RectIndexer._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RectIndexer(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_RectIndexer _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_DistanceMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMap", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMap(MR.Misc._PassBy _other_pass_by, MR.DistanceMap._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMap(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_DistanceMap _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR.Std._ByValue_Vector_StdString _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_std_vector_std_string", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_std_vector_std_string(MR.Misc._PassBy _other_pass_by, MR.Std.Vector_StdString._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_std_vector_std_string(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR.Std._ByValue_Vector_StdString _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_FastWindingNumber _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FastWindingNumber", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FastWindingNumber(MR.Misc._PassBy _other_pass_by, MR.FastWindingNumber._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FastWindingNumber(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_FastWindingNumber _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_IFastWindingNumber _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IFastWindingNumber", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IFastWindingNumber(MR.Misc._PassBy _other_pass_by, MR.IFastWindingNumber._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IFastWindingNumber(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_IFastWindingNumber _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_OpenVdbFloatGrid _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid(MR.Misc._PassBy _other_pass_by, MR.OpenVdbFloatGrid._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_OpenVdbFloatGrid _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_PointsToMeshProjector _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointsToMeshProjector", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointsToMeshProjector(MR.Misc._PassBy _other_pass_by, MR.PointsToMeshProjector._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointsToMeshProjector(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_PointsToMeshProjector _other) {return new(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe Const_SharedPtr_Void(MR._ByValue_IPointsToMeshProjector _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector(MR.Misc._PassBy _other_pass_by, MR.IPointsToMeshProjector._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator Const_SharedPtr_Void(MR._ByValue_IPointsToMeshProjector _other) {return new(_other);}
        }

        /// Wraps a pointer to a single shared reference-counted heap-allocated `void`.
        /// This is the non-const half of the class.
        public class SharedPtr_Void : Const_SharedPtr_Void
        {
            internal unsafe SharedPtr_Void(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe SharedPtr_Void() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_DefaultConstruct();
                _UnderlyingPtr = __MR_std_shared_ptr_void_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe SharedPtr_Void(MR.Std._ByValue_SharedPtr_Void other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.SharedPtr_Void._Underlying *other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_SharedPtr_Void other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.SharedPtr_Void._Underlying *other);
                __MR_std_shared_ptr_void_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Create a new instance, storing a non-owning pointer.
            /// Parameter `ptr` is a mutable pointer.
            public unsafe SharedPtr_Void(void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructNonOwning", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructNonOwning(void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructNonOwning(ptr);
            }

            /// Overwrite the existing instance with a non-owning pointer. The previously owned object, if any, has its reference count decremented.
            /// Parameter `ptr` is a mutable pointer.
            public unsafe void Assign(void *ptr)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignNonOwning", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignNonOwning(_Underlying *_this, void *ptr);
                __MR_std_shared_ptr_void_AssignNonOwning(_UnderlyingPtr, ptr);
            }

            /// The aliasing constructor. Create a new instance, copying ownership from an existing shared pointer and storing an arbitrary raw pointer.
            /// The input pointer can be reinterpreted from any other `std::shared_ptr<T>` to avoid constructing a new `std::shared_ptr<void>`.
            /// Parameter `ptr` is a mutable pointer.
            public unsafe SharedPtr_Void(MR.Std._ByValue_SharedPtr_ConstVoid ownership, void *ptr) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructAliasing", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *ownership, void *ptr);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructAliasing(ownership.PassByMode, ownership.Value is not null ? ownership.Value._UnderlyingPtr : null, ptr);
            }

            /// The aliasing assignment. Overwrite an existing instance, copying ownership from an existing shared pointer and storing an arbitrary raw pointer.
            /// The input pointer can be reinterpreted from any other `std::shared_ptr<T>` to avoid constructing a new `std::shared_ptr<void>`.
            /// Parameter `ptr` is a mutable pointer.
            public unsafe void AssignAliasing(MR.Std._ByValue_SharedPtr_ConstVoid ownership, void *ptr)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignAliasing", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignAliasing(_Underlying *_this, MR.Misc._PassBy ownership_pass_by, MR.Std.SharedPtr_ConstVoid._Underlying *ownership, void *ptr);
                __MR_std_shared_ptr_void_AssignAliasing(_UnderlyingPtr, ownership.PassByMode, ownership.Value is not null ? ownership.Value._UnderlyingPtr : null, ptr);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_CircleObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CircleObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CircleObject(MR.Misc._PassBy _other_pass_by, MR.CircleObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CircleObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_CircleObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_CircleObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CircleObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CircleObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CircleObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CircleObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_SphereObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SphereObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SphereObject(MR.Misc._PassBy _other_pass_by, MR.SphereObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SphereObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_SphereObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_SphereObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_SphereObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_SphereObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SphereObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_SphereObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ConeObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ConeObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ConeObject(MR.Misc._PassBy _other_pass_by, MR.ConeObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ConeObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ConeObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ConeObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ConeObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ConeObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ConeObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ConeObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_CylinderObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CylinderObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CylinderObject(MR.Misc._PassBy _other_pass_by, MR.CylinderObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CylinderObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_CylinderObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_CylinderObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CylinderObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CylinderObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CylinderObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CylinderObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_LineObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_LineObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_LineObject(MR.Misc._PassBy _other_pass_by, MR.LineObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_LineObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_LineObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_LineObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_LineObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_LineObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LineObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_LineObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PlaneObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PlaneObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PlaneObject(MR.Misc._PassBy _other_pass_by, MR.PlaneObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PlaneObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PlaneObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PlaneObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PlaneObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PlaneObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PlaneObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PlaneObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PointObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointObject(MR.Misc._PassBy _other_pass_by, MR.PointObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PointObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PointObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_FeatureObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FeatureObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FeatureObject(MR.Misc._PassBy _other_pass_by, MR.FeatureObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FeatureObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_FeatureObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_FeatureObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_FeatureObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_FeatureObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FeatureObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_FeatureObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PointMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.PointMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PointMeasurementObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PointMeasurementObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointMeasurementObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointMeasurementObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointMeasurementObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointMeasurementObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectComparableWithReference _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference(MR.Misc._PassBy _other_pass_by, MR.ObjectComparableWithReference._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectComparableWithReference _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectComparableWithReference _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectComparableWithReference._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectComparableWithReference(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_DistanceMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.DistanceMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_DistanceMeasurementObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_DistanceMeasurementObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceMeasurementObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_DistanceMeasurementObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_RadiusMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.RadiusMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_RadiusMeasurementObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_RadiusMeasurementObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RadiusMeasurementObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_RadiusMeasurementObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_MeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_MeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_MeasurementObject(MR.Misc._PassBy _other_pass_by, MR.MeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_MeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_MeasurementObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_MeasurementObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_MeasurementObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_MeasurementObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeasurementObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_MeasurementObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_AngleMeasurementObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AngleMeasurementObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AngleMeasurementObject(MR.Misc._PassBy _other_pass_by, MR.AngleMeasurementObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AngleMeasurementObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_AngleMeasurementObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_AngleMeasurementObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AngleMeasurementObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AngleMeasurementObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AngleMeasurementObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AngleMeasurementObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectMesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMesh", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMesh(MR.Misc._PassBy _other_pass_by, MR.ObjectMesh._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMesh(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectMesh _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectMesh _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectMesh", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectMesh(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectMesh._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectMesh(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectVoxels _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectVoxels", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectVoxels(MR.Misc._PassBy _other_pass_by, MR.ObjectVoxels._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectVoxels(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectVoxels _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectVoxels _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectVoxels", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectVoxels(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectVoxels._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectVoxels(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectMeshHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMeshHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMeshHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectMeshHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectMeshHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectMeshHolder _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectMeshHolder _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectMeshHolder", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectMeshHolder(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectMeshHolder._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectMeshHolder(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectDistanceMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectDistanceMap", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectDistanceMap(MR.Misc._PassBy _other_pass_by, MR.ObjectDistanceMap._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectDistanceMap(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectDistanceMap _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectDistanceMap _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectDistanceMap", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectDistanceMap(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectDistanceMap._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectDistanceMap(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectLines _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLines", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLines(MR.Misc._PassBy _other_pass_by, MR.ObjectLines._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLines(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectLines _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectLines _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLines", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLines(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectLines._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLines(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectLinesHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLinesHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLinesHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectLinesHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLinesHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectLinesHolder _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectLinesHolder _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLinesHolder", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLinesHolder(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectLinesHolder._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLinesHolder(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectGcode _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectGcode", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectGcode(MR.Misc._PassBy _other_pass_by, MR.ObjectGcode._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectGcode(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectGcode _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectGcode _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectGcode", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectGcode(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectGcode._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectGcode(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectLabel _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLabel", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLabel(MR.Misc._PassBy _other_pass_by, MR.ObjectLabel._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectLabel(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectLabel _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectLabel _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLabel", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLabel(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectLabel._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectLabel(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectPointsHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPointsHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPointsHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectPointsHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPointsHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectPointsHolder _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectPointsHolder _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectPointsHolder", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectPointsHolder(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectPointsHolder._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectPointsHolder(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectPoints _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPoints(MR.Misc._PassBy _other_pass_by, MR.ObjectPoints._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectPoints(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectPoints _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectPoints _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectPoints(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectPoints._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectPoints(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_VisualObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_VisualObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_VisualObject(MR.Misc._PassBy _other_pass_by, MR.VisualObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_VisualObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_VisualObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_VisualObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_VisualObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_VisualObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VisualObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_VisualObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_SceneRootObject _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SceneRootObject", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SceneRootObject(MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_SceneRootObject(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_SceneRootObject _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_SceneRootObject _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_SceneRootObject", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_SceneRootObject(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_SceneRootObject(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ObjectChildrenHolder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder(MR.Misc._PassBy _other_pass_by, MR.ObjectChildrenHolder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ObjectChildrenHolder _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ObjectChildrenHolder _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectChildrenHolder._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ObjectChildrenHolder(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_Object _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Object", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Object(MR.Misc._PassBy _other_pass_by, MR.Object._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Object(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_Object _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Object _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Object", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Object(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Object._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Object(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_Mesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Mesh", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Mesh(MR.Misc._PassBy _other_pass_by, MR.Mesh._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Mesh(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_Mesh _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Mesh _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Mesh", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Mesh(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Mesh._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Mesh(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PointCloud _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointCloud", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointCloud(MR.Misc._PassBy _other_pass_by, MR.PointCloud._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointCloud(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PointCloud _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PointCloud _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointCloud", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointCloud(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointCloud._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointCloud(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_Polyline3 _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Polyline3", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Polyline3(MR.Misc._PassBy _other_pass_by, MR.Polyline3._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Polyline3(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_Polyline3 _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Polyline3 _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Polyline3", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Polyline3(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polyline3._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Polyline3(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_BasicUiRenderTask _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_BasicUiRenderTask", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_BasicUiRenderTask(MR.Misc._PassBy _other_pass_by, MR.BasicUiRenderTask._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_BasicUiRenderTask(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_BasicUiRenderTask _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_BasicUiRenderTask _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_BasicUiRenderTask", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_BasicUiRenderTask(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BasicUiRenderTask._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_BasicUiRenderTask(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangVoxelSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangVoxelSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangVoxelSelectionAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangVoxelSelectionAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangVoxelSelectionAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangVoxelSelectionAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeActiveBoxAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction(MR.Misc._PassBy _other_pass_by, MR.ChangeActiveBoxAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeActiveBoxAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeActiveBoxAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeActiveBoxAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeActiveBoxAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeColoringType _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeColoringType", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeColoringType(MR.Misc._PassBy _other_pass_by, MR.ChangeColoringType._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeColoringType(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeColoringType _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeColoringType _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeColoringType", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeColoringType(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeColoringType._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeColoringType(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeDualMarchingCubesAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction(MR.Misc._PassBy _other_pass_by, MR.ChangeDualMarchingCubesAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeDualMarchingCubesAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeDualMarchingCubesAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeDualMarchingCubesAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeFacesColorMapAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction(MR.Misc._PassBy _other_pass_by, MR.ChangeFacesColorMapAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeFacesColorMapAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeFacesColorMapAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeFacesColorMapAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeFacesColorMapAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeGridAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeGridAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeGridAction(MR.Misc._PassBy _other_pass_by, MR.ChangeGridAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeGridAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeGridAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeGridAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeGridAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeGridAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeGridAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeGridAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeIsoAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeIsoAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeIsoAction(MR.Misc._PassBy _other_pass_by, MR.ChangeIsoAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeIsoAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeIsoAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeIsoAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeIsoAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeIsoAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeIsoAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeIsoAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeLabelAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLabelAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLabelAction(MR.Misc._PassBy _other_pass_by, MR.ChangeLabelAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLabelAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeLabelAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeLabelAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeLabelAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeLabelAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeLabelAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeLabelAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeLinesColorMapAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction(MR.Misc._PassBy _other_pass_by, MR.ChangeLinesColorMapAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeLinesColorMapAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeLinesColorMapAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeLinesColorMapAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeLinesColorMapAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshCreasesAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshCreasesAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshCreasesAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshCreasesAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshCreasesAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshCreasesAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshDataAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshDataAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshDataAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshDataAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshDataAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshDataAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshEdgeSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshEdgeSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshEdgeSelectionAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshEdgeSelectionAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshEdgeSelectionAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshFaceSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshFaceSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshFaceSelectionAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshFaceSelectionAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshFaceSelectionAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshPointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshPointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshPointsAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshPointsAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshPointsAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshPointsAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshTexturePerFaceAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTexturePerFaceAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshTexturePerFaceAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshTexturePerFaceAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTexturePerFaceAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshTopologyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTopologyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshTopologyAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshTopologyAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTopologyAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshTopologyAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeMeshUVCoordsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshUVCoordsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeMeshUVCoordsAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeMeshUVCoordsAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshUVCoordsAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeNameAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeNameAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeNameAction(MR.Misc._PassBy _other_pass_by, MR.ChangeNameAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeNameAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeNameAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeNameAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeNameAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeNameAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeNameAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeNameAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeObjectAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeObjectAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeObjectAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeObjectColorAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectColorAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeObjectColorAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeObjectColorAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectColorAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectColorAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeObjectSelectedAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectSelectedAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeObjectSelectedAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeObjectSelectedAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectSelectedAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectSelectedAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeObjectVisibilityAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectVisibilityAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeObjectVisibilityAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeObjectVisibilityAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectVisibilityAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeObjectVisibilityAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeOneNormalInCloudAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction(MR.Misc._PassBy _other_pass_by, MR.ChangeOneNormalInCloudAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeOneNormalInCloudAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeOneNormalInCloudAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeOneNormalInCloudAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeOnePointInCloudAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInCloudAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeOnePointInCloudAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeOnePointInCloudAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInCloudAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOnePointInCloudAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeOnePointInPolylineAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInPolylineAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeOnePointInPolylineAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeOnePointInPolylineAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInPolylineAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePointCloudAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePointCloudAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePointCloudAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePointCloudNormalsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudNormalsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePointCloudNormalsAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePointCloudNormalsAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudNormalsAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudNormalsAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePointCloudPointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudPointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePointCloudPointsAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePointCloudPointsAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudPointsAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointCloudPointsAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePointPointSelectionAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction(MR.Misc._PassBy _other_pass_by, MR.ChangePointPointSelectionAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePointPointSelectionAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePointPointSelectionAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointPointSelectionAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePointPointSelectionAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePolylineAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineAction(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePolylineAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePolylineAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylineAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylineAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePolylineAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylineAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePolylinePointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction(MR.Misc._PassBy _other_pass_by, MR.ChangePolylinePointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePolylinePointsAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePolylinePointsAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePolylinePointsAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylinePointsAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangePolylineTopologyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineTopologyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangePolylineTopologyAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangePolylineTopologyAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePolylineTopologyAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangePolylineTopologyAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeScaleAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeScaleAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeScaleAction(MR.Misc._PassBy _other_pass_by, MR.ChangeScaleAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeScaleAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeScaleAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeScaleAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeScaleAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeScaleAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeScaleAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeScaleAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeSceneAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneAction(MR.Misc._PassBy _other_pass_by, MR.ChangeSceneAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeSceneAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeSceneAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSceneAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSceneAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeSceneAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSceneAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeSceneObjectsOrder _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder(MR.Misc._PassBy _other_pass_by, MR.ChangeSceneObjectsOrder._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeSceneObjectsOrder _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeSceneObjectsOrder _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeSceneObjectsOrder._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSceneObjectsOrder(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeSurfaceAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction(MR.Misc._PassBy _other_pass_by, MR.ChangeSurfaceAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeSurfaceAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeSurfaceAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeSurfaceAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeSurfaceAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeTextureAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeTextureAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeTextureAction(MR.Misc._PassBy _other_pass_by, MR.ChangeTextureAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeTextureAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeTextureAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeTextureAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeTextureAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeTextureAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeTextureAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeTextureAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeVisualizePropertyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction(MR.Misc._PassBy _other_pass_by, MR.ChangeVisualizePropertyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeVisualizePropertyAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeVisualizePropertyAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeVisualizePropertyAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeVisualizePropertyAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_ChangeXfAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeXfAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeXfAction(MR.Misc._PassBy _other_pass_by, MR.ChangeXfAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_ChangeXfAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_ChangeXfAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_ChangeXfAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeXfAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeXfAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeXfAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_ChangeXfAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_CombinedHistoryAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CombinedHistoryAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CombinedHistoryAction(MR.Misc._PassBy _other_pass_by, MR.CombinedHistoryAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_CombinedHistoryAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_CombinedHistoryAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_CombinedHistoryAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CombinedHistoryAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CombinedHistoryAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CombinedHistoryAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_CombinedHistoryAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PartialChangeMeshAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PartialChangeMeshAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PartialChangeMeshAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PartialChangeMeshDataAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshDataAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PartialChangeMeshDataAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PartialChangeMeshDataAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshDataAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshDataAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PartialChangeMeshPointsAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshPointsAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PartialChangeMeshPointsAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PartialChangeMeshPointsAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshPointsAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshPointsAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PartialChangeMeshTopologyAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshTopologyAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PartialChangeMeshTopologyAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PartialChangeMeshTopologyAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshTopologyAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_HistoryAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_HistoryAction", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_HistoryAction(MR.Misc._PassBy _other_pass_by, MR.HistoryAction._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_HistoryAction(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_HistoryAction _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_HistoryAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_HistoryAction", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_HistoryAction(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.HistoryAction._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_HistoryAction(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_Matrix_Float _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Matrix_float", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Matrix_float(MR.Misc._PassBy _other_pass_by, MR.Matrix_Float._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_Matrix_float(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_Matrix_Float _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_Matrix_Float _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Matrix_float", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Matrix_float(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Matrix_Float._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_Matrix_float(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_RectIndexer _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RectIndexer", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RectIndexer(MR.Misc._PassBy _other_pass_by, MR.RectIndexer._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_RectIndexer(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_RectIndexer _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_RectIndexer _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_RectIndexer", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_RectIndexer(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RectIndexer._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_RectIndexer(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_DistanceMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMap", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMap(MR.Misc._PassBy _other_pass_by, MR.DistanceMap._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_DistanceMap(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_DistanceMap _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_DistanceMap _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_DistanceMap", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_DistanceMap(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceMap._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_DistanceMap(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR.Std._ByValue_Vector_StdString _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_std_vector_std_string", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_std_vector_std_string(MR.Misc._PassBy _other_pass_by, MR.Std.Vector_StdString._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_std_vector_std_string(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR.Std._ByValue_Vector_StdString _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR.Std._ByValue_Vector_StdString _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_std_vector_std_string", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_std_vector_std_string(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Std.Vector_StdString._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_std_vector_std_string(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_FastWindingNumber _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FastWindingNumber", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FastWindingNumber(MR.Misc._PassBy _other_pass_by, MR.FastWindingNumber._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_FastWindingNumber(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_FastWindingNumber _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_FastWindingNumber _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_FastWindingNumber", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_FastWindingNumber(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FastWindingNumber._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_FastWindingNumber(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_IFastWindingNumber _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IFastWindingNumber", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IFastWindingNumber(MR.Misc._PassBy _other_pass_by, MR.IFastWindingNumber._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IFastWindingNumber(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_IFastWindingNumber _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_IFastWindingNumber _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_IFastWindingNumber", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_IFastWindingNumber(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.IFastWindingNumber._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_IFastWindingNumber(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_OpenVdbFloatGrid _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid(MR.Misc._PassBy _other_pass_by, MR.OpenVdbFloatGrid._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_OpenVdbFloatGrid _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_OpenVdbFloatGrid _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.OpenVdbFloatGrid._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_OpenVdbFloatGrid(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_PointsToMeshProjector _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointsToMeshProjector", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointsToMeshProjector(MR.Misc._PassBy _other_pass_by, MR.PointsToMeshProjector._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_PointsToMeshProjector(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_PointsToMeshProjector _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_PointsToMeshProjector _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointsToMeshProjector", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointsToMeshProjector(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsToMeshProjector._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_PointsToMeshProjector(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public unsafe SharedPtr_Void(MR._ByValue_IPointsToMeshProjector _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector", ExactSpelling = true)]
                extern static MR.Std.SharedPtr_Void._Underlying *__MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector(MR.Misc._PassBy _other_pass_by, MR.IPointsToMeshProjector._UnderlyingShared *_other);
                _UnderlyingPtr = __MR_std_shared_ptr_void_ConstructFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator SharedPtr_Void(MR._ByValue_IPointsToMeshProjector _other) {return new(_other);}

            /// Overwrites an existing `std::shared_ptr<void>` to point to the same object as this instance.
            public unsafe void Assign(MR._ByValue_IPointsToMeshProjector _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector", ExactSpelling = true)]
                extern static void __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.IPointsToMeshProjector._UnderlyingShared *_other);
                __MR_std_shared_ptr_void_AssignFrom_MR_std_shared_ptr_MR_IPointsToMeshProjector(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingSharedPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `SharedPtr_Void` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `SharedPtr_Void`/`Const_SharedPtr_Void` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_SharedPtr_Void
        {
            internal readonly Const_SharedPtr_Void? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_SharedPtr_Void() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_SharedPtr_Void(Const_SharedPtr_Void new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_SharedPtr_Void(Const_SharedPtr_Void arg) {return new(arg);}
            public _ByValue_SharedPtr_Void(MR.Misc._Moved<SharedPtr_Void> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_SharedPtr_Void(MR.Misc._Moved<SharedPtr_Void> arg) {return new(arg);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_CircleObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_SphereObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ConeObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_CylinderObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_LineObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PlaneObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PointObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_FeatureObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PointMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectComparableWithReference _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_DistanceMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_RadiusMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_MeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_AngleMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectMesh _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectVoxels _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectMeshHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectDistanceMap _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectLines _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectLinesHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectGcode _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectLabel _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectPointsHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectPoints _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_VisualObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_SceneRootObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ObjectChildrenHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_Object _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_Mesh _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PointCloud _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_Polyline3 _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_BasicUiRenderTask _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangVoxelSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeActiveBoxAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeColoringType _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeDualMarchingCubesAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeFacesColorMapAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeGridAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeIsoAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeLabelAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeLinesColorMapAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshCreasesAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshDataAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshEdgeSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshFaceSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshPointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshTexturePerFaceAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshTopologyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeMeshUVCoordsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeNameAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeObjectAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeObjectColorAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeObjectSelectedAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeObjectVisibilityAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeOneNormalInCloudAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeOnePointInCloudAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeOnePointInPolylineAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePointCloudAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePointCloudNormalsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePointCloudPointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePointPointSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePolylineAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePolylinePointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangePolylineTopologyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeScaleAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeSceneAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeSceneObjectsOrder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeSurfaceAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeTextureAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeVisualizePropertyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_ChangeXfAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_CombinedHistoryAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PartialChangeMeshAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PartialChangeMeshDataAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PartialChangeMeshPointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PartialChangeMeshTopologyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_HistoryAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_Matrix_Float _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_RectIndexer _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_DistanceMap _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR.Std._ByValue_Vector_StdString _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_FastWindingNumber _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_IFastWindingNumber _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_OpenVdbFloatGrid _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_PointsToMeshProjector _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _ByValue_SharedPtr_Void(MR._ByValue_IPointsToMeshProjector _other) {return new MR.Std.SharedPtr_Void(_other);}
        }

        /// This is used for optional parameters of class `SharedPtr_Void` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_SharedPtr_Void`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SharedPtr_Void`/`Const_SharedPtr_Void` directly.
        public class _InOptMut_SharedPtr_Void
        {
            public SharedPtr_Void? Opt;

            public _InOptMut_SharedPtr_Void() {}
            public _InOptMut_SharedPtr_Void(SharedPtr_Void value) {Opt = value;}
            public static implicit operator _InOptMut_SharedPtr_Void(SharedPtr_Void value) {return new(value);}
        }

        /// This is used for optional parameters of class `SharedPtr_Void` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_SharedPtr_Void`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SharedPtr_Void`/`Const_SharedPtr_Void` to pass it to the function.
        public class _InOptConst_SharedPtr_Void
        {
            public Const_SharedPtr_Void? Opt;

            public _InOptConst_SharedPtr_Void() {}
            public _InOptConst_SharedPtr_Void(Const_SharedPtr_Void value) {Opt = value;}
            public static implicit operator _InOptConst_SharedPtr_Void(Const_SharedPtr_Void value) {return new(value);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_CircleObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_SphereObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ConeObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_CylinderObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_LineObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PlaneObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PointObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_FeatureObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PointMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectComparableWithReference _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_DistanceMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_RadiusMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_MeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_AngleMeasurementObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectMesh _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectVoxels _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectMeshHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectDistanceMap _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectLines _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectLinesHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectGcode _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectLabel _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectPointsHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectPoints _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_VisualObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_SceneRootObject _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ObjectChildrenHolder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_Object _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_Mesh _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PointCloud _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_Polyline3 _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_BasicUiRenderTask _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangVoxelSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeActiveBoxAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeColoringType _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeDualMarchingCubesAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeFacesColorMapAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeGridAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeIsoAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeLabelAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeLinesColorMapAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshCreasesAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshDataAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshEdgeSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshFaceSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshPointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshTexturePerFaceAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshTopologyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeMeshUVCoordsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeNameAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeObjectAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeObjectColorAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeObjectSelectedAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeObjectVisibilityAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeOneNormalInCloudAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeOnePointInCloudAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeOnePointInPolylineAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePointCloudAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePointCloudNormalsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePointCloudPointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePointPointSelectionAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePolylineAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePolylinePointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangePolylineTopologyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeScaleAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeSceneAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeSceneObjectsOrder _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeSurfaceAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeTextureAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeVisualizePropertyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_ChangeXfAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_CombinedHistoryAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PartialChangeMeshAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PartialChangeMeshDataAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PartialChangeMeshPointsAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PartialChangeMeshTopologyAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_HistoryAction _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_Matrix_Float _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_RectIndexer _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_DistanceMap _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR.Std._ByValue_Vector_StdString _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_FastWindingNumber _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_IFastWindingNumber _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_OpenVdbFloatGrid _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_PointsToMeshProjector _other) {return new MR.Std.SharedPtr_Void(_other);}

            /// Creates an untyped `std::shared_ptr<void>` pointing to the same object as the source typed pointer.
            public static unsafe implicit operator _InOptConst_SharedPtr_Void(MR._ByValue_IPointsToMeshProjector _other) {return new MR.Std.SharedPtr_Void(_other);}
        }
    }
}
