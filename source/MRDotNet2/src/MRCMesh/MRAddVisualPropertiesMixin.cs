public static partial class MR
{
    // Inherits from a datamodel object, adding some visual property masks to it.
    // `BaseObjectType` is the datamodel type to inherit from.
    // `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
    /// Generated from class `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::CircleObject`
    ///     `MR::SphereObject`
    /// This is the const half of the class.
    public class Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UseCount();
                return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructNonOwning(ptr);
        }

        internal unsafe Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_FeatureObject(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.Const_FeatureObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter?(MR.Const_FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.Const_FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticTypeName();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_typeName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_typeName(_Underlying *_this);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticClassName();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticClassNameInPlural();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Create and generate list of bounded getters and setters for the main properties of feature object, together with prop. name for display and edit into UI.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getAllSharedProperties`.
        public unsafe MR.Std.Const_Vector_MRFeatureObjectSharedProperty GetAllSharedProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAllSharedProperties", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRFeatureObjectSharedProperty._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAllSharedProperties(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAllSharedProperties(_UnderlyingPtr), is_owning: false);
        }

        // Since a point on an abstract feature is difficult to uniquely parameterize,
        // the projection function simultaneously returns the normal to the surface at the projection point.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::projectPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.FeatureObjectProjectPointResult ProjectPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_projectPoint", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_projectPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_projectPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: true);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getNormal`.
        public unsafe MR.Std.Optional_MRVector3f GetNormal(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getNormal", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVector3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getNormal(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getNormal(_UnderlyingPtr, point._UnderlyingPtr), is_owning: true);
        }

        // Returns point considered as base for the feature
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getBasePoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetBasePoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBasePoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBasePoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBasePoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        // The cached orthonormalized rotation matrix.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getRotationMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetRotationMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getRotationMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getRotationMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getRotationMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // The cached scale and shear matrix. The main diagnoal stores the scale, and some other elements store the shearing.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getScaleShearMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetScaleShearMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getScaleShearMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getScaleShearMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getScaleShearMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // This color is used for subfeatures.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetDecorationsColor(bool selected, MR._InOpt_ViewportId viewportId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDecorationsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDecorationsColor(_Underlying *_this, byte selected, MR.ViewportId *viewportId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDecorationsColor(_UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getDecorationsColorForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetDecorationsColorForAllViewports(bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDecorationsColorForAllViewports(_Underlying *_this, byte selected);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDecorationsColorForAllViewports(_UnderlyingPtr, selected ? (byte)1 : (byte)0), is_owning: false);
        }

        // Point size and line width, for primary rendering rather than subfeatures.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getPointSize", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getPointSize(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getLineWidth", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getLineWidth(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getLineWidth(_UnderlyingPtr);
        }

        // Point size and line width, for subfeatures rather than primary rendering.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSubfeaturePointSize`.
        public unsafe float GetSubfeaturePointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeaturePointSize", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeaturePointSize(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeaturePointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSubfeatureLineWidth`.
        public unsafe float GetSubfeatureLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureLineWidth", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureLineWidth(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureLineWidth(_UnderlyingPtr);
        }

        // Per-component alpha multipliers. The global alpha is multiplied by thise.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getMainFeatureAlpha`.
        public unsafe float GetMainFeatureAlpha()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getMainFeatureAlpha", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getMainFeatureAlpha(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getMainFeatureAlpha(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSubfeatureAlphaPoints`.
        public unsafe float GetSubfeatureAlphaPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaPoints(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaPoints(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSubfeatureAlphaLines`.
        public unsafe float GetSubfeatureAlphaLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaLines", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaLines(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaLines(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSubfeatureAlphaMesh`.
        public unsafe float GetSubfeatureAlphaMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaMesh(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSubfeatureAlphaMesh(_UnderlyingPtr);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDirtyFlags(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetDirty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetDirty(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBoundingBox(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isPickable", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getColoringType(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getShininess", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getShininess(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSpecularStrength(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAmbientStrength(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getAmbientStrength(_UnderlyingPtr);
        }

        /// clones this object only, without its children,
        /// making new object the owner of all copied resources
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones this object only, without its children,
        /// making new object to share resources with this object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_render", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_renderForPicker", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_renderUi", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_heapBytes(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_name(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_xfsForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalVisibilityMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isLocked", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isLocked(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isParentLocked(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isAncestor", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isSelected", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isSelected(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isAncillary", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isAncillary(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isGlobalAncillary(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isVisible", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_visibilityMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetRedrawFlag(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_hasVisualRepresentation(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_hasModel", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_hasModel(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_tags(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_sameModels", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getModelHash(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    // Inherits from a datamodel object, adding some visual property masks to it.
    // `BaseObjectType` is the datamodel type to inherit from.
    // `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
    /// Generated from class `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::CircleObject`
    ///     `MR::SphereObject`
    /// This is the non-const half of the class.
    public class AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter : Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter
    {
        internal unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.FeatureObject(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.FeatureObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter?(MR.FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(MR.FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetDecorationsColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDecorationsColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDecorationsColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDecorationsColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setDecorationsColorForAllViewports`.
        public unsafe void SetDecorationsColorForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDecorationsColorForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte selected);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDecorationsColorForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setPointSize`.
        public unsafe void SetPointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setPointSize", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setPointSize(_Underlying *_this, float pointSize);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setPointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setLineWidth`.
        public unsafe void SetLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setLineWidth", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setLineWidth(_Underlying *_this, float lineWidth);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setSubfeaturePointSize`.
        public unsafe void SetSubfeaturePointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeaturePointSize", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeaturePointSize(_Underlying *_this, float pointSize);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeaturePointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setSubfeatureLineWidth`.
        public unsafe void SetSubfeatureLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureLineWidth", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureLineWidth(_Underlying *_this, float lineWidth);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setMainFeatureAlpha`.
        public unsafe void SetMainFeatureAlpha(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setMainFeatureAlpha", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setMainFeatureAlpha(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setMainFeatureAlpha(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setSubfeatureAlphaPoints`.
        public unsafe void SetSubfeatureAlphaPoints(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaPoints(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaPoints(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setSubfeatureAlphaLines`.
        public unsafe void SetSubfeatureAlphaLines(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaLines", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaLines(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaLines(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setSubfeatureAlphaMesh`.
        public unsafe void SetSubfeatureAlphaMesh(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaMesh(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSubfeatureAlphaMesh(_UnderlyingPtr, alpha);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setFrontColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setBackColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setPickable", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setColoringType", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setShininess", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setShininess(_Underlying *_this, float shininess);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetFrontColor(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetColors", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetColors(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setName", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setWorldXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_applyScale", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_applyScale(_Underlying *_this, float scaleFactor);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setLocked", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setLocked(_Underlying *_this, byte on);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setParentLocked", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setParentLocked(_Underlying *_this, byte lock_);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_detachFromParent(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addChild", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_removeAllChildren(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_sortChildren", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_sortChildren(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_select", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_select(_Underlying *_this, byte on);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAncillary", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAncillary(_Underlying *_this, byte ancillary);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisible", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_swap", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addTag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_removeTag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter
    {
        internal readonly Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter` directly.
    public class _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter
    {
        public AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter? Opt;

        public _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter() {}
        public _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter value) {Opt = value;}
        public static implicit operator _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter value) {return new(value);}
    }

    /// This is used for optional parameters of class `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter` to pass it to the function.
    public class _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter
    {
        public Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter? Opt;

        public _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter() {}
        public _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter value) {Opt = value;}
        public static implicit operator _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter value) {return new(value);}
    }

    // Inherits from a datamodel object, adding some visual property masks to it.
    // `BaseObjectType` is the datamodel type to inherit from.
    // `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
    /// Generated from class `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ConeObject`
    /// This is the const half of the class.
    public class Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UseCount();
                return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructNonOwning(ptr);
        }

        internal unsafe Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_FeatureObject(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.Const_FeatureObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength?(MR.Const_FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.Const_FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticTypeName();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_typeName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_typeName(_Underlying *_this);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticClassName();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticClassNameInPlural();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Create and generate list of bounded getters and setters for the main properties of feature object, together with prop. name for display and edit into UI.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getAllSharedProperties`.
        public unsafe MR.Std.Const_Vector_MRFeatureObjectSharedProperty GetAllSharedProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAllSharedProperties", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRFeatureObjectSharedProperty._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAllSharedProperties(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAllSharedProperties(_UnderlyingPtr), is_owning: false);
        }

        // Since a point on an abstract feature is difficult to uniquely parameterize,
        // the projection function simultaneously returns the normal to the surface at the projection point.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::projectPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.FeatureObjectProjectPointResult ProjectPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_projectPoint", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_projectPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_projectPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: true);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getNormal`.
        public unsafe MR.Std.Optional_MRVector3f GetNormal(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getNormal", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVector3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getNormal(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getNormal(_UnderlyingPtr, point._UnderlyingPtr), is_owning: true);
        }

        // Returns point considered as base for the feature
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getBasePoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetBasePoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBasePoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBasePoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBasePoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        // The cached orthonormalized rotation matrix.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getRotationMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetRotationMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getRotationMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getRotationMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getRotationMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // The cached scale and shear matrix. The main diagnoal stores the scale, and some other elements store the shearing.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getScaleShearMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetScaleShearMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getScaleShearMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getScaleShearMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getScaleShearMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // This color is used for subfeatures.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetDecorationsColor(bool selected, MR._InOpt_ViewportId viewportId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDecorationsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDecorationsColor(_Underlying *_this, byte selected, MR.ViewportId *viewportId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDecorationsColor(_UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getDecorationsColorForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetDecorationsColorForAllViewports(bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDecorationsColorForAllViewports(_Underlying *_this, byte selected);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDecorationsColorForAllViewports(_UnderlyingPtr, selected ? (byte)1 : (byte)0), is_owning: false);
        }

        // Point size and line width, for primary rendering rather than subfeatures.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getPointSize", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getPointSize(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getLineWidth", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getLineWidth(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getLineWidth(_UnderlyingPtr);
        }

        // Point size and line width, for subfeatures rather than primary rendering.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSubfeaturePointSize`.
        public unsafe float GetSubfeaturePointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeaturePointSize", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeaturePointSize(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeaturePointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSubfeatureLineWidth`.
        public unsafe float GetSubfeatureLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureLineWidth", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureLineWidth(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureLineWidth(_UnderlyingPtr);
        }

        // Per-component alpha multipliers. The global alpha is multiplied by thise.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getMainFeatureAlpha`.
        public unsafe float GetMainFeatureAlpha()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getMainFeatureAlpha", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getMainFeatureAlpha(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getMainFeatureAlpha(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSubfeatureAlphaPoints`.
        public unsafe float GetSubfeatureAlphaPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaPoints(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaPoints(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSubfeatureAlphaLines`.
        public unsafe float GetSubfeatureAlphaLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaLines", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaLines(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaLines(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSubfeatureAlphaMesh`.
        public unsafe float GetSubfeatureAlphaMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaMesh(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaMesh(_UnderlyingPtr);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDirtyFlags(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetDirty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetDirty(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBoundingBox(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isPickable", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getColoringType(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getShininess", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getShininess(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSpecularStrength(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAmbientStrength(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getAmbientStrength(_UnderlyingPtr);
        }

        /// clones this object only, without its children,
        /// making new object the owner of all copied resources
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones this object only, without its children,
        /// making new object to share resources with this object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_render", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_renderForPicker", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_renderUi", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_heapBytes(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_name(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_xfsForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalVisibilityMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isLocked", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isLocked(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isParentLocked(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isAncestor", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isSelected", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isSelected(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isAncillary", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isAncillary(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isGlobalAncillary(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isVisible", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_visibilityMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetRedrawFlag(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_hasVisualRepresentation(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_hasModel", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_hasModel(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_tags(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_sameModels", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getModelHash(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    // Inherits from a datamodel object, adding some visual property masks to it.
    // `BaseObjectType` is the datamodel type to inherit from.
    // `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
    /// Generated from class `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ConeObject`
    /// This is the non-const half of the class.
    public class AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength : Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength
    {
        internal unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.FeatureObject(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.FeatureObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength?(MR.FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(MR.FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetDecorationsColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDecorationsColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDecorationsColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDecorationsColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setDecorationsColorForAllViewports`.
        public unsafe void SetDecorationsColorForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDecorationsColorForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte selected);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDecorationsColorForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setPointSize`.
        public unsafe void SetPointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setPointSize", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setPointSize(_Underlying *_this, float pointSize);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setPointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setLineWidth`.
        public unsafe void SetLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setLineWidth", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setLineWidth(_Underlying *_this, float lineWidth);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setSubfeaturePointSize`.
        public unsafe void SetSubfeaturePointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeaturePointSize", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeaturePointSize(_Underlying *_this, float pointSize);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeaturePointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setSubfeatureLineWidth`.
        public unsafe void SetSubfeatureLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureLineWidth", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureLineWidth(_Underlying *_this, float lineWidth);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setMainFeatureAlpha`.
        public unsafe void SetMainFeatureAlpha(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setMainFeatureAlpha", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setMainFeatureAlpha(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setMainFeatureAlpha(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setSubfeatureAlphaPoints`.
        public unsafe void SetSubfeatureAlphaPoints(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaPoints(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaPoints(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setSubfeatureAlphaLines`.
        public unsafe void SetSubfeatureAlphaLines(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaLines", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaLines(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaLines(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setSubfeatureAlphaMesh`.
        public unsafe void SetSubfeatureAlphaMesh(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaMesh(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaMesh(_UnderlyingPtr, alpha);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setFrontColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setBackColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setPickable", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setColoringType", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setShininess", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setShininess(_Underlying *_this, float shininess);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetFrontColor(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetColors", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetColors(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setName", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setWorldXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_applyScale", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_applyScale(_Underlying *_this, float scaleFactor);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setLocked", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setLocked(_Underlying *_this, byte on);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setParentLocked", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setParentLocked(_Underlying *_this, byte lock_);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_detachFromParent(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addChild", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_removeAllChildren(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_sortChildren", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_sortChildren(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_select", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_select(_Underlying *_this, byte on);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAncillary", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAncillary(_Underlying *_this, byte ancillary);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisible", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_swap", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addTag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_removeTag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_angle_MR_DimensionsVisualizePropertyType_length_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength
    {
        internal readonly Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength` directly.
    public class _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength
    {
        public AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength? Opt;

        public _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength() {}
        public _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength value) {Opt = value;}
        public static implicit operator _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength value) {return new(value);}
    }

    /// This is used for optional parameters of class `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength` to pass it to the function.
    public class _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength
    {
        public Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength? Opt;

        public _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength() {}
        public _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength value) {Opt = value;}
        public static implicit operator _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeAngle_MRDimensionsVisualizePropertyTypeLength value) {return new(value);}
    }

    // Inherits from a datamodel object, adding some visual property masks to it.
    // `BaseObjectType` is the datamodel type to inherit from.
    // `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
    /// Generated from class `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::CylinderObject`
    /// This is the const half of the class.
    public class Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UseCount();
                return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructNonOwning(ptr);
        }

        internal unsafe Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_FeatureObject(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.Const_FeatureObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength?(MR.Const_FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.Const_FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticTypeName();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_typeName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_typeName(_Underlying *_this);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticClassName();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticClassNameInPlural();
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Create and generate list of bounded getters and setters for the main properties of feature object, together with prop. name for display and edit into UI.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getAllSharedProperties`.
        public unsafe MR.Std.Const_Vector_MRFeatureObjectSharedProperty GetAllSharedProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAllSharedProperties", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRFeatureObjectSharedProperty._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAllSharedProperties(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAllSharedProperties(_UnderlyingPtr), is_owning: false);
        }

        // Since a point on an abstract feature is difficult to uniquely parameterize,
        // the projection function simultaneously returns the normal to the surface at the projection point.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::projectPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.FeatureObjectProjectPointResult ProjectPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_projectPoint", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_projectPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_projectPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: true);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getNormal`.
        public unsafe MR.Std.Optional_MRVector3f GetNormal(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getNormal", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVector3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getNormal(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getNormal(_UnderlyingPtr, point._UnderlyingPtr), is_owning: true);
        }

        // Returns point considered as base for the feature
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getBasePoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetBasePoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBasePoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBasePoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBasePoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        // The cached orthonormalized rotation matrix.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getRotationMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetRotationMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getRotationMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getRotationMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getRotationMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // The cached scale and shear matrix. The main diagnoal stores the scale, and some other elements store the shearing.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getScaleShearMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetScaleShearMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getScaleShearMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getScaleShearMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getScaleShearMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // This color is used for subfeatures.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetDecorationsColor(bool selected, MR._InOpt_ViewportId viewportId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDecorationsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDecorationsColor(_Underlying *_this, byte selected, MR.ViewportId *viewportId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDecorationsColor(_UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getDecorationsColorForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetDecorationsColorForAllViewports(bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDecorationsColorForAllViewports(_Underlying *_this, byte selected);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDecorationsColorForAllViewports(_UnderlyingPtr, selected ? (byte)1 : (byte)0), is_owning: false);
        }

        // Point size and line width, for primary rendering rather than subfeatures.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getPointSize", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getPointSize(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getLineWidth", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getLineWidth(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getLineWidth(_UnderlyingPtr);
        }

        // Point size and line width, for subfeatures rather than primary rendering.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSubfeaturePointSize`.
        public unsafe float GetSubfeaturePointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeaturePointSize", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeaturePointSize(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeaturePointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSubfeatureLineWidth`.
        public unsafe float GetSubfeatureLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureLineWidth", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureLineWidth(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureLineWidth(_UnderlyingPtr);
        }

        // Per-component alpha multipliers. The global alpha is multiplied by thise.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getMainFeatureAlpha`.
        public unsafe float GetMainFeatureAlpha()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getMainFeatureAlpha", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getMainFeatureAlpha(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getMainFeatureAlpha(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSubfeatureAlphaPoints`.
        public unsafe float GetSubfeatureAlphaPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaPoints(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaPoints(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSubfeatureAlphaLines`.
        public unsafe float GetSubfeatureAlphaLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaLines", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaLines(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaLines(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSubfeatureAlphaMesh`.
        public unsafe float GetSubfeatureAlphaMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaMesh(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSubfeatureAlphaMesh(_UnderlyingPtr);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDirtyFlags(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetDirty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetDirty(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBoundingBox(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isPickable", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getColoringType(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getShininess", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getShininess(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSpecularStrength(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAmbientStrength(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getAmbientStrength(_UnderlyingPtr);
        }

        /// clones this object only, without its children,
        /// making new object the owner of all copied resources
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones this object only, without its children,
        /// making new object to share resources with this object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_render", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_renderForPicker", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_renderUi", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_heapBytes(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_name(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_xfsForAllViewports(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalVisibilityMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isLocked", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isLocked(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isParentLocked(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isAncestor", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isSelected", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isSelected(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isAncillary", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isAncillary(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isGlobalAncillary(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isVisible", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_visibilityMask(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetRedrawFlag(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_hasVisualRepresentation(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_hasModel", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_hasModel(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_tags(_Underlying *_this);
            return new(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_sameModels", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getModelHash(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    // Inherits from a datamodel object, adding some visual property masks to it.
    // `BaseObjectType` is the datamodel type to inherit from.
    // `Properties...` is the list of properties to add. Each must be a value from a enum marked as `IsVisualizeMaskEnum`.
    /// Generated from class `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::CylinderObject`
    /// This is the non-const half of the class.
    public class AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength : Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength
    {
        internal unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.FeatureObject(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.FeatureObject._Underlying *__MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength?(MR.FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(MR.FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetDecorationsColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDecorationsColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDecorationsColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDecorationsColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setDecorationsColorForAllViewports`.
        public unsafe void SetDecorationsColorForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDecorationsColorForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte selected);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDecorationsColorForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setPointSize`.
        public unsafe void SetPointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setPointSize", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setPointSize(_Underlying *_this, float pointSize);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setPointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setLineWidth`.
        public unsafe void SetLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setLineWidth", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setLineWidth(_Underlying *_this, float lineWidth);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setSubfeaturePointSize`.
        public unsafe void SetSubfeaturePointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeaturePointSize", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeaturePointSize(_Underlying *_this, float pointSize);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeaturePointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setSubfeatureLineWidth`.
        public unsafe void SetSubfeatureLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureLineWidth", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureLineWidth(_Underlying *_this, float lineWidth);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setMainFeatureAlpha`.
        public unsafe void SetMainFeatureAlpha(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setMainFeatureAlpha", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setMainFeatureAlpha(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setMainFeatureAlpha(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setSubfeatureAlphaPoints`.
        public unsafe void SetSubfeatureAlphaPoints(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaPoints(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaPoints(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setSubfeatureAlphaLines`.
        public unsafe void SetSubfeatureAlphaLines(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaLines", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaLines(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaLines(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setSubfeatureAlphaMesh`.
        public unsafe void SetSubfeatureAlphaMesh(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaMesh(_Underlying *_this, float alpha);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSubfeatureAlphaMesh(_UnderlyingPtr, alpha);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setFrontColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setBackColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setPickable", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setColoringType", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setShininess", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setShininess(_Underlying *_this, float shininess);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetFrontColor(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetColors", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetColors(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setName", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setWorldXf", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_applyScale", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_applyScale(_Underlying *_this, float scaleFactor);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setLocked", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setLocked(_Underlying *_this, byte on);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setParentLocked", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setParentLocked(_Underlying *_this, byte lock_);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_detachFromParent(_Underlying *_this);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addChild", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_removeAllChildren(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_sortChildren", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_sortChildren(_Underlying *_this);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_select", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_select(_Underlying *_this, byte on);
            return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAncillary", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAncillary(_Underlying *_this, byte ancillary);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisible", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_swap", ExactSpelling = true)]
            extern static void __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addTag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_removeTag", ExactSpelling = true)]
            extern static byte __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_AddVisualProperties_MR_FeatureObject_MR_DimensionsVisualizePropertyType_diameter_MR_DimensionsVisualizePropertyType_length_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength
    {
        internal readonly Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength` directly.
    public class _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength
    {
        public AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength? Opt;

        public _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength() {}
        public _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength value) {Opt = value;}
        public static implicit operator _InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength value) {return new(value);}
    }

    /// This is used for optional parameters of class `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength`/`Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength` to pass it to the function.
    public class _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength
    {
        public Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength? Opt;

        public _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength() {}
        public _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength value) {Opt = value;}
        public static implicit operator _InOptConst_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength(Const_AddVisualProperties_MRFeatureObject_MRDimensionsVisualizePropertyTypeDiameter_MRDimensionsVisualizePropertyTypeLength value) {return new(value);}
    }
}
