public static partial class MR
{
    // A common base class for measurement objects.
    // Can't be constructed directly.
    /// Generated from class `MR::MeasurementObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AngleMeasurementObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::RadiusMeasurementObject`
    /// This is the const half of the class.
    public class Const_MeasurementObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_MeasurementObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_MeasurementObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_MeasurementObject_UseCount();
                return __MR_std_shared_ptr_MR_MeasurementObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_MeasurementObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_MeasurementObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_MeasurementObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_MeasurementObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_MeasurementObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_MeasurementObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_MeasurementObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_MeasurementObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe MeasurementObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_MeasurementObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_MeasurementObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_MeasurementObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_MeasurementObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_MeasurementObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_MeasurementObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_MeasurementObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeasurementObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_MeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_MeasurementObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_MeasurementObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_MeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_MeasurementObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_MeasurementObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_MeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_MeasurementObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_MeasurementObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_MeasurementObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_MeasurementObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_MeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_MeasurementObject?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_MeasurementObject(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_MeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::MeasurementObject::MeasurementObject`.
        public unsafe Const_MeasurementObject(MR._ByValue_MeasurementObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeasurementObject._Underlying *__MR_MeasurementObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeasurementObject._Underlying *_other);
            _LateMakeShared(__MR_MeasurementObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::MeasurementObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_MeasurementObject_StaticTypeName();
            var __ret = __MR_MeasurementObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::MeasurementObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_MeasurementObject_typeName(_Underlying *_this);
            var __ret = __MR_MeasurementObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::MeasurementObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_MeasurementObject_StaticClassName();
            var __ret = __MR_MeasurementObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::MeasurementObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_MeasurementObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_MeasurementObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::MeasurementObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_MeasurementObject_StaticClassNameInPlural();
            var __ret = __MR_MeasurementObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::MeasurementObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_MeasurementObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_MeasurementObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Returns true if this class supports the property `type`. Otherwise passing it to the functions below is illegal.
        /// Generated from method `MR::MeasurementObject::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_MeasurementObject_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::MeasurementObject::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_MeasurementObject_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::MeasurementObject::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_MeasurementObject_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_MeasurementObject_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// get all visualize properties masks
        /// Generated from method `MR::MeasurementObject::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_MeasurementObject_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_MeasurementObject_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::MeasurementObject::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_MeasurementObject_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_MeasurementObject_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::MeasurementObject::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_MeasurementObject_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::MeasurementObject::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_MeasurementObject_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_MeasurementObject_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::MeasurementObject::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_MeasurementObject_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_MeasurementObject_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::MeasurementObject::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_MeasurementObject_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_MeasurementObject_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::MeasurementObject::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_MeasurementObject_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_MeasurementObject_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::MeasurementObject::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_MeasurementObject_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_MeasurementObject_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::MeasurementObject::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_MeasurementObject_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_MeasurementObject_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::MeasurementObject::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_MeasurementObject_getDirtyFlags(_Underlying *_this);
            return __MR_MeasurementObject_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::MeasurementObject::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_resetDirty", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_resetDirty(_Underlying *_this);
            __MR_MeasurementObject_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::MeasurementObject::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_MeasurementObject_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::MeasurementObject::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_MeasurementObject_getBoundingBox(_Underlying *_this);
            return __MR_MeasurementObject_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::MeasurementObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_MeasurementObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_MeasurementObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::MeasurementObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_MeasurementObject_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::MeasurementObject::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isPickable", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_MeasurementObject_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::MeasurementObject::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_MeasurementObject_getColoringType(_Underlying *_this);
            return __MR_MeasurementObject_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::MeasurementObject::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getShininess", ExactSpelling = true)]
            extern static float __MR_MeasurementObject_getShininess(_Underlying *_this);
            return __MR_MeasurementObject_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::MeasurementObject::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_MeasurementObject_getSpecularStrength(_Underlying *_this);
            return __MR_MeasurementObject_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::MeasurementObject::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_MeasurementObject_getAmbientStrength(_Underlying *_this);
            return __MR_MeasurementObject_getAmbientStrength(_UnderlyingPtr);
        }

        /// clones this object only, without its children,
        /// making new object the owner of all copied resources
        /// Generated from method `MR::MeasurementObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_MeasurementObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_MeasurementObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones this object only, without its children,
        /// making new object to share resources with this object
        /// Generated from method `MR::MeasurementObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_MeasurementObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_MeasurementObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::MeasurementObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_render", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_MeasurementObject_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::MeasurementObject::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_renderForPicker", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_MeasurementObject_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::MeasurementObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_renderUi", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_MeasurementObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::MeasurementObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_MeasurementObject_heapBytes(_Underlying *_this);
            return __MR_MeasurementObject_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::MeasurementObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_MeasurementObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_MeasurementObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::MeasurementObject::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_MeasurementObject_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::MeasurementObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_MeasurementObject_name(_Underlying *_this);
            return new(__MR_MeasurementObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::MeasurementObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_MeasurementObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_MeasurementObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::MeasurementObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_MeasurementObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_MeasurementObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::MeasurementObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeasurementObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_MeasurementObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::MeasurementObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_MeasurementObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_MeasurementObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::MeasurementObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_MeasurementObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::MeasurementObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isLocked(_Underlying *_this);
            return __MR_MeasurementObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::MeasurementObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isParentLocked(_Underlying *_this);
            return __MR_MeasurementObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::MeasurementObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_MeasurementObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::MeasurementObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isSelected(_Underlying *_this);
            return __MR_MeasurementObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::MeasurementObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isAncillary(_Underlying *_this);
            return __MR_MeasurementObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::MeasurementObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isGlobalAncillary(_Underlying *_this);
            return __MR_MeasurementObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::MeasurementObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_MeasurementObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::MeasurementObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_MeasurementObject_visibilityMask(_Underlying *_this);
            return new(__MR_MeasurementObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::MeasurementObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_resetRedrawFlag(_Underlying *_this);
            __MR_MeasurementObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::MeasurementObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_MeasurementObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_MeasurementObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::MeasurementObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_MeasurementObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_MeasurementObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::MeasurementObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_MeasurementObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_MeasurementObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::MeasurementObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_MeasurementObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::MeasurementObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_hasModel(_Underlying *_this);
            return __MR_MeasurementObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::MeasurementObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_MeasurementObject_tags(_Underlying *_this);
            return new(__MR_MeasurementObject_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::MeasurementObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_MeasurementObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::MeasurementObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_MeasurementObject_getModelHash(_Underlying *_this);
            return __MR_MeasurementObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::MeasurementObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_MeasurementObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_MeasurementObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    // A common base class for measurement objects.
    // Can't be constructed directly.
    /// Generated from class `MR::MeasurementObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AngleMeasurementObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::RadiusMeasurementObject`
    /// This is the non-const half of the class.
    public class MeasurementObject : Const_MeasurementObject
    {
        internal unsafe MeasurementObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe MeasurementObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(MeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_MeasurementObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_MeasurementObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(MeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_MeasurementObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_MeasurementObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(MeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_MeasurementObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_MeasurementObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator MeasurementObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_MeasurementObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_MeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator MeasurementObject?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_MeasurementObject(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_MeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::MeasurementObject::MeasurementObject`.
        public unsafe MeasurementObject(MR._ByValue_MeasurementObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeasurementObject._Underlying *__MR_MeasurementObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeasurementObject._Underlying *_other);
            _LateMakeShared(__MR_MeasurementObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::MeasurementObject::operator=`.
        public unsafe MR.MeasurementObject Assign(MR._ByValue_MeasurementObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeasurementObject._Underlying *__MR_MeasurementObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeasurementObject._Underlying *_other);
            return new(__MR_MeasurementObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::MeasurementObject::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::MeasurementObject::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::MeasurementObject::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::MeasurementObject::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_MeasurementObject_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::MeasurementObject::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_MeasurementObject_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::MeasurementObject::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::MeasurementObject::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setFrontColor", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_MeasurementObject_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::MeasurementObject::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_MeasurementObject_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::MeasurementObject::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_MeasurementObject_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::MeasurementObject::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setBackColor", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_MeasurementObject_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::MeasurementObject::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_MeasurementObject_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::MeasurementObject::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_MeasurementObject_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::MeasurementObject::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_MeasurementObject_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::MeasurementObject::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setPickable", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::MeasurementObject::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setColoringType", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_MeasurementObject_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::MeasurementObject::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setShininess", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setShininess(_Underlying *_this, float shininess);
            __MR_MeasurementObject_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::MeasurementObject::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_MeasurementObject_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::MeasurementObject::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_MeasurementObject_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::MeasurementObject::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_MeasurementObject_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::MeasurementObject::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_resetFrontColor(_Underlying *_this);
            __MR_MeasurementObject_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::MeasurementObject::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_resetColors", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_resetColors(_Underlying *_this);
            __MR_MeasurementObject_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeasurementObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setName", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_MeasurementObject_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::MeasurementObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setXf", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_MeasurementObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::MeasurementObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_resetXf", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_MeasurementObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::MeasurementObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_MeasurementObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeasurementObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_MeasurementObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::MeasurementObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_applyScale", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_MeasurementObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::MeasurementObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeasurementObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setLocked", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setLocked(_Underlying *_this, byte on);
            __MR_MeasurementObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::MeasurementObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_MeasurementObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::MeasurementObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_detachFromParent(_Underlying *_this);
            return __MR_MeasurementObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::MeasurementObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_addChild", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_MeasurementObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::MeasurementObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_MeasurementObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::MeasurementObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_removeAllChildren(_Underlying *_this);
            __MR_MeasurementObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::MeasurementObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_sortChildren(_Underlying *_this);
            __MR_MeasurementObject_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::MeasurementObject::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_select", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_select(_Underlying *_this, byte on);
            return __MR_MeasurementObject_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::MeasurementObject::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setAncillary(_Underlying *_this, byte ancillary);
            __MR_MeasurementObject_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::MeasurementObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setVisible", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::MeasurementObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_MeasurementObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::MeasurementObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_swap", ExactSpelling = true)]
            extern static void __MR_MeasurementObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_MeasurementObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::MeasurementObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_addTag", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_MeasurementObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::MeasurementObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_MeasurementObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_MeasurementObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeasurementObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeasurementObject`/`Const_MeasurementObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeasurementObject
    {
        internal readonly Const_MeasurementObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeasurementObject(Const_MeasurementObject new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeasurementObject(Const_MeasurementObject arg) {return new(arg);}
        public _ByValue_MeasurementObject(MR.Misc._Moved<MeasurementObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeasurementObject(MR.Misc._Moved<MeasurementObject> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeasurementObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeasurementObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeasurementObject`/`Const_MeasurementObject` directly.
    public class _InOptMut_MeasurementObject
    {
        public MeasurementObject? Opt;

        public _InOptMut_MeasurementObject() {}
        public _InOptMut_MeasurementObject(MeasurementObject value) {Opt = value;}
        public static implicit operator _InOptMut_MeasurementObject(MeasurementObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeasurementObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeasurementObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeasurementObject`/`Const_MeasurementObject` to pass it to the function.
    public class _InOptConst_MeasurementObject
    {
        public Const_MeasurementObject? Opt;

        public _InOptConst_MeasurementObject() {}
        public _InOptConst_MeasurementObject(Const_MeasurementObject value) {Opt = value;}
        public static implicit operator _InOptConst_MeasurementObject(Const_MeasurementObject value) {return new(value);}
    }
}
