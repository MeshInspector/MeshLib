public static partial class MR
{
    // Represents a radius measurement.
    /// Generated from class `MR::RadiusMeasurementObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeasurementObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_RadiusMeasurementObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_RadiusMeasurementObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_RadiusMeasurementObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_RadiusMeasurementObject_UseCount();
                return __MR_std_shared_ptr_MR_RadiusMeasurementObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_RadiusMeasurementObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RadiusMeasurementObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_RadiusMeasurementObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_RadiusMeasurementObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe RadiusMeasurementObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_RadiusMeasurementObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_RadiusMeasurementObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_RadiusMeasurementObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_RadiusMeasurementObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_RadiusMeasurementObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_RadiusMeasurementObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RadiusMeasurementObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_MeasurementObject(Const_RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static MR.Const_MeasurementObject._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_MeasurementObject(_Underlying *_this);
            return MR.Const_MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_MeasurementObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_RadiusMeasurementObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_RadiusMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_RadiusMeasurementObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_RadiusMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_RadiusMeasurementObject?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_RadiusMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_RadiusMeasurementObject(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_RadiusMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_RadiusMeasurementObject?(MR.Const_MeasurementObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_DynamicDowncastTo_MR_RadiusMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_MeasurementObject_DynamicDowncastTo_MR_RadiusMeasurementObject(MR.Const_MeasurementObject._Underlying *_this);
            var ptr = __MR_MeasurementObject_DynamicDowncastTo_MR_RadiusMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RadiusMeasurementObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RadiusMeasurementObject._Underlying *__MR_RadiusMeasurementObject_DefaultConstruct();
            _LateMakeShared(__MR_RadiusMeasurementObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::RadiusMeasurementObject::RadiusMeasurementObject`.
        public unsafe Const_RadiusMeasurementObject(MR._ByValue_RadiusMeasurementObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RadiusMeasurementObject._Underlying *__MR_RadiusMeasurementObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RadiusMeasurementObject._Underlying *_other);
            _LateMakeShared(__MR_RadiusMeasurementObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::RadiusMeasurementObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_RadiusMeasurementObject_StaticTypeName();
            var __ret = __MR_RadiusMeasurementObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::RadiusMeasurementObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_RadiusMeasurementObject_typeName(_Underlying *_this);
            var __ret = __MR_RadiusMeasurementObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::RadiusMeasurementObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_RadiusMeasurementObject_StaticClassName();
            var __ret = __MR_RadiusMeasurementObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::RadiusMeasurementObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_RadiusMeasurementObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_RadiusMeasurementObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RadiusMeasurementObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_RadiusMeasurementObject_StaticClassNameInPlural();
            var __ret = __MR_RadiusMeasurementObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::RadiusMeasurementObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_RadiusMeasurementObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_RadiusMeasurementObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RadiusMeasurementObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_RadiusMeasurementObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_RadiusMeasurementObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RadiusMeasurementObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_RadiusMeasurementObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_RadiusMeasurementObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        // Get the center in world coordinates.
        /// Generated from method `MR::RadiusMeasurementObject::getWorldCenter`.
        public unsafe MR.Vector3f GetWorldCenter()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getWorldCenter", ExactSpelling = true)]
            extern static MR.Vector3f __MR_RadiusMeasurementObject_getWorldCenter(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getWorldCenter(_UnderlyingPtr);
        }

        // Get the center in local coordinates.
        /// Generated from method `MR::RadiusMeasurementObject::getLocalCenter`.
        public unsafe MR.Vector3f GetLocalCenter()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getLocalCenter", ExactSpelling = true)]
            extern static MR.Vector3f __MR_RadiusMeasurementObject_getLocalCenter(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getLocalCenter(_UnderlyingPtr);
        }

        // The length of this vector is the radius, and the direction is the preferred line drawing direction.
        /// Generated from method `MR::RadiusMeasurementObject::getWorldRadiusAsVector`.
        public unsafe MR.Vector3f GetWorldRadiusAsVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getWorldRadiusAsVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_RadiusMeasurementObject_getWorldRadiusAsVector(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getWorldRadiusAsVector(_UnderlyingPtr);
        }

        /// Generated from method `MR::RadiusMeasurementObject::getLocalRadiusAsVector`.
        public unsafe MR.Vector3f GetLocalRadiusAsVector()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getLocalRadiusAsVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_RadiusMeasurementObject_getLocalRadiusAsVector(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getLocalRadiusAsVector(_UnderlyingPtr);
        }

        // The preferred radius normal, for non-spherical radiuses.
        /// Generated from method `MR::RadiusMeasurementObject::getWorldNormal`.
        public unsafe MR.Vector3f GetWorldNormal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getWorldNormal", ExactSpelling = true)]
            extern static MR.Vector3f __MR_RadiusMeasurementObject_getWorldNormal(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getWorldNormal(_UnderlyingPtr);
        }

        /// Generated from method `MR::RadiusMeasurementObject::getLocalNormal`.
        public unsafe MR.Vector3f GetLocalNormal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getLocalNormal", ExactSpelling = true)]
            extern static MR.Vector3f __MR_RadiusMeasurementObject_getLocalNormal(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getLocalNormal(_UnderlyingPtr);
        }

        // Whether we should draw this as a diameter instead of a radius.
        /// Generated from method `MR::RadiusMeasurementObject::getDrawAsDiameter`.
        public unsafe bool GetDrawAsDiameter()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getDrawAsDiameter", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_getDrawAsDiameter(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getDrawAsDiameter(_UnderlyingPtr) != 0;
        }

        // Whether this is a sphere radius, as opposed to circle/cylinder radius.
        /// Generated from method `MR::RadiusMeasurementObject::getIsSpherical`.
        public unsafe bool GetIsSpherical()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getIsSpherical", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_getIsSpherical(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getIsSpherical(_UnderlyingPtr) != 0;
        }

        // The visual leader line length multiplier, relative to the radius.
        // You're recommended to set a min absolute value for the resulting length when rendering.
        /// Generated from method `MR::RadiusMeasurementObject::getVisualLengthMultiplier`.
        public unsafe float GetVisualLengthMultiplier()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getVisualLengthMultiplier", ExactSpelling = true)]
            extern static float __MR_RadiusMeasurementObject_getVisualLengthMultiplier(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getVisualLengthMultiplier(_UnderlyingPtr);
        }

        // Computes the radius/diameter value, as if by `getLocalRadiusAsVector()`, possibly multiplied by two if `getDrawAsDiameter()`.
        /// Generated from method `MR::RadiusMeasurementObject::computeRadiusOrDiameter`.
        public unsafe float ComputeRadiusOrDiameter()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_computeRadiusOrDiameter", ExactSpelling = true)]
            extern static float __MR_RadiusMeasurementObject_computeRadiusOrDiameter(_Underlying *_this);
            return __MR_RadiusMeasurementObject_computeRadiusOrDiameter(_UnderlyingPtr);
        }

        /// Generated from method `MR::RadiusMeasurementObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_RadiusMeasurementObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_RadiusMeasurementObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// Returns true if this class supports the property `type`. Otherwise passing it to the functions below is illegal.
        /// Generated from method `MR::RadiusMeasurementObject::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_RadiusMeasurementObject_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::RadiusMeasurementObject::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_RadiusMeasurementObject_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::RadiusMeasurementObject::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_RadiusMeasurementObject_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_RadiusMeasurementObject_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// get all visualize properties masks
        /// Generated from method `MR::RadiusMeasurementObject::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_RadiusMeasurementObject_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_RadiusMeasurementObject_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::RadiusMeasurementObject::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_RadiusMeasurementObject_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::RadiusMeasurementObject::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_RadiusMeasurementObject_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::RadiusMeasurementObject::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_RadiusMeasurementObject_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_RadiusMeasurementObject_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::RadiusMeasurementObject::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_RadiusMeasurementObject_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_RadiusMeasurementObject_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::RadiusMeasurementObject::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_RadiusMeasurementObject_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::RadiusMeasurementObject::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_RadiusMeasurementObject_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_RadiusMeasurementObject_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::RadiusMeasurementObject::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_RadiusMeasurementObject_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_RadiusMeasurementObject_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::RadiusMeasurementObject::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_RadiusMeasurementObject_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::RadiusMeasurementObject::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_RadiusMeasurementObject_getDirtyFlags(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::RadiusMeasurementObject::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_resetDirty", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_resetDirty(_Underlying *_this);
            __MR_RadiusMeasurementObject_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::RadiusMeasurementObject::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_RadiusMeasurementObject_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::RadiusMeasurementObject::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_RadiusMeasurementObject_getBoundingBox(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::RadiusMeasurementObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_RadiusMeasurementObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_RadiusMeasurementObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::RadiusMeasurementObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_RadiusMeasurementObject_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::RadiusMeasurementObject::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isPickable", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_RadiusMeasurementObject_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::RadiusMeasurementObject::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_RadiusMeasurementObject_getColoringType(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::RadiusMeasurementObject::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getShininess", ExactSpelling = true)]
            extern static float __MR_RadiusMeasurementObject_getShininess(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::RadiusMeasurementObject::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_RadiusMeasurementObject_getSpecularStrength(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::RadiusMeasurementObject::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_RadiusMeasurementObject_getAmbientStrength(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::RadiusMeasurementObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_render", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_RadiusMeasurementObject_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::RadiusMeasurementObject::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_renderForPicker", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_RadiusMeasurementObject_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::RadiusMeasurementObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_renderUi", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_RadiusMeasurementObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::RadiusMeasurementObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_RadiusMeasurementObject_heapBytes(_Underlying *_this);
            return __MR_RadiusMeasurementObject_heapBytes(_UnderlyingPtr);
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::RadiusMeasurementObject::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_RadiusMeasurementObject_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RadiusMeasurementObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_RadiusMeasurementObject_name(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::RadiusMeasurementObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_RadiusMeasurementObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_RadiusMeasurementObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::RadiusMeasurementObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_RadiusMeasurementObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::RadiusMeasurementObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_RadiusMeasurementObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_RadiusMeasurementObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::RadiusMeasurementObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_RadiusMeasurementObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::RadiusMeasurementObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_RadiusMeasurementObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::RadiusMeasurementObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isLocked(_Underlying *_this);
            return __MR_RadiusMeasurementObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::RadiusMeasurementObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isParentLocked(_Underlying *_this);
            return __MR_RadiusMeasurementObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::RadiusMeasurementObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_RadiusMeasurementObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::RadiusMeasurementObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isSelected(_Underlying *_this);
            return __MR_RadiusMeasurementObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RadiusMeasurementObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isAncillary(_Underlying *_this);
            return __MR_RadiusMeasurementObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::RadiusMeasurementObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isGlobalAncillary(_Underlying *_this);
            return __MR_RadiusMeasurementObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::RadiusMeasurementObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_RadiusMeasurementObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::RadiusMeasurementObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_RadiusMeasurementObject_visibilityMask(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::RadiusMeasurementObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_resetRedrawFlag(_Underlying *_this);
            __MR_RadiusMeasurementObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::RadiusMeasurementObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_RadiusMeasurementObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_RadiusMeasurementObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::RadiusMeasurementObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_RadiusMeasurementObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_RadiusMeasurementObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::RadiusMeasurementObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_RadiusMeasurementObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_RadiusMeasurementObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::RadiusMeasurementObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_RadiusMeasurementObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::RadiusMeasurementObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_hasModel(_Underlying *_this);
            return __MR_RadiusMeasurementObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::RadiusMeasurementObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_RadiusMeasurementObject_tags(_Underlying *_this);
            return new(__MR_RadiusMeasurementObject_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::RadiusMeasurementObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_RadiusMeasurementObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::RadiusMeasurementObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_RadiusMeasurementObject_getModelHash(_Underlying *_this);
            return __MR_RadiusMeasurementObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::RadiusMeasurementObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_RadiusMeasurementObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_RadiusMeasurementObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    // Represents a radius measurement.
    /// Generated from class `MR::RadiusMeasurementObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeasurementObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class RadiusMeasurementObject : Const_RadiusMeasurementObject
    {
        internal unsafe RadiusMeasurementObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe RadiusMeasurementObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.MeasurementObject(RadiusMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_UpcastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static MR.MeasurementObject._Underlying *__MR_RadiusMeasurementObject_UpcastTo_MR_MeasurementObject(_Underlying *_this);
            return MR.MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_RadiusMeasurementObject_UpcastTo_MR_MeasurementObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator RadiusMeasurementObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_RadiusMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_RadiusMeasurementObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_RadiusMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator RadiusMeasurementObject?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_RadiusMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_RadiusMeasurementObject(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_RadiusMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator RadiusMeasurementObject?(MR.MeasurementObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_DynamicDowncastTo_MR_RadiusMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_MeasurementObject_DynamicDowncastTo_MR_RadiusMeasurementObject(MR.MeasurementObject._Underlying *_this);
            var ptr = __MR_MeasurementObject_DynamicDowncastTo_MR_RadiusMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RadiusMeasurementObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RadiusMeasurementObject._Underlying *__MR_RadiusMeasurementObject_DefaultConstruct();
            _LateMakeShared(__MR_RadiusMeasurementObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::RadiusMeasurementObject::RadiusMeasurementObject`.
        public unsafe RadiusMeasurementObject(MR._ByValue_RadiusMeasurementObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RadiusMeasurementObject._Underlying *__MR_RadiusMeasurementObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RadiusMeasurementObject._Underlying *_other);
            _LateMakeShared(__MR_RadiusMeasurementObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::RadiusMeasurementObject::operator=`.
        public unsafe MR.RadiusMeasurementObject Assign(MR._ByValue_RadiusMeasurementObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RadiusMeasurementObject._Underlying *__MR_RadiusMeasurementObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RadiusMeasurementObject._Underlying *_other);
            return new(__MR_RadiusMeasurementObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setLocalCenter`.
        public unsafe void SetLocalCenter(MR.Const_Vector3f center)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setLocalCenter", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setLocalCenter(_Underlying *_this, MR.Const_Vector3f._Underlying *center);
            __MR_RadiusMeasurementObject_setLocalCenter(_UnderlyingPtr, center._UnderlyingPtr);
        }

        // Sets the local radius vector (the length of which is the radius value),
        //   and also the radius normal (which is ignored for spherical radiuses).
        // The normal is automatically normalized and made perpendicular to the `radiusVec`.
        /// Generated from method `MR::RadiusMeasurementObject::setLocalRadiusAsVector`.
        public unsafe void SetLocalRadiusAsVector(MR.Const_Vector3f radiusVec, MR.Const_Vector3f normal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setLocalRadiusAsVector_2", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setLocalRadiusAsVector_2(_Underlying *_this, MR.Const_Vector3f._Underlying *radiusVec, MR.Const_Vector3f._Underlying *normal);
            __MR_RadiusMeasurementObject_setLocalRadiusAsVector_2(_UnderlyingPtr, radiusVec._UnderlyingPtr, normal._UnderlyingPtr);
        }

        // Same, but without a preferred normal.
        /// Generated from method `MR::RadiusMeasurementObject::setLocalRadiusAsVector`.
        public unsafe void SetLocalRadiusAsVector(MR.Const_Vector3f radiusVec)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setLocalRadiusAsVector_1", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setLocalRadiusAsVector_1(_Underlying *_this, MR.Const_Vector3f._Underlying *radiusVec);
            __MR_RadiusMeasurementObject_setLocalRadiusAsVector_1(_UnderlyingPtr, radiusVec._UnderlyingPtr);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setDrawAsDiameter`.
        public unsafe void SetDrawAsDiameter(bool value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setDrawAsDiameter", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setDrawAsDiameter(_Underlying *_this, byte value);
            __MR_RadiusMeasurementObject_setDrawAsDiameter(_UnderlyingPtr, value ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setIsSpherical`.
        public unsafe void SetIsSpherical(bool value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setIsSpherical", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setIsSpherical(_Underlying *_this, byte value);
            __MR_RadiusMeasurementObject_setIsSpherical(_UnderlyingPtr, value ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setVisualLengthMultiplier`.
        public unsafe void SetVisualLengthMultiplier(float value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setVisualLengthMultiplier", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setVisualLengthMultiplier(_Underlying *_this, float value);
            __MR_RadiusMeasurementObject_setVisualLengthMultiplier(_UnderlyingPtr, value);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::RadiusMeasurementObject::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::RadiusMeasurementObject::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::RadiusMeasurementObject::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::RadiusMeasurementObject::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_RadiusMeasurementObject_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::RadiusMeasurementObject::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_RadiusMeasurementObject_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::RadiusMeasurementObject::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::RadiusMeasurementObject::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setFrontColor", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_RadiusMeasurementObject_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::RadiusMeasurementObject::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_RadiusMeasurementObject_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::RadiusMeasurementObject::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_RadiusMeasurementObject_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::RadiusMeasurementObject::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setBackColor", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_RadiusMeasurementObject_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::RadiusMeasurementObject::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_RadiusMeasurementObject_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::RadiusMeasurementObject::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_RadiusMeasurementObject_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::RadiusMeasurementObject::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_RadiusMeasurementObject_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::RadiusMeasurementObject::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setPickable", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::RadiusMeasurementObject::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setColoringType", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_RadiusMeasurementObject_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::RadiusMeasurementObject::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setShininess", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setShininess(_Underlying *_this, float shininess);
            __MR_RadiusMeasurementObject_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::RadiusMeasurementObject::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_RadiusMeasurementObject_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::RadiusMeasurementObject::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_RadiusMeasurementObject_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::RadiusMeasurementObject::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_RadiusMeasurementObject_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::RadiusMeasurementObject::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_resetFrontColor(_Underlying *_this);
            __MR_RadiusMeasurementObject_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::RadiusMeasurementObject::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_resetColors", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_resetColors(_Underlying *_this);
            __MR_RadiusMeasurementObject_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setName", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_RadiusMeasurementObject_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::RadiusMeasurementObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setXf", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_RadiusMeasurementObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::RadiusMeasurementObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_resetXf", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_RadiusMeasurementObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::RadiusMeasurementObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_RadiusMeasurementObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_RadiusMeasurementObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::RadiusMeasurementObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_applyScale", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_RadiusMeasurementObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::RadiusMeasurementObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setLocked", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setLocked(_Underlying *_this, byte on);
            __MR_RadiusMeasurementObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RadiusMeasurementObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_RadiusMeasurementObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::RadiusMeasurementObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_detachFromParent(_Underlying *_this);
            return __MR_RadiusMeasurementObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::RadiusMeasurementObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_addChild", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_RadiusMeasurementObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::RadiusMeasurementObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_RadiusMeasurementObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::RadiusMeasurementObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_removeAllChildren(_Underlying *_this);
            __MR_RadiusMeasurementObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::RadiusMeasurementObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_sortChildren(_Underlying *_this);
            __MR_RadiusMeasurementObject_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::RadiusMeasurementObject::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_select", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_select(_Underlying *_this, byte on);
            return __MR_RadiusMeasurementObject_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::RadiusMeasurementObject::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setAncillary(_Underlying *_this, byte ancillary);
            __MR_RadiusMeasurementObject_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::RadiusMeasurementObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setVisible", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::RadiusMeasurementObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_RadiusMeasurementObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::RadiusMeasurementObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_swap", ExactSpelling = true)]
            extern static void __MR_RadiusMeasurementObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_RadiusMeasurementObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::RadiusMeasurementObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_addTag", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_RadiusMeasurementObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::RadiusMeasurementObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RadiusMeasurementObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_RadiusMeasurementObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_RadiusMeasurementObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `RadiusMeasurementObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RadiusMeasurementObject`/`Const_RadiusMeasurementObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RadiusMeasurementObject
    {
        internal readonly Const_RadiusMeasurementObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RadiusMeasurementObject() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RadiusMeasurementObject(MR.Misc._Moved<RadiusMeasurementObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RadiusMeasurementObject(MR.Misc._Moved<RadiusMeasurementObject> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `RadiusMeasurementObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RadiusMeasurementObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RadiusMeasurementObject`/`Const_RadiusMeasurementObject` directly.
    public class _InOptMut_RadiusMeasurementObject
    {
        public RadiusMeasurementObject? Opt;

        public _InOptMut_RadiusMeasurementObject() {}
        public _InOptMut_RadiusMeasurementObject(RadiusMeasurementObject value) {Opt = value;}
        public static implicit operator _InOptMut_RadiusMeasurementObject(RadiusMeasurementObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `RadiusMeasurementObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RadiusMeasurementObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RadiusMeasurementObject`/`Const_RadiusMeasurementObject` to pass it to the function.
    public class _InOptConst_RadiusMeasurementObject
    {
        public Const_RadiusMeasurementObject? Opt;

        public _InOptConst_RadiusMeasurementObject() {}
        public _InOptConst_RadiusMeasurementObject(Const_RadiusMeasurementObject value) {Opt = value;}
        public static implicit operator _InOptConst_RadiusMeasurementObject(Const_RadiusMeasurementObject value) {return new(value);}
    }
}
