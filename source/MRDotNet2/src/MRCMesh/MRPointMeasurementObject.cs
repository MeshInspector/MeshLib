public static partial class MR
{
    public enum PointMeasurementVisualizePropertyType : int
    {
        CapVisibility = 0,
        Count = 1,
    }

    /// Generated from class `MR::PointMeasurementObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectComparableWithReference`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_PointMeasurementObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PointMeasurementObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PointMeasurementObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PointMeasurementObject_UseCount();
                return __MR_std_shared_ptr_MR_PointMeasurementObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointMeasurementObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PointMeasurementObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PointMeasurementObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointMeasurementObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointMeasurementObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointMeasurementObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointMeasurementObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PointMeasurementObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PointMeasurementObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointMeasurementObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PointMeasurementObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointMeasurementObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointMeasurementObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointMeasurementObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PointMeasurementObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PointMeasurementObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointMeasurementObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_MeasurementObject(Const_PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static MR.Const_MeasurementObject._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_MeasurementObject(_Underlying *_this);
            return MR.Const_MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_MeasurementObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_ObjectComparableWithReference(Const_PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_ObjectComparableWithReference", ExactSpelling = true)]
            extern static MR.Const_ObjectComparableWithReference._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_ObjectComparableWithReference(_Underlying *_this);
            return MR.Const_ObjectComparableWithReference._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_ObjectComparableWithReference(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PointMeasurementObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_PointMeasurementObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_PointMeasurementObject?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_PointMeasurementObject(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_PointMeasurementObject?(MR.Const_MeasurementObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_MeasurementObject_DynamicDowncastTo_MR_PointMeasurementObject(MR.Const_MeasurementObject._Underlying *_this);
            var ptr = __MR_MeasurementObject_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_PointMeasurementObject?(MR.Const_ObjectComparableWithReference parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectComparableWithReference_DynamicDowncastTo_MR_PointMeasurementObject(MR.Const_ObjectComparableWithReference._Underlying *_this);
            var ptr = __MR_ObjectComparableWithReference_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_ObjectComparableWithReference._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointMeasurementObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointMeasurementObject._Underlying *__MR_PointMeasurementObject_DefaultConstruct();
            _LateMakeShared(__MR_PointMeasurementObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::PointMeasurementObject::PointMeasurementObject`.
        public unsafe Const_PointMeasurementObject(MR._ByValue_PointMeasurementObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointMeasurementObject._Underlying *__MR_PointMeasurementObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointMeasurementObject._Underlying *_other);
            _LateMakeShared(__MR_PointMeasurementObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::PointMeasurementObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_PointMeasurementObject_StaticTypeName();
            var __ret = __MR_PointMeasurementObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PointMeasurementObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_PointMeasurementObject_typeName(_Underlying *_this);
            var __ret = __MR_PointMeasurementObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PointMeasurementObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_PointMeasurementObject_StaticClassName();
            var __ret = __MR_PointMeasurementObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PointMeasurementObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PointMeasurementObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PointMeasurementObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PointMeasurementObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_PointMeasurementObject_StaticClassNameInPlural();
            var __ret = __MR_PointMeasurementObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PointMeasurementObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PointMeasurementObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PointMeasurementObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PointMeasurementObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PointMeasurementObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PointMeasurementObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PointMeasurementObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PointMeasurementObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PointMeasurementObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PointMeasurementObject::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_PointMeasurementObject_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PointMeasurementObject::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_PointMeasurementObject_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_PointMeasurementObject_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PointMeasurementObject::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_PointMeasurementObject_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_PointMeasurementObject_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// calculates point from xf
        /// Generated from method `MR::PointMeasurementObject::getLocalPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetLocalPoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getLocalPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PointMeasurementObject_getLocalPoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_PointMeasurementObject_getLocalPoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PointMeasurementObject::getWorldPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetWorldPoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getWorldPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PointMeasurementObject_getWorldPoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_PointMeasurementObject_getWorldPoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        // Implement `ObjectComparableWithReference`:
        /// Generated from method `MR::PointMeasurementObject::numComparableProperties`.
        public unsafe ulong NumComparableProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_numComparableProperties", ExactSpelling = true)]
            extern static ulong __MR_PointMeasurementObject_numComparableProperties(_Underlying *_this);
            return __MR_PointMeasurementObject_numComparableProperties(_UnderlyingPtr);
        }

        /// Generated from method `MR::PointMeasurementObject::getComparablePropertyName`.
        public unsafe MR.Std.StringView GetComparablePropertyName(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getComparablePropertyName", ExactSpelling = true)]
            extern static MR.Std.StringView._Underlying *__MR_PointMeasurementObject_getComparablePropertyName(_Underlying *_this, ulong i);
            return new(__MR_PointMeasurementObject_getComparablePropertyName(_UnderlyingPtr, i), is_owning: true);
        }

        /// Generated from method `MR::PointMeasurementObject::computeComparableProperty`.
        public unsafe MR.Std.Optional_MRObjectComparableWithReferenceComparableProperty ComputeComparableProperty(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_computeComparableProperty", ExactSpelling = true)]
            extern static MR.Std.Optional_MRObjectComparableWithReferenceComparableProperty._Underlying *__MR_PointMeasurementObject_computeComparableProperty(_Underlying *_this, ulong i);
            return new(__MR_PointMeasurementObject_computeComparableProperty(_UnderlyingPtr, i), is_owning: true);
        }

        /// Generated from method `MR::PointMeasurementObject::getComparisonTolerence`.
        public unsafe MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance GetComparisonTolerence(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getComparisonTolerence", ExactSpelling = true)]
            extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_PointMeasurementObject_getComparisonTolerence(_Underlying *_this, ulong i);
            return new(__MR_PointMeasurementObject_getComparisonTolerence(_UnderlyingPtr, i), is_owning: true);
        }

        /// Generated from method `MR::PointMeasurementObject::comparisonToleranceIsAlwaysOnlyPositive`.
        public unsafe bool ComparisonToleranceIsAlwaysOnlyPositive(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_comparisonToleranceIsAlwaysOnlyPositive", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_comparisonToleranceIsAlwaysOnlyPositive(_Underlying *_this, ulong i);
            return __MR_PointMeasurementObject_comparisonToleranceIsAlwaysOnlyPositive(_UnderlyingPtr, i) != 0;
        }

        // This returns 2: the point, and the optional normal direction. The normal doesn't need to be normalized, its length doesn't affect calculations.
        // If the normal isn't specified, the Euclidean distance gets used.
        /// Generated from method `MR::PointMeasurementObject::numComparisonReferenceValues`.
        public unsafe ulong NumComparisonReferenceValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_numComparisonReferenceValues", ExactSpelling = true)]
            extern static ulong __MR_PointMeasurementObject_numComparisonReferenceValues(_Underlying *_this);
            return __MR_PointMeasurementObject_numComparisonReferenceValues(_UnderlyingPtr);
        }

        /// Generated from method `MR::PointMeasurementObject::getComparisonReferenceValueName`.
        public unsafe MR.Std.StringView GetComparisonReferenceValueName(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getComparisonReferenceValueName", ExactSpelling = true)]
            extern static MR.Std.StringView._Underlying *__MR_PointMeasurementObject_getComparisonReferenceValueName(_Underlying *_this, ulong i);
            return new(__MR_PointMeasurementObject_getComparisonReferenceValueName(_UnderlyingPtr, i), is_owning: true);
        }

        /// Generated from method `MR::PointMeasurementObject::getComparisonReferenceValue`.
        public unsafe MR.ObjectComparableWithReference.ComparisonReferenceValue GetComparisonReferenceValue(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getComparisonReferenceValue", ExactSpelling = true)]
            extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_PointMeasurementObject_getComparisonReferenceValue(_Underlying *_this, ulong i);
            return new(__MR_PointMeasurementObject_getComparisonReferenceValue(_UnderlyingPtr, i), is_owning: true);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::PointMeasurementObject::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PointMeasurementObject_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::PointMeasurementObject::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_PointMeasurementObject_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_PointMeasurementObject_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::PointMeasurementObject::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PointMeasurementObject_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::PointMeasurementObject::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_PointMeasurementObject_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_PointMeasurementObject_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::PointMeasurementObject::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_PointMeasurementObject_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_PointMeasurementObject_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::PointMeasurementObject::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_PointMeasurementObject_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_PointMeasurementObject_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::PointMeasurementObject::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_PointMeasurementObject_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_PointMeasurementObject_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::PointMeasurementObject::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_PointMeasurementObject_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_PointMeasurementObject_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::PointMeasurementObject::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_PointMeasurementObject_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_PointMeasurementObject_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::PointMeasurementObject::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_PointMeasurementObject_getDirtyFlags(_Underlying *_this);
            return __MR_PointMeasurementObject_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::PointMeasurementObject::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_resetDirty", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_resetDirty(_Underlying *_this);
            __MR_PointMeasurementObject_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::PointMeasurementObject::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_PointMeasurementObject_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::PointMeasurementObject::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PointMeasurementObject_getBoundingBox(_Underlying *_this);
            return __MR_PointMeasurementObject_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::PointMeasurementObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PointMeasurementObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_PointMeasurementObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::PointMeasurementObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PointMeasurementObject_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::PointMeasurementObject::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isPickable", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PointMeasurementObject_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::PointMeasurementObject::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_PointMeasurementObject_getColoringType(_Underlying *_this);
            return __MR_PointMeasurementObject_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::PointMeasurementObject::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getShininess", ExactSpelling = true)]
            extern static float __MR_PointMeasurementObject_getShininess(_Underlying *_this);
            return __MR_PointMeasurementObject_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::PointMeasurementObject::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_PointMeasurementObject_getSpecularStrength(_Underlying *_this);
            return __MR_PointMeasurementObject_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::PointMeasurementObject::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_PointMeasurementObject_getAmbientStrength(_Underlying *_this);
            return __MR_PointMeasurementObject_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::PointMeasurementObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_render", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_PointMeasurementObject_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::PointMeasurementObject::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_renderForPicker", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_PointMeasurementObject_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::PointMeasurementObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_renderUi", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_PointMeasurementObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::PointMeasurementObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PointMeasurementObject_heapBytes(_Underlying *_this);
            return __MR_PointMeasurementObject_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::PointMeasurementObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_PointMeasurementObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_PointMeasurementObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::PointMeasurementObject::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_PointMeasurementObject_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PointMeasurementObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_PointMeasurementObject_name(_Underlying *_this);
            return new(__MR_PointMeasurementObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::PointMeasurementObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_PointMeasurementObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PointMeasurementObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::PointMeasurementObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_PointMeasurementObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_PointMeasurementObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::PointMeasurementObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_PointMeasurementObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PointMeasurementObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::PointMeasurementObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_PointMeasurementObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_PointMeasurementObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::PointMeasurementObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PointMeasurementObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::PointMeasurementObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isLocked(_Underlying *_this);
            return __MR_PointMeasurementObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::PointMeasurementObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isParentLocked(_Underlying *_this);
            return __MR_PointMeasurementObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::PointMeasurementObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_PointMeasurementObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::PointMeasurementObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isSelected(_Underlying *_this);
            return __MR_PointMeasurementObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PointMeasurementObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isAncillary(_Underlying *_this);
            return __MR_PointMeasurementObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::PointMeasurementObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isGlobalAncillary(_Underlying *_this);
            return __MR_PointMeasurementObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::PointMeasurementObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PointMeasurementObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::PointMeasurementObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_PointMeasurementObject_visibilityMask(_Underlying *_this);
            return new(__MR_PointMeasurementObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::PointMeasurementObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_resetRedrawFlag(_Underlying *_this);
            __MR_PointMeasurementObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::PointMeasurementObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PointMeasurementObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PointMeasurementObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::PointMeasurementObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PointMeasurementObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PointMeasurementObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::PointMeasurementObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PointMeasurementObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_PointMeasurementObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::PointMeasurementObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_PointMeasurementObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::PointMeasurementObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_hasModel(_Underlying *_this);
            return __MR_PointMeasurementObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::PointMeasurementObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_PointMeasurementObject_tags(_Underlying *_this);
            return new(__MR_PointMeasurementObject_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::PointMeasurementObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_PointMeasurementObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::PointMeasurementObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_PointMeasurementObject_getModelHash(_Underlying *_this);
            return __MR_PointMeasurementObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::PointMeasurementObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PointMeasurementObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PointMeasurementObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// Generated from class `MR::PointMeasurementObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectComparableWithReference`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class PointMeasurementObject : Const_PointMeasurementObject
    {
        internal unsafe PointMeasurementObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PointMeasurementObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.MeasurementObject(PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_MeasurementObject", ExactSpelling = true)]
            extern static MR.MeasurementObject._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_MeasurementObject(_Underlying *_this);
            return MR.MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_MeasurementObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.ObjectComparableWithReference(PointMeasurementObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_UpcastTo_MR_ObjectComparableWithReference", ExactSpelling = true)]
            extern static MR.ObjectComparableWithReference._Underlying *__MR_PointMeasurementObject_UpcastTo_MR_ObjectComparableWithReference(_Underlying *_this);
            return MR.ObjectComparableWithReference._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointMeasurementObject_UpcastTo_MR_ObjectComparableWithReference(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PointMeasurementObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_PointMeasurementObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator PointMeasurementObject?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_PointMeasurementObject(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator PointMeasurementObject?(MR.MeasurementObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeasurementObject_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_MeasurementObject_DynamicDowncastTo_MR_PointMeasurementObject(MR.MeasurementObject._Underlying *_this);
            var ptr = __MR_MeasurementObject_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.MeasurementObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator PointMeasurementObject?(MR.ObjectComparableWithReference parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_DynamicDowncastTo_MR_PointMeasurementObject", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectComparableWithReference_DynamicDowncastTo_MR_PointMeasurementObject(MR.ObjectComparableWithReference._Underlying *_this);
            var ptr = __MR_ObjectComparableWithReference_DynamicDowncastTo_MR_PointMeasurementObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.ObjectComparableWithReference._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointMeasurementObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointMeasurementObject._Underlying *__MR_PointMeasurementObject_DefaultConstruct();
            _LateMakeShared(__MR_PointMeasurementObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::PointMeasurementObject::PointMeasurementObject`.
        public unsafe PointMeasurementObject(MR._ByValue_PointMeasurementObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointMeasurementObject._Underlying *__MR_PointMeasurementObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointMeasurementObject._Underlying *_other);
            _LateMakeShared(__MR_PointMeasurementObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::PointMeasurementObject::operator=`.
        public unsafe MR.PointMeasurementObject Assign(MR._ByValue_PointMeasurementObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointMeasurementObject._Underlying *__MR_PointMeasurementObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointMeasurementObject._Underlying *_other);
            return new(__MR_PointMeasurementObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// updates xf to fit given point
        /// Generated from method `MR::PointMeasurementObject::setLocalPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetLocalPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setLocalPoint", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setLocalPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            __MR_PointMeasurementObject_setLocalPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PointMeasurementObject::setWorldPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setWorldPoint", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setWorldPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            __MR_PointMeasurementObject_setWorldPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PointMeasurementObject::setComparisonTolerance`.
        public unsafe void SetComparisonTolerance(ulong i, MR.ObjectComparableWithReference.Const_ComparisonTolerance? newTolerance)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setComparisonTolerance", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setComparisonTolerance(_Underlying *_this, ulong i, MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *newTolerance);
            __MR_PointMeasurementObject_setComparisonTolerance(_UnderlyingPtr, i, newTolerance is not null ? newTolerance._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointMeasurementObject::setComparisonReferenceValue`.
        public unsafe void SetComparisonReferenceValue(ulong i, MR.Std.Const_Variant_Float_MRVector3f? value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setComparisonReferenceValue", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setComparisonReferenceValue(_Underlying *_this, ulong i, MR.Std.Variant_Float_MRVector3f._Underlying *value);
            __MR_PointMeasurementObject_setComparisonReferenceValue(_UnderlyingPtr, i, value is not null ? value._UnderlyingPtr : null);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::PointMeasurementObject::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::PointMeasurementObject::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::PointMeasurementObject::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::PointMeasurementObject::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_PointMeasurementObject_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::PointMeasurementObject::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_PointMeasurementObject_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::PointMeasurementObject::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::PointMeasurementObject::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setFrontColor", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_PointMeasurementObject_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::PointMeasurementObject::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PointMeasurementObject_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::PointMeasurementObject::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_PointMeasurementObject_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::PointMeasurementObject::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setBackColor", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_PointMeasurementObject_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::PointMeasurementObject::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_PointMeasurementObject_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::PointMeasurementObject::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_PointMeasurementObject_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::PointMeasurementObject::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PointMeasurementObject_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::PointMeasurementObject::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setPickable", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::PointMeasurementObject::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setColoringType", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_PointMeasurementObject_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::PointMeasurementObject::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setShininess", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setShininess(_Underlying *_this, float shininess);
            __MR_PointMeasurementObject_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::PointMeasurementObject::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_PointMeasurementObject_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::PointMeasurementObject::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_PointMeasurementObject_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::PointMeasurementObject::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_PointMeasurementObject_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::PointMeasurementObject::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_resetFrontColor(_Underlying *_this);
            __MR_PointMeasurementObject_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::PointMeasurementObject::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_resetColors", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_resetColors(_Underlying *_this);
            __MR_PointMeasurementObject_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::PointMeasurementObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setName", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_PointMeasurementObject_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::PointMeasurementObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setXf", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_PointMeasurementObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::PointMeasurementObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_resetXf", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_PointMeasurementObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::PointMeasurementObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_PointMeasurementObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointMeasurementObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_PointMeasurementObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::PointMeasurementObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_applyScale", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_PointMeasurementObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::PointMeasurementObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointMeasurementObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setLocked", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setLocked(_Underlying *_this, byte on);
            __MR_PointMeasurementObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::PointMeasurementObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_PointMeasurementObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::PointMeasurementObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_detachFromParent(_Underlying *_this);
            return __MR_PointMeasurementObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::PointMeasurementObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_addChild", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_PointMeasurementObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::PointMeasurementObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_PointMeasurementObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::PointMeasurementObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_removeAllChildren(_Underlying *_this);
            __MR_PointMeasurementObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::PointMeasurementObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_sortChildren(_Underlying *_this);
            __MR_PointMeasurementObject_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::PointMeasurementObject::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_select", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_select(_Underlying *_this, byte on);
            return __MR_PointMeasurementObject_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::PointMeasurementObject::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setAncillary(_Underlying *_this, byte ancillary);
            __MR_PointMeasurementObject_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::PointMeasurementObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setVisible", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::PointMeasurementObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_PointMeasurementObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::PointMeasurementObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_swap", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_PointMeasurementObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::PointMeasurementObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_addTag", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_PointMeasurementObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::PointMeasurementObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_PointMeasurementObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_PointMeasurementObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// Generated from method `MR::PointMeasurementObject::setComparisonReferenceVal`.
        public unsafe void SetComparisonReferenceVal(ulong i, MR.ObjectComparableWithReference.Const_ComparisonReferenceValue value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointMeasurementObject_setComparisonReferenceVal", ExactSpelling = true)]
            extern static void __MR_PointMeasurementObject_setComparisonReferenceVal(_Underlying *_this, ulong i, MR.ObjectComparableWithReference.Const_ComparisonReferenceValue._Underlying *value);
            __MR_PointMeasurementObject_setComparisonReferenceVal(_UnderlyingPtr, i, value._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointMeasurementObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointMeasurementObject`/`Const_PointMeasurementObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointMeasurementObject
    {
        internal readonly Const_PointMeasurementObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointMeasurementObject() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointMeasurementObject(MR.Misc._Moved<PointMeasurementObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointMeasurementObject(MR.Misc._Moved<PointMeasurementObject> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointMeasurementObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointMeasurementObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointMeasurementObject`/`Const_PointMeasurementObject` directly.
    public class _InOptMut_PointMeasurementObject
    {
        public PointMeasurementObject? Opt;

        public _InOptMut_PointMeasurementObject() {}
        public _InOptMut_PointMeasurementObject(PointMeasurementObject value) {Opt = value;}
        public static implicit operator _InOptMut_PointMeasurementObject(PointMeasurementObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointMeasurementObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointMeasurementObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointMeasurementObject`/`Const_PointMeasurementObject` to pass it to the function.
    public class _InOptConst_PointMeasurementObject
    {
        public Const_PointMeasurementObject? Opt;

        public _InOptConst_PointMeasurementObject() {}
        public _InOptConst_PointMeasurementObject(Const_PointMeasurementObject value) {Opt = value;}
        public static implicit operator _InOptConst_PointMeasurementObject(Const_PointMeasurementObject value) {return new(value);}
    }
}
