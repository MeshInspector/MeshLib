public static partial class MR
{
    /// Object to show plane feature
    /// Generated from class `MR::PlaneObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_PlaneObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PlaneObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PlaneObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PlaneObject_UseCount();
                return __MR_std_shared_ptr_MR_PlaneObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PlaneObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PlaneObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PlaneObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PlaneObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PlaneObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PlaneObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PlaneObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PlaneObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PlaneObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PlaneObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PlaneObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PlaneObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PlaneObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PlaneObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PlaneObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PlaneObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PlaneObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_PlaneObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_PlaneObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_PlaneObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_FeatureObject(Const_PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.Const_FeatureObject._Underlying *__MR_PlaneObject_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PlaneObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_PlaneObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_PlaneObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_PlaneObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_PlaneObject?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_PlaneObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_PlaneObject(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_PlaneObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_PlaneObject?(MR.Const_FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_PlaneObject", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_PlaneObject(MR.Const_FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_PlaneObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PlaneObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_DefaultConstruct();
            _LateMakeShared(__MR_PlaneObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public unsafe Const_PlaneObject(MR._ByValue_PlaneObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PlaneObject._Underlying *_other);
            _LateMakeShared(__MR_PlaneObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Finds best plane to approx given points
        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public unsafe Const_PlaneObject(MR.Std.Const_Vector_MRVector3f pointsToApprox) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_Construct", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_Construct(MR.Std.Const_Vector_MRVector3f._Underlying *pointsToApprox);
            _LateMakeShared(__MR_PlaneObject_Construct(pointsToApprox._UnderlyingPtr));
        }

        /// Finds best plane to approx given points
        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public static unsafe implicit operator Const_PlaneObject(MR.Std.Const_Vector_MRVector3f pointsToApprox) {return new(pointsToApprox);}

        /// Generated from method `MR::PlaneObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_PlaneObject_StaticTypeName();
            var __ret = __MR_PlaneObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PlaneObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_PlaneObject_typeName(_Underlying *_this);
            var __ret = __MR_PlaneObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PlaneObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_PlaneObject_StaticClassName();
            var __ret = __MR_PlaneObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PlaneObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PlaneObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PlaneObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PlaneObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_PlaneObject_StaticClassNameInPlural();
            var __ret = __MR_PlaneObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::PlaneObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PlaneObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PlaneObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PlaneObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PlaneObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PlaneObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PlaneObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PlaneObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PlaneObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// calculates normal from xf
        /// Generated from method `MR::PlaneObject::getNormal`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetNormal(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getNormal", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PlaneObject_getNormal(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_getNormal(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// calculates center from xf
        /// Generated from method `MR::PlaneObject::getCenter`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetCenter(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getCenter", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PlaneObject_getCenter(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_getCenter(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// calculates plane size from xf
        /// Generated from method `MR::PlaneObject::getSize`.
        /// Parameter `id` defaults to `{}`.
        public unsafe float GetSize(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSize", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSize(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_getSize(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::getSizeX`.
        /// Parameter `id` defaults to `{}`.
        public unsafe float GetSizeX(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSizeX", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSizeX(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_getSizeX(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::getSizeY`.
        /// Parameter `id` defaults to `{}`.
        public unsafe float GetSizeY(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSizeY", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSizeY(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_getSizeY(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// calculates normalized directions of X,Y axis of the plane and normal as Z
        /// Generated from method `MR::PlaneObject::calcLocalBasis`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f CalcLocalBasis(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_calcLocalBasis", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_PlaneObject_calcLocalBasis(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_calcLocalBasis(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        // Returns point considered as base for the feature
        /// Generated from method `MR::PlaneObject::getBasePoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetBasePoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getBasePoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PlaneObject_getBasePoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_PlaneObject_getBasePoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::projectPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.FeatureObjectProjectPointResult ProjectPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_projectPoint", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_PlaneObject_projectPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            return new(__MR_PlaneObject_projectPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: true);
        }

        /// Generated from method `MR::PlaneObject::getAllSharedProperties`.
        public unsafe MR.Std.Const_Vector_MRFeatureObjectSharedProperty GetAllSharedProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getAllSharedProperties", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRFeatureObjectSharedProperty._Underlying *__MR_PlaneObject_getAllSharedProperties(_Underlying *_this);
            return new(__MR_PlaneObject_getAllSharedProperties(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PlaneObject::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_PlaneObject_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PlaneObject::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_PlaneObject_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_PlaneObject_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PlaneObject::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_PlaneObject_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_PlaneObject_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        // The cached orthonormalized rotation matrix.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::PlaneObject::getRotationMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetRotationMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getRotationMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_PlaneObject_getRotationMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PlaneObject_getRotationMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // The cached scale and shear matrix. The main diagnoal stores the scale, and some other elements store the shearing.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::PlaneObject::getScaleShearMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetScaleShearMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getScaleShearMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_PlaneObject_getScaleShearMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PlaneObject_getScaleShearMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // This color is used for subfeatures.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::PlaneObject::getDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetDecorationsColor(bool selected, MR._InOpt_ViewportId viewportId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getDecorationsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_PlaneObject_getDecorationsColor(_Underlying *_this, byte selected, MR.ViewportId *viewportId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PlaneObject_getDecorationsColor(_UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::PlaneObject::getDecorationsColorForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetDecorationsColorForAllViewports(bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_PlaneObject_getDecorationsColorForAllViewports(_Underlying *_this, byte selected);
            return new(__MR_PlaneObject_getDecorationsColorForAllViewports(_UnderlyingPtr, selected ? (byte)1 : (byte)0), is_owning: false);
        }

        // Point size and line width, for primary rendering rather than subfeatures.
        /// Generated from method `MR::PlaneObject::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getPointSize", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getPointSize(_Underlying *_this);
            return __MR_PlaneObject_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneObject::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getLineWidth", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getLineWidth(_Underlying *_this);
            return __MR_PlaneObject_getLineWidth(_UnderlyingPtr);
        }

        // Point size and line width, for subfeatures rather than primary rendering.
        /// Generated from method `MR::PlaneObject::getSubfeaturePointSize`.
        public unsafe float GetSubfeaturePointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSubfeaturePointSize", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSubfeaturePointSize(_Underlying *_this);
            return __MR_PlaneObject_getSubfeaturePointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneObject::getSubfeatureLineWidth`.
        public unsafe float GetSubfeatureLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSubfeatureLineWidth", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSubfeatureLineWidth(_Underlying *_this);
            return __MR_PlaneObject_getSubfeatureLineWidth(_UnderlyingPtr);
        }

        // Per-component alpha multipliers. The global alpha is multiplied by thise.
        /// Generated from method `MR::PlaneObject::getMainFeatureAlpha`.
        public unsafe float GetMainFeatureAlpha()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getMainFeatureAlpha", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getMainFeatureAlpha(_Underlying *_this);
            return __MR_PlaneObject_getMainFeatureAlpha(_UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneObject::getSubfeatureAlphaPoints`.
        public unsafe float GetSubfeatureAlphaPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSubfeatureAlphaPoints(_Underlying *_this);
            return __MR_PlaneObject_getSubfeatureAlphaPoints(_UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneObject::getSubfeatureAlphaLines`.
        public unsafe float GetSubfeatureAlphaLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSubfeatureAlphaLines", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSubfeatureAlphaLines(_Underlying *_this);
            return __MR_PlaneObject_getSubfeatureAlphaLines(_UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneObject::getSubfeatureAlphaMesh`.
        public unsafe float GetSubfeatureAlphaMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSubfeatureAlphaMesh(_Underlying *_this);
            return __MR_PlaneObject_getSubfeatureAlphaMesh(_UnderlyingPtr);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::PlaneObject::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PlaneObject_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::PlaneObject::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_PlaneObject_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_PlaneObject_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::PlaneObject::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PlaneObject_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::PlaneObject::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_PlaneObject_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_PlaneObject_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::PlaneObject::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_PlaneObject_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_PlaneObject_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::PlaneObject::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_PlaneObject_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_PlaneObject_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::PlaneObject::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_PlaneObject_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_PlaneObject_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::PlaneObject::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_PlaneObject_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_PlaneObject_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::PlaneObject::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_PlaneObject_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_PlaneObject_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::PlaneObject::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_PlaneObject_getDirtyFlags(_Underlying *_this);
            return __MR_PlaneObject_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::PlaneObject::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_resetDirty", ExactSpelling = true)]
            extern static void __MR_PlaneObject_resetDirty(_Underlying *_this);
            __MR_PlaneObject_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::PlaneObject::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_PlaneObject_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_PlaneObject_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::PlaneObject::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PlaneObject_getBoundingBox(_Underlying *_this);
            return __MR_PlaneObject_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::PlaneObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PlaneObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_PlaneObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::PlaneObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PlaneObject_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::PlaneObject::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isPickable", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PlaneObject_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::PlaneObject::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_PlaneObject_getColoringType(_Underlying *_this);
            return __MR_PlaneObject_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::PlaneObject::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getShininess", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getShininess(_Underlying *_this);
            return __MR_PlaneObject_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::PlaneObject::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getSpecularStrength(_Underlying *_this);
            return __MR_PlaneObject_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::PlaneObject::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_PlaneObject_getAmbientStrength(_Underlying *_this);
            return __MR_PlaneObject_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::PlaneObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_render", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_PlaneObject_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::PlaneObject::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_renderForPicker", ExactSpelling = true)]
            extern static void __MR_PlaneObject_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_PlaneObject_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::PlaneObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_renderUi", ExactSpelling = true)]
            extern static void __MR_PlaneObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_PlaneObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::PlaneObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PlaneObject_heapBytes(_Underlying *_this);
            return __MR_PlaneObject_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::PlaneObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_PlaneObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_PlaneObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::PlaneObject::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_PlaneObject_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PlaneObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_PlaneObject_name(_Underlying *_this);
            return new(__MR_PlaneObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::PlaneObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_PlaneObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PlaneObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::PlaneObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_PlaneObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_PlaneObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::PlaneObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_PlaneObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_PlaneObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::PlaneObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_PlaneObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_PlaneObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::PlaneObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PlaneObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::PlaneObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isLocked(_Underlying *_this);
            return __MR_PlaneObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::PlaneObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isParentLocked(_Underlying *_this);
            return __MR_PlaneObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::PlaneObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_PlaneObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::PlaneObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isSelected(_Underlying *_this);
            return __MR_PlaneObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PlaneObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isAncillary(_Underlying *_this);
            return __MR_PlaneObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::PlaneObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isGlobalAncillary(_Underlying *_this);
            return __MR_PlaneObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::PlaneObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_PlaneObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::PlaneObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_PlaneObject_visibilityMask(_Underlying *_this);
            return new(__MR_PlaneObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::PlaneObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_PlaneObject_resetRedrawFlag(_Underlying *_this);
            __MR_PlaneObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::PlaneObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PlaneObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PlaneObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::PlaneObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PlaneObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PlaneObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::PlaneObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PlaneObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_PlaneObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::PlaneObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_PlaneObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::PlaneObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_hasModel(_Underlying *_this);
            return __MR_PlaneObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::PlaneObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_PlaneObject_tags(_Underlying *_this);
            return new(__MR_PlaneObject_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::PlaneObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_PlaneObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::PlaneObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_PlaneObject_getModelHash(_Underlying *_this);
            return __MR_PlaneObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::PlaneObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_PlaneObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_PlaneObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// Object to show plane feature
    /// Generated from class `MR::PlaneObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class PlaneObject : Const_PlaneObject
    {
        internal unsafe PlaneObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PlaneObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_PlaneObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_PlaneObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_PlaneObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.FeatureObject(PlaneObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_UpcastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static MR.FeatureObject._Underlying *__MR_PlaneObject_UpcastTo_MR_FeatureObject(_Underlying *_this);
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PlaneObject_UpcastTo_MR_FeatureObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PlaneObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_PlaneObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_PlaneObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_PlaneObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator PlaneObject?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_PlaneObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_PlaneObject(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_PlaneObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator PlaneObject?(MR.FeatureObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_DynamicDowncastTo_MR_PlaneObject", ExactSpelling = true)]
            extern static _Underlying *__MR_FeatureObject_DynamicDowncastTo_MR_PlaneObject(MR.FeatureObject._Underlying *_this);
            var ptr = __MR_FeatureObject_DynamicDowncastTo_MR_PlaneObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.FeatureObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PlaneObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_DefaultConstruct();
            _LateMakeShared(__MR_PlaneObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public unsafe PlaneObject(MR._ByValue_PlaneObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PlaneObject._Underlying *_other);
            _LateMakeShared(__MR_PlaneObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Finds best plane to approx given points
        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public unsafe PlaneObject(MR.Std.Const_Vector_MRVector3f pointsToApprox) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_Construct", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_Construct(MR.Std.Const_Vector_MRVector3f._Underlying *pointsToApprox);
            _LateMakeShared(__MR_PlaneObject_Construct(pointsToApprox._UnderlyingPtr));
        }

        /// Finds best plane to approx given points
        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public static unsafe implicit operator PlaneObject(MR.Std.Const_Vector_MRVector3f pointsToApprox) {return new(pointsToApprox);}

        /// Generated from method `MR::PlaneObject::operator=`.
        public unsafe MR.PlaneObject Assign(MR._ByValue_PlaneObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PlaneObject._Underlying *__MR_PlaneObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PlaneObject._Underlying *_other);
            return new(__MR_PlaneObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// updates xf to fit given normal
        /// Generated from method `MR::PlaneObject::setNormal`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetNormal(MR.Const_Vector3f normal, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setNormal", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setNormal(_Underlying *_this, MR.Const_Vector3f._Underlying *normal, MR.ViewportId *id);
            __MR_PlaneObject_setNormal(_UnderlyingPtr, normal._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// updates xf to fit given center
        /// Generated from method `MR::PlaneObject::setCenter`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetCenter(MR.Const_Vector3f center, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setCenter", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setCenter(_Underlying *_this, MR.Const_Vector3f._Underlying *center, MR.ViewportId *id);
            __MR_PlaneObject_setCenter(_UnderlyingPtr, center._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// updates xf to scale size
        /// Generated from method `MR::PlaneObject::setSize`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSize(float size, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSize", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSize(_Underlying *_this, float size, MR.ViewportId *id);
            __MR_PlaneObject_setSize(_UnderlyingPtr, size, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::setSizeX`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSizeX(float size, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSizeX", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSizeX(_Underlying *_this, float size, MR.ViewportId *id);
            __MR_PlaneObject_setSizeX(_UnderlyingPtr, size, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::setSizeY`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSizeY(float size, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSizeY", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSizeY(_Underlying *_this, float size, MR.ViewportId *id);
            __MR_PlaneObject_setSizeY(_UnderlyingPtr, size, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setXf", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_PlaneObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_resetXf", ExactSpelling = true)]
            extern static void __MR_PlaneObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_PlaneObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::PlaneObject::setDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetDecorationsColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setDecorationsColor", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setDecorationsColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_PlaneObject_setDecorationsColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// Generated from method `MR::PlaneObject::setDecorationsColorForAllViewports`.
        public unsafe void SetDecorationsColorForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setDecorationsColorForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte selected);
            __MR_PlaneObject_setDecorationsColorForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::PlaneObject::setPointSize`.
        public unsafe void SetPointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setPointSize", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setPointSize(_Underlying *_this, float pointSize);
            __MR_PlaneObject_setPointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::PlaneObject::setLineWidth`.
        public unsafe void SetLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setLineWidth", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setLineWidth(_Underlying *_this, float lineWidth);
            __MR_PlaneObject_setLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::PlaneObject::setSubfeaturePointSize`.
        public unsafe void SetSubfeaturePointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSubfeaturePointSize", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSubfeaturePointSize(_Underlying *_this, float pointSize);
            __MR_PlaneObject_setSubfeaturePointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::PlaneObject::setSubfeatureLineWidth`.
        public unsafe void SetSubfeatureLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSubfeatureLineWidth", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSubfeatureLineWidth(_Underlying *_this, float lineWidth);
            __MR_PlaneObject_setSubfeatureLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::PlaneObject::setMainFeatureAlpha`.
        public unsafe void SetMainFeatureAlpha(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setMainFeatureAlpha", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setMainFeatureAlpha(_Underlying *_this, float alpha);
            __MR_PlaneObject_setMainFeatureAlpha(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::PlaneObject::setSubfeatureAlphaPoints`.
        public unsafe void SetSubfeatureAlphaPoints(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSubfeatureAlphaPoints(_Underlying *_this, float alpha);
            __MR_PlaneObject_setSubfeatureAlphaPoints(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::PlaneObject::setSubfeatureAlphaLines`.
        public unsafe void SetSubfeatureAlphaLines(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSubfeatureAlphaLines", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSubfeatureAlphaLines(_Underlying *_this, float alpha);
            __MR_PlaneObject_setSubfeatureAlphaLines(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::PlaneObject::setSubfeatureAlphaMesh`.
        public unsafe void SetSubfeatureAlphaMesh(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSubfeatureAlphaMesh(_Underlying *_this, float alpha);
            __MR_PlaneObject_setSubfeatureAlphaMesh(_UnderlyingPtr, alpha);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::PlaneObject::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::PlaneObject::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::PlaneObject::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_PlaneObject_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::PlaneObject::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_PlaneObject_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::PlaneObject::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_PlaneObject_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_PlaneObject_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::PlaneObject::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::PlaneObject::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setFrontColor", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_PlaneObject_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::PlaneObject::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PlaneObject_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::PlaneObject::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_PlaneObject_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::PlaneObject::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setBackColor", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_PlaneObject_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::PlaneObject::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_PlaneObject_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::PlaneObject::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_PlaneObject_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::PlaneObject::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PlaneObject_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::PlaneObject::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setPickable", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::PlaneObject::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setColoringType", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_PlaneObject_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::PlaneObject::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setShininess", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setShininess(_Underlying *_this, float shininess);
            __MR_PlaneObject_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::PlaneObject::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_PlaneObject_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::PlaneObject::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_PlaneObject_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::PlaneObject::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_PlaneObject_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::PlaneObject::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_PlaneObject_resetFrontColor(_Underlying *_this);
            __MR_PlaneObject_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::PlaneObject::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_resetColors", ExactSpelling = true)]
            extern static void __MR_PlaneObject_resetColors(_Underlying *_this);
            __MR_PlaneObject_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setName", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_PlaneObject_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::PlaneObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_PlaneObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PlaneObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_PlaneObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::PlaneObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_applyScale", ExactSpelling = true)]
            extern static void __MR_PlaneObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_PlaneObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::PlaneObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PlaneObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setLocked", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setLocked(_Underlying *_this, byte on);
            __MR_PlaneObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::PlaneObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_PlaneObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::PlaneObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_detachFromParent(_Underlying *_this);
            return __MR_PlaneObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::PlaneObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_addChild", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_PlaneObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::PlaneObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_PlaneObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::PlaneObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_PlaneObject_removeAllChildren(_Underlying *_this);
            __MR_PlaneObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::PlaneObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_PlaneObject_sortChildren(_Underlying *_this);
            __MR_PlaneObject_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::PlaneObject::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_select", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_select(_Underlying *_this, byte on);
            return __MR_PlaneObject_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::PlaneObject::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setAncillary(_Underlying *_this, byte ancillary);
            __MR_PlaneObject_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::PlaneObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setVisible", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::PlaneObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_PlaneObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_PlaneObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::PlaneObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_swap", ExactSpelling = true)]
            extern static void __MR_PlaneObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_PlaneObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::PlaneObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_addTag", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_PlaneObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::PlaneObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_PlaneObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_PlaneObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `PlaneObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PlaneObject`/`Const_PlaneObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PlaneObject
    {
        internal readonly Const_PlaneObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PlaneObject() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PlaneObject(MR.Misc._Moved<PlaneObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PlaneObject(MR.Misc._Moved<PlaneObject> arg) {return new(arg);}

        /// Finds best plane to approx given points
        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public static unsafe implicit operator _ByValue_PlaneObject(MR.Std.Const_Vector_MRVector3f pointsToApprox) {return new MR.PlaneObject(pointsToApprox);}
    }

    /// This is used for optional parameters of class `PlaneObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PlaneObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PlaneObject`/`Const_PlaneObject` directly.
    public class _InOptMut_PlaneObject
    {
        public PlaneObject? Opt;

        public _InOptMut_PlaneObject() {}
        public _InOptMut_PlaneObject(PlaneObject value) {Opt = value;}
        public static implicit operator _InOptMut_PlaneObject(PlaneObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `PlaneObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PlaneObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PlaneObject`/`Const_PlaneObject` to pass it to the function.
    public class _InOptConst_PlaneObject
    {
        public Const_PlaneObject? Opt;

        public _InOptConst_PlaneObject() {}
        public _InOptConst_PlaneObject(Const_PlaneObject value) {Opt = value;}
        public static implicit operator _InOptConst_PlaneObject(Const_PlaneObject value) {return new(value);}

        /// Finds best plane to approx given points
        /// Generated from constructor `MR::PlaneObject::PlaneObject`.
        public static unsafe implicit operator _InOptConst_PlaneObject(MR.Std.Const_Vector_MRVector3f pointsToApprox) {return new MR.PlaneObject(pointsToApprox);}
    }
}
