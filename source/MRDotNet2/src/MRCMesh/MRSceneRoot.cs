public static partial class MR
{
    /// Object that is parent of all scene
    /// Generated from class `MR::SceneRootObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Object`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    /// This is the const half of the class.
    public class Const_SceneRootObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_SceneRootObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_SceneRootObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_SceneRootObject_UseCount();
                return __MR_std_shared_ptr_MR_SceneRootObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_SceneRootObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_SceneRootObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_SceneRootObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_SceneRootObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_SceneRootObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_SceneRootObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_SceneRootObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_SceneRootObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe SceneRootObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_SceneRootObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_SceneRootObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_SceneRootObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_SceneRootObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_SceneRootObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_SceneRootObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_SceneRootObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SceneRootObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_SceneRootObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_SceneRootObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_SceneRootObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_SceneRootObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_SceneRootObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_SceneRootObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_SceneRootObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_SceneRootObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_SceneRootObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_SceneRootObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SceneRootObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SceneRootObject._Underlying *__MR_SceneRootObject_DefaultConstruct();
            _LateMakeShared(__MR_SceneRootObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::SceneRootObject::SceneRootObject`.
        public unsafe Const_SceneRootObject(MR._ByValue_SceneRootObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SceneRootObject._Underlying *__MR_SceneRootObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._Underlying *_other);
            _LateMakeShared(__MR_SceneRootObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::SceneRootObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_SceneRootObject_StaticTypeName();
            var __ret = __MR_SceneRootObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::SceneRootObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_SceneRootObject_typeName(_Underlying *_this);
            var __ret = __MR_SceneRootObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::SceneRootObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_SceneRootObject_StaticClassName();
            var __ret = __MR_SceneRootObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::SceneRootObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_SceneRootObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_SceneRootObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::SceneRootObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_SceneRootObject_StaticClassNameInPlural();
            var __ret = __MR_SceneRootObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::SceneRootObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_SceneRootObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_SceneRootObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::SceneRootObject::RootName`.
        public static unsafe byte? RootName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_RootName", ExactSpelling = true)]
            extern static byte *__MR_SceneRootObject_RootName();
            var __ret = __MR_SceneRootObject_RootName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::SceneRootObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_SceneRootObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_SceneRootObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// same as clone but returns correct type
        /// Generated from method `MR::SceneRootObject::cloneRoot`.
        public unsafe MR.Misc._Moved<MR.SceneRootObject> CloneRoot()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_cloneRoot", ExactSpelling = true)]
            extern static MR.SceneRootObject._UnderlyingShared *__MR_SceneRootObject_cloneRoot(_Underlying *_this);
            return MR.Misc.Move(new MR.SceneRootObject(__MR_SceneRootObject_cloneRoot(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::SceneRootObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_SceneRootObject_name(_Underlying *_this);
            return new(__MR_SceneRootObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::SceneRootObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_SceneRootObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_SceneRootObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::SceneRootObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_SceneRootObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_SceneRootObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::SceneRootObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_SceneRootObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_SceneRootObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::SceneRootObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_SceneRootObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_SceneRootObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::SceneRootObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_SceneRootObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::SceneRootObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isLocked(_Underlying *_this);
            return __MR_SceneRootObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::SceneRootObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isParentLocked(_Underlying *_this);
            return __MR_SceneRootObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::SceneRootObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_SceneRootObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::SceneRootObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isSelected(_Underlying *_this);
            return __MR_SceneRootObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::SceneRootObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isAncillary(_Underlying *_this);
            return __MR_SceneRootObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::SceneRootObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isGlobalAncillary(_Underlying *_this);
            return __MR_SceneRootObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::SceneRootObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_SceneRootObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::SceneRootObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_SceneRootObject_visibilityMask(_Underlying *_this);
            return new(__MR_SceneRootObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// this method virtual because others data model types could have dirty flags or something
        /// Generated from method `MR::SceneRootObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *_1);
            return __MR_SceneRootObject_getRedrawFlag(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::SceneRootObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_resetRedrawFlag(_Underlying *_this);
            __MR_SceneRootObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::SceneRootObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_SceneRootObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_SceneRootObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::SceneRootObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_SceneRootObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_SceneRootObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones current object only, without parent and/or children
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::SceneRootObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_SceneRootObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_SceneRootObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// return several info lines that can better describe object in the UI
        /// Generated from method `MR::SceneRootObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_SceneRootObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_SceneRootObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// returns bounding box of this object in world coordinates for default or specific viewport
        /// Generated from method `MR::SceneRootObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_SceneRootObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_SceneRootObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::SceneRootObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_SceneRootObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_SceneRootObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::SceneRootObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_SceneRootObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::SceneRootObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_hasModel(_Underlying *_this);
            return __MR_SceneRootObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::SceneRootObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_SceneRootObject_tags(_Underlying *_this);
            return new(__MR_SceneRootObject_tags(_UnderlyingPtr), is_owning: false);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::SceneRootObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_SceneRootObject_heapBytes(_Underlying *_this);
            return __MR_SceneRootObject_heapBytes(_UnderlyingPtr);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::SceneRootObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_SceneRootObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::SceneRootObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_SceneRootObject_getModelHash(_Underlying *_this);
            return __MR_SceneRootObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::SceneRootObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_SceneRootObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_SceneRootObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// Object that is parent of all scene
    /// Generated from class `MR::SceneRootObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Object`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    /// This is the non-const half of the class.
    public class SceneRootObject : Const_SceneRootObject
    {
        internal unsafe SceneRootObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe SceneRootObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(SceneRootObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_SceneRootObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_SceneRootObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(SceneRootObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_SceneRootObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_SceneRootObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator SceneRootObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_SceneRootObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_SceneRootObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_SceneRootObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SceneRootObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SceneRootObject._Underlying *__MR_SceneRootObject_DefaultConstruct();
            _LateMakeShared(__MR_SceneRootObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::SceneRootObject::SceneRootObject`.
        public unsafe SceneRootObject(MR._ByValue_SceneRootObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SceneRootObject._Underlying *__MR_SceneRootObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._Underlying *_other);
            _LateMakeShared(__MR_SceneRootObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::SceneRootObject::operator=`.
        public unsafe MR.SceneRootObject Assign(MR._ByValue_SceneRootObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SceneRootObject._Underlying *__MR_SceneRootObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SceneRootObject._Underlying *_other);
            return new(__MR_SceneRootObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::SceneRootObject::setAncillary`.
        public unsafe void SetAncillary(bool _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setAncillary(_Underlying *_this, byte _1);
            __MR_SceneRootObject_setAncillary(_UnderlyingPtr, _1 ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::SceneRootObject::select`.
        public unsafe bool Select(bool _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_select", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_select(_Underlying *_this, byte _1);
            return __MR_SceneRootObject_select(_UnderlyingPtr, _1 ? (byte)1 : (byte)0) != 0;
        }

        /// Generated from method `MR::SceneRootObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setName", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setName(_Underlying *_this, byte *_1, byte *_1_end);
            byte[] __bytes__1 = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(_1.Length)];
            int __len__1 = System.Text.Encoding.UTF8.GetBytes(_1, __bytes__1);
            fixed (byte *__ptr__1 = __bytes__1)
            {
                __MR_SceneRootObject_setName(_UnderlyingPtr, __ptr__1, __ptr__1 + __len__1);
            }
        }

        /// Generated from method `MR::SceneRootObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setXf", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_SceneRootObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::SceneRootObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_resetXf", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_SceneRootObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::SceneRootObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_SceneRootObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SceneRootObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_SceneRootObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::SceneRootObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_applyScale", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_SceneRootObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::SceneRootObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_SceneRootObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SceneRootObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setLocked", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setLocked(_Underlying *_this, byte on);
            __MR_SceneRootObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::SceneRootObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_SceneRootObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::SceneRootObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_detachFromParent(_Underlying *_this);
            return __MR_SceneRootObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::SceneRootObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_addChild", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_SceneRootObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::SceneRootObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_SceneRootObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::SceneRootObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_removeAllChildren(_Underlying *_this);
            __MR_SceneRootObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::SceneRootObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_sortChildren(_Underlying *_this);
            __MR_SceneRootObject_sortChildren(_UnderlyingPtr);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::SceneRootObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setVisible", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_SceneRootObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::SceneRootObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_SceneRootObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::SceneRootObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_swap", ExactSpelling = true)]
            extern static void __MR_SceneRootObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_SceneRootObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::SceneRootObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_addTag", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_SceneRootObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::SceneRootObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRootObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_SceneRootObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_SceneRootObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `SceneRootObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SceneRootObject`/`Const_SceneRootObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SceneRootObject
    {
        internal readonly Const_SceneRootObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SceneRootObject() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SceneRootObject(MR.Misc._Moved<SceneRootObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SceneRootObject(MR.Misc._Moved<SceneRootObject> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SceneRootObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SceneRootObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneRootObject`/`Const_SceneRootObject` directly.
    public class _InOptMut_SceneRootObject
    {
        public SceneRootObject? Opt;

        public _InOptMut_SceneRootObject() {}
        public _InOptMut_SceneRootObject(SceneRootObject value) {Opt = value;}
        public static implicit operator _InOptMut_SceneRootObject(SceneRootObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `SceneRootObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SceneRootObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneRootObject`/`Const_SceneRootObject` to pass it to the function.
    public class _InOptConst_SceneRootObject
    {
        public Const_SceneRootObject? Opt;

        public _InOptConst_SceneRootObject() {}
        public _InOptConst_SceneRootObject(Const_SceneRootObject value) {Opt = value;}
        public static implicit operator _InOptConst_SceneRootObject(Const_SceneRootObject value) {return new(value);}
    }

    /// Singleton to store scene root object
    /// Generated from class `MR::SceneRoot`.
    /// This is the const half of the class.
    public class Const_SceneRoot : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SceneRoot(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_Destroy", ExactSpelling = true)]
            extern static void __MR_SceneRoot_Destroy(_Underlying *_this);
            __MR_SceneRoot_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SceneRoot() {Dispose(false);}

        /// Generated from constructor `MR::SceneRoot::SceneRoot`.
        public unsafe Const_SceneRoot(MR._ByValue_SceneRoot _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SceneRoot._Underlying *__MR_SceneRoot_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneRoot._Underlying *_other);
            _UnderlyingPtr = __MR_SceneRoot_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SceneRoot::get`.
        public static unsafe MR.SceneRootObject Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_get", ExactSpelling = true)]
            extern static MR.SceneRootObject._Underlying *__MR_SceneRoot_get();
            return new(__MR_SceneRoot_get(), is_owning: false);
        }

        /// Generated from method `MR::SceneRoot::getSharedPtr`.
        public static unsafe MR.SceneRootObject GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_getSharedPtr", ExactSpelling = true)]
            extern static MR.SceneRootObject._UnderlyingShared *__MR_SceneRoot_getSharedPtr();
            return new(__MR_SceneRoot_getSharedPtr(), is_owning: false);
        }

        /// Generated from method `MR::SceneRoot::setScenePath`.
        public static unsafe void SetScenePath(ReadOnlySpan<char> scenePath)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_setScenePath", ExactSpelling = true)]
            extern static void __MR_SceneRoot_setScenePath(byte *scenePath, byte *scenePath_end);
            byte[] __bytes_scenePath = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(scenePath.Length)];
            int __len_scenePath = System.Text.Encoding.UTF8.GetBytes(scenePath, __bytes_scenePath);
            fixed (byte *__ptr_scenePath = __bytes_scenePath)
            {
                __MR_SceneRoot_setScenePath(__ptr_scenePath, __ptr_scenePath + __len_scenePath);
            }
        }

        /// Generated from method `MR::SceneRoot::constGet`.
        public static unsafe MR.Const_SceneRootObject ConstGet()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_constGet", ExactSpelling = true)]
            extern static MR.Const_SceneRootObject._Underlying *__MR_SceneRoot_constGet();
            return new(__MR_SceneRoot_constGet(), is_owning: false);
        }

        /// Generated from method `MR::SceneRoot::constGetSharedPtr`.
        public static unsafe MR.Misc._Moved<MR.SceneRootObject> ConstGetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_constGetSharedPtr", ExactSpelling = true)]
            extern static MR.SceneRootObject._UnderlyingShared *__MR_SceneRoot_constGetSharedPtr();
            return MR.Misc.Move(new MR.SceneRootObject(__MR_SceneRoot_constGetSharedPtr(), is_owning: true));
        }

        /// Generated from method `MR::SceneRoot::getScenePath`.
        public static unsafe MR.Std.Filesystem.Const_Path GetScenePath()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_getScenePath", ExactSpelling = true)]
            extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_SceneRoot_getScenePath();
            return new(__MR_SceneRoot_getScenePath(), is_owning: false);
        }
    }

    /// Singleton to store scene root object
    /// Generated from class `MR::SceneRoot`.
    /// This is the non-const half of the class.
    public class SceneRoot : Const_SceneRoot
    {
        internal unsafe SceneRoot(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::SceneRoot::SceneRoot`.
        public unsafe SceneRoot(MR._ByValue_SceneRoot _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SceneRoot._Underlying *__MR_SceneRoot_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneRoot._Underlying *_other);
            _UnderlyingPtr = __MR_SceneRoot_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SceneRoot::operator=`.
        public unsafe MR.SceneRoot Assign(MR._ByValue_SceneRoot _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneRoot_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SceneRoot._Underlying *__MR_SceneRoot_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SceneRoot._Underlying *_other);
            return new(__MR_SceneRoot_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SceneRoot` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SceneRoot`/`Const_SceneRoot` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SceneRoot
    {
        internal readonly Const_SceneRoot? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SceneRoot(Const_SceneRoot new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SceneRoot(Const_SceneRoot arg) {return new(arg);}
        public _ByValue_SceneRoot(MR.Misc._Moved<SceneRoot> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SceneRoot(MR.Misc._Moved<SceneRoot> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SceneRoot` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SceneRoot`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneRoot`/`Const_SceneRoot` directly.
    public class _InOptMut_SceneRoot
    {
        public SceneRoot? Opt;

        public _InOptMut_SceneRoot() {}
        public _InOptMut_SceneRoot(SceneRoot value) {Opt = value;}
        public static implicit operator _InOptMut_SceneRoot(SceneRoot value) {return new(value);}
    }

    /// This is used for optional parameters of class `SceneRoot` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SceneRoot`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneRoot`/`Const_SceneRoot` to pass it to the function.
    public class _InOptConst_SceneRoot
    {
        public Const_SceneRoot? Opt;

        public _InOptConst_SceneRoot() {}
        public _InOptConst_SceneRoot(Const_SceneRoot value) {Opt = value;}
        public static implicit operator _InOptConst_SceneRoot(Const_SceneRoot value) {return new(value);}
    }

    /// Removes all `obj` children and attaches it to newly created `SceneRootObject`
    /// note that it does not respect `obj` xf
    /// Generated from function `MR::createRootFormObject`.
    public static unsafe MR.Misc._Moved<MR.SceneRootObject> CreateRootFormObject(MR._ByValue_Object obj)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_createRootFormObject", ExactSpelling = true)]
        extern static MR.SceneRootObject._UnderlyingShared *__MR_createRootFormObject(MR.Misc._PassBy obj_pass_by, MR.Object._UnderlyingShared *obj);
        return MR.Misc.Move(new MR.SceneRootObject(__MR_createRootFormObject(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null), is_owning: true));
    }
}
