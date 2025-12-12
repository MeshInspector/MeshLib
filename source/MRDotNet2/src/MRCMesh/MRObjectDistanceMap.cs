public static partial class MR
{
    /// This class stores information about distance map object
    /// Generated from class `MR::ObjectDistanceMap`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectMeshHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_ObjectDistanceMap : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectDistanceMap_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectDistanceMap_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectDistanceMap_UseCount();
                return __MR_std_shared_ptr_MR_ObjectDistanceMap_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectDistanceMap(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectDistanceMap_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectDistanceMap_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectDistanceMap(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectDistanceMap _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectDistanceMap_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectDistanceMap_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectDistanceMap_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectDistanceMap_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectDistanceMap_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectDistanceMap_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectDistanceMap() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_ObjectMeshHolder(Const_ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectMeshHolder._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_ObjectMeshHolder(_Underlying *_this);
            return MR.Const_ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_ObjectMeshHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ObjectDistanceMap?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectDistanceMap", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectDistanceMap(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectDistanceMap(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectDistanceMap?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectDistanceMap", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectDistanceMap(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectDistanceMap(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectDistanceMap?(MR.Const_ObjectMeshHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectDistanceMap", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectDistanceMap(MR.Const_ObjectMeshHolder._Underlying *_this);
            var ptr = __MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectDistanceMap(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectDistanceMap() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectDistanceMap._Underlying *__MR_ObjectDistanceMap_DefaultConstruct();
            _LateMakeShared(__MR_ObjectDistanceMap_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectDistanceMap::ObjectDistanceMap`.
        public unsafe Const_ObjectDistanceMap(MR._ByValue_ObjectDistanceMap _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectDistanceMap._Underlying *__MR_ObjectDistanceMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectDistanceMap._Underlying *_other);
            _LateMakeShared(__MR_ObjectDistanceMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectDistanceMap::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_StaticTypeName();
            var __ret = __MR_ObjectDistanceMap_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectDistanceMap::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_typeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_typeName(_Underlying *_this);
            var __ret = __MR_ObjectDistanceMap_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectDistanceMap::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_StaticClassName();
            var __ret = __MR_ObjectDistanceMap_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectDistanceMap::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectDistanceMap_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectDistanceMap_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectDistanceMap::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_StaticClassNameInPlural();
            var __ret = __MR_ObjectDistanceMap_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectDistanceMap::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectDistanceMap_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectDistanceMap_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectDistanceMap::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectDistanceMap_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectDistanceMap_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectDistanceMap::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectDistanceMap_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectDistanceMap_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectDistanceMap::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_ObjectDistanceMap_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_ObjectDistanceMap_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// creates a grid for this object
        /// Generated from method `MR::ObjectDistanceMap::calculateMesh`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Mesh> CalculateMesh(MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_calculateMesh", ExactSpelling = true)]
            extern static MR.Mesh._UnderlyingShared *__MR_ObjectDistanceMap_calculateMesh(_Underlying *_this, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Mesh(__MR_ObjectDistanceMap_calculateMesh(_UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from method `MR::ObjectDistanceMap::getDistanceMap`.
        public unsafe MR.Const_DistanceMap GetDistanceMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getDistanceMap", ExactSpelling = true)]
            extern static MR.Const_DistanceMap._UnderlyingShared *__MR_ObjectDistanceMap_getDistanceMap(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getDistanceMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_hasModel", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_hasModel(_Underlying *_this);
            return __MR_ObjectDistanceMap_hasModel(_UnderlyingPtr) != 0;
        }

        /// unlike the name, actually it is the transformation from distance map in local space
        /// Generated from method `MR::ObjectDistanceMap::getToWorldParameters`.
        public unsafe MR.Const_AffineXf3f GetToWorldParameters()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getToWorldParameters", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectDistanceMap_getToWorldParameters(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getToWorldParameters(_UnderlyingPtr), is_owning: false);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectDistanceMap::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_heapBytes(_Underlying *_this);
            return __MR_ObjectDistanceMap_heapBytes(_UnderlyingPtr);
        }

        /// mesh object can be seen if the mesh has at least one edge
        /// Generated from method `MR::ObjectDistanceMap::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_hasVisualRepresentation(_Underlying *_this);
            return __MR_ObjectDistanceMap_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectDistanceMap::mesh`.
        public unsafe MR.Const_Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_mesh", ExactSpelling = true)]
            extern static MR.Const_Mesh._UnderlyingShared *__MR_ObjectDistanceMap_mesh(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// \return the pair ( mesh, selected triangles ) if any triangle is selected or whole mesh otherwise
        /// Generated from method `MR::ObjectDistanceMap::meshPart`.
        public unsafe MR.MeshPart MeshPart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_meshPart", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_ObjectDistanceMap_meshPart(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_meshPart(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectDistanceMap::getSelectedFaces`.
        public unsafe MR.Const_FaceBitSet GetSelectedFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSelectedFaces", ExactSpelling = true)]
            extern static MR.Const_FaceBitSet._Underlying *__MR_ObjectDistanceMap_getSelectedFaces(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getSelectedFaces(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected triangles
        /// Generated from method `MR::ObjectDistanceMap::getSelectedFacesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedFacesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSelectedFacesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getSelectedFacesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectDistanceMap_getSelectedFacesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getSelectedEdges`.
        public unsafe MR.Const_UndirectedEdgeBitSet GetSelectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSelectedEdges", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectDistanceMap_getSelectedEdges(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getSelectedEdges(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected edges
        /// Generated from method `MR::ObjectDistanceMap::getSelectedEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedEdgesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSelectedEdgesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getSelectedEdgesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectDistanceMap_getSelectedEdgesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getSelectedEdgesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedEdgesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSelectedEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getSelectedEdgesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getSelectedEdgesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getSelectedFacesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedFacesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSelectedFacesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getSelectedFacesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getSelectedFacesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getEdgesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetEdgesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getEdgesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getEdgesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getPointsColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetPointsColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getPointsColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getPointsColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getPointsColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getBordersColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBordersColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getBordersColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getBordersColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getBordersColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Edges on mesh, that will have sharp visualization even with smooth shading
        /// Generated from method `MR::ObjectDistanceMap::creases`.
        public unsafe MR.Const_UndirectedEdgeBitSet Creases()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_creases", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectDistanceMap_creases(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_creases(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::flatShading`.
        public unsafe bool FlatShading()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_flatShading", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_flatShading(_Underlying *_this);
            return __MR_ObjectDistanceMap_flatShading(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectDistanceMap::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_ObjectDistanceMap_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// get all visualize properties masks
        /// Generated from method `MR::ObjectDistanceMap::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_ObjectDistanceMap_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_ObjectDistanceMap_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::ObjectDistanceMap::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_ObjectDistanceMap_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_ObjectDistanceMap_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// provides read-only access to whole ObjectMeshData
        /// Generated from method `MR::ObjectDistanceMap::data`.
        public unsafe MR.Const_ObjectMeshData Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_data", ExactSpelling = true)]
            extern static MR.Const_ObjectMeshData._Underlying *__MR_ObjectDistanceMap_data(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_data(_UnderlyingPtr), is_owning: false);
        }

        /// returns per-vertex colors of the object
        /// Generated from method `MR::ObjectDistanceMap::getVertsColorMap`.
        public unsafe MR.Const_VertColors GetVertsColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getVertsColorMap", ExactSpelling = true)]
            extern static MR.Const_VertColors._Underlying *__MR_ObjectDistanceMap_getVertsColorMap(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getVertsColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getFacesColorMap`.
        public unsafe MR.Const_FaceColors GetFacesColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getFacesColorMap", ExactSpelling = true)]
            extern static MR.Const_FaceColors._Underlying *__MR_ObjectDistanceMap_getFacesColorMap(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getFacesColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getEdgeWidth`.
        public unsafe float GetEdgeWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getEdgeWidth", ExactSpelling = true)]
            extern static float __MR_ObjectDistanceMap_getEdgeWidth(_Underlying *_this);
            return __MR_ObjectDistanceMap_getEdgeWidth(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getPointSize", ExactSpelling = true)]
            extern static float __MR_ObjectDistanceMap_getPointSize(_Underlying *_this);
            return __MR_ObjectDistanceMap_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::getEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetEdgesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getEdgesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getEdgesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectDistanceMap_getEdgesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getPointsColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetPointsColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getPointsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getPointsColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectDistanceMap_getPointsColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getBordersColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetBordersColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getBordersColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getBordersColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectDistanceMap_getBordersColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// returns first texture in the vector. If there is no textures, returns empty texture
        /// Generated from method `MR::ObjectDistanceMap::getTexture`.
        public unsafe MR.Const_MeshTexture GetTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getTexture", ExactSpelling = true)]
            extern static MR.Const_MeshTexture._Underlying *__MR_ObjectDistanceMap_getTexture(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getTexture(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getTextures`.
        public unsafe MR.Const_Vector_MRMeshTexture_MRTextureId GetTextures()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getTextures", ExactSpelling = true)]
            extern static MR.Const_Vector_MRMeshTexture_MRTextureId._Underlying *__MR_ObjectDistanceMap_getTextures(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getTextures(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getTexturePerFace`.
        public unsafe MR.Const_TexturePerFace GetTexturePerFace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getTexturePerFace", ExactSpelling = true)]
            extern static MR.Const_TexturePerFace._Underlying *__MR_ObjectDistanceMap_getTexturePerFace(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getTexturePerFace(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getUVCoords`.
        public unsafe MR.Const_VertCoords2 GetUVCoords()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getUVCoords", ExactSpelling = true)]
            extern static MR.Const_VertCoords2._Underlying *__MR_ObjectDistanceMap_getUVCoords(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getUVCoords(_UnderlyingPtr), is_owning: false);
        }

        // ancillary texture can be used to have custom features visualization without affecting real one
        /// Generated from method `MR::ObjectDistanceMap::getAncillaryTexture`.
        public unsafe MR.Const_MeshTexture GetAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getAncillaryTexture", ExactSpelling = true)]
            extern static MR.Const_MeshTexture._Underlying *__MR_ObjectDistanceMap_getAncillaryTexture(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getAncillaryTexture(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::getAncillaryUVCoords`.
        public unsafe MR.Const_VertCoords2 GetAncillaryUVCoords()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getAncillaryUVCoords", ExactSpelling = true)]
            extern static MR.Const_VertCoords2._Underlying *__MR_ObjectDistanceMap_getAncillaryUVCoords(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getAncillaryUVCoords(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::hasAncillaryTexture`.
        public unsafe bool HasAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_hasAncillaryTexture", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_hasAncillaryTexture(_Underlying *_this);
            return __MR_ObjectDistanceMap_hasAncillaryTexture(_UnderlyingPtr) != 0;
        }

        /// returns dirty flag of currently using normal type if they are dirty in render representation
        /// Generated from method `MR::ObjectDistanceMap::getNeededNormalsRenderDirtyValue`.
        public unsafe uint GetNeededNormalsRenderDirtyValue(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getNeededNormalsRenderDirtyValue", ExactSpelling = true)]
            extern static uint __MR_ObjectDistanceMap_getNeededNormalsRenderDirtyValue(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_getNeededNormalsRenderDirtyValue(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns cached information whether the mesh is closed
        /// Generated from method `MR::ObjectDistanceMap::isMeshClosed`.
        public unsafe bool IsMeshClosed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isMeshClosed", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isMeshClosed(_Underlying *_this);
            return __MR_ObjectDistanceMap_isMeshClosed(_UnderlyingPtr) != 0;
        }

        /// returns cached bounding box of this mesh object in world coordinates;
        /// if you need bounding box in local coordinates please call getBoundingBox()
        /// Generated from method `MR::ObjectDistanceMap::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectDistanceMap_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectDistanceMap_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns cached information about the number of selected faces in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numSelectedFaces`.
        public unsafe ulong NumSelectedFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numSelectedFaces", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numSelectedFaces(_Underlying *_this);
            return __MR_ObjectDistanceMap_numSelectedFaces(_UnderlyingPtr);
        }

        /// returns cached information about the number of selected undirected edges in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numSelectedEdges`.
        public unsafe ulong NumSelectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numSelectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numSelectedEdges(_Underlying *_this);
            return __MR_ObjectDistanceMap_numSelectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of crease undirected edges in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numCreaseEdges`.
        public unsafe ulong NumCreaseEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numCreaseEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numCreaseEdges(_Underlying *_this);
            return __MR_ObjectDistanceMap_numCreaseEdges(_UnderlyingPtr);
        }

        /// returns cached summed area of mesh triangles
        /// Generated from method `MR::ObjectDistanceMap::totalArea`.
        public unsafe double TotalArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_totalArea", ExactSpelling = true)]
            extern static double __MR_ObjectDistanceMap_totalArea(_Underlying *_this);
            return __MR_ObjectDistanceMap_totalArea(_UnderlyingPtr);
        }

        /// returns cached area of selected triangles
        /// Generated from method `MR::ObjectDistanceMap::selectedArea`.
        public unsafe double SelectedArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_selectedArea", ExactSpelling = true)]
            extern static double __MR_ObjectDistanceMap_selectedArea(_Underlying *_this);
            return __MR_ObjectDistanceMap_selectedArea(_UnderlyingPtr);
        }

        /// returns cached volume of space surrounded by the mesh, which is valid only if mesh is closed
        /// Generated from method `MR::ObjectDistanceMap::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_volume", ExactSpelling = true)]
            extern static double __MR_ObjectDistanceMap_volume(_Underlying *_this);
            return __MR_ObjectDistanceMap_volume(_UnderlyingPtr);
        }

        /// returns cached average edge length
        /// Generated from method `MR::ObjectDistanceMap::avgEdgeLen`.
        public unsafe float AvgEdgeLen()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_avgEdgeLen", ExactSpelling = true)]
            extern static float __MR_ObjectDistanceMap_avgEdgeLen(_Underlying *_this);
            return __MR_ObjectDistanceMap_avgEdgeLen(_UnderlyingPtr);
        }

        /// returns cached information about the number of undirected edges in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numUndirectedEdges`.
        public unsafe ulong NumUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numUndirectedEdges(_Underlying *_this);
            return __MR_ObjectDistanceMap_numUndirectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of holes in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numHoles`.
        public unsafe ulong NumHoles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numHoles", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numHoles(_Underlying *_this);
            return __MR_ObjectDistanceMap_numHoles(_UnderlyingPtr);
        }

        /// returns cached information about the number of components in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numComponents`.
        public unsafe ulong NumComponents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numComponents", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numComponents(_Underlying *_this);
            return __MR_ObjectDistanceMap_numComponents(_UnderlyingPtr);
        }

        /// returns cached information about the number of handles in the mesh
        /// Generated from method `MR::ObjectDistanceMap::numHandles`.
        public unsafe ulong NumHandles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_numHandles", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_numHandles(_Underlying *_this);
            return __MR_ObjectDistanceMap_numHandles(_UnderlyingPtr);
        }

        /// returns overriden file extension used to serialize mesh inside this object, nullptr means defaultSerializeMeshFormat()
        /// Generated from method `MR::ObjectDistanceMap::serializeFormat`.
        public unsafe byte? SerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_serializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_serializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectDistanceMap_serializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// returns overriden file extension used to serialize mesh inside this object if set, or defaultSerializeMeshFormat().c_str() otherwise; never returns nullptr
        /// Generated from method `MR::ObjectDistanceMap::actualSerializeFormat`.
        public unsafe byte? ActualSerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_actualSerializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_actualSerializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectDistanceMap_actualSerializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectDistanceMap::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_ObjectDistanceMap_getModelHash(_Underlying *_this);
            return __MR_ObjectDistanceMap_getModelHash(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_sameModels", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_ObjectDistanceMap_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::ObjectDistanceMap::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::ObjectDistanceMap::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectDistanceMap_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::ObjectDistanceMap::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectDistanceMap::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectDistanceMap_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectDistanceMap::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectDistanceMap_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::ObjectDistanceMap::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectDistanceMap_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::ObjectDistanceMap::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectDistanceMap_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_ObjectDistanceMap_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectDistanceMap::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_ObjectDistanceMap_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_ObjectDistanceMap_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectDistanceMap::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_ObjectDistanceMap_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::ObjectDistanceMap::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_ObjectDistanceMap_getDirtyFlags(_Underlying *_this);
            return __MR_ObjectDistanceMap_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::ObjectDistanceMap::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_resetDirty", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_resetDirty(_Underlying *_this);
            __MR_ObjectDistanceMap_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::ObjectDistanceMap::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_ObjectDistanceMap_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::ObjectDistanceMap::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectDistanceMap_getBoundingBox(_Underlying *_this);
            return __MR_ObjectDistanceMap_getBoundingBox(_UnderlyingPtr);
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::ObjectDistanceMap::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isPickable", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::ObjectDistanceMap::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_ObjectDistanceMap_getColoringType(_Underlying *_this);
            return __MR_ObjectDistanceMap_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::ObjectDistanceMap::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getShininess", ExactSpelling = true)]
            extern static float __MR_ObjectDistanceMap_getShininess(_Underlying *_this);
            return __MR_ObjectDistanceMap_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::ObjectDistanceMap::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_ObjectDistanceMap_getSpecularStrength(_Underlying *_this);
            return __MR_ObjectDistanceMap_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::ObjectDistanceMap::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_ObjectDistanceMap_getAmbientStrength(_Underlying *_this);
            return __MR_ObjectDistanceMap_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::ObjectDistanceMap::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_render", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_ObjectDistanceMap_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::ObjectDistanceMap::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_renderForPicker", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_ObjectDistanceMap_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::ObjectDistanceMap::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_renderUi", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_ObjectDistanceMap_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectDistanceMap::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_ObjectDistanceMap_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectDistanceMap::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_ObjectDistanceMap_name(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::ObjectDistanceMap::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectDistanceMap_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectDistanceMap_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::ObjectDistanceMap::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_ObjectDistanceMap_xfsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::ObjectDistanceMap::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ObjectDistanceMap_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectDistanceMap_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::ObjectDistanceMap::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectDistanceMap_globalVisibilityMask(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::ObjectDistanceMap::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::ObjectDistanceMap::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isLocked(_Underlying *_this);
            return __MR_ObjectDistanceMap_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::ObjectDistanceMap::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isParentLocked(_Underlying *_this);
            return __MR_ObjectDistanceMap_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::ObjectDistanceMap::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isAncestor", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_ObjectDistanceMap_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectDistanceMap::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isSelected", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isSelected(_Underlying *_this);
            return __MR_ObjectDistanceMap_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectDistanceMap::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isAncillary(_Underlying *_this);
            return __MR_ObjectDistanceMap_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::ObjectDistanceMap::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isGlobalAncillary(_Underlying *_this);
            return __MR_ObjectDistanceMap_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::ObjectDistanceMap::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_isVisible", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectDistanceMap_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectDistanceMap::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectDistanceMap_visibilityMask(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectDistanceMap::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_resetRedrawFlag(_Underlying *_this);
            __MR_ObjectDistanceMap_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::ObjectDistanceMap::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectDistanceMap_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectDistanceMap_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::ObjectDistanceMap::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectDistanceMap_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectDistanceMap_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::ObjectDistanceMap::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectDistanceMap_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectDistanceMap_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::ObjectDistanceMap::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_ObjectDistanceMap_tags(_Underlying *_this);
            return new(__MR_ObjectDistanceMap_tags(_UnderlyingPtr), is_owning: false);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectDistanceMap::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectDistanceMap_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectDistanceMap_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// This class stores information about distance map object
    /// Generated from class `MR::ObjectDistanceMap`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectMeshHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class ObjectDistanceMap : Const_ObjectDistanceMap
    {
        internal unsafe ObjectDistanceMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectDistanceMap(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.ObjectMeshHolder(ObjectDistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_UpcastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectDistanceMap_UpcastTo_MR_ObjectMeshHolder(_Underlying *_this);
            return MR.ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectDistanceMap_UpcastTo_MR_ObjectMeshHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ObjectDistanceMap?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectDistanceMap", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectDistanceMap(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectDistanceMap(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectDistanceMap?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectDistanceMap", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectDistanceMap(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectDistanceMap(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectDistanceMap?(MR.ObjectMeshHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectDistanceMap", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectDistanceMap(MR.ObjectMeshHolder._Underlying *_this);
            var ptr = __MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectDistanceMap(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectDistanceMap() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectDistanceMap._Underlying *__MR_ObjectDistanceMap_DefaultConstruct();
            _LateMakeShared(__MR_ObjectDistanceMap_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectDistanceMap::ObjectDistanceMap`.
        public unsafe ObjectDistanceMap(MR._ByValue_ObjectDistanceMap _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectDistanceMap._Underlying *__MR_ObjectDistanceMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectDistanceMap._Underlying *_other);
            _LateMakeShared(__MR_ObjectDistanceMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectDistanceMap::operator=`.
        public unsafe MR.ObjectDistanceMap Assign(MR._ByValue_ObjectDistanceMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectDistanceMap._Underlying *__MR_ObjectDistanceMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectDistanceMap._Underlying *_other);
            return new(__MR_ObjectDistanceMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectDistanceMap::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_applyScale", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_applyScale(_Underlying *_this, float scaleFactor);
            __MR_ObjectDistanceMap_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// rebuilds the mesh;
        /// if it is executed in the rendering stream then you can set the needUpdateMesh = true
        /// otherwise you should set the needUpdateMesh = false and call the function calculateMesh
        /// and after finishing in the rendering stream, call the function updateMesh
        /// Generated from method `MR::ObjectDistanceMap::setDistanceMap`.
        /// Parameter `needUpdateMesh` defaults to `true`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe bool SetDistanceMap(MR.Const_DistanceMap dmap, MR.Const_AffineXf3f dmap2local, bool? needUpdateMesh = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setDistanceMap", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_setDistanceMap(_Underlying *_this, MR.Const_DistanceMap._UnderlyingShared *dmap, MR.Const_AffineXf3f._Underlying *dmap2local, byte *needUpdateMesh, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            byte __deref_needUpdateMesh = needUpdateMesh.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectDistanceMap_setDistanceMap(_UnderlyingPtr, dmap._UnderlyingSharedPtr, dmap2local._UnderlyingPtr, needUpdateMesh.HasValue ? &__deref_needUpdateMesh : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
        }

        /// updates the grid to the current one
        /// Generated from method `MR::ObjectDistanceMap::updateMesh`.
        public unsafe void UpdateMesh(MR.Const_Mesh mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateMesh", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateMesh(_Underlying *_this, MR.Const_Mesh._UnderlyingShared *mesh);
            __MR_ObjectDistanceMap_updateMesh(_UnderlyingPtr, mesh._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectDistanceMap_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::selectFaces`.
        public unsafe void SelectFaces(MR._ByValue_FaceBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_selectFaces", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_selectFaces(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.FaceBitSet._Underlying *newSelection);
            __MR_ObjectDistanceMap_selectFaces(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// sets colors of selected triangles
        /// Generated from method `MR::ObjectDistanceMap::setSelectedFacesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedFacesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setSelectedFacesColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setSelectedFacesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setSelectedFacesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::selectEdges`.
        public unsafe void SelectEdges(MR._ByValue_UndirectedEdgeBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_selectEdges", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_selectEdges(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.UndirectedEdgeBitSet._Underlying *newSelection);
            __MR_ObjectDistanceMap_selectEdges(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// sets colors of selected edges
        /// Generated from method `MR::ObjectDistanceMap::setSelectedEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedEdgesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setSelectedEdgesColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setSelectedEdgesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setSelectedEdgesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setSelectedEdgesColorsForAllViewports`.
        public unsafe void SetSelectedEdgesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setSelectedEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setSelectedEdgesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectDistanceMap_setSelectedEdgesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setSelectedFacesColorsForAllViewports`.
        public unsafe void SetSelectedFacesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setSelectedFacesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setSelectedFacesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectDistanceMap_setSelectedFacesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setEdgesColorsForAllViewports`.
        public unsafe void SetEdgesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setEdgesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectDistanceMap_setEdgesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setPointsColorsForAllViewports`.
        public unsafe void SetPointsColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setPointsColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setPointsColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectDistanceMap_setPointsColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setBordersColorsForAllViewports`.
        public unsafe void SetBordersColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setBordersColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setBordersColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectDistanceMap_setBordersColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::ObjectDistanceMap::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_ObjectMeshHolder other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_copyAllSolidColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *other);
            __MR_ObjectDistanceMap_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::setCreases`.
        public unsafe void SetCreases(MR._ByValue_UndirectedEdgeBitSet creases)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setCreases", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setCreases(_Underlying *_this, MR.Misc._PassBy creases_pass_by, MR.UndirectedEdgeBitSet._Underlying *creases);
            __MR_ObjectDistanceMap_setCreases(_UnderlyingPtr, creases.PassByMode, creases.Value is not null ? creases.Value._UnderlyingPtr : null);
        }

        /// sets flat (true) or smooth (false) shading
        /// Generated from method `MR::ObjectDistanceMap::setFlatShading`.
        public unsafe void SetFlatShading(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setFlatShading", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setFlatShading(_Underlying *_this, byte on);
            __MR_ObjectDistanceMap_setFlatShading(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// sets whole new ObjectMeshData
        /// Generated from method `MR::ObjectDistanceMap::setData`.
        public unsafe void SetData(MR.Misc._Moved<MR.ObjectMeshData> data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setData", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setData(_Underlying *_this, MR.ObjectMeshData._Underlying *data);
            __MR_ObjectDistanceMap_setData(_UnderlyingPtr, data.Value._UnderlyingPtr);
        }

        /// swaps whole ObjectMeshData with given argument
        /// Generated from method `MR::ObjectDistanceMap::updateData`.
        public unsafe void UpdateData(MR.ObjectMeshData data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateData", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateData(_Underlying *_this, MR.ObjectMeshData._Underlying *data);
            __MR_ObjectDistanceMap_updateData(_UnderlyingPtr, data._UnderlyingPtr);
        }

        /// sets per-vertex colors of the object
        /// Generated from method `MR::ObjectDistanceMap::setVertsColorMap`.
        public unsafe void SetVertsColorMap(MR._ByValue_VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setVertsColorMap(_Underlying *_this, MR.Misc._PassBy vertsColorMap_pass_by, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectDistanceMap_setVertsColorMap(_UnderlyingPtr, vertsColorMap.PassByMode, vertsColorMap.Value is not null ? vertsColorMap.Value._UnderlyingPtr : null);
        }

        /// swaps per-vertex colors of the object with given argument
        /// Generated from method `MR::ObjectDistanceMap::updateVertsColorMap`.
        public unsafe void UpdateVertsColorMap(MR.VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateVertsColorMap(_Underlying *_this, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectDistanceMap_updateVertsColorMap(_UnderlyingPtr, vertsColorMap._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::setFacesColorMap`.
        public unsafe void SetFacesColorMap(MR._ByValue_FaceColors facesColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setFacesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setFacesColorMap(_Underlying *_this, MR.Misc._PassBy facesColorMap_pass_by, MR.FaceColors._Underlying *facesColorMap);
            __MR_ObjectDistanceMap_setFacesColorMap(_UnderlyingPtr, facesColorMap.PassByMode, facesColorMap.Value is not null ? facesColorMap.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::updateFacesColorMap`.
        public unsafe void UpdateFacesColorMap(MR.FaceColors updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateFacesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateFacesColorMap(_Underlying *_this, MR.FaceColors._Underlying *updated);
            __MR_ObjectDistanceMap_updateFacesColorMap(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::setEdgeWidth`.
        public unsafe void SetEdgeWidth(float edgeWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setEdgeWidth", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setEdgeWidth(_Underlying *_this, float edgeWidth);
            __MR_ObjectDistanceMap_setEdgeWidth(_UnderlyingPtr, edgeWidth);
        }

        /// Generated from method `MR::ObjectDistanceMap::setPointSize`.
        public unsafe void SetPointSize(float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setPointSize", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setPointSize(_Underlying *_this, float size);
            __MR_ObjectDistanceMap_setPointSize(_UnderlyingPtr, size);
        }

        /// Generated from method `MR::ObjectDistanceMap::setEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetEdgesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setEdgesColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setEdgesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setEdgesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setPointsColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetPointsColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setPointsColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setPointsColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setPointsColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setBordersColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetBordersColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setBordersColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setBordersColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setBordersColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setTextures`.
        public unsafe void SetTextures(MR._ByValue_Vector_MRMeshTexture_MRTextureId texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setTextures", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setTextures(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.Vector_MRMeshTexture_MRTextureId._Underlying *texture);
            __MR_ObjectDistanceMap_setTextures(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::updateTextures`.
        public unsafe void UpdateTextures(MR.Vector_MRMeshTexture_MRTextureId updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateTextures", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateTextures(_Underlying *_this, MR.Vector_MRMeshTexture_MRTextureId._Underlying *updated);
            __MR_ObjectDistanceMap_updateTextures(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// the texture ids for the faces if more than one texture is used to texture the object
        /// texture coordinates (data_.uvCoordinates) at a point can belong to different textures, depending on which face the point belongs to
        /// Generated from method `MR::ObjectDistanceMap::setTexturePerFace`.
        public unsafe void SetTexturePerFace(MR._ByValue_TexturePerFace texturePerFace)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setTexturePerFace", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setTexturePerFace(_Underlying *_this, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace);
            __MR_ObjectDistanceMap_setTexturePerFace(_UnderlyingPtr, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::updateTexturePerFace`.
        public unsafe void UpdateTexturePerFace(MR.TexturePerFace texturePerFace)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateTexturePerFace", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateTexturePerFace(_Underlying *_this, MR.TexturePerFace._Underlying *texturePerFace);
            __MR_ObjectDistanceMap_updateTexturePerFace(_UnderlyingPtr, texturePerFace._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::addTexture`.
        public unsafe void AddTexture(MR._ByValue_MeshTexture texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_addTexture", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_addTexture(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.MeshTexture._Underlying *texture);
            __MR_ObjectDistanceMap_addTexture(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setUVCoords`.
        public unsafe void SetUVCoords(MR._ByValue_VertCoords2 uvCoordinates)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setUVCoords(_Underlying *_this, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates);
            __MR_ObjectDistanceMap_setUVCoords(_UnderlyingPtr, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::updateUVCoords`.
        public unsafe void UpdateUVCoords(MR.VertCoords2 updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateUVCoords(_Underlying *_this, MR.VertCoords2._Underlying *updated);
            __MR_ObjectDistanceMap_updateUVCoords(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// copies texture, UV-coordinates and vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectDistanceMap::copyTextureAndColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyTextureAndColors(MR.Const_ObjectMeshHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_copyTextureAndColors", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_copyTextureAndColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectDistanceMap_copyTextureAndColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// copies vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectDistanceMap::copyColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyColors(MR.Const_ObjectMeshHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_copyColors", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_copyColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectDistanceMap_copyColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setAncillaryTexture`.
        public unsafe void SetAncillaryTexture(MR._ByValue_MeshTexture texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setAncillaryTexture", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setAncillaryTexture(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.MeshTexture._Underlying *texture);
            __MR_ObjectDistanceMap_setAncillaryTexture(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setAncillaryUVCoords`.
        public unsafe void SetAncillaryUVCoords(MR._ByValue_VertCoords2 uvCoordinates)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setAncillaryUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setAncillaryUVCoords(_Underlying *_this, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates);
            __MR_ObjectDistanceMap_setAncillaryUVCoords(_UnderlyingPtr, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::updateAncillaryUVCoords`.
        public unsafe void UpdateAncillaryUVCoords(MR.VertCoords2 updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_updateAncillaryUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_updateAncillaryUVCoords(_Underlying *_this, MR.VertCoords2._Underlying *updated);
            __MR_ObjectDistanceMap_updateAncillaryUVCoords(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectDistanceMap::clearAncillaryTexture`.
        public unsafe void ClearAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_clearAncillaryTexture", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_clearAncillaryTexture(_Underlying *_this);
            __MR_ObjectDistanceMap_clearAncillaryTexture(_UnderlyingPtr);
        }

        /// overrides file extension used to serialize mesh inside this object: must start from '.',
        /// nullptr means serialize in defaultSerializeMeshFormat()
        /// Generated from method `MR::ObjectDistanceMap::setSerializeFormat`.
        public unsafe void SetSerializeFormat(byte? newFormat)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setSerializeFormat", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setSerializeFormat(_Underlying *_this, byte *newFormat);
            byte __deref_newFormat = newFormat.GetValueOrDefault();
            __MR_ObjectDistanceMap_setSerializeFormat(_UnderlyingPtr, newFormat.HasValue ? &__deref_newFormat : null);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::ObjectDistanceMap::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_resetFrontColor(_Underlying *_this);
            __MR_ObjectDistanceMap_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::ObjectDistanceMap::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_resetColors", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_resetColors(_Underlying *_this);
            __MR_ObjectDistanceMap_resetColors(_UnderlyingPtr);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectDistanceMap::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::ObjectDistanceMap::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectDistanceMap::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::ObjectDistanceMap::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_ObjectDistanceMap_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::ObjectDistanceMap::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectDistanceMap::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_ObjectDistanceMap_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectDistanceMap::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectDistanceMap_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::ObjectDistanceMap::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectDistanceMap_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::ObjectDistanceMap::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setBackColor", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_ObjectDistanceMap_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectDistanceMap::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_ObjectDistanceMap_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectDistanceMap::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_ObjectDistanceMap_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::ObjectDistanceMap::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setPickable", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::ObjectDistanceMap::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setColoringType", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_ObjectDistanceMap_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::ObjectDistanceMap::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setShininess", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setShininess(_Underlying *_this, float shininess);
            __MR_ObjectDistanceMap_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::ObjectDistanceMap::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_ObjectDistanceMap_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::ObjectDistanceMap::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_ObjectDistanceMap_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectDistanceMap::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_ObjectDistanceMap_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectDistanceMap::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setName", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_ObjectDistanceMap_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::ObjectDistanceMap::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setXf", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::ObjectDistanceMap::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_resetXf", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_ObjectDistanceMap_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::ObjectDistanceMap::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_ObjectDistanceMap_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setWorldXf", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectDistanceMap_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::ObjectDistanceMap::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectDistanceMap::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setLocked", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setLocked(_Underlying *_this, byte on);
            __MR_ObjectDistanceMap_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectDistanceMap::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setParentLocked", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setParentLocked(_Underlying *_this, byte lock_);
            __MR_ObjectDistanceMap_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::ObjectDistanceMap::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_detachFromParent(_Underlying *_this);
            return __MR_ObjectDistanceMap_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::ObjectDistanceMap::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_addChild", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectDistanceMap_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::ObjectDistanceMap::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_ObjectDistanceMap_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::ObjectDistanceMap::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_removeAllChildren(_Underlying *_this);
            __MR_ObjectDistanceMap_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::ObjectDistanceMap::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_sortChildren", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_sortChildren(_Underlying *_this);
            __MR_ObjectDistanceMap_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::ObjectDistanceMap::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_select", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_select(_Underlying *_this, byte on);
            return __MR_ObjectDistanceMap_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::ObjectDistanceMap::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setAncillary", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setAncillary(_Underlying *_this, byte ancillary);
            __MR_ObjectDistanceMap_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::ObjectDistanceMap::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setVisible", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectDistanceMap::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectDistanceMap_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::ObjectDistanceMap::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_swap", ExactSpelling = true)]
            extern static void __MR_ObjectDistanceMap_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_ObjectDistanceMap_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::ObjectDistanceMap::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_addTag", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectDistanceMap_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::ObjectDistanceMap::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectDistanceMap_removeTag", ExactSpelling = true)]
            extern static byte __MR_ObjectDistanceMap_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectDistanceMap_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectDistanceMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectDistanceMap`/`Const_ObjectDistanceMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectDistanceMap
    {
        internal readonly Const_ObjectDistanceMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectDistanceMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectDistanceMap(MR.Misc._Moved<ObjectDistanceMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectDistanceMap(MR.Misc._Moved<ObjectDistanceMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectDistanceMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectDistanceMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectDistanceMap`/`Const_ObjectDistanceMap` directly.
    public class _InOptMut_ObjectDistanceMap
    {
        public ObjectDistanceMap? Opt;

        public _InOptMut_ObjectDistanceMap() {}
        public _InOptMut_ObjectDistanceMap(ObjectDistanceMap value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectDistanceMap(ObjectDistanceMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectDistanceMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectDistanceMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectDistanceMap`/`Const_ObjectDistanceMap` to pass it to the function.
    public class _InOptConst_ObjectDistanceMap
    {
        public Const_ObjectDistanceMap? Opt;

        public _InOptConst_ObjectDistanceMap() {}
        public _InOptConst_ObjectDistanceMap(Const_ObjectDistanceMap value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectDistanceMap(Const_ObjectDistanceMap value) {return new(value);}
    }
}
