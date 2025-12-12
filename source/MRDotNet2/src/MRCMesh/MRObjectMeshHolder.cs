public static partial class MR
{
    public enum MeshVisualizePropertyType : int
    {
        Faces = 0,
        Texture = 1,
        Edges = 2,
        Points = 3,
        SelectedFaces = 4,
        SelectedEdges = 5,
        EnableShading = 6,
        FlatShading = 7,
        OnlyOddFragments = 8,
        BordersHighlight = 9,
        // recommended for drawing edges on top of mesh
        PolygonOffsetFromCamera = 10,
        Count = 11,
    }

    /// an object that stores a mesh
    /// Generated from class `MR::ObjectMeshHolder`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectVoxels`
    /// This is the const half of the class.
    public class Const_ObjectMeshHolder : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectMeshHolder_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectMeshHolder_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectMeshHolder_UseCount();
                return __MR_std_shared_ptr_MR_ObjectMeshHolder_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectMeshHolder(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectMeshHolder_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectMeshHolder_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectMeshHolder(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectMeshHolder _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectMeshHolder_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectMeshHolder_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectMeshHolder_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectMeshHolder_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectMeshHolder_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectMeshHolder_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectMeshHolder() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_ObjectMeshHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_ObjectMeshHolder_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectMeshHolder_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_ObjectMeshHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_ObjectMeshHolder_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectMeshHolder_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_ObjectMeshHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_ObjectMeshHolder_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectMeshHolder_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ObjectMeshHolder?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectMeshHolder(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectMeshHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectMeshHolder?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectMeshHolder(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectMeshHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectMeshHolder() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectMeshHolder_DefaultConstruct();
            _LateMakeShared(__MR_ObjectMeshHolder_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectMeshHolder::ObjectMeshHolder`.
        public unsafe Const_ObjectMeshHolder(MR._ByValue_ObjectMeshHolder _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectMeshHolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectMeshHolder._Underlying *_other);
            _LateMakeShared(__MR_ObjectMeshHolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectMeshHolder::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_StaticTypeName();
            var __ret = __MR_ObjectMeshHolder_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectMeshHolder::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_typeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_typeName(_Underlying *_this);
            var __ret = __MR_ObjectMeshHolder_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// mesh object can be seen if the mesh has at least one edge
        /// Generated from method `MR::ObjectMeshHolder::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_hasVisualRepresentation(_Underlying *_this);
            return __MR_ObjectMeshHolder_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_hasModel", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_hasModel(_Underlying *_this);
            return __MR_ObjectMeshHolder_hasModel(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::mesh`.
        public unsafe MR.Const_Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_mesh", ExactSpelling = true)]
            extern static MR.Const_Mesh._UnderlyingShared *__MR_ObjectMeshHolder_mesh(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// \return the pair ( mesh, selected triangles ) if any triangle is selected or whole mesh otherwise
        /// Generated from method `MR::ObjectMeshHolder::meshPart`.
        public unsafe MR.MeshPart MeshPart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_meshPart", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_ObjectMeshHolder_meshPart(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_meshPart(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectMeshHolder::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectMeshHolder_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectMeshHolder_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectMeshHolder::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectMeshHolder_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectMeshHolder_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectMeshHolder::getSelectedFaces`.
        public unsafe MR.Const_FaceBitSet GetSelectedFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSelectedFaces", ExactSpelling = true)]
            extern static MR.Const_FaceBitSet._Underlying *__MR_ObjectMeshHolder_getSelectedFaces(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getSelectedFaces(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected triangles
        /// Generated from method `MR::ObjectMeshHolder::getSelectedFacesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedFacesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSelectedFacesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getSelectedFacesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectMeshHolder_getSelectedFacesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getSelectedEdges`.
        public unsafe MR.Const_UndirectedEdgeBitSet GetSelectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSelectedEdges", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectMeshHolder_getSelectedEdges(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getSelectedEdges(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected edges
        /// Generated from method `MR::ObjectMeshHolder::getSelectedEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedEdgesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSelectedEdgesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getSelectedEdgesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectMeshHolder_getSelectedEdgesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getSelectedEdgesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedEdgesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSelectedEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getSelectedEdgesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getSelectedEdgesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getSelectedFacesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedFacesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSelectedFacesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getSelectedFacesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getSelectedFacesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getEdgesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetEdgesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getEdgesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getEdgesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getPointsColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetPointsColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getPointsColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getPointsColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getPointsColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getBordersColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBordersColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getBordersColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getBordersColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getBordersColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Edges on mesh, that will have sharp visualization even with smooth shading
        /// Generated from method `MR::ObjectMeshHolder::creases`.
        public unsafe MR.Const_UndirectedEdgeBitSet Creases()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_creases", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectMeshHolder_creases(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_creases(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::flatShading`.
        public unsafe bool FlatShading()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_flatShading", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_flatShading(_Underlying *_this);
            return __MR_ObjectMeshHolder_flatShading(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_ObjectMeshHolder_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// get all visualize properties masks
        /// Generated from method `MR::ObjectMeshHolder::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_ObjectMeshHolder_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_ObjectMeshHolder_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::ObjectMeshHolder::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_ObjectMeshHolder_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_ObjectMeshHolder_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// provides read-only access to whole ObjectMeshData
        /// Generated from method `MR::ObjectMeshHolder::data`.
        public unsafe MR.Const_ObjectMeshData Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_data", ExactSpelling = true)]
            extern static MR.Const_ObjectMeshData._Underlying *__MR_ObjectMeshHolder_data(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_data(_UnderlyingPtr), is_owning: false);
        }

        /// returns per-vertex colors of the object
        /// Generated from method `MR::ObjectMeshHolder::getVertsColorMap`.
        public unsafe MR.Const_VertColors GetVertsColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getVertsColorMap", ExactSpelling = true)]
            extern static MR.Const_VertColors._Underlying *__MR_ObjectMeshHolder_getVertsColorMap(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getVertsColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getFacesColorMap`.
        public unsafe MR.Const_FaceColors GetFacesColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getFacesColorMap", ExactSpelling = true)]
            extern static MR.Const_FaceColors._Underlying *__MR_ObjectMeshHolder_getFacesColorMap(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getFacesColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getEdgeWidth`.
        public unsafe float GetEdgeWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getEdgeWidth", ExactSpelling = true)]
            extern static float __MR_ObjectMeshHolder_getEdgeWidth(_Underlying *_this);
            return __MR_ObjectMeshHolder_getEdgeWidth(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getPointSize", ExactSpelling = true)]
            extern static float __MR_ObjectMeshHolder_getPointSize(_Underlying *_this);
            return __MR_ObjectMeshHolder_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::getEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetEdgesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getEdgesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getEdgesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectMeshHolder_getEdgesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getPointsColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetPointsColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getPointsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getPointsColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectMeshHolder_getPointsColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getBordersColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetBordersColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getBordersColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getBordersColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectMeshHolder_getBordersColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// returns first texture in the vector. If there is no textures, returns empty texture
        /// Generated from method `MR::ObjectMeshHolder::getTexture`.
        public unsafe MR.Const_MeshTexture GetTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getTexture", ExactSpelling = true)]
            extern static MR.Const_MeshTexture._Underlying *__MR_ObjectMeshHolder_getTexture(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getTexture(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getTextures`.
        public unsafe MR.Const_Vector_MRMeshTexture_MRTextureId GetTextures()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getTextures", ExactSpelling = true)]
            extern static MR.Const_Vector_MRMeshTexture_MRTextureId._Underlying *__MR_ObjectMeshHolder_getTextures(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getTextures(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getTexturePerFace`.
        public unsafe MR.Const_TexturePerFace GetTexturePerFace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getTexturePerFace", ExactSpelling = true)]
            extern static MR.Const_TexturePerFace._Underlying *__MR_ObjectMeshHolder_getTexturePerFace(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getTexturePerFace(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getUVCoords`.
        public unsafe MR.Const_VertCoords2 GetUVCoords()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getUVCoords", ExactSpelling = true)]
            extern static MR.Const_VertCoords2._Underlying *__MR_ObjectMeshHolder_getUVCoords(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getUVCoords(_UnderlyingPtr), is_owning: false);
        }

        // ancillary texture can be used to have custom features visualization without affecting real one
        /// Generated from method `MR::ObjectMeshHolder::getAncillaryTexture`.
        public unsafe MR.Const_MeshTexture GetAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getAncillaryTexture", ExactSpelling = true)]
            extern static MR.Const_MeshTexture._Underlying *__MR_ObjectMeshHolder_getAncillaryTexture(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getAncillaryTexture(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::getAncillaryUVCoords`.
        public unsafe MR.Const_VertCoords2 GetAncillaryUVCoords()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getAncillaryUVCoords", ExactSpelling = true)]
            extern static MR.Const_VertCoords2._Underlying *__MR_ObjectMeshHolder_getAncillaryUVCoords(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getAncillaryUVCoords(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::hasAncillaryTexture`.
        public unsafe bool HasAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_hasAncillaryTexture", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_hasAncillaryTexture(_Underlying *_this);
            return __MR_ObjectMeshHolder_hasAncillaryTexture(_UnderlyingPtr) != 0;
        }

        /// returns dirty flag of currently using normal type if they are dirty in render representation
        /// Generated from method `MR::ObjectMeshHolder::getNeededNormalsRenderDirtyValue`.
        public unsafe uint GetNeededNormalsRenderDirtyValue(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getNeededNormalsRenderDirtyValue", ExactSpelling = true)]
            extern static uint __MR_ObjectMeshHolder_getNeededNormalsRenderDirtyValue(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_getNeededNormalsRenderDirtyValue(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns cached information whether the mesh is closed
        /// Generated from method `MR::ObjectMeshHolder::isMeshClosed`.
        public unsafe bool IsMeshClosed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isMeshClosed", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isMeshClosed(_Underlying *_this);
            return __MR_ObjectMeshHolder_isMeshClosed(_UnderlyingPtr) != 0;
        }

        /// returns cached bounding box of this mesh object in world coordinates;
        /// if you need bounding box in local coordinates please call getBoundingBox()
        /// Generated from method `MR::ObjectMeshHolder::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectMeshHolder_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectMeshHolder_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns cached information about the number of selected faces in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numSelectedFaces`.
        public unsafe ulong NumSelectedFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numSelectedFaces", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numSelectedFaces(_Underlying *_this);
            return __MR_ObjectMeshHolder_numSelectedFaces(_UnderlyingPtr);
        }

        /// returns cached information about the number of selected undirected edges in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numSelectedEdges`.
        public unsafe ulong NumSelectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numSelectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numSelectedEdges(_Underlying *_this);
            return __MR_ObjectMeshHolder_numSelectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of crease undirected edges in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numCreaseEdges`.
        public unsafe ulong NumCreaseEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numCreaseEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numCreaseEdges(_Underlying *_this);
            return __MR_ObjectMeshHolder_numCreaseEdges(_UnderlyingPtr);
        }

        /// returns cached summed area of mesh triangles
        /// Generated from method `MR::ObjectMeshHolder::totalArea`.
        public unsafe double TotalArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_totalArea", ExactSpelling = true)]
            extern static double __MR_ObjectMeshHolder_totalArea(_Underlying *_this);
            return __MR_ObjectMeshHolder_totalArea(_UnderlyingPtr);
        }

        /// returns cached area of selected triangles
        /// Generated from method `MR::ObjectMeshHolder::selectedArea`.
        public unsafe double SelectedArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_selectedArea", ExactSpelling = true)]
            extern static double __MR_ObjectMeshHolder_selectedArea(_Underlying *_this);
            return __MR_ObjectMeshHolder_selectedArea(_UnderlyingPtr);
        }

        /// returns cached volume of space surrounded by the mesh, which is valid only if mesh is closed
        /// Generated from method `MR::ObjectMeshHolder::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_volume", ExactSpelling = true)]
            extern static double __MR_ObjectMeshHolder_volume(_Underlying *_this);
            return __MR_ObjectMeshHolder_volume(_UnderlyingPtr);
        }

        /// returns cached average edge length
        /// Generated from method `MR::ObjectMeshHolder::avgEdgeLen`.
        public unsafe float AvgEdgeLen()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_avgEdgeLen", ExactSpelling = true)]
            extern static float __MR_ObjectMeshHolder_avgEdgeLen(_Underlying *_this);
            return __MR_ObjectMeshHolder_avgEdgeLen(_UnderlyingPtr);
        }

        /// returns cached information about the number of undirected edges in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numUndirectedEdges`.
        public unsafe ulong NumUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numUndirectedEdges(_Underlying *_this);
            return __MR_ObjectMeshHolder_numUndirectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of holes in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numHoles`.
        public unsafe ulong NumHoles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numHoles", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numHoles(_Underlying *_this);
            return __MR_ObjectMeshHolder_numHoles(_UnderlyingPtr);
        }

        /// returns cached information about the number of components in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numComponents`.
        public unsafe ulong NumComponents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numComponents", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numComponents(_Underlying *_this);
            return __MR_ObjectMeshHolder_numComponents(_UnderlyingPtr);
        }

        /// returns cached information about the number of handles in the mesh
        /// Generated from method `MR::ObjectMeshHolder::numHandles`.
        public unsafe ulong NumHandles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_numHandles", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_numHandles(_Underlying *_this);
            return __MR_ObjectMeshHolder_numHandles(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectMeshHolder::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_heapBytes(_Underlying *_this);
            return __MR_ObjectMeshHolder_heapBytes(_UnderlyingPtr);
        }

        /// returns overriden file extension used to serialize mesh inside this object, nullptr means defaultSerializeMeshFormat()
        /// Generated from method `MR::ObjectMeshHolder::serializeFormat`.
        public unsafe byte? SerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_serializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_serializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectMeshHolder_serializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// returns overriden file extension used to serialize mesh inside this object if set, or defaultSerializeMeshFormat().c_str() otherwise; never returns nullptr
        /// Generated from method `MR::ObjectMeshHolder::actualSerializeFormat`.
        public unsafe byte? ActualSerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_actualSerializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_actualSerializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectMeshHolder_actualSerializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectMeshHolder::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshHolder_getModelHash(_Underlying *_this);
            return __MR_ObjectMeshHolder_getModelHash(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_sameModels", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_ObjectMeshHolder_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_StaticClassName();
            var __ret = __MR_ObjectMeshHolder_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectMeshHolder::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectMeshHolder_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectMeshHolder_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectMeshHolder::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_StaticClassNameInPlural();
            var __ret = __MR_ObjectMeshHolder_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectMeshHolder::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectMeshHolder_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectMeshHolder_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::ObjectMeshHolder::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::ObjectMeshHolder::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectMeshHolder_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::ObjectMeshHolder::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectMeshHolder::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectMeshHolder_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectMeshHolder::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectMeshHolder_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::ObjectMeshHolder::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectMeshHolder_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::ObjectMeshHolder::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectMeshHolder_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_ObjectMeshHolder_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectMeshHolder::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_ObjectMeshHolder_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_ObjectMeshHolder_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectMeshHolder::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_ObjectMeshHolder_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::ObjectMeshHolder::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_ObjectMeshHolder_getDirtyFlags(_Underlying *_this);
            return __MR_ObjectMeshHolder_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::ObjectMeshHolder::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_resetDirty", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_resetDirty(_Underlying *_this);
            __MR_ObjectMeshHolder_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::ObjectMeshHolder::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_ObjectMeshHolder_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::ObjectMeshHolder::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectMeshHolder_getBoundingBox(_Underlying *_this);
            return __MR_ObjectMeshHolder_getBoundingBox(_UnderlyingPtr);
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::ObjectMeshHolder::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isPickable", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::ObjectMeshHolder::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_ObjectMeshHolder_getColoringType(_Underlying *_this);
            return __MR_ObjectMeshHolder_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::ObjectMeshHolder::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getShininess", ExactSpelling = true)]
            extern static float __MR_ObjectMeshHolder_getShininess(_Underlying *_this);
            return __MR_ObjectMeshHolder_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::ObjectMeshHolder::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_ObjectMeshHolder_getSpecularStrength(_Underlying *_this);
            return __MR_ObjectMeshHolder_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::ObjectMeshHolder::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_ObjectMeshHolder_getAmbientStrength(_Underlying *_this);
            return __MR_ObjectMeshHolder_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::ObjectMeshHolder::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_render", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_ObjectMeshHolder_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::ObjectMeshHolder::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_renderForPicker", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_ObjectMeshHolder_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::ObjectMeshHolder::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_renderUi", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_ObjectMeshHolder_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::ObjectMeshHolder::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_ObjectMeshHolder_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_ObjectMeshHolder_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectMeshHolder::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_ObjectMeshHolder_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_ObjectMeshHolder_name(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::ObjectMeshHolder::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectMeshHolder_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectMeshHolder_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::ObjectMeshHolder::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_ObjectMeshHolder_xfsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::ObjectMeshHolder::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ObjectMeshHolder_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectMeshHolder_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::ObjectMeshHolder::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectMeshHolder_globalVisibilityMask(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::ObjectMeshHolder::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::ObjectMeshHolder::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isLocked(_Underlying *_this);
            return __MR_ObjectMeshHolder_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::ObjectMeshHolder::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isParentLocked(_Underlying *_this);
            return __MR_ObjectMeshHolder_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::ObjectMeshHolder::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isAncestor", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_ObjectMeshHolder_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isSelected", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isSelected(_Underlying *_this);
            return __MR_ObjectMeshHolder_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectMeshHolder::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isAncillary(_Underlying *_this);
            return __MR_ObjectMeshHolder_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::ObjectMeshHolder::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isGlobalAncillary(_Underlying *_this);
            return __MR_ObjectMeshHolder_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::ObjectMeshHolder::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_isVisible", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectMeshHolder_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectMeshHolder::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectMeshHolder_visibilityMask(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectMeshHolder::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_resetRedrawFlag(_Underlying *_this);
            __MR_ObjectMeshHolder_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::ObjectMeshHolder::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectMeshHolder_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectMeshHolder_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::ObjectMeshHolder::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectMeshHolder_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectMeshHolder_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::ObjectMeshHolder::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectMeshHolder_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectMeshHolder_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::ObjectMeshHolder::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_ObjectMeshHolder_tags(_Underlying *_this);
            return new(__MR_ObjectMeshHolder_tags(_UnderlyingPtr), is_owning: false);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectMeshHolder::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectMeshHolder_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectMeshHolder_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// an object that stores a mesh
    /// Generated from class `MR::ObjectMeshHolder`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectVoxels`
    /// This is the non-const half of the class.
    public class ObjectMeshHolder : Const_ObjectMeshHolder
    {
        internal unsafe ObjectMeshHolder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectMeshHolder(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(ObjectMeshHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectMeshHolder_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectMeshHolder_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(ObjectMeshHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_ObjectMeshHolder_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectMeshHolder_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(ObjectMeshHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_ObjectMeshHolder_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectMeshHolder_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ObjectMeshHolder?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectMeshHolder(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectMeshHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectMeshHolder?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectMeshHolder(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectMeshHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectMeshHolder() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectMeshHolder_DefaultConstruct();
            _LateMakeShared(__MR_ObjectMeshHolder_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectMeshHolder::ObjectMeshHolder`.
        public unsafe ObjectMeshHolder(MR._ByValue_ObjectMeshHolder _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectMeshHolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectMeshHolder._Underlying *_other);
            _LateMakeShared(__MR_ObjectMeshHolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectMeshHolder::operator=`.
        public unsafe MR.ObjectMeshHolder Assign(MR._ByValue_ObjectMeshHolder _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectMeshHolder_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectMeshHolder._Underlying *_other);
            return new(__MR_ObjectMeshHolder_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectMeshHolder::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_applyScale", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_applyScale(_Underlying *_this, float scaleFactor);
            __MR_ObjectMeshHolder_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// Generated from method `MR::ObjectMeshHolder::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectMeshHolder_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::selectFaces`.
        public unsafe void SelectFaces(MR._ByValue_FaceBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_selectFaces", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_selectFaces(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.FaceBitSet._Underlying *newSelection);
            __MR_ObjectMeshHolder_selectFaces(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// sets colors of selected triangles
        /// Generated from method `MR::ObjectMeshHolder::setSelectedFacesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedFacesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setSelectedFacesColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setSelectedFacesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setSelectedFacesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::selectEdges`.
        public unsafe void SelectEdges(MR._ByValue_UndirectedEdgeBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_selectEdges", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_selectEdges(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.UndirectedEdgeBitSet._Underlying *newSelection);
            __MR_ObjectMeshHolder_selectEdges(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// sets colors of selected edges
        /// Generated from method `MR::ObjectMeshHolder::setSelectedEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedEdgesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setSelectedEdgesColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setSelectedEdgesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setSelectedEdgesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setSelectedEdgesColorsForAllViewports`.
        public unsafe void SetSelectedEdgesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setSelectedEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setSelectedEdgesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectMeshHolder_setSelectedEdgesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setSelectedFacesColorsForAllViewports`.
        public unsafe void SetSelectedFacesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setSelectedFacesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setSelectedFacesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectMeshHolder_setSelectedFacesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setEdgesColorsForAllViewports`.
        public unsafe void SetEdgesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setEdgesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectMeshHolder_setEdgesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setPointsColorsForAllViewports`.
        public unsafe void SetPointsColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setPointsColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setPointsColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectMeshHolder_setPointsColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setBordersColorsForAllViewports`.
        public unsafe void SetBordersColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setBordersColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setBordersColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectMeshHolder_setBordersColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::ObjectMeshHolder::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_ObjectMeshHolder other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_copyAllSolidColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *other);
            __MR_ObjectMeshHolder_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::setCreases`.
        public unsafe void SetCreases(MR._ByValue_UndirectedEdgeBitSet creases)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setCreases", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setCreases(_Underlying *_this, MR.Misc._PassBy creases_pass_by, MR.UndirectedEdgeBitSet._Underlying *creases);
            __MR_ObjectMeshHolder_setCreases(_UnderlyingPtr, creases.PassByMode, creases.Value is not null ? creases.Value._UnderlyingPtr : null);
        }

        /// sets flat (true) or smooth (false) shading
        /// Generated from method `MR::ObjectMeshHolder::setFlatShading`.
        public unsafe void SetFlatShading(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setFlatShading", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setFlatShading(_Underlying *_this, byte on);
            __MR_ObjectMeshHolder_setFlatShading(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// sets whole new ObjectMeshData
        /// Generated from method `MR::ObjectMeshHolder::setData`.
        public unsafe void SetData(MR.Misc._Moved<MR.ObjectMeshData> data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setData", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setData(_Underlying *_this, MR.ObjectMeshData._Underlying *data);
            __MR_ObjectMeshHolder_setData(_UnderlyingPtr, data.Value._UnderlyingPtr);
        }

        /// swaps whole ObjectMeshData with given argument
        /// Generated from method `MR::ObjectMeshHolder::updateData`.
        public unsafe void UpdateData(MR.ObjectMeshData data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateData", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateData(_Underlying *_this, MR.ObjectMeshData._Underlying *data);
            __MR_ObjectMeshHolder_updateData(_UnderlyingPtr, data._UnderlyingPtr);
        }

        /// sets per-vertex colors of the object
        /// Generated from method `MR::ObjectMeshHolder::setVertsColorMap`.
        public unsafe void SetVertsColorMap(MR._ByValue_VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setVertsColorMap(_Underlying *_this, MR.Misc._PassBy vertsColorMap_pass_by, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectMeshHolder_setVertsColorMap(_UnderlyingPtr, vertsColorMap.PassByMode, vertsColorMap.Value is not null ? vertsColorMap.Value._UnderlyingPtr : null);
        }

        /// swaps per-vertex colors of the object with given argument
        /// Generated from method `MR::ObjectMeshHolder::updateVertsColorMap`.
        public unsafe void UpdateVertsColorMap(MR.VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateVertsColorMap(_Underlying *_this, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectMeshHolder_updateVertsColorMap(_UnderlyingPtr, vertsColorMap._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::setFacesColorMap`.
        public unsafe void SetFacesColorMap(MR._ByValue_FaceColors facesColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setFacesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setFacesColorMap(_Underlying *_this, MR.Misc._PassBy facesColorMap_pass_by, MR.FaceColors._Underlying *facesColorMap);
            __MR_ObjectMeshHolder_setFacesColorMap(_UnderlyingPtr, facesColorMap.PassByMode, facesColorMap.Value is not null ? facesColorMap.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::updateFacesColorMap`.
        public unsafe void UpdateFacesColorMap(MR.FaceColors updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateFacesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateFacesColorMap(_Underlying *_this, MR.FaceColors._Underlying *updated);
            __MR_ObjectMeshHolder_updateFacesColorMap(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::setEdgeWidth`.
        public unsafe void SetEdgeWidth(float edgeWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setEdgeWidth", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setEdgeWidth(_Underlying *_this, float edgeWidth);
            __MR_ObjectMeshHolder_setEdgeWidth(_UnderlyingPtr, edgeWidth);
        }

        /// Generated from method `MR::ObjectMeshHolder::setPointSize`.
        public unsafe void SetPointSize(float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setPointSize", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setPointSize(_Underlying *_this, float size);
            __MR_ObjectMeshHolder_setPointSize(_UnderlyingPtr, size);
        }

        /// Generated from method `MR::ObjectMeshHolder::setEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetEdgesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setEdgesColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setEdgesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setEdgesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setPointsColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetPointsColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setPointsColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setPointsColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setPointsColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setBordersColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetBordersColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setBordersColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setBordersColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setBordersColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setTextures`.
        public unsafe void SetTextures(MR._ByValue_Vector_MRMeshTexture_MRTextureId texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setTextures", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setTextures(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.Vector_MRMeshTexture_MRTextureId._Underlying *texture);
            __MR_ObjectMeshHolder_setTextures(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::updateTextures`.
        public unsafe void UpdateTextures(MR.Vector_MRMeshTexture_MRTextureId updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateTextures", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateTextures(_Underlying *_this, MR.Vector_MRMeshTexture_MRTextureId._Underlying *updated);
            __MR_ObjectMeshHolder_updateTextures(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// the texture ids for the faces if more than one texture is used to texture the object
        /// texture coordinates (data_.uvCoordinates) at a point can belong to different textures, depending on which face the point belongs to
        /// Generated from method `MR::ObjectMeshHolder::setTexturePerFace`.
        public unsafe void SetTexturePerFace(MR._ByValue_TexturePerFace texturePerFace)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setTexturePerFace", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setTexturePerFace(_Underlying *_this, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace);
            __MR_ObjectMeshHolder_setTexturePerFace(_UnderlyingPtr, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::updateTexturePerFace`.
        public unsafe void UpdateTexturePerFace(MR.TexturePerFace texturePerFace)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateTexturePerFace", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateTexturePerFace(_Underlying *_this, MR.TexturePerFace._Underlying *texturePerFace);
            __MR_ObjectMeshHolder_updateTexturePerFace(_UnderlyingPtr, texturePerFace._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::addTexture`.
        public unsafe void AddTexture(MR._ByValue_MeshTexture texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_addTexture", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_addTexture(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.MeshTexture._Underlying *texture);
            __MR_ObjectMeshHolder_addTexture(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setUVCoords`.
        public unsafe void SetUVCoords(MR._ByValue_VertCoords2 uvCoordinates)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setUVCoords(_Underlying *_this, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates);
            __MR_ObjectMeshHolder_setUVCoords(_UnderlyingPtr, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::updateUVCoords`.
        public unsafe void UpdateUVCoords(MR.VertCoords2 updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateUVCoords(_Underlying *_this, MR.VertCoords2._Underlying *updated);
            __MR_ObjectMeshHolder_updateUVCoords(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// copies texture, UV-coordinates and vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectMeshHolder::copyTextureAndColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyTextureAndColors(MR.Const_ObjectMeshHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_copyTextureAndColors", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_copyTextureAndColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectMeshHolder_copyTextureAndColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// copies vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectMeshHolder::copyColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyColors(MR.Const_ObjectMeshHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_copyColors", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_copyColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectMeshHolder_copyColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setAncillaryTexture`.
        public unsafe void SetAncillaryTexture(MR._ByValue_MeshTexture texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setAncillaryTexture", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setAncillaryTexture(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.MeshTexture._Underlying *texture);
            __MR_ObjectMeshHolder_setAncillaryTexture(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setAncillaryUVCoords`.
        public unsafe void SetAncillaryUVCoords(MR._ByValue_VertCoords2 uvCoordinates)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setAncillaryUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setAncillaryUVCoords(_Underlying *_this, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates);
            __MR_ObjectMeshHolder_setAncillaryUVCoords(_UnderlyingPtr, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::updateAncillaryUVCoords`.
        public unsafe void UpdateAncillaryUVCoords(MR.VertCoords2 updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_updateAncillaryUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_updateAncillaryUVCoords(_Underlying *_this, MR.VertCoords2._Underlying *updated);
            __MR_ObjectMeshHolder_updateAncillaryUVCoords(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectMeshHolder::clearAncillaryTexture`.
        public unsafe void ClearAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_clearAncillaryTexture", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_clearAncillaryTexture(_Underlying *_this);
            __MR_ObjectMeshHolder_clearAncillaryTexture(_UnderlyingPtr);
        }

        /// overrides file extension used to serialize mesh inside this object: must start from '.',
        /// nullptr means serialize in defaultSerializeMeshFormat()
        /// Generated from method `MR::ObjectMeshHolder::setSerializeFormat`.
        public unsafe void SetSerializeFormat(byte? newFormat)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setSerializeFormat", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setSerializeFormat(_Underlying *_this, byte *newFormat);
            byte __deref_newFormat = newFormat.GetValueOrDefault();
            __MR_ObjectMeshHolder_setSerializeFormat(_UnderlyingPtr, newFormat.HasValue ? &__deref_newFormat : null);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::ObjectMeshHolder::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_resetFrontColor(_Underlying *_this);
            __MR_ObjectMeshHolder_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::ObjectMeshHolder::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_resetColors", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_resetColors(_Underlying *_this);
            __MR_ObjectMeshHolder_resetColors(_UnderlyingPtr);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectMeshHolder::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::ObjectMeshHolder::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectMeshHolder::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::ObjectMeshHolder::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_ObjectMeshHolder_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::ObjectMeshHolder::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectMeshHolder::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_ObjectMeshHolder_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectMeshHolder::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectMeshHolder_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::ObjectMeshHolder::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectMeshHolder_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::ObjectMeshHolder::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setBackColor", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_ObjectMeshHolder_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectMeshHolder::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_ObjectMeshHolder_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectMeshHolder::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_ObjectMeshHolder_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::ObjectMeshHolder::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setPickable", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::ObjectMeshHolder::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setColoringType", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_ObjectMeshHolder_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::ObjectMeshHolder::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setShininess", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setShininess(_Underlying *_this, float shininess);
            __MR_ObjectMeshHolder_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::ObjectMeshHolder::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_ObjectMeshHolder_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::ObjectMeshHolder::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_ObjectMeshHolder_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectMeshHolder::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_ObjectMeshHolder_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectMeshHolder::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setName", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_ObjectMeshHolder_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::ObjectMeshHolder::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setXf", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::ObjectMeshHolder::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_resetXf", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_ObjectMeshHolder_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::ObjectMeshHolder::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_ObjectMeshHolder_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setWorldXf", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectMeshHolder_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::ObjectMeshHolder::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshHolder::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setLocked", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setLocked(_Underlying *_this, byte on);
            __MR_ObjectMeshHolder_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectMeshHolder::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setParentLocked", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setParentLocked(_Underlying *_this, byte lock_);
            __MR_ObjectMeshHolder_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::ObjectMeshHolder::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_detachFromParent(_Underlying *_this);
            return __MR_ObjectMeshHolder_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::ObjectMeshHolder::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_addChild", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectMeshHolder_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::ObjectMeshHolder::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_ObjectMeshHolder_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::ObjectMeshHolder::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_removeAllChildren(_Underlying *_this);
            __MR_ObjectMeshHolder_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::ObjectMeshHolder::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_sortChildren", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_sortChildren(_Underlying *_this);
            __MR_ObjectMeshHolder_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::ObjectMeshHolder::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_select", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_select(_Underlying *_this, byte on);
            return __MR_ObjectMeshHolder_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::ObjectMeshHolder::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setAncillary", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setAncillary(_Underlying *_this, byte ancillary);
            __MR_ObjectMeshHolder_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::ObjectMeshHolder::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setVisible", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectMeshHolder::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectMeshHolder_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::ObjectMeshHolder::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_swap", ExactSpelling = true)]
            extern static void __MR_ObjectMeshHolder_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_ObjectMeshHolder_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::ObjectMeshHolder::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_addTag", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectMeshHolder_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::ObjectMeshHolder::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_removeTag", ExactSpelling = true)]
            extern static byte __MR_ObjectMeshHolder_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectMeshHolder_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectMeshHolder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectMeshHolder`/`Const_ObjectMeshHolder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectMeshHolder
    {
        internal readonly Const_ObjectMeshHolder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectMeshHolder() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectMeshHolder(MR.Misc._Moved<ObjectMeshHolder> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectMeshHolder(MR.Misc._Moved<ObjectMeshHolder> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectMeshHolder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectMeshHolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectMeshHolder`/`Const_ObjectMeshHolder` directly.
    public class _InOptMut_ObjectMeshHolder
    {
        public ObjectMeshHolder? Opt;

        public _InOptMut_ObjectMeshHolder() {}
        public _InOptMut_ObjectMeshHolder(ObjectMeshHolder value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectMeshHolder(ObjectMeshHolder value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectMeshHolder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectMeshHolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectMeshHolder`/`Const_ObjectMeshHolder` to pass it to the function.
    public class _InOptConst_ObjectMeshHolder
    {
        public Const_ObjectMeshHolder? Opt;

        public _InOptConst_ObjectMeshHolder() {}
        public _InOptConst_ObjectMeshHolder(Const_ObjectMeshHolder value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectMeshHolder(Const_ObjectMeshHolder value) {return new(value);}
    }

    /// returns file extension used to serialize ObjectMeshHolder by default (if not overridden in specific object),
    /// the string starts with '.'
    /// Generated from function `MR::defaultSerializeMeshFormat`.
    public static unsafe MR.Std.Const_String DefaultSerializeMeshFormat()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_defaultSerializeMeshFormat", ExactSpelling = true)]
        extern static MR.Std.Const_String._Underlying *__MR_defaultSerializeMeshFormat();
        return new(__MR_defaultSerializeMeshFormat(), is_owning: false);
    }

    /// sets file extension used to serialize serialize ObjectMeshHolder by default (if not overridden in specific object),
    /// the string must start from '.';
    // serialization falls back to the PLY format if given format support is available
    // NOTE: CTM format support is available in the MRIOExtras library; make sure to load it if you prefer CTM
    /// Generated from function `MR::setDefaultSerializeMeshFormat`.
    public static unsafe void SetDefaultSerializeMeshFormat(ReadOnlySpan<char> newFormat)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setDefaultSerializeMeshFormat", ExactSpelling = true)]
        extern static void __MR_setDefaultSerializeMeshFormat(byte *newFormat, byte *newFormat_end);
        byte[] __bytes_newFormat = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(newFormat.Length)];
        int __len_newFormat = System.Text.Encoding.UTF8.GetBytes(newFormat, __bytes_newFormat);
        fixed (byte *__ptr_newFormat = __bytes_newFormat)
        {
            __MR_setDefaultSerializeMeshFormat(__ptr_newFormat, __ptr_newFormat + __len_newFormat);
        }
    }
}
