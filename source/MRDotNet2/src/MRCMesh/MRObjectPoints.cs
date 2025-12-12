public static partial class MR
{
    /// an object that stores a points
    /// Generated from class `MR::ObjectPoints`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectPointsHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_ObjectPoints : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectPoints_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectPoints_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectPoints_UseCount();
                return __MR_std_shared_ptr_MR_ObjectPoints_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectPoints_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectPoints_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectPoints(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectPoints_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectPoints_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectPoints_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectPoints_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectPoints(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectPoints _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectPoints_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectPoints_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectPoints_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectPoints_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectPoints_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectPoints_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectPoints_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectPoints() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_ObjectPoints_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_ObjectPoints_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_ObjectPoints_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_ObjectPointsHolder(Const_ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_ObjectPointsHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectPointsHolder._Underlying *__MR_ObjectPoints_UpcastTo_MR_ObjectPointsHolder(_Underlying *_this);
            return MR.Const_ObjectPointsHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_ObjectPointsHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ObjectPoints?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectPoints", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectPoints(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectPoints(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectPoints?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectPoints", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectPoints(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectPoints(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectPoints?(MR.Const_ObjectPointsHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPointsHolder_DynamicDowncastTo_MR_ObjectPoints", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectPointsHolder_DynamicDowncastTo_MR_ObjectPoints(MR.Const_ObjectPointsHolder._Underlying *_this);
            var ptr = __MR_ObjectPointsHolder_DynamicDowncastTo_MR_ObjectPoints(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_ObjectPointsHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// default value for maximum rendered points number
        public static unsafe int MaxRenderingPointsDefault
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_Get_MaxRenderingPointsDefault", ExactSpelling = true)]
                extern static int *__MR_ObjectPoints_Get_MaxRenderingPointsDefault();
                return *__MR_ObjectPoints_Get_MaxRenderingPointsDefault();
            }
        }

        /// recommended value for maximum rendered points number to disable discretization
        public static unsafe int MaxRenderingPointsUnlimited
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_Get_MaxRenderingPointsUnlimited", ExactSpelling = true)]
                extern static int *__MR_ObjectPoints_Get_MaxRenderingPointsUnlimited();
                return *__MR_ObjectPoints_Get_MaxRenderingPointsUnlimited();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectPoints() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_DefaultConstruct();
            _LateMakeShared(__MR_ObjectPoints_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectPoints::ObjectPoints`.
        public unsafe Const_ObjectPoints(MR._ByValue_ObjectPoints _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectPoints._Underlying *_other);
            _LateMakeShared(__MR_ObjectPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from constructor `MR::ObjectPoints::ObjectPoints`.
        /// Parameter `saveNormals` defaults to `true`.
        public unsafe Const_ObjectPoints(MR.Const_ObjectMesh objMesh, bool? saveNormals = null) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_Construct", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_Construct(MR.Const_ObjectMesh._Underlying *objMesh, byte *saveNormals);
            byte __deref_saveNormals = saveNormals.GetValueOrDefault() ? (byte)1 : (byte)0;
            _LateMakeShared(__MR_ObjectPoints_Construct(objMesh._UnderlyingPtr, saveNormals.HasValue ? &__deref_saveNormals : null));
        }

        /// Generated from method `MR::ObjectPoints::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectPoints_StaticTypeName();
            var __ret = __MR_ObjectPoints_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectPoints::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_typeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectPoints_typeName(_Underlying *_this);
            var __ret = __MR_ObjectPoints_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectPoints::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_ObjectPoints_StaticClassName();
            var __ret = __MR_ObjectPoints_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectPoints::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectPoints_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectPoints_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectPoints::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_ObjectPoints_StaticClassNameInPlural();
            var __ret = __MR_ObjectPoints_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectPoints::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectPoints_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectPoints_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectPoints::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectPoints_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectPoints_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectPoints::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectPoints_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectPoints_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectPoints::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_ObjectPoints_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_ObjectPoints_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectPoints::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_hasVisualRepresentation(_Underlying *_this);
            return __MR_ObjectPoints_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectPoints::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_hasModel", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_hasModel(_Underlying *_this);
            return __MR_ObjectPoints_hasModel(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectPoints::pointCloud`.
        public unsafe MR.Const_PointCloud PointCloud()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_pointCloud", ExactSpelling = true)]
            extern static MR.Const_PointCloud._UnderlyingShared *__MR_ObjectPoints_pointCloud(_Underlying *_this);
            return new(__MR_ObjectPoints_pointCloud(_UnderlyingPtr), is_owning: false);
        }

        /// \return the pair ( point cloud, selected points ) if any point is selected or full point cloud otherwise
        /// Generated from method `MR::ObjectPoints::pointCloudPart`.
        public unsafe MR.PointCloudPart PointCloudPart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_pointCloudPart", ExactSpelling = true)]
            extern static MR.PointCloudPart._Underlying *__MR_ObjectPoints_pointCloudPart(_Underlying *_this);
            return new(__MR_ObjectPoints_pointCloudPart(_UnderlyingPtr), is_owning: true);
        }

        /// gets current selected points
        /// Generated from method `MR::ObjectPoints::getSelectedPoints`.
        public unsafe MR.Const_VertBitSet GetSelectedPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getSelectedPoints", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_ObjectPoints_getSelectedPoints(_Underlying *_this);
            return new(__MR_ObjectPoints_getSelectedPoints(_UnderlyingPtr), is_owning: false);
        }

        /// returns selected points if any, otherwise returns all valid points
        /// Generated from method `MR::ObjectPoints::getSelectedPointsOrAll`.
        public unsafe MR.Const_VertBitSet GetSelectedPointsOrAll()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getSelectedPointsOrAll", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_ObjectPoints_getSelectedPointsOrAll(_Underlying *_this);
            return new(__MR_ObjectPoints_getSelectedPointsOrAll(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected vertices
        /// Generated from method `MR::ObjectPoints::getSelectedVerticesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedVerticesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getSelectedVerticesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectPoints_getSelectedVerticesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectPoints_getSelectedVerticesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectPoints::getSelectedVerticesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedVerticesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getSelectedVerticesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectPoints_getSelectedVerticesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectPoints_getSelectedVerticesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectPoints::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_ObjectPoints_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// returns per-point colors of the object
        /// Generated from method `MR::ObjectPoints::getVertsColorMap`.
        public unsafe MR.Const_VertColors GetVertsColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getVertsColorMap", ExactSpelling = true)]
            extern static MR.Const_VertColors._Underlying *__MR_ObjectPoints_getVertsColorMap(_Underlying *_this);
            return new(__MR_ObjectPoints_getVertsColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// get all visualize properties masks
        /// Generated from method `MR::ObjectPoints::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_ObjectPoints_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_ObjectPoints_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::ObjectPoints::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_ObjectPoints_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_ObjectPoints_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// returns size of points on screen in pixels
        /// Generated from method `MR::ObjectPoints::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getPointSize", ExactSpelling = true)]
            extern static float __MR_ObjectPoints_getPointSize(_Underlying *_this);
            return __MR_ObjectPoints_getPointSize(_UnderlyingPtr);
        }

        /// returns cached bounding box of this point object in world coordinates;
        /// if you need bounding box in local coordinates please call getBoundingBox()
        /// Generated from method `MR::ObjectPoints::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectPoints_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectPoints_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns cached information about the number of valid points
        /// Generated from method `MR::ObjectPoints::numValidPoints`.
        public unsafe ulong NumValidPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_numValidPoints", ExactSpelling = true)]
            extern static ulong __MR_ObjectPoints_numValidPoints(_Underlying *_this);
            return __MR_ObjectPoints_numValidPoints(_UnderlyingPtr);
        }

        /// returns cached information about the number of selected points
        /// Generated from method `MR::ObjectPoints::numSelectedPoints`.
        public unsafe ulong NumSelectedPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_numSelectedPoints", ExactSpelling = true)]
            extern static ulong __MR_ObjectPoints_numSelectedPoints(_Underlying *_this);
            return __MR_ObjectPoints_numSelectedPoints(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectPoints::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectPoints_heapBytes(_Underlying *_this);
            return __MR_ObjectPoints_heapBytes(_UnderlyingPtr);
        }

        /// returns rendering discretization
        /// display each `renderDiscretization`-th point only,
        /// starting from 0 index, total number is \ref numRenderingValidPoints()
        /// \detail defined by maximum rendered points number as:
        /// \ref numValidPoints() / \ref getMaxRenderingPoints() (rounded up)
        /// updated when setting `maxRenderingPoints` or changing the cloud (setting `DIRTY_FACE` flag)
        /// Generated from method `MR::ObjectPoints::getRenderDiscretization`.
        public unsafe int GetRenderDiscretization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getRenderDiscretization", ExactSpelling = true)]
            extern static int __MR_ObjectPoints_getRenderDiscretization(_Underlying *_this);
            return __MR_ObjectPoints_getRenderDiscretization(_UnderlyingPtr);
        }

        /// returns count of valid points that will be rendered
        /// Generated from method `MR::ObjectPoints::numRenderingValidPoints`.
        public unsafe ulong NumRenderingValidPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_numRenderingValidPoints", ExactSpelling = true)]
            extern static ulong __MR_ObjectPoints_numRenderingValidPoints(_Underlying *_this);
            return __MR_ObjectPoints_numRenderingValidPoints(_UnderlyingPtr);
        }

        /// returns maximal number of points that will be rendered
        /// if actual count of valid points is greater then the points will be sampled
        /// Generated from method `MR::ObjectPoints::getMaxRenderingPoints`.
        public unsafe int GetMaxRenderingPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getMaxRenderingPoints", ExactSpelling = true)]
            extern static int __MR_ObjectPoints_getMaxRenderingPoints(_Underlying *_this);
            return __MR_ObjectPoints_getMaxRenderingPoints(_UnderlyingPtr);
        }

        /// returns overriden file extension used to serialize point cloud inside this object, nullptr means defaultSerializePointsFormat()
        /// Generated from method `MR::ObjectPoints::serializeFormat`.
        public unsafe byte? SerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_serializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectPoints_serializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectPoints_serializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::ObjectPoints::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectPoints_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::ObjectPoints::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectPoints_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_ObjectPoints_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::ObjectPoints::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectPoints_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectPoints::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectPoints_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectPoints_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectPoints::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectPoints_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectPoints_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::ObjectPoints::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectPoints_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectPoints_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::ObjectPoints::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectPoints_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_ObjectPoints_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectPoints::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_ObjectPoints_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_ObjectPoints_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectPoints::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_ObjectPoints_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_ObjectPoints_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::ObjectPoints::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_ObjectPoints_getDirtyFlags(_Underlying *_this);
            return __MR_ObjectPoints_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::ObjectPoints::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_resetDirty", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_resetDirty(_Underlying *_this);
            __MR_ObjectPoints_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::ObjectPoints::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_ObjectPoints_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::ObjectPoints::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectPoints_getBoundingBox(_Underlying *_this);
            return __MR_ObjectPoints_getBoundingBox(_UnderlyingPtr);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::ObjectPoints::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectPoints_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::ObjectPoints::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isPickable", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectPoints_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::ObjectPoints::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_ObjectPoints_getColoringType(_Underlying *_this);
            return __MR_ObjectPoints_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::ObjectPoints::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getShininess", ExactSpelling = true)]
            extern static float __MR_ObjectPoints_getShininess(_Underlying *_this);
            return __MR_ObjectPoints_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::ObjectPoints::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_ObjectPoints_getSpecularStrength(_Underlying *_this);
            return __MR_ObjectPoints_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::ObjectPoints::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_ObjectPoints_getAmbientStrength(_Underlying *_this);
            return __MR_ObjectPoints_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::ObjectPoints::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_render", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_ObjectPoints_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::ObjectPoints::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_renderForPicker", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_ObjectPoints_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::ObjectPoints::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_renderUi", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_ObjectPoints_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectPoints::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_ObjectPoints_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectPoints::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_ObjectPoints_name(_Underlying *_this);
            return new(__MR_ObjectPoints_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::ObjectPoints::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectPoints_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectPoints_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::ObjectPoints::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_ObjectPoints_xfsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectPoints_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::ObjectPoints::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ObjectPoints_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectPoints_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::ObjectPoints::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectPoints_globalVisibilityMask(_Underlying *_this);
            return new(__MR_ObjectPoints_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::ObjectPoints::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectPoints_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::ObjectPoints::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isLocked(_Underlying *_this);
            return __MR_ObjectPoints_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::ObjectPoints::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isParentLocked(_Underlying *_this);
            return __MR_ObjectPoints_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::ObjectPoints::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isAncestor", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_ObjectPoints_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectPoints::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isSelected", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isSelected(_Underlying *_this);
            return __MR_ObjectPoints_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectPoints::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isAncillary(_Underlying *_this);
            return __MR_ObjectPoints_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::ObjectPoints::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isGlobalAncillary(_Underlying *_this);
            return __MR_ObjectPoints_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::ObjectPoints::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_isVisible", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectPoints_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectPoints::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectPoints_visibilityMask(_Underlying *_this);
            return new(__MR_ObjectPoints_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectPoints::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_resetRedrawFlag(_Underlying *_this);
            __MR_ObjectPoints_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::ObjectPoints::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectPoints_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectPoints_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::ObjectPoints::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectPoints_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectPoints_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::ObjectPoints::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectPoints_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectPoints_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::ObjectPoints::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_ObjectPoints_tags(_Underlying *_this);
            return new(__MR_ObjectPoints_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::ObjectPoints::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_sameModels", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_ObjectPoints_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::ObjectPoints::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_ObjectPoints_getModelHash(_Underlying *_this);
            return __MR_ObjectPoints_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectPoints::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectPoints_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectPoints_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// an object that stores a points
    /// Generated from class `MR::ObjectPoints`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectPointsHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class ObjectPoints : Const_ObjectPoints
    {
        internal unsafe ObjectPoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectPoints(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectPoints_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_ObjectPoints_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_ObjectPoints_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.ObjectPointsHolder(ObjectPoints self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_UpcastTo_MR_ObjectPointsHolder", ExactSpelling = true)]
            extern static MR.ObjectPointsHolder._Underlying *__MR_ObjectPoints_UpcastTo_MR_ObjectPointsHolder(_Underlying *_this);
            return MR.ObjectPointsHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectPoints_UpcastTo_MR_ObjectPointsHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ObjectPoints?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectPoints", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectPoints(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectPoints(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectPoints?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectPoints", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectPoints(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectPoints(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectPoints?(MR.ObjectPointsHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPointsHolder_DynamicDowncastTo_MR_ObjectPoints", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectPointsHolder_DynamicDowncastTo_MR_ObjectPoints(MR.ObjectPointsHolder._Underlying *_this);
            var ptr = __MR_ObjectPointsHolder_DynamicDowncastTo_MR_ObjectPoints(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.ObjectPointsHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectPoints() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_DefaultConstruct();
            _LateMakeShared(__MR_ObjectPoints_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectPoints::ObjectPoints`.
        public unsafe ObjectPoints(MR._ByValue_ObjectPoints _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectPoints._Underlying *_other);
            _LateMakeShared(__MR_ObjectPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from constructor `MR::ObjectPoints::ObjectPoints`.
        /// Parameter `saveNormals` defaults to `true`.
        public unsafe ObjectPoints(MR.Const_ObjectMesh objMesh, bool? saveNormals = null) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_Construct", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_Construct(MR.Const_ObjectMesh._Underlying *objMesh, byte *saveNormals);
            byte __deref_saveNormals = saveNormals.GetValueOrDefault() ? (byte)1 : (byte)0;
            _LateMakeShared(__MR_ObjectPoints_Construct(objMesh._UnderlyingPtr, saveNormals.HasValue ? &__deref_saveNormals : null));
        }

        /// Generated from method `MR::ObjectPoints::operator=`.
        public unsafe MR.ObjectPoints Assign(MR._ByValue_ObjectPoints _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectPoints._Underlying *__MR_ObjectPoints_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectPoints._Underlying *_other);
            return new(__MR_ObjectPoints_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// returns variable point cloud, if const point cloud is needed use `pointCloud()` instead
        /// Generated from method `MR::ObjectPoints::varPointCloud`.
        public unsafe MR.Const_PointCloud VarPointCloud()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_varPointCloud", ExactSpelling = true)]
            extern static MR.Const_PointCloud._UnderlyingShared *__MR_ObjectPoints_varPointCloud(_Underlying *_this);
            return new(__MR_ObjectPoints_varPointCloud(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectPoints::setPointCloud`.
        public unsafe void SetPointCloud(MR.Const_PointCloud pointCloud)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setPointCloud", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setPointCloud(_Underlying *_this, MR.Const_PointCloud._UnderlyingShared *pointCloud);
            __MR_ObjectPoints_setPointCloud(_UnderlyingPtr, pointCloud._UnderlyingSharedPtr);
        }

        /// sets given point cloud to this, and returns back previous mesh of this;
        /// does not touch selection
        /// Generated from method `MR::ObjectPoints::swapPointCloud`.
        public unsafe void SwapPointCloud(MR.PointCloud points)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_swapPointCloud", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_swapPointCloud(_Underlying *_this, MR.PointCloud._UnderlyingShared *points);
            __MR_ObjectPoints_swapPointCloud(_UnderlyingPtr, points._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ObjectPoints::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectPoints_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// Generated from method `MR::ObjectPoints::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_applyScale", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_applyScale(_Underlying *_this, float scaleFactor);
            __MR_ObjectPoints_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// sets current selected points
        /// Generated from method `MR::ObjectPoints::selectPoints`.
        public unsafe void SelectPoints(MR._ByValue_VertBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_selectPoints", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_selectPoints(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.VertBitSet._Underlying *newSelection);
            __MR_ObjectPoints_selectPoints(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// swaps current selected points with the argument
        /// Generated from method `MR::ObjectPoints::updateSelectedPoints`.
        public unsafe void UpdateSelectedPoints(MR.VertBitSet selection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_updateSelectedPoints", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_updateSelectedPoints(_Underlying *_this, MR.VertBitSet._Underlying *selection);
            __MR_ObjectPoints_updateSelectedPoints(_UnderlyingPtr, selection._UnderlyingPtr);
        }

        /// sets colors of selected vertices
        /// Generated from method `MR::ObjectPoints::setSelectedVerticesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedVerticesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setSelectedVerticesColor", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setSelectedVerticesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectPoints_setSelectedVerticesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectPoints::setSelectedVerticesColorsForAllViewports`.
        public unsafe void SetSelectedVerticesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setSelectedVerticesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setSelectedVerticesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectPoints_setSelectedVerticesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::ObjectPoints::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_ObjectPointsHolder other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_copyAllSolidColors(_Underlying *_this, MR.Const_ObjectPointsHolder._Underlying *other);
            __MR_ObjectPoints_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// sets per-point colors of the object
        /// Generated from method `MR::ObjectPoints::setVertsColorMap`.
        public unsafe void SetVertsColorMap(MR._ByValue_VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setVertsColorMap(_Underlying *_this, MR.Misc._PassBy vertsColorMap_pass_by, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectPoints_setVertsColorMap(_UnderlyingPtr, vertsColorMap.PassByMode, vertsColorMap.Value is not null ? vertsColorMap.Value._UnderlyingPtr : null);
        }

        /// swaps per-point colors of the object with given argument
        /// Generated from method `MR::ObjectPoints::updateVertsColorMap`.
        public unsafe void UpdateVertsColorMap(MR.VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_updateVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_updateVertsColorMap(_Underlying *_this, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectPoints_updateVertsColorMap(_UnderlyingPtr, vertsColorMap._UnderlyingPtr);
        }

        /// copies point colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectPoints::copyColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyColors(MR.Const_ObjectPointsHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_copyColors", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_copyColors(_Underlying *_this, MR.Const_ObjectPointsHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectPoints_copyColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// sets size of points on screen in pixels
        /// Generated from method `MR::ObjectPoints::setPointSize`.
        public unsafe void SetPointSize(float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setPointSize", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setPointSize(_Underlying *_this, float size);
            __MR_ObjectPoints_setPointSize(_UnderlyingPtr, size);
        }

        /// sets maximal number of points that will be rendered
        /// \sa \ref getRenderDiscretization, \ref MaxRenderingPointsDefault, \ref MaxRenderingPointsUnlimited
        /// Generated from method `MR::ObjectPoints::setMaxRenderingPoints`.
        public unsafe void SetMaxRenderingPoints(int val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setMaxRenderingPoints", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setMaxRenderingPoints(_Underlying *_this, int val);
            __MR_ObjectPoints_setMaxRenderingPoints(_UnderlyingPtr, val);
        }

        /// overrides file extension used to serialize point cloud inside this object: must start from '.',
        /// nullptr means serialize in defaultSerializePointsFormat()
        /// Generated from method `MR::ObjectPoints::setSerializeFormat`.
        public unsafe void SetSerializeFormat(byte? newFormat)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setSerializeFormat", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setSerializeFormat(_Underlying *_this, byte *newFormat);
            byte __deref_newFormat = newFormat.GetValueOrDefault();
            __MR_ObjectPoints_setSerializeFormat(_UnderlyingPtr, newFormat.HasValue ? &__deref_newFormat : null);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::ObjectPoints::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_resetFrontColor(_Underlying *_this);
            __MR_ObjectPoints_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::ObjectPoints::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_resetColors", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_resetColors(_Underlying *_this);
            __MR_ObjectPoints_resetColors(_UnderlyingPtr);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectPoints::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::ObjectPoints::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectPoints::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::ObjectPoints::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_ObjectPoints_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::ObjectPoints::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectPoints::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_ObjectPoints_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectPoints::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectPoints_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::ObjectPoints::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectPoints_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::ObjectPoints::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setBackColor", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_ObjectPoints_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectPoints::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_ObjectPoints_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectPoints::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_ObjectPoints_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::ObjectPoints::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setPickable", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::ObjectPoints::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setColoringType", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_ObjectPoints_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::ObjectPoints::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setShininess", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setShininess(_Underlying *_this, float shininess);
            __MR_ObjectPoints_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::ObjectPoints::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_ObjectPoints_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::ObjectPoints::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_ObjectPoints_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectPoints::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_ObjectPoints_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectPoints::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setName", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_ObjectPoints_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::ObjectPoints::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setXf", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectPoints_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::ObjectPoints::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_resetXf", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_ObjectPoints_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::ObjectPoints::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_ObjectPoints_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectPoints::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setWorldXf", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectPoints_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::ObjectPoints::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectPoints::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setLocked", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setLocked(_Underlying *_this, byte on);
            __MR_ObjectPoints_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectPoints::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setParentLocked", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setParentLocked(_Underlying *_this, byte lock_);
            __MR_ObjectPoints_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::ObjectPoints::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_detachFromParent(_Underlying *_this);
            return __MR_ObjectPoints_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::ObjectPoints::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_addChild", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectPoints_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::ObjectPoints::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_ObjectPoints_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::ObjectPoints::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_removeAllChildren(_Underlying *_this);
            __MR_ObjectPoints_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::ObjectPoints::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_sortChildren", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_sortChildren(_Underlying *_this);
            __MR_ObjectPoints_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::ObjectPoints::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_select", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_select(_Underlying *_this, byte on);
            return __MR_ObjectPoints_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::ObjectPoints::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setAncillary", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setAncillary(_Underlying *_this, byte ancillary);
            __MR_ObjectPoints_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::ObjectPoints::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setVisible", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectPoints::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectPoints_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::ObjectPoints::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_swap", ExactSpelling = true)]
            extern static void __MR_ObjectPoints_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_ObjectPoints_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::ObjectPoints::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_addTag", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectPoints_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::ObjectPoints::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectPoints_removeTag", ExactSpelling = true)]
            extern static byte __MR_ObjectPoints_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectPoints_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectPoints` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectPoints`/`Const_ObjectPoints` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectPoints
    {
        internal readonly Const_ObjectPoints? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectPoints() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectPoints(MR.Misc._Moved<ObjectPoints> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectPoints(MR.Misc._Moved<ObjectPoints> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectPoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectPoints`/`Const_ObjectPoints` directly.
    public class _InOptMut_ObjectPoints
    {
        public ObjectPoints? Opt;

        public _InOptMut_ObjectPoints() {}
        public _InOptMut_ObjectPoints(ObjectPoints value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectPoints(ObjectPoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectPoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectPoints`/`Const_ObjectPoints` to pass it to the function.
    public class _InOptConst_ObjectPoints
    {
        public Const_ObjectPoints? Opt;

        public _InOptConst_ObjectPoints() {}
        public _InOptConst_ObjectPoints(Const_ObjectPoints value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectPoints(Const_ObjectPoints value) {return new(value);}
    }

    /// constructs new ObjectPoints containing the union of valid points from all input objects
    /// Generated from function `MR::merge`.
    public static unsafe MR.Misc._Moved<MR.ObjectPoints> Merge(MR.Std.Const_Vector_StdSharedPtrMRObjectPoints objsPoints)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_merge_1_std_vector_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
        extern static MR.ObjectPoints._UnderlyingShared *__MR_merge_1_std_vector_std_shared_ptr_MR_ObjectPoints(MR.Std.Const_Vector_StdSharedPtrMRObjectPoints._Underlying *objsPoints);
        return MR.Misc.Move(new MR.ObjectPoints(__MR_merge_1_std_vector_std_shared_ptr_MR_ObjectPoints(objsPoints._UnderlyingPtr), is_owning: true));
    }

    /// constructs new ObjectPoints containing the region of data from input object
    /// does not copy selection
    /// Generated from function `MR::cloneRegion`.
    public static unsafe MR.Misc._Moved<MR.ObjectPoints> CloneRegion(MR.Const_ObjectPoints objPoints, MR.Const_VertBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cloneRegion_2_std_shared_ptr_MR_ObjectPoints", ExactSpelling = true)]
        extern static MR.ObjectPoints._UnderlyingShared *__MR_cloneRegion_2_std_shared_ptr_MR_ObjectPoints(MR.Const_ObjectPoints._UnderlyingShared *objPoints, MR.Const_VertBitSet._Underlying *region);
        return MR.Misc.Move(new MR.ObjectPoints(__MR_cloneRegion_2_std_shared_ptr_MR_ObjectPoints(objPoints._UnderlyingSharedPtr, region._UnderlyingPtr), is_owning: true));
    }

    /// constructs new ObjectPoints containing the packed version of input points,
    /// \param newValidVerts if given, then use them instead of valid points from pts
    /// \return nullptr if the operation was cancelled
    /// Generated from function `MR::pack`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.ObjectPoints> Pack(MR.Const_ObjectPoints pts, MR.Reorder reorder, MR.VertBitSet? newValidVerts = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pack", ExactSpelling = true)]
        extern static MR.ObjectPoints._UnderlyingShared *__MR_pack(MR.Const_ObjectPoints._Underlying *pts, MR.Reorder reorder, MR.VertBitSet._Underlying *newValidVerts, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.ObjectPoints(__MR_pack(pts._UnderlyingPtr, reorder, newValidVerts is not null ? newValidVerts._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }
}
