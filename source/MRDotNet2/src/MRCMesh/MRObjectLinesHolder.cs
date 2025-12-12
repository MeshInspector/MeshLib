public static partial class MR
{
    public enum LinesVisualizePropertyType : int
    {
        Points = 0,
        Smooth = 1,
        Dashed = 2,
        Count = 3,
    }

    /// an object that stores a lines
    /// Generated from class `MR::ObjectLinesHolder`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLines`
    /// This is the const half of the class.
    public class Const_ObjectLinesHolder : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectLinesHolder_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectLinesHolder_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectLinesHolder_UseCount();
                return __MR_std_shared_ptr_MR_ObjectLinesHolder_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectLinesHolder(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectLinesHolder_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectLinesHolder_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectLinesHolder(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectLinesHolder _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectLinesHolder_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectLinesHolder_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectLinesHolder_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectLinesHolder_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectLinesHolder_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectLinesHolder_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectLinesHolder() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_ObjectLinesHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_ObjectLinesHolder_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectLinesHolder_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_ObjectLinesHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_ObjectLinesHolder_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectLinesHolder_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_ObjectLinesHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_ObjectLinesHolder_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectLinesHolder_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ObjectLinesHolder?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectLinesHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectLinesHolder(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectLinesHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectLinesHolder?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectLinesHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectLinesHolder(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectLinesHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectLinesHolder() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectLinesHolder._Underlying *__MR_ObjectLinesHolder_DefaultConstruct();
            _LateMakeShared(__MR_ObjectLinesHolder_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectLinesHolder::ObjectLinesHolder`.
        public unsafe Const_ObjectLinesHolder(MR._ByValue_ObjectLinesHolder _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectLinesHolder._Underlying *__MR_ObjectLinesHolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectLinesHolder._Underlying *_other);
            _LateMakeShared(__MR_ObjectLinesHolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectLinesHolder::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectLinesHolder_StaticTypeName();
            var __ret = __MR_ObjectLinesHolder_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectLinesHolder::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_typeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectLinesHolder_typeName(_Underlying *_this);
            var __ret = __MR_ObjectLinesHolder_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectLinesHolder::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_hasVisualRepresentation(_Underlying *_this);
            return __MR_ObjectLinesHolder_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectLinesHolder::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_hasModel", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_hasModel(_Underlying *_this);
            return __MR_ObjectLinesHolder_hasModel(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectLinesHolder::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectLinesHolder_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectLinesHolder_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectLinesHolder::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectLinesHolder_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectLinesHolder_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectLinesHolder::polyline`.
        public unsafe MR.Const_Polyline3 Polyline()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_polyline", ExactSpelling = true)]
            extern static MR.Const_Polyline3._UnderlyingShared *__MR_ObjectLinesHolder_polyline(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_polyline(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectLinesHolder::getDashPattern`.
        /// Parameter `vpId` defaults to `{}`.
        public unsafe MR.Const_Vector4_UnsignedChar GetDashPattern(MR._InOpt_ViewportId vpId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getDashPattern", ExactSpelling = true)]
            extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_ObjectLinesHolder_getDashPattern(_Underlying *_this, MR.ViewportId *vpId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectLinesHolder_getDashPattern(_UnderlyingPtr, vpId.HasValue ? &vpId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::ObjectLinesHolder::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getLineWidth", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_getLineWidth(_Underlying *_this);
            return __MR_ObjectLinesHolder_getLineWidth(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectLinesHolder::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getPointSize", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_getPointSize(_Underlying *_this);
            return __MR_ObjectLinesHolder_getPointSize(_UnderlyingPtr);
        }

        /// returns per-vertex colors of the object
        /// Generated from method `MR::ObjectLinesHolder::getVertsColorMap`.
        public unsafe MR.Const_VertColors GetVertsColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getVertsColorMap", ExactSpelling = true)]
            extern static MR.Const_VertColors._Underlying *__MR_ObjectLinesHolder_getVertsColorMap(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_getVertsColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectLinesHolder::getLinesColorMap`.
        public unsafe MR.Const_UndirectedEdgeColors GetLinesColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getLinesColorMap", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeColors._Underlying *__MR_ObjectLinesHolder_getLinesColorMap(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_getLinesColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectLinesHolder::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_ObjectLinesHolder_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// get all visualize properties masks
        /// Generated from method `MR::ObjectLinesHolder::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_ObjectLinesHolder_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_ObjectLinesHolder_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::ObjectLinesHolder::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_ObjectLinesHolder_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_ObjectLinesHolder_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// returns cached bounding box of this point object in world coordinates;
        /// if you need bounding box in local coordinates please call getBoundingBox()
        /// Generated from method `MR::ObjectLinesHolder::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectLinesHolder_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectLinesHolder_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectLinesHolder::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectLinesHolder_heapBytes(_Underlying *_this);
            return __MR_ObjectLinesHolder_heapBytes(_UnderlyingPtr);
        }

        /// returns cached average edge length
        /// Generated from method `MR::ObjectLinesHolder::avgEdgeLen`.
        public unsafe float AvgEdgeLen()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_avgEdgeLen", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_avgEdgeLen(_Underlying *_this);
            return __MR_ObjectLinesHolder_avgEdgeLen(_UnderlyingPtr);
        }

        /// returns cached information about the number of undirected edges in the polyline
        /// Generated from method `MR::ObjectLinesHolder::numUndirectedEdges`.
        public unsafe ulong NumUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_numUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectLinesHolder_numUndirectedEdges(_Underlying *_this);
            return __MR_ObjectLinesHolder_numUndirectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of components in the polyline
        /// Generated from method `MR::ObjectLinesHolder::numComponents`.
        public unsafe ulong NumComponents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_numComponents", ExactSpelling = true)]
            extern static ulong __MR_ObjectLinesHolder_numComponents(_Underlying *_this);
            return __MR_ObjectLinesHolder_numComponents(_UnderlyingPtr);
        }

        /// return cached total length
        /// Generated from method `MR::ObjectLinesHolder::totalLength`.
        public unsafe float TotalLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_totalLength", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_totalLength(_Underlying *_this);
            return __MR_ObjectLinesHolder_totalLength(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectLinesHolder::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_ObjectLinesHolder_StaticClassName();
            var __ret = __MR_ObjectLinesHolder_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectLinesHolder::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectLinesHolder_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectLinesHolder_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectLinesHolder::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_ObjectLinesHolder_StaticClassNameInPlural();
            var __ret = __MR_ObjectLinesHolder_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectLinesHolder::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectLinesHolder_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectLinesHolder_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::ObjectLinesHolder::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectLinesHolder_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::ObjectLinesHolder::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectLinesHolder_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::ObjectLinesHolder::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectLinesHolder_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectLinesHolder::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectLinesHolder_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectLinesHolder_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectLinesHolder::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectLinesHolder_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectLinesHolder_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::ObjectLinesHolder::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectLinesHolder_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::ObjectLinesHolder::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectLinesHolder_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_ObjectLinesHolder_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectLinesHolder::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_ObjectLinesHolder_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_ObjectLinesHolder_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectLinesHolder::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_ObjectLinesHolder_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::ObjectLinesHolder::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_ObjectLinesHolder_getDirtyFlags(_Underlying *_this);
            return __MR_ObjectLinesHolder_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::ObjectLinesHolder::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_resetDirty", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_resetDirty(_Underlying *_this);
            __MR_ObjectLinesHolder_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::ObjectLinesHolder::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_ObjectLinesHolder_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::ObjectLinesHolder::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectLinesHolder_getBoundingBox(_Underlying *_this);
            return __MR_ObjectLinesHolder_getBoundingBox(_UnderlyingPtr);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::ObjectLinesHolder::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectLinesHolder_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::ObjectLinesHolder::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isPickable", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectLinesHolder_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::ObjectLinesHolder::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_ObjectLinesHolder_getColoringType(_Underlying *_this);
            return __MR_ObjectLinesHolder_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::ObjectLinesHolder::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getShininess", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_getShininess(_Underlying *_this);
            return __MR_ObjectLinesHolder_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::ObjectLinesHolder::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_getSpecularStrength(_Underlying *_this);
            return __MR_ObjectLinesHolder_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::ObjectLinesHolder::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_ObjectLinesHolder_getAmbientStrength(_Underlying *_this);
            return __MR_ObjectLinesHolder_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::ObjectLinesHolder::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_render", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_ObjectLinesHolder_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::ObjectLinesHolder::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_renderForPicker", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_ObjectLinesHolder_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::ObjectLinesHolder::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_renderUi", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_ObjectLinesHolder_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::ObjectLinesHolder::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_ObjectLinesHolder_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_ObjectLinesHolder_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectLinesHolder::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_ObjectLinesHolder_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectLinesHolder::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_ObjectLinesHolder_name(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::ObjectLinesHolder::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectLinesHolder_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectLinesHolder_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::ObjectLinesHolder::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_ObjectLinesHolder_xfsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::ObjectLinesHolder::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ObjectLinesHolder_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectLinesHolder_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::ObjectLinesHolder::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectLinesHolder_globalVisibilityMask(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::ObjectLinesHolder::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectLinesHolder_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::ObjectLinesHolder::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isLocked(_Underlying *_this);
            return __MR_ObjectLinesHolder_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::ObjectLinesHolder::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isParentLocked(_Underlying *_this);
            return __MR_ObjectLinesHolder_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::ObjectLinesHolder::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isAncestor", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_ObjectLinesHolder_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectLinesHolder::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isSelected", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isSelected(_Underlying *_this);
            return __MR_ObjectLinesHolder_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectLinesHolder::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isAncillary(_Underlying *_this);
            return __MR_ObjectLinesHolder_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::ObjectLinesHolder::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isGlobalAncillary(_Underlying *_this);
            return __MR_ObjectLinesHolder_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::ObjectLinesHolder::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_isVisible", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectLinesHolder_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectLinesHolder::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectLinesHolder_visibilityMask(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectLinesHolder::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_resetRedrawFlag(_Underlying *_this);
            __MR_ObjectLinesHolder_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::ObjectLinesHolder::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectLinesHolder_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectLinesHolder_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::ObjectLinesHolder::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectLinesHolder_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectLinesHolder_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::ObjectLinesHolder::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectLinesHolder_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectLinesHolder_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::ObjectLinesHolder::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_ObjectLinesHolder_tags(_Underlying *_this);
            return new(__MR_ObjectLinesHolder_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::ObjectLinesHolder::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_sameModels", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_ObjectLinesHolder_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::ObjectLinesHolder::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_ObjectLinesHolder_getModelHash(_Underlying *_this);
            return __MR_ObjectLinesHolder_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectLinesHolder::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectLinesHolder_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectLinesHolder_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// an object that stores a lines
    /// Generated from class `MR::ObjectLinesHolder`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLines`
    /// This is the non-const half of the class.
    public class ObjectLinesHolder : Const_ObjectLinesHolder
    {
        internal unsafe ObjectLinesHolder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectLinesHolder(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(ObjectLinesHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectLinesHolder_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectLinesHolder_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(ObjectLinesHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_ObjectLinesHolder_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectLinesHolder_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(ObjectLinesHolder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_ObjectLinesHolder_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectLinesHolder_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ObjectLinesHolder?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectLinesHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectLinesHolder(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectLinesHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectLinesHolder?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectLinesHolder", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectLinesHolder(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectLinesHolder(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectLinesHolder() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectLinesHolder._Underlying *__MR_ObjectLinesHolder_DefaultConstruct();
            _LateMakeShared(__MR_ObjectLinesHolder_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectLinesHolder::ObjectLinesHolder`.
        public unsafe ObjectLinesHolder(MR._ByValue_ObjectLinesHolder _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectLinesHolder._Underlying *__MR_ObjectLinesHolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectLinesHolder._Underlying *_other);
            _LateMakeShared(__MR_ObjectLinesHolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectLinesHolder::operator=`.
        public unsafe MR.ObjectLinesHolder Assign(MR._ByValue_ObjectLinesHolder _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectLinesHolder._Underlying *__MR_ObjectLinesHolder_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectLinesHolder._Underlying *_other);
            return new(__MR_ObjectLinesHolder_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectLinesHolder::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_applyScale", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_applyScale(_Underlying *_this, float scaleFactor);
            __MR_ObjectLinesHolder_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// Generated from method `MR::ObjectLinesHolder::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectLinesHolder_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// specify dash pattern in pixels
        /// [0] - dash
        /// [1] - space
        /// [2] - dash
        /// [3] - space
        /// Generated from method `MR::ObjectLinesHolder::setDashPattern`.
        /// Parameter `vpId` defaults to `{}`.
        public unsafe void SetDashPattern(MR.Const_Vector4_UnsignedChar pattern, MR._InOpt_ViewportId vpId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setDashPattern", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setDashPattern(_Underlying *_this, MR.Const_Vector4_UnsignedChar._Underlying *pattern, MR.ViewportId *vpId);
            __MR_ObjectLinesHolder_setDashPattern(_UnderlyingPtr, pattern._UnderlyingPtr, vpId.HasValue ? &vpId.Object : null);
        }

        /// Generated from method `MR::ObjectLinesHolder::setLineWidth`.
        public unsafe void SetLineWidth(float width)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setLineWidth", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setLineWidth(_Underlying *_this, float width);
            __MR_ObjectLinesHolder_setLineWidth(_UnderlyingPtr, width);
        }

        /// Generated from method `MR::ObjectLinesHolder::setPointSize`.
        public unsafe void SetPointSize(float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setPointSize", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setPointSize(_Underlying *_this, float size);
            __MR_ObjectLinesHolder_setPointSize(_UnderlyingPtr, size);
        }

        /// sets per-vertex colors of the object
        /// Generated from method `MR::ObjectLinesHolder::setVertsColorMap`.
        public unsafe void SetVertsColorMap(MR._ByValue_VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setVertsColorMap(_Underlying *_this, MR.Misc._PassBy vertsColorMap_pass_by, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectLinesHolder_setVertsColorMap(_UnderlyingPtr, vertsColorMap.PassByMode, vertsColorMap.Value is not null ? vertsColorMap.Value._UnderlyingPtr : null);
        }

        /// swaps per-vertex colors of the object with given argument
        /// Generated from method `MR::ObjectLinesHolder::updateVertsColorMap`.
        public unsafe void UpdateVertsColorMap(MR.VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_updateVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_updateVertsColorMap(_Underlying *_this, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectLinesHolder_updateVertsColorMap(_UnderlyingPtr, vertsColorMap._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectLinesHolder::setLinesColorMap`.
        public unsafe void SetLinesColorMap(MR._ByValue_UndirectedEdgeColors linesColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setLinesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setLinesColorMap(_Underlying *_this, MR.Misc._PassBy linesColorMap_pass_by, MR.UndirectedEdgeColors._Underlying *linesColorMap);
            __MR_ObjectLinesHolder_setLinesColorMap(_UnderlyingPtr, linesColorMap.PassByMode, linesColorMap.Value is not null ? linesColorMap.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectLinesHolder::updateLinesColorMap`.
        public unsafe void UpdateLinesColorMap(MR.UndirectedEdgeColors updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_updateLinesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_updateLinesColorMap(_Underlying *_this, MR.UndirectedEdgeColors._Underlying *updated);
            __MR_ObjectLinesHolder_updateLinesColorMap(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// copies vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectLinesHolder::copyColors`.
        public unsafe void CopyColors(MR.Const_ObjectLinesHolder src, MR.Const_VertMap thisToSrc)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_copyColors", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_copyColors(_Underlying *_this, MR.Const_ObjectLinesHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc);
            __MR_ObjectLinesHolder_copyColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::ObjectLinesHolder::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_resetFrontColor(_Underlying *_this);
            __MR_ObjectLinesHolder_resetFrontColor(_UnderlyingPtr);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectLinesHolder::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::ObjectLinesHolder::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectLinesHolder::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::ObjectLinesHolder::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_ObjectLinesHolder_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::ObjectLinesHolder::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_ObjectLinesHolder_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::ObjectLinesHolder::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectLinesHolder::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_ObjectLinesHolder_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectLinesHolder::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectLinesHolder_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::ObjectLinesHolder::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectLinesHolder_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::ObjectLinesHolder::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setBackColor", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_ObjectLinesHolder_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectLinesHolder::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_ObjectLinesHolder_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectLinesHolder::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_ObjectLinesHolder_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::ObjectLinesHolder::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setPickable", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::ObjectLinesHolder::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setColoringType", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_ObjectLinesHolder_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::ObjectLinesHolder::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setShininess", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setShininess(_Underlying *_this, float shininess);
            __MR_ObjectLinesHolder_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::ObjectLinesHolder::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_ObjectLinesHolder_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::ObjectLinesHolder::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_ObjectLinesHolder_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectLinesHolder::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_ObjectLinesHolder_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::ObjectLinesHolder::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_resetColors", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_resetColors(_Underlying *_this);
            __MR_ObjectLinesHolder_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectLinesHolder::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setName", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_ObjectLinesHolder_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::ObjectLinesHolder::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setXf", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectLinesHolder_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::ObjectLinesHolder::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_resetXf", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_ObjectLinesHolder_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::ObjectLinesHolder::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_ObjectLinesHolder_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectLinesHolder::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setWorldXf", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectLinesHolder_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::ObjectLinesHolder::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectLinesHolder::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setLocked", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setLocked(_Underlying *_this, byte on);
            __MR_ObjectLinesHolder_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectLinesHolder::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setParentLocked", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setParentLocked(_Underlying *_this, byte lock_);
            __MR_ObjectLinesHolder_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::ObjectLinesHolder::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_detachFromParent(_Underlying *_this);
            return __MR_ObjectLinesHolder_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::ObjectLinesHolder::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_addChild", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectLinesHolder_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::ObjectLinesHolder::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_ObjectLinesHolder_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::ObjectLinesHolder::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_removeAllChildren(_Underlying *_this);
            __MR_ObjectLinesHolder_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::ObjectLinesHolder::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_sortChildren", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_sortChildren(_Underlying *_this);
            __MR_ObjectLinesHolder_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::ObjectLinesHolder::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_select", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_select(_Underlying *_this, byte on);
            return __MR_ObjectLinesHolder_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::ObjectLinesHolder::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setAncillary", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setAncillary(_Underlying *_this, byte ancillary);
            __MR_ObjectLinesHolder_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::ObjectLinesHolder::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setVisible", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectLinesHolder::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectLinesHolder_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::ObjectLinesHolder::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_swap", ExactSpelling = true)]
            extern static void __MR_ObjectLinesHolder_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_ObjectLinesHolder_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::ObjectLinesHolder::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_addTag", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectLinesHolder_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::ObjectLinesHolder::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_removeTag", ExactSpelling = true)]
            extern static byte __MR_ObjectLinesHolder_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectLinesHolder_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectLinesHolder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectLinesHolder`/`Const_ObjectLinesHolder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectLinesHolder
    {
        internal readonly Const_ObjectLinesHolder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectLinesHolder() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectLinesHolder(MR.Misc._Moved<ObjectLinesHolder> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectLinesHolder(MR.Misc._Moved<ObjectLinesHolder> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectLinesHolder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectLinesHolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectLinesHolder`/`Const_ObjectLinesHolder` directly.
    public class _InOptMut_ObjectLinesHolder
    {
        public ObjectLinesHolder? Opt;

        public _InOptMut_ObjectLinesHolder() {}
        public _InOptMut_ObjectLinesHolder(ObjectLinesHolder value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectLinesHolder(ObjectLinesHolder value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectLinesHolder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectLinesHolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectLinesHolder`/`Const_ObjectLinesHolder` to pass it to the function.
    public class _InOptConst_ObjectLinesHolder
    {
        public Const_ObjectLinesHolder? Opt;

        public _InOptConst_ObjectLinesHolder() {}
        public _InOptConst_ObjectLinesHolder(Const_ObjectLinesHolder value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectLinesHolder(Const_ObjectLinesHolder value) {return new(value);}
    }
}
