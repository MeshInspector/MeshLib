public static partial class MR
{
    /// an object that stores a g-code
    /// Generated from class `MR::ObjectGcode`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectLinesHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_ObjectGcode : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectGcode_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectGcode_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectGcode_UseCount();
                return __MR_std_shared_ptr_MR_ObjectGcode_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectGcode_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectGcode_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectGcode(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectGcode_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectGcode_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectGcode_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectGcode_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectGcode(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectGcode _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectGcode_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectGcode_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectGcode_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectGcode_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectGcode_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectGcode_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectGcode_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectGcode() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_ObjectGcode_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_ObjectGcode_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_ObjectGcode_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_ObjectLinesHolder(Const_ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_ObjectLinesHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectLinesHolder._Underlying *__MR_ObjectGcode_UpcastTo_MR_ObjectLinesHolder(_Underlying *_this);
            return MR.Const_ObjectLinesHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_ObjectLinesHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ObjectGcode?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectGcode", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectGcode(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectGcode(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectGcode?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectGcode", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectGcode(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectGcode(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectGcode?(MR.Const_ObjectLinesHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_DynamicDowncastTo_MR_ObjectGcode", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectLinesHolder_DynamicDowncastTo_MR_ObjectGcode(MR.Const_ObjectLinesHolder._Underlying *_this);
            var ptr = __MR_ObjectLinesHolder_DynamicDowncastTo_MR_ObjectGcode(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_ObjectLinesHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectGcode() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectGcode._Underlying *__MR_ObjectGcode_DefaultConstruct();
            _LateMakeShared(__MR_ObjectGcode_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectGcode::ObjectGcode`.
        public unsafe Const_ObjectGcode(MR._ByValue_ObjectGcode _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectGcode._Underlying *__MR_ObjectGcode_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectGcode._Underlying *_other);
            _LateMakeShared(__MR_ObjectGcode_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectGcode::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectGcode_StaticTypeName();
            var __ret = __MR_ObjectGcode_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectGcode::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_typeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectGcode_typeName(_Underlying *_this);
            var __ret = __MR_ObjectGcode_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectGcode::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_ObjectGcode_StaticClassName();
            var __ret = __MR_ObjectGcode_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectGcode::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectGcode_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectGcode_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectGcode::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_ObjectGcode_StaticClassNameInPlural();
            var __ret = __MR_ObjectGcode_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectGcode::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectGcode_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectGcode_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectGcode::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectGcode_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectGcode_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectGcode::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectGcode_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectGcode_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectGcode::getCNCMachineSettings`.
        public unsafe MR.Const_CNCMachineSettings GetCNCMachineSettings()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getCNCMachineSettings", ExactSpelling = true)]
            extern static MR.Const_CNCMachineSettings._Underlying *__MR_ObjectGcode_getCNCMachineSettings(_Underlying *_this);
            return new(__MR_ObjectGcode_getCNCMachineSettings(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::gcodeSource`.
        public unsafe MR.Std.Const_Vector_StdString GcodeSource()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_gcodeSource", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_StdString._UnderlyingShared *__MR_ObjectGcode_gcodeSource(_Underlying *_this);
            return new(__MR_ObjectGcode_gcodeSource(_UnderlyingPtr), is_owning: false);
        }

        // get action list (produced from g-code source)
        /// Generated from method `MR::ObjectGcode::actionList`.
        public unsafe MR.Std.Const_Vector_MRGcodeProcessorMoveAction ActionList()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_actionList", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRGcodeProcessorMoveAction._Underlying *__MR_ObjectGcode_actionList(_Underlying *_this);
            return new(__MR_ObjectGcode_actionList(_UnderlyingPtr), is_owning: false);
        }

        // get mapping of tool path polyline segment id to source line number of g-code source
        /// Generated from method `MR::ObjectGcode::segmentToSourceLineMap`.
        public unsafe MR.Std.Const_Vector_Int SegmentToSourceLineMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_segmentToSourceLineMap", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_Int._Underlying *__MR_ObjectGcode_segmentToSourceLineMap(_Underlying *_this);
            return new(__MR_ObjectGcode_segmentToSourceLineMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_ObjectGcode_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_ObjectGcode_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectGcode::isFeedrateGradient`.
        public unsafe bool IsFeedrateGradient()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isFeedrateGradient", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isFeedrateGradient(_Underlying *_this);
            return __MR_ObjectGcode_isFeedrateGradient(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectGcode::getIdleColor`.
        public unsafe MR.Const_Color GetIdleColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getIdleColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectGcode_getIdleColor(_Underlying *_this);
            return new(__MR_ObjectGcode_getIdleColor(_UnderlyingPtr), is_owning: false);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectGcode::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectGcode_heapBytes(_Underlying *_this);
            return __MR_ObjectGcode_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectGcode::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_hasVisualRepresentation(_Underlying *_this);
            return __MR_ObjectGcode_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectGcode::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_hasModel", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_hasModel(_Underlying *_this);
            return __MR_ObjectGcode_hasModel(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectGcode::polyline`.
        public unsafe MR.Const_Polyline3 Polyline()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_polyline", ExactSpelling = true)]
            extern static MR.Const_Polyline3._UnderlyingShared *__MR_ObjectGcode_polyline(_Underlying *_this);
            return new(__MR_ObjectGcode_polyline(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::getDashPattern`.
        /// Parameter `vpId` defaults to `{}`.
        public unsafe MR.Const_Vector4_UnsignedChar GetDashPattern(MR._InOpt_ViewportId vpId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getDashPattern", ExactSpelling = true)]
            extern static MR.Const_Vector4_UnsignedChar._Underlying *__MR_ObjectGcode_getDashPattern(_Underlying *_this, MR.ViewportId *vpId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectGcode_getDashPattern(_UnderlyingPtr, vpId.HasValue ? &vpId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getLineWidth", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_getLineWidth(_Underlying *_this);
            return __MR_ObjectGcode_getLineWidth(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectGcode::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getPointSize", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_getPointSize(_Underlying *_this);
            return __MR_ObjectGcode_getPointSize(_UnderlyingPtr);
        }

        /// returns per-vertex colors of the object
        /// Generated from method `MR::ObjectGcode::getVertsColorMap`.
        public unsafe MR.Const_VertColors GetVertsColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getVertsColorMap", ExactSpelling = true)]
            extern static MR.Const_VertColors._Underlying *__MR_ObjectGcode_getVertsColorMap(_Underlying *_this);
            return new(__MR_ObjectGcode_getVertsColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::getLinesColorMap`.
        public unsafe MR.Const_UndirectedEdgeColors GetLinesColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getLinesColorMap", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeColors._Underlying *__MR_ObjectGcode_getLinesColorMap(_Underlying *_this);
            return new(__MR_ObjectGcode_getLinesColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_ObjectGcode_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// get all visualize properties masks
        /// Generated from method `MR::ObjectGcode::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_ObjectGcode_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_ObjectGcode_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::ObjectGcode::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_ObjectGcode_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_ObjectGcode_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// returns cached bounding box of this point object in world coordinates;
        /// if you need bounding box in local coordinates please call getBoundingBox()
        /// Generated from method `MR::ObjectGcode::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectGcode_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectGcode_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns cached average edge length
        /// Generated from method `MR::ObjectGcode::avgEdgeLen`.
        public unsafe float AvgEdgeLen()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_avgEdgeLen", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_avgEdgeLen(_Underlying *_this);
            return __MR_ObjectGcode_avgEdgeLen(_UnderlyingPtr);
        }

        /// returns cached information about the number of undirected edges in the polyline
        /// Generated from method `MR::ObjectGcode::numUndirectedEdges`.
        public unsafe ulong NumUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_numUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectGcode_numUndirectedEdges(_Underlying *_this);
            return __MR_ObjectGcode_numUndirectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of components in the polyline
        /// Generated from method `MR::ObjectGcode::numComponents`.
        public unsafe ulong NumComponents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_numComponents", ExactSpelling = true)]
            extern static ulong __MR_ObjectGcode_numComponents(_Underlying *_this);
            return __MR_ObjectGcode_numComponents(_UnderlyingPtr);
        }

        /// return cached total length
        /// Generated from method `MR::ObjectGcode::totalLength`.
        public unsafe float TotalLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_totalLength", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_totalLength(_Underlying *_this);
            return __MR_ObjectGcode_totalLength(_UnderlyingPtr);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::ObjectGcode::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectGcode_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::ObjectGcode::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectGcode_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_ObjectGcode_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::ObjectGcode::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectGcode_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectGcode::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectGcode_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectGcode_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectGcode::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectGcode_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectGcode_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::ObjectGcode::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectGcode_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectGcode_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::ObjectGcode::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectGcode_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_ObjectGcode_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectGcode::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_ObjectGcode_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_ObjectGcode_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectGcode::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_ObjectGcode_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_ObjectGcode_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::ObjectGcode::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_ObjectGcode_getDirtyFlags(_Underlying *_this);
            return __MR_ObjectGcode_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::ObjectGcode::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_resetDirty", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_resetDirty(_Underlying *_this);
            __MR_ObjectGcode_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::ObjectGcode::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_ObjectGcode_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::ObjectGcode::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectGcode_getBoundingBox(_Underlying *_this);
            return __MR_ObjectGcode_getBoundingBox(_UnderlyingPtr);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::ObjectGcode::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectGcode_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::ObjectGcode::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isPickable", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectGcode_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::ObjectGcode::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_ObjectGcode_getColoringType(_Underlying *_this);
            return __MR_ObjectGcode_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::ObjectGcode::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getShininess", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_getShininess(_Underlying *_this);
            return __MR_ObjectGcode_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::ObjectGcode::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_getSpecularStrength(_Underlying *_this);
            return __MR_ObjectGcode_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::ObjectGcode::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_ObjectGcode_getAmbientStrength(_Underlying *_this);
            return __MR_ObjectGcode_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::ObjectGcode::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_render", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_ObjectGcode_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::ObjectGcode::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_renderForPicker", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_ObjectGcode_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::ObjectGcode::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_renderUi", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_ObjectGcode_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectGcode::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_ObjectGcode_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectGcode::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_ObjectGcode_name(_Underlying *_this);
            return new(__MR_ObjectGcode_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::ObjectGcode::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectGcode_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectGcode_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::ObjectGcode::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_ObjectGcode_xfsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectGcode_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::ObjectGcode::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ObjectGcode_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectGcode_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::ObjectGcode::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectGcode_globalVisibilityMask(_Underlying *_this);
            return new(__MR_ObjectGcode_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::ObjectGcode::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectGcode_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::ObjectGcode::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isLocked(_Underlying *_this);
            return __MR_ObjectGcode_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::ObjectGcode::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isParentLocked(_Underlying *_this);
            return __MR_ObjectGcode_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::ObjectGcode::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isAncestor", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_ObjectGcode_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectGcode::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isSelected", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isSelected(_Underlying *_this);
            return __MR_ObjectGcode_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectGcode::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isAncillary(_Underlying *_this);
            return __MR_ObjectGcode_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::ObjectGcode::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isGlobalAncillary(_Underlying *_this);
            return __MR_ObjectGcode_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::ObjectGcode::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_isVisible", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectGcode_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectGcode::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectGcode_visibilityMask(_Underlying *_this);
            return new(__MR_ObjectGcode_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectGcode::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_resetRedrawFlag(_Underlying *_this);
            __MR_ObjectGcode_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::ObjectGcode::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectGcode_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectGcode_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::ObjectGcode::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectGcode_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectGcode_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::ObjectGcode::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectGcode_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectGcode_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::ObjectGcode::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_ObjectGcode_tags(_Underlying *_this);
            return new(__MR_ObjectGcode_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::ObjectGcode::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_sameModels", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_ObjectGcode_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::ObjectGcode::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_ObjectGcode_getModelHash(_Underlying *_this);
            return __MR_ObjectGcode_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectGcode::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectGcode_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectGcode_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// an object that stores a g-code
    /// Generated from class `MR::ObjectGcode`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectLinesHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class ObjectGcode : Const_ObjectGcode
    {
        internal unsafe ObjectGcode(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectGcode(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectGcode_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_ObjectGcode_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_ObjectGcode_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.ObjectLinesHolder(ObjectGcode self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_UpcastTo_MR_ObjectLinesHolder", ExactSpelling = true)]
            extern static MR.ObjectLinesHolder._Underlying *__MR_ObjectGcode_UpcastTo_MR_ObjectLinesHolder(_Underlying *_this);
            return MR.ObjectLinesHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectGcode_UpcastTo_MR_ObjectLinesHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ObjectGcode?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectGcode", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectGcode(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectGcode(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectGcode?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectGcode", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectGcode(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectGcode(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectGcode?(MR.ObjectLinesHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectLinesHolder_DynamicDowncastTo_MR_ObjectGcode", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectLinesHolder_DynamicDowncastTo_MR_ObjectGcode(MR.ObjectLinesHolder._Underlying *_this);
            var ptr = __MR_ObjectLinesHolder_DynamicDowncastTo_MR_ObjectGcode(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.ObjectLinesHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectGcode() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectGcode._Underlying *__MR_ObjectGcode_DefaultConstruct();
            _LateMakeShared(__MR_ObjectGcode_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectGcode::ObjectGcode`.
        public unsafe ObjectGcode(MR._ByValue_ObjectGcode _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectGcode._Underlying *__MR_ObjectGcode_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectGcode._Underlying *_other);
            _LateMakeShared(__MR_ObjectGcode_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectGcode::operator=`.
        public unsafe MR.ObjectGcode Assign(MR._ByValue_ObjectGcode _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectGcode._Underlying *__MR_ObjectGcode_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectGcode._Underlying *_other);
            return new(__MR_ObjectGcode_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectGcode::setCNCMachineSettings`.
        public unsafe void SetCNCMachineSettings(MR.Const_CNCMachineSettings cncSettings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setCNCMachineSettings", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setCNCMachineSettings(_Underlying *_this, MR.Const_CNCMachineSettings._Underlying *cncSettings);
            __MR_ObjectGcode_setCNCMachineSettings(_UnderlyingPtr, cncSettings._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectGcode::setGcodeSource`.
        public unsafe void SetGcodeSource(MR.Std.Const_Vector_StdString gcodeSource)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setGcodeSource", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setGcodeSource(_Underlying *_this, MR.Std.Const_Vector_StdString._UnderlyingShared *gcodeSource);
            __MR_ObjectGcode_setGcodeSource(_UnderlyingPtr, gcodeSource._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ObjectGcode::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectGcode_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        // set drawing feedrate as gradient of brightness
        /// Generated from method `MR::ObjectGcode::switchFeedrateGradient`.
        public unsafe void SwitchFeedrateGradient(bool isFeedrateGradientEnabled)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_switchFeedrateGradient", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_switchFeedrateGradient(_Underlying *_this, byte isFeedrateGradientEnabled);
            __MR_ObjectGcode_switchFeedrateGradient(_UnderlyingPtr, isFeedrateGradientEnabled ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectGcode::setIdleColor`.
        public unsafe void SetIdleColor(MR.Const_Color color)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setIdleColor", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setIdleColor(_Underlying *_this, MR.Const_Color._Underlying *color);
            __MR_ObjectGcode_setIdleColor(_UnderlyingPtr, color._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectGcode::select`.
        public unsafe bool Select(bool isSelected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_select", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_select(_Underlying *_this, byte isSelected);
            return __MR_ObjectGcode_select(_UnderlyingPtr, isSelected ? (byte)1 : (byte)0) != 0;
        }

        /// Generated from method `MR::ObjectGcode::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_ObjectGcode_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// Generated from method `MR::ObjectGcode::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_applyScale", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_applyScale(_Underlying *_this, float scaleFactor);
            __MR_ObjectGcode_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// specify dash pattern in pixels
        /// [0] - dash
        /// [1] - space
        /// [2] - dash
        /// [3] - space
        /// Generated from method `MR::ObjectGcode::setDashPattern`.
        /// Parameter `vpId` defaults to `{}`.
        public unsafe void SetDashPattern(MR.Const_Vector4_UnsignedChar pattern, MR._InOpt_ViewportId vpId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setDashPattern", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setDashPattern(_Underlying *_this, MR.Const_Vector4_UnsignedChar._Underlying *pattern, MR.ViewportId *vpId);
            __MR_ObjectGcode_setDashPattern(_UnderlyingPtr, pattern._UnderlyingPtr, vpId.HasValue ? &vpId.Object : null);
        }

        /// Generated from method `MR::ObjectGcode::setLineWidth`.
        public unsafe void SetLineWidth(float width)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setLineWidth", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setLineWidth(_Underlying *_this, float width);
            __MR_ObjectGcode_setLineWidth(_UnderlyingPtr, width);
        }

        /// Generated from method `MR::ObjectGcode::setPointSize`.
        public unsafe void SetPointSize(float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setPointSize", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setPointSize(_Underlying *_this, float size);
            __MR_ObjectGcode_setPointSize(_UnderlyingPtr, size);
        }

        /// sets per-vertex colors of the object
        /// Generated from method `MR::ObjectGcode::setVertsColorMap`.
        public unsafe void SetVertsColorMap(MR._ByValue_VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setVertsColorMap(_Underlying *_this, MR.Misc._PassBy vertsColorMap_pass_by, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectGcode_setVertsColorMap(_UnderlyingPtr, vertsColorMap.PassByMode, vertsColorMap.Value is not null ? vertsColorMap.Value._UnderlyingPtr : null);
        }

        /// swaps per-vertex colors of the object with given argument
        /// Generated from method `MR::ObjectGcode::updateVertsColorMap`.
        public unsafe void UpdateVertsColorMap(MR.VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_updateVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_updateVertsColorMap(_Underlying *_this, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectGcode_updateVertsColorMap(_UnderlyingPtr, vertsColorMap._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectGcode::setLinesColorMap`.
        public unsafe void SetLinesColorMap(MR._ByValue_UndirectedEdgeColors linesColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setLinesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setLinesColorMap(_Underlying *_this, MR.Misc._PassBy linesColorMap_pass_by, MR.UndirectedEdgeColors._Underlying *linesColorMap);
            __MR_ObjectGcode_setLinesColorMap(_UnderlyingPtr, linesColorMap.PassByMode, linesColorMap.Value is not null ? linesColorMap.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectGcode::updateLinesColorMap`.
        public unsafe void UpdateLinesColorMap(MR.UndirectedEdgeColors updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_updateLinesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_updateLinesColorMap(_Underlying *_this, MR.UndirectedEdgeColors._Underlying *updated);
            __MR_ObjectGcode_updateLinesColorMap(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// copies vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectGcode::copyColors`.
        public unsafe void CopyColors(MR.Const_ObjectLinesHolder src, MR.Const_VertMap thisToSrc)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_copyColors", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_copyColors(_Underlying *_this, MR.Const_ObjectLinesHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc);
            __MR_ObjectGcode_copyColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::ObjectGcode::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_resetFrontColor(_Underlying *_this);
            __MR_ObjectGcode_resetFrontColor(_UnderlyingPtr);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectGcode::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::ObjectGcode::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectGcode::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::ObjectGcode::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_ObjectGcode_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::ObjectGcode::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_ObjectGcode_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::ObjectGcode::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectGcode::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectGcode_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::ObjectGcode::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectGcode_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::ObjectGcode::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setBackColor", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_ObjectGcode_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectGcode::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_ObjectGcode_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectGcode::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_ObjectGcode_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::ObjectGcode::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setPickable", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::ObjectGcode::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setColoringType", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_ObjectGcode_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::ObjectGcode::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setShininess", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setShininess(_Underlying *_this, float shininess);
            __MR_ObjectGcode_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::ObjectGcode::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_ObjectGcode_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::ObjectGcode::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_ObjectGcode_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectGcode::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_ObjectGcode_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::ObjectGcode::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_resetColors", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_resetColors(_Underlying *_this);
            __MR_ObjectGcode_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectGcode::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setName", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_ObjectGcode_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::ObjectGcode::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setXf", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectGcode_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::ObjectGcode::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_resetXf", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_ObjectGcode_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::ObjectGcode::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_ObjectGcode_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectGcode::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setWorldXf", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectGcode_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::ObjectGcode::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectGcode::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setLocked", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setLocked(_Underlying *_this, byte on);
            __MR_ObjectGcode_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectGcode::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setParentLocked", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setParentLocked(_Underlying *_this, byte lock_);
            __MR_ObjectGcode_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::ObjectGcode::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_detachFromParent(_Underlying *_this);
            return __MR_ObjectGcode_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::ObjectGcode::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_addChild", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectGcode_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::ObjectGcode::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_ObjectGcode_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::ObjectGcode::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_removeAllChildren(_Underlying *_this);
            __MR_ObjectGcode_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::ObjectGcode::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_sortChildren", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_sortChildren(_Underlying *_this);
            __MR_ObjectGcode_sortChildren(_UnderlyingPtr);
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::ObjectGcode::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setAncillary", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setAncillary(_Underlying *_this, byte ancillary);
            __MR_ObjectGcode_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::ObjectGcode::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setVisible", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectGcode::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectGcode_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::ObjectGcode::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_swap", ExactSpelling = true)]
            extern static void __MR_ObjectGcode_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_ObjectGcode_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::ObjectGcode::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_addTag", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectGcode_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::ObjectGcode::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectGcode_removeTag", ExactSpelling = true)]
            extern static byte __MR_ObjectGcode_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectGcode_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectGcode` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectGcode`/`Const_ObjectGcode` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectGcode
    {
        internal readonly Const_ObjectGcode? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectGcode() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectGcode(MR.Misc._Moved<ObjectGcode> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectGcode(MR.Misc._Moved<ObjectGcode> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectGcode` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectGcode`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectGcode`/`Const_ObjectGcode` directly.
    public class _InOptMut_ObjectGcode
    {
        public ObjectGcode? Opt;

        public _InOptMut_ObjectGcode() {}
        public _InOptMut_ObjectGcode(ObjectGcode value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectGcode(ObjectGcode value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectGcode` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectGcode`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectGcode`/`Const_ObjectGcode` to pass it to the function.
    public class _InOptConst_ObjectGcode
    {
        public Const_ObjectGcode? Opt;

        public _InOptConst_ObjectGcode() {}
        public _InOptConst_ObjectGcode(Const_ObjectGcode value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectGcode(Const_ObjectGcode value) {return new(value);}
    }
}
