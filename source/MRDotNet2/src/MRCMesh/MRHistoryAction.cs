public static partial class MR
{
    /// Abstract class for history actions
    /// Generated from class `MR::HistoryAction`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ChangVoxelSelectionAction`
    ///     `MR::ChangeActiveBoxAction`
    ///     `MR::ChangeColoringType`
    ///     `MR::ChangeDualMarchingCubesAction`
    ///     `MR::ChangeFacesColorMapAction`
    ///     `MR::ChangeGridAction`
    ///     `MR::ChangeIsoAction`
    ///     `MR::ChangeLabelAction`
    ///     `MR::ChangeLinesColorMapAction`
    ///     `MR::ChangeMeshAction`
    ///     `MR::ChangeMeshCreasesAction`
    ///     `MR::ChangeMeshDataAction`
    ///     `MR::ChangeMeshEdgeSelectionAction`
    ///     `MR::ChangeMeshFaceSelectionAction`
    ///     `MR::ChangeMeshPointsAction`
    ///     `MR::ChangeMeshTexturePerFaceAction`
    ///     `MR::ChangeMeshTopologyAction`
    ///     `MR::ChangeMeshUVCoordsAction`
    ///     `MR::ChangeNameAction`
    ///     `MR::ChangeObjectAction`
    ///     `MR::ChangeObjectColorAction`
    ///     `MR::ChangeObjectSelectedAction`
    ///     `MR::ChangeObjectVisibilityAction`
    ///     `MR::ChangeOneNormalInCloudAction`
    ///     `MR::ChangeOnePointInCloudAction`
    ///     `MR::ChangeOnePointInPolylineAction`
    ///     `MR::ChangePointCloudAction`
    ///     `MR::ChangePointCloudNormalsAction`
    ///     `MR::ChangePointCloudPointsAction`
    ///     `MR::ChangePointPointSelectionAction`
    ///     `MR::ChangePolylineAction`
    ///     `MR::ChangePolylinePointsAction`
    ///     `MR::ChangePolylineTopologyAction`
    ///     `MR::ChangeScaleAction`
    ///     `MR::ChangeSceneAction`
    ///     `MR::ChangeSceneObjectsOrder`
    ///     `MR::ChangeSurfaceAction`
    ///     `MR::ChangeTextureAction`
    ///     `MR::ChangeVisualizePropertyAction`
    ///     `MR::ChangeXfAction`
    ///     `MR::CombinedHistoryAction`
    ///     `MR::PartialChangeMeshAction`
    ///     `MR::PartialChangeMeshDataAction`
    ///     `MR::PartialChangeMeshPointsAction`
    ///     `MR::PartialChangeMeshTopologyAction`
    /// This is the const half of the class.
    public class Const_HistoryAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_HistoryAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_HistoryAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_HistoryAction_UseCount();
                return __MR_std_shared_ptr_MR_HistoryAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_HistoryAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_HistoryAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_HistoryAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_HistoryAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_HistoryAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_HistoryAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_HistoryAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_HistoryAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe HistoryAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_HistoryAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_HistoryAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_HistoryAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_HistoryAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_HistoryAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_HistoryAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_HistoryAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_HistoryAction() {Dispose(false);}

        /// Generated from method `MR::HistoryAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_HistoryAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_HistoryAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::HistoryAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_HistoryAction_heapBytes(_Underlying *_this);
            return __MR_HistoryAction_heapBytes(_UnderlyingPtr);
        }

        public enum Type : int
        {
            Undo = 0,
            Redo = 1,
        }
    }

    /// Abstract class for history actions
    /// Generated from class `MR::HistoryAction`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ChangVoxelSelectionAction`
    ///     `MR::ChangeActiveBoxAction`
    ///     `MR::ChangeColoringType`
    ///     `MR::ChangeDualMarchingCubesAction`
    ///     `MR::ChangeFacesColorMapAction`
    ///     `MR::ChangeGridAction`
    ///     `MR::ChangeIsoAction`
    ///     `MR::ChangeLabelAction`
    ///     `MR::ChangeLinesColorMapAction`
    ///     `MR::ChangeMeshAction`
    ///     `MR::ChangeMeshCreasesAction`
    ///     `MR::ChangeMeshDataAction`
    ///     `MR::ChangeMeshEdgeSelectionAction`
    ///     `MR::ChangeMeshFaceSelectionAction`
    ///     `MR::ChangeMeshPointsAction`
    ///     `MR::ChangeMeshTexturePerFaceAction`
    ///     `MR::ChangeMeshTopologyAction`
    ///     `MR::ChangeMeshUVCoordsAction`
    ///     `MR::ChangeNameAction`
    ///     `MR::ChangeObjectAction`
    ///     `MR::ChangeObjectColorAction`
    ///     `MR::ChangeObjectSelectedAction`
    ///     `MR::ChangeObjectVisibilityAction`
    ///     `MR::ChangeOneNormalInCloudAction`
    ///     `MR::ChangeOnePointInCloudAction`
    ///     `MR::ChangeOnePointInPolylineAction`
    ///     `MR::ChangePointCloudAction`
    ///     `MR::ChangePointCloudNormalsAction`
    ///     `MR::ChangePointCloudPointsAction`
    ///     `MR::ChangePointPointSelectionAction`
    ///     `MR::ChangePolylineAction`
    ///     `MR::ChangePolylinePointsAction`
    ///     `MR::ChangePolylineTopologyAction`
    ///     `MR::ChangeScaleAction`
    ///     `MR::ChangeSceneAction`
    ///     `MR::ChangeSceneObjectsOrder`
    ///     `MR::ChangeSurfaceAction`
    ///     `MR::ChangeTextureAction`
    ///     `MR::ChangeVisualizePropertyAction`
    ///     `MR::ChangeXfAction`
    ///     `MR::CombinedHistoryAction`
    ///     `MR::PartialChangeMeshAction`
    ///     `MR::PartialChangeMeshDataAction`
    ///     `MR::PartialChangeMeshPointsAction`
    ///     `MR::PartialChangeMeshTopologyAction`
    /// This is the non-const half of the class.
    public class HistoryAction : Const_HistoryAction
    {
        internal unsafe HistoryAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe HistoryAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// This function is called on history action (undo, redo, etc.)
        /// Generated from method `MR::HistoryAction::action`.
        public unsafe void Action(MR.HistoryAction.Type actionType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_action", ExactSpelling = true)]
            extern static void __MR_HistoryAction_action(_Underlying *_this, MR.HistoryAction.Type actionType);
            __MR_HistoryAction_action(_UnderlyingPtr, actionType);
        }
    }

    /// This is used as a function parameter when the underlying function receives `HistoryAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `HistoryAction`/`Const_HistoryAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_HistoryAction
    {
        internal readonly Const_HistoryAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `HistoryAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_HistoryAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `HistoryAction`/`Const_HistoryAction` directly.
    public class _InOptMut_HistoryAction
    {
        public HistoryAction? Opt;

        public _InOptMut_HistoryAction() {}
        public _InOptMut_HistoryAction(HistoryAction value) {Opt = value;}
        public static implicit operator _InOptMut_HistoryAction(HistoryAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `HistoryAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_HistoryAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `HistoryAction`/`Const_HistoryAction` to pass it to the function.
    public class _InOptConst_HistoryAction
    {
        public Const_HistoryAction? Opt;

        public _InOptConst_HistoryAction() {}
        public _InOptConst_HistoryAction(Const_HistoryAction value) {Opt = value;}
        public static implicit operator _InOptConst_HistoryAction(Const_HistoryAction value) {return new(value);}
    }

    /**
    * \brief Remove actions from history actions vector that match the condition
    * \param firstRedoIndex - set redo index for calculate how many actions removed before it
    * \param deepFiltering - filter actions into combined actions
    * \return pair (anything removed, how many removed before firstRedoIndex)
    */
    /// Generated from function `MR::filterHistoryActionsVector`.
    /// Parameter `firstRedoIndex` defaults to `0`.
    /// Parameter `deepFiltering` defaults to `true`.
    public static unsafe MR.Std.Pair_Bool_Int FilterHistoryActionsVector(MR.Std.Vector_StdSharedPtrMRHistoryAction historyVector, MR.Std._ByValue_Function_BoolFuncFromConstStdSharedPtrMRHistoryActionRef filteringCondition, ulong? firstRedoIndex = null, bool? deepFiltering = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_filterHistoryActionsVector", ExactSpelling = true)]
        extern static MR.Std.Pair_Bool_Int._Underlying *__MR_filterHistoryActionsVector(MR.Std.Vector_StdSharedPtrMRHistoryAction._Underlying *historyVector, MR.Misc._PassBy filteringCondition_pass_by, MR.Std.Function_BoolFuncFromConstStdSharedPtrMRHistoryActionRef._Underlying *filteringCondition, ulong *firstRedoIndex, byte *deepFiltering);
        ulong __deref_firstRedoIndex = firstRedoIndex.GetValueOrDefault();
        byte __deref_deepFiltering = deepFiltering.GetValueOrDefault() ? (byte)1 : (byte)0;
        return new(__MR_filterHistoryActionsVector(historyVector._UnderlyingPtr, filteringCondition.PassByMode, filteringCondition.Value is not null ? filteringCondition.Value._UnderlyingPtr : null, firstRedoIndex.HasValue ? &__deref_firstRedoIndex : null, deepFiltering.HasValue ? &__deref_deepFiltering : null), is_owning: true);
    }
}
