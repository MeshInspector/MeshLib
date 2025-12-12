public static partial class MR
{
    public enum ObjectSelectivityType : int
    {
        /// object itself and all its ancestors are selectable
        Selectable = 0,
        /// object itself is selectable
        LocalSelectable = 1,
        /// object itself is selected and all its ancestors are selectable
        Selected = 2,
        /// object itself is selected
        LocalSelected = 3,
        /// any object
        Any = 4,
    }

    /// if input object is of given type then returns another pointer on it
    /// Generated from function `MR::asSelectivityType<MR::Object>`.
    public static unsafe MR.Misc._Moved<MR.Object> AsSelectivityType(MR._ByValue_Object obj, MR.ObjectSelectivityType type)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_asSelectivityType", ExactSpelling = true)]
        extern static MR.Object._UnderlyingShared *__MR_asSelectivityType(MR.Misc._PassBy obj_pass_by, MR.Object._UnderlyingShared *obj, MR.ObjectSelectivityType *type);
        return MR.Misc.Move(new MR.Object(__MR_asSelectivityType(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, &type), is_owning: true));
    }

    /// Traverses tree and collect objects of given type excluding root
    /// returns vector
    /// Generated from function `MR::getAllObjectsInTree<MR::Object>`.
    /// Parameter `type` defaults to `ObjectSelectivityType::Selectable`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdSharedPtrMRObject> GetAllObjectsInTree(MR.Object? root, MR.ObjectSelectivityType? type = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getAllObjectsInTree", ExactSpelling = true)]
        extern static MR.Std.Vector_StdSharedPtrMRObject._Underlying *__MR_getAllObjectsInTree(MR.Object._Underlying *root, MR.ObjectSelectivityType *type);
        MR.ObjectSelectivityType __deref_type = type.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_StdSharedPtrMRObject(__MR_getAllObjectsInTree(root is not null ? root._UnderlyingPtr : null, type.HasValue ? &__deref_type : null), is_owning: true));
    }

    /// Returns all topmost visible objects of given type (if an object is returned, its children are not) excluding root
    /// Generated from function `MR::getTopmostVisibleObjects<MR::Object>`.
    /// Parameter `type` defaults to `ObjectSelectivityType::Selectable`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdSharedPtrMRObject> GetTopmostVisibleObjects(MR.Object? root, MR.ObjectSelectivityType? type = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getTopmostVisibleObjects", ExactSpelling = true)]
        extern static MR.Std.Vector_StdSharedPtrMRObject._Underlying *__MR_getTopmostVisibleObjects(MR.Object._Underlying *root, MR.ObjectSelectivityType *type);
        MR.ObjectSelectivityType __deref_type = type.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_StdSharedPtrMRObject(__MR_getTopmostVisibleObjects(root is not null ? root._UnderlyingPtr : null, type.HasValue ? &__deref_type : null), is_owning: true));
    }

    /// Returns all topmost objects of given type (if an object is returned, its children are not) excluding root
    /// Generated from function `MR::getTopmostObjects<MR::Object>`.
    /// Parameter `type` defaults to `ObjectSelectivityType::Selectable`.
    /// Parameter `visibilityCheck` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdSharedPtrMRObject> GetTopmostObjects(MR.Object? root, MR.ObjectSelectivityType? type = null, bool? visibilityCheck = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getTopmostObjects", ExactSpelling = true)]
        extern static MR.Std.Vector_StdSharedPtrMRObject._Underlying *__MR_getTopmostObjects(MR.Object._Underlying *root, MR.ObjectSelectivityType *type, byte *visibilityCheck);
        MR.ObjectSelectivityType __deref_type = type.GetValueOrDefault();
        byte __deref_visibilityCheck = visibilityCheck.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_StdSharedPtrMRObject(__MR_getTopmostObjects(root is not null ? root._UnderlyingPtr : null, type.HasValue ? &__deref_type : null, visibilityCheck.HasValue ? &__deref_visibilityCheck : null), is_owning: true));
    }

    /// return first object of given type in depth-first traverse order excluding root
    /// Generated from function `MR::getDepthFirstObject<MR::Object>`.
    public static unsafe MR.Misc._Moved<MR.Object> GetDepthFirstObject(MR.Object? root, MR.ObjectSelectivityType type)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getDepthFirstObject", ExactSpelling = true)]
        extern static MR.Object._UnderlyingShared *__MR_getDepthFirstObject(MR.Object._Underlying *root, MR.ObjectSelectivityType *type);
        return MR.Misc.Move(new MR.Object(__MR_getDepthFirstObject(root is not null ? root._UnderlyingPtr : null, &type), is_owning: true));
    }

    /// returns whether the object has selectable children
    /// \param recurse - if true, look up for selectable children at any depth
    /// Generated from function `MR::objectHasSelectableChildren`.
    /// Parameter `recurse` defaults to `false`.
    public static unsafe bool ObjectHasSelectableChildren(MR.Const_Object object_, bool? recurse = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_objectHasSelectableChildren", ExactSpelling = true)]
        extern static byte __MR_objectHasSelectableChildren(MR.Const_Object._Underlying *object_, byte *recurse);
        byte __deref_recurse = recurse.GetValueOrDefault() ? (byte)1 : (byte)0;
        return __MR_objectHasSelectableChildren(object_._UnderlyingPtr, recurse.HasValue ? &__deref_recurse : null) != 0;
    }
}
