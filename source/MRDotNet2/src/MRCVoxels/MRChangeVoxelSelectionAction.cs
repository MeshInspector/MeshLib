public static partial class MR
{
    /// Undo action for ObjectVoxels face selection
    /// Generated from class `MR::ChangVoxelSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangVoxelSelectionAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangVoxelSelectionAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangVoxelSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangVoxelSelectionAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangVoxelSelectionAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangVoxelSelectionAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangVoxelSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangVoxelSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangVoxelSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangVoxelSelectionAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangVoxelSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangVoxelSelectionAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangVoxelSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangVoxelSelectionAction::ChangVoxelSelectionAction`.
        public unsafe Const_ChangVoxelSelectionAction(MR._ByValue_ChangVoxelSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangVoxelSelectionAction._Underlying *__MR_ChangVoxelSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangVoxelSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangVoxelSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's face selection before making any changes in it
        /// Generated from constructor `MR::ChangVoxelSelectionAction::ChangVoxelSelectionAction`.
        public unsafe Const_ChangVoxelSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels objVoxels) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_Construct", ExactSpelling = true)]
            extern static MR.ChangVoxelSelectionAction._Underlying *__MR_ChangVoxelSelectionAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *objVoxels);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangVoxelSelectionAction_Construct(__ptr_name, __ptr_name + __len_name, objVoxels._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangVoxelSelectionAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangVoxelSelectionAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangVoxelSelectionAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangVoxelSelectionAction::selection`.
        public unsafe MR.Const_VoxelBitSet Selection()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_selection", ExactSpelling = true)]
            extern static MR.Const_VoxelBitSet._Underlying *__MR_ChangVoxelSelectionAction_selection(_Underlying *_this);
            return new(__MR_ChangVoxelSelectionAction_selection(_UnderlyingPtr), is_owning: false);
        }

        /// empty because set dirty is inside selectFaces
        /// Generated from method `MR::ChangVoxelSelectionAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectVoxels _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangVoxelSelectionAction_setObjectDirty(MR.Const_ObjectVoxels._UnderlyingShared *_1);
            __MR_ChangVoxelSelectionAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangVoxelSelectionAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangVoxelSelectionAction_heapBytes(_Underlying *_this);
            return __MR_ChangVoxelSelectionAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectVoxels face selection
    /// Generated from class `MR::ChangVoxelSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangVoxelSelectionAction : Const_ChangVoxelSelectionAction
    {
        internal unsafe ChangVoxelSelectionAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangVoxelSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangVoxelSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangVoxelSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangVoxelSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangVoxelSelectionAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangVoxelSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangVoxelSelectionAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangVoxelSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangVoxelSelectionAction::ChangVoxelSelectionAction`.
        public unsafe ChangVoxelSelectionAction(MR._ByValue_ChangVoxelSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangVoxelSelectionAction._Underlying *__MR_ChangVoxelSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangVoxelSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangVoxelSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's face selection before making any changes in it
        /// Generated from constructor `MR::ChangVoxelSelectionAction::ChangVoxelSelectionAction`.
        public unsafe ChangVoxelSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels objVoxels) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_Construct", ExactSpelling = true)]
            extern static MR.ChangVoxelSelectionAction._Underlying *__MR_ChangVoxelSelectionAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *objVoxels);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangVoxelSelectionAction_Construct(__ptr_name, __ptr_name + __len_name, objVoxels._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangVoxelSelectionAction::operator=`.
        public unsafe MR.ChangVoxelSelectionAction Assign(MR._ByValue_ChangVoxelSelectionAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangVoxelSelectionAction._Underlying *__MR_ChangVoxelSelectionAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangVoxelSelectionAction._Underlying *_other);
            return new(__MR_ChangVoxelSelectionAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangVoxelSelectionAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangVoxelSelectionAction_action", ExactSpelling = true)]
            extern static void __MR_ChangVoxelSelectionAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangVoxelSelectionAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangVoxelSelectionAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangVoxelSelectionAction`/`Const_ChangVoxelSelectionAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangVoxelSelectionAction
    {
        internal readonly Const_ChangVoxelSelectionAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangVoxelSelectionAction(Const_ChangVoxelSelectionAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangVoxelSelectionAction(Const_ChangVoxelSelectionAction arg) {return new(arg);}
        public _ByValue_ChangVoxelSelectionAction(MR.Misc._Moved<ChangVoxelSelectionAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangVoxelSelectionAction(MR.Misc._Moved<ChangVoxelSelectionAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangVoxelSelectionAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangVoxelSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangVoxelSelectionAction`/`Const_ChangVoxelSelectionAction` directly.
    public class _InOptMut_ChangVoxelSelectionAction
    {
        public ChangVoxelSelectionAction? Opt;

        public _InOptMut_ChangVoxelSelectionAction() {}
        public _InOptMut_ChangVoxelSelectionAction(ChangVoxelSelectionAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangVoxelSelectionAction(ChangVoxelSelectionAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangVoxelSelectionAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangVoxelSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangVoxelSelectionAction`/`Const_ChangVoxelSelectionAction` to pass it to the function.
    public class _InOptConst_ChangVoxelSelectionAction
    {
        public Const_ChangVoxelSelectionAction? Opt;

        public _InOptConst_ChangVoxelSelectionAction() {}
        public _InOptConst_ChangVoxelSelectionAction(Const_ChangVoxelSelectionAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangVoxelSelectionAction(Const_ChangVoxelSelectionAction value) {return new(value);}
    }
}
