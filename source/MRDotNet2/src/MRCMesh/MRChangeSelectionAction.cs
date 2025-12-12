public static partial class MR
{
    /// Undo action for ObjectMesh face selection
    /// Generated from class `MR::ChangeMeshFaceSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshFaceSelectionAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshFaceSelectionAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshFaceSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshFaceSelectionAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshFaceSelectionAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshFaceSelectionAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshFaceSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshFaceSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshFaceSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshFaceSelectionAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshFaceSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshFaceSelectionAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshFaceSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshFaceSelectionAction::ChangeMeshFaceSelectionAction`.
        public unsafe Const_ChangeMeshFaceSelectionAction(MR._ByValue_ChangeMeshFaceSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshFaceSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshFaceSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's face selection before making any changes in it
        /// Generated from constructor `MR::ChangeMeshFaceSelectionAction::ChangeMeshFaceSelectionAction`.
        public unsafe Const_ChangeMeshFaceSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshFaceSelectionAction_Construct_2(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's face selection and immediate set new value
        /// Generated from constructor `MR::ChangeMeshFaceSelectionAction::ChangeMeshFaceSelectionAction`.
        public unsafe Const_ChangeMeshFaceSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh, MR.Misc._Moved<MR.FaceBitSet> newSelection) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh, MR.FaceBitSet._Underlying *newSelection);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshFaceSelectionAction_Construct_3(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr, newSelection.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshFaceSelectionAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshFaceSelectionAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshFaceSelectionAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshFaceSelectionAction::selection`.
        public unsafe MR.Const_FaceBitSet Selection()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_selection", ExactSpelling = true)]
            extern static MR.Const_FaceBitSet._Underlying *__MR_ChangeMeshFaceSelectionAction_selection(_Underlying *_this);
            return new(__MR_ChangeMeshFaceSelectionAction_selection(_UnderlyingPtr), is_owning: false);
        }

        /// empty because set dirty is inside selectFaces
        /// Generated from method `MR::ChangeMeshFaceSelectionAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshFaceSelectionAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *_1);
            __MR_ChangeMeshFaceSelectionAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshFaceSelectionAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshFaceSelectionAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshFaceSelectionAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMesh face selection
    /// Generated from class `MR::ChangeMeshFaceSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshFaceSelectionAction : Const_ChangeMeshFaceSelectionAction
    {
        internal unsafe ChangeMeshFaceSelectionAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshFaceSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshFaceSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshFaceSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshFaceSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshFaceSelectionAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshFaceSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshFaceSelectionAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshFaceSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshFaceSelectionAction::ChangeMeshFaceSelectionAction`.
        public unsafe ChangeMeshFaceSelectionAction(MR._ByValue_ChangeMeshFaceSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshFaceSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshFaceSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's face selection before making any changes in it
        /// Generated from constructor `MR::ChangeMeshFaceSelectionAction::ChangeMeshFaceSelectionAction`.
        public unsafe ChangeMeshFaceSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshFaceSelectionAction_Construct_2(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's face selection and immediate set new value
        /// Generated from constructor `MR::ChangeMeshFaceSelectionAction::ChangeMeshFaceSelectionAction`.
        public unsafe ChangeMeshFaceSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh, MR.Misc._Moved<MR.FaceBitSet> newSelection) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh, MR.FaceBitSet._Underlying *newSelection);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshFaceSelectionAction_Construct_3(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr, newSelection.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshFaceSelectionAction::operator=`.
        public unsafe MR.ChangeMeshFaceSelectionAction Assign(MR._ByValue_ChangeMeshFaceSelectionAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshFaceSelectionAction._Underlying *__MR_ChangeMeshFaceSelectionAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshFaceSelectionAction._Underlying *_other);
            return new(__MR_ChangeMeshFaceSelectionAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshFaceSelectionAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshFaceSelectionAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshFaceSelectionAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshFaceSelectionAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshFaceSelectionAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshFaceSelectionAction`/`Const_ChangeMeshFaceSelectionAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshFaceSelectionAction
    {
        internal readonly Const_ChangeMeshFaceSelectionAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshFaceSelectionAction(Const_ChangeMeshFaceSelectionAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshFaceSelectionAction(Const_ChangeMeshFaceSelectionAction arg) {return new(arg);}
        public _ByValue_ChangeMeshFaceSelectionAction(MR.Misc._Moved<ChangeMeshFaceSelectionAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshFaceSelectionAction(MR.Misc._Moved<ChangeMeshFaceSelectionAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshFaceSelectionAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshFaceSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshFaceSelectionAction`/`Const_ChangeMeshFaceSelectionAction` directly.
    public class _InOptMut_ChangeMeshFaceSelectionAction
    {
        public ChangeMeshFaceSelectionAction? Opt;

        public _InOptMut_ChangeMeshFaceSelectionAction() {}
        public _InOptMut_ChangeMeshFaceSelectionAction(ChangeMeshFaceSelectionAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshFaceSelectionAction(ChangeMeshFaceSelectionAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshFaceSelectionAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshFaceSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshFaceSelectionAction`/`Const_ChangeMeshFaceSelectionAction` to pass it to the function.
    public class _InOptConst_ChangeMeshFaceSelectionAction
    {
        public Const_ChangeMeshFaceSelectionAction? Opt;

        public _InOptConst_ChangeMeshFaceSelectionAction() {}
        public _InOptConst_ChangeMeshFaceSelectionAction(Const_ChangeMeshFaceSelectionAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshFaceSelectionAction(Const_ChangeMeshFaceSelectionAction value) {return new(value);}
    }

    /// Undo action for ObjectMesh edge selection
    /// Generated from class `MR::ChangeMeshEdgeSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshEdgeSelectionAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshEdgeSelectionAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshEdgeSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshEdgeSelectionAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshEdgeSelectionAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshEdgeSelectionAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshEdgeSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshEdgeSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshEdgeSelectionAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshEdgeSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshEdgeSelectionAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshEdgeSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshEdgeSelectionAction::ChangeMeshEdgeSelectionAction`.
        public unsafe Const_ChangeMeshEdgeSelectionAction(MR._ByValue_ChangeMeshEdgeSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshEdgeSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's edge selection before making any changes in it
        /// Generated from constructor `MR::ChangeMeshEdgeSelectionAction::ChangeMeshEdgeSelectionAction`.
        public unsafe Const_ChangeMeshEdgeSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshEdgeSelectionAction_Construct_2(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's edge selection and immediate set new selection
        /// Generated from constructor `MR::ChangeMeshEdgeSelectionAction::ChangeMeshEdgeSelectionAction`.
        public unsafe Const_ChangeMeshEdgeSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh, MR.Misc._Moved<MR.UndirectedEdgeBitSet> newSelection) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh, MR.UndirectedEdgeBitSet._Underlying *newSelection);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshEdgeSelectionAction_Construct_3(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr, newSelection.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshEdgeSelectionAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshEdgeSelectionAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshEdgeSelectionAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshEdgeSelectionAction::selection`.
        public unsafe MR.Const_UndirectedEdgeBitSet Selection()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_selection", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ChangeMeshEdgeSelectionAction_selection(_Underlying *_this);
            return new(__MR_ChangeMeshEdgeSelectionAction_selection(_UnderlyingPtr), is_owning: false);
        }

        /// empty because set dirty is inside selectEdges
        /// Generated from method `MR::ChangeMeshEdgeSelectionAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshEdgeSelectionAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *_1);
            __MR_ChangeMeshEdgeSelectionAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshEdgeSelectionAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshEdgeSelectionAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshEdgeSelectionAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMesh edge selection
    /// Generated from class `MR::ChangeMeshEdgeSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshEdgeSelectionAction : Const_ChangeMeshEdgeSelectionAction
    {
        internal unsafe ChangeMeshEdgeSelectionAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshEdgeSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshEdgeSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshEdgeSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshEdgeSelectionAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshEdgeSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshEdgeSelectionAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshEdgeSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshEdgeSelectionAction::ChangeMeshEdgeSelectionAction`.
        public unsafe ChangeMeshEdgeSelectionAction(MR._ByValue_ChangeMeshEdgeSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshEdgeSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshEdgeSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's edge selection before making any changes in it
        /// Generated from constructor `MR::ChangeMeshEdgeSelectionAction::ChangeMeshEdgeSelectionAction`.
        public unsafe ChangeMeshEdgeSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshEdgeSelectionAction_Construct_2(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's edge selection and immediate set new selection
        /// Generated from constructor `MR::ChangeMeshEdgeSelectionAction::ChangeMeshEdgeSelectionAction`.
        public unsafe ChangeMeshEdgeSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh, MR.Misc._Moved<MR.UndirectedEdgeBitSet> newSelection) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh, MR.UndirectedEdgeBitSet._Underlying *newSelection);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshEdgeSelectionAction_Construct_3(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr, newSelection.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshEdgeSelectionAction::operator=`.
        public unsafe MR.ChangeMeshEdgeSelectionAction Assign(MR._ByValue_ChangeMeshEdgeSelectionAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshEdgeSelectionAction._Underlying *__MR_ChangeMeshEdgeSelectionAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshEdgeSelectionAction._Underlying *_other);
            return new(__MR_ChangeMeshEdgeSelectionAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshEdgeSelectionAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshEdgeSelectionAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshEdgeSelectionAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshEdgeSelectionAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshEdgeSelectionAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshEdgeSelectionAction`/`Const_ChangeMeshEdgeSelectionAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshEdgeSelectionAction
    {
        internal readonly Const_ChangeMeshEdgeSelectionAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshEdgeSelectionAction(Const_ChangeMeshEdgeSelectionAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshEdgeSelectionAction(Const_ChangeMeshEdgeSelectionAction arg) {return new(arg);}
        public _ByValue_ChangeMeshEdgeSelectionAction(MR.Misc._Moved<ChangeMeshEdgeSelectionAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshEdgeSelectionAction(MR.Misc._Moved<ChangeMeshEdgeSelectionAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshEdgeSelectionAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshEdgeSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshEdgeSelectionAction`/`Const_ChangeMeshEdgeSelectionAction` directly.
    public class _InOptMut_ChangeMeshEdgeSelectionAction
    {
        public ChangeMeshEdgeSelectionAction? Opt;

        public _InOptMut_ChangeMeshEdgeSelectionAction() {}
        public _InOptMut_ChangeMeshEdgeSelectionAction(ChangeMeshEdgeSelectionAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshEdgeSelectionAction(ChangeMeshEdgeSelectionAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshEdgeSelectionAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshEdgeSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshEdgeSelectionAction`/`Const_ChangeMeshEdgeSelectionAction` to pass it to the function.
    public class _InOptConst_ChangeMeshEdgeSelectionAction
    {
        public Const_ChangeMeshEdgeSelectionAction? Opt;

        public _InOptConst_ChangeMeshEdgeSelectionAction() {}
        public _InOptConst_ChangeMeshEdgeSelectionAction(Const_ChangeMeshEdgeSelectionAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshEdgeSelectionAction(Const_ChangeMeshEdgeSelectionAction value) {return new(value);}
    }

    /// Undo action for ObjectMesh creases
    /// Generated from class `MR::ChangeMeshCreasesAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshCreasesAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshCreasesAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshCreasesAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshCreasesAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshCreasesAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshCreasesAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshCreasesAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshCreasesAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshCreasesAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshCreasesAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshCreasesAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshCreasesAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshCreasesAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshCreasesAction::ChangeMeshCreasesAction`.
        public unsafe Const_ChangeMeshCreasesAction(MR._ByValue_ChangeMeshCreasesAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshCreasesAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshCreasesAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's creases before making any changes in it
        /// Generated from constructor `MR::ChangeMeshCreasesAction::ChangeMeshCreasesAction`.
        public unsafe Const_ChangeMeshCreasesAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshCreasesAction_Construct_2(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's current creases and immediate set new creases
        /// Generated from constructor `MR::ChangeMeshCreasesAction::ChangeMeshCreasesAction`.
        public unsafe Const_ChangeMeshCreasesAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh, MR.Misc._Moved<MR.UndirectedEdgeBitSet> newCreases) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh, MR.UndirectedEdgeBitSet._Underlying *newCreases);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshCreasesAction_Construct_3(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr, newCreases.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshCreasesAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshCreasesAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshCreasesAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshCreasesAction::creases`.
        public unsafe MR.Const_UndirectedEdgeBitSet Creases()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_creases", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ChangeMeshCreasesAction_creases(_Underlying *_this);
            return new(__MR_ChangeMeshCreasesAction_creases(_UnderlyingPtr), is_owning: false);
        }

        /// empty because set dirty is inside setCreases
        /// Generated from method `MR::ChangeMeshCreasesAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshCreasesAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *_1);
            __MR_ChangeMeshCreasesAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshCreasesAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshCreasesAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshCreasesAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMesh creases
    /// Generated from class `MR::ChangeMeshCreasesAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshCreasesAction : Const_ChangeMeshCreasesAction
    {
        internal unsafe ChangeMeshCreasesAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshCreasesAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshCreasesAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshCreasesAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshCreasesAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshCreasesAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshCreasesAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshCreasesAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshCreasesAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshCreasesAction::ChangeMeshCreasesAction`.
        public unsafe ChangeMeshCreasesAction(MR._ByValue_ChangeMeshCreasesAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshCreasesAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshCreasesAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's creases before making any changes in it
        /// Generated from constructor `MR::ChangeMeshCreasesAction::ChangeMeshCreasesAction`.
        public unsafe ChangeMeshCreasesAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshCreasesAction_Construct_2(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's current creases and immediate set new creases
        /// Generated from constructor `MR::ChangeMeshCreasesAction::ChangeMeshCreasesAction`.
        public unsafe ChangeMeshCreasesAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh objMesh, MR.Misc._Moved<MR.UndirectedEdgeBitSet> newCreases) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *objMesh, MR.UndirectedEdgeBitSet._Underlying *newCreases);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshCreasesAction_Construct_3(__ptr_name, __ptr_name + __len_name, objMesh._UnderlyingSharedPtr, newCreases.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshCreasesAction::operator=`.
        public unsafe MR.ChangeMeshCreasesAction Assign(MR._ByValue_ChangeMeshCreasesAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshCreasesAction._Underlying *__MR_ChangeMeshCreasesAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshCreasesAction._Underlying *_other);
            return new(__MR_ChangeMeshCreasesAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshCreasesAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshCreasesAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshCreasesAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshCreasesAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshCreasesAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshCreasesAction`/`Const_ChangeMeshCreasesAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshCreasesAction
    {
        internal readonly Const_ChangeMeshCreasesAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshCreasesAction(Const_ChangeMeshCreasesAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshCreasesAction(Const_ChangeMeshCreasesAction arg) {return new(arg);}
        public _ByValue_ChangeMeshCreasesAction(MR.Misc._Moved<ChangeMeshCreasesAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshCreasesAction(MR.Misc._Moved<ChangeMeshCreasesAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshCreasesAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshCreasesAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshCreasesAction`/`Const_ChangeMeshCreasesAction` directly.
    public class _InOptMut_ChangeMeshCreasesAction
    {
        public ChangeMeshCreasesAction? Opt;

        public _InOptMut_ChangeMeshCreasesAction() {}
        public _InOptMut_ChangeMeshCreasesAction(ChangeMeshCreasesAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshCreasesAction(ChangeMeshCreasesAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshCreasesAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshCreasesAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshCreasesAction`/`Const_ChangeMeshCreasesAction` to pass it to the function.
    public class _InOptConst_ChangeMeshCreasesAction
    {
        public Const_ChangeMeshCreasesAction? Opt;

        public _InOptConst_ChangeMeshCreasesAction() {}
        public _InOptConst_ChangeMeshCreasesAction(Const_ChangeMeshCreasesAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshCreasesAction(Const_ChangeMeshCreasesAction value) {return new(value);}
    }

    /// Undo action for ObjectPoints point selection
    /// Generated from class `MR::ChangePointPointSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePointPointSelectionAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePointPointSelectionAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePointPointSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePointPointSelectionAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePointPointSelectionAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePointPointSelectionAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePointPointSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePointPointSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointPointSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePointPointSelectionAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointPointSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointPointSelectionAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointPointSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointPointSelectionAction::ChangePointPointSelectionAction`.
        public unsafe Const_ChangePointPointSelectionAction(MR._ByValue_ChangePointPointSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointPointSelectionAction._Underlying *__MR_ChangePointPointSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointPointSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointPointSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's vertex selection before making any changes in it
        /// Generated from constructor `MR::ChangePointPointSelectionAction::ChangePointPointSelectionAction`.
        public unsafe Const_ChangePointPointSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints objPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePointPointSelectionAction._Underlying *__MR_ChangePointPointSelectionAction_Construct(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *objPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointPointSelectionAction_Construct(__ptr_name, __ptr_name + __len_name, objPoints._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePointPointSelectionAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePointPointSelectionAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePointPointSelectionAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePointPointSelectionAction::selection`.
        public unsafe MR.Const_VertBitSet Selection()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_selection", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_ChangePointPointSelectionAction_selection(_Underlying *_this);
            return new(__MR_ChangePointPointSelectionAction_selection(_UnderlyingPtr), is_owning: false);
        }

        /// empty because set dirty is inside selectPoints
        /// Generated from method `MR::ChangePointPointSelectionAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectPoints _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePointPointSelectionAction_setObjectDirty(MR.Const_ObjectPoints._UnderlyingShared *_1);
            __MR_ChangePointPointSelectionAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePointPointSelectionAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePointPointSelectionAction_heapBytes(_Underlying *_this);
            return __MR_ChangePointPointSelectionAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectPoints point selection
    /// Generated from class `MR::ChangePointPointSelectionAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePointPointSelectionAction : Const_ChangePointPointSelectionAction
    {
        internal unsafe ChangePointPointSelectionAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePointPointSelectionAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePointPointSelectionAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePointPointSelectionAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointPointSelectionAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePointPointSelectionAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointPointSelectionAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointPointSelectionAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointPointSelectionAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointPointSelectionAction::ChangePointPointSelectionAction`.
        public unsafe ChangePointPointSelectionAction(MR._ByValue_ChangePointPointSelectionAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointPointSelectionAction._Underlying *__MR_ChangePointPointSelectionAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointPointSelectionAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointPointSelectionAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's vertex selection before making any changes in it
        /// Generated from constructor `MR::ChangePointPointSelectionAction::ChangePointPointSelectionAction`.
        public unsafe ChangePointPointSelectionAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints objPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePointPointSelectionAction._Underlying *__MR_ChangePointPointSelectionAction_Construct(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *objPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointPointSelectionAction_Construct(__ptr_name, __ptr_name + __len_name, objPoints._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePointPointSelectionAction::operator=`.
        public unsafe MR.ChangePointPointSelectionAction Assign(MR._ByValue_ChangePointPointSelectionAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointPointSelectionAction._Underlying *__MR_ChangePointPointSelectionAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointPointSelectionAction._Underlying *_other);
            return new(__MR_ChangePointPointSelectionAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePointPointSelectionAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointPointSelectionAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePointPointSelectionAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePointPointSelectionAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePointPointSelectionAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePointPointSelectionAction`/`Const_ChangePointPointSelectionAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePointPointSelectionAction
    {
        internal readonly Const_ChangePointPointSelectionAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePointPointSelectionAction(Const_ChangePointPointSelectionAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePointPointSelectionAction(Const_ChangePointPointSelectionAction arg) {return new(arg);}
        public _ByValue_ChangePointPointSelectionAction(MR.Misc._Moved<ChangePointPointSelectionAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePointPointSelectionAction(MR.Misc._Moved<ChangePointPointSelectionAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePointPointSelectionAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePointPointSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointPointSelectionAction`/`Const_ChangePointPointSelectionAction` directly.
    public class _InOptMut_ChangePointPointSelectionAction
    {
        public ChangePointPointSelectionAction? Opt;

        public _InOptMut_ChangePointPointSelectionAction() {}
        public _InOptMut_ChangePointPointSelectionAction(ChangePointPointSelectionAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePointPointSelectionAction(ChangePointPointSelectionAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePointPointSelectionAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePointPointSelectionAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointPointSelectionAction`/`Const_ChangePointPointSelectionAction` to pass it to the function.
    public class _InOptConst_ChangePointPointSelectionAction
    {
        public Const_ChangePointPointSelectionAction? Opt;

        public _InOptConst_ChangePointPointSelectionAction() {}
        public _InOptConst_ChangePointPointSelectionAction(Const_ChangePointPointSelectionAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePointPointSelectionAction(Const_ChangePointPointSelectionAction value) {return new(value);}
    }
}
