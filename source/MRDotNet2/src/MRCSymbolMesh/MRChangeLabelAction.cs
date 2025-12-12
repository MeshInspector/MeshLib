public static partial class MR
{
    /// Generated from class `MR::ChangeLabelAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeLabelAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeLabelAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeLabelAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeLabelAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeLabelAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLabelAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeLabelAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeLabelAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLabelAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLabelAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeLabelAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeLabelAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeLabelAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeLabelAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLabelAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeLabelAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLabelAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeLabelAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLabelAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeLabelAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeLabelAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeLabelAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeLabelAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeLabelAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeLabelAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeLabelAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeLabelAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeLabelAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeLabelAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeLabelAction::ChangeLabelAction`.
        public unsafe Const_ChangeLabelAction(MR._ByValue_ChangeLabelAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeLabelAction._Underlying *__MR_ChangeLabelAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeLabelAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeLabelAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from constructor `MR::ChangeLabelAction::ChangeLabelAction`.
        public unsafe Const_ChangeLabelAction(ReadOnlySpan<char> actionName, MR._ByValue_ObjectLabel obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeLabelAction._Underlying *__MR_ChangeLabelAction_Construct(byte *actionName, byte *actionName_end, MR.Misc._PassBy obj_pass_by, MR.ObjectLabel._UnderlyingShared *obj);
            byte[] __bytes_actionName = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(actionName.Length)];
            int __len_actionName = System.Text.Encoding.UTF8.GetBytes(actionName, __bytes_actionName);
            fixed (byte *__ptr_actionName = __bytes_actionName)
            {
                _LateMakeShared(__MR_ChangeLabelAction_Construct(__ptr_actionName, __ptr_actionName + __len_actionName, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null));
            }
        }

        /// Generated from method `MR::ChangeLabelAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeLabelAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeLabelAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeLabelAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeLabelAction_heapBytes(_Underlying *_this);
            return __MR_ChangeLabelAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ChangeLabelAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeLabelAction : Const_ChangeLabelAction
    {
        internal unsafe ChangeLabelAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeLabelAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeLabelAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeLabelAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeLabelAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeLabelAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeLabelAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeLabelAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeLabelAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeLabelAction::ChangeLabelAction`.
        public unsafe ChangeLabelAction(MR._ByValue_ChangeLabelAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeLabelAction._Underlying *__MR_ChangeLabelAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeLabelAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeLabelAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from constructor `MR::ChangeLabelAction::ChangeLabelAction`.
        public unsafe ChangeLabelAction(ReadOnlySpan<char> actionName, MR._ByValue_ObjectLabel obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeLabelAction._Underlying *__MR_ChangeLabelAction_Construct(byte *actionName, byte *actionName_end, MR.Misc._PassBy obj_pass_by, MR.ObjectLabel._UnderlyingShared *obj);
            byte[] __bytes_actionName = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(actionName.Length)];
            int __len_actionName = System.Text.Encoding.UTF8.GetBytes(actionName, __bytes_actionName);
            fixed (byte *__ptr_actionName = __bytes_actionName)
            {
                _LateMakeShared(__MR_ChangeLabelAction_Construct(__ptr_actionName, __ptr_actionName + __len_actionName, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null));
            }
        }

        /// Generated from method `MR::ChangeLabelAction::operator=`.
        public unsafe MR.ChangeLabelAction Assign(MR._ByValue_ChangeLabelAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeLabelAction._Underlying *__MR_ChangeLabelAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeLabelAction._Underlying *_other);
            return new(__MR_ChangeLabelAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeLabelAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLabelAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeLabelAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeLabelAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeLabelAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeLabelAction`/`Const_ChangeLabelAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeLabelAction
    {
        internal readonly Const_ChangeLabelAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeLabelAction(Const_ChangeLabelAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeLabelAction(Const_ChangeLabelAction arg) {return new(arg);}
        public _ByValue_ChangeLabelAction(MR.Misc._Moved<ChangeLabelAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeLabelAction(MR.Misc._Moved<ChangeLabelAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeLabelAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeLabelAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeLabelAction`/`Const_ChangeLabelAction` directly.
    public class _InOptMut_ChangeLabelAction
    {
        public ChangeLabelAction? Opt;

        public _InOptMut_ChangeLabelAction() {}
        public _InOptMut_ChangeLabelAction(ChangeLabelAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeLabelAction(ChangeLabelAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeLabelAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeLabelAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeLabelAction`/`Const_ChangeLabelAction` to pass it to the function.
    public class _InOptConst_ChangeLabelAction
    {
        public Const_ChangeLabelAction? Opt;

        public _InOptConst_ChangeLabelAction() {}
        public _InOptConst_ChangeLabelAction(Const_ChangeLabelAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeLabelAction(Const_ChangeLabelAction value) {return new(value);}
    }
}
