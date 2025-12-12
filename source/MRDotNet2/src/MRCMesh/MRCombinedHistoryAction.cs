public static partial class MR
{
    /// History action for combine some history actions
    /// Generated from class `MR::CombinedHistoryAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_CombinedHistoryAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_CombinedHistoryAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_CombinedHistoryAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_CombinedHistoryAction_UseCount();
                return __MR_std_shared_ptr_MR_CombinedHistoryAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_CombinedHistoryAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_CombinedHistoryAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_CombinedHistoryAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_CombinedHistoryAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe CombinedHistoryAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_CombinedHistoryAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_CombinedHistoryAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_CombinedHistoryAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_CombinedHistoryAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_CombinedHistoryAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_CombinedHistoryAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CombinedHistoryAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_CombinedHistoryAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_CombinedHistoryAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_CombinedHistoryAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_CombinedHistoryAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_CombinedHistoryAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_CombinedHistoryAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_CombinedHistoryAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::CombinedHistoryAction::CombinedHistoryAction`.
        public unsafe Const_CombinedHistoryAction(MR._ByValue_CombinedHistoryAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CombinedHistoryAction._Underlying *__MR_CombinedHistoryAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CombinedHistoryAction._Underlying *_other);
            _LateMakeShared(__MR_CombinedHistoryAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Will call action() for each actions in given order (undo in reverse, redo in forward)
        /// Generated from constructor `MR::CombinedHistoryAction::CombinedHistoryAction`.
        public unsafe Const_CombinedHistoryAction(ReadOnlySpan<char> name, MR.Std.Const_Vector_StdSharedPtrMRHistoryAction actions) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_Construct", ExactSpelling = true)]
            extern static MR.CombinedHistoryAction._Underlying *__MR_CombinedHistoryAction_Construct(byte *name, byte *name_end, MR.Std.Const_Vector_StdSharedPtrMRHistoryAction._Underlying *actions);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_CombinedHistoryAction_Construct(__ptr_name, __ptr_name + __len_name, actions._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::CombinedHistoryAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_CombinedHistoryAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_CombinedHistoryAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::CombinedHistoryAction::getStack`.
        public unsafe MR.Std.Const_Vector_StdSharedPtrMRHistoryAction GetStack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_getStack_const", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_StdSharedPtrMRHistoryAction._Underlying *__MR_CombinedHistoryAction_getStack_const(_Underlying *_this);
            return new(__MR_CombinedHistoryAction_getStack_const(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::CombinedHistoryAction::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_empty", ExactSpelling = true)]
            extern static byte __MR_CombinedHistoryAction_empty(_Underlying *_this);
            return __MR_CombinedHistoryAction_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::CombinedHistoryAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_CombinedHistoryAction_heapBytes(_Underlying *_this);
            return __MR_CombinedHistoryAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for combine some history actions
    /// Generated from class `MR::CombinedHistoryAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class CombinedHistoryAction : Const_CombinedHistoryAction
    {
        internal unsafe CombinedHistoryAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe CombinedHistoryAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(CombinedHistoryAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_CombinedHistoryAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_CombinedHistoryAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator CombinedHistoryAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_CombinedHistoryAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_CombinedHistoryAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_CombinedHistoryAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::CombinedHistoryAction::CombinedHistoryAction`.
        public unsafe CombinedHistoryAction(MR._ByValue_CombinedHistoryAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CombinedHistoryAction._Underlying *__MR_CombinedHistoryAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CombinedHistoryAction._Underlying *_other);
            _LateMakeShared(__MR_CombinedHistoryAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Will call action() for each actions in given order (undo in reverse, redo in forward)
        /// Generated from constructor `MR::CombinedHistoryAction::CombinedHistoryAction`.
        public unsafe CombinedHistoryAction(ReadOnlySpan<char> name, MR.Std.Const_Vector_StdSharedPtrMRHistoryAction actions) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_Construct", ExactSpelling = true)]
            extern static MR.CombinedHistoryAction._Underlying *__MR_CombinedHistoryAction_Construct(byte *name, byte *name_end, MR.Std.Const_Vector_StdSharedPtrMRHistoryAction._Underlying *actions);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_CombinedHistoryAction_Construct(__ptr_name, __ptr_name + __len_name, actions._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::CombinedHistoryAction::operator=`.
        public unsafe MR.CombinedHistoryAction Assign(MR._ByValue_CombinedHistoryAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CombinedHistoryAction._Underlying *__MR_CombinedHistoryAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CombinedHistoryAction._Underlying *_other);
            return new(__MR_CombinedHistoryAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::CombinedHistoryAction::action`.
        public unsafe void Action(MR.HistoryAction.Type type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_action", ExactSpelling = true)]
            extern static void __MR_CombinedHistoryAction_action(_Underlying *_this, MR.HistoryAction.Type type);
            __MR_CombinedHistoryAction_action(_UnderlyingPtr, type);
        }

        /// Generated from method `MR::CombinedHistoryAction::getStack`.
        public unsafe new MR.Std.Vector_StdSharedPtrMRHistoryAction GetStack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_getStack", ExactSpelling = true)]
            extern static MR.Std.Vector_StdSharedPtrMRHistoryAction._Underlying *__MR_CombinedHistoryAction_getStack(_Underlying *_this);
            return new(__MR_CombinedHistoryAction_getStack(_UnderlyingPtr), is_owning: false);
        }

        /// Remove some actions according to condition inside combined actions.
        /// Do deep filtering.
        /// Generated from method `MR::CombinedHistoryAction::filter`.
        public unsafe bool Filter(MR.Std._ByValue_Function_BoolFuncFromConstStdSharedPtrMRHistoryActionRef filteringCondition)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CombinedHistoryAction_filter", ExactSpelling = true)]
            extern static byte __MR_CombinedHistoryAction_filter(_Underlying *_this, MR.Misc._PassBy filteringCondition_pass_by, MR.Std.Function_BoolFuncFromConstStdSharedPtrMRHistoryActionRef._Underlying *filteringCondition);
            return __MR_CombinedHistoryAction_filter(_UnderlyingPtr, filteringCondition.PassByMode, filteringCondition.Value is not null ? filteringCondition.Value._UnderlyingPtr : null) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `CombinedHistoryAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CombinedHistoryAction`/`Const_CombinedHistoryAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CombinedHistoryAction
    {
        internal readonly Const_CombinedHistoryAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CombinedHistoryAction(Const_CombinedHistoryAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CombinedHistoryAction(Const_CombinedHistoryAction arg) {return new(arg);}
        public _ByValue_CombinedHistoryAction(MR.Misc._Moved<CombinedHistoryAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CombinedHistoryAction(MR.Misc._Moved<CombinedHistoryAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CombinedHistoryAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CombinedHistoryAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CombinedHistoryAction`/`Const_CombinedHistoryAction` directly.
    public class _InOptMut_CombinedHistoryAction
    {
        public CombinedHistoryAction? Opt;

        public _InOptMut_CombinedHistoryAction() {}
        public _InOptMut_CombinedHistoryAction(CombinedHistoryAction value) {Opt = value;}
        public static implicit operator _InOptMut_CombinedHistoryAction(CombinedHistoryAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `CombinedHistoryAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CombinedHistoryAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CombinedHistoryAction`/`Const_CombinedHistoryAction` to pass it to the function.
    public class _InOptConst_CombinedHistoryAction
    {
        public Const_CombinedHistoryAction? Opt;

        public _InOptConst_CombinedHistoryAction() {}
        public _InOptConst_CombinedHistoryAction(Const_CombinedHistoryAction value) {Opt = value;}
        public static implicit operator _InOptConst_CombinedHistoryAction(Const_CombinedHistoryAction value) {return new(value);}
    }
}
