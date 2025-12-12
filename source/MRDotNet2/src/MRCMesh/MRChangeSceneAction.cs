public static partial class MR
{
    /// Change scene action
    /// Generated from class `MR::ChangeSceneAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeSceneAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeSceneAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeSceneAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeSceneAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeSceneAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSceneAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeSceneAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeSceneAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSceneAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSceneAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeSceneAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeSceneAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeSceneAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeSceneAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSceneAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeSceneAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSceneAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeSceneAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSceneAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeSceneAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeSceneAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeSceneAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeSceneAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeSceneAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeSceneAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeSceneAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeSceneAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeSceneAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeSceneAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeSceneAction::ChangeSceneAction`.
        public unsafe Const_ChangeSceneAction(MR._ByValue_ChangeSceneAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeSceneAction._Underlying *__MR_ChangeSceneAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeSceneAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeSceneAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Constructed before removal or addition
        /// Generated from constructor `MR::ChangeSceneAction::ChangeSceneAction`.
        public unsafe Const_ChangeSceneAction(ReadOnlySpan<char> name, MR.Const_Object obj, MR.ChangeSceneAction.Type type) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeSceneAction._Underlying *__MR_ChangeSceneAction_Construct(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj, MR.ChangeSceneAction.Type type);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeSceneAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, type));
            }
        }

        /// Generated from method `MR::ChangeSceneAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeSceneAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeSceneAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeSceneAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeSceneAction_heapBytes(_Underlying *_this);
            return __MR_ChangeSceneAction_heapBytes(_UnderlyingPtr);
        }

        public enum Type : int
        {
            AddObject = 0,
            RemoveObject = 1,
        }
    }

    /// Change scene action
    /// Generated from class `MR::ChangeSceneAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeSceneAction : Const_ChangeSceneAction
    {
        internal unsafe ChangeSceneAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeSceneAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeSceneAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeSceneAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeSceneAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeSceneAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeSceneAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeSceneAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeSceneAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeSceneAction::ChangeSceneAction`.
        public unsafe ChangeSceneAction(MR._ByValue_ChangeSceneAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeSceneAction._Underlying *__MR_ChangeSceneAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeSceneAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeSceneAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Constructed before removal or addition
        /// Generated from constructor `MR::ChangeSceneAction::ChangeSceneAction`.
        public unsafe ChangeSceneAction(ReadOnlySpan<char> name, MR.Const_Object obj, MR.ChangeSceneAction.Type type) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeSceneAction._Underlying *__MR_ChangeSceneAction_Construct(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj, MR.ChangeSceneAction.Type type);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeSceneAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, type));
            }
        }

        /// Generated from method `MR::ChangeSceneAction::operator=`.
        public unsafe MR.ChangeSceneAction Assign(MR._ByValue_ChangeSceneAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeSceneAction._Underlying *__MR_ChangeSceneAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeSceneAction._Underlying *_other);
            return new(__MR_ChangeSceneAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeSceneAction::action`.
        public unsafe void Action(MR.HistoryAction.Type actionType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSceneAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeSceneAction_action(_Underlying *_this, MR.HistoryAction.Type actionType);
            __MR_ChangeSceneAction_action(_UnderlyingPtr, actionType);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeSceneAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeSceneAction`/`Const_ChangeSceneAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeSceneAction
    {
        internal readonly Const_ChangeSceneAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeSceneAction(Const_ChangeSceneAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeSceneAction(Const_ChangeSceneAction arg) {return new(arg);}
        public _ByValue_ChangeSceneAction(MR.Misc._Moved<ChangeSceneAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeSceneAction(MR.Misc._Moved<ChangeSceneAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeSceneAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeSceneAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeSceneAction`/`Const_ChangeSceneAction` directly.
    public class _InOptMut_ChangeSceneAction
    {
        public ChangeSceneAction? Opt;

        public _InOptMut_ChangeSceneAction() {}
        public _InOptMut_ChangeSceneAction(ChangeSceneAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeSceneAction(ChangeSceneAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeSceneAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeSceneAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeSceneAction`/`Const_ChangeSceneAction` to pass it to the function.
    public class _InOptConst_ChangeSceneAction
    {
        public Const_ChangeSceneAction? Opt;

        public _InOptConst_ChangeSceneAction() {}
        public _InOptConst_ChangeSceneAction(Const_ChangeSceneAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeSceneAction(Const_ChangeSceneAction value) {return new(value);}
    }
}
