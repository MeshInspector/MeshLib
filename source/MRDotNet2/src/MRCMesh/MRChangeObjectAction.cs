public static partial class MR
{
    /// History action for object change
    /// Generated from class `MR::ChangeObjectAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeObjectAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeObjectAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeObjectAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeObjectAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeObjectAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeObjectAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeObjectAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeObjectAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeObjectAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeObjectAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeObjectAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeObjectAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeObjectAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeObjectAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeObjectAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeObjectAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectAction::ChangeObjectAction`.
        public unsafe Const_ChangeObjectAction(MR._ByValue_ChangeObjectAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectAction._Underlying *__MR_ChangeObjectAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Constructed from original Object
        /// Generated from constructor `MR::ChangeObjectAction::ChangeObjectAction`.
        public unsafe Const_ChangeObjectAction(ReadOnlySpan<char> name, MR.Const_Object obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeObjectAction._Underlying *__MR_ChangeObjectAction_Construct(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeObjectAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeObjectAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeObjectAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeObjectAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeObjectAction_heapBytes(_Underlying *_this);
            return __MR_ChangeObjectAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for object change
    /// Generated from class `MR::ChangeObjectAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeObjectAction : Const_ChangeObjectAction
    {
        internal unsafe ChangeObjectAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeObjectAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeObjectAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeObjectAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeObjectAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectAction::ChangeObjectAction`.
        public unsafe ChangeObjectAction(MR._ByValue_ChangeObjectAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectAction._Underlying *__MR_ChangeObjectAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Constructed from original Object
        /// Generated from constructor `MR::ChangeObjectAction::ChangeObjectAction`.
        public unsafe ChangeObjectAction(ReadOnlySpan<char> name, MR.Const_Object obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeObjectAction._Underlying *__MR_ChangeObjectAction_Construct(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeObjectAction::operator=`.
        public unsafe MR.ChangeObjectAction Assign(MR._ByValue_ChangeObjectAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectAction._Underlying *__MR_ChangeObjectAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectAction._Underlying *_other);
            return new(__MR_ChangeObjectAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeObjectAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeObjectAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeObjectAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeObjectAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeObjectAction`/`Const_ChangeObjectAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeObjectAction
    {
        internal readonly Const_ChangeObjectAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeObjectAction(Const_ChangeObjectAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeObjectAction(Const_ChangeObjectAction arg) {return new(arg);}
        public _ByValue_ChangeObjectAction(MR.Misc._Moved<ChangeObjectAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeObjectAction(MR.Misc._Moved<ChangeObjectAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeObjectAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeObjectAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectAction`/`Const_ChangeObjectAction` directly.
    public class _InOptMut_ChangeObjectAction
    {
        public ChangeObjectAction? Opt;

        public _InOptMut_ChangeObjectAction() {}
        public _InOptMut_ChangeObjectAction(ChangeObjectAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeObjectAction(ChangeObjectAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeObjectAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeObjectAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectAction`/`Const_ChangeObjectAction` to pass it to the function.
    public class _InOptConst_ChangeObjectAction
    {
        public Const_ChangeObjectAction? Opt;

        public _InOptConst_ChangeObjectAction() {}
        public _InOptConst_ChangeObjectAction(Const_ChangeObjectAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeObjectAction(Const_ChangeObjectAction value) {return new(value);}
    }
}
