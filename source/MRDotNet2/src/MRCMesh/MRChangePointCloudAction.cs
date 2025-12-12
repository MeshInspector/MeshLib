public static partial class MR
{
    /// Undo action for ObjectMesh mesh change
    /// Generated from class `MR::ChangePointCloudAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePointCloudAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePointCloudAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePointCloudAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePointCloudAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePointCloudAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePointCloudAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePointCloudAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePointCloudAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePointCloudAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePointCloudAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePointCloudAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePointCloudAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePointCloudAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePointCloudAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointCloudAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePointCloudAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointCloudAction::ChangePointCloudAction`.
        public unsafe Const_ChangePointCloudAction(MR._ByValue_ChangePointCloudAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudAction._Underlying *__MR_ChangePointCloudAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointCloudAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's point cloud before making any changes in it
        /// Generated from constructor `MR::ChangePointCloudAction::ChangePointCloudAction`.
        public unsafe Const_ChangePointCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePointCloudAction._Underlying *__MR_ChangePointCloudAction_Construct(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePointCloudAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePointCloudAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePointCloudAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePointCloudAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectPoints obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePointCloudAction_setObjectDirty(MR.Const_ObjectPoints._UnderlyingShared *obj);
            __MR_ChangePointCloudAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePointCloudAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePointCloudAction_heapBytes(_Underlying *_this);
            return __MR_ChangePointCloudAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMesh mesh change
    /// Generated from class `MR::ChangePointCloudAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePointCloudAction : Const_ChangePointCloudAction
    {
        internal unsafe ChangePointCloudAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePointCloudAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePointCloudAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePointCloudAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointCloudAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePointCloudAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointCloudAction::ChangePointCloudAction`.
        public unsafe ChangePointCloudAction(MR._ByValue_ChangePointCloudAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudAction._Underlying *__MR_ChangePointCloudAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointCloudAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's point cloud before making any changes in it
        /// Generated from constructor `MR::ChangePointCloudAction::ChangePointCloudAction`.
        public unsafe ChangePointCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePointCloudAction._Underlying *__MR_ChangePointCloudAction_Construct(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePointCloudAction::operator=`.
        public unsafe MR.ChangePointCloudAction Assign(MR._ByValue_ChangePointCloudAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudAction._Underlying *__MR_ChangePointCloudAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudAction._Underlying *_other);
            return new(__MR_ChangePointCloudAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePointCloudAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePointCloudAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePointCloudAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePointCloudAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePointCloudAction`/`Const_ChangePointCloudAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePointCloudAction
    {
        internal readonly Const_ChangePointCloudAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePointCloudAction(Const_ChangePointCloudAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePointCloudAction(Const_ChangePointCloudAction arg) {return new(arg);}
        public _ByValue_ChangePointCloudAction(MR.Misc._Moved<ChangePointCloudAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePointCloudAction(MR.Misc._Moved<ChangePointCloudAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePointCloudAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePointCloudAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointCloudAction`/`Const_ChangePointCloudAction` directly.
    public class _InOptMut_ChangePointCloudAction
    {
        public ChangePointCloudAction? Opt;

        public _InOptMut_ChangePointCloudAction() {}
        public _InOptMut_ChangePointCloudAction(ChangePointCloudAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePointCloudAction(ChangePointCloudAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePointCloudAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePointCloudAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointCloudAction`/`Const_ChangePointCloudAction` to pass it to the function.
    public class _InOptConst_ChangePointCloudAction
    {
        public Const_ChangePointCloudAction? Opt;

        public _InOptConst_ChangePointCloudAction() {}
        public _InOptConst_ChangePointCloudAction(Const_ChangePointCloudAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePointCloudAction(Const_ChangePointCloudAction value) {return new(value);}
    }

    /// Undo action for points field inside ObjectPoints
    /// Generated from class `MR::ChangePointCloudPointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePointCloudPointsAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePointCloudPointsAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePointCloudPointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePointCloudPointsAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePointCloudPointsAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePointCloudPointsAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePointCloudPointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePointCloudPointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointCloudPointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePointCloudPointsAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudPointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudPointsAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudPointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointCloudPointsAction::ChangePointCloudPointsAction`.
        public unsafe Const_ChangePointCloudPointsAction(MR._ByValue_ChangePointCloudPointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudPointsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointCloudPointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's points field before making any changes in it
        /// Generated from constructor `MR::ChangePointCloudPointsAction::ChangePointCloudPointsAction`.
        public unsafe Const_ChangePointCloudPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudPointsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's points field and immediate set new value
        /// Generated from constructor `MR::ChangePointCloudPointsAction::ChangePointCloudPointsAction`.
        public unsafe Const_ChangePointCloudPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.Misc._Moved<MR.VertCoords> newPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertCoords._Underlying *newPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudPointsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newPoints.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangePointCloudPointsAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePointCloudPointsAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePointCloudPointsAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePointCloudPointsAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectPoints obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePointCloudPointsAction_setObjectDirty(MR.Const_ObjectPoints._UnderlyingShared *obj);
            __MR_ChangePointCloudPointsAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePointCloudPointsAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePointCloudPointsAction_heapBytes(_Underlying *_this);
            return __MR_ChangePointCloudPointsAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for points field inside ObjectPoints
    /// Generated from class `MR::ChangePointCloudPointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePointCloudPointsAction : Const_ChangePointCloudPointsAction
    {
        internal unsafe ChangePointCloudPointsAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePointCloudPointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePointCloudPointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePointCloudPointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointCloudPointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePointCloudPointsAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudPointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudPointsAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudPointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointCloudPointsAction::ChangePointCloudPointsAction`.
        public unsafe ChangePointCloudPointsAction(MR._ByValue_ChangePointCloudPointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudPointsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointCloudPointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's points field before making any changes in it
        /// Generated from constructor `MR::ChangePointCloudPointsAction::ChangePointCloudPointsAction`.
        public unsafe ChangePointCloudPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudPointsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's points field and immediate set new value
        /// Generated from constructor `MR::ChangePointCloudPointsAction::ChangePointCloudPointsAction`.
        public unsafe ChangePointCloudPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.Misc._Moved<MR.VertCoords> newPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertCoords._Underlying *newPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudPointsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newPoints.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangePointCloudPointsAction::operator=`.
        public unsafe MR.ChangePointCloudPointsAction Assign(MR._ByValue_ChangePointCloudPointsAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudPointsAction._Underlying *__MR_ChangePointCloudPointsAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudPointsAction._Underlying *_other);
            return new(__MR_ChangePointCloudPointsAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePointCloudPointsAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudPointsAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePointCloudPointsAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePointCloudPointsAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePointCloudPointsAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePointCloudPointsAction`/`Const_ChangePointCloudPointsAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePointCloudPointsAction
    {
        internal readonly Const_ChangePointCloudPointsAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePointCloudPointsAction(Const_ChangePointCloudPointsAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePointCloudPointsAction(Const_ChangePointCloudPointsAction arg) {return new(arg);}
        public _ByValue_ChangePointCloudPointsAction(MR.Misc._Moved<ChangePointCloudPointsAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePointCloudPointsAction(MR.Misc._Moved<ChangePointCloudPointsAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePointCloudPointsAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePointCloudPointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointCloudPointsAction`/`Const_ChangePointCloudPointsAction` directly.
    public class _InOptMut_ChangePointCloudPointsAction
    {
        public ChangePointCloudPointsAction? Opt;

        public _InOptMut_ChangePointCloudPointsAction() {}
        public _InOptMut_ChangePointCloudPointsAction(ChangePointCloudPointsAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePointCloudPointsAction(ChangePointCloudPointsAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePointCloudPointsAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePointCloudPointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointCloudPointsAction`/`Const_ChangePointCloudPointsAction` to pass it to the function.
    public class _InOptConst_ChangePointCloudPointsAction
    {
        public Const_ChangePointCloudPointsAction? Opt;

        public _InOptConst_ChangePointCloudPointsAction() {}
        public _InOptConst_ChangePointCloudPointsAction(Const_ChangePointCloudPointsAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePointCloudPointsAction(Const_ChangePointCloudPointsAction value) {return new(value);}
    }

    /// Undo action that modifies one point's coordinates inside ObjectPoints
    /// Generated from class `MR::ChangeOnePointInCloudAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeOnePointInCloudAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeOnePointInCloudAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeOnePointInCloudAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeOnePointInCloudAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeOnePointInCloudAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeOnePointInCloudAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeOnePointInCloudAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeOnePointInCloudAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeOnePointInCloudAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeOnePointInCloudAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInCloudAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInCloudAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInCloudAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeOnePointInCloudAction::ChangeOnePointInCloudAction`.
        public unsafe Const_ChangeOnePointInCloudAction(MR._ByValue_ChangeOnePointInCloudAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInCloudAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeOnePointInCloudAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point's coordinates before making any changes in it
        /// Generated from constructor `MR::ChangeOnePointInCloudAction::ChangeOnePointInCloudAction`.
        public unsafe Const_ChangeOnePointInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInCloudAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId));
            }
        }

        /// use this constructor to remember point's coordinates and immediate set new coordinates
        /// Generated from constructor `MR::ChangeOnePointInCloudAction::ChangeOnePointInCloudAction`.
        public unsafe Const_ChangeOnePointInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId, MR.Const_Vector3f newCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_Construct_4(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId, MR.Const_Vector3f._Underlying *newCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInCloudAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId, newCoords._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeOnePointInCloudAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeOnePointInCloudAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeOnePointInCloudAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeOnePointInCloudAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectPoints obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeOnePointInCloudAction_setObjectDirty(MR.Const_ObjectPoints._UnderlyingShared *obj);
            __MR_ChangeOnePointInCloudAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeOnePointInCloudAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeOnePointInCloudAction_heapBytes(_Underlying *_this);
            return __MR_ChangeOnePointInCloudAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action that modifies one point's coordinates inside ObjectPoints
    /// Generated from class `MR::ChangeOnePointInCloudAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeOnePointInCloudAction : Const_ChangeOnePointInCloudAction
    {
        internal unsafe ChangeOnePointInCloudAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeOnePointInCloudAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeOnePointInCloudAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeOnePointInCloudAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeOnePointInCloudAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeOnePointInCloudAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInCloudAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInCloudAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInCloudAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeOnePointInCloudAction::ChangeOnePointInCloudAction`.
        public unsafe ChangeOnePointInCloudAction(MR._ByValue_ChangeOnePointInCloudAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInCloudAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeOnePointInCloudAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point's coordinates before making any changes in it
        /// Generated from constructor `MR::ChangeOnePointInCloudAction::ChangeOnePointInCloudAction`.
        public unsafe ChangeOnePointInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInCloudAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId));
            }
        }

        /// use this constructor to remember point's coordinates and immediate set new coordinates
        /// Generated from constructor `MR::ChangeOnePointInCloudAction::ChangeOnePointInCloudAction`.
        public unsafe ChangeOnePointInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId, MR.Const_Vector3f newCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_Construct_4(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId, MR.Const_Vector3f._Underlying *newCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInCloudAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId, newCoords._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeOnePointInCloudAction::operator=`.
        public unsafe MR.ChangeOnePointInCloudAction Assign(MR._ByValue_ChangeOnePointInCloudAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOnePointInCloudAction._Underlying *__MR_ChangeOnePointInCloudAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInCloudAction._Underlying *_other);
            return new(__MR_ChangeOnePointInCloudAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeOnePointInCloudAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInCloudAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeOnePointInCloudAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeOnePointInCloudAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeOnePointInCloudAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeOnePointInCloudAction`/`Const_ChangeOnePointInCloudAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeOnePointInCloudAction
    {
        internal readonly Const_ChangeOnePointInCloudAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeOnePointInCloudAction(Const_ChangeOnePointInCloudAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeOnePointInCloudAction(Const_ChangeOnePointInCloudAction arg) {return new(arg);}
        public _ByValue_ChangeOnePointInCloudAction(MR.Misc._Moved<ChangeOnePointInCloudAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeOnePointInCloudAction(MR.Misc._Moved<ChangeOnePointInCloudAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeOnePointInCloudAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeOnePointInCloudAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeOnePointInCloudAction`/`Const_ChangeOnePointInCloudAction` directly.
    public class _InOptMut_ChangeOnePointInCloudAction
    {
        public ChangeOnePointInCloudAction? Opt;

        public _InOptMut_ChangeOnePointInCloudAction() {}
        public _InOptMut_ChangeOnePointInCloudAction(ChangeOnePointInCloudAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeOnePointInCloudAction(ChangeOnePointInCloudAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeOnePointInCloudAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeOnePointInCloudAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeOnePointInCloudAction`/`Const_ChangeOnePointInCloudAction` to pass it to the function.
    public class _InOptConst_ChangeOnePointInCloudAction
    {
        public Const_ChangeOnePointInCloudAction? Opt;

        public _InOptConst_ChangeOnePointInCloudAction() {}
        public _InOptConst_ChangeOnePointInCloudAction(Const_ChangeOnePointInCloudAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeOnePointInCloudAction(Const_ChangeOnePointInCloudAction value) {return new(value);}
    }
}
