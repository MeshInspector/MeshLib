public static partial class MR
{
    /// History action for visualizeMaskType change
    /// Generated from class `MR::ChangeVisualizePropertyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeVisualizePropertyAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeVisualizePropertyAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeVisualizePropertyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeVisualizePropertyAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeVisualizePropertyAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeVisualizePropertyAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeVisualizePropertyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeVisualizePropertyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeVisualizePropertyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeVisualizePropertyAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeVisualizePropertyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeVisualizePropertyAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeVisualizePropertyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeVisualizePropertyAction::ChangeVisualizePropertyAction`.
        public unsafe Const_ChangeVisualizePropertyAction(MR._ByValue_ChangeVisualizePropertyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeVisualizePropertyAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeVisualizePropertyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's visualize property mask before making any changes in it
        /// Generated from constructor `MR::ChangeVisualizePropertyAction::ChangeVisualizePropertyAction`.
        public unsafe Const_ChangeVisualizePropertyAction(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.Const_AnyVisualizeMaskEnum visualizeMaskType) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_Construct_3(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.AnyVisualizeMaskEnum._Underlying *visualizeMaskType);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeVisualizePropertyAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, visualizeMaskType._UnderlyingPtr));
            }
        }

        /// use this constructor to remember object's visualize property mask and immediately set new value
        /// Generated from constructor `MR::ChangeVisualizePropertyAction::ChangeVisualizePropertyAction`.
        public unsafe Const_ChangeVisualizePropertyAction(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.Const_AnyVisualizeMaskEnum visualizeMaskType, MR.Const_ViewportMask newMask) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_Construct_4(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.AnyVisualizeMaskEnum._Underlying *visualizeMaskType, MR.ViewportMask._Underlying *newMask);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeVisualizePropertyAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, visualizeMaskType._UnderlyingPtr, newMask._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeVisualizePropertyAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeVisualizePropertyAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeVisualizePropertyAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeVisualizePropertyAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_VisualObject _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeVisualizePropertyAction_setObjectDirty(MR.Const_VisualObject._UnderlyingShared *_1);
            __MR_ChangeVisualizePropertyAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeVisualizePropertyAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeVisualizePropertyAction_heapBytes(_Underlying *_this);
            return __MR_ChangeVisualizePropertyAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for visualizeMaskType change
    /// Generated from class `MR::ChangeVisualizePropertyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeVisualizePropertyAction : Const_ChangeVisualizePropertyAction
    {
        internal unsafe ChangeVisualizePropertyAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeVisualizePropertyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeVisualizePropertyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeVisualizePropertyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeVisualizePropertyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeVisualizePropertyAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeVisualizePropertyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeVisualizePropertyAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeVisualizePropertyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeVisualizePropertyAction::ChangeVisualizePropertyAction`.
        public unsafe ChangeVisualizePropertyAction(MR._ByValue_ChangeVisualizePropertyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeVisualizePropertyAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeVisualizePropertyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's visualize property mask before making any changes in it
        /// Generated from constructor `MR::ChangeVisualizePropertyAction::ChangeVisualizePropertyAction`.
        public unsafe ChangeVisualizePropertyAction(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.Const_AnyVisualizeMaskEnum visualizeMaskType) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_Construct_3(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.AnyVisualizeMaskEnum._Underlying *visualizeMaskType);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeVisualizePropertyAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, visualizeMaskType._UnderlyingPtr));
            }
        }

        /// use this constructor to remember object's visualize property mask and immediately set new value
        /// Generated from constructor `MR::ChangeVisualizePropertyAction::ChangeVisualizePropertyAction`.
        public unsafe ChangeVisualizePropertyAction(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.Const_AnyVisualizeMaskEnum visualizeMaskType, MR.Const_ViewportMask newMask) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_Construct_4(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.AnyVisualizeMaskEnum._Underlying *visualizeMaskType, MR.ViewportMask._Underlying *newMask);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeVisualizePropertyAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, visualizeMaskType._UnderlyingPtr, newMask._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeVisualizePropertyAction::operator=`.
        public unsafe MR.ChangeVisualizePropertyAction Assign(MR._ByValue_ChangeVisualizePropertyAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeVisualizePropertyAction._Underlying *__MR_ChangeVisualizePropertyAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeVisualizePropertyAction._Underlying *_other);
            return new(__MR_ChangeVisualizePropertyAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeVisualizePropertyAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeVisualizePropertyAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeVisualizePropertyAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeVisualizePropertyAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeVisualizePropertyAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeVisualizePropertyAction`/`Const_ChangeVisualizePropertyAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeVisualizePropertyAction
    {
        internal readonly Const_ChangeVisualizePropertyAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeVisualizePropertyAction(Const_ChangeVisualizePropertyAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeVisualizePropertyAction(Const_ChangeVisualizePropertyAction arg) {return new(arg);}
        public _ByValue_ChangeVisualizePropertyAction(MR.Misc._Moved<ChangeVisualizePropertyAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeVisualizePropertyAction(MR.Misc._Moved<ChangeVisualizePropertyAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeVisualizePropertyAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeVisualizePropertyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeVisualizePropertyAction`/`Const_ChangeVisualizePropertyAction` directly.
    public class _InOptMut_ChangeVisualizePropertyAction
    {
        public ChangeVisualizePropertyAction? Opt;

        public _InOptMut_ChangeVisualizePropertyAction() {}
        public _InOptMut_ChangeVisualizePropertyAction(ChangeVisualizePropertyAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeVisualizePropertyAction(ChangeVisualizePropertyAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeVisualizePropertyAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeVisualizePropertyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeVisualizePropertyAction`/`Const_ChangeVisualizePropertyAction` to pass it to the function.
    public class _InOptConst_ChangeVisualizePropertyAction
    {
        public Const_ChangeVisualizePropertyAction? Opt;

        public _InOptConst_ChangeVisualizePropertyAction() {}
        public _InOptConst_ChangeVisualizePropertyAction(Const_ChangeVisualizePropertyAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeVisualizePropertyAction(Const_ChangeVisualizePropertyAction value) {return new(value);}
    }

    /// History action for object selected state
    /// Generated from class `MR::ChangeObjectSelectedAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeObjectSelectedAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeObjectSelectedAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeObjectSelectedAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeObjectSelectedAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeObjectSelectedAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeObjectSelectedAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeObjectSelectedAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeObjectSelectedAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectSelectedAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeObjectSelectedAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectSelectedAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectSelectedAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectSelectedAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectSelectedAction::ChangeObjectSelectedAction`.
        public unsafe Const_ChangeObjectSelectedAction(MR._ByValue_ChangeObjectSelectedAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectSelectedAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectSelectedAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's isSelected property before making any changes in it
        /// Generated from constructor `MR::ChangeObjectSelectedAction::ChangeObjectSelectedAction`.
        public unsafe Const_ChangeObjectSelectedAction(ReadOnlySpan<char> name, MR.Const_Object obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_Construct_2(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectSelectedAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's isSelected property and immediately set new value
        /// Generated from constructor `MR::ChangeObjectSelectedAction::ChangeObjectSelectedAction`.
        public unsafe Const_ChangeObjectSelectedAction(ReadOnlySpan<char> name, MR.Const_Object obj, bool newValue) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_Construct_3(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj, byte newValue);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectSelectedAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newValue ? (byte)1 : (byte)0));
            }
        }

        /// Generated from method `MR::ChangeObjectSelectedAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeObjectSelectedAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeObjectSelectedAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeObjectSelectedAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_Object _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeObjectSelectedAction_setObjectDirty(MR.Const_Object._UnderlyingShared *_1);
            __MR_ChangeObjectSelectedAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeObjectSelectedAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeObjectSelectedAction_heapBytes(_Underlying *_this);
            return __MR_ChangeObjectSelectedAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for object selected state
    /// Generated from class `MR::ChangeObjectSelectedAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeObjectSelectedAction : Const_ChangeObjectSelectedAction
    {
        internal unsafe ChangeObjectSelectedAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeObjectSelectedAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeObjectSelectedAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeObjectSelectedAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectSelectedAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeObjectSelectedAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectSelectedAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectSelectedAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectSelectedAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectSelectedAction::ChangeObjectSelectedAction`.
        public unsafe ChangeObjectSelectedAction(MR._ByValue_ChangeObjectSelectedAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectSelectedAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectSelectedAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's isSelected property before making any changes in it
        /// Generated from constructor `MR::ChangeObjectSelectedAction::ChangeObjectSelectedAction`.
        public unsafe ChangeObjectSelectedAction(ReadOnlySpan<char> name, MR.Const_Object obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_Construct_2(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectSelectedAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's isSelected property and immediately set new value
        /// Generated from constructor `MR::ChangeObjectSelectedAction::ChangeObjectSelectedAction`.
        public unsafe ChangeObjectSelectedAction(ReadOnlySpan<char> name, MR.Const_Object obj, bool newValue) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_Construct_3(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj, byte newValue);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectSelectedAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newValue ? (byte)1 : (byte)0));
            }
        }

        /// Generated from method `MR::ChangeObjectSelectedAction::operator=`.
        public unsafe MR.ChangeObjectSelectedAction Assign(MR._ByValue_ChangeObjectSelectedAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectSelectedAction._Underlying *__MR_ChangeObjectSelectedAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectSelectedAction._Underlying *_other);
            return new(__MR_ChangeObjectSelectedAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeObjectSelectedAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectSelectedAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeObjectSelectedAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeObjectSelectedAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeObjectSelectedAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeObjectSelectedAction`/`Const_ChangeObjectSelectedAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeObjectSelectedAction
    {
        internal readonly Const_ChangeObjectSelectedAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeObjectSelectedAction(Const_ChangeObjectSelectedAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeObjectSelectedAction(Const_ChangeObjectSelectedAction arg) {return new(arg);}
        public _ByValue_ChangeObjectSelectedAction(MR.Misc._Moved<ChangeObjectSelectedAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeObjectSelectedAction(MR.Misc._Moved<ChangeObjectSelectedAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeObjectSelectedAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeObjectSelectedAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectSelectedAction`/`Const_ChangeObjectSelectedAction` directly.
    public class _InOptMut_ChangeObjectSelectedAction
    {
        public ChangeObjectSelectedAction? Opt;

        public _InOptMut_ChangeObjectSelectedAction() {}
        public _InOptMut_ChangeObjectSelectedAction(ChangeObjectSelectedAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeObjectSelectedAction(ChangeObjectSelectedAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeObjectSelectedAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeObjectSelectedAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectSelectedAction`/`Const_ChangeObjectSelectedAction` to pass it to the function.
    public class _InOptConst_ChangeObjectSelectedAction
    {
        public Const_ChangeObjectSelectedAction? Opt;

        public _InOptConst_ChangeObjectSelectedAction() {}
        public _InOptConst_ChangeObjectSelectedAction(Const_ChangeObjectSelectedAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeObjectSelectedAction(Const_ChangeObjectSelectedAction value) {return new(value);}
    }

    /// History action for object visibility
    /// Generated from class `MR::ChangeObjectVisibilityAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeObjectVisibilityAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeObjectVisibilityAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeObjectVisibilityAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeObjectVisibilityAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeObjectVisibilityAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeObjectVisibilityAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeObjectVisibilityAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeObjectVisibilityAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectVisibilityAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeObjectVisibilityAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectVisibilityAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectVisibilityAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectVisibilityAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectVisibilityAction::ChangeObjectVisibilityAction`.
        public unsafe Const_ChangeObjectVisibilityAction(MR._ByValue_ChangeObjectVisibilityAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectVisibilityAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectVisibilityAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's visibility mask before making any changes in it
        /// Generated from constructor `MR::ChangeObjectVisibilityAction::ChangeObjectVisibilityAction`.
        public unsafe Const_ChangeObjectVisibilityAction(ReadOnlySpan<char> name, MR.Const_Object obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_Construct_2(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectVisibilityAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's visibility mask and immediately set new mask
        /// Generated from constructor `MR::ChangeObjectVisibilityAction::ChangeObjectVisibilityAction`.
        public unsafe Const_ChangeObjectVisibilityAction(ReadOnlySpan<char> name, MR.Const_Object obj, MR.Const_ViewportMask newVisibilityMask) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_Construct_3(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj, MR.ViewportMask._Underlying *newVisibilityMask);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectVisibilityAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newVisibilityMask._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeObjectVisibilityAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeObjectVisibilityAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeObjectVisibilityAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeObjectVisibilityAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_Object _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeObjectVisibilityAction_setObjectDirty(MR.Const_Object._UnderlyingShared *_1);
            __MR_ChangeObjectVisibilityAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeObjectVisibilityAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeObjectVisibilityAction_heapBytes(_Underlying *_this);
            return __MR_ChangeObjectVisibilityAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for object visibility
    /// Generated from class `MR::ChangeObjectVisibilityAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeObjectVisibilityAction : Const_ChangeObjectVisibilityAction
    {
        internal unsafe ChangeObjectVisibilityAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeObjectVisibilityAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeObjectVisibilityAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeObjectVisibilityAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectVisibilityAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeObjectVisibilityAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectVisibilityAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectVisibilityAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectVisibilityAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectVisibilityAction::ChangeObjectVisibilityAction`.
        public unsafe ChangeObjectVisibilityAction(MR._ByValue_ChangeObjectVisibilityAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectVisibilityAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectVisibilityAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's visibility mask before making any changes in it
        /// Generated from constructor `MR::ChangeObjectVisibilityAction::ChangeObjectVisibilityAction`.
        public unsafe ChangeObjectVisibilityAction(ReadOnlySpan<char> name, MR.Const_Object obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_Construct_2(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectVisibilityAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's visibility mask and immediately set new mask
        /// Generated from constructor `MR::ChangeObjectVisibilityAction::ChangeObjectVisibilityAction`.
        public unsafe ChangeObjectVisibilityAction(ReadOnlySpan<char> name, MR.Const_Object obj, MR.Const_ViewportMask newVisibilityMask) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_Construct_3(byte *name, byte *name_end, MR.Const_Object._UnderlyingShared *obj, MR.ViewportMask._Underlying *newVisibilityMask);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectVisibilityAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newVisibilityMask._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeObjectVisibilityAction::operator=`.
        public unsafe MR.ChangeObjectVisibilityAction Assign(MR._ByValue_ChangeObjectVisibilityAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectVisibilityAction._Underlying *__MR_ChangeObjectVisibilityAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectVisibilityAction._Underlying *_other);
            return new(__MR_ChangeObjectVisibilityAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeObjectVisibilityAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectVisibilityAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeObjectVisibilityAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeObjectVisibilityAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeObjectVisibilityAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeObjectVisibilityAction`/`Const_ChangeObjectVisibilityAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeObjectVisibilityAction
    {
        internal readonly Const_ChangeObjectVisibilityAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeObjectVisibilityAction(Const_ChangeObjectVisibilityAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeObjectVisibilityAction(Const_ChangeObjectVisibilityAction arg) {return new(arg);}
        public _ByValue_ChangeObjectVisibilityAction(MR.Misc._Moved<ChangeObjectVisibilityAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeObjectVisibilityAction(MR.Misc._Moved<ChangeObjectVisibilityAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeObjectVisibilityAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeObjectVisibilityAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectVisibilityAction`/`Const_ChangeObjectVisibilityAction` directly.
    public class _InOptMut_ChangeObjectVisibilityAction
    {
        public ChangeObjectVisibilityAction? Opt;

        public _InOptMut_ChangeObjectVisibilityAction() {}
        public _InOptMut_ChangeObjectVisibilityAction(ChangeObjectVisibilityAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeObjectVisibilityAction(ChangeObjectVisibilityAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeObjectVisibilityAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeObjectVisibilityAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectVisibilityAction`/`Const_ChangeObjectVisibilityAction` to pass it to the function.
    public class _InOptConst_ChangeObjectVisibilityAction
    {
        public Const_ChangeObjectVisibilityAction? Opt;

        public _InOptConst_ChangeObjectVisibilityAction() {}
        public _InOptConst_ChangeObjectVisibilityAction(Const_ChangeObjectVisibilityAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeObjectVisibilityAction(Const_ChangeObjectVisibilityAction value) {return new(value);}
    }
}
