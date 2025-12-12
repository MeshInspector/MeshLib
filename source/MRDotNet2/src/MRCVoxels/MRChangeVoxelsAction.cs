public static partial class MR
{
    /// Undo action for ObjectVoxels iso-value change
    /// Generated from class `MR::ChangeIsoAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeIsoAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeIsoAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeIsoAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeIsoAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeIsoAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeIsoAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeIsoAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeIsoAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeIsoAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeIsoAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeIsoAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeIsoAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeIsoAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeIsoAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeIsoAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeIsoAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeIsoAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeIsoAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeIsoAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeIsoAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeIsoAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeIsoAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeIsoAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeIsoAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeIsoAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeIsoAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeIsoAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeIsoAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeIsoAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeIsoAction::ChangeIsoAction`.
        public unsafe Const_ChangeIsoAction(MR._ByValue_ChangeIsoAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeIsoAction._Underlying *__MR_ChangeIsoAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeIsoAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeIsoAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's iso before making any changes in it
        /// Generated from constructor `MR::ChangeIsoAction::ChangeIsoAction`.
        public unsafe Const_ChangeIsoAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeIsoAction._Underlying *__MR_ChangeIsoAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeIsoAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeIsoAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeIsoAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeIsoAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeIsoAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectVoxels _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeIsoAction_setObjectDirty(MR.Const_ObjectVoxels._UnderlyingShared *_1);
            __MR_ChangeIsoAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeIsoAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeIsoAction_heapBytes(_Underlying *_this);
            return __MR_ChangeIsoAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectVoxels iso-value change
    /// Generated from class `MR::ChangeIsoAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeIsoAction : Const_ChangeIsoAction
    {
        internal unsafe ChangeIsoAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeIsoAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeIsoAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeIsoAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeIsoAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeIsoAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeIsoAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeIsoAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeIsoAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeIsoAction::ChangeIsoAction`.
        public unsafe ChangeIsoAction(MR._ByValue_ChangeIsoAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeIsoAction._Underlying *__MR_ChangeIsoAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeIsoAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeIsoAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's iso before making any changes in it
        /// Generated from constructor `MR::ChangeIsoAction::ChangeIsoAction`.
        public unsafe ChangeIsoAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeIsoAction._Underlying *__MR_ChangeIsoAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeIsoAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeIsoAction::operator=`.
        public unsafe MR.ChangeIsoAction Assign(MR._ByValue_ChangeIsoAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeIsoAction._Underlying *__MR_ChangeIsoAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeIsoAction._Underlying *_other);
            return new(__MR_ChangeIsoAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeIsoAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeIsoAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeIsoAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeIsoAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeIsoAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeIsoAction`/`Const_ChangeIsoAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeIsoAction
    {
        internal readonly Const_ChangeIsoAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeIsoAction(Const_ChangeIsoAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeIsoAction(Const_ChangeIsoAction arg) {return new(arg);}
        public _ByValue_ChangeIsoAction(MR.Misc._Moved<ChangeIsoAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeIsoAction(MR.Misc._Moved<ChangeIsoAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeIsoAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeIsoAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeIsoAction`/`Const_ChangeIsoAction` directly.
    public class _InOptMut_ChangeIsoAction
    {
        public ChangeIsoAction? Opt;

        public _InOptMut_ChangeIsoAction() {}
        public _InOptMut_ChangeIsoAction(ChangeIsoAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeIsoAction(ChangeIsoAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeIsoAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeIsoAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeIsoAction`/`Const_ChangeIsoAction` to pass it to the function.
    public class _InOptConst_ChangeIsoAction
    {
        public Const_ChangeIsoAction? Opt;

        public _InOptConst_ChangeIsoAction() {}
        public _InOptConst_ChangeIsoAction(Const_ChangeIsoAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeIsoAction(Const_ChangeIsoAction value) {return new(value);}
    }

    /// Undo action for ObjectVoxels dual/standard marching cubes change
    /// Generated from class `MR::ChangeDualMarchingCubesAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeDualMarchingCubesAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeDualMarchingCubesAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeDualMarchingCubesAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeDualMarchingCubesAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeDualMarchingCubesAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeDualMarchingCubesAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeDualMarchingCubesAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeDualMarchingCubesAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeDualMarchingCubesAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeDualMarchingCubesAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeDualMarchingCubesAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeDualMarchingCubesAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeDualMarchingCubesAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeDualMarchingCubesAction::ChangeDualMarchingCubesAction`.
        public unsafe Const_ChangeDualMarchingCubesAction(MR._ByValue_ChangeDualMarchingCubesAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeDualMarchingCubesAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeDualMarchingCubesAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's dual-value before making any changes in it
        /// Generated from constructor `MR::ChangeDualMarchingCubesAction::ChangeDualMarchingCubesAction`.
        public unsafe Const_ChangeDualMarchingCubesAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeDualMarchingCubesAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember given dual-value (and not the current value in the object)
        /// Generated from constructor `MR::ChangeDualMarchingCubesAction::ChangeDualMarchingCubesAction`.
        public unsafe Const_ChangeDualMarchingCubesAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj, bool storeDual) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj, byte storeDual);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeDualMarchingCubesAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, storeDual ? (byte)1 : (byte)0));
            }
        }

        /// Generated from method `MR::ChangeDualMarchingCubesAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeDualMarchingCubesAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeDualMarchingCubesAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeDualMarchingCubesAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectVoxels _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeDualMarchingCubesAction_setObjectDirty(MR.Const_ObjectVoxels._UnderlyingShared *_1);
            __MR_ChangeDualMarchingCubesAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeDualMarchingCubesAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeDualMarchingCubesAction_heapBytes(_Underlying *_this);
            return __MR_ChangeDualMarchingCubesAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectVoxels dual/standard marching cubes change
    /// Generated from class `MR::ChangeDualMarchingCubesAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeDualMarchingCubesAction : Const_ChangeDualMarchingCubesAction
    {
        internal unsafe ChangeDualMarchingCubesAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeDualMarchingCubesAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeDualMarchingCubesAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeDualMarchingCubesAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeDualMarchingCubesAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeDualMarchingCubesAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeDualMarchingCubesAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeDualMarchingCubesAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeDualMarchingCubesAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeDualMarchingCubesAction::ChangeDualMarchingCubesAction`.
        public unsafe ChangeDualMarchingCubesAction(MR._ByValue_ChangeDualMarchingCubesAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeDualMarchingCubesAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeDualMarchingCubesAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's dual-value before making any changes in it
        /// Generated from constructor `MR::ChangeDualMarchingCubesAction::ChangeDualMarchingCubesAction`.
        public unsafe ChangeDualMarchingCubesAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeDualMarchingCubesAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember given dual-value (and not the current value in the object)
        /// Generated from constructor `MR::ChangeDualMarchingCubesAction::ChangeDualMarchingCubesAction`.
        public unsafe ChangeDualMarchingCubesAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj, bool storeDual) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj, byte storeDual);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeDualMarchingCubesAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, storeDual ? (byte)1 : (byte)0));
            }
        }

        /// Generated from method `MR::ChangeDualMarchingCubesAction::operator=`.
        public unsafe MR.ChangeDualMarchingCubesAction Assign(MR._ByValue_ChangeDualMarchingCubesAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeDualMarchingCubesAction._Underlying *__MR_ChangeDualMarchingCubesAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeDualMarchingCubesAction._Underlying *_other);
            return new(__MR_ChangeDualMarchingCubesAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeDualMarchingCubesAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeDualMarchingCubesAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeDualMarchingCubesAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeDualMarchingCubesAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeDualMarchingCubesAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeDualMarchingCubesAction`/`Const_ChangeDualMarchingCubesAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeDualMarchingCubesAction
    {
        internal readonly Const_ChangeDualMarchingCubesAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeDualMarchingCubesAction(Const_ChangeDualMarchingCubesAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeDualMarchingCubesAction(Const_ChangeDualMarchingCubesAction arg) {return new(arg);}
        public _ByValue_ChangeDualMarchingCubesAction(MR.Misc._Moved<ChangeDualMarchingCubesAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeDualMarchingCubesAction(MR.Misc._Moved<ChangeDualMarchingCubesAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeDualMarchingCubesAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeDualMarchingCubesAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeDualMarchingCubesAction`/`Const_ChangeDualMarchingCubesAction` directly.
    public class _InOptMut_ChangeDualMarchingCubesAction
    {
        public ChangeDualMarchingCubesAction? Opt;

        public _InOptMut_ChangeDualMarchingCubesAction() {}
        public _InOptMut_ChangeDualMarchingCubesAction(ChangeDualMarchingCubesAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeDualMarchingCubesAction(ChangeDualMarchingCubesAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeDualMarchingCubesAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeDualMarchingCubesAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeDualMarchingCubesAction`/`Const_ChangeDualMarchingCubesAction` to pass it to the function.
    public class _InOptConst_ChangeDualMarchingCubesAction
    {
        public Const_ChangeDualMarchingCubesAction? Opt;

        public _InOptConst_ChangeDualMarchingCubesAction() {}
        public _InOptConst_ChangeDualMarchingCubesAction(Const_ChangeDualMarchingCubesAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeDualMarchingCubesAction(Const_ChangeDualMarchingCubesAction value) {return new(value);}
    }

    // Undo action for ObjectVoxels active bounds change
    /// Generated from class `MR::ChangeActiveBoxAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeActiveBoxAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeActiveBoxAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeActiveBoxAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeActiveBoxAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeActiveBoxAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeActiveBoxAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeActiveBoxAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeActiveBoxAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeActiveBoxAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeActiveBoxAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeActiveBoxAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeActiveBoxAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeActiveBoxAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeActiveBoxAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeActiveBoxAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeActiveBoxAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeActiveBoxAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeActiveBoxAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeActiveBoxAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeActiveBoxAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeActiveBoxAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeActiveBoxAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeActiveBoxAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeActiveBoxAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeActiveBoxAction::ChangeActiveBoxAction`.
        public unsafe Const_ChangeActiveBoxAction(MR._ByValue_ChangeActiveBoxAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeActiveBoxAction._Underlying *__MR_ChangeActiveBoxAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeActiveBoxAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeActiveBoxAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's active box before making any changes in it
        /// Generated from constructor `MR::ChangeActiveBoxAction::ChangeActiveBoxAction`.
        public unsafe Const_ChangeActiveBoxAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeActiveBoxAction._Underlying *__MR_ChangeActiveBoxAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeActiveBoxAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeActiveBoxAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeActiveBoxAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeActiveBoxAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeActiveBoxAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectVoxels _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeActiveBoxAction_setObjectDirty(MR.Const_ObjectVoxels._UnderlyingShared *_1);
            __MR_ChangeActiveBoxAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeActiveBoxAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeActiveBoxAction_heapBytes(_Underlying *_this);
            return __MR_ChangeActiveBoxAction_heapBytes(_UnderlyingPtr);
        }
    }

    // Undo action for ObjectVoxels active bounds change
    /// Generated from class `MR::ChangeActiveBoxAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeActiveBoxAction : Const_ChangeActiveBoxAction
    {
        internal unsafe ChangeActiveBoxAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeActiveBoxAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeActiveBoxAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeActiveBoxAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeActiveBoxAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeActiveBoxAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeActiveBoxAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeActiveBoxAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeActiveBoxAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeActiveBoxAction::ChangeActiveBoxAction`.
        public unsafe ChangeActiveBoxAction(MR._ByValue_ChangeActiveBoxAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeActiveBoxAction._Underlying *__MR_ChangeActiveBoxAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeActiveBoxAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeActiveBoxAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's active box before making any changes in it
        /// Generated from constructor `MR::ChangeActiveBoxAction::ChangeActiveBoxAction`.
        public unsafe ChangeActiveBoxAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeActiveBoxAction._Underlying *__MR_ChangeActiveBoxAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeActiveBoxAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeActiveBoxAction::operator=`.
        public unsafe MR.ChangeActiveBoxAction Assign(MR._ByValue_ChangeActiveBoxAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeActiveBoxAction._Underlying *__MR_ChangeActiveBoxAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeActiveBoxAction._Underlying *_other);
            return new(__MR_ChangeActiveBoxAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeActiveBoxAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeActiveBoxAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeActiveBoxAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeActiveBoxAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeActiveBoxAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeActiveBoxAction`/`Const_ChangeActiveBoxAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeActiveBoxAction
    {
        internal readonly Const_ChangeActiveBoxAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeActiveBoxAction(Const_ChangeActiveBoxAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeActiveBoxAction(Const_ChangeActiveBoxAction arg) {return new(arg);}
        public _ByValue_ChangeActiveBoxAction(MR.Misc._Moved<ChangeActiveBoxAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeActiveBoxAction(MR.Misc._Moved<ChangeActiveBoxAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeActiveBoxAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeActiveBoxAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeActiveBoxAction`/`Const_ChangeActiveBoxAction` directly.
    public class _InOptMut_ChangeActiveBoxAction
    {
        public ChangeActiveBoxAction? Opt;

        public _InOptMut_ChangeActiveBoxAction() {}
        public _InOptMut_ChangeActiveBoxAction(ChangeActiveBoxAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeActiveBoxAction(ChangeActiveBoxAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeActiveBoxAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeActiveBoxAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeActiveBoxAction`/`Const_ChangeActiveBoxAction` to pass it to the function.
    public class _InOptConst_ChangeActiveBoxAction
    {
        public Const_ChangeActiveBoxAction? Opt;

        public _InOptConst_ChangeActiveBoxAction() {}
        public _InOptConst_ChangeActiveBoxAction(Const_ChangeActiveBoxAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeActiveBoxAction(Const_ChangeActiveBoxAction value) {return new(value);}
    }

    // Undo action for ObjectVoxels surface change (need for faster undo redo)
    /// Generated from class `MR::ChangeSurfaceAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeSurfaceAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeSurfaceAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeSurfaceAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeSurfaceAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeSurfaceAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeSurfaceAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSurfaceAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeSurfaceAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeSurfaceAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeSurfaceAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeSurfaceAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeSurfaceAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeSurfaceAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeSurfaceAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeSurfaceAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeSurfaceAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeSurfaceAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeSurfaceAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeSurfaceAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeSurfaceAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeSurfaceAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeSurfaceAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeSurfaceAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeSurfaceAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeSurfaceAction::ChangeSurfaceAction`.
        public unsafe Const_ChangeSurfaceAction(MR._ByValue_ChangeSurfaceAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeSurfaceAction._Underlying *__MR_ChangeSurfaceAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeSurfaceAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeSurfaceAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's surface before making any changes in it
        /// Generated from constructor `MR::ChangeSurfaceAction::ChangeSurfaceAction`.
        public unsafe Const_ChangeSurfaceAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeSurfaceAction._Underlying *__MR_ChangeSurfaceAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeSurfaceAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeSurfaceAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeSurfaceAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeSurfaceAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeSurfaceAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectVoxels obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeSurfaceAction_setObjectDirty(MR.Const_ObjectVoxels._UnderlyingShared *obj);
            __MR_ChangeSurfaceAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeSurfaceAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeSurfaceAction_heapBytes(_Underlying *_this);
            return __MR_ChangeSurfaceAction_heapBytes(_UnderlyingPtr);
        }
    }

    // Undo action for ObjectVoxels surface change (need for faster undo redo)
    /// Generated from class `MR::ChangeSurfaceAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeSurfaceAction : Const_ChangeSurfaceAction
    {
        internal unsafe ChangeSurfaceAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeSurfaceAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeSurfaceAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeSurfaceAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeSurfaceAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeSurfaceAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeSurfaceAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeSurfaceAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeSurfaceAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeSurfaceAction::ChangeSurfaceAction`.
        public unsafe ChangeSurfaceAction(MR._ByValue_ChangeSurfaceAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeSurfaceAction._Underlying *__MR_ChangeSurfaceAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeSurfaceAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeSurfaceAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's surface before making any changes in it
        /// Generated from constructor `MR::ChangeSurfaceAction::ChangeSurfaceAction`.
        public unsafe ChangeSurfaceAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeSurfaceAction._Underlying *__MR_ChangeSurfaceAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeSurfaceAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeSurfaceAction::operator=`.
        public unsafe MR.ChangeSurfaceAction Assign(MR._ByValue_ChangeSurfaceAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeSurfaceAction._Underlying *__MR_ChangeSurfaceAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeSurfaceAction._Underlying *_other);
            return new(__MR_ChangeSurfaceAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeSurfaceAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeSurfaceAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeSurfaceAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeSurfaceAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeSurfaceAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeSurfaceAction`/`Const_ChangeSurfaceAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeSurfaceAction
    {
        internal readonly Const_ChangeSurfaceAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeSurfaceAction(Const_ChangeSurfaceAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeSurfaceAction(Const_ChangeSurfaceAction arg) {return new(arg);}
        public _ByValue_ChangeSurfaceAction(MR.Misc._Moved<ChangeSurfaceAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeSurfaceAction(MR.Misc._Moved<ChangeSurfaceAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeSurfaceAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeSurfaceAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeSurfaceAction`/`Const_ChangeSurfaceAction` directly.
    public class _InOptMut_ChangeSurfaceAction
    {
        public ChangeSurfaceAction? Opt;

        public _InOptMut_ChangeSurfaceAction() {}
        public _InOptMut_ChangeSurfaceAction(ChangeSurfaceAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeSurfaceAction(ChangeSurfaceAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeSurfaceAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeSurfaceAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeSurfaceAction`/`Const_ChangeSurfaceAction` to pass it to the function.
    public class _InOptConst_ChangeSurfaceAction
    {
        public Const_ChangeSurfaceAction? Opt;

        public _InOptConst_ChangeSurfaceAction() {}
        public _InOptConst_ChangeSurfaceAction(Const_ChangeSurfaceAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeSurfaceAction(Const_ChangeSurfaceAction value) {return new(value);}
    }

    // Undo action for ObjectVoxels all data change (need for faster undo redo)
    /// Generated from class `MR::ChangeGridAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeGridAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeGridAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeGridAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeGridAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeGridAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeGridAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeGridAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeGridAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeGridAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeGridAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeGridAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeGridAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeGridAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeGridAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeGridAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeGridAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeGridAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeGridAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeGridAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeGridAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeGridAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeGridAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeGridAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeGridAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeGridAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeGridAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeGridAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeGridAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeGridAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeGridAction::ChangeGridAction`.
        public unsafe Const_ChangeGridAction(MR._ByValue_ChangeGridAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeGridAction._Underlying *__MR_ChangeGridAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeGridAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeGridAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's data before making any changes in it
        /// Generated from constructor `MR::ChangeGridAction::ChangeGridAction`.
        public unsafe Const_ChangeGridAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeGridAction._Underlying *__MR_ChangeGridAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeGridAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeGridAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeGridAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeGridAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeGridAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectVoxels obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeGridAction_setObjectDirty(MR.Const_ObjectVoxels._UnderlyingShared *obj);
            __MR_ChangeGridAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeGridAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeGridAction_heapBytes(_Underlying *_this);
            return __MR_ChangeGridAction_heapBytes(_UnderlyingPtr);
        }
    }

    // Undo action for ObjectVoxels all data change (need for faster undo redo)
    /// Generated from class `MR::ChangeGridAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeGridAction : Const_ChangeGridAction
    {
        internal unsafe ChangeGridAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeGridAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeGridAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeGridAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeGridAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeGridAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeGridAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeGridAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeGridAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeGridAction::ChangeGridAction`.
        public unsafe ChangeGridAction(MR._ByValue_ChangeGridAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeGridAction._Underlying *__MR_ChangeGridAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeGridAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeGridAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's data before making any changes in it
        /// Generated from constructor `MR::ChangeGridAction::ChangeGridAction`.
        public unsafe ChangeGridAction(ReadOnlySpan<char> name, MR.Const_ObjectVoxels obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeGridAction._Underlying *__MR_ChangeGridAction_Construct(byte *name, byte *name_end, MR.Const_ObjectVoxels._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeGridAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangeGridAction::operator=`.
        public unsafe MR.ChangeGridAction Assign(MR._ByValue_ChangeGridAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeGridAction._Underlying *__MR_ChangeGridAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeGridAction._Underlying *_other);
            return new(__MR_ChangeGridAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeGridAction::action`.
        public unsafe void Action(MR.HistoryAction.Type obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeGridAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeGridAction_action(_Underlying *_this, MR.HistoryAction.Type obj);
            __MR_ChangeGridAction_action(_UnderlyingPtr, obj);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeGridAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeGridAction`/`Const_ChangeGridAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeGridAction
    {
        internal readonly Const_ChangeGridAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeGridAction(Const_ChangeGridAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeGridAction(Const_ChangeGridAction arg) {return new(arg);}
        public _ByValue_ChangeGridAction(MR.Misc._Moved<ChangeGridAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeGridAction(MR.Misc._Moved<ChangeGridAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeGridAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeGridAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeGridAction`/`Const_ChangeGridAction` directly.
    public class _InOptMut_ChangeGridAction
    {
        public ChangeGridAction? Opt;

        public _InOptMut_ChangeGridAction() {}
        public _InOptMut_ChangeGridAction(ChangeGridAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeGridAction(ChangeGridAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeGridAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeGridAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeGridAction`/`Const_ChangeGridAction` to pass it to the function.
    public class _InOptConst_ChangeGridAction
    {
        public Const_ChangeGridAction? Opt;

        public _InOptConst_ChangeGridAction() {}
        public _InOptConst_ChangeGridAction(Const_ChangeGridAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeGridAction(Const_ChangeGridAction value) {return new(value);}
    }
}
