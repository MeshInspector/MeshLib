public static partial class MR
{
    /// Undo action for ObjectMesh mesh change
    /// Generated from class `MR::ChangeMeshAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshAction::ChangeMeshAction`.
        public unsafe Const_ChangeMeshAction(MR._ByValue_ChangeMeshAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's mesh before making any changes in it
        /// Generated from constructor `MR::ChangeMeshAction::ChangeMeshAction`.
        public unsafe Const_ChangeMeshAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's mesh and immediately set new mesh
        /// Generated from constructor `MR::ChangeMeshAction::ChangeMeshAction`.
        public unsafe Const_ChangeMeshAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR._ByValue_Mesh newMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.Misc._PassBy newMesh_pass_by, MR.Mesh._UnderlyingShared *newMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newMesh.PassByMode, newMesh.Value is not null ? newMesh.Value._UnderlyingSharedPtr : null));
            }
        }

        /// Generated from method `MR::ChangeMeshAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *obj);
            __MR_ChangeMeshAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMesh mesh change
    /// Generated from class `MR::ChangeMeshAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshAction : Const_ChangeMeshAction
    {
        internal unsafe ChangeMeshAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshAction::ChangeMeshAction`.
        public unsafe ChangeMeshAction(MR._ByValue_ChangeMeshAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's mesh before making any changes in it
        /// Generated from constructor `MR::ChangeMeshAction::ChangeMeshAction`.
        public unsafe ChangeMeshAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's mesh and immediately set new mesh
        /// Generated from constructor `MR::ChangeMeshAction::ChangeMeshAction`.
        public unsafe ChangeMeshAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR._ByValue_Mesh newMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.Misc._PassBy newMesh_pass_by, MR.Mesh._UnderlyingShared *newMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newMesh.PassByMode, newMesh.Value is not null ? newMesh.Value._UnderlyingSharedPtr : null));
            }
        }

        /// Generated from method `MR::ChangeMeshAction::operator=`.
        public unsafe MR.ChangeMeshAction Assign(MR._ByValue_ChangeMeshAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshAction._Underlying *__MR_ChangeMeshAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshAction._Underlying *_other);
            return new(__MR_ChangeMeshAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshAction`/`Const_ChangeMeshAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshAction
    {
        internal readonly Const_ChangeMeshAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshAction(Const_ChangeMeshAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshAction(Const_ChangeMeshAction arg) {return new(arg);}
        public _ByValue_ChangeMeshAction(MR.Misc._Moved<ChangeMeshAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshAction(MR.Misc._Moved<ChangeMeshAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshAction`/`Const_ChangeMeshAction` directly.
    public class _InOptMut_ChangeMeshAction
    {
        public ChangeMeshAction? Opt;

        public _InOptMut_ChangeMeshAction() {}
        public _InOptMut_ChangeMeshAction(ChangeMeshAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshAction(ChangeMeshAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshAction`/`Const_ChangeMeshAction` to pass it to the function.
    public class _InOptConst_ChangeMeshAction
    {
        public Const_ChangeMeshAction? Opt;

        public _InOptConst_ChangeMeshAction() {}
        public _InOptConst_ChangeMeshAction(Const_ChangeMeshAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshAction(Const_ChangeMeshAction value) {return new(value);}
    }

    /// Undo action for ObjectMeshHolder uvCoords change
    /// Generated from class `MR::ChangeMeshUVCoordsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshUVCoordsAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshUVCoordsAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshUVCoordsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshUVCoordsAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshUVCoordsAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshUVCoordsAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshUVCoordsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshUVCoordsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshUVCoordsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshUVCoordsAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshUVCoordsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshUVCoordsAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshUVCoordsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshUVCoordsAction::ChangeMeshUVCoordsAction`.
        public unsafe Const_ChangeMeshUVCoordsAction(MR._ByValue_ChangeMeshUVCoordsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshUVCoordsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshUVCoordsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's uv-coordinates before making any changes in them
        /// Generated from constructor `MR::ChangeMeshUVCoordsAction::ChangeMeshUVCoordsAction`.
        public unsafe Const_ChangeMeshUVCoordsAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshUVCoordsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's uv-coordinates and immediate set new value
        /// Generated from constructor `MR::ChangeMeshUVCoordsAction::ChangeMeshUVCoordsAction`.
        public unsafe Const_ChangeMeshUVCoordsAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.VertCoords2> newUvCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.VertCoords2._Underlying *newUvCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshUVCoordsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newUvCoords.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshUVCoordsAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshUVCoordsAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshUVCoordsAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshUVCoordsAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMeshHolder obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshUVCoordsAction_setObjectDirty(MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            __MR_ChangeMeshUVCoordsAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshUVCoordsAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshUVCoordsAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshUVCoordsAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMeshHolder uvCoords change
    /// Generated from class `MR::ChangeMeshUVCoordsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshUVCoordsAction : Const_ChangeMeshUVCoordsAction
    {
        internal unsafe ChangeMeshUVCoordsAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshUVCoordsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshUVCoordsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshUVCoordsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshUVCoordsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshUVCoordsAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshUVCoordsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshUVCoordsAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshUVCoordsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshUVCoordsAction::ChangeMeshUVCoordsAction`.
        public unsafe ChangeMeshUVCoordsAction(MR._ByValue_ChangeMeshUVCoordsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshUVCoordsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshUVCoordsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's uv-coordinates before making any changes in them
        /// Generated from constructor `MR::ChangeMeshUVCoordsAction::ChangeMeshUVCoordsAction`.
        public unsafe ChangeMeshUVCoordsAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshUVCoordsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's uv-coordinates and immediate set new value
        /// Generated from constructor `MR::ChangeMeshUVCoordsAction::ChangeMeshUVCoordsAction`.
        public unsafe ChangeMeshUVCoordsAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.VertCoords2> newUvCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.VertCoords2._Underlying *newUvCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshUVCoordsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newUvCoords.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshUVCoordsAction::operator=`.
        public unsafe MR.ChangeMeshUVCoordsAction Assign(MR._ByValue_ChangeMeshUVCoordsAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshUVCoordsAction._Underlying *__MR_ChangeMeshUVCoordsAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshUVCoordsAction._Underlying *_other);
            return new(__MR_ChangeMeshUVCoordsAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshUVCoordsAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshUVCoordsAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshUVCoordsAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshUVCoordsAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshUVCoordsAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshUVCoordsAction`/`Const_ChangeMeshUVCoordsAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshUVCoordsAction
    {
        internal readonly Const_ChangeMeshUVCoordsAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshUVCoordsAction(Const_ChangeMeshUVCoordsAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshUVCoordsAction(Const_ChangeMeshUVCoordsAction arg) {return new(arg);}
        public _ByValue_ChangeMeshUVCoordsAction(MR.Misc._Moved<ChangeMeshUVCoordsAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshUVCoordsAction(MR.Misc._Moved<ChangeMeshUVCoordsAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshUVCoordsAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshUVCoordsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshUVCoordsAction`/`Const_ChangeMeshUVCoordsAction` directly.
    public class _InOptMut_ChangeMeshUVCoordsAction
    {
        public ChangeMeshUVCoordsAction? Opt;

        public _InOptMut_ChangeMeshUVCoordsAction() {}
        public _InOptMut_ChangeMeshUVCoordsAction(ChangeMeshUVCoordsAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshUVCoordsAction(ChangeMeshUVCoordsAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshUVCoordsAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshUVCoordsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshUVCoordsAction`/`Const_ChangeMeshUVCoordsAction` to pass it to the function.
    public class _InOptConst_ChangeMeshUVCoordsAction
    {
        public Const_ChangeMeshUVCoordsAction? Opt;

        public _InOptConst_ChangeMeshUVCoordsAction() {}
        public _InOptConst_ChangeMeshUVCoordsAction(Const_ChangeMeshUVCoordsAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshUVCoordsAction(Const_ChangeMeshUVCoordsAction value) {return new(value);}
    }

    /// History action for texture change
    /// Generated from class `MR::ChangeTextureAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeTextureAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeTextureAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeTextureAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeTextureAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeTextureAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeTextureAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeTextureAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeTextureAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeTextureAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeTextureAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeTextureAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeTextureAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeTextureAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeTextureAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeTextureAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeTextureAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeTextureAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeTextureAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeTextureAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeTextureAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeTextureAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeTextureAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeTextureAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeTextureAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeTextureAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeTextureAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeTextureAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeTextureAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeTextureAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeTextureAction::ChangeTextureAction`.
        public unsafe Const_ChangeTextureAction(MR._ByValue_ChangeTextureAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeTextureAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeTextureAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's textures before making any changes in them
        /// Generated from constructor `MR::ChangeTextureAction::ChangeTextureAction`.
        public unsafe Const_ChangeTextureAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeTextureAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's textures and immediate set new value
        /// Generated from constructor `MR::ChangeTextureAction::ChangeTextureAction`.
        public unsafe Const_ChangeTextureAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.Vector_MRMeshTexture_MRTextureId> newTextures) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.Vector_MRMeshTexture_MRTextureId._Underlying *newTextures);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeTextureAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newTextures.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeTextureAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeTextureAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeTextureAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeTextureAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMeshHolder obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeTextureAction_setObjectDirty(MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            __MR_ChangeTextureAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeTextureAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeTextureAction_heapBytes(_Underlying *_this);
            return __MR_ChangeTextureAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for texture change
    /// Generated from class `MR::ChangeTextureAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeTextureAction : Const_ChangeTextureAction
    {
        internal unsafe ChangeTextureAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeTextureAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeTextureAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeTextureAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeTextureAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeTextureAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeTextureAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeTextureAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeTextureAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeTextureAction::ChangeTextureAction`.
        public unsafe ChangeTextureAction(MR._ByValue_ChangeTextureAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeTextureAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeTextureAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's textures before making any changes in them
        /// Generated from constructor `MR::ChangeTextureAction::ChangeTextureAction`.
        public unsafe ChangeTextureAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeTextureAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's textures and immediate set new value
        /// Generated from constructor `MR::ChangeTextureAction::ChangeTextureAction`.
        public unsafe ChangeTextureAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.Vector_MRMeshTexture_MRTextureId> newTextures) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.Vector_MRMeshTexture_MRTextureId._Underlying *newTextures);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeTextureAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newTextures.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeTextureAction::operator=`.
        public unsafe MR.ChangeTextureAction Assign(MR._ByValue_ChangeTextureAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeTextureAction._Underlying *__MR_ChangeTextureAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeTextureAction._Underlying *_other);
            return new(__MR_ChangeTextureAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeTextureAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeTextureAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeTextureAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeTextureAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeTextureAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeTextureAction`/`Const_ChangeTextureAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeTextureAction
    {
        internal readonly Const_ChangeTextureAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeTextureAction(Const_ChangeTextureAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeTextureAction(Const_ChangeTextureAction arg) {return new(arg);}
        public _ByValue_ChangeTextureAction(MR.Misc._Moved<ChangeTextureAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeTextureAction(MR.Misc._Moved<ChangeTextureAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeTextureAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeTextureAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeTextureAction`/`Const_ChangeTextureAction` directly.
    public class _InOptMut_ChangeTextureAction
    {
        public ChangeTextureAction? Opt;

        public _InOptMut_ChangeTextureAction() {}
        public _InOptMut_ChangeTextureAction(ChangeTextureAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeTextureAction(ChangeTextureAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeTextureAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeTextureAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeTextureAction`/`Const_ChangeTextureAction` to pass it to the function.
    public class _InOptConst_ChangeTextureAction
    {
        public Const_ChangeTextureAction? Opt;

        public _InOptConst_ChangeTextureAction() {}
        public _InOptConst_ChangeTextureAction(Const_ChangeTextureAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeTextureAction(Const_ChangeTextureAction value) {return new(value);}
    }

    /// Undo action for ObjectMesh points only (not topology) change
    /// Generated from class `MR::ChangeMeshPointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshPointsAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshPointsAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshPointsAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshPointsAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshPointsAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshPointsAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshPointsAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshPointsAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshPointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshPointsAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshPointsAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshPointsAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshPointsAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshPointsAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshPointsAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshPointsAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshPointsAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshPointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshPointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshPointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshPointsAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshPointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshPointsAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshPointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshPointsAction::ChangeMeshPointsAction`.
        public unsafe Const_ChangeMeshPointsAction(MR._ByValue_ChangeMeshPointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshPointsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshPointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's mesh points before making any changes in it
        /// Generated from constructor `MR::ChangeMeshPointsAction::ChangeMeshPointsAction`.
        public unsafe Const_ChangeMeshPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshPointsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's mesh points and immediate set new value
        /// Generated from constructor `MR::ChangeMeshPointsAction::ChangeMeshPointsAction`.
        public unsafe Const_ChangeMeshPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.VertCoords> newCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.VertCoords._Underlying *newCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshPointsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newCoords.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshPointsAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshPointsAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshPointsAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshPointsAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshPointsAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *obj);
            __MR_ChangeMeshPointsAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshPointsAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshPointsAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshPointsAction_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::ChangeMeshPointsAction::obj`.
        public unsafe MR.Const_ObjectMesh Obj()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_obj", ExactSpelling = true)]
            extern static MR.Const_ObjectMesh._UnderlyingShared *__MR_ChangeMeshPointsAction_obj(_Underlying *_this);
            return new(__MR_ChangeMeshPointsAction_obj(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshPointsAction::clonePoints`.
        public unsafe MR.Const_VertCoords ClonePoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_clonePoints", ExactSpelling = true)]
            extern static MR.Const_VertCoords._Underlying *__MR_ChangeMeshPointsAction_clonePoints(_Underlying *_this);
            return new(__MR_ChangeMeshPointsAction_clonePoints(_UnderlyingPtr), is_owning: false);
        }
    }

    /// Undo action for ObjectMesh points only (not topology) change
    /// Generated from class `MR::ChangeMeshPointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshPointsAction : Const_ChangeMeshPointsAction
    {
        internal unsafe ChangeMeshPointsAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshPointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshPointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshPointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshPointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshPointsAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshPointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshPointsAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshPointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshPointsAction::ChangeMeshPointsAction`.
        public unsafe ChangeMeshPointsAction(MR._ByValue_ChangeMeshPointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshPointsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshPointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's mesh points before making any changes in it
        /// Generated from constructor `MR::ChangeMeshPointsAction::ChangeMeshPointsAction`.
        public unsafe ChangeMeshPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshPointsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's mesh points and immediate set new value
        /// Generated from constructor `MR::ChangeMeshPointsAction::ChangeMeshPointsAction`.
        public unsafe ChangeMeshPointsAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.VertCoords> newCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.VertCoords._Underlying *newCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshPointsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newCoords.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshPointsAction::operator=`.
        public unsafe MR.ChangeMeshPointsAction Assign(MR._ByValue_ChangeMeshPointsAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshPointsAction._Underlying *__MR_ChangeMeshPointsAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshPointsAction._Underlying *_other);
            return new(__MR_ChangeMeshPointsAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshPointsAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshPointsAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshPointsAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshPointsAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshPointsAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshPointsAction`/`Const_ChangeMeshPointsAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshPointsAction
    {
        internal readonly Const_ChangeMeshPointsAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshPointsAction(Const_ChangeMeshPointsAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshPointsAction(Const_ChangeMeshPointsAction arg) {return new(arg);}
        public _ByValue_ChangeMeshPointsAction(MR.Misc._Moved<ChangeMeshPointsAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshPointsAction(MR.Misc._Moved<ChangeMeshPointsAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshPointsAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshPointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshPointsAction`/`Const_ChangeMeshPointsAction` directly.
    public class _InOptMut_ChangeMeshPointsAction
    {
        public ChangeMeshPointsAction? Opt;

        public _InOptMut_ChangeMeshPointsAction() {}
        public _InOptMut_ChangeMeshPointsAction(ChangeMeshPointsAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshPointsAction(ChangeMeshPointsAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshPointsAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshPointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshPointsAction`/`Const_ChangeMeshPointsAction` to pass it to the function.
    public class _InOptConst_ChangeMeshPointsAction
    {
        public Const_ChangeMeshPointsAction? Opt;

        public _InOptConst_ChangeMeshPointsAction() {}
        public _InOptConst_ChangeMeshPointsAction(Const_ChangeMeshPointsAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshPointsAction(Const_ChangeMeshPointsAction value) {return new(value);}
    }

    /// Undo action for ObjectMesh topology only (not points) change
    /// Generated from class `MR::ChangeMeshTopologyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshTopologyAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshTopologyAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshTopologyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshTopologyAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshTopologyAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshTopologyAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshTopologyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshTopologyAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTopologyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTopologyAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTopologyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshTopologyAction::ChangeMeshTopologyAction`.
        public unsafe Const_ChangeMeshTopologyAction(MR._ByValue_ChangeMeshTopologyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTopologyAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshTopologyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's mesh points before making any changes in it
        /// Generated from constructor `MR::ChangeMeshTopologyAction::ChangeMeshTopologyAction`.
        public unsafe Const_ChangeMeshTopologyAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTopologyAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's mesh topology and immediate set new value
        /// Generated from constructor `MR::ChangeMeshTopologyAction::ChangeMeshTopologyAction`.
        public unsafe Const_ChangeMeshTopologyAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.MeshTopology> newTopology) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.MeshTopology._Underlying *newTopology);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTopologyAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newTopology.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshTopologyAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshTopologyAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshTopologyAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshTopologyAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshTopologyAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *obj);
            __MR_ChangeMeshTopologyAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshTopologyAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshTopologyAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshTopologyAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMesh topology only (not points) change
    /// Generated from class `MR::ChangeMeshTopologyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshTopologyAction : Const_ChangeMeshTopologyAction
    {
        internal unsafe ChangeMeshTopologyAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshTopologyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshTopologyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshTopologyAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTopologyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTopologyAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTopologyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshTopologyAction::ChangeMeshTopologyAction`.
        public unsafe ChangeMeshTopologyAction(MR._ByValue_ChangeMeshTopologyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTopologyAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshTopologyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's mesh points before making any changes in it
        /// Generated from constructor `MR::ChangeMeshTopologyAction::ChangeMeshTopologyAction`.
        public unsafe ChangeMeshTopologyAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTopologyAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's mesh topology and immediate set new value
        /// Generated from constructor `MR::ChangeMeshTopologyAction::ChangeMeshTopologyAction`.
        public unsafe ChangeMeshTopologyAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.MeshTopology> newTopology) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.MeshTopology._Underlying *newTopology);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTopologyAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newTopology.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshTopologyAction::operator=`.
        public unsafe MR.ChangeMeshTopologyAction Assign(MR._ByValue_ChangeMeshTopologyAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshTopologyAction._Underlying *__MR_ChangeMeshTopologyAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTopologyAction._Underlying *_other);
            return new(__MR_ChangeMeshTopologyAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshTopologyAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTopologyAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshTopologyAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshTopologyAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshTopologyAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshTopologyAction`/`Const_ChangeMeshTopologyAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshTopologyAction
    {
        internal readonly Const_ChangeMeshTopologyAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshTopologyAction(Const_ChangeMeshTopologyAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshTopologyAction(Const_ChangeMeshTopologyAction arg) {return new(arg);}
        public _ByValue_ChangeMeshTopologyAction(MR.Misc._Moved<ChangeMeshTopologyAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshTopologyAction(MR.Misc._Moved<ChangeMeshTopologyAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshTopologyAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshTopologyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshTopologyAction`/`Const_ChangeMeshTopologyAction` directly.
    public class _InOptMut_ChangeMeshTopologyAction
    {
        public ChangeMeshTopologyAction? Opt;

        public _InOptMut_ChangeMeshTopologyAction() {}
        public _InOptMut_ChangeMeshTopologyAction(ChangeMeshTopologyAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshTopologyAction(ChangeMeshTopologyAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshTopologyAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshTopologyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshTopologyAction`/`Const_ChangeMeshTopologyAction` to pass it to the function.
    public class _InOptConst_ChangeMeshTopologyAction
    {
        public Const_ChangeMeshTopologyAction? Opt;

        public _InOptConst_ChangeMeshTopologyAction() {}
        public _InOptConst_ChangeMeshTopologyAction(Const_ChangeMeshTopologyAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshTopologyAction(Const_ChangeMeshTopologyAction value) {return new(value);}
    }

    /// Undo action for ObjectMeshHolder texturePerFace change
    /// Generated from class `MR::ChangeMeshTexturePerFaceAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshTexturePerFaceAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshTexturePerFaceAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshTexturePerFaceAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshTexturePerFaceAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshTexturePerFaceAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshTexturePerFaceAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshTexturePerFaceAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshTexturePerFaceAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshTexturePerFaceAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTexturePerFaceAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTexturePerFaceAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTexturePerFaceAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshTexturePerFaceAction::ChangeMeshTexturePerFaceAction`.
        public unsafe Const_ChangeMeshTexturePerFaceAction(MR._ByValue_ChangeMeshTexturePerFaceAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTexturePerFaceAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's texturePerFace data before making any changes in them
        /// Generated from constructor `MR::ChangeMeshTexturePerFaceAction::ChangeMeshTexturePerFaceAction`.
        public unsafe Const_ChangeMeshTexturePerFaceAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTexturePerFaceAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's texturePerFace data and immediate set new value
        /// Generated from constructor `MR::ChangeMeshTexturePerFaceAction::ChangeMeshTexturePerFaceAction`.
        public unsafe Const_ChangeMeshTexturePerFaceAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.TexturePerFace> newTexturePerFace) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.TexturePerFace._Underlying *newTexturePerFace);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTexturePerFaceAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newTexturePerFace.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshTexturePerFaceAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshTexturePerFaceAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshTexturePerFaceAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshTexturePerFaceAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMeshHolder obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshTexturePerFaceAction_setObjectDirty(MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            __MR_ChangeMeshTexturePerFaceAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshTexturePerFaceAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshTexturePerFaceAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshTexturePerFaceAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMeshHolder texturePerFace change
    /// Generated from class `MR::ChangeMeshTexturePerFaceAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshTexturePerFaceAction : Const_ChangeMeshTexturePerFaceAction
    {
        internal unsafe ChangeMeshTexturePerFaceAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshTexturePerFaceAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshTexturePerFaceAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshTexturePerFaceAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshTexturePerFaceAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTexturePerFaceAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTexturePerFaceAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshTexturePerFaceAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshTexturePerFaceAction::ChangeMeshTexturePerFaceAction`.
        public unsafe ChangeMeshTexturePerFaceAction(MR._ByValue_ChangeMeshTexturePerFaceAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTexturePerFaceAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshTexturePerFaceAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's texturePerFace data before making any changes in them
        /// Generated from constructor `MR::ChangeMeshTexturePerFaceAction::ChangeMeshTexturePerFaceAction`.
        public unsafe ChangeMeshTexturePerFaceAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTexturePerFaceAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's texturePerFace data and immediate set new value
        /// Generated from constructor `MR::ChangeMeshTexturePerFaceAction::ChangeMeshTexturePerFaceAction`.
        public unsafe ChangeMeshTexturePerFaceAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.TexturePerFace> newTexturePerFace) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.TexturePerFace._Underlying *newTexturePerFace);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshTexturePerFaceAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newTexturePerFace.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshTexturePerFaceAction::operator=`.
        public unsafe MR.ChangeMeshTexturePerFaceAction Assign(MR._ByValue_ChangeMeshTexturePerFaceAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshTexturePerFaceAction._Underlying *__MR_ChangeMeshTexturePerFaceAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshTexturePerFaceAction._Underlying *_other);
            return new(__MR_ChangeMeshTexturePerFaceAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshTexturePerFaceAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshTexturePerFaceAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshTexturePerFaceAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshTexturePerFaceAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshTexturePerFaceAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshTexturePerFaceAction`/`Const_ChangeMeshTexturePerFaceAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshTexturePerFaceAction
    {
        internal readonly Const_ChangeMeshTexturePerFaceAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshTexturePerFaceAction(Const_ChangeMeshTexturePerFaceAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshTexturePerFaceAction(Const_ChangeMeshTexturePerFaceAction arg) {return new(arg);}
        public _ByValue_ChangeMeshTexturePerFaceAction(MR.Misc._Moved<ChangeMeshTexturePerFaceAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshTexturePerFaceAction(MR.Misc._Moved<ChangeMeshTexturePerFaceAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshTexturePerFaceAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshTexturePerFaceAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshTexturePerFaceAction`/`Const_ChangeMeshTexturePerFaceAction` directly.
    public class _InOptMut_ChangeMeshTexturePerFaceAction
    {
        public ChangeMeshTexturePerFaceAction? Opt;

        public _InOptMut_ChangeMeshTexturePerFaceAction() {}
        public _InOptMut_ChangeMeshTexturePerFaceAction(ChangeMeshTexturePerFaceAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshTexturePerFaceAction(ChangeMeshTexturePerFaceAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshTexturePerFaceAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshTexturePerFaceAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshTexturePerFaceAction`/`Const_ChangeMeshTexturePerFaceAction` to pass it to the function.
    public class _InOptConst_ChangeMeshTexturePerFaceAction
    {
        public Const_ChangeMeshTexturePerFaceAction? Opt;

        public _InOptConst_ChangeMeshTexturePerFaceAction() {}
        public _InOptConst_ChangeMeshTexturePerFaceAction(Const_ChangeMeshTexturePerFaceAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshTexturePerFaceAction(Const_ChangeMeshTexturePerFaceAction value) {return new(value);}
    }
}
