public static partial class MR
{
    /// Undo action for ObjectMeshData change
    /// Generated from class `MR::ChangeMeshDataAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeMeshDataAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeMeshDataAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeMeshDataAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeMeshDataAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeMeshDataAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeMeshDataAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshDataAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshDataAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeMeshDataAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeMeshDataAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeMeshDataAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeMeshDataAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeMeshDataAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeMeshDataAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeMeshDataAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeMeshDataAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeMeshDataAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeMeshDataAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeMeshDataAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshDataAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeMeshDataAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshDataAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshDataAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshDataAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshDataAction::ChangeMeshDataAction`.
        public unsafe Const_ChangeMeshDataAction(MR._ByValue_ChangeMeshDataAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshDataAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshDataAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's data before making any changes in it
        /// Generated from constructor `MR::ChangeMeshDataAction::ChangeMeshDataAction`.
        public unsafe Const_ChangeMeshDataAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, bool cloneMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_Construct_bool", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_Construct_bool(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, byte cloneMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshDataAction_Construct_bool(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, cloneMesh ? (byte)1 : (byte)0));
            }
        }

        /// use this constructor to remember object's data and immediately set new data
        /// Generated from constructor `MR::ChangeMeshDataAction::ChangeMeshDataAction`.
        public unsafe Const_ChangeMeshDataAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.ObjectMeshData> newData) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_Construct_MR_ObjectMeshData", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_Construct_MR_ObjectMeshData(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.ObjectMeshData._Underlying *newData);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshDataAction_Construct_MR_ObjectMeshData(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newData.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshDataAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeMeshDataAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeMeshDataAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeMeshDataAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMesh obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeMeshDataAction_setObjectDirty(MR.Const_ObjectMesh._UnderlyingShared *obj);
            __MR_ChangeMeshDataAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeMeshDataAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeMeshDataAction_heapBytes(_Underlying *_this);
            return __MR_ChangeMeshDataAction_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::ChangeMeshDataAction::obj`.
        public unsafe MR.Const_ObjectMesh Obj()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_obj", ExactSpelling = true)]
            extern static MR.Const_ObjectMesh._UnderlyingShared *__MR_ChangeMeshDataAction_obj(_Underlying *_this);
            return new(__MR_ChangeMeshDataAction_obj(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshDataAction::data`.
        public unsafe MR.Const_ObjectMeshData Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_data", ExactSpelling = true)]
            extern static MR.Const_ObjectMeshData._Underlying *__MR_ChangeMeshDataAction_data(_Underlying *_this);
            return new(__MR_ChangeMeshDataAction_data(_UnderlyingPtr), is_owning: false);
        }
    }

    /// Undo action for ObjectMeshData change
    /// Generated from class `MR::ChangeMeshDataAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeMeshDataAction : Const_ChangeMeshDataAction
    {
        internal unsafe ChangeMeshDataAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeMeshDataAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeMeshDataAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeMeshDataAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeMeshDataAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeMeshDataAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshDataAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshDataAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeMeshDataAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeMeshDataAction::ChangeMeshDataAction`.
        public unsafe ChangeMeshDataAction(MR._ByValue_ChangeMeshDataAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeMeshDataAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeMeshDataAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's data before making any changes in it
        /// Generated from constructor `MR::ChangeMeshDataAction::ChangeMeshDataAction`.
        public unsafe ChangeMeshDataAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, bool cloneMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_Construct_bool", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_Construct_bool(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, byte cloneMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshDataAction_Construct_bool(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, cloneMesh ? (byte)1 : (byte)0));
            }
        }

        /// use this constructor to remember object's data and immediately set new data
        /// Generated from constructor `MR::ChangeMeshDataAction::ChangeMeshDataAction`.
        public unsafe ChangeMeshDataAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.ObjectMeshData> newData) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_Construct_MR_ObjectMeshData", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_Construct_MR_ObjectMeshData(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.ObjectMeshData._Underlying *newData);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeMeshDataAction_Construct_MR_ObjectMeshData(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newData.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeMeshDataAction::operator=`.
        public unsafe MR.ChangeMeshDataAction Assign(MR._ByValue_ChangeMeshDataAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeMeshDataAction._Underlying *__MR_ChangeMeshDataAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeMeshDataAction._Underlying *_other);
            return new(__MR_ChangeMeshDataAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeMeshDataAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeMeshDataAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeMeshDataAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeMeshDataAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeMeshDataAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeMeshDataAction`/`Const_ChangeMeshDataAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeMeshDataAction
    {
        internal readonly Const_ChangeMeshDataAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeMeshDataAction(Const_ChangeMeshDataAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeMeshDataAction(Const_ChangeMeshDataAction arg) {return new(arg);}
        public _ByValue_ChangeMeshDataAction(MR.Misc._Moved<ChangeMeshDataAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeMeshDataAction(MR.Misc._Moved<ChangeMeshDataAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeMeshDataAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeMeshDataAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshDataAction`/`Const_ChangeMeshDataAction` directly.
    public class _InOptMut_ChangeMeshDataAction
    {
        public ChangeMeshDataAction? Opt;

        public _InOptMut_ChangeMeshDataAction() {}
        public _InOptMut_ChangeMeshDataAction(ChangeMeshDataAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeMeshDataAction(ChangeMeshDataAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeMeshDataAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeMeshDataAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeMeshDataAction`/`Const_ChangeMeshDataAction` to pass it to the function.
    public class _InOptConst_ChangeMeshDataAction
    {
        public Const_ChangeMeshDataAction? Opt;

        public _InOptConst_ChangeMeshDataAction() {}
        public _InOptConst_ChangeMeshDataAction(Const_ChangeMeshDataAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeMeshDataAction(Const_ChangeMeshDataAction value) {return new(value);}
    }

    /// Undo action for ObjectMeshData change partially
    /// Generated from class `MR::PartialChangeMeshDataAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_PartialChangeMeshDataAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_UseCount();
                return __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PartialChangeMeshDataAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PartialChangeMeshDataAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PartialChangeMeshDataAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PartialChangeMeshDataAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PartialChangeMeshDataAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_PartialChangeMeshDataAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_PartialChangeMeshDataAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshDataAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PartialChangeMeshDataAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshDataAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshDataAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshDataAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshDataAction::PartialChangeMeshDataAction`.
        public unsafe Const_PartialChangeMeshDataAction(MR._ByValue_PartialChangeMeshDataAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshDataAction._Underlying *__MR_PartialChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshDataAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshDataAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's data and immediately set new data
        /// Generated from constructor `MR::PartialChangeMeshDataAction::PartialChangeMeshDataAction`.
        public unsafe Const_PartialChangeMeshDataAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.ObjectMeshData> newData) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_Construct", ExactSpelling = true)]
            extern static MR.PartialChangeMeshDataAction._Underlying *__MR_PartialChangeMeshDataAction_Construct(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.ObjectMeshData._Underlying *newData);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshDataAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newData.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshDataAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PartialChangeMeshDataAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PartialChangeMeshDataAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PartialChangeMeshDataAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PartialChangeMeshDataAction_heapBytes(_Underlying *_this);
            return __MR_PartialChangeMeshDataAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectMeshData change partially
    /// Generated from class `MR::PartialChangeMeshDataAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class PartialChangeMeshDataAction : Const_PartialChangeMeshDataAction
    {
        internal unsafe PartialChangeMeshDataAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PartialChangeMeshDataAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(PartialChangeMeshDataAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_PartialChangeMeshDataAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshDataAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PartialChangeMeshDataAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshDataAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshDataAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshDataAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshDataAction::PartialChangeMeshDataAction`.
        public unsafe PartialChangeMeshDataAction(MR._ByValue_PartialChangeMeshDataAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshDataAction._Underlying *__MR_PartialChangeMeshDataAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshDataAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshDataAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's data and immediately set new data
        /// Generated from constructor `MR::PartialChangeMeshDataAction::PartialChangeMeshDataAction`.
        public unsafe PartialChangeMeshDataAction(ReadOnlySpan<char> name, MR.Const_ObjectMesh obj, MR.Misc._Moved<MR.ObjectMeshData> newData) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_Construct", ExactSpelling = true)]
            extern static MR.PartialChangeMeshDataAction._Underlying *__MR_PartialChangeMeshDataAction_Construct(byte *name, byte *name_end, MR.Const_ObjectMesh._UnderlyingShared *obj, MR.ObjectMeshData._Underlying *newData);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshDataAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newData.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshDataAction::operator=`.
        public unsafe MR.PartialChangeMeshDataAction Assign(MR._ByValue_PartialChangeMeshDataAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshDataAction._Underlying *__MR_PartialChangeMeshDataAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshDataAction._Underlying *_other);
            return new(__MR_PartialChangeMeshDataAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::PartialChangeMeshDataAction::action`.
        public unsafe void Action(MR.HistoryAction.Type type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshDataAction_action", ExactSpelling = true)]
            extern static void __MR_PartialChangeMeshDataAction_action(_Underlying *_this, MR.HistoryAction.Type type);
            __MR_PartialChangeMeshDataAction_action(_UnderlyingPtr, type);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PartialChangeMeshDataAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PartialChangeMeshDataAction`/`Const_PartialChangeMeshDataAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PartialChangeMeshDataAction
    {
        internal readonly Const_PartialChangeMeshDataAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PartialChangeMeshDataAction(MR.Misc._Moved<PartialChangeMeshDataAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PartialChangeMeshDataAction(MR.Misc._Moved<PartialChangeMeshDataAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshDataAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialChangeMeshDataAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshDataAction`/`Const_PartialChangeMeshDataAction` directly.
    public class _InOptMut_PartialChangeMeshDataAction
    {
        public PartialChangeMeshDataAction? Opt;

        public _InOptMut_PartialChangeMeshDataAction() {}
        public _InOptMut_PartialChangeMeshDataAction(PartialChangeMeshDataAction value) {Opt = value;}
        public static implicit operator _InOptMut_PartialChangeMeshDataAction(PartialChangeMeshDataAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshDataAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialChangeMeshDataAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshDataAction`/`Const_PartialChangeMeshDataAction` to pass it to the function.
    public class _InOptConst_PartialChangeMeshDataAction
    {
        public Const_PartialChangeMeshDataAction? Opt;

        public _InOptConst_PartialChangeMeshDataAction() {}
        public _InOptConst_PartialChangeMeshDataAction(Const_PartialChangeMeshDataAction value) {Opt = value;}
        public static implicit operator _InOptConst_PartialChangeMeshDataAction(Const_PartialChangeMeshDataAction value) {return new(value);}
    }
}
