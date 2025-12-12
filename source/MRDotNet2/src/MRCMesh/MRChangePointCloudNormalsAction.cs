public static partial class MR
{
    /// Undo action for changing normals in PointCloud
    /// Generated from class `MR::ChangePointCloudNormalsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePointCloudNormalsAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePointCloudNormalsAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePointCloudNormalsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePointCloudNormalsAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePointCloudNormalsAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePointCloudNormalsAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePointCloudNormalsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePointCloudNormalsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointCloudNormalsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePointCloudNormalsAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudNormalsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudNormalsAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudNormalsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointCloudNormalsAction::ChangePointCloudNormalsAction`.
        public unsafe Const_ChangePointCloudNormalsAction(MR._ByValue_ChangePointCloudNormalsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudNormalsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointCloudNormalsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point cloud's normals before making any changes in them
        /// Generated from constructor `MR::ChangePointCloudNormalsAction::ChangePointCloudNormalsAction`.
        public unsafe Const_ChangePointCloudNormalsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudNormalsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember point cloud's normals and immediate set new value
        /// Generated from constructor `MR::ChangePointCloudNormalsAction::ChangePointCloudNormalsAction`.
        public unsafe Const_ChangePointCloudNormalsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.Misc._Moved<MR.VertCoords> newNormals) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertCoords._Underlying *newNormals);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudNormalsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newNormals.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangePointCloudNormalsAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePointCloudNormalsAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePointCloudNormalsAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePointCloudNormalsAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectPoints obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePointCloudNormalsAction_setObjectDirty(MR.Const_ObjectPoints._UnderlyingShared *obj);
            __MR_ChangePointCloudNormalsAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePointCloudNormalsAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePointCloudNormalsAction_heapBytes(_Underlying *_this);
            return __MR_ChangePointCloudNormalsAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for changing normals in PointCloud
    /// Generated from class `MR::ChangePointCloudNormalsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePointCloudNormalsAction : Const_ChangePointCloudNormalsAction
    {
        internal unsafe ChangePointCloudNormalsAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePointCloudNormalsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePointCloudNormalsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePointCloudNormalsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePointCloudNormalsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePointCloudNormalsAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudNormalsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudNormalsAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePointCloudNormalsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePointCloudNormalsAction::ChangePointCloudNormalsAction`.
        public unsafe ChangePointCloudNormalsAction(MR._ByValue_ChangePointCloudNormalsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudNormalsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePointCloudNormalsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point cloud's normals before making any changes in them
        /// Generated from constructor `MR::ChangePointCloudNormalsAction::ChangePointCloudNormalsAction`.
        public unsafe ChangePointCloudNormalsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudNormalsAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember point cloud's normals and immediate set new value
        /// Generated from constructor `MR::ChangePointCloudNormalsAction::ChangePointCloudNormalsAction`.
        public unsafe ChangePointCloudNormalsAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.Misc._Moved<MR.VertCoords> newNormals) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertCoords._Underlying *newNormals);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePointCloudNormalsAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newNormals.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangePointCloudNormalsAction::operator=`.
        public unsafe MR.ChangePointCloudNormalsAction Assign(MR._ByValue_ChangePointCloudNormalsAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePointCloudNormalsAction._Underlying *__MR_ChangePointCloudNormalsAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePointCloudNormalsAction._Underlying *_other);
            return new(__MR_ChangePointCloudNormalsAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePointCloudNormalsAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePointCloudNormalsAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePointCloudNormalsAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePointCloudNormalsAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePointCloudNormalsAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePointCloudNormalsAction`/`Const_ChangePointCloudNormalsAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePointCloudNormalsAction
    {
        internal readonly Const_ChangePointCloudNormalsAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePointCloudNormalsAction(Const_ChangePointCloudNormalsAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePointCloudNormalsAction(Const_ChangePointCloudNormalsAction arg) {return new(arg);}
        public _ByValue_ChangePointCloudNormalsAction(MR.Misc._Moved<ChangePointCloudNormalsAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePointCloudNormalsAction(MR.Misc._Moved<ChangePointCloudNormalsAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePointCloudNormalsAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePointCloudNormalsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointCloudNormalsAction`/`Const_ChangePointCloudNormalsAction` directly.
    public class _InOptMut_ChangePointCloudNormalsAction
    {
        public ChangePointCloudNormalsAction? Opt;

        public _InOptMut_ChangePointCloudNormalsAction() {}
        public _InOptMut_ChangePointCloudNormalsAction(ChangePointCloudNormalsAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePointCloudNormalsAction(ChangePointCloudNormalsAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePointCloudNormalsAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePointCloudNormalsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePointCloudNormalsAction`/`Const_ChangePointCloudNormalsAction` to pass it to the function.
    public class _InOptConst_ChangePointCloudNormalsAction
    {
        public Const_ChangePointCloudNormalsAction? Opt;

        public _InOptConst_ChangePointCloudNormalsAction() {}
        public _InOptConst_ChangePointCloudNormalsAction(Const_ChangePointCloudNormalsAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePointCloudNormalsAction(Const_ChangePointCloudNormalsAction value) {return new(value);}
    }

    /// Undo action that modifies one point's normal inside ObjectPoints
    /// Generated from class `MR::ChangeOneNormalInCloudAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeOneNormalInCloudAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeOneNormalInCloudAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeOneNormalInCloudAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeOneNormalInCloudAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeOneNormalInCloudAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeOneNormalInCloudAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeOneNormalInCloudAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeOneNormalInCloudAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeOneNormalInCloudAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeOneNormalInCloudAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeOneNormalInCloudAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeOneNormalInCloudAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeOneNormalInCloudAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeOneNormalInCloudAction::ChangeOneNormalInCloudAction`.
        public unsafe Const_ChangeOneNormalInCloudAction(MR._ByValue_ChangeOneNormalInCloudAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeOneNormalInCloudAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeOneNormalInCloudAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point's normal before making any changes in it
        /// Generated from constructor `MR::ChangeOneNormalInCloudAction::ChangeOneNormalInCloudAction`.
        public unsafe Const_ChangeOneNormalInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOneNormalInCloudAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId));
            }
        }

        /// use this constructor to remember point's normal and immediate set new normal
        /// Generated from constructor `MR::ChangeOneNormalInCloudAction::ChangeOneNormalInCloudAction`.
        public unsafe Const_ChangeOneNormalInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId, MR.Const_Vector3f newNormal) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_Construct_4(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId, MR.Const_Vector3f._Underlying *newNormal);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOneNormalInCloudAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId, newNormal._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeOneNormalInCloudAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeOneNormalInCloudAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeOneNormalInCloudAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeOneNormalInCloudAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectPoints obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeOneNormalInCloudAction_setObjectDirty(MR.Const_ObjectPoints._UnderlyingShared *obj);
            __MR_ChangeOneNormalInCloudAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeOneNormalInCloudAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeOneNormalInCloudAction_heapBytes(_Underlying *_this);
            return __MR_ChangeOneNormalInCloudAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action that modifies one point's normal inside ObjectPoints
    /// Generated from class `MR::ChangeOneNormalInCloudAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeOneNormalInCloudAction : Const_ChangeOneNormalInCloudAction
    {
        internal unsafe ChangeOneNormalInCloudAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeOneNormalInCloudAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeOneNormalInCloudAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeOneNormalInCloudAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeOneNormalInCloudAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeOneNormalInCloudAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeOneNormalInCloudAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeOneNormalInCloudAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeOneNormalInCloudAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeOneNormalInCloudAction::ChangeOneNormalInCloudAction`.
        public unsafe ChangeOneNormalInCloudAction(MR._ByValue_ChangeOneNormalInCloudAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeOneNormalInCloudAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeOneNormalInCloudAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point's normal before making any changes in it
        /// Generated from constructor `MR::ChangeOneNormalInCloudAction::ChangeOneNormalInCloudAction`.
        public unsafe ChangeOneNormalInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOneNormalInCloudAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId));
            }
        }

        /// use this constructor to remember point's normal and immediate set new normal
        /// Generated from constructor `MR::ChangeOneNormalInCloudAction::ChangeOneNormalInCloudAction`.
        public unsafe ChangeOneNormalInCloudAction(ReadOnlySpan<char> name, MR.Const_ObjectPoints obj, MR.VertId pointId, MR.Const_Vector3f newNormal) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_Construct_4(byte *name, byte *name_end, MR.Const_ObjectPoints._UnderlyingShared *obj, MR.VertId pointId, MR.Const_Vector3f._Underlying *newNormal);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOneNormalInCloudAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId, newNormal._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeOneNormalInCloudAction::operator=`.
        public unsafe MR.ChangeOneNormalInCloudAction Assign(MR._ByValue_ChangeOneNormalInCloudAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOneNormalInCloudAction._Underlying *__MR_ChangeOneNormalInCloudAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeOneNormalInCloudAction._Underlying *_other);
            return new(__MR_ChangeOneNormalInCloudAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeOneNormalInCloudAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOneNormalInCloudAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeOneNormalInCloudAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeOneNormalInCloudAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeOneNormalInCloudAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeOneNormalInCloudAction`/`Const_ChangeOneNormalInCloudAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeOneNormalInCloudAction
    {
        internal readonly Const_ChangeOneNormalInCloudAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeOneNormalInCloudAction(Const_ChangeOneNormalInCloudAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeOneNormalInCloudAction(Const_ChangeOneNormalInCloudAction arg) {return new(arg);}
        public _ByValue_ChangeOneNormalInCloudAction(MR.Misc._Moved<ChangeOneNormalInCloudAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeOneNormalInCloudAction(MR.Misc._Moved<ChangeOneNormalInCloudAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeOneNormalInCloudAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeOneNormalInCloudAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeOneNormalInCloudAction`/`Const_ChangeOneNormalInCloudAction` directly.
    public class _InOptMut_ChangeOneNormalInCloudAction
    {
        public ChangeOneNormalInCloudAction? Opt;

        public _InOptMut_ChangeOneNormalInCloudAction() {}
        public _InOptMut_ChangeOneNormalInCloudAction(ChangeOneNormalInCloudAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeOneNormalInCloudAction(ChangeOneNormalInCloudAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeOneNormalInCloudAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeOneNormalInCloudAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeOneNormalInCloudAction`/`Const_ChangeOneNormalInCloudAction` to pass it to the function.
    public class _InOptConst_ChangeOneNormalInCloudAction
    {
        public Const_ChangeOneNormalInCloudAction? Opt;

        public _InOptConst_ChangeOneNormalInCloudAction() {}
        public _InOptConst_ChangeOneNormalInCloudAction(Const_ChangeOneNormalInCloudAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeOneNormalInCloudAction(Const_ChangeOneNormalInCloudAction value) {return new(value);}
    }
}
