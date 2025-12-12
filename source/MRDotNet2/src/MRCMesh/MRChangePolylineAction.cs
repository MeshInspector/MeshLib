public static partial class MR
{
    /// Undo action for ObjectLines polyline change
    /// Generated from class `MR::ChangePolylineAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePolylineAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePolylineAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePolylineAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePolylineAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePolylineAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePolylineAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePolylineAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylineAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylineAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePolylineAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePolylineAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePolylineAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylineAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePolylineAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePolylineAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePolylineAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePolylineAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePolylineAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePolylineAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePolylineAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePolylineAction::ChangePolylineAction`.
        public unsafe Const_ChangePolylineAction(MR._ByValue_ChangePolylineAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePolylineAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's polyline before making any changes in it
        /// Generated from constructor `MR::ChangePolylineAction::ChangePolylineAction`.
        public unsafe Const_ChangePolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylineAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's polyline and immediately set new polyline
        /// Generated from constructor `MR::ChangePolylineAction::ChangePolylineAction`.
        public unsafe Const_ChangePolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj, MR._ByValue_Polyline3 newPolyline) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj, MR.Misc._PassBy newPolyline_pass_by, MR.Polyline3._UnderlyingShared *newPolyline);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylineAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newPolyline.PassByMode, newPolyline.Value is not null ? newPolyline.Value._UnderlyingSharedPtr : null));
            }
        }

        /// Generated from method `MR::ChangePolylineAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePolylineAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePolylineAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePolylineAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectLines obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePolylineAction_setObjectDirty(MR.Const_ObjectLines._UnderlyingShared *obj);
            __MR_ChangePolylineAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePolylineAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePolylineAction_heapBytes(_Underlying *_this);
            return __MR_ChangePolylineAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectLines polyline change
    /// Generated from class `MR::ChangePolylineAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePolylineAction : Const_ChangePolylineAction
    {
        internal unsafe ChangePolylineAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePolylineAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePolylineAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePolylineAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePolylineAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePolylineAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePolylineAction::ChangePolylineAction`.
        public unsafe ChangePolylineAction(MR._ByValue_ChangePolylineAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePolylineAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's polyline before making any changes in it
        /// Generated from constructor `MR::ChangePolylineAction::ChangePolylineAction`.
        public unsafe ChangePolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylineAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's polyline and immediately set new polyline
        /// Generated from constructor `MR::ChangePolylineAction::ChangePolylineAction`.
        public unsafe ChangePolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj, MR._ByValue_Polyline3 newPolyline) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj, MR.Misc._PassBy newPolyline_pass_by, MR.Polyline3._UnderlyingShared *newPolyline);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylineAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newPolyline.PassByMode, newPolyline.Value is not null ? newPolyline.Value._UnderlyingSharedPtr : null));
            }
        }

        /// Generated from method `MR::ChangePolylineAction::operator=`.
        public unsafe MR.ChangePolylineAction Assign(MR._ByValue_ChangePolylineAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylineAction._Underlying *__MR_ChangePolylineAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePolylineAction._Underlying *_other);
            return new(__MR_ChangePolylineAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePolylineAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePolylineAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePolylineAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePolylineAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePolylineAction`/`Const_ChangePolylineAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePolylineAction
    {
        internal readonly Const_ChangePolylineAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePolylineAction(Const_ChangePolylineAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePolylineAction(Const_ChangePolylineAction arg) {return new(arg);}
        public _ByValue_ChangePolylineAction(MR.Misc._Moved<ChangePolylineAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePolylineAction(MR.Misc._Moved<ChangePolylineAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePolylineAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePolylineAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePolylineAction`/`Const_ChangePolylineAction` directly.
    public class _InOptMut_ChangePolylineAction
    {
        public ChangePolylineAction? Opt;

        public _InOptMut_ChangePolylineAction() {}
        public _InOptMut_ChangePolylineAction(ChangePolylineAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePolylineAction(ChangePolylineAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePolylineAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePolylineAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePolylineAction`/`Const_ChangePolylineAction` to pass it to the function.
    public class _InOptConst_ChangePolylineAction
    {
        public Const_ChangePolylineAction? Opt;

        public _InOptConst_ChangePolylineAction() {}
        public _InOptConst_ChangePolylineAction(Const_ChangePolylineAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePolylineAction(Const_ChangePolylineAction value) {return new(value);}
    }

    /// Undo action for ObjectLines points only (not topology) change
    /// Generated from class `MR::ChangePolylinePointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePolylinePointsAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePolylinePointsAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePolylinePointsAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePolylinePointsAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePolylinePointsAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePolylinePointsAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylinePointsAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylinePointsAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePolylinePointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePolylinePointsAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePolylinePointsAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylinePointsAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylinePointsAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylinePointsAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePolylinePointsAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePolylinePointsAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePolylinePointsAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePolylinePointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePolylinePointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePolylinePointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePolylinePointsAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylinePointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylinePointsAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylinePointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePolylinePointsAction::ChangePolylinePointsAction`.
        public unsafe Const_ChangePolylinePointsAction(MR._ByValue_ChangePolylinePointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylinePointsAction._Underlying *__MR_ChangePolylinePointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePolylinePointsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePolylinePointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's lines points before making any changes in it
        /// Generated from constructor `MR::ChangePolylinePointsAction::ChangePolylinePointsAction`.
        public unsafe Const_ChangePolylinePointsAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePolylinePointsAction._Underlying *__MR_ChangePolylinePointsAction_Construct(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylinePointsAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePolylinePointsAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePolylinePointsAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePolylinePointsAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePolylinePointsAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectLines obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePolylinePointsAction_setObjectDirty(MR.Const_ObjectLines._UnderlyingShared *obj);
            __MR_ChangePolylinePointsAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePolylinePointsAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePolylinePointsAction_heapBytes(_Underlying *_this);
            return __MR_ChangePolylinePointsAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectLines points only (not topology) change
    /// Generated from class `MR::ChangePolylinePointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePolylinePointsAction : Const_ChangePolylinePointsAction
    {
        internal unsafe ChangePolylinePointsAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePolylinePointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePolylinePointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePolylinePointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePolylinePointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePolylinePointsAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylinePointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylinePointsAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylinePointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePolylinePointsAction::ChangePolylinePointsAction`.
        public unsafe ChangePolylinePointsAction(MR._ByValue_ChangePolylinePointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylinePointsAction._Underlying *__MR_ChangePolylinePointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePolylinePointsAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePolylinePointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's lines points before making any changes in it
        /// Generated from constructor `MR::ChangePolylinePointsAction::ChangePolylinePointsAction`.
        public unsafe ChangePolylinePointsAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePolylinePointsAction._Underlying *__MR_ChangePolylinePointsAction_Construct(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylinePointsAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePolylinePointsAction::operator=`.
        public unsafe MR.ChangePolylinePointsAction Assign(MR._ByValue_ChangePolylinePointsAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylinePointsAction._Underlying *__MR_ChangePolylinePointsAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePolylinePointsAction._Underlying *_other);
            return new(__MR_ChangePolylinePointsAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePolylinePointsAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylinePointsAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePolylinePointsAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePolylinePointsAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePolylinePointsAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePolylinePointsAction`/`Const_ChangePolylinePointsAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePolylinePointsAction
    {
        internal readonly Const_ChangePolylinePointsAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePolylinePointsAction(Const_ChangePolylinePointsAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePolylinePointsAction(Const_ChangePolylinePointsAction arg) {return new(arg);}
        public _ByValue_ChangePolylinePointsAction(MR.Misc._Moved<ChangePolylinePointsAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePolylinePointsAction(MR.Misc._Moved<ChangePolylinePointsAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePolylinePointsAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePolylinePointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePolylinePointsAction`/`Const_ChangePolylinePointsAction` directly.
    public class _InOptMut_ChangePolylinePointsAction
    {
        public ChangePolylinePointsAction? Opt;

        public _InOptMut_ChangePolylinePointsAction() {}
        public _InOptMut_ChangePolylinePointsAction(ChangePolylinePointsAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePolylinePointsAction(ChangePolylinePointsAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePolylinePointsAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePolylinePointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePolylinePointsAction`/`Const_ChangePolylinePointsAction` to pass it to the function.
    public class _InOptConst_ChangePolylinePointsAction
    {
        public Const_ChangePolylinePointsAction? Opt;

        public _InOptConst_ChangePolylinePointsAction() {}
        public _InOptConst_ChangePolylinePointsAction(Const_ChangePolylinePointsAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePolylinePointsAction(Const_ChangePolylinePointsAction value) {return new(value);}
    }

    /// Undo action for ObjectLines topology only (not points) change
    /// Generated from class `MR::ChangePolylineTopologyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangePolylineTopologyAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangePolylineTopologyAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangePolylineTopologyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangePolylineTopologyAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangePolylineTopologyAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangePolylineTopologyAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangePolylineTopologyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangePolylineTopologyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePolylineTopologyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangePolylineTopologyAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineTopologyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineTopologyAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineTopologyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePolylineTopologyAction::ChangePolylineTopologyAction`.
        public unsafe Const_ChangePolylineTopologyAction(MR._ByValue_ChangePolylineTopologyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylineTopologyAction._Underlying *__MR_ChangePolylineTopologyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineTopologyAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePolylineTopologyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's lines points before making any changes in it
        /// Generated from constructor `MR::ChangePolylineTopologyAction::ChangePolylineTopologyAction`.
        public unsafe Const_ChangePolylineTopologyAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePolylineTopologyAction._Underlying *__MR_ChangePolylineTopologyAction_Construct(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylineTopologyAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePolylineTopologyAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangePolylineTopologyAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangePolylineTopologyAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangePolylineTopologyAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectLines obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangePolylineTopologyAction_setObjectDirty(MR.Const_ObjectLines._UnderlyingShared *obj);
            __MR_ChangePolylineTopologyAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangePolylineTopologyAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangePolylineTopologyAction_heapBytes(_Underlying *_this);
            return __MR_ChangePolylineTopologyAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for ObjectLines topology only (not points) change
    /// Generated from class `MR::ChangePolylineTopologyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangePolylineTopologyAction : Const_ChangePolylineTopologyAction
    {
        internal unsafe ChangePolylineTopologyAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangePolylineTopologyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangePolylineTopologyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangePolylineTopologyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangePolylineTopologyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangePolylineTopologyAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineTopologyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineTopologyAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangePolylineTopologyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangePolylineTopologyAction::ChangePolylineTopologyAction`.
        public unsafe ChangePolylineTopologyAction(MR._ByValue_ChangePolylineTopologyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylineTopologyAction._Underlying *__MR_ChangePolylineTopologyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangePolylineTopologyAction._Underlying *_other);
            _LateMakeShared(__MR_ChangePolylineTopologyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's lines points before making any changes in it
        /// Generated from constructor `MR::ChangePolylineTopologyAction::ChangePolylineTopologyAction`.
        public unsafe ChangePolylineTopologyAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_Construct", ExactSpelling = true)]
            extern static MR.ChangePolylineTopologyAction._Underlying *__MR_ChangePolylineTopologyAction_Construct(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangePolylineTopologyAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::ChangePolylineTopologyAction::operator=`.
        public unsafe MR.ChangePolylineTopologyAction Assign(MR._ByValue_ChangePolylineTopologyAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangePolylineTopologyAction._Underlying *__MR_ChangePolylineTopologyAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangePolylineTopologyAction._Underlying *_other);
            return new(__MR_ChangePolylineTopologyAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangePolylineTopologyAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangePolylineTopologyAction_action", ExactSpelling = true)]
            extern static void __MR_ChangePolylineTopologyAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangePolylineTopologyAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangePolylineTopologyAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangePolylineTopologyAction`/`Const_ChangePolylineTopologyAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangePolylineTopologyAction
    {
        internal readonly Const_ChangePolylineTopologyAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangePolylineTopologyAction(Const_ChangePolylineTopologyAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangePolylineTopologyAction(Const_ChangePolylineTopologyAction arg) {return new(arg);}
        public _ByValue_ChangePolylineTopologyAction(MR.Misc._Moved<ChangePolylineTopologyAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangePolylineTopologyAction(MR.Misc._Moved<ChangePolylineTopologyAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangePolylineTopologyAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangePolylineTopologyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePolylineTopologyAction`/`Const_ChangePolylineTopologyAction` directly.
    public class _InOptMut_ChangePolylineTopologyAction
    {
        public ChangePolylineTopologyAction? Opt;

        public _InOptMut_ChangePolylineTopologyAction() {}
        public _InOptMut_ChangePolylineTopologyAction(ChangePolylineTopologyAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangePolylineTopologyAction(ChangePolylineTopologyAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangePolylineTopologyAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangePolylineTopologyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangePolylineTopologyAction`/`Const_ChangePolylineTopologyAction` to pass it to the function.
    public class _InOptConst_ChangePolylineTopologyAction
    {
        public Const_ChangePolylineTopologyAction? Opt;

        public _InOptConst_ChangePolylineTopologyAction() {}
        public _InOptConst_ChangePolylineTopologyAction(Const_ChangePolylineTopologyAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangePolylineTopologyAction(Const_ChangePolylineTopologyAction value) {return new(value);}
    }

    /// Undo action that modifies one point's coordinates inside ObjectPolyline
    /// Generated from class `MR::ChangeOnePointInPolylineAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeOnePointInPolylineAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeOnePointInPolylineAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeOnePointInPolylineAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeOnePointInPolylineAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeOnePointInPolylineAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeOnePointInPolylineAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeOnePointInPolylineAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeOnePointInPolylineAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeOnePointInPolylineAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeOnePointInPolylineAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInPolylineAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInPolylineAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInPolylineAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeOnePointInPolylineAction::ChangeOnePointInPolylineAction`.
        public unsafe Const_ChangeOnePointInPolylineAction(MR._ByValue_ChangeOnePointInPolylineAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInPolylineAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeOnePointInPolylineAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point's coordinates before making any changes in it
        /// Generated from constructor `MR::ChangeOnePointInPolylineAction::ChangeOnePointInPolylineAction`.
        public unsafe Const_ChangeOnePointInPolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj, MR.VertId pointId) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj, MR.VertId pointId);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInPolylineAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId));
            }
        }

        /// use this constructor to remember point's coordinates and immediate set new coordinates
        /// Generated from constructor `MR::ChangeOnePointInPolylineAction::ChangeOnePointInPolylineAction`.
        public unsafe Const_ChangeOnePointInPolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj, MR.VertId pointId, MR.Const_Vector3f newCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_Construct_4(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj, MR.VertId pointId, MR.Const_Vector3f._Underlying *newCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInPolylineAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId, newCoords._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeOnePointInPolylineAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeOnePointInPolylineAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeOnePointInPolylineAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeOnePointInPolylineAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectLines obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeOnePointInPolylineAction_setObjectDirty(MR.Const_ObjectLines._UnderlyingShared *obj);
            __MR_ChangeOnePointInPolylineAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeOnePointInPolylineAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeOnePointInPolylineAction_heapBytes(_Underlying *_this);
            return __MR_ChangeOnePointInPolylineAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action that modifies one point's coordinates inside ObjectPolyline
    /// Generated from class `MR::ChangeOnePointInPolylineAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeOnePointInPolylineAction : Const_ChangeOnePointInPolylineAction
    {
        internal unsafe ChangeOnePointInPolylineAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeOnePointInPolylineAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeOnePointInPolylineAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeOnePointInPolylineAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeOnePointInPolylineAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeOnePointInPolylineAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInPolylineAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInPolylineAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeOnePointInPolylineAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeOnePointInPolylineAction::ChangeOnePointInPolylineAction`.
        public unsafe ChangeOnePointInPolylineAction(MR._ByValue_ChangeOnePointInPolylineAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInPolylineAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeOnePointInPolylineAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember point's coordinates before making any changes in it
        /// Generated from constructor `MR::ChangeOnePointInPolylineAction::ChangeOnePointInPolylineAction`.
        public unsafe ChangeOnePointInPolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj, MR.VertId pointId) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj, MR.VertId pointId);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInPolylineAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId));
            }
        }

        /// use this constructor to remember point's coordinates and immediate set new coordinates
        /// Generated from constructor `MR::ChangeOnePointInPolylineAction::ChangeOnePointInPolylineAction`.
        public unsafe ChangeOnePointInPolylineAction(ReadOnlySpan<char> name, MR.Const_ObjectLines obj, MR.VertId pointId, MR.Const_Vector3f newCoords) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_Construct_4", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_Construct_4(byte *name, byte *name_end, MR.Const_ObjectLines._UnderlyingShared *obj, MR.VertId pointId, MR.Const_Vector3f._Underlying *newCoords);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeOnePointInPolylineAction_Construct_4(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, pointId, newCoords._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeOnePointInPolylineAction::operator=`.
        public unsafe MR.ChangeOnePointInPolylineAction Assign(MR._ByValue_ChangeOnePointInPolylineAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeOnePointInPolylineAction._Underlying *__MR_ChangeOnePointInPolylineAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeOnePointInPolylineAction._Underlying *_other);
            return new(__MR_ChangeOnePointInPolylineAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeOnePointInPolylineAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeOnePointInPolylineAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeOnePointInPolylineAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeOnePointInPolylineAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeOnePointInPolylineAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeOnePointInPolylineAction`/`Const_ChangeOnePointInPolylineAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeOnePointInPolylineAction
    {
        internal readonly Const_ChangeOnePointInPolylineAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeOnePointInPolylineAction(Const_ChangeOnePointInPolylineAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeOnePointInPolylineAction(Const_ChangeOnePointInPolylineAction arg) {return new(arg);}
        public _ByValue_ChangeOnePointInPolylineAction(MR.Misc._Moved<ChangeOnePointInPolylineAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeOnePointInPolylineAction(MR.Misc._Moved<ChangeOnePointInPolylineAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeOnePointInPolylineAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeOnePointInPolylineAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeOnePointInPolylineAction`/`Const_ChangeOnePointInPolylineAction` directly.
    public class _InOptMut_ChangeOnePointInPolylineAction
    {
        public ChangeOnePointInPolylineAction? Opt;

        public _InOptMut_ChangeOnePointInPolylineAction() {}
        public _InOptMut_ChangeOnePointInPolylineAction(ChangeOnePointInPolylineAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeOnePointInPolylineAction(ChangeOnePointInPolylineAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeOnePointInPolylineAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeOnePointInPolylineAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeOnePointInPolylineAction`/`Const_ChangeOnePointInPolylineAction` to pass it to the function.
    public class _InOptConst_ChangeOnePointInPolylineAction
    {
        public Const_ChangeOnePointInPolylineAction? Opt;

        public _InOptConst_ChangeOnePointInPolylineAction() {}
        public _InOptConst_ChangeOnePointInPolylineAction(Const_ChangeOnePointInPolylineAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeOnePointInPolylineAction(Const_ChangeOnePointInPolylineAction value) {return new(value);}
    }
}
