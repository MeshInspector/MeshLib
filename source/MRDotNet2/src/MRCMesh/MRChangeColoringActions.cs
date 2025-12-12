public static partial class MR
{
    /// History action for object color palette change
    /// To use with setFrontColorsForAllViewports, setBackColorsForAllViewports, setFrontColor, setBackColor
    /// Generated from class `MR::ChangeObjectColorAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeObjectColorAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeObjectColorAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeObjectColorAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeObjectColorAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeObjectColorAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeObjectColorAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectColorAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectColorAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeObjectColorAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeObjectColorAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeObjectColorAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeObjectColorAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeObjectColorAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeObjectColorAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeObjectColorAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeObjectColorAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeObjectColorAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeObjectColorAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeObjectColorAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectColorAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeObjectColorAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectColorAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectColorAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectColorAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectColorAction::ChangeObjectColorAction`.
        public unsafe Const_ChangeObjectColorAction(MR._ByValue_ChangeObjectColorAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectColorAction._Underlying *__MR_ChangeObjectColorAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectColorAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectColorAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Constructed from original obj
        /// Generated from constructor `MR::ChangeObjectColorAction::ChangeObjectColorAction`.
        public unsafe Const_ChangeObjectColorAction(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.ChangeObjectColorAction.Type type) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeObjectColorAction._Underlying *__MR_ChangeObjectColorAction_Construct(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.ChangeObjectColorAction.Type type);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectColorAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, type));
            }
        }

        /// Generated from method `MR::ChangeObjectColorAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeObjectColorAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeObjectColorAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeObjectColorAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_VisualObject _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeObjectColorAction_setObjectDirty(MR.Const_VisualObject._UnderlyingShared *_1);
            __MR_ChangeObjectColorAction_setObjectDirty(_1._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeObjectColorAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeObjectColorAction_heapBytes(_Underlying *_this);
            return __MR_ChangeObjectColorAction_heapBytes(_UnderlyingPtr);
        }

        public enum Type : int
        {
            Unselected = 0,
            Selected = 1,
            Back = 2,
        }
    }

    /// History action for object color palette change
    /// To use with setFrontColorsForAllViewports, setBackColorsForAllViewports, setFrontColor, setBackColor
    /// Generated from class `MR::ChangeObjectColorAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeObjectColorAction : Const_ChangeObjectColorAction
    {
        internal unsafe ChangeObjectColorAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeObjectColorAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeObjectColorAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeObjectColorAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeObjectColorAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeObjectColorAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectColorAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectColorAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeObjectColorAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeObjectColorAction::ChangeObjectColorAction`.
        public unsafe ChangeObjectColorAction(MR._ByValue_ChangeObjectColorAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectColorAction._Underlying *__MR_ChangeObjectColorAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeObjectColorAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeObjectColorAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Constructed from original obj
        /// Generated from constructor `MR::ChangeObjectColorAction::ChangeObjectColorAction`.
        public unsafe ChangeObjectColorAction(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.ChangeObjectColorAction.Type type) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_Construct", ExactSpelling = true)]
            extern static MR.ChangeObjectColorAction._Underlying *__MR_ChangeObjectColorAction_Construct(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.ChangeObjectColorAction.Type type);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeObjectColorAction_Construct(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, type));
            }
        }

        /// Generated from method `MR::ChangeObjectColorAction::operator=`.
        public unsafe MR.ChangeObjectColorAction Assign(MR._ByValue_ChangeObjectColorAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeObjectColorAction._Underlying *__MR_ChangeObjectColorAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeObjectColorAction._Underlying *_other);
            return new(__MR_ChangeObjectColorAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeObjectColorAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeObjectColorAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeObjectColorAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeObjectColorAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeObjectColorAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeObjectColorAction`/`Const_ChangeObjectColorAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeObjectColorAction
    {
        internal readonly Const_ChangeObjectColorAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeObjectColorAction(Const_ChangeObjectColorAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeObjectColorAction(Const_ChangeObjectColorAction arg) {return new(arg);}
        public _ByValue_ChangeObjectColorAction(MR.Misc._Moved<ChangeObjectColorAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeObjectColorAction(MR.Misc._Moved<ChangeObjectColorAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeObjectColorAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeObjectColorAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectColorAction`/`Const_ChangeObjectColorAction` directly.
    public class _InOptMut_ChangeObjectColorAction
    {
        public ChangeObjectColorAction? Opt;

        public _InOptMut_ChangeObjectColorAction() {}
        public _InOptMut_ChangeObjectColorAction(ChangeObjectColorAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeObjectColorAction(ChangeObjectColorAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeObjectColorAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeObjectColorAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeObjectColorAction`/`Const_ChangeObjectColorAction` to pass it to the function.
    public class _InOptConst_ChangeObjectColorAction
    {
        public Const_ChangeObjectColorAction? Opt;

        public _InOptConst_ChangeObjectColorAction() {}
        public _InOptConst_ChangeObjectColorAction(Const_ChangeObjectColorAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeObjectColorAction(Const_ChangeObjectColorAction value) {return new(value);}
    }

    /// History action for faces color map change
    /// Generated from class `MR::ChangeFacesColorMapAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeFacesColorMapAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeFacesColorMapAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeFacesColorMapAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeFacesColorMapAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeFacesColorMapAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeFacesColorMapAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeFacesColorMapAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeFacesColorMapAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeFacesColorMapAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeFacesColorMapAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeFacesColorMapAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeFacesColorMapAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeFacesColorMapAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeFacesColorMapAction::ChangeFacesColorMapAction`.
        public unsafe Const_ChangeFacesColorMapAction(MR._ByValue_ChangeFacesColorMapAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeFacesColorMapAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeFacesColorMapAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's face colors before making any changes in them
        /// Generated from constructor `MR::ChangeFacesColorMapAction::ChangeFacesColorMapAction`.
        public unsafe Const_ChangeFacesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeFacesColorMapAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's face colors and immediate set new value
        /// Generated from constructor `MR::ChangeFacesColorMapAction::ChangeFacesColorMapAction`.
        public unsafe Const_ChangeFacesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.FaceColors> newColorMap) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.FaceColors._Underlying *newColorMap);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeFacesColorMapAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newColorMap.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeFacesColorMapAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeFacesColorMapAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeFacesColorMapAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeFacesColorMapAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectMeshHolder obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeFacesColorMapAction_setObjectDirty(MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            __MR_ChangeFacesColorMapAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeFacesColorMapAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeFacesColorMapAction_heapBytes(_Underlying *_this);
            return __MR_ChangeFacesColorMapAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for faces color map change
    /// Generated from class `MR::ChangeFacesColorMapAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeFacesColorMapAction : Const_ChangeFacesColorMapAction
    {
        internal unsafe ChangeFacesColorMapAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeFacesColorMapAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeFacesColorMapAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeFacesColorMapAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeFacesColorMapAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeFacesColorMapAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeFacesColorMapAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeFacesColorMapAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeFacesColorMapAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeFacesColorMapAction::ChangeFacesColorMapAction`.
        public unsafe ChangeFacesColorMapAction(MR._ByValue_ChangeFacesColorMapAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeFacesColorMapAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeFacesColorMapAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's face colors before making any changes in them
        /// Generated from constructor `MR::ChangeFacesColorMapAction::ChangeFacesColorMapAction`.
        public unsafe ChangeFacesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeFacesColorMapAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's face colors and immediate set new value
        /// Generated from constructor `MR::ChangeFacesColorMapAction::ChangeFacesColorMapAction`.
        public unsafe ChangeFacesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectMeshHolder obj, MR.Misc._Moved<MR.FaceColors> newColorMap) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectMeshHolder._UnderlyingShared *obj, MR.FaceColors._Underlying *newColorMap);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeFacesColorMapAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newColorMap.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeFacesColorMapAction::operator=`.
        public unsafe MR.ChangeFacesColorMapAction Assign(MR._ByValue_ChangeFacesColorMapAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeFacesColorMapAction._Underlying *__MR_ChangeFacesColorMapAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeFacesColorMapAction._Underlying *_other);
            return new(__MR_ChangeFacesColorMapAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeFacesColorMapAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeFacesColorMapAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeFacesColorMapAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeFacesColorMapAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeFacesColorMapAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeFacesColorMapAction`/`Const_ChangeFacesColorMapAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeFacesColorMapAction
    {
        internal readonly Const_ChangeFacesColorMapAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeFacesColorMapAction(Const_ChangeFacesColorMapAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeFacesColorMapAction(Const_ChangeFacesColorMapAction arg) {return new(arg);}
        public _ByValue_ChangeFacesColorMapAction(MR.Misc._Moved<ChangeFacesColorMapAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeFacesColorMapAction(MR.Misc._Moved<ChangeFacesColorMapAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeFacesColorMapAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeFacesColorMapAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeFacesColorMapAction`/`Const_ChangeFacesColorMapAction` directly.
    public class _InOptMut_ChangeFacesColorMapAction
    {
        public ChangeFacesColorMapAction? Opt;

        public _InOptMut_ChangeFacesColorMapAction() {}
        public _InOptMut_ChangeFacesColorMapAction(ChangeFacesColorMapAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeFacesColorMapAction(ChangeFacesColorMapAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeFacesColorMapAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeFacesColorMapAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeFacesColorMapAction`/`Const_ChangeFacesColorMapAction` to pass it to the function.
    public class _InOptConst_ChangeFacesColorMapAction
    {
        public Const_ChangeFacesColorMapAction? Opt;

        public _InOptConst_ChangeFacesColorMapAction() {}
        public _InOptConst_ChangeFacesColorMapAction(Const_ChangeFacesColorMapAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeFacesColorMapAction(Const_ChangeFacesColorMapAction value) {return new(value);}
    }

    /// History action for lines color map change
    /// Generated from class `MR::ChangeLinesColorMapAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeLinesColorMapAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_UseCount();
                return __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeLinesColorMapAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeLinesColorMapAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeLinesColorMapAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeLinesColorMapAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeLinesColorMapAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeLinesColorMapAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeLinesColorMapAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeLinesColorMapAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeLinesColorMapAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeLinesColorMapAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeLinesColorMapAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeLinesColorMapAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeLinesColorMapAction::ChangeLinesColorMapAction`.
        public unsafe Const_ChangeLinesColorMapAction(MR._ByValue_ChangeLinesColorMapAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeLinesColorMapAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeLinesColorMapAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's line colors before making any changes in them
        /// Generated from constructor `MR::ChangeLinesColorMapAction::ChangeLinesColorMapAction`.
        public unsafe Const_ChangeLinesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectLinesHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectLinesHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeLinesColorMapAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's lines colors and immediate set new value
        /// Generated from constructor `MR::ChangeLinesColorMapAction::ChangeLinesColorMapAction`.
        public unsafe Const_ChangeLinesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectLinesHolder obj, MR.Misc._Moved<MR.UndirectedEdgeColors> newColorMap) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectLinesHolder._UnderlyingShared *obj, MR.UndirectedEdgeColors._Underlying *newColorMap);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeLinesColorMapAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newColorMap.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeLinesColorMapAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeLinesColorMapAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeLinesColorMapAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeLinesColorMapAction::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_ObjectLinesHolder obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeLinesColorMapAction_setObjectDirty(MR.Const_ObjectLinesHolder._UnderlyingShared *obj);
            __MR_ChangeLinesColorMapAction_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeLinesColorMapAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeLinesColorMapAction_heapBytes(_Underlying *_this);
            return __MR_ChangeLinesColorMapAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for lines color map change
    /// Generated from class `MR::ChangeLinesColorMapAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeLinesColorMapAction : Const_ChangeLinesColorMapAction
    {
        internal unsafe ChangeLinesColorMapAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeLinesColorMapAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeLinesColorMapAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeLinesColorMapAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeLinesColorMapAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeLinesColorMapAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeLinesColorMapAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeLinesColorMapAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeLinesColorMapAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeLinesColorMapAction::ChangeLinesColorMapAction`.
        public unsafe ChangeLinesColorMapAction(MR._ByValue_ChangeLinesColorMapAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeLinesColorMapAction._Underlying *_other);
            _LateMakeShared(__MR_ChangeLinesColorMapAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's line colors before making any changes in them
        /// Generated from constructor `MR::ChangeLinesColorMapAction::ChangeLinesColorMapAction`.
        public unsafe ChangeLinesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectLinesHolder obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_Construct_2(byte *name, byte *name_end, MR.Const_ObjectLinesHolder._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeLinesColorMapAction_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's lines colors and immediate set new value
        /// Generated from constructor `MR::ChangeLinesColorMapAction::ChangeLinesColorMapAction`.
        public unsafe ChangeLinesColorMapAction(ReadOnlySpan<char> name, MR.Const_ObjectLinesHolder obj, MR.Misc._Moved<MR.UndirectedEdgeColors> newColorMap) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_Construct_3(byte *name, byte *name_end, MR.Const_ObjectLinesHolder._UnderlyingShared *obj, MR.UndirectedEdgeColors._Underlying *newColorMap);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeLinesColorMapAction_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newColorMap.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::ChangeLinesColorMapAction::operator=`.
        public unsafe MR.ChangeLinesColorMapAction Assign(MR._ByValue_ChangeLinesColorMapAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeLinesColorMapAction._Underlying *__MR_ChangeLinesColorMapAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeLinesColorMapAction._Underlying *_other);
            return new(__MR_ChangeLinesColorMapAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeLinesColorMapAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeLinesColorMapAction_action", ExactSpelling = true)]
            extern static void __MR_ChangeLinesColorMapAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeLinesColorMapAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeLinesColorMapAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeLinesColorMapAction`/`Const_ChangeLinesColorMapAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeLinesColorMapAction
    {
        internal readonly Const_ChangeLinesColorMapAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeLinesColorMapAction(Const_ChangeLinesColorMapAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeLinesColorMapAction(Const_ChangeLinesColorMapAction arg) {return new(arg);}
        public _ByValue_ChangeLinesColorMapAction(MR.Misc._Moved<ChangeLinesColorMapAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeLinesColorMapAction(MR.Misc._Moved<ChangeLinesColorMapAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeLinesColorMapAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeLinesColorMapAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeLinesColorMapAction`/`Const_ChangeLinesColorMapAction` directly.
    public class _InOptMut_ChangeLinesColorMapAction
    {
        public ChangeLinesColorMapAction? Opt;

        public _InOptMut_ChangeLinesColorMapAction() {}
        public _InOptMut_ChangeLinesColorMapAction(ChangeLinesColorMapAction value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeLinesColorMapAction(ChangeLinesColorMapAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeLinesColorMapAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeLinesColorMapAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeLinesColorMapAction`/`Const_ChangeLinesColorMapAction` to pass it to the function.
    public class _InOptConst_ChangeLinesColorMapAction
    {
        public Const_ChangeLinesColorMapAction? Opt;

        public _InOptConst_ChangeLinesColorMapAction() {}
        public _InOptConst_ChangeLinesColorMapAction(Const_ChangeLinesColorMapAction value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeLinesColorMapAction(Const_ChangeLinesColorMapAction value) {return new(value);}
    }
}
