public static partial class MR
{
    /// History action for ColoringType change
    /// Generated from class `MR::ChangeColoringType`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_ChangeColoringType : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ChangeColoringType_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ChangeColoringType_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ChangeColoringType_UseCount();
                return __MR_std_shared_ptr_MR_ChangeColoringType_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeColoringType_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ChangeColoringType_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ChangeColoringType(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeColoringType_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeColoringType_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeColoringType_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeColoringType_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ChangeColoringType(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ChangeColoringType _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeColoringType_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ChangeColoringType_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ChangeColoringType_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ChangeColoringType_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ChangeColoringType_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ChangeColoringType_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ChangeColoringType_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChangeColoringType() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_ChangeColoringType self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_ChangeColoringType_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeColoringType_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ChangeColoringType?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeColoringType", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeColoringType(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeColoringType(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeColoringType::ChangeColoringType`.
        public unsafe Const_ChangeColoringType(MR._ByValue_ChangeColoringType _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeColoringType._Underlying *_other);
            _LateMakeShared(__MR_ChangeColoringType_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's coloring type before making any changes in it
        /// Generated from constructor `MR::ChangeColoringType::ChangeColoringType`.
        public unsafe Const_ChangeColoringType(ReadOnlySpan<char> name, MR.Const_VisualObject obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_Construct_2(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeColoringType_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's coloring type and immediate set new value
        /// Generated from constructor `MR::ChangeColoringType::ChangeColoringType`.
        public unsafe Const_ChangeColoringType(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.ColoringType newType) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_Construct_3(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.ColoringType newType);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeColoringType_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newType));
            }
        }

        /// Generated from method `MR::ChangeColoringType::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ChangeColoringType_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ChangeColoringType_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ChangeColoringType::setObjectDirty`.
        public static unsafe void SetObjectDirty(MR.Const_VisualObject obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_setObjectDirty", ExactSpelling = true)]
            extern static void __MR_ChangeColoringType_setObjectDirty(MR.Const_VisualObject._UnderlyingShared *obj);
            __MR_ChangeColoringType_setObjectDirty(obj._UnderlyingSharedPtr);
        }

        /// Generated from method `MR::ChangeColoringType::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ChangeColoringType_heapBytes(_Underlying *_this);
            return __MR_ChangeColoringType_heapBytes(_UnderlyingPtr);
        }
    }

    /// History action for ColoringType change
    /// Generated from class `MR::ChangeColoringType`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class ChangeColoringType : Const_ChangeColoringType
    {
        internal unsafe ChangeColoringType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ChangeColoringType(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(ChangeColoringType self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_ChangeColoringType_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ChangeColoringType_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ChangeColoringType?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_ChangeColoringType", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_ChangeColoringType(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_ChangeColoringType(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::ChangeColoringType::ChangeColoringType`.
        public unsafe ChangeColoringType(MR._ByValue_ChangeColoringType _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ChangeColoringType._Underlying *_other);
            _LateMakeShared(__MR_ChangeColoringType_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor to remember object's coloring type before making any changes in it
        /// Generated from constructor `MR::ChangeColoringType::ChangeColoringType`.
        public unsafe ChangeColoringType(ReadOnlySpan<char> name, MR.Const_VisualObject obj) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_Construct_2", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_Construct_2(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeColoringType_Construct_2(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr));
            }
        }

        /// use this constructor to remember object's coloring type and immediate set new value
        /// Generated from constructor `MR::ChangeColoringType::ChangeColoringType`.
        public unsafe ChangeColoringType(ReadOnlySpan<char> name, MR.Const_VisualObject obj, MR.ColoringType newType) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_Construct_3", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_Construct_3(byte *name, byte *name_end, MR.Const_VisualObject._UnderlyingShared *obj, MR.ColoringType newType);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_ChangeColoringType_Construct_3(__ptr_name, __ptr_name + __len_name, obj._UnderlyingSharedPtr, newType));
            }
        }

        /// Generated from method `MR::ChangeColoringType::operator=`.
        public unsafe MR.ChangeColoringType Assign(MR._ByValue_ChangeColoringType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChangeColoringType._Underlying *__MR_ChangeColoringType_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ChangeColoringType._Underlying *_other);
            return new(__MR_ChangeColoringType_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ChangeColoringType::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChangeColoringType_action", ExactSpelling = true)]
            extern static void __MR_ChangeColoringType_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_ChangeColoringType_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ChangeColoringType` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ChangeColoringType`/`Const_ChangeColoringType` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ChangeColoringType
    {
        internal readonly Const_ChangeColoringType? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ChangeColoringType(Const_ChangeColoringType new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ChangeColoringType(Const_ChangeColoringType arg) {return new(arg);}
        public _ByValue_ChangeColoringType(MR.Misc._Moved<ChangeColoringType> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ChangeColoringType(MR.Misc._Moved<ChangeColoringType> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ChangeColoringType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChangeColoringType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeColoringType`/`Const_ChangeColoringType` directly.
    public class _InOptMut_ChangeColoringType
    {
        public ChangeColoringType? Opt;

        public _InOptMut_ChangeColoringType() {}
        public _InOptMut_ChangeColoringType(ChangeColoringType value) {Opt = value;}
        public static implicit operator _InOptMut_ChangeColoringType(ChangeColoringType value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChangeColoringType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChangeColoringType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChangeColoringType`/`Const_ChangeColoringType` to pass it to the function.
    public class _InOptConst_ChangeColoringType
    {
        public Const_ChangeColoringType? Opt;

        public _InOptConst_ChangeColoringType() {}
        public _InOptConst_ChangeColoringType(Const_ChangeColoringType value) {Opt = value;}
        public static implicit operator _InOptConst_ChangeColoringType(Const_ChangeColoringType value) {return new(value);}
    }
}
