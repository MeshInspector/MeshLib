public static partial class MR
{
    /// argument of this type indicates that the object is already in new state, and the following argument is old state
    /// Generated from class `MR::CmpOld`.
    /// This is the const half of the class.
    public class Const_CmpOld : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CmpOld(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CmpOld_Destroy", ExactSpelling = true)]
            extern static void __MR_CmpOld_Destroy(_Underlying *_this);
            __MR_CmpOld_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CmpOld() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CmpOld() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CmpOld_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CmpOld._Underlying *__MR_CmpOld_DefaultConstruct();
            _UnderlyingPtr = __MR_CmpOld_DefaultConstruct();
        }

        /// Generated from constructor `MR::CmpOld::CmpOld`.
        public unsafe Const_CmpOld(MR.Const_CmpOld _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CmpOld_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CmpOld._Underlying *__MR_CmpOld_ConstructFromAnother(MR.CmpOld._Underlying *_other);
            _UnderlyingPtr = __MR_CmpOld_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// argument of this type indicates that the object is already in new state, and the following argument is old state
    /// Generated from class `MR::CmpOld`.
    /// This is the non-const half of the class.
    public class CmpOld : Const_CmpOld
    {
        internal unsafe CmpOld(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe CmpOld() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CmpOld_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CmpOld._Underlying *__MR_CmpOld_DefaultConstruct();
            _UnderlyingPtr = __MR_CmpOld_DefaultConstruct();
        }

        /// Generated from constructor `MR::CmpOld::CmpOld`.
        public unsafe CmpOld(MR.Const_CmpOld _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CmpOld_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CmpOld._Underlying *__MR_CmpOld_ConstructFromAnother(MR.CmpOld._Underlying *_other);
            _UnderlyingPtr = __MR_CmpOld_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CmpOld::operator=`.
        public unsafe MR.CmpOld Assign(MR.Const_CmpOld _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CmpOld_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CmpOld._Underlying *__MR_CmpOld_AssignFromAnother(_Underlying *_this, MR.CmpOld._Underlying *_other);
            return new(__MR_CmpOld_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CmpOld` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CmpOld`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CmpOld`/`Const_CmpOld` directly.
    public class _InOptMut_CmpOld
    {
        public CmpOld? Opt;

        public _InOptMut_CmpOld() {}
        public _InOptMut_CmpOld(CmpOld value) {Opt = value;}
        public static implicit operator _InOptMut_CmpOld(CmpOld value) {return new(value);}
    }

    /// This is used for optional parameters of class `CmpOld` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CmpOld`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CmpOld`/`Const_CmpOld` to pass it to the function.
    public class _InOptConst_CmpOld
    {
        public Const_CmpOld? Opt;

        public _InOptConst_CmpOld() {}
        public _InOptConst_CmpOld(Const_CmpOld value) {Opt = value;}
        public static implicit operator _InOptConst_CmpOld(Const_CmpOld value) {return new(value);}
    }

    /// argument of this type indicates that the object is in old state, and the following argument is new state to be set
    /// Generated from class `MR::SetNew`.
    /// This is the const half of the class.
    public class Const_SetNew : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SetNew(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetNew_Destroy", ExactSpelling = true)]
            extern static void __MR_SetNew_Destroy(_Underlying *_this);
            __MR_SetNew_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SetNew() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SetNew() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetNew_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SetNew._Underlying *__MR_SetNew_DefaultConstruct();
            _UnderlyingPtr = __MR_SetNew_DefaultConstruct();
        }

        /// Generated from constructor `MR::SetNew::SetNew`.
        public unsafe Const_SetNew(MR.Const_SetNew _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetNew_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SetNew._Underlying *__MR_SetNew_ConstructFromAnother(MR.SetNew._Underlying *_other);
            _UnderlyingPtr = __MR_SetNew_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// argument of this type indicates that the object is in old state, and the following argument is new state to be set
    /// Generated from class `MR::SetNew`.
    /// This is the non-const half of the class.
    public class SetNew : Const_SetNew
    {
        internal unsafe SetNew(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe SetNew() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetNew_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SetNew._Underlying *__MR_SetNew_DefaultConstruct();
            _UnderlyingPtr = __MR_SetNew_DefaultConstruct();
        }

        /// Generated from constructor `MR::SetNew::SetNew`.
        public unsafe SetNew(MR.Const_SetNew _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetNew_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SetNew._Underlying *__MR_SetNew_ConstructFromAnother(MR.SetNew._Underlying *_other);
            _UnderlyingPtr = __MR_SetNew_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SetNew::operator=`.
        public unsafe MR.SetNew Assign(MR.Const_SetNew _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SetNew_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SetNew._Underlying *__MR_SetNew_AssignFromAnother(_Underlying *_this, MR.SetNew._Underlying *_other);
            return new(__MR_SetNew_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SetNew` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SetNew`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SetNew`/`Const_SetNew` directly.
    public class _InOptMut_SetNew
    {
        public SetNew? Opt;

        public _InOptMut_SetNew() {}
        public _InOptMut_SetNew(SetNew value) {Opt = value;}
        public static implicit operator _InOptMut_SetNew(SetNew value) {return new(value);}
    }

    /// This is used for optional parameters of class `SetNew` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SetNew`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SetNew`/`Const_SetNew` to pass it to the function.
    public class _InOptConst_SetNew
    {
        public Const_SetNew? Opt;

        public _InOptConst_SetNew() {}
        public _InOptConst_SetNew(Const_SetNew value) {Opt = value;}
        public static implicit operator _InOptConst_SetNew(Const_SetNew value) {return new(value);}
    }

    /// Undo action for efficiently storage of partial change in mesh (e.g. a modification of small region)
    /// Generated from class `MR::PartialChangeMeshAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_PartialChangeMeshAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PartialChangeMeshAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PartialChangeMeshAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PartialChangeMeshAction_UseCount();
                return __MR_std_shared_ptr_MR_PartialChangeMeshAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PartialChangeMeshAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PartialChangeMeshAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PartialChangeMeshAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PartialChangeMeshAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PartialChangeMeshAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PartialChangeMeshAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PartialChangeMeshAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_PartialChangeMeshAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_PartialChangeMeshAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PartialChangeMeshAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshAction::PartialChangeMeshAction`.
        public unsafe Const_PartialChangeMeshAction(MR._ByValue_PartialChangeMeshAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor after the object already contains new mesh,
        /// and old mesh is passed to remember the difference for future undoing
        /// Generated from constructor `MR::PartialChangeMeshAction::PartialChangeMeshAction`.
        public unsafe Const_PartialChangeMeshAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_CmpOld _3, MR.Const_Mesh oldMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_Construct_MR_CmpOld", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_Construct_MR_CmpOld(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.CmpOld._Underlying *_3, MR.Const_Mesh._Underlying *oldMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshAction_Construct_MR_CmpOld(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, oldMesh._UnderlyingPtr));
            }
        }

        /// use this constructor to set new object's mesh and remember its difference from existed mesh for future undoing
        /// Generated from constructor `MR::PartialChangeMeshAction::PartialChangeMeshAction`.
        public unsafe Const_PartialChangeMeshAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_SetNew _3, MR.Misc._Moved<MR.Mesh> newMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_Construct_MR_SetNew", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_Construct_MR_SetNew(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.SetNew._Underlying *_3, MR.Mesh._UnderlyingShared *newMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshAction_Construct_MR_SetNew(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, newMesh.Value._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PartialChangeMeshAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PartialChangeMeshAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PartialChangeMeshAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PartialChangeMeshAction_heapBytes(_Underlying *_this);
            return __MR_PartialChangeMeshAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for efficiently storage of partial change in mesh (e.g. a modification of small region)
    /// Generated from class `MR::PartialChangeMeshAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class PartialChangeMeshAction : Const_PartialChangeMeshAction
    {
        internal unsafe PartialChangeMeshAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PartialChangeMeshAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(PartialChangeMeshAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_PartialChangeMeshAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PartialChangeMeshAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshAction::PartialChangeMeshAction`.
        public unsafe PartialChangeMeshAction(MR._ByValue_PartialChangeMeshAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor after the object already contains new mesh,
        /// and old mesh is passed to remember the difference for future undoing
        /// Generated from constructor `MR::PartialChangeMeshAction::PartialChangeMeshAction`.
        public unsafe PartialChangeMeshAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_CmpOld _3, MR.Const_Mesh oldMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_Construct_MR_CmpOld", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_Construct_MR_CmpOld(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.CmpOld._Underlying *_3, MR.Const_Mesh._Underlying *oldMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshAction_Construct_MR_CmpOld(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, oldMesh._UnderlyingPtr));
            }
        }

        /// use this constructor to set new object's mesh and remember its difference from existed mesh for future undoing
        /// Generated from constructor `MR::PartialChangeMeshAction::PartialChangeMeshAction`.
        public unsafe PartialChangeMeshAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_SetNew _3, MR.Misc._Moved<MR.Mesh> newMesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_Construct_MR_SetNew", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_Construct_MR_SetNew(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.SetNew._Underlying *_3, MR.Mesh._UnderlyingShared *newMesh);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshAction_Construct_MR_SetNew(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, newMesh.Value._UnderlyingSharedPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshAction::operator=`.
        public unsafe MR.PartialChangeMeshAction Assign(MR._ByValue_PartialChangeMeshAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshAction._Underlying *__MR_PartialChangeMeshAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshAction._Underlying *_other);
            return new(__MR_PartialChangeMeshAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::PartialChangeMeshAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshAction_action", ExactSpelling = true)]
            extern static void __MR_PartialChangeMeshAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_PartialChangeMeshAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PartialChangeMeshAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PartialChangeMeshAction`/`Const_PartialChangeMeshAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PartialChangeMeshAction
    {
        internal readonly Const_PartialChangeMeshAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PartialChangeMeshAction(Const_PartialChangeMeshAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PartialChangeMeshAction(Const_PartialChangeMeshAction arg) {return new(arg);}
        public _ByValue_PartialChangeMeshAction(MR.Misc._Moved<PartialChangeMeshAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PartialChangeMeshAction(MR.Misc._Moved<PartialChangeMeshAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialChangeMeshAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshAction`/`Const_PartialChangeMeshAction` directly.
    public class _InOptMut_PartialChangeMeshAction
    {
        public PartialChangeMeshAction? Opt;

        public _InOptMut_PartialChangeMeshAction() {}
        public _InOptMut_PartialChangeMeshAction(PartialChangeMeshAction value) {Opt = value;}
        public static implicit operator _InOptMut_PartialChangeMeshAction(PartialChangeMeshAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialChangeMeshAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshAction`/`Const_PartialChangeMeshAction` to pass it to the function.
    public class _InOptConst_PartialChangeMeshAction
    {
        public Const_PartialChangeMeshAction? Opt;

        public _InOptConst_PartialChangeMeshAction() {}
        public _InOptConst_PartialChangeMeshAction(Const_PartialChangeMeshAction value) {Opt = value;}
        public static implicit operator _InOptConst_PartialChangeMeshAction(Const_PartialChangeMeshAction value) {return new(value);}
    }

    /// Undo action for efficiently storage of partial change in mesh points (e.g. a modification of small region)
    /// Generated from class `MR::PartialChangeMeshPointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_PartialChangeMeshPointsAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_UseCount();
                return __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PartialChangeMeshPointsAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PartialChangeMeshPointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PartialChangeMeshPointsAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PartialChangeMeshPointsAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PartialChangeMeshPointsAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_PartialChangeMeshPointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_PartialChangeMeshPointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshPointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PartialChangeMeshPointsAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshPointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshPointsAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshPointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshPointsAction::PartialChangeMeshPointsAction`.
        public unsafe Const_PartialChangeMeshPointsAction(MR._ByValue_PartialChangeMeshPointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshPointsAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshPointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor after the object already contains new points,
        /// and old points are passed to remember the difference for future undoing
        /// Generated from constructor `MR::PartialChangeMeshPointsAction::PartialChangeMeshPointsAction`.
        public unsafe Const_PartialChangeMeshPointsAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_CmpOld _3, MR.Const_VertCoords oldPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_Construct_MR_CmpOld", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_Construct_MR_CmpOld(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.CmpOld._Underlying *_3, MR.Const_VertCoords._Underlying *oldPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshPointsAction_Construct_MR_CmpOld(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, oldPoints._UnderlyingPtr));
            }
        }

        /// use this constructor to set new object's points and remember its difference from existed points for future undoing
        /// Generated from constructor `MR::PartialChangeMeshPointsAction::PartialChangeMeshPointsAction`.
        public unsafe Const_PartialChangeMeshPointsAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_SetNew _3, MR.Misc._Moved<MR.VertCoords> newPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_Construct_MR_SetNew", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_Construct_MR_SetNew(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.SetNew._Underlying *_3, MR.VertCoords._Underlying *newPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshPointsAction_Construct_MR_SetNew(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, newPoints.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshPointsAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PartialChangeMeshPointsAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PartialChangeMeshPointsAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PartialChangeMeshPointsAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PartialChangeMeshPointsAction_heapBytes(_Underlying *_this);
            return __MR_PartialChangeMeshPointsAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for efficiently storage of partial change in mesh points (e.g. a modification of small region)
    /// Generated from class `MR::PartialChangeMeshPointsAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class PartialChangeMeshPointsAction : Const_PartialChangeMeshPointsAction
    {
        internal unsafe PartialChangeMeshPointsAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PartialChangeMeshPointsAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(PartialChangeMeshPointsAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_PartialChangeMeshPointsAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshPointsAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PartialChangeMeshPointsAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshPointsAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshPointsAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshPointsAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshPointsAction::PartialChangeMeshPointsAction`.
        public unsafe PartialChangeMeshPointsAction(MR._ByValue_PartialChangeMeshPointsAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshPointsAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshPointsAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor after the object already contains new points,
        /// and old points are passed to remember the difference for future undoing
        /// Generated from constructor `MR::PartialChangeMeshPointsAction::PartialChangeMeshPointsAction`.
        public unsafe PartialChangeMeshPointsAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_CmpOld _3, MR.Const_VertCoords oldPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_Construct_MR_CmpOld", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_Construct_MR_CmpOld(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.CmpOld._Underlying *_3, MR.Const_VertCoords._Underlying *oldPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshPointsAction_Construct_MR_CmpOld(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, oldPoints._UnderlyingPtr));
            }
        }

        /// use this constructor to set new object's points and remember its difference from existed points for future undoing
        /// Generated from constructor `MR::PartialChangeMeshPointsAction::PartialChangeMeshPointsAction`.
        public unsafe PartialChangeMeshPointsAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_SetNew _3, MR.Misc._Moved<MR.VertCoords> newPoints) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_Construct_MR_SetNew", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_Construct_MR_SetNew(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.SetNew._Underlying *_3, MR.VertCoords._Underlying *newPoints);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshPointsAction_Construct_MR_SetNew(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, newPoints.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshPointsAction::operator=`.
        public unsafe MR.PartialChangeMeshPointsAction Assign(MR._ByValue_PartialChangeMeshPointsAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshPointsAction._Underlying *__MR_PartialChangeMeshPointsAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshPointsAction._Underlying *_other);
            return new(__MR_PartialChangeMeshPointsAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::PartialChangeMeshPointsAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshPointsAction_action", ExactSpelling = true)]
            extern static void __MR_PartialChangeMeshPointsAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_PartialChangeMeshPointsAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PartialChangeMeshPointsAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PartialChangeMeshPointsAction`/`Const_PartialChangeMeshPointsAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PartialChangeMeshPointsAction
    {
        internal readonly Const_PartialChangeMeshPointsAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PartialChangeMeshPointsAction(Const_PartialChangeMeshPointsAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PartialChangeMeshPointsAction(Const_PartialChangeMeshPointsAction arg) {return new(arg);}
        public _ByValue_PartialChangeMeshPointsAction(MR.Misc._Moved<PartialChangeMeshPointsAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PartialChangeMeshPointsAction(MR.Misc._Moved<PartialChangeMeshPointsAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshPointsAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialChangeMeshPointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshPointsAction`/`Const_PartialChangeMeshPointsAction` directly.
    public class _InOptMut_PartialChangeMeshPointsAction
    {
        public PartialChangeMeshPointsAction? Opt;

        public _InOptMut_PartialChangeMeshPointsAction() {}
        public _InOptMut_PartialChangeMeshPointsAction(PartialChangeMeshPointsAction value) {Opt = value;}
        public static implicit operator _InOptMut_PartialChangeMeshPointsAction(PartialChangeMeshPointsAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshPointsAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialChangeMeshPointsAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshPointsAction`/`Const_PartialChangeMeshPointsAction` to pass it to the function.
    public class _InOptConst_PartialChangeMeshPointsAction
    {
        public Const_PartialChangeMeshPointsAction? Opt;

        public _InOptConst_PartialChangeMeshPointsAction() {}
        public _InOptConst_PartialChangeMeshPointsAction(Const_PartialChangeMeshPointsAction value) {Opt = value;}
        public static implicit operator _InOptConst_PartialChangeMeshPointsAction(Const_PartialChangeMeshPointsAction value) {return new(value);}
    }

    /// Undo action for efficiently storage of partial change in mesh topology (e.g. a modification of small region)
    /// Generated from class `MR::PartialChangeMeshTopologyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the const half of the class.
    public class Const_PartialChangeMeshTopologyAction : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_UseCount();
                return __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PartialChangeMeshTopologyAction(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PartialChangeMeshTopologyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PartialChangeMeshTopologyAction _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PartialChangeMeshTopologyAction_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PartialChangeMeshTopologyAction() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_HistoryAction(Const_PartialChangeMeshTopologyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.Const_HistoryAction._Underlying *__MR_PartialChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PartialChangeMeshTopologyAction?(MR.Const_HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshTopologyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshTopologyAction(MR.Const_HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshTopologyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshTopologyAction::PartialChangeMeshTopologyAction`.
        public unsafe Const_PartialChangeMeshTopologyAction(MR._ByValue_PartialChangeMeshTopologyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshTopologyAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshTopologyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor after the object already contains new topology,
        /// and old topology is passed to remember the difference for future undoing
        /// Generated from constructor `MR::PartialChangeMeshTopologyAction::PartialChangeMeshTopologyAction`.
        public unsafe Const_PartialChangeMeshTopologyAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_CmpOld _3, MR.Const_MeshTopology oldTopology) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_Construct_MR_CmpOld", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_Construct_MR_CmpOld(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.CmpOld._Underlying *_3, MR.Const_MeshTopology._Underlying *oldTopology);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshTopologyAction_Construct_MR_CmpOld(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, oldTopology._UnderlyingPtr));
            }
        }

        /// use this constructor to set new object's topology and remember its difference from existed topology for future undoing
        /// Generated from constructor `MR::PartialChangeMeshTopologyAction::PartialChangeMeshTopologyAction`.
        public unsafe Const_PartialChangeMeshTopologyAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_SetNew _3, MR.Misc._Moved<MR.MeshTopology> newTopology) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_Construct_MR_SetNew", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_Construct_MR_SetNew(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.SetNew._Underlying *_3, MR.MeshTopology._Underlying *newTopology);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshTopologyAction_Construct_MR_SetNew(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, newTopology.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshTopologyAction::name`.
        public unsafe MR.Misc._Moved<MR.Std.String> Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_PartialChangeMeshTopologyAction_name(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_PartialChangeMeshTopologyAction_name(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PartialChangeMeshTopologyAction::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PartialChangeMeshTopologyAction_heapBytes(_Underlying *_this);
            return __MR_PartialChangeMeshTopologyAction_heapBytes(_UnderlyingPtr);
        }
    }

    /// Undo action for efficiently storage of partial change in mesh topology (e.g. a modification of small region)
    /// Generated from class `MR::PartialChangeMeshTopologyAction`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::HistoryAction`
    /// This is the non-const half of the class.
    public class PartialChangeMeshTopologyAction : Const_PartialChangeMeshTopologyAction
    {
        internal unsafe PartialChangeMeshTopologyAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PartialChangeMeshTopologyAction(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.HistoryAction(PartialChangeMeshTopologyAction self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_UpcastTo_MR_HistoryAction", ExactSpelling = true)]
            extern static MR.HistoryAction._Underlying *__MR_PartialChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(_Underlying *_this);
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PartialChangeMeshTopologyAction_UpcastTo_MR_HistoryAction(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PartialChangeMeshTopologyAction?(MR.HistoryAction parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshTopologyAction", ExactSpelling = true)]
            extern static _Underlying *__MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshTopologyAction(MR.HistoryAction._Underlying *_this);
            var ptr = __MR_HistoryAction_DynamicDowncastTo_MR_PartialChangeMeshTopologyAction(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.HistoryAction._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::PartialChangeMeshTopologyAction::PartialChangeMeshTopologyAction`.
        public unsafe PartialChangeMeshTopologyAction(MR._ByValue_PartialChangeMeshTopologyAction _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshTopologyAction._Underlying *_other);
            _LateMakeShared(__MR_PartialChangeMeshTopologyAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// use this constructor after the object already contains new topology,
        /// and old topology is passed to remember the difference for future undoing
        /// Generated from constructor `MR::PartialChangeMeshTopologyAction::PartialChangeMeshTopologyAction`.
        public unsafe PartialChangeMeshTopologyAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_CmpOld _3, MR.Const_MeshTopology oldTopology) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_Construct_MR_CmpOld", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_Construct_MR_CmpOld(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.CmpOld._Underlying *_3, MR.Const_MeshTopology._Underlying *oldTopology);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshTopologyAction_Construct_MR_CmpOld(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, oldTopology._UnderlyingPtr));
            }
        }

        /// use this constructor to set new object's topology and remember its difference from existed topology for future undoing
        /// Generated from constructor `MR::PartialChangeMeshTopologyAction::PartialChangeMeshTopologyAction`.
        public unsafe PartialChangeMeshTopologyAction(ReadOnlySpan<char> name, MR._ByValue_ObjectMesh obj, MR.Const_SetNew _3, MR.Misc._Moved<MR.MeshTopology> newTopology) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_Construct_MR_SetNew", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_Construct_MR_SetNew(byte *name, byte *name_end, MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, MR.SetNew._Underlying *_3, MR.MeshTopology._Underlying *newTopology);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                _LateMakeShared(__MR_PartialChangeMeshTopologyAction_Construct_MR_SetNew(__ptr_name, __ptr_name + __len_name, obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, _3._UnderlyingPtr, newTopology.Value._UnderlyingPtr));
            }
        }

        /// Generated from method `MR::PartialChangeMeshTopologyAction::operator=`.
        public unsafe MR.PartialChangeMeshTopologyAction Assign(MR._ByValue_PartialChangeMeshTopologyAction _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PartialChangeMeshTopologyAction._Underlying *__MR_PartialChangeMeshTopologyAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PartialChangeMeshTopologyAction._Underlying *_other);
            return new(__MR_PartialChangeMeshTopologyAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::PartialChangeMeshTopologyAction::action`.
        public unsafe void Action(MR.HistoryAction.Type _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartialChangeMeshTopologyAction_action", ExactSpelling = true)]
            extern static void __MR_PartialChangeMeshTopologyAction_action(_Underlying *_this, MR.HistoryAction.Type _1);
            __MR_PartialChangeMeshTopologyAction_action(_UnderlyingPtr, _1);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PartialChangeMeshTopologyAction` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PartialChangeMeshTopologyAction`/`Const_PartialChangeMeshTopologyAction` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PartialChangeMeshTopologyAction
    {
        internal readonly Const_PartialChangeMeshTopologyAction? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PartialChangeMeshTopologyAction(Const_PartialChangeMeshTopologyAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PartialChangeMeshTopologyAction(Const_PartialChangeMeshTopologyAction arg) {return new(arg);}
        public _ByValue_PartialChangeMeshTopologyAction(MR.Misc._Moved<PartialChangeMeshTopologyAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PartialChangeMeshTopologyAction(MR.Misc._Moved<PartialChangeMeshTopologyAction> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshTopologyAction` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialChangeMeshTopologyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshTopologyAction`/`Const_PartialChangeMeshTopologyAction` directly.
    public class _InOptMut_PartialChangeMeshTopologyAction
    {
        public PartialChangeMeshTopologyAction? Opt;

        public _InOptMut_PartialChangeMeshTopologyAction() {}
        public _InOptMut_PartialChangeMeshTopologyAction(PartialChangeMeshTopologyAction value) {Opt = value;}
        public static implicit operator _InOptMut_PartialChangeMeshTopologyAction(PartialChangeMeshTopologyAction value) {return new(value);}
    }

    /// This is used for optional parameters of class `PartialChangeMeshTopologyAction` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialChangeMeshTopologyAction`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartialChangeMeshTopologyAction`/`Const_PartialChangeMeshTopologyAction` to pass it to the function.
    public class _InOptConst_PartialChangeMeshTopologyAction
    {
        public Const_PartialChangeMeshTopologyAction? Opt;

        public _InOptConst_PartialChangeMeshTopologyAction() {}
        public _InOptConst_PartialChangeMeshTopologyAction(Const_PartialChangeMeshTopologyAction value) {Opt = value;}
        public static implicit operator _InOptConst_PartialChangeMeshTopologyAction(Const_PartialChangeMeshTopologyAction value) {return new(value);}
    }
}
