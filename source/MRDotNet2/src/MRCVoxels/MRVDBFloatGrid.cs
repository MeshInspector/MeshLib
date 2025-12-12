public static partial class MR
{
    /// this class just hides very complex type of typedef openvdb::FloatGrid
    /// Generated from class `MR::OpenVdbFloatGrid`.
    /// This is the const half of the class.
    public class Const_OpenVdbFloatGrid : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_OpenVdbFloatGrid_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_OpenVdbFloatGrid_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_OpenVdbFloatGrid_UseCount();
                return __MR_std_shared_ptr_MR_OpenVdbFloatGrid_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_OpenVdbFloatGrid(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_OpenVdbFloatGrid_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_OpenVdbFloatGrid_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructNonOwning(ptr);
        }

        internal unsafe Const_OpenVdbFloatGrid(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe OpenVdbFloatGrid _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_OpenVdbFloatGrid_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_OpenVdbFloatGrid_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_OpenVdbFloatGrid_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_OpenVdbFloatGrid_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_OpenVdbFloatGrid_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_OpenVdbFloatGrid_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OpenVdbFloatGrid() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OpenVdbFloatGrid() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenVdbFloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_OpenVdbFloatGrid_DefaultConstruct();
            _LateMakeShared(__MR_OpenVdbFloatGrid_DefaultConstruct());
        }

        /// Generated from constructor `MR::OpenVdbFloatGrid::OpenVdbFloatGrid`.
        public unsafe Const_OpenVdbFloatGrid(MR._ByValue_OpenVdbFloatGrid _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenVdbFloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_OpenVdbFloatGrid_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OpenVdbFloatGrid._Underlying *_other);
            _LateMakeShared(__MR_OpenVdbFloatGrid_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::OpenVdbFloatGrid::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenVdbFloatGrid_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_OpenVdbFloatGrid_heapBytes(_Underlying *_this);
            return __MR_OpenVdbFloatGrid_heapBytes(_UnderlyingPtr);
        }
    }

    /// this class just hides very complex type of typedef openvdb::FloatGrid
    /// Generated from class `MR::OpenVdbFloatGrid`.
    /// This is the non-const half of the class.
    public class OpenVdbFloatGrid : Const_OpenVdbFloatGrid
    {
        internal unsafe OpenVdbFloatGrid(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe OpenVdbFloatGrid(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe OpenVdbFloatGrid() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenVdbFloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_OpenVdbFloatGrid_DefaultConstruct();
            _LateMakeShared(__MR_OpenVdbFloatGrid_DefaultConstruct());
        }

        /// Generated from constructor `MR::OpenVdbFloatGrid::OpenVdbFloatGrid`.
        public unsafe OpenVdbFloatGrid(MR._ByValue_OpenVdbFloatGrid _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OpenVdbFloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_OpenVdbFloatGrid_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OpenVdbFloatGrid._Underlying *_other);
            _LateMakeShared(__MR_OpenVdbFloatGrid_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }
    }

    /// This is used as a function parameter when the underlying function receives `OpenVdbFloatGrid` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `OpenVdbFloatGrid`/`Const_OpenVdbFloatGrid` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_OpenVdbFloatGrid
    {
        internal readonly Const_OpenVdbFloatGrid? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_OpenVdbFloatGrid() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_OpenVdbFloatGrid(Const_OpenVdbFloatGrid new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_OpenVdbFloatGrid(Const_OpenVdbFloatGrid arg) {return new(arg);}
        public _ByValue_OpenVdbFloatGrid(MR.Misc._Moved<OpenVdbFloatGrid> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_OpenVdbFloatGrid(MR.Misc._Moved<OpenVdbFloatGrid> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `OpenVdbFloatGrid` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OpenVdbFloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OpenVdbFloatGrid`/`Const_OpenVdbFloatGrid` directly.
    public class _InOptMut_OpenVdbFloatGrid
    {
        public OpenVdbFloatGrid? Opt;

        public _InOptMut_OpenVdbFloatGrid() {}
        public _InOptMut_OpenVdbFloatGrid(OpenVdbFloatGrid value) {Opt = value;}
        public static implicit operator _InOptMut_OpenVdbFloatGrid(OpenVdbFloatGrid value) {return new(value);}
    }

    /// This is used for optional parameters of class `OpenVdbFloatGrid` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OpenVdbFloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OpenVdbFloatGrid`/`Const_OpenVdbFloatGrid` to pass it to the function.
    public class _InOptConst_OpenVdbFloatGrid
    {
        public Const_OpenVdbFloatGrid? Opt;

        public _InOptConst_OpenVdbFloatGrid() {}
        public _InOptConst_OpenVdbFloatGrid(Const_OpenVdbFloatGrid value) {Opt = value;}
        public static implicit operator _InOptConst_OpenVdbFloatGrid(Const_OpenVdbFloatGrid value) {return new(value);}
    }
}
