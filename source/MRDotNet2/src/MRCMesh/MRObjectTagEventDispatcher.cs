public static partial class MR
{
    /// class for dispatching object tag addition/removal events
    /// Generated from class `MR::ObjectTagEventDispatcher`.
    /// This is the const half of the class.
    public class Const_ObjectTagEventDispatcher : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjectTagEventDispatcher(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectTagEventDispatcher_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjectTagEventDispatcher_Destroy(_Underlying *_this);
            __MR_ObjectTagEventDispatcher_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectTagEventDispatcher() {Dispose(false);}

        /// Generated from constructor `MR::ObjectTagEventDispatcher::ObjectTagEventDispatcher`.
        public unsafe Const_ObjectTagEventDispatcher(MR._ByValue_ObjectTagEventDispatcher _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectTagEventDispatcher_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectTagEventDispatcher._Underlying *__MR_ObjectTagEventDispatcher_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectTagEventDispatcher._Underlying *_other);
            _UnderlyingPtr = __MR_ObjectTagEventDispatcher_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// returns singleton instance
        /// Generated from method `MR::ObjectTagEventDispatcher::instance`.
        public static unsafe MR.ObjectTagEventDispatcher Instance()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectTagEventDispatcher_instance", ExactSpelling = true)]
            extern static MR.ObjectTagEventDispatcher._Underlying *__MR_ObjectTagEventDispatcher_instance();
            return new(__MR_ObjectTagEventDispatcher_instance(), is_owning: false);
        }
    }

    /// class for dispatching object tag addition/removal events
    /// Generated from class `MR::ObjectTagEventDispatcher`.
    /// This is the non-const half of the class.
    public class ObjectTagEventDispatcher : Const_ObjectTagEventDispatcher
    {
        internal unsafe ObjectTagEventDispatcher(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::ObjectTagEventDispatcher::ObjectTagEventDispatcher`.
        public unsafe ObjectTagEventDispatcher(MR._ByValue_ObjectTagEventDispatcher _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectTagEventDispatcher_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectTagEventDispatcher._Underlying *__MR_ObjectTagEventDispatcher_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectTagEventDispatcher._Underlying *_other);
            _UnderlyingPtr = __MR_ObjectTagEventDispatcher_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectTagEventDispatcher::operator=`.
        public unsafe MR.ObjectTagEventDispatcher Assign(MR._ByValue_ObjectTagEventDispatcher _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectTagEventDispatcher_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectTagEventDispatcher._Underlying *__MR_ObjectTagEventDispatcher_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectTagEventDispatcher._Underlying *_other);
            return new(__MR_ObjectTagEventDispatcher_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectTagEventDispatcher` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectTagEventDispatcher`/`Const_ObjectTagEventDispatcher` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectTagEventDispatcher
    {
        internal readonly Const_ObjectTagEventDispatcher? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectTagEventDispatcher(Const_ObjectTagEventDispatcher new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ObjectTagEventDispatcher(Const_ObjectTagEventDispatcher arg) {return new(arg);}
        public _ByValue_ObjectTagEventDispatcher(MR.Misc._Moved<ObjectTagEventDispatcher> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectTagEventDispatcher(MR.Misc._Moved<ObjectTagEventDispatcher> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectTagEventDispatcher` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectTagEventDispatcher`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectTagEventDispatcher`/`Const_ObjectTagEventDispatcher` directly.
    public class _InOptMut_ObjectTagEventDispatcher
    {
        public ObjectTagEventDispatcher? Opt;

        public _InOptMut_ObjectTagEventDispatcher() {}
        public _InOptMut_ObjectTagEventDispatcher(ObjectTagEventDispatcher value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectTagEventDispatcher(ObjectTagEventDispatcher value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectTagEventDispatcher` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectTagEventDispatcher`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectTagEventDispatcher`/`Const_ObjectTagEventDispatcher` to pass it to the function.
    public class _InOptConst_ObjectTagEventDispatcher
    {
        public Const_ObjectTagEventDispatcher? Opt;

        public _InOptConst_ObjectTagEventDispatcher() {}
        public _InOptConst_ObjectTagEventDispatcher(Const_ObjectTagEventDispatcher value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectTagEventDispatcher(Const_ObjectTagEventDispatcher value) {return new(value);}
    }
}
