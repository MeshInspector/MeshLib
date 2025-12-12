public static partial class MR
{
    /// optional T-object container, which stores a transformation as well for which the object is valid
    /// Generated from class `MR::XfBasedCache<MR::Box3f>`.
    /// This is the const half of the class.
    public class Const_XfBasedCache_MRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_XfBasedCache_MRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_XfBasedCache_MR_Box3f_Destroy(_Underlying *_this);
            __MR_XfBasedCache_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_XfBasedCache_MRBox3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_XfBasedCache_MRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_XfBasedCache_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_XfBasedCache_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::XfBasedCache<MR::Box3f>::XfBasedCache`.
        public unsafe Const_XfBasedCache_MRBox3f(MR.Const_XfBasedCache_MRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_XfBasedCache_MR_Box3f_ConstructFromAnother(MR.XfBasedCache_MRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_XfBasedCache_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns stored object only if requested transformation is the same as stored one
        /// Generated from method `MR::XfBasedCache<MR::Box3f>::get`.
        public unsafe MR.Std.Const_Optional_MRBox3f Get(MR.Const_AffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_get", ExactSpelling = true)]
            extern static MR.Std.Const_Optional_MRBox3f._Underlying *__MR_XfBasedCache_MR_Box3f_get(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf);
            return new(__MR_XfBasedCache_MR_Box3f_get(_UnderlyingPtr, xf._UnderlyingPtr), is_owning: false);
        }
    }

    /// optional T-object container, which stores a transformation as well for which the object is valid
    /// Generated from class `MR::XfBasedCache<MR::Box3f>`.
    /// This is the non-const half of the class.
    public class XfBasedCache_MRBox3f : Const_XfBasedCache_MRBox3f
    {
        internal unsafe XfBasedCache_MRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe XfBasedCache_MRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_XfBasedCache_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_XfBasedCache_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::XfBasedCache<MR::Box3f>::XfBasedCache`.
        public unsafe XfBasedCache_MRBox3f(MR.Const_XfBasedCache_MRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_XfBasedCache_MR_Box3f_ConstructFromAnother(MR.XfBasedCache_MRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_XfBasedCache_MR_Box3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::XfBasedCache<MR::Box3f>::operator=`.
        public unsafe MR.XfBasedCache_MRBox3f Assign(MR.Const_XfBasedCache_MRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.XfBasedCache_MRBox3f._Underlying *__MR_XfBasedCache_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.XfBasedCache_MRBox3f._Underlying *_other);
            return new(__MR_XfBasedCache_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// sets new transformation and the object
        /// Generated from method `MR::XfBasedCache<MR::Box3f>::set`.
        public unsafe void Set(MR.Const_AffineXf3f xf, MR.Box3f t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_set", ExactSpelling = true)]
            extern static void __MR_XfBasedCache_MR_Box3f_set(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.Box3f t);
            __MR_XfBasedCache_MR_Box3f_set(_UnderlyingPtr, xf._UnderlyingPtr, t);
        }

        /// clears stored object
        /// Generated from method `MR::XfBasedCache<MR::Box3f>::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_XfBasedCache_MR_Box3f_reset", ExactSpelling = true)]
            extern static void __MR_XfBasedCache_MR_Box3f_reset(_Underlying *_this);
            __MR_XfBasedCache_MR_Box3f_reset(_UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `XfBasedCache_MRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_XfBasedCache_MRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `XfBasedCache_MRBox3f`/`Const_XfBasedCache_MRBox3f` directly.
    public class _InOptMut_XfBasedCache_MRBox3f
    {
        public XfBasedCache_MRBox3f? Opt;

        public _InOptMut_XfBasedCache_MRBox3f() {}
        public _InOptMut_XfBasedCache_MRBox3f(XfBasedCache_MRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_XfBasedCache_MRBox3f(XfBasedCache_MRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `XfBasedCache_MRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_XfBasedCache_MRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `XfBasedCache_MRBox3f`/`Const_XfBasedCache_MRBox3f` to pass it to the function.
    public class _InOptConst_XfBasedCache_MRBox3f
    {
        public Const_XfBasedCache_MRBox3f? Opt;

        public _InOptConst_XfBasedCache_MRBox3f() {}
        public _InOptConst_XfBasedCache_MRBox3f(Const_XfBasedCache_MRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_XfBasedCache_MRBox3f(Const_XfBasedCache_MRBox3f value) {return new(value);}
    }
}
