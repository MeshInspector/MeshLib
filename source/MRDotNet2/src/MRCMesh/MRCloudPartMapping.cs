public static partial class MR
{
    // mapping among elements of source point cloud, from which a part is taken, and target (this) point cloud
    /// Generated from class `MR::CloudPartMapping`.
    /// This is the const half of the class.
    public class Const_CloudPartMapping : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CloudPartMapping(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_Destroy", ExactSpelling = true)]
            extern static void __MR_CloudPartMapping_Destroy(_Underlying *_this);
            __MR_CloudPartMapping_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CloudPartMapping() {Dispose(false);}

        // from.id -> this.id, efficient when full cloud without many invalid points is added into another cloud
        public unsafe ref void * Src2tgtVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_Get_src2tgtVerts", ExactSpelling = true)]
                extern static void **__MR_CloudPartMapping_Get_src2tgtVerts(_Underlying *_this);
                return ref *__MR_CloudPartMapping_Get_src2tgtVerts(_UnderlyingPtr);
            }
        }

        // this.id -> from.id, efficient when any cloud or its part is added into empty cloud
        public unsafe ref void * Tgt2srcVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_Get_tgt2srcVerts", ExactSpelling = true)]
                extern static void **__MR_CloudPartMapping_Get_tgt2srcVerts(_Underlying *_this);
                return ref *__MR_CloudPartMapping_Get_tgt2srcVerts(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CloudPartMapping() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_DefaultConstruct();
            _UnderlyingPtr = __MR_CloudPartMapping_DefaultConstruct();
        }

        /// Constructs `MR::CloudPartMapping` elementwise.
        public unsafe Const_CloudPartMapping(MR.VertMap? src2tgtVerts, MR.VertMap? tgt2srcVerts) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_ConstructFrom", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_ConstructFrom(MR.VertMap._Underlying *src2tgtVerts, MR.VertMap._Underlying *tgt2srcVerts);
            _UnderlyingPtr = __MR_CloudPartMapping_ConstructFrom(src2tgtVerts is not null ? src2tgtVerts._UnderlyingPtr : null, tgt2srcVerts is not null ? tgt2srcVerts._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CloudPartMapping::CloudPartMapping`.
        public unsafe Const_CloudPartMapping(MR.Const_CloudPartMapping _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_ConstructFromAnother(MR.CloudPartMapping._Underlying *_other);
            _UnderlyingPtr = __MR_CloudPartMapping_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // mapping among elements of source point cloud, from which a part is taken, and target (this) point cloud
    /// Generated from class `MR::CloudPartMapping`.
    /// This is the non-const half of the class.
    public class CloudPartMapping : Const_CloudPartMapping
    {
        internal unsafe CloudPartMapping(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // from.id -> this.id, efficient when full cloud without many invalid points is added into another cloud
        public new unsafe ref void * Src2tgtVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_GetMutable_src2tgtVerts", ExactSpelling = true)]
                extern static void **__MR_CloudPartMapping_GetMutable_src2tgtVerts(_Underlying *_this);
                return ref *__MR_CloudPartMapping_GetMutable_src2tgtVerts(_UnderlyingPtr);
            }
        }

        // this.id -> from.id, efficient when any cloud or its part is added into empty cloud
        public new unsafe ref void * Tgt2srcVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_GetMutable_tgt2srcVerts", ExactSpelling = true)]
                extern static void **__MR_CloudPartMapping_GetMutable_tgt2srcVerts(_Underlying *_this);
                return ref *__MR_CloudPartMapping_GetMutable_tgt2srcVerts(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CloudPartMapping() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_DefaultConstruct();
            _UnderlyingPtr = __MR_CloudPartMapping_DefaultConstruct();
        }

        /// Constructs `MR::CloudPartMapping` elementwise.
        public unsafe CloudPartMapping(MR.VertMap? src2tgtVerts, MR.VertMap? tgt2srcVerts) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_ConstructFrom", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_ConstructFrom(MR.VertMap._Underlying *src2tgtVerts, MR.VertMap._Underlying *tgt2srcVerts);
            _UnderlyingPtr = __MR_CloudPartMapping_ConstructFrom(src2tgtVerts is not null ? src2tgtVerts._UnderlyingPtr : null, tgt2srcVerts is not null ? tgt2srcVerts._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CloudPartMapping::CloudPartMapping`.
        public unsafe CloudPartMapping(MR.Const_CloudPartMapping _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_ConstructFromAnother(MR.CloudPartMapping._Underlying *_other);
            _UnderlyingPtr = __MR_CloudPartMapping_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CloudPartMapping::operator=`.
        public unsafe MR.CloudPartMapping Assign(MR.Const_CloudPartMapping _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloudPartMapping_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CloudPartMapping._Underlying *__MR_CloudPartMapping_AssignFromAnother(_Underlying *_this, MR.CloudPartMapping._Underlying *_other);
            return new(__MR_CloudPartMapping_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CloudPartMapping` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CloudPartMapping`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CloudPartMapping`/`Const_CloudPartMapping` directly.
    public class _InOptMut_CloudPartMapping
    {
        public CloudPartMapping? Opt;

        public _InOptMut_CloudPartMapping() {}
        public _InOptMut_CloudPartMapping(CloudPartMapping value) {Opt = value;}
        public static implicit operator _InOptMut_CloudPartMapping(CloudPartMapping value) {return new(value);}
    }

    /// This is used for optional parameters of class `CloudPartMapping` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CloudPartMapping`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CloudPartMapping`/`Const_CloudPartMapping` to pass it to the function.
    public class _InOptConst_CloudPartMapping
    {
        public Const_CloudPartMapping? Opt;

        public _InOptConst_CloudPartMapping() {}
        public _InOptConst_CloudPartMapping(Const_CloudPartMapping value) {Opt = value;}
        public static implicit operator _InOptConst_CloudPartMapping(Const_CloudPartMapping value) {return new(value);}
    }
}
