public static partial class MR
{
    /// represents full point cloud (if region is nullptr) or some portion of point cloud (if region pointer is valid)
    /// Generated from class `MR::PointCloudPart`.
    /// This is the const half of the class.
    public class Const_PointCloudPart : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointCloudPart(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_Destroy", ExactSpelling = true)]
            extern static void __MR_PointCloudPart_Destroy(_Underlying *_this);
            __MR_PointCloudPart_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointCloudPart() {Dispose(false);}

        public unsafe MR.Const_PointCloud Cloud
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_Get_cloud", ExactSpelling = true)]
                extern static MR.Const_PointCloud._Underlying *__MR_PointCloudPart_Get_cloud(_Underlying *_this);
                return new(__MR_PointCloudPart_Get_cloud(_UnderlyingPtr), is_owning: false);
            }
        }

        // nullptr here means all valid points of point cloud
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_Get_region", ExactSpelling = true)]
                extern static void **__MR_PointCloudPart_Get_region(_Underlying *_this);
                return ref *__MR_PointCloudPart_Get_region(_UnderlyingPtr);
            }
        }

        // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
        /// Generated from constructor `MR::PointCloudPart::PointCloudPart`.
        public unsafe Const_PointCloudPart(MR.Const_PointCloudPart other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudPart._Underlying *__MR_PointCloudPart_ConstructFromAnother(MR.PointCloudPart._Underlying *other);
            _UnderlyingPtr = __MR_PointCloudPart_ConstructFromAnother(other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PointCloudPart::PointCloudPart`.
        public unsafe Const_PointCloudPart(MR.Const_PointCloud c, MR.Const_VertBitSet? bs = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_Construct", ExactSpelling = true)]
            extern static MR.PointCloudPart._Underlying *__MR_PointCloudPart_Construct(MR.Const_PointCloud._Underlying *c, MR.Const_VertBitSet._Underlying *bs);
            _UnderlyingPtr = __MR_PointCloudPart_Construct(c._UnderlyingPtr, bs is not null ? bs._UnderlyingPtr : null);
        }
    }

    /// represents full point cloud (if region is nullptr) or some portion of point cloud (if region pointer is valid)
    /// Generated from class `MR::PointCloudPart`.
    /// This is the non-const half of the class.
    public class PointCloudPart : Const_PointCloudPart
    {
        internal unsafe PointCloudPart(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // nullptr here means all valid points of point cloud
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_PointCloudPart_GetMutable_region(_Underlying *_this);
                return ref *__MR_PointCloudPart_GetMutable_region(_UnderlyingPtr);
            }
        }

        // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
        /// Generated from constructor `MR::PointCloudPart::PointCloudPart`.
        public unsafe PointCloudPart(MR.Const_PointCloudPart other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudPart._Underlying *__MR_PointCloudPart_ConstructFromAnother(MR.PointCloudPart._Underlying *other);
            _UnderlyingPtr = __MR_PointCloudPart_ConstructFromAnother(other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PointCloudPart::PointCloudPart`.
        public unsafe PointCloudPart(MR.Const_PointCloud c, MR.Const_VertBitSet? bs = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudPart_Construct", ExactSpelling = true)]
            extern static MR.PointCloudPart._Underlying *__MR_PointCloudPart_Construct(MR.Const_PointCloud._Underlying *c, MR.Const_VertBitSet._Underlying *bs);
            _UnderlyingPtr = __MR_PointCloudPart_Construct(c._UnderlyingPtr, bs is not null ? bs._UnderlyingPtr : null);
        }
    }

    /// This is used for optional parameters of class `PointCloudPart` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointCloudPart`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloudPart`/`Const_PointCloudPart` directly.
    public class _InOptMut_PointCloudPart
    {
        public PointCloudPart? Opt;

        public _InOptMut_PointCloudPart() {}
        public _InOptMut_PointCloudPart(PointCloudPart value) {Opt = value;}
        public static implicit operator _InOptMut_PointCloudPart(PointCloudPart value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointCloudPart` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointCloudPart`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloudPart`/`Const_PointCloudPart` to pass it to the function.
    public class _InOptConst_PointCloudPart
    {
        public Const_PointCloudPart? Opt;

        public _InOptConst_PointCloudPart() {}
        public _InOptConst_PointCloudPart(Const_PointCloudPart value) {Opt = value;}
        public static implicit operator _InOptConst_PointCloudPart(Const_PointCloudPart value) {return new(value);}
    }
}
