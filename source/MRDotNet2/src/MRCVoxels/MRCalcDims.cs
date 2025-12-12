public static partial class MR
{
    /// shift of zero voxel in 3D space and dimensions of voxel-grid
    /// Generated from class `MR::OriginAndDimensions`.
    /// This is the const half of the class.
    public class Const_OriginAndDimensions : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OriginAndDimensions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_Destroy", ExactSpelling = true)]
            extern static void __MR_OriginAndDimensions_Destroy(_Underlying *_this);
            __MR_OriginAndDimensions_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OriginAndDimensions() {Dispose(false);}

        public unsafe MR.Const_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_Get_origin", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_OriginAndDimensions_Get_origin(_Underlying *_this);
                return new(__MR_OriginAndDimensions_Get_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dimensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_Get_dimensions", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_OriginAndDimensions_Get_dimensions(_Underlying *_this);
                return new(__MR_OriginAndDimensions_Get_dimensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OriginAndDimensions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_DefaultConstruct();
            _UnderlyingPtr = __MR_OriginAndDimensions_DefaultConstruct();
        }

        /// Constructs `MR::OriginAndDimensions` elementwise.
        public unsafe Const_OriginAndDimensions(MR.Vector3f origin, MR.Vector3i dimensions) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_ConstructFrom", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_ConstructFrom(MR.Vector3f origin, MR.Vector3i dimensions);
            _UnderlyingPtr = __MR_OriginAndDimensions_ConstructFrom(origin, dimensions);
        }

        /// Generated from constructor `MR::OriginAndDimensions::OriginAndDimensions`.
        public unsafe Const_OriginAndDimensions(MR.Const_OriginAndDimensions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_ConstructFromAnother(MR.OriginAndDimensions._Underlying *_other);
            _UnderlyingPtr = __MR_OriginAndDimensions_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// shift of zero voxel in 3D space and dimensions of voxel-grid
    /// Generated from class `MR::OriginAndDimensions`.
    /// This is the non-const half of the class.
    public class OriginAndDimensions : Const_OriginAndDimensions
    {
        internal unsafe OriginAndDimensions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_GetMutable_origin", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_OriginAndDimensions_GetMutable_origin(_Underlying *_this);
                return new(__MR_OriginAndDimensions_GetMutable_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dimensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_GetMutable_dimensions", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_OriginAndDimensions_GetMutable_dimensions(_Underlying *_this);
                return new(__MR_OriginAndDimensions_GetMutable_dimensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OriginAndDimensions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_DefaultConstruct();
            _UnderlyingPtr = __MR_OriginAndDimensions_DefaultConstruct();
        }

        /// Constructs `MR::OriginAndDimensions` elementwise.
        public unsafe OriginAndDimensions(MR.Vector3f origin, MR.Vector3i dimensions) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_ConstructFrom", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_ConstructFrom(MR.Vector3f origin, MR.Vector3i dimensions);
            _UnderlyingPtr = __MR_OriginAndDimensions_ConstructFrom(origin, dimensions);
        }

        /// Generated from constructor `MR::OriginAndDimensions::OriginAndDimensions`.
        public unsafe OriginAndDimensions(MR.Const_OriginAndDimensions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_ConstructFromAnother(MR.OriginAndDimensions._Underlying *_other);
            _UnderlyingPtr = __MR_OriginAndDimensions_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OriginAndDimensions::operator=`.
        public unsafe MR.OriginAndDimensions Assign(MR.Const_OriginAndDimensions _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OriginAndDimensions_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OriginAndDimensions._Underlying *__MR_OriginAndDimensions_AssignFromAnother(_Underlying *_this, MR.OriginAndDimensions._Underlying *_other);
            return new(__MR_OriginAndDimensions_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OriginAndDimensions` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OriginAndDimensions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OriginAndDimensions`/`Const_OriginAndDimensions` directly.
    public class _InOptMut_OriginAndDimensions
    {
        public OriginAndDimensions? Opt;

        public _InOptMut_OriginAndDimensions() {}
        public _InOptMut_OriginAndDimensions(OriginAndDimensions value) {Opt = value;}
        public static implicit operator _InOptMut_OriginAndDimensions(OriginAndDimensions value) {return new(value);}
    }

    /// This is used for optional parameters of class `OriginAndDimensions` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OriginAndDimensions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OriginAndDimensions`/`Const_OriginAndDimensions` to pass it to the function.
    public class _InOptConst_OriginAndDimensions
    {
        public Const_OriginAndDimensions? Opt;

        public _InOptConst_OriginAndDimensions() {}
        public _InOptConst_OriginAndDimensions(Const_OriginAndDimensions value) {Opt = value;}
        public static implicit operator _InOptConst_OriginAndDimensions(Const_OriginAndDimensions value) {return new(value);}
    }

    /// computes origin and dimensions of voxel-grid to cover given 3D box with given spacing (voxelSize)
    /// Generated from function `MR::calcOriginAndDimensions`.
    public static unsafe MR.OriginAndDimensions CalcOriginAndDimensions(MR.Const_Box3f box, float voxelSize)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcOriginAndDimensions", ExactSpelling = true)]
        extern static MR.OriginAndDimensions._Underlying *__MR_calcOriginAndDimensions(MR.Const_Box3f._Underlying *box, float voxelSize);
        return new(__MR_calcOriginAndDimensions(box._UnderlyingPtr, voxelSize), is_owning: true);
    }
}
