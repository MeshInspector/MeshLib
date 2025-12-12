public static partial class MR
{
    /// Parameters for aligning 2d contour onto mesh surface
    /// Generated from class `MR::ContoursMeshAlignParams`.
    /// This is the const half of the class.
    public class Const_ContoursMeshAlignParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ContoursMeshAlignParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ContoursMeshAlignParams_Destroy(_Underlying *_this);
            __MR_ContoursMeshAlignParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ContoursMeshAlignParams() {Dispose(false);}

        /// Point coordinate on mesh, represent position of contours box 'pivotPoint' on mesh
        public unsafe MR.Const_MeshTriPoint MeshPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Get_meshPoint", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_ContoursMeshAlignParams_Get_meshPoint(_Underlying *_this);
                return new(__MR_ContoursMeshAlignParams_Get_meshPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Relative position of 'meshPoint' in contours bounding box
        /// (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
        public unsafe MR.Const_Vector2f PivotPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Get_pivotPoint", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_ContoursMeshAlignParams_Get_pivotPoint(_Underlying *_this);
                return new(__MR_ContoursMeshAlignParams_Get_pivotPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Represents 2d contours xDirection in mesh space
        public unsafe MR.Const_Vector3f XDirection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Get_xDirection", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ContoursMeshAlignParams_Get_xDirection(_Underlying *_this);
                return new(__MR_ContoursMeshAlignParams_Get_xDirection(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Represents contours normal in mesh space 
        /// if nullptr - use mesh normal at 'meshPoint'
        public unsafe ref readonly MR.Vector3f * ZDirection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Get_zDirection", ExactSpelling = true)]
                extern static MR.Vector3f **__MR_ContoursMeshAlignParams_Get_zDirection(_Underlying *_this);
                return ref *__MR_ContoursMeshAlignParams_Get_zDirection(_UnderlyingPtr);
            }
        }

        /// Contours extrusion in +z and -z direction
        public unsafe float Extrusion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Get_extrusion", ExactSpelling = true)]
                extern static float *__MR_ContoursMeshAlignParams_Get_extrusion(_Underlying *_this);
                return *__MR_ContoursMeshAlignParams_Get_extrusion(_UnderlyingPtr);
            }
        }

        /// Maximum allowed shift along 'zDirection' for alignment
        public unsafe float MaximumShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_Get_maximumShift", ExactSpelling = true)]
                extern static float *__MR_ContoursMeshAlignParams_Get_maximumShift(_Underlying *_this);
                return *__MR_ContoursMeshAlignParams_Get_maximumShift(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ContoursMeshAlignParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ContoursMeshAlignParams_DefaultConstruct();
        }

        /// Constructs `MR::ContoursMeshAlignParams` elementwise.
        public unsafe Const_ContoursMeshAlignParams(MR.Const_MeshTriPoint meshPoint, MR.Vector2f pivotPoint, MR.Vector3f xDirection, MR.Const_Vector3f? zDirection, float extrusion, float maximumShift) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_ConstructFrom(MR.MeshTriPoint._Underlying *meshPoint, MR.Vector2f pivotPoint, MR.Vector3f xDirection, MR.Const_Vector3f._Underlying *zDirection, float extrusion, float maximumShift);
            _UnderlyingPtr = __MR_ContoursMeshAlignParams_ConstructFrom(meshPoint._UnderlyingPtr, pivotPoint, xDirection, zDirection is not null ? zDirection._UnderlyingPtr : null, extrusion, maximumShift);
        }

        /// Generated from constructor `MR::ContoursMeshAlignParams::ContoursMeshAlignParams`.
        public unsafe Const_ContoursMeshAlignParams(MR.Const_ContoursMeshAlignParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_ConstructFromAnother(MR.ContoursMeshAlignParams._Underlying *_other);
            _UnderlyingPtr = __MR_ContoursMeshAlignParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Parameters for aligning 2d contour onto mesh surface
    /// Generated from class `MR::ContoursMeshAlignParams`.
    /// This is the non-const half of the class.
    public class ContoursMeshAlignParams : Const_ContoursMeshAlignParams
    {
        internal unsafe ContoursMeshAlignParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Point coordinate on mesh, represent position of contours box 'pivotPoint' on mesh
        public new unsafe MR.MeshTriPoint MeshPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_GetMutable_meshPoint", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_ContoursMeshAlignParams_GetMutable_meshPoint(_Underlying *_this);
                return new(__MR_ContoursMeshAlignParams_GetMutable_meshPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Relative position of 'meshPoint' in contours bounding box
        /// (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
        public new unsafe MR.Mut_Vector2f PivotPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_GetMutable_pivotPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_ContoursMeshAlignParams_GetMutable_pivotPoint(_Underlying *_this);
                return new(__MR_ContoursMeshAlignParams_GetMutable_pivotPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Represents 2d contours xDirection in mesh space
        public new unsafe MR.Mut_Vector3f XDirection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_GetMutable_xDirection", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ContoursMeshAlignParams_GetMutable_xDirection(_Underlying *_this);
                return new(__MR_ContoursMeshAlignParams_GetMutable_xDirection(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Represents contours normal in mesh space 
        /// if nullptr - use mesh normal at 'meshPoint'
        public new unsafe ref readonly MR.Vector3f * ZDirection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_GetMutable_zDirection", ExactSpelling = true)]
                extern static MR.Vector3f **__MR_ContoursMeshAlignParams_GetMutable_zDirection(_Underlying *_this);
                return ref *__MR_ContoursMeshAlignParams_GetMutable_zDirection(_UnderlyingPtr);
            }
        }

        /// Contours extrusion in +z and -z direction
        public new unsafe ref float Extrusion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_GetMutable_extrusion", ExactSpelling = true)]
                extern static float *__MR_ContoursMeshAlignParams_GetMutable_extrusion(_Underlying *_this);
                return ref *__MR_ContoursMeshAlignParams_GetMutable_extrusion(_UnderlyingPtr);
            }
        }

        /// Maximum allowed shift along 'zDirection' for alignment
        public new unsafe ref float MaximumShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_GetMutable_maximumShift", ExactSpelling = true)]
                extern static float *__MR_ContoursMeshAlignParams_GetMutable_maximumShift(_Underlying *_this);
                return ref *__MR_ContoursMeshAlignParams_GetMutable_maximumShift(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ContoursMeshAlignParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ContoursMeshAlignParams_DefaultConstruct();
        }

        /// Constructs `MR::ContoursMeshAlignParams` elementwise.
        public unsafe ContoursMeshAlignParams(MR.Const_MeshTriPoint meshPoint, MR.Vector2f pivotPoint, MR.Vector3f xDirection, MR.Const_Vector3f? zDirection, float extrusion, float maximumShift) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_ConstructFrom(MR.MeshTriPoint._Underlying *meshPoint, MR.Vector2f pivotPoint, MR.Vector3f xDirection, MR.Const_Vector3f._Underlying *zDirection, float extrusion, float maximumShift);
            _UnderlyingPtr = __MR_ContoursMeshAlignParams_ConstructFrom(meshPoint._UnderlyingPtr, pivotPoint, xDirection, zDirection is not null ? zDirection._UnderlyingPtr : null, extrusion, maximumShift);
        }

        /// Generated from constructor `MR::ContoursMeshAlignParams::ContoursMeshAlignParams`.
        public unsafe ContoursMeshAlignParams(MR.Const_ContoursMeshAlignParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_ConstructFromAnother(MR.ContoursMeshAlignParams._Underlying *_other);
            _UnderlyingPtr = __MR_ContoursMeshAlignParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ContoursMeshAlignParams::operator=`.
        public unsafe MR.ContoursMeshAlignParams Assign(MR.Const_ContoursMeshAlignParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursMeshAlignParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ContoursMeshAlignParams._Underlying *__MR_ContoursMeshAlignParams_AssignFromAnother(_Underlying *_this, MR.ContoursMeshAlignParams._Underlying *_other);
            return new(__MR_ContoursMeshAlignParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ContoursMeshAlignParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ContoursMeshAlignParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContoursMeshAlignParams`/`Const_ContoursMeshAlignParams` directly.
    public class _InOptMut_ContoursMeshAlignParams
    {
        public ContoursMeshAlignParams? Opt;

        public _InOptMut_ContoursMeshAlignParams() {}
        public _InOptMut_ContoursMeshAlignParams(ContoursMeshAlignParams value) {Opt = value;}
        public static implicit operator _InOptMut_ContoursMeshAlignParams(ContoursMeshAlignParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ContoursMeshAlignParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ContoursMeshAlignParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContoursMeshAlignParams`/`Const_ContoursMeshAlignParams` to pass it to the function.
    public class _InOptConst_ContoursMeshAlignParams
    {
        public Const_ContoursMeshAlignParams? Opt;

        public _InOptConst_ContoursMeshAlignParams() {}
        public _InOptConst_ContoursMeshAlignParams(Const_ContoursMeshAlignParams value) {Opt = value;}
        public static implicit operator _InOptConst_ContoursMeshAlignParams(Const_ContoursMeshAlignParams value) {return new(value);}
    }

    /// Creates planar mesh out of given contour and aligns it to given surface
    /// Generated from function `MR::alignContoursToMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> AlignContoursToMesh(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMRVector2f contours, MR.Const_ContoursMeshAlignParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_alignContoursToMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_alignContoursToMesh(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, MR.Const_ContoursMeshAlignParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_alignContoursToMesh(mesh._UnderlyingPtr, contours._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset (along -Z if zOffset is negative) to make a volumetric closed mesh
    /// note that this function also packs the mesh
    /// Generated from function `MR::addBaseToPlanarMesh`.
    public static unsafe void AddBaseToPlanarMesh(MR.Mesh mesh, float zOffset)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_addBaseToPlanarMesh", ExactSpelling = true)]
        extern static void __MR_addBaseToPlanarMesh(MR.Mesh._Underlying *mesh, float zOffset);
        __MR_addBaseToPlanarMesh(mesh._UnderlyingPtr, zOffset);
    }
}
