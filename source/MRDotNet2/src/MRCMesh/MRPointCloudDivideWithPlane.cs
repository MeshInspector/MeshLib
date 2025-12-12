public static partial class MR
{
    /// Generated from class `MR::DividePointCloudOptionalOutput`.
    /// This is the const half of the class.
    public class Const_DividePointCloudOptionalOutput : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DividePointCloudOptionalOutput(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_Destroy", ExactSpelling = true)]
            extern static void __MR_DividePointCloudOptionalOutput_Destroy(_Underlying *_this);
            __MR_DividePointCloudOptionalOutput_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DividePointCloudOptionalOutput() {Dispose(false);}

        /// optional out map from input points to output
        public unsafe ref void * OutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_Get_outVmap", ExactSpelling = true)]
                extern static void **__MR_DividePointCloudOptionalOutput_Get_outVmap(_Underlying *_this);
                return ref *__MR_DividePointCloudOptionalOutput_Get_outVmap(_UnderlyingPtr);
            }
        }

        /// optional out other part of the point cloud
        public unsafe ref void * OtherPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_Get_otherPart", ExactSpelling = true)]
                extern static void **__MR_DividePointCloudOptionalOutput_Get_otherPart(_Underlying *_this);
                return ref *__MR_DividePointCloudOptionalOutput_Get_otherPart(_UnderlyingPtr);
            }
        }

        /// optional out map from input points to other part output
        public unsafe ref void * OtherOutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_Get_otherOutVmap", ExactSpelling = true)]
                extern static void **__MR_DividePointCloudOptionalOutput_Get_otherOutVmap(_Underlying *_this);
                return ref *__MR_DividePointCloudOptionalOutput_Get_otherOutVmap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DividePointCloudOptionalOutput() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_DefaultConstruct();
            _UnderlyingPtr = __MR_DividePointCloudOptionalOutput_DefaultConstruct();
        }

        /// Constructs `MR::DividePointCloudOptionalOutput` elementwise.
        public unsafe Const_DividePointCloudOptionalOutput(MR.VertMap? outVmap, MR.PointCloud? otherPart, MR.VertMap? otherOutVmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_ConstructFrom", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_ConstructFrom(MR.VertMap._Underlying *outVmap, MR.PointCloud._Underlying *otherPart, MR.VertMap._Underlying *otherOutVmap);
            _UnderlyingPtr = __MR_DividePointCloudOptionalOutput_ConstructFrom(outVmap is not null ? outVmap._UnderlyingPtr : null, otherPart is not null ? otherPart._UnderlyingPtr : null, otherOutVmap is not null ? otherOutVmap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DividePointCloudOptionalOutput::DividePointCloudOptionalOutput`.
        public unsafe Const_DividePointCloudOptionalOutput(MR.Const_DividePointCloudOptionalOutput _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_ConstructFromAnother(MR.DividePointCloudOptionalOutput._Underlying *_other);
            _UnderlyingPtr = __MR_DividePointCloudOptionalOutput_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::DividePointCloudOptionalOutput`.
    /// This is the non-const half of the class.
    public class DividePointCloudOptionalOutput : Const_DividePointCloudOptionalOutput
    {
        internal unsafe DividePointCloudOptionalOutput(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// optional out map from input points to output
        public new unsafe ref void * OutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_GetMutable_outVmap", ExactSpelling = true)]
                extern static void **__MR_DividePointCloudOptionalOutput_GetMutable_outVmap(_Underlying *_this);
                return ref *__MR_DividePointCloudOptionalOutput_GetMutable_outVmap(_UnderlyingPtr);
            }
        }

        /// optional out other part of the point cloud
        public new unsafe ref void * OtherPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_GetMutable_otherPart", ExactSpelling = true)]
                extern static void **__MR_DividePointCloudOptionalOutput_GetMutable_otherPart(_Underlying *_this);
                return ref *__MR_DividePointCloudOptionalOutput_GetMutable_otherPart(_UnderlyingPtr);
            }
        }

        /// optional out map from input points to other part output
        public new unsafe ref void * OtherOutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_GetMutable_otherOutVmap", ExactSpelling = true)]
                extern static void **__MR_DividePointCloudOptionalOutput_GetMutable_otherOutVmap(_Underlying *_this);
                return ref *__MR_DividePointCloudOptionalOutput_GetMutable_otherOutVmap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DividePointCloudOptionalOutput() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_DefaultConstruct();
            _UnderlyingPtr = __MR_DividePointCloudOptionalOutput_DefaultConstruct();
        }

        /// Constructs `MR::DividePointCloudOptionalOutput` elementwise.
        public unsafe DividePointCloudOptionalOutput(MR.VertMap? outVmap, MR.PointCloud? otherPart, MR.VertMap? otherOutVmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_ConstructFrom", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_ConstructFrom(MR.VertMap._Underlying *outVmap, MR.PointCloud._Underlying *otherPart, MR.VertMap._Underlying *otherOutVmap);
            _UnderlyingPtr = __MR_DividePointCloudOptionalOutput_ConstructFrom(outVmap is not null ? outVmap._UnderlyingPtr : null, otherPart is not null ? otherPart._UnderlyingPtr : null, otherOutVmap is not null ? otherOutVmap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DividePointCloudOptionalOutput::DividePointCloudOptionalOutput`.
        public unsafe DividePointCloudOptionalOutput(MR.Const_DividePointCloudOptionalOutput _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_ConstructFromAnother(MR.DividePointCloudOptionalOutput._Underlying *_other);
            _UnderlyingPtr = __MR_DividePointCloudOptionalOutput_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::DividePointCloudOptionalOutput::operator=`.
        public unsafe MR.DividePointCloudOptionalOutput Assign(MR.Const_DividePointCloudOptionalOutput _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePointCloudOptionalOutput_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DividePointCloudOptionalOutput._Underlying *__MR_DividePointCloudOptionalOutput_AssignFromAnother(_Underlying *_this, MR.DividePointCloudOptionalOutput._Underlying *_other);
            return new(__MR_DividePointCloudOptionalOutput_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DividePointCloudOptionalOutput` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DividePointCloudOptionalOutput`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DividePointCloudOptionalOutput`/`Const_DividePointCloudOptionalOutput` directly.
    public class _InOptMut_DividePointCloudOptionalOutput
    {
        public DividePointCloudOptionalOutput? Opt;

        public _InOptMut_DividePointCloudOptionalOutput() {}
        public _InOptMut_DividePointCloudOptionalOutput(DividePointCloudOptionalOutput value) {Opt = value;}
        public static implicit operator _InOptMut_DividePointCloudOptionalOutput(DividePointCloudOptionalOutput value) {return new(value);}
    }

    /// This is used for optional parameters of class `DividePointCloudOptionalOutput` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DividePointCloudOptionalOutput`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DividePointCloudOptionalOutput`/`Const_DividePointCloudOptionalOutput` to pass it to the function.
    public class _InOptConst_DividePointCloudOptionalOutput
    {
        public Const_DividePointCloudOptionalOutput? Opt;

        public _InOptConst_DividePointCloudOptionalOutput() {}
        public _InOptConst_DividePointCloudOptionalOutput(Const_DividePointCloudOptionalOutput value) {Opt = value;}
        public static implicit operator _InOptConst_DividePointCloudOptionalOutput(Const_DividePointCloudOptionalOutput value) {return new(value);}
    }

    /// \return All vertices on the positive side of the plane
    /// \param pc Input point cloud that will be cut by the plane
    /// \param plane Input plane to cut point cloud with
    /// Generated from function `MR::findHalfSpacePoints`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> FindHalfSpacePoints(MR.Const_PointCloud pc, MR.Const_Plane3f plane)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findHalfSpacePoints", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_findHalfSpacePoints(MR.Const_PointCloud._Underlying *pc, MR.Const_Plane3f._Underlying *plane);
        return MR.Misc.Move(new MR.VertBitSet(__MR_findHalfSpacePoints(pc._UnderlyingPtr, plane._UnderlyingPtr), is_owning: true));
    }

    /// This function cuts a point cloud with a plane, leaving only the part of mesh that lay in positive direction of normal
    /// \return Point cloud object with vertices on the positive side of the plane
    /// \param pc Input point cloud that will be cut by the plane
    /// \param plane Input plane to cut point cloud with
    /// \param optOut optional output of the function
    /// Generated from function `MR::divideWithPlane`.
    /// Parameter `optOut` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.PointCloud> DivideWithPlane(MR.Const_PointCloud points, MR.Const_Plane3f plane, MR.Const_DividePointCloudOptionalOutput? optOut = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_divideWithPlane", ExactSpelling = true)]
        extern static MR.PointCloud._Underlying *__MR_divideWithPlane(MR.Const_PointCloud._Underlying *points, MR.Const_Plane3f._Underlying *plane, MR.Const_DividePointCloudOptionalOutput._Underlying *optOut);
        return MR.Misc.Move(new MR.PointCloud(__MR_divideWithPlane(points._UnderlyingPtr, plane._UnderlyingPtr, optOut is not null ? optOut._UnderlyingPtr : null), is_owning: true));
    }
}
