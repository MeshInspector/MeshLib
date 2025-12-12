public static partial class MR
{
    public enum QuarterBit : byte
    {
        LeftLeft = 1,
        LeftRight = 2,
        RightLeft = 4,
        RightRight = 8,
        All = 15,
    }

    /// Plane of slice in which to find path
    public enum SlicePlane : int
    {
        ///< = 0 cause main axis is x - [0]
        YZ = 0,
        ///< = 1 cause main axis is y - [1]
        ZX = 1,
        ///< = 2 cause main axis is z - [2]
        XY = 2,
        ///< special value not to limit path in one slice
        None = 3,
    }

    /// Parameters for building metric function
    /// Generated from class `MR::VoxelMetricParameters`.
    /// This is the const half of the class.
    public class Const_VoxelMetricParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelMetricParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelMetricParameters_Destroy(_Underlying *_this);
            __MR_VoxelMetricParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelMetricParameters() {Dispose(false);}

        ///< start voxel index
        public unsafe ulong Start
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_Get_start", ExactSpelling = true)]
                extern static ulong *__MR_VoxelMetricParameters_Get_start(_Underlying *_this);
                return *__MR_VoxelMetricParameters_Get_start(_UnderlyingPtr);
            }
        }

        ///< stop voxel index
        public unsafe ulong Stop
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_Get_stop", ExactSpelling = true)]
                extern static ulong *__MR_VoxelMetricParameters_Get_stop(_Underlying *_this);
                return *__MR_VoxelMetricParameters_Get_stop(_UnderlyingPtr);
            }
        }

        ///< max distance ratio: if (dist^2(next,start) + dist^2(next,stop) > maxDistRatio^2*dist^2(start,stop)) - candidate is not processed
        public unsafe float MaxDistRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_Get_maxDistRatio", ExactSpelling = true)]
                extern static float *__MR_VoxelMetricParameters_Get_maxDistRatio(_Underlying *_this);
                return *__MR_VoxelMetricParameters_Get_maxDistRatio(_UnderlyingPtr);
            }
        }

        ///< if not None - builds path in one slice of voxels (make sure start and stop has same main axis coordinate)
        public unsafe MR.SlicePlane Plane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_Get_plane", ExactSpelling = true)]
                extern static MR.SlicePlane *__MR_VoxelMetricParameters_Get_plane(_Underlying *_this);
                return *__MR_VoxelMetricParameters_Get_plane(_UnderlyingPtr);
            }
        }

        ///< quarter of building path, if plane is selected, it should be (LeftLeft | LeftRigth) or (RigthLeft | RightRight) or All
        public unsafe MR.QuarterBit QuatersMask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_Get_quatersMask", ExactSpelling = true)]
                extern static MR.QuarterBit *__MR_VoxelMetricParameters_Get_quatersMask(_Underlying *_this);
                return *__MR_VoxelMetricParameters_Get_quatersMask(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelMetricParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelMetricParameters_DefaultConstruct();
        }

        /// Constructs `MR::VoxelMetricParameters` elementwise.
        public unsafe Const_VoxelMetricParameters(ulong start, ulong stop, float maxDistRatio, MR.SlicePlane plane, MR.QuarterBit quatersMask) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_ConstructFrom(ulong start, ulong stop, float maxDistRatio, MR.SlicePlane plane, MR.QuarterBit quatersMask);
            _UnderlyingPtr = __MR_VoxelMetricParameters_ConstructFrom(start, stop, maxDistRatio, plane, quatersMask);
        }

        /// Generated from constructor `MR::VoxelMetricParameters::VoxelMetricParameters`.
        public unsafe Const_VoxelMetricParameters(MR.Const_VoxelMetricParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_ConstructFromAnother(MR.VoxelMetricParameters._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelMetricParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Parameters for building metric function
    /// Generated from class `MR::VoxelMetricParameters`.
    /// This is the non-const half of the class.
    public class VoxelMetricParameters : Const_VoxelMetricParameters
    {
        internal unsafe VoxelMetricParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< start voxel index
        public new unsafe ref ulong Start
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_GetMutable_start", ExactSpelling = true)]
                extern static ulong *__MR_VoxelMetricParameters_GetMutable_start(_Underlying *_this);
                return ref *__MR_VoxelMetricParameters_GetMutable_start(_UnderlyingPtr);
            }
        }

        ///< stop voxel index
        public new unsafe ref ulong Stop
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_GetMutable_stop", ExactSpelling = true)]
                extern static ulong *__MR_VoxelMetricParameters_GetMutable_stop(_Underlying *_this);
                return ref *__MR_VoxelMetricParameters_GetMutable_stop(_UnderlyingPtr);
            }
        }

        ///< max distance ratio: if (dist^2(next,start) + dist^2(next,stop) > maxDistRatio^2*dist^2(start,stop)) - candidate is not processed
        public new unsafe ref float MaxDistRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_GetMutable_maxDistRatio", ExactSpelling = true)]
                extern static float *__MR_VoxelMetricParameters_GetMutable_maxDistRatio(_Underlying *_this);
                return ref *__MR_VoxelMetricParameters_GetMutable_maxDistRatio(_UnderlyingPtr);
            }
        }

        ///< if not None - builds path in one slice of voxels (make sure start and stop has same main axis coordinate)
        public new unsafe ref MR.SlicePlane Plane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_GetMutable_plane", ExactSpelling = true)]
                extern static MR.SlicePlane *__MR_VoxelMetricParameters_GetMutable_plane(_Underlying *_this);
                return ref *__MR_VoxelMetricParameters_GetMutable_plane(_UnderlyingPtr);
            }
        }

        ///< quarter of building path, if plane is selected, it should be (LeftLeft | LeftRigth) or (RigthLeft | RightRight) or All
        public new unsafe ref MR.QuarterBit QuatersMask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_GetMutable_quatersMask", ExactSpelling = true)]
                extern static MR.QuarterBit *__MR_VoxelMetricParameters_GetMutable_quatersMask(_Underlying *_this);
                return ref *__MR_VoxelMetricParameters_GetMutable_quatersMask(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelMetricParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelMetricParameters_DefaultConstruct();
        }

        /// Constructs `MR::VoxelMetricParameters` elementwise.
        public unsafe VoxelMetricParameters(ulong start, ulong stop, float maxDistRatio, MR.SlicePlane plane, MR.QuarterBit quatersMask) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_ConstructFrom(ulong start, ulong stop, float maxDistRatio, MR.SlicePlane plane, MR.QuarterBit quatersMask);
            _UnderlyingPtr = __MR_VoxelMetricParameters_ConstructFrom(start, stop, maxDistRatio, plane, quatersMask);
        }

        /// Generated from constructor `MR::VoxelMetricParameters::VoxelMetricParameters`.
        public unsafe VoxelMetricParameters(MR.Const_VoxelMetricParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_ConstructFromAnother(MR.VoxelMetricParameters._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelMetricParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelMetricParameters::operator=`.
        public unsafe MR.VoxelMetricParameters Assign(MR.Const_VoxelMetricParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelMetricParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelMetricParameters._Underlying *__MR_VoxelMetricParameters_AssignFromAnother(_Underlying *_this, MR.VoxelMetricParameters._Underlying *_other);
            return new(__MR_VoxelMetricParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelMetricParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelMetricParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelMetricParameters`/`Const_VoxelMetricParameters` directly.
    public class _InOptMut_VoxelMetricParameters
    {
        public VoxelMetricParameters? Opt;

        public _InOptMut_VoxelMetricParameters() {}
        public _InOptMut_VoxelMetricParameters(VoxelMetricParameters value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelMetricParameters(VoxelMetricParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelMetricParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelMetricParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelMetricParameters`/`Const_VoxelMetricParameters` to pass it to the function.
    public class _InOptConst_VoxelMetricParameters
    {
        public Const_VoxelMetricParameters? Opt;

        public _InOptConst_VoxelMetricParameters() {}
        public _InOptConst_VoxelMetricParameters(Const_VoxelMetricParameters value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelMetricParameters(Const_VoxelMetricParameters value) {return new(value);}
    }

    /// Generated from function `MR::operator&`.
    public static MR.QuarterBit Bitand(MR.QuarterBit a, MR.QuarterBit b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_QuarterBit", ExactSpelling = true)]
        extern static MR.QuarterBit __MR_bitand_MR_QuarterBit(MR.QuarterBit a, MR.QuarterBit b);
        return __MR_bitand_MR_QuarterBit(a, b);
    }

    /// Generated from function `MR::operator|`.
    public static MR.QuarterBit Bitor(MR.QuarterBit a, MR.QuarterBit b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_QuarterBit", ExactSpelling = true)]
        extern static MR.QuarterBit __MR_bitor_MR_QuarterBit(MR.QuarterBit a, MR.QuarterBit b);
        return __MR_bitor_MR_QuarterBit(a, b);
    }

    /// Generated from function `MR::operator~`.
    public static MR.QuarterBit Compl(MR.QuarterBit a)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compl_MR_QuarterBit", ExactSpelling = true)]
        extern static MR.QuarterBit __MR_compl_MR_QuarterBit(MR.QuarterBit a);
        return __MR_compl_MR_QuarterBit(a);
    }

    /// Generated from function `MR::operator&=`.
    public static unsafe ref MR.QuarterBit BitandAssign(ref MR.QuarterBit a, MR.QuarterBit b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_assign_MR_QuarterBit", ExactSpelling = true)]
        extern static MR.QuarterBit *__MR_bitand_assign_MR_QuarterBit(MR.QuarterBit *a, MR.QuarterBit b);
        fixed (MR.QuarterBit *__ptr_a = &a)
        {
            return ref *__MR_bitand_assign_MR_QuarterBit(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator|=`.
    public static unsafe ref MR.QuarterBit BitorAssign(ref MR.QuarterBit a, MR.QuarterBit b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_assign_MR_QuarterBit", ExactSpelling = true)]
        extern static MR.QuarterBit *__MR_bitor_assign_MR_QuarterBit(MR.QuarterBit *a, MR.QuarterBit b);
        fixed (MR.QuarterBit *__ptr_a = &a)
        {
            return ref *__MR_bitor_assign_MR_QuarterBit(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator*`.
    public static MR.QuarterBit Mul(MR.QuarterBit a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_QuarterBit_bool", ExactSpelling = true)]
        extern static MR.QuarterBit __MR_mul_MR_QuarterBit_bool(MR.QuarterBit a, byte b);
        return __MR_mul_MR_QuarterBit_bool(a, b ? (byte)1 : (byte)0);
    }

    /// Generated from function `MR::operator*`.
    public static MR.QuarterBit Mul(bool a, MR.QuarterBit b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_QuarterBit", ExactSpelling = true)]
        extern static MR.QuarterBit __MR_mul_bool_MR_QuarterBit(byte a, MR.QuarterBit b);
        return __MR_mul_bool_MR_QuarterBit(a ? (byte)1 : (byte)0, b);
    }

    /// Generated from function `MR::operator*=`.
    public static unsafe ref MR.QuarterBit MulAssign(ref MR.QuarterBit a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_QuarterBit_bool", ExactSpelling = true)]
        extern static MR.QuarterBit *__MR_mul_assign_MR_QuarterBit_bool(MR.QuarterBit *a, byte b);
        fixed (MR.QuarterBit *__ptr_a = &a)
        {
            return ref *__MR_mul_assign_MR_QuarterBit_bool(__ptr_a, b ? (byte)1 : (byte)0);
        }
    }

    /// e^(modifier*(dens1+dens2))
    /// Generated from function `MR::voxelsExponentMetric`.
    /// Parameter `modifier` defaults to `-1.0f`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMRUint64TMRUint64T> VoxelsExponentMetric(MR.Const_VdbVolume voxels, MR.Const_VoxelMetricParameters parameters, float? modifier = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_voxelsExponentMetric", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMRUint64TMRUint64T._Underlying *__MR_voxelsExponentMetric(MR.Const_VdbVolume._Underlying *voxels, MR.Const_VoxelMetricParameters._Underlying *parameters, float *modifier);
        float __deref_modifier = modifier.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMRUint64TMRUint64T(__MR_voxelsExponentMetric(voxels._UnderlyingPtr, parameters._UnderlyingPtr, modifier.HasValue ? &__deref_modifier : null), is_owning: true));
    }

    /// sum of dense differences with start and stop voxels
    /// Generated from function `MR::voxelsSumDiffsMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMRUint64TMRUint64T> VoxelsSumDiffsMetric(MR.Const_VdbVolume voxels, MR.Const_VoxelMetricParameters parameters)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_voxelsSumDiffsMetric", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMRUint64TMRUint64T._Underlying *__MR_voxelsSumDiffsMetric(MR.Const_VdbVolume._Underlying *voxels, MR.Const_VoxelMetricParameters._Underlying *parameters);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMRUint64TMRUint64T(__MR_voxelsSumDiffsMetric(voxels._UnderlyingPtr, parameters._UnderlyingPtr), is_owning: true));
    }

    /// builds shortest path in given metric from start to finish voxels; if no path can be found then empty path is returned
    /// Generated from function `MR::buildSmallestMetricPath`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUint64T> BuildSmallestMetricPath(MR.Const_VdbVolume voxels, MR.Std.Const_Function_FloatFuncFromMRUint64TMRUint64T metric, ulong start, ulong finish, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildSmallestMetricPath_const_MR_VdbVolume_ref", ExactSpelling = true)]
        extern static MR.Std.Vector_MRUint64T._Underlying *__MR_buildSmallestMetricPath_const_MR_VdbVolume_ref(MR.Const_VdbVolume._Underlying *voxels, MR.Std.Const_Function_FloatFuncFromMRUint64TMRUint64T._Underlying *metric, ulong start, ulong finish, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Vector_MRUint64T(__MR_buildSmallestMetricPath_const_MR_VdbVolume_ref(voxels._UnderlyingPtr, metric._UnderlyingPtr, start, finish, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }
}
