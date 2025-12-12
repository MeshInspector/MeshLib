public static partial class MR
{
    /**
    * \brief Parameters for volume segmentation
    * 
    * \sa \ref segmentVolume
    */
    /// Generated from class `MR::VolumeSegmentationParameters`.
    /// This is the const half of the class.
    public class Const_VolumeSegmentationParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VolumeSegmentationParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_VolumeSegmentationParameters_Destroy(_Underlying *_this);
            __MR_VolumeSegmentationParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VolumeSegmentationParameters() {Dispose(false);}

        /// Exponent modifier of path building metric (paths are built between voxel pairs and then marked as tooth seed)
        public unsafe float BuildPathExponentModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_Get_buildPathExponentModifier", ExactSpelling = true)]
                extern static float *__MR_VolumeSegmentationParameters_Get_buildPathExponentModifier(_Underlying *_this);
                return *__MR_VolumeSegmentationParameters_Get_buildPathExponentModifier(_UnderlyingPtr);
            }
        }

        /// Exponent modifier of graph cutting metric (volume presents graph with seeds, this graph are min cut)
        public unsafe float SegmentationExponentModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_Get_segmentationExponentModifier", ExactSpelling = true)]
                extern static float *__MR_VolumeSegmentationParameters_Get_segmentationExponentModifier(_Underlying *_this);
                return *__MR_VolumeSegmentationParameters_Get_segmentationExponentModifier(_UnderlyingPtr);
            }
        }

        /// Segment box expansion (only part of volume are segmented, this parameter shows how much to expand this part)
        public unsafe int VoxelsExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_Get_voxelsExpansion", ExactSpelling = true)]
                extern static int *__MR_VolumeSegmentationParameters_Get_voxelsExpansion(_Underlying *_this);
                return *__MR_VolumeSegmentationParameters_Get_voxelsExpansion(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VolumeSegmentationParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_VolumeSegmentationParameters_DefaultConstruct();
        }

        /// Constructs `MR::VolumeSegmentationParameters` elementwise.
        public unsafe Const_VolumeSegmentationParameters(float buildPathExponentModifier, float segmentationExponentModifier, int voxelsExpansion) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_ConstructFrom(float buildPathExponentModifier, float segmentationExponentModifier, int voxelsExpansion);
            _UnderlyingPtr = __MR_VolumeSegmentationParameters_ConstructFrom(buildPathExponentModifier, segmentationExponentModifier, voxelsExpansion);
        }

        /// Generated from constructor `MR::VolumeSegmentationParameters::VolumeSegmentationParameters`.
        public unsafe Const_VolumeSegmentationParameters(MR.Const_VolumeSegmentationParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_ConstructFromAnother(MR.VolumeSegmentationParameters._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeSegmentationParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /**
    * \brief Parameters for volume segmentation
    * 
    * \sa \ref segmentVolume
    */
    /// Generated from class `MR::VolumeSegmentationParameters`.
    /// This is the non-const half of the class.
    public class VolumeSegmentationParameters : Const_VolumeSegmentationParameters
    {
        internal unsafe VolumeSegmentationParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Exponent modifier of path building metric (paths are built between voxel pairs and then marked as tooth seed)
        public new unsafe ref float BuildPathExponentModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_GetMutable_buildPathExponentModifier", ExactSpelling = true)]
                extern static float *__MR_VolumeSegmentationParameters_GetMutable_buildPathExponentModifier(_Underlying *_this);
                return ref *__MR_VolumeSegmentationParameters_GetMutable_buildPathExponentModifier(_UnderlyingPtr);
            }
        }

        /// Exponent modifier of graph cutting metric (volume presents graph with seeds, this graph are min cut)
        public new unsafe ref float SegmentationExponentModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_GetMutable_segmentationExponentModifier", ExactSpelling = true)]
                extern static float *__MR_VolumeSegmentationParameters_GetMutable_segmentationExponentModifier(_Underlying *_this);
                return ref *__MR_VolumeSegmentationParameters_GetMutable_segmentationExponentModifier(_UnderlyingPtr);
            }
        }

        /// Segment box expansion (only part of volume are segmented, this parameter shows how much to expand this part)
        public new unsafe ref int VoxelsExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_GetMutable_voxelsExpansion", ExactSpelling = true)]
                extern static int *__MR_VolumeSegmentationParameters_GetMutable_voxelsExpansion(_Underlying *_this);
                return ref *__MR_VolumeSegmentationParameters_GetMutable_voxelsExpansion(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VolumeSegmentationParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_VolumeSegmentationParameters_DefaultConstruct();
        }

        /// Constructs `MR::VolumeSegmentationParameters` elementwise.
        public unsafe VolumeSegmentationParameters(float buildPathExponentModifier, float segmentationExponentModifier, int voxelsExpansion) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_ConstructFrom(float buildPathExponentModifier, float segmentationExponentModifier, int voxelsExpansion);
            _UnderlyingPtr = __MR_VolumeSegmentationParameters_ConstructFrom(buildPathExponentModifier, segmentationExponentModifier, voxelsExpansion);
        }

        /// Generated from constructor `MR::VolumeSegmentationParameters::VolumeSegmentationParameters`.
        public unsafe VolumeSegmentationParameters(MR.Const_VolumeSegmentationParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_ConstructFromAnother(MR.VolumeSegmentationParameters._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeSegmentationParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VolumeSegmentationParameters::operator=`.
        public unsafe MR.VolumeSegmentationParameters Assign(MR.Const_VolumeSegmentationParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmentationParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VolumeSegmentationParameters._Underlying *__MR_VolumeSegmentationParameters_AssignFromAnother(_Underlying *_this, MR.VolumeSegmentationParameters._Underlying *_other);
            return new(__MR_VolumeSegmentationParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VolumeSegmentationParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VolumeSegmentationParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeSegmentationParameters`/`Const_VolumeSegmentationParameters` directly.
    public class _InOptMut_VolumeSegmentationParameters
    {
        public VolumeSegmentationParameters? Opt;

        public _InOptMut_VolumeSegmentationParameters() {}
        public _InOptMut_VolumeSegmentationParameters(VolumeSegmentationParameters value) {Opt = value;}
        public static implicit operator _InOptMut_VolumeSegmentationParameters(VolumeSegmentationParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `VolumeSegmentationParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VolumeSegmentationParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeSegmentationParameters`/`Const_VolumeSegmentationParameters` to pass it to the function.
    public class _InOptConst_VolumeSegmentationParameters
    {
        public Const_VolumeSegmentationParameters? Opt;

        public _InOptConst_VolumeSegmentationParameters() {}
        public _InOptConst_VolumeSegmentationParameters(Const_VolumeSegmentationParameters value) {Opt = value;}
        public static implicit operator _InOptConst_VolumeSegmentationParameters(Const_VolumeSegmentationParameters value) {return new(value);}
    }

    /**
    * \brief Class for voxels segmentation
    *
    * <table border=0> <caption id="VolumeSegmenter_examples"></caption>
    * <tr> <td> \image html voxel_segmentation/voxel_segmentation_0_0.png "Before (a)" width = 350cm </td>
    *      <td> \image html voxel_segmentation/voxel_segmentation_0_1.png "Before (b)" width = 350cm </td> </tr>
    *      <td> \image html voxel_segmentation/voxel_segmentation_0_2.png "After" width = 350cm </td> </tr>
    * </table>
    */
    /// Generated from class `MR::VolumeSegmenter`.
    /// This is the const half of the class.
    public class Const_VolumeSegmenter : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VolumeSegmenter(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_Destroy", ExactSpelling = true)]
            extern static void __MR_VolumeSegmenter_Destroy(_Underlying *_this);
            __MR_VolumeSegmenter_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VolumeSegmenter() {Dispose(false);}

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public unsafe Const_VolumeSegmenter(MR._ByValue_VolumeSegmenter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeSegmenter._Underlying *__MR_VolumeSegmenter_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VolumeSegmenter._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeSegmenter_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public unsafe Const_VolumeSegmenter(MR.Const_VdbVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_Construct", ExactSpelling = true)]
            extern static MR.VolumeSegmenter._Underlying *__MR_VolumeSegmenter_Construct(MR.Const_VdbVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VolumeSegmenter_Construct(volume._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public static unsafe implicit operator Const_VolumeSegmenter(MR.Const_VdbVolume volume) {return new(volume);}

        /// Return currently stored seeds
        /// Generated from method `MR::VolumeSegmenter::getSeeds`.
        public unsafe MR.Std.Const_Vector_MRVector3i GetSeeds(MR.VolumeSegmenter.SeedType seedType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_getSeeds", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRVector3i._Underlying *__MR_VolumeSegmenter_getSeeds(_Underlying *_this, MR.VolumeSegmenter.SeedType seedType);
            return new(__MR_VolumeSegmenter_getSeeds(_UnderlyingPtr, seedType), is_owning: false);
        }

        /// Returns mesh of given segment
        /// Generated from method `MR::VolumeSegmenter::createMeshFromSegmentation`.
        public unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> CreateMeshFromSegmentation(MR.Const_VoxelBitSet segmentation)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_createMeshFromSegmentation", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_VolumeSegmenter_createMeshFromSegmentation(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *segmentation);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_VolumeSegmenter_createMeshFromSegmentation(_UnderlyingPtr, segmentation._UnderlyingPtr), is_owning: true));
        }

        /// Dimensions of volume part, filled after segmentation
        /// Generated from method `MR::VolumeSegmenter::getVolumePartDimensions`.
        public unsafe MR.Const_Vector3i GetVolumePartDimensions()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_getVolumePartDimensions", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_VolumeSegmenter_getVolumePartDimensions(_Underlying *_this);
            return new(__MR_VolumeSegmenter_getVolumePartDimensions(_UnderlyingPtr), is_owning: false);
        }

        /// Min voxel of volume part box in whole volume space, filled after segmentation
        /// Generated from method `MR::VolumeSegmenter::getMinVoxel`.
        public unsafe MR.Const_Vector3i GetMinVoxel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_getMinVoxel", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_VolumeSegmenter_getMinVoxel(_Underlying *_this);
            return new(__MR_VolumeSegmenter_getMinVoxel(_UnderlyingPtr), is_owning: false);
        }

        public enum SeedType : int
        {
            Inside = 0,
            Outside = 1,
            Count = 2,
        }
    }

    /**
    * \brief Class for voxels segmentation
    *
    * <table border=0> <caption id="VolumeSegmenter_examples"></caption>
    * <tr> <td> \image html voxel_segmentation/voxel_segmentation_0_0.png "Before (a)" width = 350cm </td>
    *      <td> \image html voxel_segmentation/voxel_segmentation_0_1.png "Before (b)" width = 350cm </td> </tr>
    *      <td> \image html voxel_segmentation/voxel_segmentation_0_2.png "After" width = 350cm </td> </tr>
    * </table>
    */
    /// Generated from class `MR::VolumeSegmenter`.
    /// This is the non-const half of the class.
    public class VolumeSegmenter : Const_VolumeSegmenter
    {
        internal unsafe VolumeSegmenter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public unsafe VolumeSegmenter(MR._ByValue_VolumeSegmenter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeSegmenter._Underlying *__MR_VolumeSegmenter_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VolumeSegmenter._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeSegmenter_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public unsafe VolumeSegmenter(MR.Const_VdbVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_Construct", ExactSpelling = true)]
            extern static MR.VolumeSegmenter._Underlying *__MR_VolumeSegmenter_Construct(MR.Const_VdbVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VolumeSegmenter_Construct(volume._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public static unsafe implicit operator VolumeSegmenter(MR.Const_VdbVolume volume) {return new(volume);}

        /// Builds path with given parameters, marks result as seedType seeds
        /// Generated from method `MR::VolumeSegmenter::addPathSeeds`.
        /// Parameter `exponentModifier` defaults to `-1.0f`.
        public unsafe void AddPathSeeds(MR.Const_VoxelMetricParameters metricParameters, MR.VolumeSegmenter.SeedType seedType, float? exponentModifier = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_addPathSeeds", ExactSpelling = true)]
            extern static void __MR_VolumeSegmenter_addPathSeeds(_Underlying *_this, MR.Const_VoxelMetricParameters._Underlying *metricParameters, MR.VolumeSegmenter.SeedType seedType, float *exponentModifier);
            float __deref_exponentModifier = exponentModifier.GetValueOrDefault();
            __MR_VolumeSegmenter_addPathSeeds(_UnderlyingPtr, metricParameters._UnderlyingPtr, seedType, exponentModifier.HasValue ? &__deref_exponentModifier : null);
        }

        /// Reset seeds with given ones
        /// Generated from method `MR::VolumeSegmenter::setSeeds`.
        public unsafe void SetSeeds(MR.Std.Const_Vector_MRVector3i seeds, MR.VolumeSegmenter.SeedType seedType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_setSeeds", ExactSpelling = true)]
            extern static void __MR_VolumeSegmenter_setSeeds(_Underlying *_this, MR.Std.Const_Vector_MRVector3i._Underlying *seeds, MR.VolumeSegmenter.SeedType seedType);
            __MR_VolumeSegmenter_setSeeds(_UnderlyingPtr, seeds._UnderlyingPtr, seedType);
        }

        /// Adds new seeds to stored
        /// Generated from method `MR::VolumeSegmenter::addSeeds`.
        public unsafe void AddSeeds(MR.Std.Const_Vector_MRVector3i seeds, MR.VolumeSegmenter.SeedType seedType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_addSeeds", ExactSpelling = true)]
            extern static void __MR_VolumeSegmenter_addSeeds(_Underlying *_this, MR.Std.Const_Vector_MRVector3i._Underlying *seeds, MR.VolumeSegmenter.SeedType seedType);
            __MR_VolumeSegmenter_addSeeds(_UnderlyingPtr, seeds._UnderlyingPtr, seedType);
        }

        /// Segments volume, return inside part segmentation (VoxelBitSet in space of VolumePart)
        /// Generated from method `MR::VolumeSegmenter::segmentVolume`.
        /// Parameter `segmentationExponentModifier` defaults to `3000.0f`.
        /// Parameter `voxelsExpansion` defaults to `25`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_MRVoxelBitSet_StdString> SegmentVolume(float? segmentationExponentModifier = null, int? voxelsExpansion = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeSegmenter_segmentVolume", ExactSpelling = true)]
            extern static MR.Expected_MRVoxelBitSet_StdString._Underlying *__MR_VolumeSegmenter_segmentVolume(_Underlying *_this, float *segmentationExponentModifier, int *voxelsExpansion, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            float __deref_segmentationExponentModifier = segmentationExponentModifier.GetValueOrDefault();
            int __deref_voxelsExpansion = voxelsExpansion.GetValueOrDefault();
            return MR.Misc.Move(new MR.Expected_MRVoxelBitSet_StdString(__MR_VolumeSegmenter_segmentVolume(_UnderlyingPtr, segmentationExponentModifier.HasValue ? &__deref_segmentationExponentModifier : null, voxelsExpansion.HasValue ? &__deref_voxelsExpansion : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `VolumeSegmenter` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VolumeSegmenter`/`Const_VolumeSegmenter` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VolumeSegmenter
    {
        internal readonly Const_VolumeSegmenter? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VolumeSegmenter(Const_VolumeSegmenter new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VolumeSegmenter(Const_VolumeSegmenter arg) {return new(arg);}
        public _ByValue_VolumeSegmenter(MR.Misc._Moved<VolumeSegmenter> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VolumeSegmenter(MR.Misc._Moved<VolumeSegmenter> arg) {return new(arg);}

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public static unsafe implicit operator _ByValue_VolumeSegmenter(MR.Const_VdbVolume volume) {return new MR.VolumeSegmenter(volume);}
    }

    /// This is used for optional parameters of class `VolumeSegmenter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VolumeSegmenter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeSegmenter`/`Const_VolumeSegmenter` directly.
    public class _InOptMut_VolumeSegmenter
    {
        public VolumeSegmenter? Opt;

        public _InOptMut_VolumeSegmenter() {}
        public _InOptMut_VolumeSegmenter(VolumeSegmenter value) {Opt = value;}
        public static implicit operator _InOptMut_VolumeSegmenter(VolumeSegmenter value) {return new(value);}
    }

    /// This is used for optional parameters of class `VolumeSegmenter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VolumeSegmenter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeSegmenter`/`Const_VolumeSegmenter` to pass it to the function.
    public class _InOptConst_VolumeSegmenter
    {
        public Const_VolumeSegmenter? Opt;

        public _InOptConst_VolumeSegmenter() {}
        public _InOptConst_VolumeSegmenter(Const_VolumeSegmenter value) {Opt = value;}
        public static implicit operator _InOptConst_VolumeSegmenter(Const_VolumeSegmenter value) {return new(value);}

        /// Generated from constructor `MR::VolumeSegmenter::VolumeSegmenter`.
        public static unsafe implicit operator _InOptConst_VolumeSegmenter(MR.Const_VdbVolume volume) {return new MR.VolumeSegmenter(volume);}
    }

    /**
    * \brief Creates mesh from voxels mask
    * \param mask in space of whole volume
    *  density inside mask is expected to be higher then outside
    */
    /// Generated from function `MR::meshFromVoxelsMask`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MeshFromVoxelsMask(MR.Const_VdbVolume volume, MR.Const_VoxelBitSet mask)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshFromVoxelsMask", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_meshFromVoxelsMask(MR.Const_VdbVolume._Underlying *volume, MR.Const_VoxelBitSet._Underlying *mask);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_meshFromVoxelsMask(volume._UnderlyingPtr, mask._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief Simple segment volume
    * \details
    * 1. Build paths between points pairs \n
    * 2. Mark paths as inside part seeds \n
    * 3. Mark volume part edges as outside part seeds \n
    * 4. Return mesh from segmented inside part
    */
    /// Generated from function `MR::segmentVolume`.
    /// Parameter `params_` defaults to `MR::VolumeSegmentationParameters()`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> SegmentVolume(MR.Const_VdbVolume volume, MR.Std.Const_Vector_StdPairMRVector3fMRVector3f pairs, MR.Const_VolumeSegmentationParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_segmentVolume", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_segmentVolume(MR.Const_VdbVolume._Underlying *volume, MR.Std.Const_Vector_StdPairMRVector3fMRVector3f._Underlying *pairs, MR.Const_VolumeSegmentationParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_segmentVolume(volume._UnderlyingPtr, pairs._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Given voxel mask, separate it into components using mincut algorithm
    /// @param minSize Minimum size of a segment (in voxels)
    /// Generated from function `MR::segmentVoxelMaskToInstances`.
    /// Parameter `minSize` defaults to `100`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRMesh_StdString> SegmentVoxelMaskToInstances(MR.Const_VdbVolume mask, ulong? minSize = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_segmentVoxelMaskToInstances", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMRMesh_StdString._Underlying *__MR_segmentVoxelMaskToInstances(MR.Const_VdbVolume._Underlying *mask, ulong *minSize, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        ulong __deref_minSize = minSize.GetValueOrDefault();
        return MR.Misc.Move(new MR.Expected_StdVectorMRMesh_StdString(__MR_segmentVoxelMaskToInstances(mask._UnderlyingPtr, minSize.HasValue ? &__deref_minSize : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }
}
