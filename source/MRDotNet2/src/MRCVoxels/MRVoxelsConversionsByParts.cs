public static partial class MR
{
    /**
    * \struct MR::MergeVolumePartSettings
    * \brief Parameters' structure for MR::mergeVolumePart
    *
    *
    * \sa \ref mergeVolumePart
    */
    /// Generated from class `MR::MergeVolumePartSettings`.
    /// This is the const half of the class.
    public class Const_MergeVolumePartSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MergeVolumePartSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_MergeVolumePartSettings_Destroy(_Underlying *_this);
            __MR_MergeVolumePartSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MergeVolumePartSettings() {Dispose(false);}

        public unsafe MR.Std.Const_Function_VoidFuncFromMRMeshRefFloatFloat PreCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_Get_preCut", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *__MR_MergeVolumePartSettings_Get_preCut(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_Get_preCut(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Function_VoidFuncFromMRMeshRef PostCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_Get_postCut", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRMeshRef._Underlying *__MR_MergeVolumePartSettings_Get_postCut(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_Get_postCut(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Function_VoidFuncFromMRMeshRefConstMRPartMappingRef PostMerge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_Get_postMerge", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRMeshRefConstMRPartMappingRef._Underlying *__MR_MergeVolumePartSettings_Get_postMerge(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_Get_postMerge(_UnderlyingPtr), is_owning: false);
            }
        }

        /// mapping with initialized maps required for the `postMerge` callback
        public unsafe MR.Const_PartMapping Mapping
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_Get_mapping", ExactSpelling = true)]
                extern static MR.Const_PartMapping._Underlying *__MR_MergeVolumePartSettings_Get_mapping(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_Get_mapping(_UnderlyingPtr), is_owning: false);
            }
        }

        /// origin (position of the (0;0;0) voxel) of the voxel volume part, usually specified for SimpleVolume
        public unsafe MR.Const_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_Get_origin", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MergeVolumePartSettings_Get_origin(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_Get_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MergeVolumePartSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MergeVolumePartSettings_DefaultConstruct();
        }

        /// Constructs `MR::MergeVolumePartSettings` elementwise.
        public unsafe Const_MergeVolumePartSettings(MR.Std._ByValue_Function_VoidFuncFromMRMeshRefFloatFloat preCut, MR.Std._ByValue_Function_VoidFuncFromMRMeshRef postCut, MR.Std._ByValue_Function_VoidFuncFromMRMeshRefConstMRPartMappingRef postMerge, MR.Const_PartMapping mapping, MR.Vector3f origin) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_ConstructFrom(MR.Misc._PassBy preCut_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *preCut, MR.Misc._PassBy postCut_pass_by, MR.Std.Function_VoidFuncFromMRMeshRef._Underlying *postCut, MR.Misc._PassBy postMerge_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefConstMRPartMappingRef._Underlying *postMerge, MR.PartMapping._Underlying *mapping, MR.Vector3f origin);
            _UnderlyingPtr = __MR_MergeVolumePartSettings_ConstructFrom(preCut.PassByMode, preCut.Value is not null ? preCut.Value._UnderlyingPtr : null, postCut.PassByMode, postCut.Value is not null ? postCut.Value._UnderlyingPtr : null, postMerge.PassByMode, postMerge.Value is not null ? postMerge.Value._UnderlyingPtr : null, mapping._UnderlyingPtr, origin);
        }

        /// Generated from constructor `MR::MergeVolumePartSettings::MergeVolumePartSettings`.
        public unsafe Const_MergeVolumePartSettings(MR._ByValue_MergeVolumePartSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MergeVolumePartSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MergeVolumePartSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * \struct MR::MergeVolumePartSettings
    * \brief Parameters' structure for MR::mergeVolumePart
    *
    *
    * \sa \ref mergeVolumePart
    */
    /// Generated from class `MR::MergeVolumePartSettings`.
    /// This is the non-const half of the class.
    public class MergeVolumePartSettings : Const_MergeVolumePartSettings
    {
        internal unsafe MergeVolumePartSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat PreCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_GetMutable_preCut", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *__MR_MergeVolumePartSettings_GetMutable_preCut(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_GetMutable_preCut(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Function_VoidFuncFromMRMeshRef PostCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_GetMutable_postCut", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRef._Underlying *__MR_MergeVolumePartSettings_GetMutable_postCut(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_GetMutable_postCut(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Function_VoidFuncFromMRMeshRefConstMRPartMappingRef PostMerge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_GetMutable_postMerge", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRefConstMRPartMappingRef._Underlying *__MR_MergeVolumePartSettings_GetMutable_postMerge(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_GetMutable_postMerge(_UnderlyingPtr), is_owning: false);
            }
        }

        /// mapping with initialized maps required for the `postMerge` callback
        public new unsafe MR.PartMapping Mapping
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_GetMutable_mapping", ExactSpelling = true)]
                extern static MR.PartMapping._Underlying *__MR_MergeVolumePartSettings_GetMutable_mapping(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_GetMutable_mapping(_UnderlyingPtr), is_owning: false);
            }
        }

        /// origin (position of the (0;0;0) voxel) of the voxel volume part, usually specified for SimpleVolume
        public new unsafe MR.Mut_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_GetMutable_origin", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MergeVolumePartSettings_GetMutable_origin(_Underlying *_this);
                return new(__MR_MergeVolumePartSettings_GetMutable_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MergeVolumePartSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MergeVolumePartSettings_DefaultConstruct();
        }

        /// Constructs `MR::MergeVolumePartSettings` elementwise.
        public unsafe MergeVolumePartSettings(MR.Std._ByValue_Function_VoidFuncFromMRMeshRefFloatFloat preCut, MR.Std._ByValue_Function_VoidFuncFromMRMeshRef postCut, MR.Std._ByValue_Function_VoidFuncFromMRMeshRefConstMRPartMappingRef postMerge, MR.Const_PartMapping mapping, MR.Vector3f origin) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_ConstructFrom(MR.Misc._PassBy preCut_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *preCut, MR.Misc._PassBy postCut_pass_by, MR.Std.Function_VoidFuncFromMRMeshRef._Underlying *postCut, MR.Misc._PassBy postMerge_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefConstMRPartMappingRef._Underlying *postMerge, MR.PartMapping._Underlying *mapping, MR.Vector3f origin);
            _UnderlyingPtr = __MR_MergeVolumePartSettings_ConstructFrom(preCut.PassByMode, preCut.Value is not null ? preCut.Value._UnderlyingPtr : null, postCut.PassByMode, postCut.Value is not null ? postCut.Value._UnderlyingPtr : null, postMerge.PassByMode, postMerge.Value is not null ? postMerge.Value._UnderlyingPtr : null, mapping._UnderlyingPtr, origin);
        }

        /// Generated from constructor `MR::MergeVolumePartSettings::MergeVolumePartSettings`.
        public unsafe MergeVolumePartSettings(MR._ByValue_MergeVolumePartSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MergeVolumePartSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MergeVolumePartSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MergeVolumePartSettings::operator=`.
        public unsafe MR.MergeVolumePartSettings Assign(MR._ByValue_MergeVolumePartSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MergeVolumePartSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MergeVolumePartSettings._Underlying *__MR_MergeVolumePartSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MergeVolumePartSettings._Underlying *_other);
            return new(__MR_MergeVolumePartSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MergeVolumePartSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MergeVolumePartSettings`/`Const_MergeVolumePartSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MergeVolumePartSettings
    {
        internal readonly Const_MergeVolumePartSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MergeVolumePartSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MergeVolumePartSettings(Const_MergeVolumePartSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MergeVolumePartSettings(Const_MergeVolumePartSettings arg) {return new(arg);}
        public _ByValue_MergeVolumePartSettings(MR.Misc._Moved<MergeVolumePartSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MergeVolumePartSettings(MR.Misc._Moved<MergeVolumePartSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MergeVolumePartSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MergeVolumePartSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MergeVolumePartSettings`/`Const_MergeVolumePartSettings` directly.
    public class _InOptMut_MergeVolumePartSettings
    {
        public MergeVolumePartSettings? Opt;

        public _InOptMut_MergeVolumePartSettings() {}
        public _InOptMut_MergeVolumePartSettings(MergeVolumePartSettings value) {Opt = value;}
        public static implicit operator _InOptMut_MergeVolumePartSettings(MergeVolumePartSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `MergeVolumePartSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MergeVolumePartSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MergeVolumePartSettings`/`Const_MergeVolumePartSettings` to pass it to the function.
    public class _InOptConst_MergeVolumePartSettings
    {
        public Const_MergeVolumePartSettings? Opt;

        public _InOptConst_MergeVolumePartSettings() {}
        public _InOptConst_MergeVolumePartSettings(Const_MergeVolumePartSettings value) {Opt = value;}
        public static implicit operator _InOptConst_MergeVolumePartSettings(Const_MergeVolumePartSettings value) {return new(value);}
    }

    /**
    * \struct MR::VolumeToMeshByPartsSettings
    * \brief Parameters' structure for MR::volumeToMeshByParts
    *
    *
    * \sa \ref volumeToMeshByParts
    */
    /// Generated from class `MR::VolumeToMeshByPartsSettings`.
    /// This is the const half of the class.
    public class Const_VolumeToMeshByPartsSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VolumeToMeshByPartsSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_VolumeToMeshByPartsSettings_Destroy(_Underlying *_this);
            __MR_VolumeToMeshByPartsSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VolumeToMeshByPartsSettings() {Dispose(false);}

        // 256 MiB
        public unsafe ulong MaxVolumePartMemoryUsage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_Get_maxVolumePartMemoryUsage", ExactSpelling = true)]
                extern static ulong *__MR_VolumeToMeshByPartsSettings_Get_maxVolumePartMemoryUsage(_Underlying *_this);
                return *__MR_VolumeToMeshByPartsSettings_Get_maxVolumePartMemoryUsage(_UnderlyingPtr);
            }
        }

        /// overlap in voxels between two parts
        public unsafe ulong StripeOverlap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_Get_stripeOverlap", ExactSpelling = true)]
                extern static ulong *__MR_VolumeToMeshByPartsSettings_Get_stripeOverlap(_Underlying *_this);
                return *__MR_VolumeToMeshByPartsSettings_Get_stripeOverlap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VolumeToMeshByPartsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_VolumeToMeshByPartsSettings_DefaultConstruct();
        }

        /// Constructs `MR::VolumeToMeshByPartsSettings` elementwise.
        public unsafe Const_VolumeToMeshByPartsSettings(ulong maxVolumePartMemoryUsage, ulong stripeOverlap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_ConstructFrom(ulong maxVolumePartMemoryUsage, ulong stripeOverlap);
            _UnderlyingPtr = __MR_VolumeToMeshByPartsSettings_ConstructFrom(maxVolumePartMemoryUsage, stripeOverlap);
        }

        /// Generated from constructor `MR::VolumeToMeshByPartsSettings::VolumeToMeshByPartsSettings`.
        public unsafe Const_VolumeToMeshByPartsSettings(MR.Const_VolumeToMeshByPartsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_ConstructFromAnother(MR.VolumeToMeshByPartsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeToMeshByPartsSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /**
    * \struct MR::VolumeToMeshByPartsSettings
    * \brief Parameters' structure for MR::volumeToMeshByParts
    *
    *
    * \sa \ref volumeToMeshByParts
    */
    /// Generated from class `MR::VolumeToMeshByPartsSettings`.
    /// This is the non-const half of the class.
    public class VolumeToMeshByPartsSettings : Const_VolumeToMeshByPartsSettings
    {
        internal unsafe VolumeToMeshByPartsSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // 256 MiB
        public new unsafe ref ulong MaxVolumePartMemoryUsage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_GetMutable_maxVolumePartMemoryUsage", ExactSpelling = true)]
                extern static ulong *__MR_VolumeToMeshByPartsSettings_GetMutable_maxVolumePartMemoryUsage(_Underlying *_this);
                return ref *__MR_VolumeToMeshByPartsSettings_GetMutable_maxVolumePartMemoryUsage(_UnderlyingPtr);
            }
        }

        /// overlap in voxels between two parts
        public new unsafe ref ulong StripeOverlap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_GetMutable_stripeOverlap", ExactSpelling = true)]
                extern static ulong *__MR_VolumeToMeshByPartsSettings_GetMutable_stripeOverlap(_Underlying *_this);
                return ref *__MR_VolumeToMeshByPartsSettings_GetMutable_stripeOverlap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VolumeToMeshByPartsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_VolumeToMeshByPartsSettings_DefaultConstruct();
        }

        /// Constructs `MR::VolumeToMeshByPartsSettings` elementwise.
        public unsafe VolumeToMeshByPartsSettings(ulong maxVolumePartMemoryUsage, ulong stripeOverlap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_ConstructFrom(ulong maxVolumePartMemoryUsage, ulong stripeOverlap);
            _UnderlyingPtr = __MR_VolumeToMeshByPartsSettings_ConstructFrom(maxVolumePartMemoryUsage, stripeOverlap);
        }

        /// Generated from constructor `MR::VolumeToMeshByPartsSettings::VolumeToMeshByPartsSettings`.
        public unsafe VolumeToMeshByPartsSettings(MR.Const_VolumeToMeshByPartsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_ConstructFromAnother(MR.VolumeToMeshByPartsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeToMeshByPartsSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VolumeToMeshByPartsSettings::operator=`.
        public unsafe MR.VolumeToMeshByPartsSettings Assign(MR.Const_VolumeToMeshByPartsSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeToMeshByPartsSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VolumeToMeshByPartsSettings._Underlying *__MR_VolumeToMeshByPartsSettings_AssignFromAnother(_Underlying *_this, MR.VolumeToMeshByPartsSettings._Underlying *_other);
            return new(__MR_VolumeToMeshByPartsSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VolumeToMeshByPartsSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VolumeToMeshByPartsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeToMeshByPartsSettings`/`Const_VolumeToMeshByPartsSettings` directly.
    public class _InOptMut_VolumeToMeshByPartsSettings
    {
        public VolumeToMeshByPartsSettings? Opt;

        public _InOptMut_VolumeToMeshByPartsSettings() {}
        public _InOptMut_VolumeToMeshByPartsSettings(VolumeToMeshByPartsSettings value) {Opt = value;}
        public static implicit operator _InOptMut_VolumeToMeshByPartsSettings(VolumeToMeshByPartsSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `VolumeToMeshByPartsSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VolumeToMeshByPartsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeToMeshByPartsSettings`/`Const_VolumeToMeshByPartsSettings` to pass it to the function.
    public class _InOptConst_VolumeToMeshByPartsSettings
    {
        public Const_VolumeToMeshByPartsSettings? Opt;

        public _InOptConst_VolumeToMeshByPartsSettings() {}
        public _InOptConst_VolumeToMeshByPartsSettings(Const_VolumeToMeshByPartsSettings value) {Opt = value;}
        public static implicit operator _InOptConst_VolumeToMeshByPartsSettings(Const_VolumeToMeshByPartsSettings value) {return new(value);}
    }
}
