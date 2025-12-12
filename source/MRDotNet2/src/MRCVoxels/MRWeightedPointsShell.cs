public static partial class MR
{
    public static partial class WeightedShell
    {
        /// Generated from class `MR::WeightedShell::DistanceVolumeCreationParams`.
        /// This is the const half of the class.
        public class Const_DistanceVolumeCreationParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_DistanceVolumeCreationParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_Destroy", ExactSpelling = true)]
                extern static void __MR_WeightedShell_DistanceVolumeCreationParams_Destroy(_Underlying *_this);
                __MR_WeightedShell_DistanceVolumeCreationParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_DistanceVolumeCreationParams() {Dispose(false);}

            public unsafe MR.Const_DistanceVolumeParams Vol
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_Get_vol", ExactSpelling = true)]
                    extern static MR.Const_DistanceVolumeParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_Get_vol(_Underlying *_this);
                    return new(__MR_WeightedShell_DistanceVolumeCreationParams_Get_vol(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_DistanceFromWeightedPointsComputeParams Dist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_Get_dist", ExactSpelling = true)]
                    extern static MR.Const_DistanceFromWeightedPointsComputeParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_Get_dist(_Underlying *_this);
                    return new(__MR_WeightedShell_DistanceVolumeCreationParams_Get_dist(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_DistanceVolumeCreationParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_DistanceVolumeCreationParams_DefaultConstruct();
            }

            /// Constructs `MR::WeightedShell::DistanceVolumeCreationParams` elementwise.
            public unsafe Const_DistanceVolumeCreationParams(MR._ByValue_DistanceVolumeParams vol, MR._ByValue_DistanceFromWeightedPointsComputeParams dist) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.Misc._PassBy dist_pass_by, MR.DistanceFromWeightedPointsComputeParams._Underlying *dist);
                _UnderlyingPtr = __MR_WeightedShell_DistanceVolumeCreationParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist.PassByMode, dist.Value is not null ? dist.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::WeightedShell::DistanceVolumeCreationParams::DistanceVolumeCreationParams`.
            public unsafe Const_DistanceVolumeCreationParams(MR.WeightedShell._ByValue_DistanceVolumeCreationParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.DistanceVolumeCreationParams._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_DistanceVolumeCreationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::WeightedShell::DistanceVolumeCreationParams`.
        /// This is the non-const half of the class.
        public class DistanceVolumeCreationParams : Const_DistanceVolumeCreationParams
        {
            internal unsafe DistanceVolumeCreationParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.DistanceVolumeParams Vol
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_GetMutable_vol", ExactSpelling = true)]
                    extern static MR.DistanceVolumeParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_GetMutable_vol(_Underlying *_this);
                    return new(__MR_WeightedShell_DistanceVolumeCreationParams_GetMutable_vol(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.DistanceFromWeightedPointsComputeParams Dist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_GetMutable_dist", ExactSpelling = true)]
                    extern static MR.DistanceFromWeightedPointsComputeParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_GetMutable_dist(_Underlying *_this);
                    return new(__MR_WeightedShell_DistanceVolumeCreationParams_GetMutable_dist(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe DistanceVolumeCreationParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_DistanceVolumeCreationParams_DefaultConstruct();
            }

            /// Constructs `MR::WeightedShell::DistanceVolumeCreationParams` elementwise.
            public unsafe DistanceVolumeCreationParams(MR._ByValue_DistanceVolumeParams vol, MR._ByValue_DistanceFromWeightedPointsComputeParams dist) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.Misc._PassBy dist_pass_by, MR.DistanceFromWeightedPointsComputeParams._Underlying *dist);
                _UnderlyingPtr = __MR_WeightedShell_DistanceVolumeCreationParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist.PassByMode, dist.Value is not null ? dist.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::WeightedShell::DistanceVolumeCreationParams::DistanceVolumeCreationParams`.
            public unsafe DistanceVolumeCreationParams(MR.WeightedShell._ByValue_DistanceVolumeCreationParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.DistanceVolumeCreationParams._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_DistanceVolumeCreationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::WeightedShell::DistanceVolumeCreationParams::operator=`.
            public unsafe MR.WeightedShell.DistanceVolumeCreationParams Assign(MR.WeightedShell._ByValue_DistanceVolumeCreationParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_DistanceVolumeCreationParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.DistanceVolumeCreationParams._Underlying *__MR_WeightedShell_DistanceVolumeCreationParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WeightedShell.DistanceVolumeCreationParams._Underlying *_other);
                return new(__MR_WeightedShell_DistanceVolumeCreationParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `DistanceVolumeCreationParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `DistanceVolumeCreationParams`/`Const_DistanceVolumeCreationParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_DistanceVolumeCreationParams
        {
            internal readonly Const_DistanceVolumeCreationParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_DistanceVolumeCreationParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_DistanceVolumeCreationParams(Const_DistanceVolumeCreationParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_DistanceVolumeCreationParams(Const_DistanceVolumeCreationParams arg) {return new(arg);}
            public _ByValue_DistanceVolumeCreationParams(MR.Misc._Moved<DistanceVolumeCreationParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_DistanceVolumeCreationParams(MR.Misc._Moved<DistanceVolumeCreationParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `DistanceVolumeCreationParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceVolumeCreationParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DistanceVolumeCreationParams`/`Const_DistanceVolumeCreationParams` directly.
        public class _InOptMut_DistanceVolumeCreationParams
        {
            public DistanceVolumeCreationParams? Opt;

            public _InOptMut_DistanceVolumeCreationParams() {}
            public _InOptMut_DistanceVolumeCreationParams(DistanceVolumeCreationParams value) {Opt = value;}
            public static implicit operator _InOptMut_DistanceVolumeCreationParams(DistanceVolumeCreationParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `DistanceVolumeCreationParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceVolumeCreationParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DistanceVolumeCreationParams`/`Const_DistanceVolumeCreationParams` to pass it to the function.
        public class _InOptConst_DistanceVolumeCreationParams
        {
            public Const_DistanceVolumeCreationParams? Opt;

            public _InOptConst_DistanceVolumeCreationParams() {}
            public _InOptConst_DistanceVolumeCreationParams(Const_DistanceVolumeCreationParams value) {Opt = value;}
            public static implicit operator _InOptConst_DistanceVolumeCreationParams(Const_DistanceVolumeCreationParams value) {return new(value);}
        }

        /// Generated from class `MR::WeightedShell::ParametersBase`.
        /// Derived classes:
        ///   Direct: (non-virtual)
        ///     `MR::WeightedShell::ParametersMetric`
        ///     `MR::WeightedShell::ParametersRegions`
        /// This is the const half of the class.
        public class Const_ParametersBase : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ParametersBase(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_Destroy", ExactSpelling = true)]
                extern static void __MR_WeightedShell_ParametersBase_Destroy(_Underlying *_this);
                __MR_WeightedShell_ParametersBase_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ParametersBase() {Dispose(false);}

            /// build iso-surface of minimal distance to points corresponding to this value
            public unsafe float Offset
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_Get_offset", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersBase_Get_offset(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersBase_Get_offset(_UnderlyingPtr);
                }
            }

            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public unsafe float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_Get_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersBase_Get_voxelSize(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersBase_Get_voxelSize(_UnderlyingPtr);
                }
            }

            /// number of voxels to compute near the offset (should be left default unless used for debugging)
            public unsafe float NumLayers
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_Get_numLayers", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersBase_Get_numLayers(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersBase_Get_numLayers(_UnderlyingPtr);
                }
            }

            /// Progress callback
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_WeightedShell_ParametersBase_Get_progress(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersBase_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ParametersBase() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_ParametersBase_DefaultConstruct();
            }

            /// Constructs `MR::WeightedShell::ParametersBase` elementwise.
            public unsafe Const_ParametersBase(float offset, float voxelSize, float numLayers, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_ConstructFrom", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_ConstructFrom(float offset, float voxelSize, float numLayers, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                _UnderlyingPtr = __MR_WeightedShell_ParametersBase_ConstructFrom(offset, voxelSize, numLayers, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::WeightedShell::ParametersBase::ParametersBase`.
            public unsafe Const_ParametersBase(MR.WeightedShell._ByValue_ParametersBase _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersBase._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_ParametersBase_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::WeightedShell::ParametersBase`.
        /// Derived classes:
        ///   Direct: (non-virtual)
        ///     `MR::WeightedShell::ParametersMetric`
        ///     `MR::WeightedShell::ParametersRegions`
        /// This is the non-const half of the class.
        public class ParametersBase : Const_ParametersBase
        {
            internal unsafe ParametersBase(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// build iso-surface of minimal distance to points corresponding to this value
            public new unsafe ref float Offset
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_GetMutable_offset", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersBase_GetMutable_offset(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersBase_GetMutable_offset(_UnderlyingPtr);
                }
            }

            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public new unsafe ref float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_GetMutable_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersBase_GetMutable_voxelSize(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersBase_GetMutable_voxelSize(_UnderlyingPtr);
                }
            }

            /// number of voxels to compute near the offset (should be left default unless used for debugging)
            public new unsafe ref float NumLayers
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_GetMutable_numLayers", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersBase_GetMutable_numLayers(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersBase_GetMutable_numLayers(_UnderlyingPtr);
                }
            }

            /// Progress callback
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_WeightedShell_ParametersBase_GetMutable_progress(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersBase_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ParametersBase() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_ParametersBase_DefaultConstruct();
            }

            /// Constructs `MR::WeightedShell::ParametersBase` elementwise.
            public unsafe ParametersBase(float offset, float voxelSize, float numLayers, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_ConstructFrom", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_ConstructFrom(float offset, float voxelSize, float numLayers, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                _UnderlyingPtr = __MR_WeightedShell_ParametersBase_ConstructFrom(offset, voxelSize, numLayers, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::WeightedShell::ParametersBase::ParametersBase`.
            public unsafe ParametersBase(MR.WeightedShell._ByValue_ParametersBase _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersBase._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_ParametersBase_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::WeightedShell::ParametersBase::operator=`.
            public unsafe MR.WeightedShell.ParametersBase Assign(MR.WeightedShell._ByValue_ParametersBase _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersBase_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersBase_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersBase._Underlying *_other);
                return new(__MR_WeightedShell_ParametersBase_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ParametersBase` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ParametersBase`/`Const_ParametersBase` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ParametersBase
        {
            internal readonly Const_ParametersBase? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ParametersBase() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ParametersBase(Const_ParametersBase new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ParametersBase(Const_ParametersBase arg) {return new(arg);}
            public _ByValue_ParametersBase(MR.Misc._Moved<ParametersBase> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ParametersBase(MR.Misc._Moved<ParametersBase> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ParametersBase` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ParametersBase`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ParametersBase`/`Const_ParametersBase` directly.
        public class _InOptMut_ParametersBase
        {
            public ParametersBase? Opt;

            public _InOptMut_ParametersBase() {}
            public _InOptMut_ParametersBase(ParametersBase value) {Opt = value;}
            public static implicit operator _InOptMut_ParametersBase(ParametersBase value) {return new(value);}
        }

        /// This is used for optional parameters of class `ParametersBase` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ParametersBase`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ParametersBase`/`Const_ParametersBase` to pass it to the function.
        public class _InOptConst_ParametersBase
        {
            public Const_ParametersBase? Opt;

            public _InOptConst_ParametersBase() {}
            public _InOptConst_ParametersBase(Const_ParametersBase value) {Opt = value;}
            public static implicit operator _InOptConst_ParametersBase(Const_ParametersBase value) {return new(value);}
        }

        /// Generated from class `MR::WeightedShell::ParametersMetric`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::WeightedShell::ParametersBase`
        /// This is the const half of the class.
        public class Const_ParametersMetric : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ParametersMetric(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_Destroy", ExactSpelling = true)]
                extern static void __MR_WeightedShell_ParametersMetric_Destroy(_Underlying *_this);
                __MR_WeightedShell_ParametersMetric_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ParametersMetric() {Dispose(false);}

            // Upcasts:
            public static unsafe implicit operator MR.WeightedShell.Const_ParametersBase(Const_ParametersMetric self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_UpcastTo_MR_WeightedShell_ParametersBase", ExactSpelling = true)]
                extern static MR.WeightedShell.Const_ParametersBase._Underlying *__MR_WeightedShell_ParametersMetric_UpcastTo_MR_WeightedShell_ParametersBase(_Underlying *_this);
                MR.WeightedShell.Const_ParametersBase ret = new(__MR_WeightedShell_ParametersMetric_UpcastTo_MR_WeightedShell_ParametersBase(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            /// parameters of distance finding
            public unsafe MR.Const_DistanceFromWeightedPointsParams Dist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_Get_dist", ExactSpelling = true)]
                    extern static MR.Const_DistanceFromWeightedPointsParams._Underlying *__MR_WeightedShell_ParametersMetric_Get_dist(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersMetric_Get_dist(_UnderlyingPtr), is_owning: false);
                }
            }

            /// build iso-surface of minimal distance to points corresponding to this value
            public unsafe float Offset
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_Get_offset", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersMetric_Get_offset(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersMetric_Get_offset(_UnderlyingPtr);
                }
            }

            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public unsafe float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_Get_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersMetric_Get_voxelSize(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersMetric_Get_voxelSize(_UnderlyingPtr);
                }
            }

            /// number of voxels to compute near the offset (should be left default unless used for debugging)
            public unsafe float NumLayers
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_Get_numLayers", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersMetric_Get_numLayers(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersMetric_Get_numLayers(_UnderlyingPtr);
                }
            }

            /// Progress callback
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_WeightedShell_ParametersMetric_Get_progress(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersMetric_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ParametersMetric() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersMetric._Underlying *__MR_WeightedShell_ParametersMetric_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_ParametersMetric_DefaultConstruct();
            }

            /// Generated from constructor `MR::WeightedShell::ParametersMetric::ParametersMetric`.
            public unsafe Const_ParametersMetric(MR.WeightedShell._ByValue_ParametersMetric _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersMetric._Underlying *__MR_WeightedShell_ParametersMetric_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersMetric._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_ParametersMetric_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::WeightedShell::ParametersMetric`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::WeightedShell::ParametersBase`
        /// This is the non-const half of the class.
        public class ParametersMetric : Const_ParametersMetric
        {
            internal unsafe ParametersMetric(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Upcasts:
            public static unsafe implicit operator MR.WeightedShell.ParametersBase(ParametersMetric self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_UpcastTo_MR_WeightedShell_ParametersBase", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersMetric_UpcastTo_MR_WeightedShell_ParametersBase(_Underlying *_this);
                MR.WeightedShell.ParametersBase ret = new(__MR_WeightedShell_ParametersMetric_UpcastTo_MR_WeightedShell_ParametersBase(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            /// parameters of distance finding
            public new unsafe MR.DistanceFromWeightedPointsParams Dist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_GetMutable_dist", ExactSpelling = true)]
                    extern static MR.DistanceFromWeightedPointsParams._Underlying *__MR_WeightedShell_ParametersMetric_GetMutable_dist(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersMetric_GetMutable_dist(_UnderlyingPtr), is_owning: false);
                }
            }

            /// build iso-surface of minimal distance to points corresponding to this value
            public new unsafe ref float Offset
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_GetMutable_offset", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersMetric_GetMutable_offset(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersMetric_GetMutable_offset(_UnderlyingPtr);
                }
            }

            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public new unsafe ref float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_GetMutable_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersMetric_GetMutable_voxelSize(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersMetric_GetMutable_voxelSize(_UnderlyingPtr);
                }
            }

            /// number of voxels to compute near the offset (should be left default unless used for debugging)
            public new unsafe ref float NumLayers
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_GetMutable_numLayers", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersMetric_GetMutable_numLayers(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersMetric_GetMutable_numLayers(_UnderlyingPtr);
                }
            }

            /// Progress callback
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_WeightedShell_ParametersMetric_GetMutable_progress(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersMetric_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ParametersMetric() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersMetric._Underlying *__MR_WeightedShell_ParametersMetric_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_ParametersMetric_DefaultConstruct();
            }

            /// Generated from constructor `MR::WeightedShell::ParametersMetric::ParametersMetric`.
            public unsafe ParametersMetric(MR.WeightedShell._ByValue_ParametersMetric _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersMetric._Underlying *__MR_WeightedShell_ParametersMetric_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersMetric._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_ParametersMetric_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::WeightedShell::ParametersMetric::operator=`.
            public unsafe MR.WeightedShell.ParametersMetric Assign(MR.WeightedShell._ByValue_ParametersMetric _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersMetric_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersMetric._Underlying *__MR_WeightedShell_ParametersMetric_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersMetric._Underlying *_other);
                return new(__MR_WeightedShell_ParametersMetric_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ParametersMetric` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ParametersMetric`/`Const_ParametersMetric` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ParametersMetric
        {
            internal readonly Const_ParametersMetric? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ParametersMetric() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ParametersMetric(Const_ParametersMetric new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ParametersMetric(Const_ParametersMetric arg) {return new(arg);}
            public _ByValue_ParametersMetric(MR.Misc._Moved<ParametersMetric> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ParametersMetric(MR.Misc._Moved<ParametersMetric> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ParametersMetric` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ParametersMetric`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ParametersMetric`/`Const_ParametersMetric` directly.
        public class _InOptMut_ParametersMetric
        {
            public ParametersMetric? Opt;

            public _InOptMut_ParametersMetric() {}
            public _InOptMut_ParametersMetric(ParametersMetric value) {Opt = value;}
            public static implicit operator _InOptMut_ParametersMetric(ParametersMetric value) {return new(value);}
        }

        /// This is used for optional parameters of class `ParametersMetric` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ParametersMetric`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ParametersMetric`/`Const_ParametersMetric` to pass it to the function.
        public class _InOptConst_ParametersMetric
        {
            public Const_ParametersMetric? Opt;

            public _InOptConst_ParametersMetric() {}
            public _InOptConst_ParametersMetric(Const_ParametersMetric value) {Opt = value;}
            public static implicit operator _InOptConst_ParametersMetric(Const_ParametersMetric value) {return new(value);}
        }

        /// Generated from class `MR::WeightedShell::ParametersRegions`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::WeightedShell::ParametersBase`
        /// This is the const half of the class.
        public class Const_ParametersRegions : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ParametersRegions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Destroy", ExactSpelling = true)]
                extern static void __MR_WeightedShell_ParametersRegions_Destroy(_Underlying *_this);
                __MR_WeightedShell_ParametersRegions_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ParametersRegions() {Dispose(false);}

            // Upcasts:
            public static unsafe implicit operator MR.WeightedShell.Const_ParametersBase(Const_ParametersRegions self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_UpcastTo_MR_WeightedShell_ParametersBase", ExactSpelling = true)]
                extern static MR.WeightedShell.Const_ParametersBase._Underlying *__MR_WeightedShell_ParametersRegions_UpcastTo_MR_WeightedShell_ParametersBase(_Underlying *_this);
                MR.WeightedShell.Const_ParametersBase ret = new(__MR_WeightedShell_ParametersRegions_UpcastTo_MR_WeightedShell_ParametersBase(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            /// list of regions (overlappings are allowed) with corresponding offsets
            /// the additional offset in overlaps is set to the average of the regions
            public unsafe MR.Std.Const_Vector_MRWeightedShellParametersRegionsRegion Regions
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_regions", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRWeightedShellParametersRegionsRegion._Underlying *__MR_WeightedShell_ParametersRegions_Get_regions(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersRegions_Get_regions(_UnderlyingPtr), is_owning: false);
                }
            }

            /// interpolation distance between the weights of the regions
            /// determines the sharpness of transitions between different regions
            public unsafe float InterpolationDist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_interpolationDist", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_Get_interpolationDist(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersRegions_Get_interpolationDist(_UnderlyingPtr);
                }
            }

            /// if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
            /// if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
            public unsafe bool BidirectionalMode
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_bidirectionalMode", ExactSpelling = true)]
                    extern static bool *__MR_WeightedShell_ParametersRegions_Get_bidirectionalMode(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersRegions_Get_bidirectionalMode(_UnderlyingPtr);
                }
            }

            /// build iso-surface of minimal distance to points corresponding to this value
            public unsafe float Offset
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_offset", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_Get_offset(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersRegions_Get_offset(_UnderlyingPtr);
                }
            }

            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public unsafe float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_Get_voxelSize(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersRegions_Get_voxelSize(_UnderlyingPtr);
                }
            }

            /// number of voxels to compute near the offset (should be left default unless used for debugging)
            public unsafe float NumLayers
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_numLayers", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_Get_numLayers(_Underlying *_this);
                    return *__MR_WeightedShell_ParametersRegions_Get_numLayers(_UnderlyingPtr);
                }
            }

            /// Progress callback
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_WeightedShell_ParametersRegions_Get_progress(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersRegions_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ParametersRegions() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersRegions._Underlying *__MR_WeightedShell_ParametersRegions_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_DefaultConstruct();
            }

            /// Generated from constructor `MR::WeightedShell::ParametersRegions::ParametersRegions`.
            public unsafe Const_ParametersRegions(MR.WeightedShell._ByValue_ParametersRegions _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersRegions._Underlying *__MR_WeightedShell_ParametersRegions_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersRegions._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from class `MR::WeightedShell::ParametersRegions::Region`.
            /// This is the const half of the class.
            public class Const_Region : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Region(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_Destroy", ExactSpelling = true)]
                    extern static void __MR_WeightedShell_ParametersRegions_Region_Destroy(_Underlying *_this);
                    __MR_WeightedShell_ParametersRegions_Region_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Region() {Dispose(false);}

                public unsafe MR.Const_VertBitSet Verts
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_Get_verts", ExactSpelling = true)]
                        extern static MR.Const_VertBitSet._Underlying *__MR_WeightedShell_ParametersRegions_Region_Get_verts(_Underlying *_this);
                        return new(__MR_WeightedShell_ParametersRegions_Region_Get_verts(_UnderlyingPtr), is_owning: false);
                    }
                }

                public unsafe float Weight
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_Get_weight", ExactSpelling = true)]
                        extern static float *__MR_WeightedShell_ParametersRegions_Region_Get_weight(_Underlying *_this);
                        return *__MR_WeightedShell_ParametersRegions_Region_Get_weight(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Region() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_DefaultConstruct();
                    _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_Region_DefaultConstruct();
                }

                /// Constructs `MR::WeightedShell::ParametersRegions::Region` elementwise.
                public unsafe Const_Region(MR._ByValue_VertBitSet verts, float weight) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_ConstructFrom", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_ConstructFrom(MR.Misc._PassBy verts_pass_by, MR.VertBitSet._Underlying *verts, float weight);
                    _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_Region_ConstructFrom(verts.PassByMode, verts.Value is not null ? verts.Value._UnderlyingPtr : null, weight);
                }

                /// Generated from constructor `MR::WeightedShell::ParametersRegions::Region::Region`.
                public unsafe Const_Region(MR.WeightedShell.ParametersRegions._ByValue_Region _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersRegions.Region._Underlying *_other);
                    _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_Region_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from class `MR::WeightedShell::ParametersRegions::Region`.
            /// This is the non-const half of the class.
            public class Region : Const_Region
            {
                internal unsafe Region(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                public new unsafe MR.VertBitSet Verts
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_GetMutable_verts", ExactSpelling = true)]
                        extern static MR.VertBitSet._Underlying *__MR_WeightedShell_ParametersRegions_Region_GetMutable_verts(_Underlying *_this);
                        return new(__MR_WeightedShell_ParametersRegions_Region_GetMutable_verts(_UnderlyingPtr), is_owning: false);
                    }
                }

                public new unsafe ref float Weight
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_GetMutable_weight", ExactSpelling = true)]
                        extern static float *__MR_WeightedShell_ParametersRegions_Region_GetMutable_weight(_Underlying *_this);
                        return ref *__MR_WeightedShell_ParametersRegions_Region_GetMutable_weight(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Region() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_DefaultConstruct();
                    _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_Region_DefaultConstruct();
                }

                /// Constructs `MR::WeightedShell::ParametersRegions::Region` elementwise.
                public unsafe Region(MR._ByValue_VertBitSet verts, float weight) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_ConstructFrom", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_ConstructFrom(MR.Misc._PassBy verts_pass_by, MR.VertBitSet._Underlying *verts, float weight);
                    _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_Region_ConstructFrom(verts.PassByMode, verts.Value is not null ? verts.Value._UnderlyingPtr : null, weight);
                }

                /// Generated from constructor `MR::WeightedShell::ParametersRegions::Region::Region`.
                public unsafe Region(MR.WeightedShell.ParametersRegions._ByValue_Region _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersRegions.Region._Underlying *_other);
                    _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_Region_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
                }

                /// Generated from method `MR::WeightedShell::ParametersRegions::Region::operator=`.
                public unsafe MR.WeightedShell.ParametersRegions.Region Assign(MR.WeightedShell.ParametersRegions._ByValue_Region _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_Region_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.WeightedShell.ParametersRegions.Region._Underlying *__MR_WeightedShell_ParametersRegions_Region_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersRegions.Region._Underlying *_other);
                    return new(__MR_WeightedShell_ParametersRegions_Region_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
                }
            }

            /// This is used as a function parameter when the underlying function receives `Region` by value.
            /// Usage:
            /// * Pass `new()` to default-construct the instance.
            /// * Pass an instance of `Region`/`Const_Region` to copy it into the function.
            /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
            ///   Be careful if your input isn't a unique reference to this object.
            /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
            public class _ByValue_Region
            {
                internal readonly Const_Region? Value;
                internal readonly MR.Misc._PassBy PassByMode;
                public _ByValue_Region() {PassByMode = MR.Misc._PassBy.default_construct;}
                public _ByValue_Region(Const_Region new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
                public static implicit operator _ByValue_Region(Const_Region arg) {return new(arg);}
                public _ByValue_Region(MR.Misc._Moved<Region> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
                public static implicit operator _ByValue_Region(MR.Misc._Moved<Region> arg) {return new(arg);}
            }

            /// This is used for optional parameters of class `Region` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Region`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Region`/`Const_Region` directly.
            public class _InOptMut_Region
            {
                public Region? Opt;

                public _InOptMut_Region() {}
                public _InOptMut_Region(Region value) {Opt = value;}
                public static implicit operator _InOptMut_Region(Region value) {return new(value);}
            }

            /// This is used for optional parameters of class `Region` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Region`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Region`/`Const_Region` to pass it to the function.
            public class _InOptConst_Region
            {
                public Const_Region? Opt;

                public _InOptConst_Region() {}
                public _InOptConst_Region(Const_Region value) {Opt = value;}
                public static implicit operator _InOptConst_Region(Const_Region value) {return new(value);}
            }
        }

        /// Generated from class `MR::WeightedShell::ParametersRegions`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::WeightedShell::ParametersBase`
        /// This is the non-const half of the class.
        public class ParametersRegions : Const_ParametersRegions
        {
            internal unsafe ParametersRegions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Upcasts:
            public static unsafe implicit operator MR.WeightedShell.ParametersBase(ParametersRegions self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_UpcastTo_MR_WeightedShell_ParametersBase", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersBase._Underlying *__MR_WeightedShell_ParametersRegions_UpcastTo_MR_WeightedShell_ParametersBase(_Underlying *_this);
                MR.WeightedShell.ParametersBase ret = new(__MR_WeightedShell_ParametersRegions_UpcastTo_MR_WeightedShell_ParametersBase(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            /// list of regions (overlappings are allowed) with corresponding offsets
            /// the additional offset in overlaps is set to the average of the regions
            public new unsafe MR.Std.Vector_MRWeightedShellParametersRegionsRegion Regions
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_regions", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRWeightedShellParametersRegionsRegion._Underlying *__MR_WeightedShell_ParametersRegions_GetMutable_regions(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersRegions_GetMutable_regions(_UnderlyingPtr), is_owning: false);
                }
            }

            /// interpolation distance between the weights of the regions
            /// determines the sharpness of transitions between different regions
            public new unsafe ref float InterpolationDist
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_interpolationDist", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_GetMutable_interpolationDist(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersRegions_GetMutable_interpolationDist(_UnderlyingPtr);
                }
            }

            /// if true the distances grow in both directions from each triangle, reaching minimum in the triangle;
            /// if false the distances grow to infinity in the direction of triangle's normals, and decrease to minus infinity in the opposite direction
            public new unsafe ref bool BidirectionalMode
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_bidirectionalMode", ExactSpelling = true)]
                    extern static bool *__MR_WeightedShell_ParametersRegions_GetMutable_bidirectionalMode(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersRegions_GetMutable_bidirectionalMode(_UnderlyingPtr);
                }
            }

            /// build iso-surface of minimal distance to points corresponding to this value
            public new unsafe ref float Offset
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_offset", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_GetMutable_offset(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersRegions_GetMutable_offset(_UnderlyingPtr);
                }
            }

            /// Size of voxel in grid conversions;
            /// The user is responsible for setting some positive value here
            public new unsafe ref float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_GetMutable_voxelSize(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersRegions_GetMutable_voxelSize(_UnderlyingPtr);
                }
            }

            /// number of voxels to compute near the offset (should be left default unless used for debugging)
            public new unsafe ref float NumLayers
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_numLayers", ExactSpelling = true)]
                    extern static float *__MR_WeightedShell_ParametersRegions_GetMutable_numLayers(_Underlying *_this);
                    return ref *__MR_WeightedShell_ParametersRegions_GetMutable_numLayers(_UnderlyingPtr);
                }
            }

            /// Progress callback
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_WeightedShell_ParametersRegions_GetMutable_progress(_Underlying *_this);
                    return new(__MR_WeightedShell_ParametersRegions_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ParametersRegions() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersRegions._Underlying *__MR_WeightedShell_ParametersRegions_DefaultConstruct();
                _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_DefaultConstruct();
            }

            /// Generated from constructor `MR::WeightedShell::ParametersRegions::ParametersRegions`.
            public unsafe ParametersRegions(MR.WeightedShell._ByValue_ParametersRegions _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersRegions._Underlying *__MR_WeightedShell_ParametersRegions_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersRegions._Underlying *_other);
                _UnderlyingPtr = __MR_WeightedShell_ParametersRegions_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::WeightedShell::ParametersRegions::operator=`.
            public unsafe MR.WeightedShell.ParametersRegions Assign(MR.WeightedShell._ByValue_ParametersRegions _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_ParametersRegions_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WeightedShell.ParametersRegions._Underlying *__MR_WeightedShell_ParametersRegions_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WeightedShell.ParametersRegions._Underlying *_other);
                return new(__MR_WeightedShell_ParametersRegions_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ParametersRegions` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ParametersRegions`/`Const_ParametersRegions` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ParametersRegions
        {
            internal readonly Const_ParametersRegions? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ParametersRegions() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ParametersRegions(Const_ParametersRegions new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ParametersRegions(Const_ParametersRegions arg) {return new(arg);}
            public _ByValue_ParametersRegions(MR.Misc._Moved<ParametersRegions> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ParametersRegions(MR.Misc._Moved<ParametersRegions> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ParametersRegions` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ParametersRegions`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ParametersRegions`/`Const_ParametersRegions` directly.
        public class _InOptMut_ParametersRegions
        {
            public ParametersRegions? Opt;

            public _InOptMut_ParametersRegions() {}
            public _InOptMut_ParametersRegions(ParametersRegions value) {Opt = value;}
            public static implicit operator _InOptMut_ParametersRegions(ParametersRegions value) {return new(value);}
        }

        /// This is used for optional parameters of class `ParametersRegions` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ParametersRegions`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ParametersRegions`/`Const_ParametersRegions` to pass it to the function.
        public class _InOptConst_ParametersRegions
        {
            public Const_ParametersRegions? Opt;

            public _InOptConst_ParametersRegions() {}
            public _InOptConst_ParametersRegions(Const_ParametersRegions value) {Opt = value;}
            public static implicit operator _InOptConst_ParametersRegions(Const_ParametersRegions value) {return new(value);}
        }

        /// makes FunctionVolume representing minimal distance to weighted points
        /// Generated from function `MR::WeightedShell::pointsToDistanceVolume`.
        public static unsafe MR.Misc._Moved<MR.FunctionVolume> PointsToDistanceVolume(MR.Const_PointCloud cloud, MR.WeightedShell.Const_DistanceVolumeCreationParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_pointsToDistanceVolume", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_WeightedShell_pointsToDistanceVolume(MR.Const_PointCloud._Underlying *cloud, MR.WeightedShell.Const_DistanceVolumeCreationParams._Underlying *params_);
            return MR.Misc.Move(new MR.FunctionVolume(__MR_WeightedShell_pointsToDistanceVolume(cloud._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
        }

        /// makes FunctionVolume representing minimal distance to mesh with weighted vertices
        /// Generated from function `MR::WeightedShell::meshToDistanceVolume`.
        public static unsafe MR.Misc._Moved<MR.FunctionVolume> MeshToDistanceVolume(MR.Const_Mesh mesh, MR.WeightedShell.Const_DistanceVolumeCreationParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_meshToDistanceVolume", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_WeightedShell_meshToDistanceVolume(MR.Const_Mesh._Underlying *mesh, MR.WeightedShell.Const_DistanceVolumeCreationParams._Underlying *params_);
            return MR.Misc.Move(new MR.FunctionVolume(__MR_WeightedShell_meshToDistanceVolume(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
        }

        /// consider a point cloud where each point has additive weight (taken from pointWeights and not from params),
        /// and the distance to a point is considered equal to (euclidean distance - weight),
        /// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
        /// Generated from function `MR::WeightedShell::pointsShell`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> PointsShell(MR.Const_PointCloud cloud, MR.Const_VertScalars pointWeights, MR.WeightedShell.Const_ParametersMetric params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_pointsShell", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_WeightedShell_pointsShell(MR.Const_PointCloud._Underlying *cloud, MR.Const_VertScalars._Underlying *pointWeights, MR.WeightedShell.Const_ParametersMetric._Underlying *params_);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_WeightedShell_pointsShell(cloud._UnderlyingPtr, pointWeights._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
        }

        /// consider a mesh where each vertex has additive weight (taken from vertWeights and not from params), and this weight is linearly interpolated in mesh triangles,
        /// and the distance to a point is considered equal to (euclidean distance - weight),
        /// constructs iso-surface of such distance field corresponding to params.offset value using marching cubes
        /// Generated from function `MR::WeightedShell::meshShell`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MeshShell(MR.Const_Mesh mesh, MR.Const_VertScalars vertWeights, MR.WeightedShell.Const_ParametersMetric params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_meshShell_3_MR_VertScalars", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_WeightedShell_meshShell_3_MR_VertScalars(MR.Const_Mesh._Underlying *mesh, MR.Const_VertScalars._Underlying *vertWeights, MR.WeightedShell.Const_ParametersMetric._Underlying *params_);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_WeightedShell_meshShell_3_MR_VertScalars(mesh._UnderlyingPtr, vertWeights._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
        }

        /// interpolate set of regions and assign weight to each vertex of the mesh
        /// Generated from function `MR::WeightedShell::calculateShellWeightsFromRegions`.
        public static unsafe MR.Misc._Moved<MR.VertScalars> CalculateShellWeightsFromRegions(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRWeightedShellParametersRegionsRegion regions, float interpolationDist)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_calculateShellWeightsFromRegions", ExactSpelling = true)]
            extern static MR.VertScalars._Underlying *__MR_WeightedShell_calculateShellWeightsFromRegions(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRWeightedShellParametersRegionsRegion._Underlying *regions, float interpolationDist);
            return MR.Misc.Move(new MR.VertScalars(__MR_WeightedShell_calculateShellWeightsFromRegions(mesh._UnderlyingPtr, regions._UnderlyingPtr, interpolationDist), is_owning: true));
        }

        /// this overload supports linear interpolation between the regions with different weight
        /// Generated from function `MR::WeightedShell::meshShell`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MeshShell(MR.Const_Mesh mesh, MR.WeightedShell.Const_ParametersRegions params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_meshShell_2", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_WeightedShell_meshShell_2(MR.Const_Mesh._Underlying *mesh, MR.WeightedShell.Const_ParametersRegions._Underlying *params_);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_WeightedShell_meshShell_2(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
        }

        /// this overload allows to control how distance volume is build during the offset
        /// Generated from function `MR::WeightedShell::meshShell`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MeshShell(MR.Const_Mesh mesh, MR.WeightedShell.Const_ParametersRegions params_, MR.Std._ByValue_Function_MRFunctionVolumeFuncFromConstMRMeshRefConstMRWeightedShellDistanceVolumeCreationParamsRef volumeBuilder)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedShell_meshShell_3_MR_WeightedShell_ParametersRegions", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_WeightedShell_meshShell_3_MR_WeightedShell_ParametersRegions(MR.Const_Mesh._Underlying *mesh, MR.WeightedShell.Const_ParametersRegions._Underlying *params_, MR.Misc._PassBy volumeBuilder_pass_by, MR.Std.Function_MRFunctionVolumeFuncFromConstMRMeshRefConstMRWeightedShellDistanceVolumeCreationParamsRef._Underlying *volumeBuilder);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_WeightedShell_meshShell_3_MR_WeightedShell_ParametersRegions(mesh._UnderlyingPtr, params_._UnderlyingPtr, volumeBuilder.PassByMode, volumeBuilder.Value is not null ? volumeBuilder.Value._UnderlyingPtr : null), is_owning: true));
        }
    }
}
