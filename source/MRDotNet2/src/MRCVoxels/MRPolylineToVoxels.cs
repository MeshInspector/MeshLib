public static partial class MR
{
    /// Generated from class `MR::PolylineToDistanceVolumeParams`.
    /// This is the const half of the class.
    public class Const_PolylineToDistanceVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineToDistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineToDistanceVolumeParams_Destroy(_Underlying *_this);
            __MR_PolylineToDistanceVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineToDistanceVolumeParams() {Dispose(false);}

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PolylineToDistanceVolumeParams_Get_voxelSize(_Underlying *_this);
                return new(__MR_PolylineToDistanceVolumeParams_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
        public unsafe float OffsetCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_Get_offsetCount", ExactSpelling = true)]
                extern static float *__MR_PolylineToDistanceVolumeParams_Get_offsetCount(_Underlying *_this);
                return *__MR_PolylineToDistanceVolumeParams_Get_offsetCount(_UnderlyingPtr);
            }
        }

        // line initial transform
        public unsafe MR.Const_AffineXf3f WorldXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_Get_worldXf", ExactSpelling = true)]
                extern static MR.Const_AffineXf3f._Underlying *__MR_PolylineToDistanceVolumeParams_Get_worldXf(_Underlying *_this);
                return new(__MR_PolylineToDistanceVolumeParams_Get_worldXf(_UnderlyingPtr), is_owning: false);
            }
        }

        // optional output: xf to original mesh (respecting worldXf)
        public unsafe ref MR.AffineXf3f * OutXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_Get_outXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_PolylineToDistanceVolumeParams_Get_outXf(_Underlying *_this);
                return ref *__MR_PolylineToDistanceVolumeParams_Get_outXf(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PolylineToDistanceVolumeParams_Get_cb(_Underlying *_this);
                return new(__MR_PolylineToDistanceVolumeParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineToDistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineToDistanceVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::PolylineToDistanceVolumeParams` elementwise.
        public unsafe Const_PolylineToDistanceVolumeParams(MR.Vector3f voxelSize, float offsetCount, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f? outXf, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_ConstructFrom(MR.Vector3f voxelSize, float offsetCount, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f._Underlying *outXf, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_PolylineToDistanceVolumeParams_ConstructFrom(voxelSize, offsetCount, worldXf, outXf is not null ? outXf._UnderlyingPtr : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PolylineToDistanceVolumeParams::PolylineToDistanceVolumeParams`.
        public unsafe Const_PolylineToDistanceVolumeParams(MR._ByValue_PolylineToDistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineToDistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineToDistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::PolylineToDistanceVolumeParams`.
    /// This is the non-const half of the class.
    public class PolylineToDistanceVolumeParams : Const_PolylineToDistanceVolumeParams
    {
        internal unsafe PolylineToDistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PolylineToDistanceVolumeParams_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_PolylineToDistanceVolumeParams_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// offsetCount - the number of voxels around polyline to calculate distance in (should be positive)
        public new unsafe ref float OffsetCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_GetMutable_offsetCount", ExactSpelling = true)]
                extern static float *__MR_PolylineToDistanceVolumeParams_GetMutable_offsetCount(_Underlying *_this);
                return ref *__MR_PolylineToDistanceVolumeParams_GetMutable_offsetCount(_UnderlyingPtr);
            }
        }

        // line initial transform
        public new unsafe MR.Mut_AffineXf3f WorldXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_GetMutable_worldXf", ExactSpelling = true)]
                extern static MR.Mut_AffineXf3f._Underlying *__MR_PolylineToDistanceVolumeParams_GetMutable_worldXf(_Underlying *_this);
                return new(__MR_PolylineToDistanceVolumeParams_GetMutable_worldXf(_UnderlyingPtr), is_owning: false);
            }
        }

        // optional output: xf to original mesh (respecting worldXf)
        public new unsafe ref MR.AffineXf3f * OutXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_GetMutable_outXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_PolylineToDistanceVolumeParams_GetMutable_outXf(_Underlying *_this);
                return ref *__MR_PolylineToDistanceVolumeParams_GetMutable_outXf(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PolylineToDistanceVolumeParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_PolylineToDistanceVolumeParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineToDistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineToDistanceVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::PolylineToDistanceVolumeParams` elementwise.
        public unsafe PolylineToDistanceVolumeParams(MR.Vector3f voxelSize, float offsetCount, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f? outXf, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_ConstructFrom(MR.Vector3f voxelSize, float offsetCount, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f._Underlying *outXf, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_PolylineToDistanceVolumeParams_ConstructFrom(voxelSize, offsetCount, worldXf, outXf is not null ? outXf._UnderlyingPtr : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PolylineToDistanceVolumeParams::PolylineToDistanceVolumeParams`.
        public unsafe PolylineToDistanceVolumeParams(MR._ByValue_PolylineToDistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineToDistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineToDistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolylineToDistanceVolumeParams::operator=`.
        public unsafe MR.PolylineToDistanceVolumeParams Assign(MR._ByValue_PolylineToDistanceVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToDistanceVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineToDistanceVolumeParams._Underlying *__MR_PolylineToDistanceVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PolylineToDistanceVolumeParams._Underlying *_other);
            return new(__MR_PolylineToDistanceVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PolylineToDistanceVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PolylineToDistanceVolumeParams`/`Const_PolylineToDistanceVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PolylineToDistanceVolumeParams
    {
        internal readonly Const_PolylineToDistanceVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PolylineToDistanceVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PolylineToDistanceVolumeParams(Const_PolylineToDistanceVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PolylineToDistanceVolumeParams(Const_PolylineToDistanceVolumeParams arg) {return new(arg);}
        public _ByValue_PolylineToDistanceVolumeParams(MR.Misc._Moved<PolylineToDistanceVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PolylineToDistanceVolumeParams(MR.Misc._Moved<PolylineToDistanceVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PolylineToDistanceVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineToDistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineToDistanceVolumeParams`/`Const_PolylineToDistanceVolumeParams` directly.
    public class _InOptMut_PolylineToDistanceVolumeParams
    {
        public PolylineToDistanceVolumeParams? Opt;

        public _InOptMut_PolylineToDistanceVolumeParams() {}
        public _InOptMut_PolylineToDistanceVolumeParams(PolylineToDistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineToDistanceVolumeParams(PolylineToDistanceVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineToDistanceVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineToDistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineToDistanceVolumeParams`/`Const_PolylineToDistanceVolumeParams` to pass it to the function.
    public class _InOptConst_PolylineToDistanceVolumeParams
    {
        public Const_PolylineToDistanceVolumeParams? Opt;

        public _InOptConst_PolylineToDistanceVolumeParams() {}
        public _InOptConst_PolylineToDistanceVolumeParams(Const_PolylineToDistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineToDistanceVolumeParams(Const_PolylineToDistanceVolumeParams value) {return new(value);}
    }

    /// Settings to conversion polyline to volume
    /// Generated from class `MR::PolylineToVolumeParams`.
    /// This is the const half of the class.
    public class Const_PolylineToVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineToVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineToVolumeParams_Destroy(_Underlying *_this);
            __MR_PolylineToVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineToVolumeParams() {Dispose(false);}

        public unsafe MR.Const_DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_Get_vol", ExactSpelling = true)]
                extern static MR.Const_DistanceVolumeParams._Underlying *__MR_PolylineToVolumeParams_Get_vol(_Underlying *_this);
                return new(__MR_PolylineToVolumeParams_Get_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_DistanceToMeshOptions Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_Get_dist", ExactSpelling = true)]
                extern static MR.Const_DistanceToMeshOptions._Underlying *__MR_PolylineToVolumeParams_Get_dist(_Underlying *_this);
                return new(__MR_PolylineToVolumeParams_Get_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineToVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineToVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::PolylineToVolumeParams` elementwise.
        public unsafe Const_PolylineToVolumeParams(MR._ByValue_DistanceVolumeParams vol, MR.Const_DistanceToMeshOptions dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.DistanceToMeshOptions._Underlying *dist);
            _UnderlyingPtr = __MR_PolylineToVolumeParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PolylineToVolumeParams::PolylineToVolumeParams`.
        public unsafe Const_PolylineToVolumeParams(MR._ByValue_PolylineToVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineToVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineToVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Settings to conversion polyline to volume
    /// Generated from class `MR::PolylineToVolumeParams`.
    /// This is the non-const half of the class.
    public class PolylineToVolumeParams : Const_PolylineToVolumeParams
    {
        internal unsafe PolylineToVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_GetMutable_vol", ExactSpelling = true)]
                extern static MR.DistanceVolumeParams._Underlying *__MR_PolylineToVolumeParams_GetMutable_vol(_Underlying *_this);
                return new(__MR_PolylineToVolumeParams_GetMutable_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.DistanceToMeshOptions Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_GetMutable_dist", ExactSpelling = true)]
                extern static MR.DistanceToMeshOptions._Underlying *__MR_PolylineToVolumeParams_GetMutable_dist(_Underlying *_this);
                return new(__MR_PolylineToVolumeParams_GetMutable_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineToVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineToVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::PolylineToVolumeParams` elementwise.
        public unsafe PolylineToVolumeParams(MR._ByValue_DistanceVolumeParams vol, MR.Const_DistanceToMeshOptions dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.DistanceToMeshOptions._Underlying *dist);
            _UnderlyingPtr = __MR_PolylineToVolumeParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PolylineToVolumeParams::PolylineToVolumeParams`.
        public unsafe PolylineToVolumeParams(MR._ByValue_PolylineToVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineToVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineToVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolylineToVolumeParams::operator=`.
        public unsafe MR.PolylineToVolumeParams Assign(MR._ByValue_PolylineToVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineToVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineToVolumeParams._Underlying *__MR_PolylineToVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PolylineToVolumeParams._Underlying *_other);
            return new(__MR_PolylineToVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PolylineToVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PolylineToVolumeParams`/`Const_PolylineToVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PolylineToVolumeParams
    {
        internal readonly Const_PolylineToVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PolylineToVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PolylineToVolumeParams(Const_PolylineToVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PolylineToVolumeParams(Const_PolylineToVolumeParams arg) {return new(arg);}
        public _ByValue_PolylineToVolumeParams(MR.Misc._Moved<PolylineToVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PolylineToVolumeParams(MR.Misc._Moved<PolylineToVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PolylineToVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineToVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineToVolumeParams`/`Const_PolylineToVolumeParams` directly.
    public class _InOptMut_PolylineToVolumeParams
    {
        public PolylineToVolumeParams? Opt;

        public _InOptMut_PolylineToVolumeParams() {}
        public _InOptMut_PolylineToVolumeParams(PolylineToVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineToVolumeParams(PolylineToVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineToVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineToVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineToVolumeParams`/`Const_PolylineToVolumeParams` to pass it to the function.
    public class _InOptConst_PolylineToVolumeParams
    {
        public Const_PolylineToVolumeParams? Opt;

        public _InOptConst_PolylineToVolumeParams() {}
        public _InOptConst_PolylineToVolumeParams(Const_PolylineToVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineToVolumeParams(Const_PolylineToVolumeParams value) {return new(value);}
    }

    /// convert polyline to voxels distance field
    /// Generated from function `MR::polylineToDistanceField`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFloatGrid_StdString> PolylineToDistanceField(MR.Const_Polyline3 polyline, MR.Const_PolylineToDistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_polylineToDistanceField", ExactSpelling = true)]
        extern static MR.Expected_MRFloatGrid_StdString._Underlying *__MR_polylineToDistanceField(MR.Const_Polyline3._Underlying *polyline, MR.Const_PolylineToDistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRFloatGrid_StdString(__MR_polylineToDistanceField(polyline._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// convert polyline to VDB volume
    /// Generated from function `MR::polylineToVdbVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> PolylineToVdbVolume(MR.Const_Polyline3 polyline, MR.Const_PolylineToDistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_polylineToVdbVolume", ExactSpelling = true)]
        extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_polylineToVdbVolume(MR.Const_Polyline3._Underlying *polyline, MR.Const_PolylineToDistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_polylineToVdbVolume(polyline._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// convert polyline to simple volume
    /// Generated from function `MR::polylineToSimpleVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolume_StdString> PolylineToSimpleVolume(MR.Const_Polyline3 polyline, MR.Const_PolylineToVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_polylineToSimpleVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolume_StdString._Underlying *__MR_polylineToSimpleVolume(MR.Const_Polyline3._Underlying *polyline, MR.Const_PolylineToVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolume_StdString(__MR_polylineToSimpleVolume(polyline._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// convert polyline to function volume
    /// Generated from function `MR::polylineToFunctionVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFunctionVolume_StdString> PolylineToFunctionVolume(MR.Const_Polyline3 polyline, MR.Const_PolylineToVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_polylineToFunctionVolume", ExactSpelling = true)]
        extern static MR.Expected_MRFunctionVolume_StdString._Underlying *__MR_polylineToFunctionVolume(MR.Const_Polyline3._Underlying *polyline, MR.Const_PolylineToVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRFunctionVolume_StdString(__MR_polylineToFunctionVolume(polyline._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
