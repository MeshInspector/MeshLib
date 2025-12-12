public static partial class MR
{
    /// Generated from class `MR::MeshToDistanceVolumeParams`.
    /// This is the const half of the class.
    public class Const_MeshToDistanceVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshToDistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshToDistanceVolumeParams_Destroy(_Underlying *_this);
            __MR_MeshToDistanceVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshToDistanceVolumeParams() {Dispose(false);}

        public unsafe MR.Const_DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_Get_vol", ExactSpelling = true)]
                extern static MR.Const_DistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_Get_vol(_Underlying *_this);
                return new(__MR_MeshToDistanceVolumeParams_Get_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_SignedDistanceToMeshOptions Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_Get_dist", ExactSpelling = true)]
                extern static MR.Const_SignedDistanceToMeshOptions._Underlying *__MR_MeshToDistanceVolumeParams_Get_dist(_Underlying *_this);
                return new(__MR_MeshToDistanceVolumeParams_Get_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_MeshToDistanceVolumeParams_Get_fwn(_Underlying *_this);
                return new(__MR_MeshToDistanceVolumeParams_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshToDistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToDistanceVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::MeshToDistanceVolumeParams` elementwise.
        public unsafe Const_MeshToDistanceVolumeParams(MR._ByValue_DistanceVolumeParams vol, MR.Const_SignedDistanceToMeshOptions dist, MR._ByValue_IFastWindingNumber fwn) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.SignedDistanceToMeshOptions._Underlying *dist, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn);
            _UnderlyingPtr = __MR_MeshToDistanceVolumeParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist._UnderlyingPtr, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshToDistanceVolumeParams::MeshToDistanceVolumeParams`.
        public unsafe Const_MeshToDistanceVolumeParams(MR._ByValue_MeshToDistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshToDistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToDistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::MeshToDistanceVolumeParams`.
    /// This is the non-const half of the class.
    public class MeshToDistanceVolumeParams : Const_MeshToDistanceVolumeParams
    {
        internal unsafe MeshToDistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_GetMutable_vol", ExactSpelling = true)]
                extern static MR.DistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_GetMutable_vol(_Underlying *_this);
                return new(__MR_MeshToDistanceVolumeParams_GetMutable_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.SignedDistanceToMeshOptions Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_GetMutable_dist", ExactSpelling = true)]
                extern static MR.SignedDistanceToMeshOptions._Underlying *__MR_MeshToDistanceVolumeParams_GetMutable_dist(_Underlying *_this);
                return new(__MR_MeshToDistanceVolumeParams_GetMutable_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_MeshToDistanceVolumeParams_GetMutable_fwn(_Underlying *_this);
                return new(__MR_MeshToDistanceVolumeParams_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshToDistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToDistanceVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::MeshToDistanceVolumeParams` elementwise.
        public unsafe MeshToDistanceVolumeParams(MR._ByValue_DistanceVolumeParams vol, MR.Const_SignedDistanceToMeshOptions dist, MR._ByValue_IFastWindingNumber fwn) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.SignedDistanceToMeshOptions._Underlying *dist, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn);
            _UnderlyingPtr = __MR_MeshToDistanceVolumeParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist._UnderlyingPtr, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshToDistanceVolumeParams::MeshToDistanceVolumeParams`.
        public unsafe MeshToDistanceVolumeParams(MR._ByValue_MeshToDistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshToDistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToDistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshToDistanceVolumeParams::operator=`.
        public unsafe MR.MeshToDistanceVolumeParams Assign(MR._ByValue_MeshToDistanceVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDistanceVolumeParams._Underlying *__MR_MeshToDistanceVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshToDistanceVolumeParams._Underlying *_other);
            return new(__MR_MeshToDistanceVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshToDistanceVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshToDistanceVolumeParams`/`Const_MeshToDistanceVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshToDistanceVolumeParams
    {
        internal readonly Const_MeshToDistanceVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshToDistanceVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshToDistanceVolumeParams(Const_MeshToDistanceVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshToDistanceVolumeParams(Const_MeshToDistanceVolumeParams arg) {return new(arg);}
        public _ByValue_MeshToDistanceVolumeParams(MR.Misc._Moved<MeshToDistanceVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshToDistanceVolumeParams(MR.Misc._Moved<MeshToDistanceVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshToDistanceVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshToDistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToDistanceVolumeParams`/`Const_MeshToDistanceVolumeParams` directly.
    public class _InOptMut_MeshToDistanceVolumeParams
    {
        public MeshToDistanceVolumeParams? Opt;

        public _InOptMut_MeshToDistanceVolumeParams() {}
        public _InOptMut_MeshToDistanceVolumeParams(MeshToDistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshToDistanceVolumeParams(MeshToDistanceVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshToDistanceVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshToDistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToDistanceVolumeParams`/`Const_MeshToDistanceVolumeParams` to pass it to the function.
    public class _InOptConst_MeshToDistanceVolumeParams
    {
        public Const_MeshToDistanceVolumeParams? Opt;

        public _InOptConst_MeshToDistanceVolumeParams() {}
        public _InOptConst_MeshToDistanceVolumeParams(Const_MeshToDistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshToDistanceVolumeParams(Const_MeshToDistanceVolumeParams value) {return new(value);}
    }

    /// Generated from class `MR::CloseToMeshVolumeParams`.
    /// This is the const half of the class.
    public class Const_CloseToMeshVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CloseToMeshVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_CloseToMeshVolumeParams_Destroy(_Underlying *_this);
            __MR_CloseToMeshVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CloseToMeshVolumeParams() {Dispose(false);}

        /// a resulting voxel will get 1 if that voxel's center is not further than unsigned (closeDist) from the surface, and 0 otherwise
        public unsafe float CloseDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_Get_closeDist", ExactSpelling = true)]
                extern static float *__MR_CloseToMeshVolumeParams_Get_closeDist(_Underlying *_this);
                return *__MR_CloseToMeshVolumeParams_Get_closeDist(_UnderlyingPtr);
            }
        }

        /// dimensions, location, and scaling in world space of the expected volume
        public unsafe MR.Const_DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_Get_vol", ExactSpelling = true)]
                extern static MR.Const_DistanceVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_Get_vol(_Underlying *_this);
                return new(__MR_CloseToMeshVolumeParams_Get_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        /// optional transformation from mesh space to world space
        public unsafe ref readonly MR.AffineXf3f * MeshToWorld
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_Get_meshToWorld", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_CloseToMeshVolumeParams_Get_meshToWorld(_Underlying *_this);
                return ref *__MR_CloseToMeshVolumeParams_Get_meshToWorld(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CloseToMeshVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_CloseToMeshVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::CloseToMeshVolumeParams` elementwise.
        public unsafe Const_CloseToMeshVolumeParams(float closeDist, MR._ByValue_DistanceVolumeParams vol, MR.Const_AffineXf3f? meshToWorld) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_ConstructFrom(float closeDist, MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.Const_AffineXf3f._Underlying *meshToWorld);
            _UnderlyingPtr = __MR_CloseToMeshVolumeParams_ConstructFrom(closeDist, vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, meshToWorld is not null ? meshToWorld._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CloseToMeshVolumeParams::CloseToMeshVolumeParams`.
        public unsafe Const_CloseToMeshVolumeParams(MR._ByValue_CloseToMeshVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CloseToMeshVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_CloseToMeshVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::CloseToMeshVolumeParams`.
    /// This is the non-const half of the class.
    public class CloseToMeshVolumeParams : Const_CloseToMeshVolumeParams
    {
        internal unsafe CloseToMeshVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// a resulting voxel will get 1 if that voxel's center is not further than unsigned (closeDist) from the surface, and 0 otherwise
        public new unsafe ref float CloseDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_GetMutable_closeDist", ExactSpelling = true)]
                extern static float *__MR_CloseToMeshVolumeParams_GetMutable_closeDist(_Underlying *_this);
                return ref *__MR_CloseToMeshVolumeParams_GetMutable_closeDist(_UnderlyingPtr);
            }
        }

        /// dimensions, location, and scaling in world space of the expected volume
        public new unsafe MR.DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_GetMutable_vol", ExactSpelling = true)]
                extern static MR.DistanceVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_GetMutable_vol(_Underlying *_this);
                return new(__MR_CloseToMeshVolumeParams_GetMutable_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        /// optional transformation from mesh space to world space
        public new unsafe ref readonly MR.AffineXf3f * MeshToWorld
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_GetMutable_meshToWorld", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_CloseToMeshVolumeParams_GetMutable_meshToWorld(_Underlying *_this);
                return ref *__MR_CloseToMeshVolumeParams_GetMutable_meshToWorld(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CloseToMeshVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_CloseToMeshVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::CloseToMeshVolumeParams` elementwise.
        public unsafe CloseToMeshVolumeParams(float closeDist, MR._ByValue_DistanceVolumeParams vol, MR.Const_AffineXf3f? meshToWorld) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_ConstructFrom(float closeDist, MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.Const_AffineXf3f._Underlying *meshToWorld);
            _UnderlyingPtr = __MR_CloseToMeshVolumeParams_ConstructFrom(closeDist, vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, meshToWorld is not null ? meshToWorld._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CloseToMeshVolumeParams::CloseToMeshVolumeParams`.
        public unsafe CloseToMeshVolumeParams(MR._ByValue_CloseToMeshVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CloseToMeshVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_CloseToMeshVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CloseToMeshVolumeParams::operator=`.
        public unsafe MR.CloseToMeshVolumeParams Assign(MR._ByValue_CloseToMeshVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CloseToMeshVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CloseToMeshVolumeParams._Underlying *__MR_CloseToMeshVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CloseToMeshVolumeParams._Underlying *_other);
            return new(__MR_CloseToMeshVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `CloseToMeshVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CloseToMeshVolumeParams`/`Const_CloseToMeshVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CloseToMeshVolumeParams
    {
        internal readonly Const_CloseToMeshVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CloseToMeshVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_CloseToMeshVolumeParams(Const_CloseToMeshVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CloseToMeshVolumeParams(Const_CloseToMeshVolumeParams arg) {return new(arg);}
        public _ByValue_CloseToMeshVolumeParams(MR.Misc._Moved<CloseToMeshVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CloseToMeshVolumeParams(MR.Misc._Moved<CloseToMeshVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CloseToMeshVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CloseToMeshVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CloseToMeshVolumeParams`/`Const_CloseToMeshVolumeParams` directly.
    public class _InOptMut_CloseToMeshVolumeParams
    {
        public CloseToMeshVolumeParams? Opt;

        public _InOptMut_CloseToMeshVolumeParams() {}
        public _InOptMut_CloseToMeshVolumeParams(CloseToMeshVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_CloseToMeshVolumeParams(CloseToMeshVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `CloseToMeshVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CloseToMeshVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CloseToMeshVolumeParams`/`Const_CloseToMeshVolumeParams` to pass it to the function.
    public class _InOptConst_CloseToMeshVolumeParams
    {
        public Const_CloseToMeshVolumeParams? Opt;

        public _InOptConst_CloseToMeshVolumeParams() {}
        public _InOptConst_CloseToMeshVolumeParams(Const_CloseToMeshVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_CloseToMeshVolumeParams(Const_CloseToMeshVolumeParams value) {return new(value);}
    }

    /// Generated from class `MR::MeshToDirectionVolumeParams`.
    /// This is the const half of the class.
    public class Const_MeshToDirectionVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshToDirectionVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshToDirectionVolumeParams_Destroy(_Underlying *_this);
            __MR_MeshToDirectionVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshToDirectionVolumeParams() {Dispose(false);}

        public unsafe MR.Const_DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_Get_vol", ExactSpelling = true)]
                extern static MR.Const_DistanceVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_Get_vol(_Underlying *_this);
                return new(__MR_MeshToDirectionVolumeParams_Get_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        // note that signMode is ignored in this algorithm
        public unsafe MR.Const_DistanceToMeshOptions Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_Get_dist", ExactSpelling = true)]
                extern static MR.Const_DistanceToMeshOptions._Underlying *__MR_MeshToDirectionVolumeParams_Get_dist(_Underlying *_this);
                return new(__MR_MeshToDirectionVolumeParams_Get_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_IPointsToMeshProjector Projector
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_Get_projector", ExactSpelling = true)]
                extern static MR.Const_IPointsToMeshProjector._UnderlyingShared *__MR_MeshToDirectionVolumeParams_Get_projector(_Underlying *_this);
                return new(__MR_MeshToDirectionVolumeParams_Get_projector(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshToDirectionVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToDirectionVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::MeshToDirectionVolumeParams` elementwise.
        public unsafe Const_MeshToDirectionVolumeParams(MR._ByValue_DistanceVolumeParams vol, MR.Const_DistanceToMeshOptions dist, MR._ByValue_IPointsToMeshProjector projector) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.DistanceToMeshOptions._Underlying *dist, MR.Misc._PassBy projector_pass_by, MR.IPointsToMeshProjector._UnderlyingShared *projector);
            _UnderlyingPtr = __MR_MeshToDirectionVolumeParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist._UnderlyingPtr, projector.PassByMode, projector.Value is not null ? projector.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshToDirectionVolumeParams::MeshToDirectionVolumeParams`.
        public unsafe Const_MeshToDirectionVolumeParams(MR._ByValue_MeshToDirectionVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshToDirectionVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToDirectionVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::MeshToDirectionVolumeParams`.
    /// This is the non-const half of the class.
    public class MeshToDirectionVolumeParams : Const_MeshToDirectionVolumeParams
    {
        internal unsafe MeshToDirectionVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.DistanceVolumeParams Vol
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_GetMutable_vol", ExactSpelling = true)]
                extern static MR.DistanceVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_GetMutable_vol(_Underlying *_this);
                return new(__MR_MeshToDirectionVolumeParams_GetMutable_vol(_UnderlyingPtr), is_owning: false);
            }
        }

        // note that signMode is ignored in this algorithm
        public new unsafe MR.DistanceToMeshOptions Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_GetMutable_dist", ExactSpelling = true)]
                extern static MR.DistanceToMeshOptions._Underlying *__MR_MeshToDirectionVolumeParams_GetMutable_dist(_Underlying *_this);
                return new(__MR_MeshToDirectionVolumeParams_GetMutable_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.IPointsToMeshProjector Projector
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_GetMutable_projector", ExactSpelling = true)]
                extern static MR.IPointsToMeshProjector._UnderlyingShared *__MR_MeshToDirectionVolumeParams_GetMutable_projector(_Underlying *_this);
                return new(__MR_MeshToDirectionVolumeParams_GetMutable_projector(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshToDirectionVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToDirectionVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::MeshToDirectionVolumeParams` elementwise.
        public unsafe MeshToDirectionVolumeParams(MR._ByValue_DistanceVolumeParams vol, MR.Const_DistanceToMeshOptions dist, MR._ByValue_IPointsToMeshProjector projector) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.DistanceVolumeParams._Underlying *vol, MR.DistanceToMeshOptions._Underlying *dist, MR.Misc._PassBy projector_pass_by, MR.IPointsToMeshProjector._UnderlyingShared *projector);
            _UnderlyingPtr = __MR_MeshToDirectionVolumeParams_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, dist._UnderlyingPtr, projector.PassByMode, projector.Value is not null ? projector.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::MeshToDirectionVolumeParams::MeshToDirectionVolumeParams`.
        public unsafe MeshToDirectionVolumeParams(MR._ByValue_MeshToDirectionVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshToDirectionVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToDirectionVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshToDirectionVolumeParams::operator=`.
        public unsafe MR.MeshToDirectionVolumeParams Assign(MR._ByValue_MeshToDirectionVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDirectionVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDirectionVolumeParams._Underlying *__MR_MeshToDirectionVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshToDirectionVolumeParams._Underlying *_other);
            return new(__MR_MeshToDirectionVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshToDirectionVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshToDirectionVolumeParams`/`Const_MeshToDirectionVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshToDirectionVolumeParams
    {
        internal readonly Const_MeshToDirectionVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshToDirectionVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshToDirectionVolumeParams(Const_MeshToDirectionVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshToDirectionVolumeParams(Const_MeshToDirectionVolumeParams arg) {return new(arg);}
        public _ByValue_MeshToDirectionVolumeParams(MR.Misc._Moved<MeshToDirectionVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshToDirectionVolumeParams(MR.Misc._Moved<MeshToDirectionVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshToDirectionVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshToDirectionVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToDirectionVolumeParams`/`Const_MeshToDirectionVolumeParams` directly.
    public class _InOptMut_MeshToDirectionVolumeParams
    {
        public MeshToDirectionVolumeParams? Opt;

        public _InOptMut_MeshToDirectionVolumeParams() {}
        public _InOptMut_MeshToDirectionVolumeParams(MeshToDirectionVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshToDirectionVolumeParams(MeshToDirectionVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshToDirectionVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshToDirectionVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToDirectionVolumeParams`/`Const_MeshToDirectionVolumeParams` to pass it to the function.
    public class _InOptConst_MeshToDirectionVolumeParams
    {
        public Const_MeshToDirectionVolumeParams? Opt;

        public _InOptConst_MeshToDirectionVolumeParams() {}
        public _InOptConst_MeshToDirectionVolumeParams(Const_MeshToDirectionVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshToDirectionVolumeParams(Const_MeshToDirectionVolumeParams value) {return new(value);}
    }

    /// makes SimpleVolume filled with (signed or unsigned) distances from Mesh with given settings
    /// Generated from function `MR::meshToDistanceVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolumeMinMax_StdString> MeshToDistanceVolume(MR.Const_MeshPart mp, MR.Const_MeshToDistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToDistanceVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolumeMinMax_StdString._Underlying *__MR_meshToDistanceVolume(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshToDistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolumeMinMax_StdString(__MR_meshToDistanceVolume(mp._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// makes FunctionVolume representing (signed or unsigned) distances from Mesh with given settings
    /// Generated from function `MR::meshToDistanceFunctionVolume`.
    public static unsafe MR.Misc._Moved<MR.FunctionVolume> MeshToDistanceFunctionVolume(MR.Const_MeshPart mp, MR.Const_MeshToDistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToDistanceFunctionVolume", ExactSpelling = true)]
        extern static MR.FunctionVolume._Underlying *__MR_meshToDistanceFunctionVolume(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshToDistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.FunctionVolume(__MR_meshToDistanceFunctionVolume(mp._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// makes a binary volume with close-to-surface predicate values according to the given parameters
    /// Generated from function `MR::makeCloseToMeshVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleBinaryVolume_StdString> MakeCloseToMeshVolume(MR.Const_MeshPart mp, MR.Const_CloseToMeshVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeCloseToMeshVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleBinaryVolume_StdString._Underlying *__MR_makeCloseToMeshVolume(MR.Const_MeshPart._Underlying *mp, MR.Const_CloseToMeshVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRSimpleBinaryVolume_StdString(__MR_makeCloseToMeshVolume(mp._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// returns a volume filled with the values: (unsigned distance to region-part) - (unsigned distance to not-region-part);
    /// v < 0: this point is within offset distance to region-part of mesh and it is closer to region-part than to not-region-part
    /// Generated from function `MR::meshRegionToIndicatorVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolumeMinMax_StdString> MeshRegionToIndicatorVolume(MR.Const_Mesh mesh, MR.Const_FaceBitSet region, float offset, MR.Const_DistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshRegionToIndicatorVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolumeMinMax_StdString._Underlying *__MR_meshRegionToIndicatorVolume(MR.Const_Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *region, float offset, MR.Const_DistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolumeMinMax_StdString(__MR_meshRegionToIndicatorVolume(mesh._UnderlyingPtr, region._UnderlyingPtr, offset, params_._UnderlyingPtr), is_owning: true));
    }

    /// Converts mesh into 4d voxels, so that each cell in 3d space holds the direction from the closest point on mesh to the cell position.
    /// Resulting volume is encoded by 3 separate 3d volumes, corresponding to `x`, `y` and `z` components of vectors respectively.
    /// \param params Expected to have valid (not null) projector, with invoked method \ref IPointsToMeshProjector::updateMeshData
    /// Generated from function `MR::meshToDirectionVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdArrayMRSimpleVolumeMinMax3_StdString> MeshToDirectionVolume(MR.Const_MeshToDirectionVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToDirectionVolume", ExactSpelling = true)]
        extern static MR.Expected_StdArrayMRSimpleVolumeMinMax3_StdString._Underlying *__MR_meshToDirectionVolume(MR.Const_MeshToDirectionVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_StdArrayMRSimpleVolumeMinMax3_StdString(__MR_meshToDirectionVolume(params_._UnderlyingPtr), is_owning: true));
    }
}
