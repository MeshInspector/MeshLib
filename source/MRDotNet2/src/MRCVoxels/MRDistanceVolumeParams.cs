public static partial class MR
{
    /// Generated from class `MR::DistanceVolumeParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointsToDistanceVolumeParams`
    /// This is the const half of the class.
    public class Const_DistanceVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceVolumeParams_Destroy(_Underlying *_this);
            __MR_DistanceVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceVolumeParams() {Dispose(false);}

        /// origin point of voxels box
        public unsafe MR.Const_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_Get_origin", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_DistanceVolumeParams_Get_origin(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_Get_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DistanceVolumeParams_Get_cb(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// size of voxel on each axis
        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_DistanceVolumeParams_Get_voxelSize(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// num voxels along each axis
        public unsafe MR.Const_Vector3i Dimensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_Get_dimensions", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_DistanceVolumeParams_Get_dimensions(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_Get_dimensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::DistanceVolumeParams` elementwise.
        public unsafe Const_DistanceVolumeParams(MR.Vector3f origin, MR.Std._ByValue_Function_BoolFuncFromFloat cb, MR.Vector3f voxelSize, MR.Vector3i dimensions) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_ConstructFrom(MR.Vector3f origin, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Vector3f voxelSize, MR.Vector3i dimensions);
            _UnderlyingPtr = __MR_DistanceVolumeParams_ConstructFrom(origin, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, voxelSize, dimensions);
        }

        /// Generated from constructor `MR::DistanceVolumeParams::DistanceVolumeParams`.
        public unsafe Const_DistanceVolumeParams(MR._ByValue_DistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DistanceVolumeParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointsToDistanceVolumeParams`
    /// This is the non-const half of the class.
    public class DistanceVolumeParams : Const_DistanceVolumeParams
    {
        internal unsafe DistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// origin point of voxels box
        public new unsafe MR.Mut_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_GetMutable_origin", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_DistanceVolumeParams_GetMutable_origin(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_GetMutable_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DistanceVolumeParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// size of voxel on each axis
        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_DistanceVolumeParams_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// num voxels along each axis
        public new unsafe MR.Mut_Vector3i Dimensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_GetMutable_dimensions", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_DistanceVolumeParams_GetMutable_dimensions(_Underlying *_this);
                return new(__MR_DistanceVolumeParams_GetMutable_dimensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::DistanceVolumeParams` elementwise.
        public unsafe DistanceVolumeParams(MR.Vector3f origin, MR.Std._ByValue_Function_BoolFuncFromFloat cb, MR.Vector3f voxelSize, MR.Vector3i dimensions) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_ConstructFrom(MR.Vector3f origin, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Vector3f voxelSize, MR.Vector3i dimensions);
            _UnderlyingPtr = __MR_DistanceVolumeParams_ConstructFrom(origin, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, voxelSize, dimensions);
        }

        /// Generated from constructor `MR::DistanceVolumeParams::DistanceVolumeParams`.
        public unsafe DistanceVolumeParams(MR._ByValue_DistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DistanceVolumeParams::operator=`.
        public unsafe MR.DistanceVolumeParams Assign(MR._ByValue_DistanceVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_DistanceVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceVolumeParams._Underlying *_other);
            return new(__MR_DistanceVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DistanceVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DistanceVolumeParams`/`Const_DistanceVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DistanceVolumeParams
    {
        internal readonly Const_DistanceVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DistanceVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DistanceVolumeParams(Const_DistanceVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DistanceVolumeParams(Const_DistanceVolumeParams arg) {return new(arg);}
        public _ByValue_DistanceVolumeParams(MR.Misc._Moved<DistanceVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DistanceVolumeParams(MR.Misc._Moved<DistanceVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DistanceVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceVolumeParams`/`Const_DistanceVolumeParams` directly.
    public class _InOptMut_DistanceVolumeParams
    {
        public DistanceVolumeParams? Opt;

        public _InOptMut_DistanceVolumeParams() {}
        public _InOptMut_DistanceVolumeParams(DistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceVolumeParams(DistanceVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceVolumeParams`/`Const_DistanceVolumeParams` to pass it to the function.
    public class _InOptConst_DistanceVolumeParams
    {
        public Const_DistanceVolumeParams? Opt;

        public _InOptConst_DistanceVolumeParams() {}
        public _InOptConst_DistanceVolumeParams(Const_DistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceVolumeParams(Const_DistanceVolumeParams value) {return new(value);}
    }
}
