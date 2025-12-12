public static partial class MR
{
    public static partial class FixUndercuts
    {
        /// Parameters that is used to find undercuts
        /// Generated from class `MR::FixUndercuts::FindParams`.
        /// This is the const half of the class.
        public class Const_FindParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_FindParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_Destroy", ExactSpelling = true)]
                extern static void __MR_FixUndercuts_FindParams_Destroy(_Underlying *_this);
                __MR_FixUndercuts_FindParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_FindParams() {Dispose(false);}

            /// Primitives that are not visible from up direction are considered as undercuts (fix undercuts is performed downwards (in `-direction`))
            public unsafe MR.Const_Vector3f UpDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_Get_upDirection", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_FixUndercuts_FindParams_Get_upDirection(_Underlying *_this);
                    return new(__MR_FixUndercuts_FindParams_Get_upDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            /// vertical angle of fixed undercut walls (note that this value is approximate - it defines "camera" position for internal projective transformation)
            /// 0 - strictly vertical walls of undercuts area
            /// positive - expanding downwards walls
            /// negative - shrinking downwards walls
            public unsafe float WallAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_Get_wallAngle", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_FindParams_Get_wallAngle(_Underlying *_this);
                    return *__MR_FixUndercuts_FindParams_Get_wallAngle(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_FindParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_FindParams_DefaultConstruct();
            }

            /// Constructs `MR::FixUndercuts::FindParams` elementwise.
            public unsafe Const_FindParams(MR.Vector3f upDirection, float wallAngle) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_ConstructFrom(MR.Vector3f upDirection, float wallAngle);
                _UnderlyingPtr = __MR_FixUndercuts_FindParams_ConstructFrom(upDirection, wallAngle);
            }

            /// Generated from constructor `MR::FixUndercuts::FindParams::FindParams`.
            public unsafe Const_FindParams(MR.FixUndercuts.Const_FindParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_ConstructFromAnother(MR.FixUndercuts.FindParams._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_FindParams_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Parameters that is used to find undercuts
        /// Generated from class `MR::FixUndercuts::FindParams`.
        /// This is the non-const half of the class.
        public class FindParams : Const_FindParams
        {
            internal unsafe FindParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Primitives that are not visible from up direction are considered as undercuts (fix undercuts is performed downwards (in `-direction`))
            public new unsafe MR.Mut_Vector3f UpDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_GetMutable_upDirection", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_FixUndercuts_FindParams_GetMutable_upDirection(_Underlying *_this);
                    return new(__MR_FixUndercuts_FindParams_GetMutable_upDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            /// vertical angle of fixed undercut walls (note that this value is approximate - it defines "camera" position for internal projective transformation)
            /// 0 - strictly vertical walls of undercuts area
            /// positive - expanding downwards walls
            /// negative - shrinking downwards walls
            public new unsafe ref float WallAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_GetMutable_wallAngle", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_FindParams_GetMutable_wallAngle(_Underlying *_this);
                    return ref *__MR_FixUndercuts_FindParams_GetMutable_wallAngle(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe FindParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_FindParams_DefaultConstruct();
            }

            /// Constructs `MR::FixUndercuts::FindParams` elementwise.
            public unsafe FindParams(MR.Vector3f upDirection, float wallAngle) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_ConstructFrom(MR.Vector3f upDirection, float wallAngle);
                _UnderlyingPtr = __MR_FixUndercuts_FindParams_ConstructFrom(upDirection, wallAngle);
            }

            /// Generated from constructor `MR::FixUndercuts::FindParams::FindParams`.
            public unsafe FindParams(MR.FixUndercuts.Const_FindParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_ConstructFromAnother(MR.FixUndercuts.FindParams._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_FindParams_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::FixUndercuts::FindParams::operator=`.
            public unsafe MR.FixUndercuts.FindParams Assign(MR.FixUndercuts.Const_FindParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FindParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FindParams_AssignFromAnother(_Underlying *_this, MR.FixUndercuts.FindParams._Underlying *_other);
                return new(__MR_FixUndercuts_FindParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `FindParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_FindParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FindParams`/`Const_FindParams` directly.
        public class _InOptMut_FindParams
        {
            public FindParams? Opt;

            public _InOptMut_FindParams() {}
            public _InOptMut_FindParams(FindParams value) {Opt = value;}
            public static implicit operator _InOptMut_FindParams(FindParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `FindParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_FindParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FindParams`/`Const_FindParams` to pass it to the function.
        public class _InOptConst_FindParams
        {
            public Const_FindParams? Opt;

            public _InOptConst_FindParams() {}
            public _InOptConst_FindParams(Const_FindParams value) {Opt = value;}
            public static implicit operator _InOptConst_FindParams(Const_FindParams value) {return new(value);}
        }

        /// Fix undercuts function parameters
        /// Generated from class `MR::FixUndercuts::FixParams`.
        /// This is the const half of the class.
        public class Const_FixParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_FixParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Destroy", ExactSpelling = true)]
                extern static void __MR_FixUndercuts_FixParams_Destroy(_Underlying *_this);
                __MR_FixUndercuts_FixParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_FixParams() {Dispose(false);}

            /// parameters of what is considered as undercut
            public unsafe MR.FixUndercuts.Const_FindParams FindParameters
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Get_findParameters", ExactSpelling = true)]
                    extern static MR.FixUndercuts.Const_FindParams._Underlying *__MR_FixUndercuts_FixParams_Get_findParameters(_Underlying *_this);
                    return new(__MR_FixUndercuts_FixParams_Get_findParameters(_UnderlyingPtr), is_owning: false);
                }
            }

            /// voxel size for internal computations: lower size - better precision but more system resources required
            public unsafe float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Get_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_FixParams_Get_voxelSize(_Underlying *_this);
                    return *__MR_FixUndercuts_FixParams_Get_voxelSize(_UnderlyingPtr);
                }
            }

            /// minimum extension of bottom part of the mesh
            public unsafe float BottomExtension
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Get_bottomExtension", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_FixParams_Get_bottomExtension(_Underlying *_this);
                    return *__MR_FixUndercuts_FixParams_Get_bottomExtension(_UnderlyingPtr);
                }
            }

            /// if set - only this region will be fixed (but still all mesh will be rebuild)
            public unsafe ref readonly void * Region
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Get_region", ExactSpelling = true)]
                    extern static void **__MR_FixUndercuts_FixParams_Get_region(_Underlying *_this);
                    return ref *__MR_FixUndercuts_FixParams_Get_region(_UnderlyingPtr);
                }
            }

            /// if true applies one iterations of gaussian filtering for voxels, useful if thin walls expected
            public unsafe bool Smooth
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Get_smooth", ExactSpelling = true)]
                    extern static bool *__MR_FixUndercuts_FixParams_Get_smooth(_Underlying *_this);
                    return *__MR_FixUndercuts_FixParams_Get_smooth(_UnderlyingPtr);
                }
            }

            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_Get_cb", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FixUndercuts_FixParams_Get_cb(_Underlying *_this);
                    return new(__MR_FixUndercuts_FixParams_Get_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_FixParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_FixParams_DefaultConstruct();
            }

            /// Constructs `MR::FixUndercuts::FixParams` elementwise.
            public unsafe Const_FixParams(MR.FixUndercuts.Const_FindParams findParameters, float voxelSize, float bottomExtension, MR.Const_FaceBitSet? region, bool smooth, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_ConstructFrom(MR.FixUndercuts.FindParams._Underlying *findParameters, float voxelSize, float bottomExtension, MR.Const_FaceBitSet._Underlying *region, byte smooth, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                _UnderlyingPtr = __MR_FixUndercuts_FixParams_ConstructFrom(findParameters._UnderlyingPtr, voxelSize, bottomExtension, region is not null ? region._UnderlyingPtr : null, smooth ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::FixUndercuts::FixParams::FixParams`.
            public unsafe Const_FixParams(MR.FixUndercuts._ByValue_FixParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FixUndercuts.FixParams._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_FixParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Fix undercuts function parameters
        /// Generated from class `MR::FixUndercuts::FixParams`.
        /// This is the non-const half of the class.
        public class FixParams : Const_FixParams
        {
            internal unsafe FixParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// parameters of what is considered as undercut
            public new unsafe MR.FixUndercuts.FindParams FindParameters
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_GetMutable_findParameters", ExactSpelling = true)]
                    extern static MR.FixUndercuts.FindParams._Underlying *__MR_FixUndercuts_FixParams_GetMutable_findParameters(_Underlying *_this);
                    return new(__MR_FixUndercuts_FixParams_GetMutable_findParameters(_UnderlyingPtr), is_owning: false);
                }
            }

            /// voxel size for internal computations: lower size - better precision but more system resources required
            public new unsafe ref float VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_GetMutable_voxelSize", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_FixParams_GetMutable_voxelSize(_Underlying *_this);
                    return ref *__MR_FixUndercuts_FixParams_GetMutable_voxelSize(_UnderlyingPtr);
                }
            }

            /// minimum extension of bottom part of the mesh
            public new unsafe ref float BottomExtension
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_GetMutable_bottomExtension", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_FixParams_GetMutable_bottomExtension(_Underlying *_this);
                    return ref *__MR_FixUndercuts_FixParams_GetMutable_bottomExtension(_UnderlyingPtr);
                }
            }

            /// if set - only this region will be fixed (but still all mesh will be rebuild)
            public new unsafe ref readonly void * Region
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_GetMutable_region", ExactSpelling = true)]
                    extern static void **__MR_FixUndercuts_FixParams_GetMutable_region(_Underlying *_this);
                    return ref *__MR_FixUndercuts_FixParams_GetMutable_region(_UnderlyingPtr);
                }
            }

            /// if true applies one iterations of gaussian filtering for voxels, useful if thin walls expected
            public new unsafe ref bool Smooth
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_GetMutable_smooth", ExactSpelling = true)]
                    extern static bool *__MR_FixUndercuts_FixParams_GetMutable_smooth(_Underlying *_this);
                    return ref *__MR_FixUndercuts_FixParams_GetMutable_smooth(_UnderlyingPtr);
                }
            }

            public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_GetMutable_cb", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FixUndercuts_FixParams_GetMutable_cb(_Underlying *_this);
                    return new(__MR_FixUndercuts_FixParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe FixParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_FixParams_DefaultConstruct();
            }

            /// Constructs `MR::FixUndercuts::FixParams` elementwise.
            public unsafe FixParams(MR.FixUndercuts.Const_FindParams findParameters, float voxelSize, float bottomExtension, MR.Const_FaceBitSet? region, bool smooth, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_ConstructFrom(MR.FixUndercuts.FindParams._Underlying *findParameters, float voxelSize, float bottomExtension, MR.Const_FaceBitSet._Underlying *region, byte smooth, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                _UnderlyingPtr = __MR_FixUndercuts_FixParams_ConstructFrom(findParameters._UnderlyingPtr, voxelSize, bottomExtension, region is not null ? region._UnderlyingPtr : null, smooth ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::FixUndercuts::FixParams::FixParams`.
            public unsafe FixParams(MR.FixUndercuts._ByValue_FixParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FixUndercuts.FixParams._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_FixParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::FixUndercuts::FixParams::operator=`.
            public unsafe MR.FixUndercuts.FixParams Assign(MR.FixUndercuts._ByValue_FixParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_FixParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.FixParams._Underlying *__MR_FixUndercuts_FixParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FixUndercuts.FixParams._Underlying *_other);
                return new(__MR_FixUndercuts_FixParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `FixParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `FixParams`/`Const_FixParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_FixParams
        {
            internal readonly Const_FixParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_FixParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_FixParams(Const_FixParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_FixParams(Const_FixParams arg) {return new(arg);}
            public _ByValue_FixParams(MR.Misc._Moved<FixParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_FixParams(MR.Misc._Moved<FixParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `FixParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_FixParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FixParams`/`Const_FixParams` directly.
        public class _InOptMut_FixParams
        {
            public FixParams? Opt;

            public _InOptMut_FixParams() {}
            public _InOptMut_FixParams(FixParams value) {Opt = value;}
            public static implicit operator _InOptMut_FixParams(FixParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `FixParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_FixParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FixParams`/`Const_FixParams` to pass it to the function.
        public class _InOptConst_FixParams
        {
            public Const_FixParams? Opt;

            public _InOptConst_FixParams() {}
            public _InOptConst_FixParams(Const_FixParams value) {Opt = value;}
            public static implicit operator _InOptConst_FixParams(Const_FixParams value) {return new(value);}
        }

        /// Generated from class `MR::FixUndercuts::ImproveDirectionParameters`.
        /// Derived classes:
        ///   Direct: (non-virtual)
        ///     `MR::FixUndercuts::DistMapImproveDirectionParameters`
        /// This is the const half of the class.
        public class Const_ImproveDirectionParameters : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ImproveDirectionParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_Destroy", ExactSpelling = true)]
                extern static void __MR_FixUndercuts_ImproveDirectionParameters_Destroy(_Underlying *_this);
                __MR_FixUndercuts_ImproveDirectionParameters_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ImproveDirectionParameters() {Dispose(false);}

            // Hint direction which will be improved
            public unsafe MR.Const_Vector3f HintDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_Get_hintDirection", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_Get_hintDirection(_Underlying *_this);
                    return new(__MR_FixUndercuts_ImproveDirectionParameters_Get_hintDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            // Radial step given in radians look improveDirection comment
            public unsafe float BaseAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_Get_baseAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_ImproveDirectionParameters_Get_baseAngleStep(_Underlying *_this);
                    return *__MR_FixUndercuts_ImproveDirectionParameters_Get_baseAngleStep(_UnderlyingPtr);
                }
            }

            // Maximum radial line given in radians look improveDirection comment
            public unsafe float MaxBaseAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_Get_maxBaseAngle", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_ImproveDirectionParameters_Get_maxBaseAngle(_Underlying *_this);
                    return *__MR_FixUndercuts_ImproveDirectionParameters_Get_maxBaseAngle(_UnderlyingPtr);
                }
            }

            // Polar angle step
            public unsafe float PolarAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_Get_polarAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_ImproveDirectionParameters_Get_polarAngleStep(_Underlying *_this);
                    return *__MR_FixUndercuts_ImproveDirectionParameters_Get_polarAngleStep(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ImproveDirectionParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_ImproveDirectionParameters_DefaultConstruct();
            }

            /// Constructs `MR::FixUndercuts::ImproveDirectionParameters` elementwise.
            public unsafe Const_ImproveDirectionParameters(MR.Vector3f hintDirection, float baseAngleStep, float maxBaseAngle, float polarAngleStep) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_ConstructFrom(MR.Vector3f hintDirection, float baseAngleStep, float maxBaseAngle, float polarAngleStep);
                _UnderlyingPtr = __MR_FixUndercuts_ImproveDirectionParameters_ConstructFrom(hintDirection, baseAngleStep, maxBaseAngle, polarAngleStep);
            }

            /// Generated from constructor `MR::FixUndercuts::ImproveDirectionParameters::ImproveDirectionParameters`.
            public unsafe Const_ImproveDirectionParameters(MR.FixUndercuts.Const_ImproveDirectionParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_ConstructFromAnother(MR.FixUndercuts.ImproveDirectionParameters._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_ImproveDirectionParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::FixUndercuts::ImproveDirectionParameters`.
        /// Derived classes:
        ///   Direct: (non-virtual)
        ///     `MR::FixUndercuts::DistMapImproveDirectionParameters`
        /// This is the non-const half of the class.
        public class ImproveDirectionParameters : Const_ImproveDirectionParameters
        {
            internal unsafe ImproveDirectionParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Hint direction which will be improved
            public new unsafe MR.Mut_Vector3f HintDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_GetMutable_hintDirection", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_hintDirection(_Underlying *_this);
                    return new(__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_hintDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            // Radial step given in radians look improveDirection comment
            public new unsafe ref float BaseAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_GetMutable_baseAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_baseAngleStep(_Underlying *_this);
                    return ref *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_baseAngleStep(_UnderlyingPtr);
                }
            }

            // Maximum radial line given in radians look improveDirection comment
            public new unsafe ref float MaxBaseAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_GetMutable_maxBaseAngle", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_maxBaseAngle(_Underlying *_this);
                    return ref *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_maxBaseAngle(_UnderlyingPtr);
                }
            }

            // Polar angle step
            public new unsafe ref float PolarAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_GetMutable_polarAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_polarAngleStep(_Underlying *_this);
                    return ref *__MR_FixUndercuts_ImproveDirectionParameters_GetMutable_polarAngleStep(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ImproveDirectionParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_ImproveDirectionParameters_DefaultConstruct();
            }

            /// Constructs `MR::FixUndercuts::ImproveDirectionParameters` elementwise.
            public unsafe ImproveDirectionParameters(MR.Vector3f hintDirection, float baseAngleStep, float maxBaseAngle, float polarAngleStep) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_ConstructFrom(MR.Vector3f hintDirection, float baseAngleStep, float maxBaseAngle, float polarAngleStep);
                _UnderlyingPtr = __MR_FixUndercuts_ImproveDirectionParameters_ConstructFrom(hintDirection, baseAngleStep, maxBaseAngle, polarAngleStep);
            }

            /// Generated from constructor `MR::FixUndercuts::ImproveDirectionParameters::ImproveDirectionParameters`.
            public unsafe ImproveDirectionParameters(MR.FixUndercuts.Const_ImproveDirectionParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_ConstructFromAnother(MR.FixUndercuts.ImproveDirectionParameters._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_ImproveDirectionParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::FixUndercuts::ImproveDirectionParameters::operator=`.
            public unsafe MR.FixUndercuts.ImproveDirectionParameters Assign(MR.FixUndercuts.Const_ImproveDirectionParameters _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_ImproveDirectionParameters_AssignFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_ImproveDirectionParameters_AssignFromAnother(_Underlying *_this, MR.FixUndercuts.ImproveDirectionParameters._Underlying *_other);
                return new(__MR_FixUndercuts_ImproveDirectionParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ImproveDirectionParameters` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ImproveDirectionParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ImproveDirectionParameters`/`Const_ImproveDirectionParameters` directly.
        public class _InOptMut_ImproveDirectionParameters
        {
            public ImproveDirectionParameters? Opt;

            public _InOptMut_ImproveDirectionParameters() {}
            public _InOptMut_ImproveDirectionParameters(ImproveDirectionParameters value) {Opt = value;}
            public static implicit operator _InOptMut_ImproveDirectionParameters(ImproveDirectionParameters value) {return new(value);}
        }

        /// This is used for optional parameters of class `ImproveDirectionParameters` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ImproveDirectionParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ImproveDirectionParameters`/`Const_ImproveDirectionParameters` to pass it to the function.
        public class _InOptConst_ImproveDirectionParameters
        {
            public Const_ImproveDirectionParameters? Opt;

            public _InOptConst_ImproveDirectionParameters() {}
            public _InOptConst_ImproveDirectionParameters(Const_ImproveDirectionParameters value) {Opt = value;}
            public static implicit operator _InOptConst_ImproveDirectionParameters(Const_ImproveDirectionParameters value) {return new(value);}
        }

        /// Generated from class `MR::FixUndercuts::DistMapImproveDirectionParameters`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::FixUndercuts::ImproveDirectionParameters`
        /// This is the const half of the class.
        public class Const_DistMapImproveDirectionParameters : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_DistMapImproveDirectionParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_Destroy", ExactSpelling = true)]
                extern static void __MR_FixUndercuts_DistMapImproveDirectionParameters_Destroy(_Underlying *_this);
                __MR_FixUndercuts_DistMapImproveDirectionParameters_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_DistMapImproveDirectionParameters() {Dispose(false);}

            // Upcasts:
            public static unsafe implicit operator MR.FixUndercuts.Const_ImproveDirectionParameters(Const_DistMapImproveDirectionParameters self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_UpcastTo_MR_FixUndercuts_ImproveDirectionParameters", ExactSpelling = true)]
                extern static MR.FixUndercuts.Const_ImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_UpcastTo_MR_FixUndercuts_ImproveDirectionParameters(_Underlying *_this);
                MR.FixUndercuts.Const_ImproveDirectionParameters ret = new(__MR_FixUndercuts_DistMapImproveDirectionParameters_UpcastTo_MR_FixUndercuts_ImproveDirectionParameters(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            // Resolution of distance map, lower it is, faster score works
            public unsafe MR.Const_Vector2i DistanceMapResolution
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_Get_distanceMapResolution", ExactSpelling = true)]
                    extern static MR.Const_Vector2i._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_distanceMapResolution(_Underlying *_this);
                    return new(__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_distanceMapResolution(_UnderlyingPtr), is_owning: false);
                }
            }

            // Hint direction which will be improved
            public unsafe MR.Const_Vector3f HintDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_Get_hintDirection", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_hintDirection(_Underlying *_this);
                    return new(__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_hintDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            // Radial step given in radians look improveDirection comment
            public unsafe float BaseAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_Get_baseAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_baseAngleStep(_Underlying *_this);
                    return *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_baseAngleStep(_UnderlyingPtr);
                }
            }

            // Maximum radial line given in radians look improveDirection comment
            public unsafe float MaxBaseAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_Get_maxBaseAngle", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_maxBaseAngle(_Underlying *_this);
                    return *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_maxBaseAngle(_UnderlyingPtr);
                }
            }

            // Polar angle step
            public unsafe float PolarAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_Get_polarAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_polarAngleStep(_Underlying *_this);
                    return *__MR_FixUndercuts_DistMapImproveDirectionParameters_Get_polarAngleStep(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_DistMapImproveDirectionParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_DistMapImproveDirectionParameters_DefaultConstruct();
            }

            /// Generated from constructor `MR::FixUndercuts::DistMapImproveDirectionParameters::DistMapImproveDirectionParameters`.
            public unsafe Const_DistMapImproveDirectionParameters(MR.FixUndercuts.Const_DistMapImproveDirectionParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_ConstructFromAnother(MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_DistMapImproveDirectionParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::FixUndercuts::DistMapImproveDirectionParameters`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::FixUndercuts::ImproveDirectionParameters`
        /// This is the non-const half of the class.
        public class DistMapImproveDirectionParameters : Const_DistMapImproveDirectionParameters
        {
            internal unsafe DistMapImproveDirectionParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Upcasts:
            public static unsafe implicit operator MR.FixUndercuts.ImproveDirectionParameters(DistMapImproveDirectionParameters self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_UpcastTo_MR_FixUndercuts_ImproveDirectionParameters", ExactSpelling = true)]
                extern static MR.FixUndercuts.ImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_UpcastTo_MR_FixUndercuts_ImproveDirectionParameters(_Underlying *_this);
                MR.FixUndercuts.ImproveDirectionParameters ret = new(__MR_FixUndercuts_DistMapImproveDirectionParameters_UpcastTo_MR_FixUndercuts_ImproveDirectionParameters(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            // Resolution of distance map, lower it is, faster score works
            public new unsafe MR.Mut_Vector2i DistanceMapResolution
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_distanceMapResolution", ExactSpelling = true)]
                    extern static MR.Mut_Vector2i._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_distanceMapResolution(_Underlying *_this);
                    return new(__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_distanceMapResolution(_UnderlyingPtr), is_owning: false);
                }
            }

            // Hint direction which will be improved
            public new unsafe MR.Mut_Vector3f HintDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_hintDirection", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_hintDirection(_Underlying *_this);
                    return new(__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_hintDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            // Radial step given in radians look improveDirection comment
            public new unsafe ref float BaseAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_baseAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_baseAngleStep(_Underlying *_this);
                    return ref *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_baseAngleStep(_UnderlyingPtr);
                }
            }

            // Maximum radial line given in radians look improveDirection comment
            public new unsafe ref float MaxBaseAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_maxBaseAngle", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_maxBaseAngle(_Underlying *_this);
                    return ref *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_maxBaseAngle(_UnderlyingPtr);
                }
            }

            // Polar angle step
            public new unsafe ref float PolarAngleStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_polarAngleStep", ExactSpelling = true)]
                    extern static float *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_polarAngleStep(_Underlying *_this);
                    return ref *__MR_FixUndercuts_DistMapImproveDirectionParameters_GetMutable_polarAngleStep(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe DistMapImproveDirectionParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_FixUndercuts_DistMapImproveDirectionParameters_DefaultConstruct();
            }

            /// Generated from constructor `MR::FixUndercuts::DistMapImproveDirectionParameters::DistMapImproveDirectionParameters`.
            public unsafe DistMapImproveDirectionParameters(MR.FixUndercuts.Const_DistMapImproveDirectionParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_ConstructFromAnother(MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *_other);
                _UnderlyingPtr = __MR_FixUndercuts_DistMapImproveDirectionParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::FixUndercuts::DistMapImproveDirectionParameters::operator=`.
            public unsafe MR.FixUndercuts.DistMapImproveDirectionParameters Assign(MR.FixUndercuts.Const_DistMapImproveDirectionParameters _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_DistMapImproveDirectionParameters_AssignFromAnother", ExactSpelling = true)]
                extern static MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *__MR_FixUndercuts_DistMapImproveDirectionParameters_AssignFromAnother(_Underlying *_this, MR.FixUndercuts.DistMapImproveDirectionParameters._Underlying *_other);
                return new(__MR_FixUndercuts_DistMapImproveDirectionParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `DistMapImproveDirectionParameters` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_DistMapImproveDirectionParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DistMapImproveDirectionParameters`/`Const_DistMapImproveDirectionParameters` directly.
        public class _InOptMut_DistMapImproveDirectionParameters
        {
            public DistMapImproveDirectionParameters? Opt;

            public _InOptMut_DistMapImproveDirectionParameters() {}
            public _InOptMut_DistMapImproveDirectionParameters(DistMapImproveDirectionParameters value) {Opt = value;}
            public static implicit operator _InOptMut_DistMapImproveDirectionParameters(DistMapImproveDirectionParameters value) {return new(value);}
        }

        /// This is used for optional parameters of class `DistMapImproveDirectionParameters` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_DistMapImproveDirectionParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DistMapImproveDirectionParameters`/`Const_DistMapImproveDirectionParameters` to pass it to the function.
        public class _InOptConst_DistMapImproveDirectionParameters
        {
            public Const_DistMapImproveDirectionParameters? Opt;

            public _InOptConst_DistMapImproveDirectionParameters() {}
            public _InOptConst_DistMapImproveDirectionParameters(Const_DistMapImproveDirectionParameters value) {Opt = value;}
            public static implicit operator _InOptConst_DistMapImproveDirectionParameters(Const_DistMapImproveDirectionParameters value) {return new(value);}
        }

        /// Fixes undercut areas by building vertical walls under it,
        /// algorithm is performed in voxel space, so the mesh is completely rebuilt after this operation
        /// Generated from function `MR::FixUndercuts::fix`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> Fix(MR.Mesh mesh, MR.FixUndercuts.Const_FixParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_fix", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_FixUndercuts_fix(MR.Mesh._Underlying *mesh, MR.FixUndercuts.Const_FixParams._Underlying *params_);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_FixUndercuts_fix(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
        }

        /// returns the metric that computes total area of undercut faces
        /// Generated from function `MR::FixUndercuts::getUndercutAreaMetric`.
        public static unsafe MR.Misc._Moved<MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef> GetUndercutAreaMetric(MR.Const_Mesh mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_getUndercutAreaMetric", ExactSpelling = true)]
            extern static MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *__MR_FixUndercuts_getUndercutAreaMetric(MR.Const_Mesh._Underlying *mesh);
            return MR.Misc.Move(new MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(__MR_FixUndercuts_getUndercutAreaMetric(mesh._UnderlyingPtr), is_owning: true));
        }

        /// returns the metric that computes summed absolute projected area of undercut
        /// Generated from function `MR::FixUndercuts::getUndercutAreaProjectionMetric`.
        public static unsafe MR.Misc._Moved<MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef> GetUndercutAreaProjectionMetric(MR.Const_Mesh mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_getUndercutAreaProjectionMetric", ExactSpelling = true)]
            extern static MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *__MR_FixUndercuts_getUndercutAreaProjectionMetric(MR.Const_Mesh._Underlying *mesh);
            return MR.Misc.Move(new MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(__MR_FixUndercuts_getUndercutAreaProjectionMetric(mesh._UnderlyingPtr), is_owning: true));
        }

        /// Adds undercuts to \param outUndercuts
        /// if metric is set returns metric of found undercuts, otherwise returns DBL_MAX
        /// Generated from function `MR::FixUndercuts::find`.
        /// Parameter `metric` defaults to `{}`.
        public static unsafe double Find(MR.Const_Mesh mesh, MR.FixUndercuts.Const_FindParams params_, MR.FaceBitSet outUndercuts, MR.Std.Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef? metric = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_find_4", ExactSpelling = true)]
            extern static double __MR_FixUndercuts_find_4(MR.Const_Mesh._Underlying *mesh, MR.FixUndercuts.Const_FindParams._Underlying *params_, MR.FaceBitSet._Underlying *outUndercuts, MR.Std.Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *metric);
            return __MR_FixUndercuts_find_4(mesh._UnderlyingPtr, params_._UnderlyingPtr, outUndercuts._UnderlyingPtr, metric is not null ? metric._UnderlyingPtr : null);
        }

        /// Adds undercuts to \param outUndercuts
        /// Generated from function `MR::FixUndercuts::find`.
        public static unsafe void Find(MR.Const_Mesh mesh, MR.FixUndercuts.Const_FindParams params_, MR.VertBitSet outUndercuts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_find_3", ExactSpelling = true)]
            extern static void __MR_FixUndercuts_find_3(MR.Const_Mesh._Underlying *mesh, MR.FixUndercuts.Const_FindParams._Underlying *params_, MR.VertBitSet._Underlying *outUndercuts);
            __MR_FixUndercuts_find_3(mesh._UnderlyingPtr, params_._UnderlyingPtr, outUndercuts._UnderlyingPtr);
        }

        /// Fast score undercuts projected area via distance map with given resolution
        /// lower resolution means lower precision, but faster work
        /// \note does not support wallAngle yet
        /// Generated from function `MR::FixUndercuts::scoreUndercuts`.
        public static unsafe double ScoreUndercuts(MR.Const_Mesh mesh, MR.Const_Vector3f upDirection, MR.Const_Vector2i resolution)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_scoreUndercuts", ExactSpelling = true)]
            extern static double __MR_FixUndercuts_scoreUndercuts(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector3f._Underlying *upDirection, MR.Const_Vector2i._Underlying *resolution);
            return __MR_FixUndercuts_scoreUndercuts(mesh._UnderlyingPtr, upDirection._UnderlyingPtr, resolution._UnderlyingPtr);
        }

        // Parallel finds best of several directions defined by ImproveDirectionParameters struct
        /// \note does not support wallAngle yet
        ///                      ________
        ///        Top view:    /  \__/  \-----> maximum radial line   Side view:  |    /    _/
        ///                    /  / \/ \  \                                        |   /   _/ - maxBaseAngle
        ///                   |--|------|--|                                       |  /  _/     difference between two angles is baseAngleStep
        ///                    \  \_/\_/  /                                        | / _/
        ///                     \__/__\__/                                         |/_/
        /// This picture shows polarAngle = 60 deg
        /// Generated from function `MR::FixUndercuts::improveDirection`.
        public static unsafe MR.Vector3f ImproveDirection(MR.Const_Mesh mesh, MR.FixUndercuts.Const_ImproveDirectionParameters params_, MR.Std.Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef metric)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_improveDirection", ExactSpelling = true)]
            extern static MR.Vector3f __MR_FixUndercuts_improveDirection(MR.Const_Mesh._Underlying *mesh, MR.FixUndercuts.Const_ImproveDirectionParameters._Underlying *params_, MR.Std.Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *metric);
            return __MR_FixUndercuts_improveDirection(mesh._UnderlyingPtr, params_._UnderlyingPtr, metric._UnderlyingPtr);
        }

        /// Score candidates with distance maps, lower resolution -> faster score
        /// \note does not support wallAngle yet
        /// Generated from function `MR::FixUndercuts::distMapImproveDirection`.
        public static unsafe MR.Vector3f DistMapImproveDirection(MR.Const_Mesh mesh, MR.FixUndercuts.Const_DistMapImproveDirectionParameters params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixUndercuts_distMapImproveDirection", ExactSpelling = true)]
            extern static MR.Vector3f __MR_FixUndercuts_distMapImproveDirection(MR.Const_Mesh._Underlying *mesh, MR.FixUndercuts.Const_DistMapImproveDirectionParameters._Underlying *params_);
            return __MR_FixUndercuts_distMapImproveDirection(mesh._UnderlyingPtr, params_._UnderlyingPtr);
        }
    }
}
