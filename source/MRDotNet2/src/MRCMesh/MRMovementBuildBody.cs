public static partial class MR
{
    /// Generated from class `MR::MovementBuildBodyParams`.
    /// This is the const half of the class.
    public class Const_MovementBuildBodyParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MovementBuildBodyParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MovementBuildBodyParams_Destroy(_Underlying *_this);
            __MR_MovementBuildBodyParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MovementBuildBodyParams() {Dispose(false);}

        /// if this flag is set, rotate body in trajectory vertices
        /// according to its rotation
        /// otherwise body movement will be done without any rotation
        public unsafe bool AllowRotation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_Get_allowRotation", ExactSpelling = true)]
                extern static bool *__MR_MovementBuildBodyParams_Get_allowRotation(_Underlying *_this);
                return *__MR_MovementBuildBodyParams_Get_allowRotation(_UnderlyingPtr);
            }
        }

        /// point in body space that follows trajectory
        /// if not set body bounding box center is used
        public unsafe MR.Std.Const_Optional_MRVector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_Get_center", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRVector3f._Underlying *__MR_MovementBuildBodyParams_Get_center(_Underlying *_this);
                return new(__MR_MovementBuildBodyParams_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        /// facing direction of body, used for initial rotation (if allowRotation)
        /// if not set body accumulative normal is used
        public unsafe MR.Std.Const_Optional_MRVector3f BodyNormal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_Get_bodyNormal", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRVector3f._Underlying *__MR_MovementBuildBodyParams_Get_bodyNormal(_Underlying *_this);
                return new(__MR_MovementBuildBodyParams_Get_bodyNormal(_UnderlyingPtr), is_owning: false);
            }
        }

        /// optional transform body space to trajectory space
        public unsafe ref readonly MR.AffineXf3f * B2tXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_Get_b2tXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MovementBuildBodyParams_Get_b2tXf(_Underlying *_this);
                return ref *__MR_MovementBuildBodyParams_Get_b2tXf(_UnderlyingPtr);
            }
        }

        /// if true, then body-contours will be located exactly on resulting mesh
        public unsafe bool StartMeshFromBody
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_Get_startMeshFromBody", ExactSpelling = true)]
                extern static bool *__MR_MovementBuildBodyParams_Get_startMeshFromBody(_Underlying *_this);
                return *__MR_MovementBuildBodyParams_Get_startMeshFromBody(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MovementBuildBodyParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MovementBuildBodyParams_DefaultConstruct();
        }

        /// Constructs `MR::MovementBuildBodyParams` elementwise.
        public unsafe Const_MovementBuildBodyParams(bool allowRotation, MR._InOpt_Vector3f center, MR._InOpt_Vector3f bodyNormal, MR.Const_AffineXf3f? b2tXf, bool startMeshFromBody) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_ConstructFrom(byte allowRotation, MR.Vector3f *center, MR.Vector3f *bodyNormal, MR.Const_AffineXf3f._Underlying *b2tXf, byte startMeshFromBody);
            _UnderlyingPtr = __MR_MovementBuildBodyParams_ConstructFrom(allowRotation ? (byte)1 : (byte)0, center.HasValue ? &center.Object : null, bodyNormal.HasValue ? &bodyNormal.Object : null, b2tXf is not null ? b2tXf._UnderlyingPtr : null, startMeshFromBody ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::MovementBuildBodyParams::MovementBuildBodyParams`.
        public unsafe Const_MovementBuildBodyParams(MR.Const_MovementBuildBodyParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_ConstructFromAnother(MR.MovementBuildBodyParams._Underlying *_other);
            _UnderlyingPtr = __MR_MovementBuildBodyParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MovementBuildBodyParams`.
    /// This is the non-const half of the class.
    public class MovementBuildBodyParams : Const_MovementBuildBodyParams
    {
        internal unsafe MovementBuildBodyParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// if this flag is set, rotate body in trajectory vertices
        /// according to its rotation
        /// otherwise body movement will be done without any rotation
        public new unsafe ref bool AllowRotation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_GetMutable_allowRotation", ExactSpelling = true)]
                extern static bool *__MR_MovementBuildBodyParams_GetMutable_allowRotation(_Underlying *_this);
                return ref *__MR_MovementBuildBodyParams_GetMutable_allowRotation(_UnderlyingPtr);
            }
        }

        /// point in body space that follows trajectory
        /// if not set body bounding box center is used
        public new unsafe MR.Std.Optional_MRVector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_GetMutable_center", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3f._Underlying *__MR_MovementBuildBodyParams_GetMutable_center(_Underlying *_this);
                return new(__MR_MovementBuildBodyParams_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        /// facing direction of body, used for initial rotation (if allowRotation)
        /// if not set body accumulative normal is used
        public new unsafe MR.Std.Optional_MRVector3f BodyNormal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_GetMutable_bodyNormal", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3f._Underlying *__MR_MovementBuildBodyParams_GetMutable_bodyNormal(_Underlying *_this);
                return new(__MR_MovementBuildBodyParams_GetMutable_bodyNormal(_UnderlyingPtr), is_owning: false);
            }
        }

        /// optional transform body space to trajectory space
        public new unsafe ref readonly MR.AffineXf3f * B2tXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_GetMutable_b2tXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MovementBuildBodyParams_GetMutable_b2tXf(_Underlying *_this);
                return ref *__MR_MovementBuildBodyParams_GetMutable_b2tXf(_UnderlyingPtr);
            }
        }

        /// if true, then body-contours will be located exactly on resulting mesh
        public new unsafe ref bool StartMeshFromBody
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_GetMutable_startMeshFromBody", ExactSpelling = true)]
                extern static bool *__MR_MovementBuildBodyParams_GetMutable_startMeshFromBody(_Underlying *_this);
                return ref *__MR_MovementBuildBodyParams_GetMutable_startMeshFromBody(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MovementBuildBodyParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MovementBuildBodyParams_DefaultConstruct();
        }

        /// Constructs `MR::MovementBuildBodyParams` elementwise.
        public unsafe MovementBuildBodyParams(bool allowRotation, MR._InOpt_Vector3f center, MR._InOpt_Vector3f bodyNormal, MR.Const_AffineXf3f? b2tXf, bool startMeshFromBody) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_ConstructFrom(byte allowRotation, MR.Vector3f *center, MR.Vector3f *bodyNormal, MR.Const_AffineXf3f._Underlying *b2tXf, byte startMeshFromBody);
            _UnderlyingPtr = __MR_MovementBuildBodyParams_ConstructFrom(allowRotation ? (byte)1 : (byte)0, center.HasValue ? &center.Object : null, bodyNormal.HasValue ? &bodyNormal.Object : null, b2tXf is not null ? b2tXf._UnderlyingPtr : null, startMeshFromBody ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::MovementBuildBodyParams::MovementBuildBodyParams`.
        public unsafe MovementBuildBodyParams(MR.Const_MovementBuildBodyParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_ConstructFromAnother(MR.MovementBuildBodyParams._Underlying *_other);
            _UnderlyingPtr = __MR_MovementBuildBodyParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MovementBuildBodyParams::operator=`.
        public unsafe MR.MovementBuildBodyParams Assign(MR.Const_MovementBuildBodyParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MovementBuildBodyParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MovementBuildBodyParams._Underlying *__MR_MovementBuildBodyParams_AssignFromAnother(_Underlying *_this, MR.MovementBuildBodyParams._Underlying *_other);
            return new(__MR_MovementBuildBodyParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MovementBuildBodyParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MovementBuildBodyParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MovementBuildBodyParams`/`Const_MovementBuildBodyParams` directly.
    public class _InOptMut_MovementBuildBodyParams
    {
        public MovementBuildBodyParams? Opt;

        public _InOptMut_MovementBuildBodyParams() {}
        public _InOptMut_MovementBuildBodyParams(MovementBuildBodyParams value) {Opt = value;}
        public static implicit operator _InOptMut_MovementBuildBodyParams(MovementBuildBodyParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MovementBuildBodyParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MovementBuildBodyParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MovementBuildBodyParams`/`Const_MovementBuildBodyParams` to pass it to the function.
    public class _InOptConst_MovementBuildBodyParams
    {
        public Const_MovementBuildBodyParams? Opt;

        public _InOptConst_MovementBuildBodyParams() {}
        public _InOptConst_MovementBuildBodyParams(Const_MovementBuildBodyParams value) {Opt = value;}
        public static implicit operator _InOptConst_MovementBuildBodyParams(Const_MovementBuildBodyParams value) {return new(value);}
    }

    /// makes mesh by moving `body` along `trajectory`
    /// if allowRotation rotate it in corners
    /// Generated from function `MR::makeMovementBuildBody`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeMovementBuildBody(MR.Std.Const_Vector_StdVectorMRVector3f body, MR.Std.Const_Vector_StdVectorMRVector3f trajectory, MR.Const_MovementBuildBodyParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeMovementBuildBody", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeMovementBuildBody(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *body, MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *trajectory, MR.Const_MovementBuildBodyParams._Underlying *params_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeMovementBuildBody(body._UnderlyingPtr, trajectory._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
