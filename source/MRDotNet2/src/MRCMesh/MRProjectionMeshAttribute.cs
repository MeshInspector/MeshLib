public static partial class MR
{
    /// this structure contains transformation for projection from one mesh to another and progress callback
    /// Generated from class `MR::ProjectAttributeParams`.
    /// This is the const half of the class.
    public class Const_ProjectAttributeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ProjectAttributeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ProjectAttributeParams_Destroy(_Underlying *_this);
            __MR_ProjectAttributeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ProjectAttributeParams() {Dispose(false);}

        public unsafe MR.Const_MeshProjectionTransforms Xfs
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_Get_xfs", ExactSpelling = true)]
                extern static MR.Const_MeshProjectionTransforms._Underlying *__MR_ProjectAttributeParams_Get_xfs(_Underlying *_this);
                return new(__MR_ProjectAttributeParams_Get_xfs(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_Get_progressCb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ProjectAttributeParams_Get_progressCb(_Underlying *_this);
                return new(__MR_ProjectAttributeParams_Get_progressCb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ProjectAttributeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ProjectAttributeParams_DefaultConstruct();
        }

        /// Constructs `MR::ProjectAttributeParams` elementwise.
        public unsafe Const_ProjectAttributeParams(MR.Const_MeshProjectionTransforms xfs, MR.Std._ByValue_Function_BoolFuncFromFloat progressCb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_ConstructFrom(MR.MeshProjectionTransforms._Underlying *xfs, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            _UnderlyingPtr = __MR_ProjectAttributeParams_ConstructFrom(xfs._UnderlyingPtr, progressCb.PassByMode, progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ProjectAttributeParams::ProjectAttributeParams`.
        public unsafe Const_ProjectAttributeParams(MR._ByValue_ProjectAttributeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ProjectAttributeParams._Underlying *_other);
            _UnderlyingPtr = __MR_ProjectAttributeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// this structure contains transformation for projection from one mesh to another and progress callback
    /// Generated from class `MR::ProjectAttributeParams`.
    /// This is the non-const half of the class.
    public class ProjectAttributeParams : Const_ProjectAttributeParams
    {
        internal unsafe ProjectAttributeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.MeshProjectionTransforms Xfs
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_GetMutable_xfs", ExactSpelling = true)]
                extern static MR.MeshProjectionTransforms._Underlying *__MR_ProjectAttributeParams_GetMutable_xfs(_Underlying *_this);
                return new(__MR_ProjectAttributeParams_GetMutable_xfs(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_GetMutable_progressCb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ProjectAttributeParams_GetMutable_progressCb(_Underlying *_this);
                return new(__MR_ProjectAttributeParams_GetMutable_progressCb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ProjectAttributeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ProjectAttributeParams_DefaultConstruct();
        }

        /// Constructs `MR::ProjectAttributeParams` elementwise.
        public unsafe ProjectAttributeParams(MR.Const_MeshProjectionTransforms xfs, MR.Std._ByValue_Function_BoolFuncFromFloat progressCb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_ConstructFrom(MR.MeshProjectionTransforms._Underlying *xfs, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            _UnderlyingPtr = __MR_ProjectAttributeParams_ConstructFrom(xfs._UnderlyingPtr, progressCb.PassByMode, progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ProjectAttributeParams::ProjectAttributeParams`.
        public unsafe ProjectAttributeParams(MR._ByValue_ProjectAttributeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ProjectAttributeParams._Underlying *_other);
            _UnderlyingPtr = __MR_ProjectAttributeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ProjectAttributeParams::operator=`.
        public unsafe MR.ProjectAttributeParams Assign(MR._ByValue_ProjectAttributeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ProjectAttributeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ProjectAttributeParams._Underlying *__MR_ProjectAttributeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ProjectAttributeParams._Underlying *_other);
            return new(__MR_ProjectAttributeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ProjectAttributeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ProjectAttributeParams`/`Const_ProjectAttributeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ProjectAttributeParams
    {
        internal readonly Const_ProjectAttributeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ProjectAttributeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ProjectAttributeParams(Const_ProjectAttributeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ProjectAttributeParams(Const_ProjectAttributeParams arg) {return new(arg);}
        public _ByValue_ProjectAttributeParams(MR.Misc._Moved<ProjectAttributeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ProjectAttributeParams(MR.Misc._Moved<ProjectAttributeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ProjectAttributeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ProjectAttributeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ProjectAttributeParams`/`Const_ProjectAttributeParams` directly.
    public class _InOptMut_ProjectAttributeParams
    {
        public ProjectAttributeParams? Opt;

        public _InOptMut_ProjectAttributeParams() {}
        public _InOptMut_ProjectAttributeParams(ProjectAttributeParams value) {Opt = value;}
        public static implicit operator _InOptMut_ProjectAttributeParams(ProjectAttributeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ProjectAttributeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ProjectAttributeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ProjectAttributeParams`/`Const_ProjectAttributeParams` to pass it to the function.
    public class _InOptConst_ProjectAttributeParams
    {
        public Const_ProjectAttributeParams? Opt;

        public _InOptConst_ProjectAttributeParams() {}
        public _InOptConst_ProjectAttributeParams(Const_ProjectAttributeParams value) {Opt = value;}
        public static implicit operator _InOptConst_ProjectAttributeParams(Const_ProjectAttributeParams value) {return new(value);}
    }

    /// <summary>
    /// finds attributes of new mesh by projecting faces/vertices on old mesh
    /// \note for now clears edges attributes
    /// </summary>
    /// <param name="oldMeshData">old mesh along with input attributes</param>
    /// <param name="newMeshData">new mesh along with outpuyt attributes</param>
    /// <param name="region">optional input region for projecting (usefull if newMesh is changed part of old mesh)</param>
    /// <param name="params">parameters of prohecting</param>
    /// Generated from function `MR::projectObjectMeshData`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ProjectObjectMeshData(MR.Const_ObjectMeshData oldMeshData, MR.ObjectMeshData newMeshData, MR.Const_FaceBitSet? region = null, MR.Const_ProjectAttributeParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_projectObjectMeshData", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_projectObjectMeshData(MR.Const_ObjectMeshData._Underlying *oldMeshData, MR.ObjectMeshData._Underlying *newMeshData, MR.Const_FaceBitSet._Underlying *region, MR.Const_ProjectAttributeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_projectObjectMeshData(oldMeshData._UnderlyingPtr, newMeshData._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
