public static partial class MR
{
    /// structure with parameters for `compensateRadius` function
    /// Generated from class `MR::CompensateRadiusParams`.
    /// This is the const half of the class.
    public class Const_CompensateRadiusParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CompensateRadiusParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Destroy", ExactSpelling = true)]
            extern static void __MR_CompensateRadiusParams_Destroy(_Underlying *_this);
            __MR_CompensateRadiusParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CompensateRadiusParams() {Dispose(false);}

        /// Z direction of milling tool
        public unsafe MR.Const_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_direction", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_CompensateRadiusParams_Get_direction(_Underlying *_this);
                return new(__MR_CompensateRadiusParams_Get_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        ///  radius of spherical tool
        public unsafe float ToolRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_toolRadius", ExactSpelling = true)]
                extern static float *__MR_CompensateRadiusParams_Get_toolRadius(_Underlying *_this);
                return *__MR_CompensateRadiusParams_Get_toolRadius(_UnderlyingPtr);
            }
        }

        /// region of the mesh that will be compensated
        /// it should not contain closed components
        /// also please note that boundaries of the region are fixed
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_CompensateRadiusParams_Get_region(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_Get_region(_UnderlyingPtr);
            }
        }

        /// maximum iteration of applying algorithm (each iteration improves result a little bit)
        public unsafe int MaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_maxIterations", ExactSpelling = true)]
                extern static int *__MR_CompensateRadiusParams_Get_maxIterations(_Underlying *_this);
                return *__MR_CompensateRadiusParams_Get_maxIterations(_UnderlyingPtr);
            }
        }

        /// how many hops to expand around each moved vertex for relaxation
        public unsafe int RelaxExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_relaxExpansion", ExactSpelling = true)]
                extern static int *__MR_CompensateRadiusParams_Get_relaxExpansion(_Underlying *_this);
                return *__MR_CompensateRadiusParams_Get_relaxExpansion(_UnderlyingPtr);
            }
        }

        /// how many iterations of relax is applied on each compensation iteration
        public unsafe int RelaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_relaxIterations", ExactSpelling = true)]
                extern static int *__MR_CompensateRadiusParams_Get_relaxIterations(_Underlying *_this);
                return *__MR_CompensateRadiusParams_Get_relaxIterations(_UnderlyingPtr);
            }
        }

        /// force of relaxations on each compensation iteration
        public unsafe float RelaxForce
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_relaxForce", ExactSpelling = true)]
                extern static float *__MR_CompensateRadiusParams_Get_relaxForce(_Underlying *_this);
                return *__MR_CompensateRadiusParams_Get_relaxForce(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_CompensateRadiusParams_Get_callback(_Underlying *_this);
                return new(__MR_CompensateRadiusParams_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CompensateRadiusParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_DefaultConstruct();
            _UnderlyingPtr = __MR_CompensateRadiusParams_DefaultConstruct();
        }

        /// Constructs `MR::CompensateRadiusParams` elementwise.
        public unsafe Const_CompensateRadiusParams(MR.Vector3f direction, float toolRadius, MR.Const_FaceBitSet? region, int maxIterations, int relaxExpansion, int relaxIterations, float relaxForce, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_ConstructFrom(MR.Vector3f direction, float toolRadius, MR.Const_FaceBitSet._Underlying *region, int maxIterations, int relaxExpansion, int relaxIterations, float relaxForce, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_CompensateRadiusParams_ConstructFrom(direction, toolRadius, region is not null ? region._UnderlyingPtr : null, maxIterations, relaxExpansion, relaxIterations, relaxForce, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CompensateRadiusParams::CompensateRadiusParams`.
        public unsafe Const_CompensateRadiusParams(MR._ByValue_CompensateRadiusParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CompensateRadiusParams._Underlying *_other);
            _UnderlyingPtr = __MR_CompensateRadiusParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// structure with parameters for `compensateRadius` function
    /// Generated from class `MR::CompensateRadiusParams`.
    /// This is the non-const half of the class.
    public class CompensateRadiusParams : Const_CompensateRadiusParams
    {
        internal unsafe CompensateRadiusParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Z direction of milling tool
        public new unsafe MR.Mut_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_direction", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_CompensateRadiusParams_GetMutable_direction(_Underlying *_this);
                return new(__MR_CompensateRadiusParams_GetMutable_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        ///  radius of spherical tool
        public new unsafe ref float ToolRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_toolRadius", ExactSpelling = true)]
                extern static float *__MR_CompensateRadiusParams_GetMutable_toolRadius(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_GetMutable_toolRadius(_UnderlyingPtr);
            }
        }

        /// region of the mesh that will be compensated
        /// it should not contain closed components
        /// also please note that boundaries of the region are fixed
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_CompensateRadiusParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// maximum iteration of applying algorithm (each iteration improves result a little bit)
        public new unsafe ref int MaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_maxIterations", ExactSpelling = true)]
                extern static int *__MR_CompensateRadiusParams_GetMutable_maxIterations(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_GetMutable_maxIterations(_UnderlyingPtr);
            }
        }

        /// how many hops to expand around each moved vertex for relaxation
        public new unsafe ref int RelaxExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_relaxExpansion", ExactSpelling = true)]
                extern static int *__MR_CompensateRadiusParams_GetMutable_relaxExpansion(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_GetMutable_relaxExpansion(_UnderlyingPtr);
            }
        }

        /// how many iterations of relax is applied on each compensation iteration
        public new unsafe ref int RelaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_relaxIterations", ExactSpelling = true)]
                extern static int *__MR_CompensateRadiusParams_GetMutable_relaxIterations(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_GetMutable_relaxIterations(_UnderlyingPtr);
            }
        }

        /// force of relaxations on each compensation iteration
        public new unsafe ref float RelaxForce
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_relaxForce", ExactSpelling = true)]
                extern static float *__MR_CompensateRadiusParams_GetMutable_relaxForce(_Underlying *_this);
                return ref *__MR_CompensateRadiusParams_GetMutable_relaxForce(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_CompensateRadiusParams_GetMutable_callback(_Underlying *_this);
                return new(__MR_CompensateRadiusParams_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CompensateRadiusParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_DefaultConstruct();
            _UnderlyingPtr = __MR_CompensateRadiusParams_DefaultConstruct();
        }

        /// Constructs `MR::CompensateRadiusParams` elementwise.
        public unsafe CompensateRadiusParams(MR.Vector3f direction, float toolRadius, MR.Const_FaceBitSet? region, int maxIterations, int relaxExpansion, int relaxIterations, float relaxForce, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_ConstructFrom(MR.Vector3f direction, float toolRadius, MR.Const_FaceBitSet._Underlying *region, int maxIterations, int relaxExpansion, int relaxIterations, float relaxForce, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_CompensateRadiusParams_ConstructFrom(direction, toolRadius, region is not null ? region._UnderlyingPtr : null, maxIterations, relaxExpansion, relaxIterations, relaxForce, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CompensateRadiusParams::CompensateRadiusParams`.
        public unsafe CompensateRadiusParams(MR._ByValue_CompensateRadiusParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CompensateRadiusParams._Underlying *_other);
            _UnderlyingPtr = __MR_CompensateRadiusParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CompensateRadiusParams::operator=`.
        public unsafe MR.CompensateRadiusParams Assign(MR._ByValue_CompensateRadiusParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CompensateRadiusParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CompensateRadiusParams._Underlying *__MR_CompensateRadiusParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CompensateRadiusParams._Underlying *_other);
            return new(__MR_CompensateRadiusParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `CompensateRadiusParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CompensateRadiusParams`/`Const_CompensateRadiusParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CompensateRadiusParams
    {
        internal readonly Const_CompensateRadiusParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CompensateRadiusParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_CompensateRadiusParams(Const_CompensateRadiusParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CompensateRadiusParams(Const_CompensateRadiusParams arg) {return new(arg);}
        public _ByValue_CompensateRadiusParams(MR.Misc._Moved<CompensateRadiusParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CompensateRadiusParams(MR.Misc._Moved<CompensateRadiusParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CompensateRadiusParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CompensateRadiusParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CompensateRadiusParams`/`Const_CompensateRadiusParams` directly.
    public class _InOptMut_CompensateRadiusParams
    {
        public CompensateRadiusParams? Opt;

        public _InOptMut_CompensateRadiusParams() {}
        public _InOptMut_CompensateRadiusParams(CompensateRadiusParams value) {Opt = value;}
        public static implicit operator _InOptMut_CompensateRadiusParams(CompensateRadiusParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `CompensateRadiusParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CompensateRadiusParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CompensateRadiusParams`/`Const_CompensateRadiusParams` to pass it to the function.
    public class _InOptConst_CompensateRadiusParams
    {
        public Const_CompensateRadiusParams? Opt;

        public _InOptConst_CompensateRadiusParams() {}
        public _InOptConst_CompensateRadiusParams(Const_CompensateRadiusParams value) {Opt = value;}
        public static implicit operator _InOptConst_CompensateRadiusParams(Const_CompensateRadiusParams value) {return new(value);}
    }

    /// compensate spherical milling tool radius in given mesh region making it possible to mill it
    /// note that tool milling outer surface of the mesh
    /// also please note that boundaries of the region are fixed
    /// Generated from function `MR::compensateRadius`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CompensateRadius(MR.Mesh mesh, MR.Const_CompensateRadiusParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compensateRadius", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_compensateRadius(MR.Mesh._Underlying *mesh, MR.Const_CompensateRadiusParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_compensateRadius(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
