public static partial class MR
{
    public enum DepthFunction : int
    {
        Never = 0,
        Less = 1,
        Equal = 2,
        Greater = 4,
        LessOrEqual = 3,
        GreaterOrEqual = 6,
        NotEqual = 5,
        Always = 7,
        // usually "Less" but may differ for different object types
        Default = 8,
    }

    /// Common rendering parameters for meshes and UI.
    /// Generated from class `MR::BaseRenderParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ModelBaseRenderParams`
    ///     `MR::UiRenderParams`
    ///   Indirect: (non-virtual)
    ///     `MR::ModelRenderParams`
    /// This is the const half of the class.
    public class Const_BaseRenderParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BaseRenderParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_Destroy", ExactSpelling = true)]
            extern static void __MR_BaseRenderParams_Destroy(_Underlying *_this);
            __MR_BaseRenderParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BaseRenderParams() {Dispose(false);}

        public unsafe MR.Const_Matrix4f ViewMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_Get_viewMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_BaseRenderParams_Get_viewMatrix(_Underlying *_this);
                return new(__MR_BaseRenderParams_Get_viewMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Matrix4f ProjMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_Get_projMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_BaseRenderParams_Get_projMatrix(_Underlying *_this);
                return new(__MR_BaseRenderParams_Get_projMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        // id of the viewport
        public unsafe MR.Const_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_Get_viewportId", ExactSpelling = true)]
                extern static MR.Const_ViewportId._Underlying *__MR_BaseRenderParams_Get_viewportId(_Underlying *_this);
                return new(__MR_BaseRenderParams_Get_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public unsafe MR.Const_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_Get_viewport", ExactSpelling = true)]
                extern static MR.Const_Vector4i._Underlying *__MR_BaseRenderParams_Get_viewport(_Underlying *_this);
                return new(__MR_BaseRenderParams_Get_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::BaseRenderParams::BaseRenderParams`.
        public unsafe Const_BaseRenderParams(MR.Const_BaseRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_BaseRenderParams_ConstructFromAnother(MR.BaseRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_BaseRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Constructs `MR::BaseRenderParams` elementwise.
        public unsafe Const_BaseRenderParams(MR.Const_Matrix4f viewMatrix, MR.Const_Matrix4f projMatrix, MR.ViewportId viewportId, MR.Vector4i viewport) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_BaseRenderParams_ConstructFrom(MR.Const_Matrix4f._Underlying *viewMatrix, MR.Const_Matrix4f._Underlying *projMatrix, MR.ViewportId viewportId, MR.Vector4i viewport);
            _UnderlyingPtr = __MR_BaseRenderParams_ConstructFrom(viewMatrix._UnderlyingPtr, projMatrix._UnderlyingPtr, viewportId, viewport);
        }
    }

    /// Common rendering parameters for meshes and UI.
    /// Generated from class `MR::BaseRenderParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ModelBaseRenderParams`
    ///     `MR::UiRenderParams`
    ///   Indirect: (non-virtual)
    ///     `MR::ModelRenderParams`
    /// This is the non-const half of the class.
    public class BaseRenderParams : Const_BaseRenderParams
    {
        internal unsafe BaseRenderParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // id of the viewport
        public new unsafe MR.Mut_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_GetMutable_viewportId", ExactSpelling = true)]
                extern static MR.Mut_ViewportId._Underlying *__MR_BaseRenderParams_GetMutable_viewportId(_Underlying *_this);
                return new(__MR_BaseRenderParams_GetMutable_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public new unsafe MR.Mut_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_GetMutable_viewport", ExactSpelling = true)]
                extern static MR.Mut_Vector4i._Underlying *__MR_BaseRenderParams_GetMutable_viewport(_Underlying *_this);
                return new(__MR_BaseRenderParams_GetMutable_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::BaseRenderParams::BaseRenderParams`.
        public unsafe BaseRenderParams(MR.Const_BaseRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_BaseRenderParams_ConstructFromAnother(MR.BaseRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_BaseRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Constructs `MR::BaseRenderParams` elementwise.
        public unsafe BaseRenderParams(MR.Const_Matrix4f viewMatrix, MR.Const_Matrix4f projMatrix, MR.ViewportId viewportId, MR.Vector4i viewport) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseRenderParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_BaseRenderParams_ConstructFrom(MR.Const_Matrix4f._Underlying *viewMatrix, MR.Const_Matrix4f._Underlying *projMatrix, MR.ViewportId viewportId, MR.Vector4i viewport);
            _UnderlyingPtr = __MR_BaseRenderParams_ConstructFrom(viewMatrix._UnderlyingPtr, projMatrix._UnderlyingPtr, viewportId, viewport);
        }
    }

    /// This is used for optional parameters of class `BaseRenderParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BaseRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BaseRenderParams`/`Const_BaseRenderParams` directly.
    public class _InOptMut_BaseRenderParams
    {
        public BaseRenderParams? Opt;

        public _InOptMut_BaseRenderParams() {}
        public _InOptMut_BaseRenderParams(BaseRenderParams value) {Opt = value;}
        public static implicit operator _InOptMut_BaseRenderParams(BaseRenderParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `BaseRenderParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BaseRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BaseRenderParams`/`Const_BaseRenderParams` to pass it to the function.
    public class _InOptConst_BaseRenderParams
    {
        public Const_BaseRenderParams? Opt;

        public _InOptConst_BaseRenderParams() {}
        public _InOptConst_BaseRenderParams(Const_BaseRenderParams value) {Opt = value;}
        public static implicit operator _InOptConst_BaseRenderParams(Const_BaseRenderParams value) {return new(value);}
    }

    /// Common rendering parameters for meshes (both for primary rendering and for the picker;
    /// the picker uses this as is while the primary rendering adds more fields).
    /// Generated from class `MR::ModelBaseRenderParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseRenderParams`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ModelRenderParams`
    /// This is the const half of the class.
    public class Const_ModelBaseRenderParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ModelBaseRenderParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ModelBaseRenderParams_Destroy(_Underlying *_this);
            __MR_ModelBaseRenderParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ModelBaseRenderParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseRenderParams(Const_ModelBaseRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_UpcastTo_MR_BaseRenderParams", ExactSpelling = true)]
            extern static MR.Const_BaseRenderParams._Underlying *__MR_ModelBaseRenderParams_UpcastTo_MR_BaseRenderParams(_Underlying *_this);
            MR.Const_BaseRenderParams ret = new(__MR_ModelBaseRenderParams_UpcastTo_MR_BaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.Const_Matrix4f ModelMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_modelMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_ModelBaseRenderParams_Get_modelMatrix(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_Get_modelMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport clip plane (it is not applied while object does not have clipping flag set)
        public unsafe MR.Const_Plane3f ClipPlane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_clipPlane", ExactSpelling = true)]
                extern static MR.Const_Plane3f._Underlying *__MR_ModelBaseRenderParams_Get_clipPlane(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_Get_clipPlane(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.DepthFunction DepthFunction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_depthFunction", ExactSpelling = true)]
                extern static MR.DepthFunction *__MR_ModelBaseRenderParams_Get_depthFunction(_Underlying *_this);
                return *__MR_ModelBaseRenderParams_Get_depthFunction(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Matrix4f ViewMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_viewMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_ModelBaseRenderParams_Get_viewMatrix(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_Get_viewMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Matrix4f ProjMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_projMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_ModelBaseRenderParams_Get_projMatrix(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_Get_projMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        // id of the viewport
        public unsafe MR.Const_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_viewportId", ExactSpelling = true)]
                extern static MR.Const_ViewportId._Underlying *__MR_ModelBaseRenderParams_Get_viewportId(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_Get_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public unsafe MR.Const_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_Get_viewport", ExactSpelling = true)]
                extern static MR.Const_Vector4i._Underlying *__MR_ModelBaseRenderParams_Get_viewport(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_Get_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::ModelBaseRenderParams::ModelBaseRenderParams`.
        public unsafe Const_ModelBaseRenderParams(MR.Const_ModelBaseRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ModelBaseRenderParams._Underlying *__MR_ModelBaseRenderParams_ConstructFromAnother(MR.ModelBaseRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_ModelBaseRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Common rendering parameters for meshes (both for primary rendering and for the picker;
    /// the picker uses this as is while the primary rendering adds more fields).
    /// Generated from class `MR::ModelBaseRenderParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseRenderParams`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ModelRenderParams`
    /// This is the non-const half of the class.
    public class ModelBaseRenderParams : Const_ModelBaseRenderParams
    {
        internal unsafe ModelBaseRenderParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseRenderParams(ModelBaseRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_UpcastTo_MR_BaseRenderParams", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_ModelBaseRenderParams_UpcastTo_MR_BaseRenderParams(_Underlying *_this);
            MR.BaseRenderParams ret = new(__MR_ModelBaseRenderParams_UpcastTo_MR_BaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref MR.DepthFunction DepthFunction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_GetMutable_depthFunction", ExactSpelling = true)]
                extern static MR.DepthFunction *__MR_ModelBaseRenderParams_GetMutable_depthFunction(_Underlying *_this);
                return ref *__MR_ModelBaseRenderParams_GetMutable_depthFunction(_UnderlyingPtr);
            }
        }

        // id of the viewport
        public new unsafe MR.Mut_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_GetMutable_viewportId", ExactSpelling = true)]
                extern static MR.Mut_ViewportId._Underlying *__MR_ModelBaseRenderParams_GetMutable_viewportId(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_GetMutable_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public new unsafe MR.Mut_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_GetMutable_viewport", ExactSpelling = true)]
                extern static MR.Mut_Vector4i._Underlying *__MR_ModelBaseRenderParams_GetMutable_viewport(_Underlying *_this);
                return new(__MR_ModelBaseRenderParams_GetMutable_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::ModelBaseRenderParams::ModelBaseRenderParams`.
        public unsafe ModelBaseRenderParams(MR.Const_ModelBaseRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelBaseRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ModelBaseRenderParams._Underlying *__MR_ModelBaseRenderParams_ConstructFromAnother(MR.ModelBaseRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_ModelBaseRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `ModelBaseRenderParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ModelBaseRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ModelBaseRenderParams`/`Const_ModelBaseRenderParams` directly.
    public class _InOptMut_ModelBaseRenderParams
    {
        public ModelBaseRenderParams? Opt;

        public _InOptMut_ModelBaseRenderParams() {}
        public _InOptMut_ModelBaseRenderParams(ModelBaseRenderParams value) {Opt = value;}
        public static implicit operator _InOptMut_ModelBaseRenderParams(ModelBaseRenderParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ModelBaseRenderParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ModelBaseRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ModelBaseRenderParams`/`Const_ModelBaseRenderParams` to pass it to the function.
    public class _InOptConst_ModelBaseRenderParams
    {
        public Const_ModelBaseRenderParams? Opt;

        public _InOptConst_ModelBaseRenderParams() {}
        public _InOptConst_ModelBaseRenderParams(Const_ModelBaseRenderParams value) {Opt = value;}
        public static implicit operator _InOptConst_ModelBaseRenderParams(Const_ModelBaseRenderParams value) {return new(value);}
    }

    /// Mesh rendering parameters for primary rendering (as opposed to the picker).
    /// Generated from class `MR::ModelRenderParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ModelBaseRenderParams`
    ///   Indirect: (non-virtual)
    ///     `MR::BaseRenderParams`
    /// This is the const half of the class.
    public class Const_ModelRenderParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ModelRenderParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ModelRenderParams_Destroy(_Underlying *_this);
            __MR_ModelRenderParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ModelRenderParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseRenderParams(Const_ModelRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_UpcastTo_MR_BaseRenderParams", ExactSpelling = true)]
            extern static MR.Const_BaseRenderParams._Underlying *__MR_ModelRenderParams_UpcastTo_MR_BaseRenderParams(_Underlying *_this);
            MR.Const_BaseRenderParams ret = new(__MR_ModelRenderParams_UpcastTo_MR_BaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_ModelBaseRenderParams(Const_ModelRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_UpcastTo_MR_ModelBaseRenderParams", ExactSpelling = true)]
            extern static MR.Const_ModelBaseRenderParams._Underlying *__MR_ModelRenderParams_UpcastTo_MR_ModelBaseRenderParams(_Underlying *_this);
            MR.Const_ModelBaseRenderParams ret = new(__MR_ModelRenderParams_UpcastTo_MR_ModelBaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // normal matrix, only necessary for triangles rendering
        public unsafe ref readonly MR.Matrix4f * NormMatrixPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_normMatrixPtr", ExactSpelling = true)]
                extern static MR.Matrix4f **__MR_ModelRenderParams_Get_normMatrixPtr(_Underlying *_this);
                return ref *__MR_ModelRenderParams_Get_normMatrixPtr(_UnderlyingPtr);
            }
        }

        // position of light source
        public unsafe MR.Const_Vector3f LightPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_lightPos", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_ModelRenderParams_Get_lightPos(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_lightPos(_UnderlyingPtr), is_owning: false);
            }
        }

        // determines how to handle transparent models
        public unsafe MR.Const_TransparencyMode TransparencyMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_transparencyMode", ExactSpelling = true)]
                extern static MR.Const_TransparencyMode._Underlying *__MR_ModelRenderParams_Get_transparencyMode(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_transparencyMode(_UnderlyingPtr), is_owning: false);
            }
        }

        // Only perform rendering if `bool( passMask & desiredPass )` is true.
        public unsafe MR.RenderModelPassMask PassMask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_passMask", ExactSpelling = true)]
                extern static MR.RenderModelPassMask *__MR_ModelRenderParams_Get_passMask(_Underlying *_this);
                return *__MR_ModelRenderParams_Get_passMask(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Matrix4f ModelMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_modelMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_ModelRenderParams_Get_modelMatrix(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_modelMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport clip plane (it is not applied while object does not have clipping flag set)
        public unsafe MR.Const_Plane3f ClipPlane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_clipPlane", ExactSpelling = true)]
                extern static MR.Const_Plane3f._Underlying *__MR_ModelRenderParams_Get_clipPlane(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_clipPlane(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.DepthFunction DepthFunction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_depthFunction", ExactSpelling = true)]
                extern static MR.DepthFunction *__MR_ModelRenderParams_Get_depthFunction(_Underlying *_this);
                return *__MR_ModelRenderParams_Get_depthFunction(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Matrix4f ViewMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_viewMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_ModelRenderParams_Get_viewMatrix(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_viewMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Matrix4f ProjMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_projMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_ModelRenderParams_Get_projMatrix(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_projMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        // id of the viewport
        public unsafe MR.Const_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_viewportId", ExactSpelling = true)]
                extern static MR.Const_ViewportId._Underlying *__MR_ModelRenderParams_Get_viewportId(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public unsafe MR.Const_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_Get_viewport", ExactSpelling = true)]
                extern static MR.Const_Vector4i._Underlying *__MR_ModelRenderParams_Get_viewport(_Underlying *_this);
                return new(__MR_ModelRenderParams_Get_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::ModelRenderParams::ModelRenderParams`.
        public unsafe Const_ModelRenderParams(MR.Const_ModelRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ModelRenderParams._Underlying *__MR_ModelRenderParams_ConstructFromAnother(MR.ModelRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_ModelRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Mesh rendering parameters for primary rendering (as opposed to the picker).
    /// Generated from class `MR::ModelRenderParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ModelBaseRenderParams`
    ///   Indirect: (non-virtual)
    ///     `MR::BaseRenderParams`
    /// This is the non-const half of the class.
    public class ModelRenderParams : Const_ModelRenderParams
    {
        internal unsafe ModelRenderParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseRenderParams(ModelRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_UpcastTo_MR_BaseRenderParams", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_ModelRenderParams_UpcastTo_MR_BaseRenderParams(_Underlying *_this);
            MR.BaseRenderParams ret = new(__MR_ModelRenderParams_UpcastTo_MR_BaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.ModelBaseRenderParams(ModelRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_UpcastTo_MR_ModelBaseRenderParams", ExactSpelling = true)]
            extern static MR.ModelBaseRenderParams._Underlying *__MR_ModelRenderParams_UpcastTo_MR_ModelBaseRenderParams(_Underlying *_this);
            MR.ModelBaseRenderParams ret = new(__MR_ModelRenderParams_UpcastTo_MR_ModelBaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // normal matrix, only necessary for triangles rendering
        public new unsafe ref readonly MR.Matrix4f * NormMatrixPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_normMatrixPtr", ExactSpelling = true)]
                extern static MR.Matrix4f **__MR_ModelRenderParams_GetMutable_normMatrixPtr(_Underlying *_this);
                return ref *__MR_ModelRenderParams_GetMutable_normMatrixPtr(_UnderlyingPtr);
            }
        }

        // position of light source
        public new unsafe MR.Mut_Vector3f LightPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_lightPos", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_ModelRenderParams_GetMutable_lightPos(_Underlying *_this);
                return new(__MR_ModelRenderParams_GetMutable_lightPos(_UnderlyingPtr), is_owning: false);
            }
        }

        // determines how to handle transparent models
        public new unsafe MR.TransparencyMode TransparencyMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_transparencyMode", ExactSpelling = true)]
                extern static MR.TransparencyMode._Underlying *__MR_ModelRenderParams_GetMutable_transparencyMode(_Underlying *_this);
                return new(__MR_ModelRenderParams_GetMutable_transparencyMode(_UnderlyingPtr), is_owning: false);
            }
        }

        // Only perform rendering if `bool( passMask & desiredPass )` is true.
        public new unsafe ref MR.RenderModelPassMask PassMask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_passMask", ExactSpelling = true)]
                extern static MR.RenderModelPassMask *__MR_ModelRenderParams_GetMutable_passMask(_Underlying *_this);
                return ref *__MR_ModelRenderParams_GetMutable_passMask(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.DepthFunction DepthFunction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_depthFunction", ExactSpelling = true)]
                extern static MR.DepthFunction *__MR_ModelRenderParams_GetMutable_depthFunction(_Underlying *_this);
                return ref *__MR_ModelRenderParams_GetMutable_depthFunction(_UnderlyingPtr);
            }
        }

        // id of the viewport
        public new unsafe MR.Mut_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_viewportId", ExactSpelling = true)]
                extern static MR.Mut_ViewportId._Underlying *__MR_ModelRenderParams_GetMutable_viewportId(_Underlying *_this);
                return new(__MR_ModelRenderParams_GetMutable_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public new unsafe MR.Mut_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_GetMutable_viewport", ExactSpelling = true)]
                extern static MR.Mut_Vector4i._Underlying *__MR_ModelRenderParams_GetMutable_viewport(_Underlying *_this);
                return new(__MR_ModelRenderParams_GetMutable_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::ModelRenderParams::ModelRenderParams`.
        public unsafe ModelRenderParams(MR.Const_ModelRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ModelRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ModelRenderParams._Underlying *__MR_ModelRenderParams_ConstructFromAnother(MR.ModelRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_ModelRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `ModelRenderParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ModelRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ModelRenderParams`/`Const_ModelRenderParams` directly.
    public class _InOptMut_ModelRenderParams
    {
        public ModelRenderParams? Opt;

        public _InOptMut_ModelRenderParams() {}
        public _InOptMut_ModelRenderParams(ModelRenderParams value) {Opt = value;}
        public static implicit operator _InOptMut_ModelRenderParams(ModelRenderParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ModelRenderParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ModelRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ModelRenderParams`/`Const_ModelRenderParams` to pass it to the function.
    public class _InOptConst_ModelRenderParams
    {
        public Const_ModelRenderParams? Opt;

        public _InOptConst_ModelRenderParams() {}
        public _InOptConst_ModelRenderParams(Const_ModelRenderParams value) {Opt = value;}
        public static implicit operator _InOptConst_ModelRenderParams(Const_ModelRenderParams value) {return new(value);}
    }

    /// `IRenderObject::renderUi()` can emit zero or more or more of those tasks. They are sorted by depth every frame.
    /// Generated from class `MR::BasicUiRenderTask`.
    /// This is the const half of the class.
    public class Const_BasicUiRenderTask : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_BasicUiRenderTask_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_BasicUiRenderTask_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_BasicUiRenderTask_UseCount();
                return __MR_std_shared_ptr_MR_BasicUiRenderTask_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_BasicUiRenderTask(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_BasicUiRenderTask_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_BasicUiRenderTask_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructNonOwning(ptr);
        }

        internal unsafe Const_BasicUiRenderTask(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe BasicUiRenderTask _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_BasicUiRenderTask_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_BasicUiRenderTask_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_BasicUiRenderTask_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_BasicUiRenderTask_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_BasicUiRenderTask_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_BasicUiRenderTask_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BasicUiRenderTask() {Dispose(false);}

        /// The tasks are sorted by this depth, descending (larger depth = further away).
        public unsafe float RenderTaskDepth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_Get_renderTaskDepth", ExactSpelling = true)]
                extern static float *__MR_BasicUiRenderTask_Get_renderTaskDepth(_Underlying *_this);
                return *__MR_BasicUiRenderTask_Get_renderTaskDepth(_UnderlyingPtr);
            }
        }

        /// Generated from class `MR::BasicUiRenderTask::BackwardPassParams`.
        /// This is the const half of the class.
        public class Const_BackwardPassParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BackwardPassParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_Destroy", ExactSpelling = true)]
                extern static void __MR_BasicUiRenderTask_BackwardPassParams_Destroy(_Underlying *_this);
                __MR_BasicUiRenderTask_BackwardPassParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BackwardPassParams() {Dispose(false);}

            // Which interactions should be blocked by this object.
            // This is passed along between all `renderUi()` calls in a single frame, and then the end result is used.
            public unsafe MR.BasicUiRenderTask.InteractionMask ConsumedInteractions
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_Get_consumedInteractions", ExactSpelling = true)]
                    extern static MR.BasicUiRenderTask.InteractionMask *__MR_BasicUiRenderTask_BackwardPassParams_Get_consumedInteractions(_Underlying *_this);
                    return *__MR_BasicUiRenderTask_BackwardPassParams_Get_consumedInteractions(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BackwardPassParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_DefaultConstruct();
                _UnderlyingPtr = __MR_BasicUiRenderTask_BackwardPassParams_DefaultConstruct();
            }

            /// Constructs `MR::BasicUiRenderTask::BackwardPassParams` elementwise.
            public unsafe Const_BackwardPassParams(MR.BasicUiRenderTask.InteractionMask consumedInteractions) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_ConstructFrom(MR.BasicUiRenderTask.InteractionMask consumedInteractions);
                _UnderlyingPtr = __MR_BasicUiRenderTask_BackwardPassParams_ConstructFrom(consumedInteractions);
            }

            /// Generated from constructor `MR::BasicUiRenderTask::BackwardPassParams::BackwardPassParams`.
            public unsafe Const_BackwardPassParams(MR.BasicUiRenderTask.Const_BackwardPassParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_ConstructFromAnother(MR.BasicUiRenderTask.BackwardPassParams._Underlying *_other);
                _UnderlyingPtr = __MR_BasicUiRenderTask_BackwardPassParams_ConstructFromAnother(_other._UnderlyingPtr);
            }

            // If nothing else is hovered, this returns true and writes `mouseHover` to `consumedInteractions`.
            /// Generated from method `MR::BasicUiRenderTask::BackwardPassParams::tryConsumeMouseHover`.
            public unsafe bool TryConsumeMouseHover()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_tryConsumeMouseHover", ExactSpelling = true)]
                extern static byte __MR_BasicUiRenderTask_BackwardPassParams_tryConsumeMouseHover(_Underlying *_this);
                return __MR_BasicUiRenderTask_BackwardPassParams_tryConsumeMouseHover(_UnderlyingPtr) != 0;
            }
        }

        /// Generated from class `MR::BasicUiRenderTask::BackwardPassParams`.
        /// This is the non-const half of the class.
        public class BackwardPassParams : Const_BackwardPassParams
        {
            internal unsafe BackwardPassParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Which interactions should be blocked by this object.
            // This is passed along between all `renderUi()` calls in a single frame, and then the end result is used.
            public new unsafe ref MR.BasicUiRenderTask.InteractionMask ConsumedInteractions
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_GetMutable_consumedInteractions", ExactSpelling = true)]
                    extern static MR.BasicUiRenderTask.InteractionMask *__MR_BasicUiRenderTask_BackwardPassParams_GetMutable_consumedInteractions(_Underlying *_this);
                    return ref *__MR_BasicUiRenderTask_BackwardPassParams_GetMutable_consumedInteractions(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BackwardPassParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_DefaultConstruct();
                _UnderlyingPtr = __MR_BasicUiRenderTask_BackwardPassParams_DefaultConstruct();
            }

            /// Constructs `MR::BasicUiRenderTask::BackwardPassParams` elementwise.
            public unsafe BackwardPassParams(MR.BasicUiRenderTask.InteractionMask consumedInteractions) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_ConstructFrom(MR.BasicUiRenderTask.InteractionMask consumedInteractions);
                _UnderlyingPtr = __MR_BasicUiRenderTask_BackwardPassParams_ConstructFrom(consumedInteractions);
            }

            /// Generated from constructor `MR::BasicUiRenderTask::BackwardPassParams::BackwardPassParams`.
            public unsafe BackwardPassParams(MR.BasicUiRenderTask.Const_BackwardPassParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_ConstructFromAnother(MR.BasicUiRenderTask.BackwardPassParams._Underlying *_other);
                _UnderlyingPtr = __MR_BasicUiRenderTask_BackwardPassParams_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::BasicUiRenderTask::BackwardPassParams::operator=`.
            public unsafe MR.BasicUiRenderTask.BackwardPassParams Assign(MR.BasicUiRenderTask.Const_BackwardPassParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_BackwardPassParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_BasicUiRenderTask_BackwardPassParams_AssignFromAnother(_Underlying *_this, MR.BasicUiRenderTask.BackwardPassParams._Underlying *_other);
                return new(__MR_BasicUiRenderTask_BackwardPassParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `BackwardPassParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BackwardPassParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BackwardPassParams`/`Const_BackwardPassParams` directly.
        public class _InOptMut_BackwardPassParams
        {
            public BackwardPassParams? Opt;

            public _InOptMut_BackwardPassParams() {}
            public _InOptMut_BackwardPassParams(BackwardPassParams value) {Opt = value;}
            public static implicit operator _InOptMut_BackwardPassParams(BackwardPassParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `BackwardPassParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BackwardPassParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BackwardPassParams`/`Const_BackwardPassParams` to pass it to the function.
        public class _InOptConst_BackwardPassParams
        {
            public Const_BackwardPassParams? Opt;

            public _InOptConst_BackwardPassParams() {}
            public _InOptConst_BackwardPassParams(Const_BackwardPassParams value) {Opt = value;}
            public static implicit operator _InOptConst_BackwardPassParams(Const_BackwardPassParams value) {return new(value);}
        }

        public enum InteractionMask : int
        {
            MouseHover = 1,
            MouseScroll = 2,
        }
    }

    /// `IRenderObject::renderUi()` can emit zero or more or more of those tasks. They are sorted by depth every frame.
    /// Generated from class `MR::BasicUiRenderTask`.
    /// This is the non-const half of the class.
    public class BasicUiRenderTask : Const_BasicUiRenderTask
    {
        internal unsafe BasicUiRenderTask(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe BasicUiRenderTask(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// The tasks are sorted by this depth, descending (larger depth = further away).
        public new unsafe ref float RenderTaskDepth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_GetMutable_renderTaskDepth", ExactSpelling = true)]
                extern static float *__MR_BasicUiRenderTask_GetMutable_renderTaskDepth(_Underlying *_this);
                return ref *__MR_BasicUiRenderTask_GetMutable_renderTaskDepth(_UnderlyingPtr);
            }
        }

        /// This is an optional early pass, where you can claim exclusive control over the mouse.
        /// This pass is executed in reverse draw order.
        /// Generated from method `MR::BasicUiRenderTask::earlyBackwardPass`.
        public unsafe void EarlyBackwardPass(MR.BasicUiRenderTask.Const_BackwardPassParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_earlyBackwardPass", ExactSpelling = true)]
            extern static void __MR_BasicUiRenderTask_earlyBackwardPass(_Underlying *_this, MR.BasicUiRenderTask.Const_BackwardPassParams._Underlying *params_);
            __MR_BasicUiRenderTask_earlyBackwardPass(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// This is the main rendering pass.
        /// Generated from method `MR::BasicUiRenderTask::renderPass`.
        public unsafe void RenderPass()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasicUiRenderTask_renderPass", ExactSpelling = true)]
            extern static void __MR_BasicUiRenderTask_renderPass(_Underlying *_this);
            __MR_BasicUiRenderTask_renderPass(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BasicUiRenderTask` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BasicUiRenderTask`/`Const_BasicUiRenderTask` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BasicUiRenderTask
    {
        internal readonly Const_BasicUiRenderTask? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `BasicUiRenderTask` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BasicUiRenderTask`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BasicUiRenderTask`/`Const_BasicUiRenderTask` directly.
    public class _InOptMut_BasicUiRenderTask
    {
        public BasicUiRenderTask? Opt;

        public _InOptMut_BasicUiRenderTask() {}
        public _InOptMut_BasicUiRenderTask(BasicUiRenderTask value) {Opt = value;}
        public static implicit operator _InOptMut_BasicUiRenderTask(BasicUiRenderTask value) {return new(value);}
    }

    /// This is used for optional parameters of class `BasicUiRenderTask` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BasicUiRenderTask`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BasicUiRenderTask`/`Const_BasicUiRenderTask` to pass it to the function.
    public class _InOptConst_BasicUiRenderTask
    {
        public Const_BasicUiRenderTask? Opt;

        public _InOptConst_BasicUiRenderTask() {}
        public _InOptConst_BasicUiRenderTask(Const_BasicUiRenderTask value) {Opt = value;}
        public static implicit operator _InOptConst_BasicUiRenderTask(Const_BasicUiRenderTask value) {return new(value);}
    }

    /// Generated from class `MR::UiRenderParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseRenderParams`
    /// This is the const half of the class.
    public class Const_UiRenderParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UiRenderParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_Destroy", ExactSpelling = true)]
            extern static void __MR_UiRenderParams_Destroy(_Underlying *_this);
            __MR_UiRenderParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UiRenderParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseRenderParams(Const_UiRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_UpcastTo_MR_BaseRenderParams", ExactSpelling = true)]
            extern static MR.Const_BaseRenderParams._Underlying *__MR_UiRenderParams_UpcastTo_MR_BaseRenderParams(_Underlying *_this);
            MR.Const_BaseRenderParams ret = new(__MR_UiRenderParams_UpcastTo_MR_BaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Those are Z-sorted and then executed.
        public unsafe ref void * Tasks
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_Get_tasks", ExactSpelling = true)]
                extern static void **__MR_UiRenderParams_Get_tasks(_Underlying *_this);
                return ref *__MR_UiRenderParams_Get_tasks(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Matrix4f ViewMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_Get_viewMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_UiRenderParams_Get_viewMatrix(_Underlying *_this);
                return new(__MR_UiRenderParams_Get_viewMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Matrix4f ProjMatrix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_Get_projMatrix", ExactSpelling = true)]
                extern static MR.Const_Matrix4f._Underlying *__MR_UiRenderParams_Get_projMatrix(_Underlying *_this);
                return new(__MR_UiRenderParams_Get_projMatrix(_UnderlyingPtr), is_owning: false);
            }
        }

        // id of the viewport
        public unsafe MR.Const_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_Get_viewportId", ExactSpelling = true)]
                extern static MR.Const_ViewportId._Underlying *__MR_UiRenderParams_Get_viewportId(_Underlying *_this);
                return new(__MR_UiRenderParams_Get_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public unsafe MR.Const_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_Get_viewport", ExactSpelling = true)]
                extern static MR.Const_Vector4i._Underlying *__MR_UiRenderParams_Get_viewport(_Underlying *_this);
                return new(__MR_UiRenderParams_Get_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::UiRenderParams::UiRenderParams`.
        public unsafe Const_UiRenderParams(MR.Const_UiRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UiRenderParams._Underlying *__MR_UiRenderParams_ConstructFromAnother(MR.UiRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_UiRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::UiRenderParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseRenderParams`
    /// This is the non-const half of the class.
    public class UiRenderParams : Const_UiRenderParams
    {
        internal unsafe UiRenderParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseRenderParams(UiRenderParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_UpcastTo_MR_BaseRenderParams", ExactSpelling = true)]
            extern static MR.BaseRenderParams._Underlying *__MR_UiRenderParams_UpcastTo_MR_BaseRenderParams(_Underlying *_this);
            MR.BaseRenderParams ret = new(__MR_UiRenderParams_UpcastTo_MR_BaseRenderParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Those are Z-sorted and then executed.
        public new unsafe ref void * Tasks
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_GetMutable_tasks", ExactSpelling = true)]
                extern static void **__MR_UiRenderParams_GetMutable_tasks(_Underlying *_this);
                return ref *__MR_UiRenderParams_GetMutable_tasks(_UnderlyingPtr);
            }
        }

        // id of the viewport
        public new unsafe MR.Mut_ViewportId ViewportId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_GetMutable_viewportId", ExactSpelling = true)]
                extern static MR.Mut_ViewportId._Underlying *__MR_UiRenderParams_GetMutable_viewportId(_Underlying *_this);
                return new(__MR_UiRenderParams_GetMutable_viewportId(_UnderlyingPtr), is_owning: false);
            }
        }

        // viewport x0, y0, width, height
        public new unsafe MR.Mut_Vector4i Viewport
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_GetMutable_viewport", ExactSpelling = true)]
                extern static MR.Mut_Vector4i._Underlying *__MR_UiRenderParams_GetMutable_viewport(_Underlying *_this);
                return new(__MR_UiRenderParams_GetMutable_viewport(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::UiRenderParams::UiRenderParams`.
        public unsafe UiRenderParams(MR.Const_UiRenderParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UiRenderParams._Underlying *__MR_UiRenderParams_ConstructFromAnother(MR.UiRenderParams._Underlying *_other);
            _UnderlyingPtr = __MR_UiRenderParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `UiRenderParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UiRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UiRenderParams`/`Const_UiRenderParams` directly.
    public class _InOptMut_UiRenderParams
    {
        public UiRenderParams? Opt;

        public _InOptMut_UiRenderParams() {}
        public _InOptMut_UiRenderParams(UiRenderParams value) {Opt = value;}
        public static implicit operator _InOptMut_UiRenderParams(UiRenderParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `UiRenderParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UiRenderParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UiRenderParams`/`Const_UiRenderParams` to pass it to the function.
    public class _InOptConst_UiRenderParams
    {
        public Const_UiRenderParams? Opt;

        public _InOptConst_UiRenderParams() {}
        public _InOptConst_UiRenderParams(Const_UiRenderParams value) {Opt = value;}
        public static implicit operator _InOptConst_UiRenderParams(Const_UiRenderParams value) {return new(value);}
    }

    /// Generated from class `MR::UiRenderManager`.
    /// This is the const half of the class.
    public class Const_UiRenderManager : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UiRenderManager(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_Destroy", ExactSpelling = true)]
            extern static void __MR_UiRenderManager_Destroy(_Underlying *_this);
            __MR_UiRenderManager_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UiRenderManager() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UiRenderManager() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UiRenderManager._Underlying *__MR_UiRenderManager_DefaultConstruct();
            _UnderlyingPtr = __MR_UiRenderManager_DefaultConstruct();
        }

        /// Generated from constructor `MR::UiRenderManager::UiRenderManager`.
        public unsafe Const_UiRenderManager(MR._ByValue_UiRenderManager _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UiRenderManager._Underlying *__MR_UiRenderManager_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UiRenderManager._Underlying *_other);
            _UnderlyingPtr = __MR_UiRenderManager_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::UiRenderManager`.
    /// This is the non-const half of the class.
    public class UiRenderManager : Const_UiRenderManager
    {
        internal unsafe UiRenderManager(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UiRenderManager() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UiRenderManager._Underlying *__MR_UiRenderManager_DefaultConstruct();
            _UnderlyingPtr = __MR_UiRenderManager_DefaultConstruct();
        }

        /// Generated from constructor `MR::UiRenderManager::UiRenderManager`.
        public unsafe UiRenderManager(MR._ByValue_UiRenderManager _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UiRenderManager._Underlying *__MR_UiRenderManager_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UiRenderManager._Underlying *_other);
            _UnderlyingPtr = __MR_UiRenderManager_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        // This is called before doing `IRenderObject::renderUi()` on even object in a viewport. Each viewport is rendered separately.
        /// Generated from method `MR::UiRenderManager::preRenderViewport`.
        public unsafe void PreRenderViewport(MR.ViewportId viewport)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_preRenderViewport", ExactSpelling = true)]
            extern static void __MR_UiRenderManager_preRenderViewport(_Underlying *_this, MR.ViewportId viewport);
            __MR_UiRenderManager_preRenderViewport(_UnderlyingPtr, viewport);
        }

        // This is called after doing `IRenderObject::renderUi()` on even object in a viewport. Each viewport is rendered separately.
        /// Generated from method `MR::UiRenderManager::postRenderViewport`.
        public unsafe void PostRenderViewport(MR.ViewportId viewport)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_postRenderViewport", ExactSpelling = true)]
            extern static void __MR_UiRenderManager_postRenderViewport(_Underlying *_this, MR.ViewportId viewport);
            __MR_UiRenderManager_postRenderViewport(_UnderlyingPtr, viewport);
        }

        // Returns the parameters for the `IRenderObject::earlyBackwardPass()`.
        // This will be called exactly once per viewport, each time the UI in it is rendered.
        /// Generated from method `MR::UiRenderManager::beginBackwardPass`.
        public unsafe MR.BasicUiRenderTask.BackwardPassParams BeginBackwardPass(MR.ViewportId viewport, MR.Std.Vector_StdSharedPtrMRBasicUiRenderTask tasks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_beginBackwardPass", ExactSpelling = true)]
            extern static MR.BasicUiRenderTask.BackwardPassParams._Underlying *__MR_UiRenderManager_beginBackwardPass(_Underlying *_this, MR.ViewportId viewport, MR.Std.Vector_StdSharedPtrMRBasicUiRenderTask._Underlying *tasks);
            return new(__MR_UiRenderManager_beginBackwardPass(_UnderlyingPtr, viewport, tasks._UnderlyingPtr), is_owning: true);
        }

        // After the backward pass is performed, the parameters should be passed back into this function.
        /// Generated from method `MR::UiRenderManager::finishBackwardPass`.
        public unsafe void FinishBackwardPass(MR.ViewportId viewport, MR.BasicUiRenderTask.Const_BackwardPassParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UiRenderManager_finishBackwardPass", ExactSpelling = true)]
            extern static void __MR_UiRenderManager_finishBackwardPass(_Underlying *_this, MR.ViewportId viewport, MR.BasicUiRenderTask.Const_BackwardPassParams._Underlying *params_);
            __MR_UiRenderManager_finishBackwardPass(_UnderlyingPtr, viewport, params_._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UiRenderManager` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UiRenderManager`/`Const_UiRenderManager` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UiRenderManager
    {
        internal readonly Const_UiRenderManager? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UiRenderManager() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UiRenderManager(Const_UiRenderManager new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UiRenderManager(Const_UiRenderManager arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UiRenderManager` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UiRenderManager`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UiRenderManager`/`Const_UiRenderManager` directly.
    public class _InOptMut_UiRenderManager
    {
        public UiRenderManager? Opt;

        public _InOptMut_UiRenderManager() {}
        public _InOptMut_UiRenderManager(UiRenderManager value) {Opt = value;}
        public static implicit operator _InOptMut_UiRenderManager(UiRenderManager value) {return new(value);}
    }

    /// This is used for optional parameters of class `UiRenderManager` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UiRenderManager`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UiRenderManager`/`Const_UiRenderManager` to pass it to the function.
    public class _InOptConst_UiRenderManager
    {
        public Const_UiRenderManager? Opt;

        public _InOptConst_UiRenderManager() {}
        public _InOptConst_UiRenderManager(Const_UiRenderManager value) {Opt = value;}
        public static implicit operator _InOptConst_UiRenderManager(Const_UiRenderManager value) {return new(value);}
    }

    /// Generated from class `MR::IRenderObject`.
    /// This is the const half of the class.
    public class Const_IRenderObject : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IRenderObject(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_Destroy", ExactSpelling = true)]
            extern static void __MR_IRenderObject_Destroy(_Underlying *_this);
            __MR_IRenderObject_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IRenderObject() {Dispose(false);}

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::IRenderObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_IRenderObject_heapBytes(_Underlying *_this);
            return __MR_IRenderObject_heapBytes(_UnderlyingPtr);
        }

        /// returns the amount of memory this object allocated in OpenGL
        /// Generated from method `MR::IRenderObject::glBytes`.
        public unsafe ulong GlBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_glBytes", ExactSpelling = true)]
            extern static ulong __MR_IRenderObject_glBytes(_Underlying *_this);
            return __MR_IRenderObject_glBytes(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IRenderObject`.
    /// This is the non-const half of the class.
    public class IRenderObject : Const_IRenderObject
    {
        internal unsafe IRenderObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Returns true if something was rendered, or false if nothing to render.
        /// Generated from method `MR::IRenderObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_render", ExactSpelling = true)]
            extern static byte __MR_IRenderObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *params_);
            return __MR_IRenderObject_render(_UnderlyingPtr, params_._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::IRenderObject::renderPicker`.
        public unsafe void RenderPicker(MR.Const_ModelBaseRenderParams params_, uint geomId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_renderPicker", ExactSpelling = true)]
            extern static void __MR_IRenderObject_renderPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *params_, uint geomId);
            __MR_IRenderObject_renderPicker(_UnderlyingPtr, params_._UnderlyingPtr, geomId);
        }

        /// binds all data for this render object, not to bind ever again (until object becomes dirty)
        /// Generated from method `MR::IRenderObject::forceBindAll`.
        public unsafe void ForceBindAll()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_forceBindAll", ExactSpelling = true)]
            extern static void __MR_IRenderObject_forceBindAll(_Underlying *_this);
            __MR_IRenderObject_forceBindAll(_UnderlyingPtr);
        }

        /// Render the UI. This is repeated for each viewport.
        /// Here you can either render immediately, or insert a task into `params.tasks`, which get Z-sorted.
        /// * `params` will remain alive as long as the tasks are used.
        /// * You'll have at most one living task at a time, so you can write a non-owning pointer to an internal task.
        /// Generated from method `MR::IRenderObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IRenderObject_renderUi", ExactSpelling = true)]
            extern static void __MR_IRenderObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_IRenderObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `IRenderObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IRenderObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IRenderObject`/`Const_IRenderObject` directly.
    public class _InOptMut_IRenderObject
    {
        public IRenderObject? Opt;

        public _InOptMut_IRenderObject() {}
        public _InOptMut_IRenderObject(IRenderObject value) {Opt = value;}
        public static implicit operator _InOptMut_IRenderObject(IRenderObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `IRenderObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IRenderObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IRenderObject`/`Const_IRenderObject` to pass it to the function.
    public class _InOptConst_IRenderObject
    {
        public Const_IRenderObject? Opt;

        public _InOptConst_IRenderObject() {}
        public _InOptConst_IRenderObject(Const_IRenderObject value) {Opt = value;}
        public static implicit operator _InOptConst_IRenderObject(Const_IRenderObject value) {return new(value);}
    }

    /// Generated from function `MR::operator&`.
    public static MR.DepthFunction Bitand(MR.DepthFunction a, MR.DepthFunction b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_DepthFunction", ExactSpelling = true)]
        extern static MR.DepthFunction __MR_bitand_MR_DepthFunction(MR.DepthFunction a, MR.DepthFunction b);
        return __MR_bitand_MR_DepthFunction(a, b);
    }

    /// Generated from function `MR::operator|`.
    public static MR.DepthFunction Bitor(MR.DepthFunction a, MR.DepthFunction b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_DepthFunction", ExactSpelling = true)]
        extern static MR.DepthFunction __MR_bitor_MR_DepthFunction(MR.DepthFunction a, MR.DepthFunction b);
        return __MR_bitor_MR_DepthFunction(a, b);
    }

    /// Generated from function `MR::operator~`.
    public static MR.DepthFunction Compl(MR.DepthFunction a)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compl_MR_DepthFunction", ExactSpelling = true)]
        extern static MR.DepthFunction __MR_compl_MR_DepthFunction(MR.DepthFunction a);
        return __MR_compl_MR_DepthFunction(a);
    }

    /// Generated from function `MR::operator&=`.
    public static unsafe ref MR.DepthFunction BitandAssign(ref MR.DepthFunction a, MR.DepthFunction b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_assign_MR_DepthFunction", ExactSpelling = true)]
        extern static MR.DepthFunction *__MR_bitand_assign_MR_DepthFunction(MR.DepthFunction *a, MR.DepthFunction b);
        fixed (MR.DepthFunction *__ptr_a = &a)
        {
            return ref *__MR_bitand_assign_MR_DepthFunction(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator|=`.
    public static unsafe ref MR.DepthFunction BitorAssign(ref MR.DepthFunction a, MR.DepthFunction b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_assign_MR_DepthFunction", ExactSpelling = true)]
        extern static MR.DepthFunction *__MR_bitor_assign_MR_DepthFunction(MR.DepthFunction *a, MR.DepthFunction b);
        fixed (MR.DepthFunction *__ptr_a = &a)
        {
            return ref *__MR_bitor_assign_MR_DepthFunction(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator*`.
    public static MR.DepthFunction Mul(MR.DepthFunction a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_DepthFunction_bool", ExactSpelling = true)]
        extern static MR.DepthFunction __MR_mul_MR_DepthFunction_bool(MR.DepthFunction a, byte b);
        return __MR_mul_MR_DepthFunction_bool(a, b ? (byte)1 : (byte)0);
    }

    /// Generated from function `MR::operator*`.
    public static MR.DepthFunction Mul(bool a, MR.DepthFunction b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_DepthFunction", ExactSpelling = true)]
        extern static MR.DepthFunction __MR_mul_bool_MR_DepthFunction(byte a, MR.DepthFunction b);
        return __MR_mul_bool_MR_DepthFunction(a ? (byte)1 : (byte)0, b);
    }

    /// Generated from function `MR::operator*=`.
    public static unsafe ref MR.DepthFunction MulAssign(ref MR.DepthFunction a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_DepthFunction_bool", ExactSpelling = true)]
        extern static MR.DepthFunction *__MR_mul_assign_MR_DepthFunction_bool(MR.DepthFunction *a, byte b);
        fixed (MR.DepthFunction *__ptr_a = &a)
        {
            return ref *__MR_mul_assign_MR_DepthFunction_bool(__ptr_a, b ? (byte)1 : (byte)0);
        }
    }

    /// Generated from function `MR::operator&`.
    public static MR.BasicUiRenderTask.InteractionMask Bitand(MR.BasicUiRenderTask.InteractionMask a, MR.BasicUiRenderTask.InteractionMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_BasicUiRenderTask_InteractionMask", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask __MR_bitand_MR_BasicUiRenderTask_InteractionMask(MR.BasicUiRenderTask.InteractionMask a, MR.BasicUiRenderTask.InteractionMask b);
        return __MR_bitand_MR_BasicUiRenderTask_InteractionMask(a, b);
    }

    /// Generated from function `MR::operator|`.
    public static MR.BasicUiRenderTask.InteractionMask Bitor(MR.BasicUiRenderTask.InteractionMask a, MR.BasicUiRenderTask.InteractionMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_BasicUiRenderTask_InteractionMask", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask __MR_bitor_MR_BasicUiRenderTask_InteractionMask(MR.BasicUiRenderTask.InteractionMask a, MR.BasicUiRenderTask.InteractionMask b);
        return __MR_bitor_MR_BasicUiRenderTask_InteractionMask(a, b);
    }

    /// Generated from function `MR::operator~`.
    public static MR.BasicUiRenderTask.InteractionMask Compl(MR.BasicUiRenderTask.InteractionMask a)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compl_MR_BasicUiRenderTask_InteractionMask", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask __MR_compl_MR_BasicUiRenderTask_InteractionMask(MR.BasicUiRenderTask.InteractionMask a);
        return __MR_compl_MR_BasicUiRenderTask_InteractionMask(a);
    }

    /// Generated from function `MR::operator&=`.
    public static unsafe ref MR.BasicUiRenderTask.InteractionMask BitandAssign(ref MR.BasicUiRenderTask.InteractionMask a, MR.BasicUiRenderTask.InteractionMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_assign_MR_BasicUiRenderTask_InteractionMask", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask *__MR_bitand_assign_MR_BasicUiRenderTask_InteractionMask(MR.BasicUiRenderTask.InteractionMask *a, MR.BasicUiRenderTask.InteractionMask b);
        fixed (MR.BasicUiRenderTask.InteractionMask *__ptr_a = &a)
        {
            return ref *__MR_bitand_assign_MR_BasicUiRenderTask_InteractionMask(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator|=`.
    public static unsafe ref MR.BasicUiRenderTask.InteractionMask BitorAssign(ref MR.BasicUiRenderTask.InteractionMask a, MR.BasicUiRenderTask.InteractionMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_assign_MR_BasicUiRenderTask_InteractionMask", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask *__MR_bitor_assign_MR_BasicUiRenderTask_InteractionMask(MR.BasicUiRenderTask.InteractionMask *a, MR.BasicUiRenderTask.InteractionMask b);
        fixed (MR.BasicUiRenderTask.InteractionMask *__ptr_a = &a)
        {
            return ref *__MR_bitor_assign_MR_BasicUiRenderTask_InteractionMask(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator*`.
    public static MR.BasicUiRenderTask.InteractionMask Mul(MR.BasicUiRenderTask.InteractionMask a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_BasicUiRenderTask_InteractionMask_bool", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask __MR_mul_MR_BasicUiRenderTask_InteractionMask_bool(MR.BasicUiRenderTask.InteractionMask a, byte b);
        return __MR_mul_MR_BasicUiRenderTask_InteractionMask_bool(a, b ? (byte)1 : (byte)0);
    }

    /// Generated from function `MR::operator*`.
    public static MR.BasicUiRenderTask.InteractionMask Mul(bool a, MR.BasicUiRenderTask.InteractionMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_BasicUiRenderTask_InteractionMask", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask __MR_mul_bool_MR_BasicUiRenderTask_InteractionMask(byte a, MR.BasicUiRenderTask.InteractionMask b);
        return __MR_mul_bool_MR_BasicUiRenderTask_InteractionMask(a ? (byte)1 : (byte)0, b);
    }

    /// Generated from function `MR::operator*=`.
    public static unsafe ref MR.BasicUiRenderTask.InteractionMask MulAssign(ref MR.BasicUiRenderTask.InteractionMask a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_BasicUiRenderTask_InteractionMask_bool", ExactSpelling = true)]
        extern static MR.BasicUiRenderTask.InteractionMask *__MR_mul_assign_MR_BasicUiRenderTask_InteractionMask_bool(MR.BasicUiRenderTask.InteractionMask *a, byte b);
        fixed (MR.BasicUiRenderTask.InteractionMask *__ptr_a = &a)
        {
            return ref *__MR_mul_assign_MR_BasicUiRenderTask_InteractionMask_bool(__ptr_a, b ? (byte)1 : (byte)0);
        }
    }
}
