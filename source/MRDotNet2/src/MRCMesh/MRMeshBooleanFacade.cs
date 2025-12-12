public static partial class MR
{
    /// just stores a mesh and its transformation to some fixed reference frame
    /// Generated from class `MR::TransformedMesh`.
    /// This is the const half of the class.
    public class Const_TransformedMesh : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TransformedMesh(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_Destroy", ExactSpelling = true)]
            extern static void __MR_TransformedMesh_Destroy(_Underlying *_this);
            __MR_TransformedMesh_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TransformedMesh() {Dispose(false);}

        public unsafe MR.Const_Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_Get_mesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_TransformedMesh_Get_mesh(_Underlying *_this);
                return new(__MR_TransformedMesh_Get_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_AffineXf3f Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_Get_xf", ExactSpelling = true)]
                extern static MR.Const_AffineXf3f._Underlying *__MR_TransformedMesh_Get_xf(_Underlying *_this);
                return new(__MR_TransformedMesh_Get_xf(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TransformedMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_TransformedMesh_DefaultConstruct();
        }

        /// Generated from constructor `MR::TransformedMesh::TransformedMesh`.
        public unsafe Const_TransformedMesh(MR._ByValue_TransformedMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TransformedMesh._Underlying *_other);
            _UnderlyingPtr = __MR_TransformedMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TransformedMesh::TransformedMesh`.
        /// Parameter `xf` defaults to `{}`.
        public unsafe Const_TransformedMesh(MR._ByValue_Mesh mesh, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_Construct", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_Construct(MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_TransformedMesh_Construct(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null);
        }
    }

    /// just stores a mesh and its transformation to some fixed reference frame
    /// Generated from class `MR::TransformedMesh`.
    /// This is the non-const half of the class.
    public class TransformedMesh : Const_TransformedMesh
    {
        internal unsafe TransformedMesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_GetMutable_mesh", ExactSpelling = true)]
                extern static MR.Mesh._Underlying *__MR_TransformedMesh_GetMutable_mesh(_Underlying *_this);
                return new(__MR_TransformedMesh_GetMutable_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_AffineXf3f Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_GetMutable_xf", ExactSpelling = true)]
                extern static MR.Mut_AffineXf3f._Underlying *__MR_TransformedMesh_GetMutable_xf(_Underlying *_this);
                return new(__MR_TransformedMesh_GetMutable_xf(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TransformedMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_TransformedMesh_DefaultConstruct();
        }

        /// Generated from constructor `MR::TransformedMesh::TransformedMesh`.
        public unsafe TransformedMesh(MR._ByValue_TransformedMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TransformedMesh._Underlying *_other);
            _UnderlyingPtr = __MR_TransformedMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TransformedMesh::TransformedMesh`.
        /// Parameter `xf` defaults to `{}`.
        public unsafe TransformedMesh(MR._ByValue_Mesh mesh, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_Construct", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_Construct(MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_TransformedMesh_Construct(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Generated from method `MR::TransformedMesh::operator=`.
        public unsafe MR.TransformedMesh Assign(MR._ByValue_TransformedMesh _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformedMesh_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_TransformedMesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TransformedMesh._Underlying *_other);
            return new(__MR_TransformedMesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// union operation on two meshes
        /// Generated from function `MR::operator+=`.
        public unsafe MR.TransformedMesh AddAssign(MR.Const_TransformedMesh b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_TransformedMesh", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_add_assign_MR_TransformedMesh(_Underlying *a, MR.Const_TransformedMesh._Underlying *b);
            return new(__MR_add_assign_MR_TransformedMesh(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// difference operation on two meshes
        /// Generated from function `MR::operator-=`.
        public unsafe MR.TransformedMesh SubAssign(MR.Const_TransformedMesh b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_TransformedMesh", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_sub_assign_MR_TransformedMesh(_Underlying *a, MR.Const_TransformedMesh._Underlying *b);
            return new(__MR_sub_assign_MR_TransformedMesh(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// intersection operation on two meshes
        /// Generated from function `MR::operator*=`.
        public unsafe MR.TransformedMesh MulAssign(MR.Const_TransformedMesh b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_TransformedMesh", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_mul_assign_MR_TransformedMesh(_Underlying *a, MR.Const_TransformedMesh._Underlying *b);
            return new(__MR_mul_assign_MR_TransformedMesh(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TransformedMesh` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TransformedMesh`/`Const_TransformedMesh` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TransformedMesh
    {
        internal readonly Const_TransformedMesh? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TransformedMesh() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TransformedMesh(Const_TransformedMesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TransformedMesh(Const_TransformedMesh arg) {return new(arg);}
        public _ByValue_TransformedMesh(MR.Misc._Moved<TransformedMesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TransformedMesh(MR.Misc._Moved<TransformedMesh> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TransformedMesh` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TransformedMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TransformedMesh`/`Const_TransformedMesh` directly.
    public class _InOptMut_TransformedMesh
    {
        public TransformedMesh? Opt;

        public _InOptMut_TransformedMesh() {}
        public _InOptMut_TransformedMesh(TransformedMesh value) {Opt = value;}
        public static implicit operator _InOptMut_TransformedMesh(TransformedMesh value) {return new(value);}
    }

    /// This is used for optional parameters of class `TransformedMesh` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TransformedMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TransformedMesh`/`Const_TransformedMesh` to pass it to the function.
    public class _InOptConst_TransformedMesh
    {
        public Const_TransformedMesh? Opt;

        public _InOptConst_TransformedMesh() {}
        public _InOptConst_TransformedMesh(Const_TransformedMesh value) {Opt = value;}
        public static implicit operator _InOptConst_TransformedMesh(Const_TransformedMesh value) {return new(value);}
    }

    /// the purpose of this class is to be a replacement for MeshVoxelsConverter
    /// in case one wants to quickly assess the change from voxel-based to mesh-based boolean
    /// Generated from class `MR::MeshMeshConverter`.
    /// This is the const half of the class.
    public class Const_MeshMeshConverter : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshMeshConverter(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshMeshConverter_Destroy(_Underlying *_this);
            __MR_MeshMeshConverter_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshMeshConverter() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshMeshConverter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshMeshConverter._Underlying *__MR_MeshMeshConverter_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshMeshConverter_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshMeshConverter::MeshMeshConverter`.
        public unsafe Const_MeshMeshConverter(MR.Const_MeshMeshConverter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshConverter._Underlying *__MR_MeshMeshConverter_ConstructFromAnother(MR.MeshMeshConverter._Underlying *_other);
            _UnderlyingPtr = __MR_MeshMeshConverter_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshMeshConverter::operator()`.
        /// Parameter `xf` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.TransformedMesh> Call(MR._ByValue_Mesh mesh, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_call_2", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_MeshMeshConverter_call_2(_Underlying *_this, MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *xf);
            return MR.Misc.Move(new MR.TransformedMesh(__MR_MeshMeshConverter_call_2(_UnderlyingPtr, mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from method `MR::MeshMeshConverter::operator()`.
        public unsafe MR.Misc._Moved<MR.TransformedMesh> Call(MR.Const_ObjectMesh obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_call_1_const_MR_ObjectMesh_ref", ExactSpelling = true)]
            extern static MR.TransformedMesh._Underlying *__MR_MeshMeshConverter_call_1_const_MR_ObjectMesh_ref(_Underlying *_this, MR.Const_ObjectMesh._Underlying *obj);
            return MR.Misc.Move(new MR.TransformedMesh(__MR_MeshMeshConverter_call_1_const_MR_ObjectMesh_ref(_UnderlyingPtr, obj._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::MeshMeshConverter::operator()`.
        public unsafe MR.Const_Mesh Call(MR.Const_TransformedMesh xm)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_call_1_const_MR_TransformedMesh_ref", ExactSpelling = true)]
            extern static MR.Const_Mesh._Underlying *__MR_MeshMeshConverter_call_1_const_MR_TransformedMesh_ref(_Underlying *_this, MR.Const_TransformedMesh._Underlying *xm);
            return new(__MR_MeshMeshConverter_call_1_const_MR_TransformedMesh_ref(_UnderlyingPtr, xm._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshMeshConverter::operator()`.
        public unsafe MR.Misc._Moved<MR.Mesh> Call(MR.Misc._Moved<MR.TransformedMesh> xm)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_call_1_MR_TransformedMesh_rvalue_ref", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_MeshMeshConverter_call_1_MR_TransformedMesh_rvalue_ref(_Underlying *_this, MR.TransformedMesh._Underlying *xm);
            return MR.Misc.Move(new MR.Mesh(__MR_MeshMeshConverter_call_1_MR_TransformedMesh_rvalue_ref(_UnderlyingPtr, xm.Value._UnderlyingPtr), is_owning: false));
        }
    }

    /// the purpose of this class is to be a replacement for MeshVoxelsConverter
    /// in case one wants to quickly assess the change from voxel-based to mesh-based boolean
    /// Generated from class `MR::MeshMeshConverter`.
    /// This is the non-const half of the class.
    public class MeshMeshConverter : Const_MeshMeshConverter
    {
        internal unsafe MeshMeshConverter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshMeshConverter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshMeshConverter._Underlying *__MR_MeshMeshConverter_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshMeshConverter_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshMeshConverter::MeshMeshConverter`.
        public unsafe MeshMeshConverter(MR.Const_MeshMeshConverter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshConverter._Underlying *__MR_MeshMeshConverter_ConstructFromAnother(MR.MeshMeshConverter._Underlying *_other);
            _UnderlyingPtr = __MR_MeshMeshConverter_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshMeshConverter::operator=`.
        public unsafe MR.MeshMeshConverter Assign(MR.Const_MeshMeshConverter _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshMeshConverter_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshMeshConverter._Underlying *__MR_MeshMeshConverter_AssignFromAnother(_Underlying *_this, MR.MeshMeshConverter._Underlying *_other);
            return new(__MR_MeshMeshConverter_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshMeshConverter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshMeshConverter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshMeshConverter`/`Const_MeshMeshConverter` directly.
    public class _InOptMut_MeshMeshConverter
    {
        public MeshMeshConverter? Opt;

        public _InOptMut_MeshMeshConverter() {}
        public _InOptMut_MeshMeshConverter(MeshMeshConverter value) {Opt = value;}
        public static implicit operator _InOptMut_MeshMeshConverter(MeshMeshConverter value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshMeshConverter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshMeshConverter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshMeshConverter`/`Const_MeshMeshConverter` to pass it to the function.
    public class _InOptConst_MeshMeshConverter
    {
        public Const_MeshMeshConverter? Opt;

        public _InOptConst_MeshMeshConverter() {}
        public _InOptConst_MeshMeshConverter(Const_MeshMeshConverter value) {Opt = value;}
        public static implicit operator _InOptConst_MeshMeshConverter(Const_MeshMeshConverter value) {return new(value);}
    }
}
