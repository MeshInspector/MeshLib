public static partial class MR
{
    /// Generated from class `MR::MeshNormals`.
    /// This is the const half of the class.
    public class Const_MeshNormals : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshNormals(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshNormals_Destroy(_Underlying *_this);
            __MR_MeshNormals_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshNormals() {Dispose(false);}

        public unsafe MR.Const_FaceNormals FaceNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_Get_faceNormals", ExactSpelling = true)]
                extern static MR.Const_FaceNormals._Underlying *__MR_MeshNormals_Get_faceNormals(_Underlying *_this);
                return new(__MR_MeshNormals_Get_faceNormals(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertCoords VertNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_Get_vertNormals", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_MeshNormals_Get_vertNormals(_Underlying *_this);
                return new(__MR_MeshNormals_Get_vertNormals(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshNormals() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshNormals_DefaultConstruct();
        }

        /// Constructs `MR::MeshNormals` elementwise.
        public unsafe Const_MeshNormals(MR._ByValue_FaceNormals faceNormals, MR._ByValue_VertCoords vertNormals) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_ConstructFrom(MR.Misc._PassBy faceNormals_pass_by, MR.FaceNormals._Underlying *faceNormals, MR.Misc._PassBy vertNormals_pass_by, MR.VertCoords._Underlying *vertNormals);
            _UnderlyingPtr = __MR_MeshNormals_ConstructFrom(faceNormals.PassByMode, faceNormals.Value is not null ? faceNormals.Value._UnderlyingPtr : null, vertNormals.PassByMode, vertNormals.Value is not null ? vertNormals.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshNormals::MeshNormals`.
        public unsafe Const_MeshNormals(MR._ByValue_MeshNormals _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshNormals._Underlying *_other);
            _UnderlyingPtr = __MR_MeshNormals_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::MeshNormals`.
    /// This is the non-const half of the class.
    public class MeshNormals : Const_MeshNormals
    {
        internal unsafe MeshNormals(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.FaceNormals FaceNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_GetMutable_faceNormals", ExactSpelling = true)]
                extern static MR.FaceNormals._Underlying *__MR_MeshNormals_GetMutable_faceNormals(_Underlying *_this);
                return new(__MR_MeshNormals_GetMutable_faceNormals(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertCoords VertNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_GetMutable_vertNormals", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_MeshNormals_GetMutable_vertNormals(_Underlying *_this);
                return new(__MR_MeshNormals_GetMutable_vertNormals(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshNormals() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshNormals_DefaultConstruct();
        }

        /// Constructs `MR::MeshNormals` elementwise.
        public unsafe MeshNormals(MR._ByValue_FaceNormals faceNormals, MR._ByValue_VertCoords vertNormals) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_ConstructFrom(MR.Misc._PassBy faceNormals_pass_by, MR.FaceNormals._Underlying *faceNormals, MR.Misc._PassBy vertNormals_pass_by, MR.VertCoords._Underlying *vertNormals);
            _UnderlyingPtr = __MR_MeshNormals_ConstructFrom(faceNormals.PassByMode, faceNormals.Value is not null ? faceNormals.Value._UnderlyingPtr : null, vertNormals.PassByMode, vertNormals.Value is not null ? vertNormals.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshNormals::MeshNormals`.
        public unsafe MeshNormals(MR._ByValue_MeshNormals _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshNormals._Underlying *_other);
            _UnderlyingPtr = __MR_MeshNormals_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshNormals::operator=`.
        public unsafe MR.MeshNormals Assign(MR._ByValue_MeshNormals _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshNormals_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshNormals._Underlying *__MR_MeshNormals_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshNormals._Underlying *_other);
            return new(__MR_MeshNormals_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshNormals` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshNormals`/`Const_MeshNormals` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshNormals
    {
        internal readonly Const_MeshNormals? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshNormals() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshNormals(Const_MeshNormals new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshNormals(Const_MeshNormals arg) {return new(arg);}
        public _ByValue_MeshNormals(MR.Misc._Moved<MeshNormals> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshNormals(MR.Misc._Moved<MeshNormals> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshNormals` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshNormals`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshNormals`/`Const_MeshNormals` directly.
    public class _InOptMut_MeshNormals
    {
        public MeshNormals? Opt;

        public _InOptMut_MeshNormals() {}
        public _InOptMut_MeshNormals(MeshNormals value) {Opt = value;}
        public static implicit operator _InOptMut_MeshNormals(MeshNormals value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshNormals` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshNormals`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshNormals`/`Const_MeshNormals` to pass it to the function.
    public class _InOptConst_MeshNormals
    {
        public Const_MeshNormals? Opt;

        public _InOptConst_MeshNormals() {}
        public _InOptConst_MeshNormals(Const_MeshNormals value) {Opt = value;}
        public static implicit operator _InOptConst_MeshNormals(Const_MeshNormals value) {return new(value);}
    }

    /// returns a vector with face-normal in every element for valid mesh faces
    /// Generated from function `MR::computePerFaceNormals`.
    public static unsafe MR.Misc._Moved<MR.FaceNormals> ComputePerFaceNormals(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computePerFaceNormals", ExactSpelling = true)]
        extern static MR.FaceNormals._Underlying *__MR_computePerFaceNormals(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FaceNormals(__MR_computePerFaceNormals(mesh._UnderlyingPtr), is_owning: true));
    }

    /// fills buffer with face-normals as Vector4f for valid mesh faces
    /// Generated from function `MR::computePerFaceNormals4`.
    public static unsafe void ComputePerFaceNormals4(MR.Const_Mesh mesh, MR.Mut_Vector4f? faceNormals, ulong size)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computePerFaceNormals4", ExactSpelling = true)]
        extern static void __MR_computePerFaceNormals4(MR.Const_Mesh._Underlying *mesh, MR.Mut_Vector4f._Underlying *faceNormals, ulong size);
        __MR_computePerFaceNormals4(mesh._UnderlyingPtr, faceNormals is not null ? faceNormals._UnderlyingPtr : null, size);
    }

    /// returns a vector with vertex normals in every element for valid mesh vertices
    /// Generated from function `MR::computePerVertNormals`.
    public static unsafe MR.Misc._Moved<MR.VertCoords> ComputePerVertNormals(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computePerVertNormals", ExactSpelling = true)]
        extern static MR.VertCoords._Underlying *__MR_computePerVertNormals(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.VertCoords(__MR_computePerVertNormals(mesh._UnderlyingPtr), is_owning: true));
    }

    /// returns a vector with vertex pseudonormals in every element for valid mesh vertices
    /// see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.9173&rep=rep1&type=pdf
    /// Generated from function `MR::computePerVertPseudoNormals`.
    public static unsafe MR.Misc._Moved<MR.VertCoords> ComputePerVertPseudoNormals(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computePerVertPseudoNormals", ExactSpelling = true)]
        extern static MR.VertCoords._Underlying *__MR_computePerVertPseudoNormals(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.VertCoords(__MR_computePerVertPseudoNormals(mesh._UnderlyingPtr), is_owning: true));
    }

    /// computes both per-face and per-vertex normals more efficiently then just calling both previous functions
    /// Generated from function `MR::computeMeshNormals`.
    public static unsafe MR.Misc._Moved<MR.MeshNormals> ComputeMeshNormals(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeMeshNormals", ExactSpelling = true)]
        extern static MR.MeshNormals._Underlying *__MR_computeMeshNormals(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.MeshNormals(__MR_computeMeshNormals(mesh._UnderlyingPtr), is_owning: true));
    }

    /// returns a vector with corner normals in every element for valid mesh faces;
    /// corner normals of adjacent triangles are equal, unless they are separated by crease edges
    /// Generated from function `MR::computePerCornerNormals`.
    public static unsafe MR.Misc._Moved<MR.Vector_StdArrayMRVector3f3_MRFaceId> ComputePerCornerNormals(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet? creases)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computePerCornerNormals", ExactSpelling = true)]
        extern static MR.Vector_StdArrayMRVector3f3_MRFaceId._Underlying *__MR_computePerCornerNormals(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *creases);
        return MR.Misc.Move(new MR.Vector_StdArrayMRVector3f3_MRFaceId(__MR_computePerCornerNormals(mesh._UnderlyingPtr, creases is not null ? creases._UnderlyingPtr : null), is_owning: true));
    }
}
