public static partial class MR
{
    /// mesh and its per-element attributes for ObjectMeshHolder
    /// Generated from class `MR::ObjectMeshData`.
    /// This is the const half of the class.
    public class Const_ObjectMeshData : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjectMeshData(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjectMeshData_Destroy(_Underlying *_this);
            __MR_ObjectMeshData_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectMeshData() {Dispose(false);}

        public unsafe MR.Const_Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_mesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._UnderlyingShared *__MR_ObjectMeshData_Get_mesh(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        // selection
        public unsafe MR.Const_FaceBitSet SelectedFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_selectedFaces", ExactSpelling = true)]
                extern static MR.Const_FaceBitSet._Underlying *__MR_ObjectMeshData_Get_selectedFaces(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_selectedFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_UndirectedEdgeBitSet SelectedEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_selectedEdges", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectMeshData_Get_selectedEdges(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_selectedEdges(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_UndirectedEdgeBitSet Creases
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_creases", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectMeshData_Get_creases(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_creases(_UnderlyingPtr), is_owning: false);
            }
        }

        // colors
        public unsafe MR.Const_VertColors VertColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_vertColors", ExactSpelling = true)]
                extern static MR.Const_VertColors._Underlying *__MR_ObjectMeshData_Get_vertColors(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_vertColors(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_FaceColors FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_faceColors", ExactSpelling = true)]
                extern static MR.Const_FaceColors._Underlying *__MR_ObjectMeshData_Get_faceColors(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_faceColors(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< vertices coordinates in texture
        public unsafe MR.Const_VertCoords2 UvCoordinates
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_uvCoordinates", ExactSpelling = true)]
                extern static MR.Const_VertCoords2._Underlying *__MR_ObjectMeshData_Get_uvCoordinates(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_uvCoordinates(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_TexturePerFace TexturePerFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_Get_texturePerFace", ExactSpelling = true)]
                extern static MR.Const_TexturePerFace._Underlying *__MR_ObjectMeshData_Get_texturePerFace(_Underlying *_this);
                return new(__MR_ObjectMeshData_Get_texturePerFace(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectMeshData() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjectMeshData_DefaultConstruct();
        }

        /// Constructs `MR::ObjectMeshData` elementwise.
        public unsafe Const_ObjectMeshData(MR._ByValue_Mesh mesh, MR._ByValue_FaceBitSet selectedFaces, MR._ByValue_UndirectedEdgeBitSet selectedEdges, MR._ByValue_UndirectedEdgeBitSet creases, MR._ByValue_VertColors vertColors, MR._ByValue_FaceColors faceColors, MR._ByValue_VertCoords2 uvCoordinates, MR._ByValue_TexturePerFace texturePerFace) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_ConstructFrom", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_ConstructFrom(MR.Misc._PassBy mesh_pass_by, MR.Mesh._UnderlyingShared *mesh, MR.Misc._PassBy selectedFaces_pass_by, MR.FaceBitSet._Underlying *selectedFaces, MR.Misc._PassBy selectedEdges_pass_by, MR.UndirectedEdgeBitSet._Underlying *selectedEdges, MR.Misc._PassBy creases_pass_by, MR.UndirectedEdgeBitSet._Underlying *creases, MR.Misc._PassBy vertColors_pass_by, MR.VertColors._Underlying *vertColors, MR.Misc._PassBy faceColors_pass_by, MR.FaceColors._Underlying *faceColors, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace);
            _UnderlyingPtr = __MR_ObjectMeshData_ConstructFrom(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingSharedPtr : null, selectedFaces.PassByMode, selectedFaces.Value is not null ? selectedFaces.Value._UnderlyingPtr : null, selectedEdges.PassByMode, selectedEdges.Value is not null ? selectedEdges.Value._UnderlyingPtr : null, creases.PassByMode, creases.Value is not null ? creases.Value._UnderlyingPtr : null, vertColors.PassByMode, vertColors.Value is not null ? vertColors.Value._UnderlyingPtr : null, faceColors.PassByMode, faceColors.Value is not null ? faceColors.Value._UnderlyingPtr : null, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ObjectMeshData::ObjectMeshData`.
        public unsafe Const_ObjectMeshData(MR._ByValue_ObjectMeshData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectMeshData._Underlying *_other);
            _UnderlyingPtr = __MR_ObjectMeshData_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// returns copy of this object with mesh cloned
        /// Generated from method `MR::ObjectMeshData::clone`.
        public unsafe MR.Misc._Moved<MR.ObjectMeshData> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_clone", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.ObjectMeshData(__MR_ObjectMeshData_clone(_UnderlyingPtr), is_owning: true));
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectMeshData::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectMeshData_heapBytes(_Underlying *_this);
            return __MR_ObjectMeshData_heapBytes(_UnderlyingPtr);
        }
    }

    /// mesh and its per-element attributes for ObjectMeshHolder
    /// Generated from class `MR::ObjectMeshData`.
    /// This is the non-const half of the class.
    public class ObjectMeshData : Const_ObjectMeshData
    {
        internal unsafe ObjectMeshData(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_mesh", ExactSpelling = true)]
                extern static MR.Mesh._UnderlyingShared *__MR_ObjectMeshData_GetMutable_mesh(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        // selection
        public new unsafe MR.FaceBitSet SelectedFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_selectedFaces", ExactSpelling = true)]
                extern static MR.FaceBitSet._Underlying *__MR_ObjectMeshData_GetMutable_selectedFaces(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_selectedFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.UndirectedEdgeBitSet SelectedEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_selectedEdges", ExactSpelling = true)]
                extern static MR.UndirectedEdgeBitSet._Underlying *__MR_ObjectMeshData_GetMutable_selectedEdges(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_selectedEdges(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.UndirectedEdgeBitSet Creases
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_creases", ExactSpelling = true)]
                extern static MR.UndirectedEdgeBitSet._Underlying *__MR_ObjectMeshData_GetMutable_creases(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_creases(_UnderlyingPtr), is_owning: false);
            }
        }

        // colors
        public new unsafe MR.VertColors VertColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_vertColors", ExactSpelling = true)]
                extern static MR.VertColors._Underlying *__MR_ObjectMeshData_GetMutable_vertColors(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_vertColors(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.FaceColors FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_faceColors", ExactSpelling = true)]
                extern static MR.FaceColors._Underlying *__MR_ObjectMeshData_GetMutable_faceColors(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_faceColors(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< vertices coordinates in texture
        public new unsafe MR.VertCoords2 UvCoordinates
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_uvCoordinates", ExactSpelling = true)]
                extern static MR.VertCoords2._Underlying *__MR_ObjectMeshData_GetMutable_uvCoordinates(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_uvCoordinates(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.TexturePerFace TexturePerFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_GetMutable_texturePerFace", ExactSpelling = true)]
                extern static MR.TexturePerFace._Underlying *__MR_ObjectMeshData_GetMutable_texturePerFace(_Underlying *_this);
                return new(__MR_ObjectMeshData_GetMutable_texturePerFace(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectMeshData() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjectMeshData_DefaultConstruct();
        }

        /// Constructs `MR::ObjectMeshData` elementwise.
        public unsafe ObjectMeshData(MR._ByValue_Mesh mesh, MR._ByValue_FaceBitSet selectedFaces, MR._ByValue_UndirectedEdgeBitSet selectedEdges, MR._ByValue_UndirectedEdgeBitSet creases, MR._ByValue_VertColors vertColors, MR._ByValue_FaceColors faceColors, MR._ByValue_VertCoords2 uvCoordinates, MR._ByValue_TexturePerFace texturePerFace) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_ConstructFrom", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_ConstructFrom(MR.Misc._PassBy mesh_pass_by, MR.Mesh._UnderlyingShared *mesh, MR.Misc._PassBy selectedFaces_pass_by, MR.FaceBitSet._Underlying *selectedFaces, MR.Misc._PassBy selectedEdges_pass_by, MR.UndirectedEdgeBitSet._Underlying *selectedEdges, MR.Misc._PassBy creases_pass_by, MR.UndirectedEdgeBitSet._Underlying *creases, MR.Misc._PassBy vertColors_pass_by, MR.VertColors._Underlying *vertColors, MR.Misc._PassBy faceColors_pass_by, MR.FaceColors._Underlying *faceColors, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace);
            _UnderlyingPtr = __MR_ObjectMeshData_ConstructFrom(mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingSharedPtr : null, selectedFaces.PassByMode, selectedFaces.Value is not null ? selectedFaces.Value._UnderlyingPtr : null, selectedEdges.PassByMode, selectedEdges.Value is not null ? selectedEdges.Value._UnderlyingPtr : null, creases.PassByMode, creases.Value is not null ? creases.Value._UnderlyingPtr : null, vertColors.PassByMode, vertColors.Value is not null ? vertColors.Value._UnderlyingPtr : null, faceColors.PassByMode, faceColors.Value is not null ? faceColors.Value._UnderlyingPtr : null, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ObjectMeshData::ObjectMeshData`.
        public unsafe ObjectMeshData(MR._ByValue_ObjectMeshData _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectMeshData._Underlying *_other);
            _UnderlyingPtr = __MR_ObjectMeshData_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectMeshData::operator=`.
        public unsafe MR.ObjectMeshData Assign(MR._ByValue_ObjectMeshData _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshData_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectMeshData._Underlying *__MR_ObjectMeshData_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectMeshData._Underlying *_other);
            return new(__MR_ObjectMeshData_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectMeshData` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectMeshData`/`Const_ObjectMeshData` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectMeshData
    {
        internal readonly Const_ObjectMeshData? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectMeshData() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectMeshData(Const_ObjectMeshData new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ObjectMeshData(Const_ObjectMeshData arg) {return new(arg);}
        public _ByValue_ObjectMeshData(MR.Misc._Moved<ObjectMeshData> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectMeshData(MR.Misc._Moved<ObjectMeshData> arg) {return new(arg);}
    }

    /// This is used as a function parameter when the underlying function receives an optional `ObjectMeshData` by value,
    ///   and also has a default argument, meaning it has two different null states.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectMeshData`/`Const_ObjectMeshData` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument.
    /// * Pass `MR.Misc.NullOptType` to pass no object.
    public class _ByValueOptOpt_ObjectMeshData
    {
        internal readonly Const_ObjectMeshData? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValueOptOpt_ObjectMeshData() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValueOptOpt_ObjectMeshData(Const_ObjectMeshData new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValueOptOpt_ObjectMeshData(Const_ObjectMeshData arg) {return new(arg);}
        public _ByValueOptOpt_ObjectMeshData(MR.Misc._Moved<ObjectMeshData> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValueOptOpt_ObjectMeshData(MR.Misc._Moved<ObjectMeshData> arg) {return new(arg);}
        public _ByValueOptOpt_ObjectMeshData(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
        public static implicit operator _ByValueOptOpt_ObjectMeshData(MR.Misc.NullOptType nullopt) {return new(nullopt);}
    }

    /// This is used for optional parameters of class `ObjectMeshData` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectMeshData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectMeshData`/`Const_ObjectMeshData` directly.
    public class _InOptMut_ObjectMeshData
    {
        public ObjectMeshData? Opt;

        public _InOptMut_ObjectMeshData() {}
        public _InOptMut_ObjectMeshData(ObjectMeshData value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectMeshData(ObjectMeshData value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectMeshData` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectMeshData`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectMeshData`/`Const_ObjectMeshData` to pass it to the function.
    public class _InOptConst_ObjectMeshData
    {
        public Const_ObjectMeshData? Opt;

        public _InOptConst_ObjectMeshData() {}
        public _InOptConst_ObjectMeshData(Const_ObjectMeshData value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectMeshData(Const_ObjectMeshData value) {return new(value);}
    }

    /// return all edges separating faces with different colors
    /// Generated from function `MR::edgesBetweenDifferentColors`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> EdgesBetweenDifferentColors(MR.Const_MeshTopology topology, MR.Const_FaceColors colors)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgesBetweenDifferentColors", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_edgesBetweenDifferentColors(MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceColors._Underlying *colors);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_edgesBetweenDifferentColors(topology._UnderlyingPtr, colors._UnderlyingPtr), is_owning: true));
    }
}
