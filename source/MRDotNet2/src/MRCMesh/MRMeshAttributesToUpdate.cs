public static partial class MR
{
    // the attribute data of the mesh that needs to be updated
    /// Generated from class `MR::MeshAttributesToUpdate`.
    /// This is the const half of the class.
    public class Const_MeshAttributesToUpdate : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshAttributesToUpdate(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshAttributesToUpdate_Destroy(_Underlying *_this);
            __MR_MeshAttributesToUpdate_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshAttributesToUpdate() {Dispose(false);}

        public unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_Get_uvCoords", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_Get_uvCoords(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_Get_uvCoords(_UnderlyingPtr);
            }
        }

        public unsafe ref void * ColorMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_Get_colorMap", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_Get_colorMap(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_Get_colorMap(_UnderlyingPtr);
            }
        }

        public unsafe ref void * TexturePerFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_Get_texturePerFace", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_Get_texturePerFace(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_Get_texturePerFace(_UnderlyingPtr);
            }
        }

        public unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_Get_faceColors", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_Get_faceColors(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_Get_faceColors(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshAttributesToUpdate() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshAttributesToUpdate_DefaultConstruct();
        }

        /// Constructs `MR::MeshAttributesToUpdate` elementwise.
        public unsafe Const_MeshAttributesToUpdate(MR.VertCoords2? uvCoords, MR.VertColors? colorMap, MR.TexturePerFace? texturePerFace, MR.FaceColors? faceColors) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_ConstructFrom(MR.VertCoords2._Underlying *uvCoords, MR.VertColors._Underlying *colorMap, MR.TexturePerFace._Underlying *texturePerFace, MR.FaceColors._Underlying *faceColors);
            _UnderlyingPtr = __MR_MeshAttributesToUpdate_ConstructFrom(uvCoords is not null ? uvCoords._UnderlyingPtr : null, colorMap is not null ? colorMap._UnderlyingPtr : null, texturePerFace is not null ? texturePerFace._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshAttributesToUpdate::MeshAttributesToUpdate`.
        public unsafe Const_MeshAttributesToUpdate(MR.Const_MeshAttributesToUpdate _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_ConstructFromAnother(MR.MeshAttributesToUpdate._Underlying *_other);
            _UnderlyingPtr = __MR_MeshAttributesToUpdate_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // the attribute data of the mesh that needs to be updated
    /// Generated from class `MR::MeshAttributesToUpdate`.
    /// This is the non-const half of the class.
    public class MeshAttributesToUpdate : Const_MeshAttributesToUpdate
    {
        internal unsafe MeshAttributesToUpdate(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_GetMutable_uvCoords", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_GetMutable_uvCoords(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_GetMutable_uvCoords(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * ColorMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_GetMutable_colorMap", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_GetMutable_colorMap(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_GetMutable_colorMap(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * TexturePerFace
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_GetMutable_texturePerFace", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_GetMutable_texturePerFace(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_GetMutable_texturePerFace(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_GetMutable_faceColors", ExactSpelling = true)]
                extern static void **__MR_MeshAttributesToUpdate_GetMutable_faceColors(_Underlying *_this);
                return ref *__MR_MeshAttributesToUpdate_GetMutable_faceColors(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshAttributesToUpdate() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshAttributesToUpdate_DefaultConstruct();
        }

        /// Constructs `MR::MeshAttributesToUpdate` elementwise.
        public unsafe MeshAttributesToUpdate(MR.VertCoords2? uvCoords, MR.VertColors? colorMap, MR.TexturePerFace? texturePerFace, MR.FaceColors? faceColors) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_ConstructFrom(MR.VertCoords2._Underlying *uvCoords, MR.VertColors._Underlying *colorMap, MR.TexturePerFace._Underlying *texturePerFace, MR.FaceColors._Underlying *faceColors);
            _UnderlyingPtr = __MR_MeshAttributesToUpdate_ConstructFrom(uvCoords is not null ? uvCoords._UnderlyingPtr : null, colorMap is not null ? colorMap._UnderlyingPtr : null, texturePerFace is not null ? texturePerFace._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshAttributesToUpdate::MeshAttributesToUpdate`.
        public unsafe MeshAttributesToUpdate(MR.Const_MeshAttributesToUpdate _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_ConstructFromAnother(MR.MeshAttributesToUpdate._Underlying *_other);
            _UnderlyingPtr = __MR_MeshAttributesToUpdate_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshAttributesToUpdate::operator=`.
        public unsafe MR.MeshAttributesToUpdate Assign(MR.Const_MeshAttributesToUpdate _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshAttributesToUpdate_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshAttributesToUpdate._Underlying *__MR_MeshAttributesToUpdate_AssignFromAnother(_Underlying *_this, MR.MeshAttributesToUpdate._Underlying *_other);
            return new(__MR_MeshAttributesToUpdate_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshAttributesToUpdate` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshAttributesToUpdate`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshAttributesToUpdate`/`Const_MeshAttributesToUpdate` directly.
    public class _InOptMut_MeshAttributesToUpdate
    {
        public MeshAttributesToUpdate? Opt;

        public _InOptMut_MeshAttributesToUpdate() {}
        public _InOptMut_MeshAttributesToUpdate(MeshAttributesToUpdate value) {Opt = value;}
        public static implicit operator _InOptMut_MeshAttributesToUpdate(MeshAttributesToUpdate value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshAttributesToUpdate` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshAttributesToUpdate`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshAttributesToUpdate`/`Const_MeshAttributesToUpdate` to pass it to the function.
    public class _InOptConst_MeshAttributesToUpdate
    {
        public Const_MeshAttributesToUpdate? Opt;

        public _InOptConst_MeshAttributesToUpdate() {}
        public _InOptConst_MeshAttributesToUpdate(Const_MeshAttributesToUpdate value) {Opt = value;}
        public static implicit operator _InOptConst_MeshAttributesToUpdate(Const_MeshAttributesToUpdate value) {return new(value);}
    }
}
