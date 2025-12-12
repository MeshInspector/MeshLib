public static partial class MR
{
    /// optional load artifacts and other setting for PLY file loading
    /// Generated from class `MR::PlyLoadParams`.
    /// This is the const half of the class.
    public class Const_PlyLoadParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PlyLoadParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PlyLoadParams_Destroy(_Underlying *_this);
            __MR_PlyLoadParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PlyLoadParams() {Dispose(false);}

        ///< optional load artifact: mesh triangles
        public unsafe ref void * Tris
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_tris", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_tris(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_tris(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: polyline edges
        public unsafe ref void * Edges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_edges", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_edges(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_edges(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex color map
        public unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_colors", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_colors(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_colors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-face color map
        public unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_faceColors", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_faceColors(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_faceColors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex uv-coordinates
        public unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_uvCoords", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_uvCoords(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_uvCoords(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-corner uv-coordinates for each triangle
        public unsafe ref void * TriCornerUvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_triCornerUvCoords", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_triCornerUvCoords(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_triCornerUvCoords(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex normals
        public unsafe ref void * Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_normals", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_normals(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_normals(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: texture image
        public unsafe ref void * Texture
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_texture", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_Get_texture(_Underlying *_this);
                return ref *__MR_PlyLoadParams_Get_texture(_UnderlyingPtr);
            }
        }

        ///< directory to load texture files from
        public unsafe MR.Std.Filesystem.Const_Path Dir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_dir", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_PlyLoadParams_Get_dir(_Underlying *_this);
                return new(__MR_PlyLoadParams_Get_dir(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< callback for set progress and stop process
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PlyLoadParams_Get_callback(_Underlying *_this);
                return new(__MR_PlyLoadParams_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PlyLoadParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PlyLoadParams_DefaultConstruct();
        }

        /// Constructs `MR::PlyLoadParams` elementwise.
        public unsafe Const_PlyLoadParams(MR.Std.Optional_MRTriangulation? tris, MR.Std.Optional_MREdges? edges, MR.VertColors? colors, MR.FaceColors? faceColors, MR.VertCoords2? uvCoords, MR.TriCornerUVCoords? triCornerUvCoords, MR.VertCoords? normals, MR.MeshTexture? texture, ReadOnlySpan<char> dir, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_ConstructFrom(MR.Std.Optional_MRTriangulation._Underlying *tris, MR.Std.Optional_MREdges._Underlying *edges, MR.VertColors._Underlying *colors, MR.FaceColors._Underlying *faceColors, MR.VertCoords2._Underlying *uvCoords, MR.TriCornerUVCoords._Underlying *triCornerUvCoords, MR.VertCoords._Underlying *normals, MR.MeshTexture._Underlying *texture, byte *dir, byte *dir_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_dir = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dir.Length)];
            int __len_dir = System.Text.Encoding.UTF8.GetBytes(dir, __bytes_dir);
            fixed (byte *__ptr_dir = __bytes_dir)
            {
                _UnderlyingPtr = __MR_PlyLoadParams_ConstructFrom(tris is not null ? tris._UnderlyingPtr : null, edges is not null ? edges._UnderlyingPtr : null, colors is not null ? colors._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null, uvCoords is not null ? uvCoords._UnderlyingPtr : null, triCornerUvCoords is not null ? triCornerUvCoords._UnderlyingPtr : null, normals is not null ? normals._UnderlyingPtr : null, texture is not null ? texture._UnderlyingPtr : null, __ptr_dir, __ptr_dir + __len_dir, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from constructor `MR::PlyLoadParams::PlyLoadParams`.
        public unsafe Const_PlyLoadParams(MR._ByValue_PlyLoadParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PlyLoadParams._Underlying *_other);
            _UnderlyingPtr = __MR_PlyLoadParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// optional load artifacts and other setting for PLY file loading
    /// Generated from class `MR::PlyLoadParams`.
    /// This is the non-const half of the class.
    public class PlyLoadParams : Const_PlyLoadParams
    {
        internal unsafe PlyLoadParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< optional load artifact: mesh triangles
        public new unsafe ref void * Tris
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_tris", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_tris(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_tris(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: polyline edges
        public new unsafe ref void * Edges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_edges", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_edges(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_edges(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex color map
        public new unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_colors", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_colors(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_colors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-face color map
        public new unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_faceColors", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_faceColors(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_faceColors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex uv-coordinates
        public new unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_uvCoords", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_uvCoords(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_uvCoords(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-corner uv-coordinates for each triangle
        public new unsafe ref void * TriCornerUvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_triCornerUvCoords", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_triCornerUvCoords(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_triCornerUvCoords(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex normals
        public new unsafe ref void * Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_normals", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_normals(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_normals(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: texture image
        public new unsafe ref void * Texture
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_texture", ExactSpelling = true)]
                extern static void **__MR_PlyLoadParams_GetMutable_texture(_Underlying *_this);
                return ref *__MR_PlyLoadParams_GetMutable_texture(_UnderlyingPtr);
            }
        }

        ///< directory to load texture files from
        public new unsafe MR.Std.Filesystem.Path Dir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_dir", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Path._Underlying *__MR_PlyLoadParams_GetMutable_dir(_Underlying *_this);
                return new(__MR_PlyLoadParams_GetMutable_dir(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< callback for set progress and stop process
        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PlyLoadParams_GetMutable_callback(_Underlying *_this);
                return new(__MR_PlyLoadParams_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PlyLoadParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PlyLoadParams_DefaultConstruct();
        }

        /// Constructs `MR::PlyLoadParams` elementwise.
        public unsafe PlyLoadParams(MR.Std.Optional_MRTriangulation? tris, MR.Std.Optional_MREdges? edges, MR.VertColors? colors, MR.FaceColors? faceColors, MR.VertCoords2? uvCoords, MR.TriCornerUVCoords? triCornerUvCoords, MR.VertCoords? normals, MR.MeshTexture? texture, ReadOnlySpan<char> dir, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_ConstructFrom(MR.Std.Optional_MRTriangulation._Underlying *tris, MR.Std.Optional_MREdges._Underlying *edges, MR.VertColors._Underlying *colors, MR.FaceColors._Underlying *faceColors, MR.VertCoords2._Underlying *uvCoords, MR.TriCornerUVCoords._Underlying *triCornerUvCoords, MR.VertCoords._Underlying *normals, MR.MeshTexture._Underlying *texture, byte *dir, byte *dir_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_dir = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dir.Length)];
            int __len_dir = System.Text.Encoding.UTF8.GetBytes(dir, __bytes_dir);
            fixed (byte *__ptr_dir = __bytes_dir)
            {
                _UnderlyingPtr = __MR_PlyLoadParams_ConstructFrom(tris is not null ? tris._UnderlyingPtr : null, edges is not null ? edges._UnderlyingPtr : null, colors is not null ? colors._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null, uvCoords is not null ? uvCoords._UnderlyingPtr : null, triCornerUvCoords is not null ? triCornerUvCoords._UnderlyingPtr : null, normals is not null ? normals._UnderlyingPtr : null, texture is not null ? texture._UnderlyingPtr : null, __ptr_dir, __ptr_dir + __len_dir, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from constructor `MR::PlyLoadParams::PlyLoadParams`.
        public unsafe PlyLoadParams(MR._ByValue_PlyLoadParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PlyLoadParams._Underlying *_other);
            _UnderlyingPtr = __MR_PlyLoadParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PlyLoadParams::operator=`.
        public unsafe MR.PlyLoadParams Assign(MR._ByValue_PlyLoadParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlyLoadParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PlyLoadParams._Underlying *__MR_PlyLoadParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PlyLoadParams._Underlying *_other);
            return new(__MR_PlyLoadParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PlyLoadParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PlyLoadParams`/`Const_PlyLoadParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PlyLoadParams
    {
        internal readonly Const_PlyLoadParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PlyLoadParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PlyLoadParams(Const_PlyLoadParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PlyLoadParams(Const_PlyLoadParams arg) {return new(arg);}
        public _ByValue_PlyLoadParams(MR.Misc._Moved<PlyLoadParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PlyLoadParams(MR.Misc._Moved<PlyLoadParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PlyLoadParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PlyLoadParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PlyLoadParams`/`Const_PlyLoadParams` directly.
    public class _InOptMut_PlyLoadParams
    {
        public PlyLoadParams? Opt;

        public _InOptMut_PlyLoadParams() {}
        public _InOptMut_PlyLoadParams(PlyLoadParams value) {Opt = value;}
        public static implicit operator _InOptMut_PlyLoadParams(PlyLoadParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PlyLoadParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PlyLoadParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PlyLoadParams`/`Const_PlyLoadParams` to pass it to the function.
    public class _InOptConst_PlyLoadParams
    {
        public Const_PlyLoadParams? Opt;

        public _InOptConst_PlyLoadParams() {}
        public _InOptConst_PlyLoadParams(Const_PlyLoadParams value) {Opt = value;}
        public static implicit operator _InOptConst_PlyLoadParams(Const_PlyLoadParams value) {return new(value);}
    }

    /// Generated from function `MR::loadPly`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertCoords_StdString> LoadPly(MR.Std.Istream in_, MR.Const_PlyLoadParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_loadPly_std_istream_ref_MR_PlyLoadParams", ExactSpelling = true)]
        extern static MR.Expected_MRVertCoords_StdString._Underlying *__MR_loadPly_std_istream_ref_MR_PlyLoadParams(MR.Std.Istream._Underlying *in_, MR.Const_PlyLoadParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRVertCoords_StdString(__MR_loadPly_std_istream_ref_MR_PlyLoadParams(in_._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
