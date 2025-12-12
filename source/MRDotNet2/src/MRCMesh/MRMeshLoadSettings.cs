public static partial class MR
{
    /// setting for mesh loading from external format, and locations of optional output data
    /// Generated from class `MR::MeshLoadSettings`.
    /// This is the const half of the class.
    public class Const_MeshLoadSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshLoadSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshLoadSettings_Destroy(_Underlying *_this);
            __MR_MeshLoadSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshLoadSettings() {Dispose(false);}

        ///< optional load artifact: polyline edges
        public unsafe ref void * Edges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_edges", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_Get_edges(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_edges(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex color map
        public unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_colors", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_Get_colors(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_colors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-face color map
        public unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_faceColors", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_Get_faceColors(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_faceColors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex uv-coordinates
        public unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_uvCoords", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_Get_uvCoords(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_uvCoords(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex normals
        public unsafe ref void * Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_normals", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_Get_normals(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_normals(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: texture image
        public unsafe ref void * Texture
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_texture", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_Get_texture(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_texture(_UnderlyingPtr);
            }
        }

        ///< optional output: counter of skipped faces during mesh creation
        public unsafe ref int * SkippedFaceCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_skippedFaceCount", ExactSpelling = true)]
                extern static int **__MR_MeshLoadSettings_Get_skippedFaceCount(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_skippedFaceCount(_UnderlyingPtr);
            }
        }

        ///< optional output: counter of duplicated vertices (that created for resolve non-manifold geometry)
        public unsafe ref int * DuplicatedVertexCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_duplicatedVertexCount", ExactSpelling = true)]
                extern static int **__MR_MeshLoadSettings_Get_duplicatedVertexCount(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_duplicatedVertexCount(_UnderlyingPtr);
            }
        }

        ///< optional output: transform for the loaded mesh to improve precision of vertex coordinates
        public unsafe ref MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshLoadSettings_Get_xf(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_Get_xf(_UnderlyingPtr);
            }
        }

        ///< callback for set progress and stop process
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MeshLoadSettings_Get_callback(_Underlying *_this);
                return new(__MR_MeshLoadSettings_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::MeshLoadSettings` elementwise.
        public unsafe Const_MeshLoadSettings(MR.Std.Optional_MREdges? edges, MR.VertColors? colors, MR.FaceColors? faceColors, MR.VertCoords2? uvCoords, MR.VertCoords? normals, MR.MeshTexture? texture, MR.Misc.InOut<int>? skippedFaceCount, MR.Misc.InOut<int>? duplicatedVertexCount, MR.Mut_AffineXf3f? xf, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_ConstructFrom(MR.Std.Optional_MREdges._Underlying *edges, MR.VertColors._Underlying *colors, MR.FaceColors._Underlying *faceColors, MR.VertCoords2._Underlying *uvCoords, MR.VertCoords._Underlying *normals, MR.MeshTexture._Underlying *texture, int *skippedFaceCount, int *duplicatedVertexCount, MR.Mut_AffineXf3f._Underlying *xf, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            int __value_skippedFaceCount = skippedFaceCount is not null ? skippedFaceCount.Value : default(int);
            int __value_duplicatedVertexCount = duplicatedVertexCount is not null ? duplicatedVertexCount.Value : default(int);
            _UnderlyingPtr = __MR_MeshLoadSettings_ConstructFrom(edges is not null ? edges._UnderlyingPtr : null, colors is not null ? colors._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null, uvCoords is not null ? uvCoords._UnderlyingPtr : null, normals is not null ? normals._UnderlyingPtr : null, texture is not null ? texture._UnderlyingPtr : null, skippedFaceCount is not null ? &__value_skippedFaceCount : null, duplicatedVertexCount is not null ? &__value_duplicatedVertexCount : null, xf is not null ? xf._UnderlyingPtr : null, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            if (duplicatedVertexCount is not null) duplicatedVertexCount.Value = __value_duplicatedVertexCount;
            if (skippedFaceCount is not null) skippedFaceCount.Value = __value_skippedFaceCount;
        }

        /// Generated from constructor `MR::MeshLoadSettings::MeshLoadSettings`.
        public unsafe Const_MeshLoadSettings(MR._ByValue_MeshLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MeshLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// setting for mesh loading from external format, and locations of optional output data
    /// Generated from class `MR::MeshLoadSettings`.
    /// This is the non-const half of the class.
    public class MeshLoadSettings : Const_MeshLoadSettings
    {
        internal unsafe MeshLoadSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< optional load artifact: polyline edges
        public new unsafe ref void * Edges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_edges", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_GetMutable_edges(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_edges(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex color map
        public new unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_colors", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_GetMutable_colors(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_colors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-face color map
        public new unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_faceColors", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_GetMutable_faceColors(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_faceColors(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex uv-coordinates
        public new unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_uvCoords", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_GetMutable_uvCoords(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_uvCoords(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: per-vertex normals
        public new unsafe ref void * Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_normals", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_GetMutable_normals(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_normals(_UnderlyingPtr);
            }
        }

        ///< optional load artifact: texture image
        public new unsafe ref void * Texture
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_texture", ExactSpelling = true)]
                extern static void **__MR_MeshLoadSettings_GetMutable_texture(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_texture(_UnderlyingPtr);
            }
        }

        ///< optional output: counter of skipped faces during mesh creation
        public new unsafe ref int * SkippedFaceCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_skippedFaceCount", ExactSpelling = true)]
                extern static int **__MR_MeshLoadSettings_GetMutable_skippedFaceCount(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_skippedFaceCount(_UnderlyingPtr);
            }
        }

        ///< optional output: counter of duplicated vertices (that created for resolve non-manifold geometry)
        public new unsafe ref int * DuplicatedVertexCount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_duplicatedVertexCount", ExactSpelling = true)]
                extern static int **__MR_MeshLoadSettings_GetMutable_duplicatedVertexCount(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_duplicatedVertexCount(_UnderlyingPtr);
            }
        }

        ///< optional output: transform for the loaded mesh to improve precision of vertex coordinates
        public new unsafe ref MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshLoadSettings_GetMutable_xf(_Underlying *_this);
                return ref *__MR_MeshLoadSettings_GetMutable_xf(_UnderlyingPtr);
            }
        }

        ///< callback for set progress and stop process
        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MeshLoadSettings_GetMutable_callback(_Underlying *_this);
                return new(__MR_MeshLoadSettings_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::MeshLoadSettings` elementwise.
        public unsafe MeshLoadSettings(MR.Std.Optional_MREdges? edges, MR.VertColors? colors, MR.FaceColors? faceColors, MR.VertCoords2? uvCoords, MR.VertCoords? normals, MR.MeshTexture? texture, MR.Misc.InOut<int>? skippedFaceCount, MR.Misc.InOut<int>? duplicatedVertexCount, MR.Mut_AffineXf3f? xf, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_ConstructFrom(MR.Std.Optional_MREdges._Underlying *edges, MR.VertColors._Underlying *colors, MR.FaceColors._Underlying *faceColors, MR.VertCoords2._Underlying *uvCoords, MR.VertCoords._Underlying *normals, MR.MeshTexture._Underlying *texture, int *skippedFaceCount, int *duplicatedVertexCount, MR.Mut_AffineXf3f._Underlying *xf, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            int __value_skippedFaceCount = skippedFaceCount is not null ? skippedFaceCount.Value : default(int);
            int __value_duplicatedVertexCount = duplicatedVertexCount is not null ? duplicatedVertexCount.Value : default(int);
            _UnderlyingPtr = __MR_MeshLoadSettings_ConstructFrom(edges is not null ? edges._UnderlyingPtr : null, colors is not null ? colors._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null, uvCoords is not null ? uvCoords._UnderlyingPtr : null, normals is not null ? normals._UnderlyingPtr : null, texture is not null ? texture._UnderlyingPtr : null, skippedFaceCount is not null ? &__value_skippedFaceCount : null, duplicatedVertexCount is not null ? &__value_duplicatedVertexCount : null, xf is not null ? xf._UnderlyingPtr : null, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            if (duplicatedVertexCount is not null) duplicatedVertexCount.Value = __value_duplicatedVertexCount;
            if (skippedFaceCount is not null) skippedFaceCount.Value = __value_skippedFaceCount;
        }

        /// Generated from constructor `MR::MeshLoadSettings::MeshLoadSettings`.
        public unsafe MeshLoadSettings(MR._ByValue_MeshLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MeshLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshLoadSettings::operator=`.
        public unsafe MR.MeshLoadSettings Assign(MR._ByValue_MeshLoadSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoadSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshLoadSettings._Underlying *__MR_MeshLoadSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshLoadSettings._Underlying *_other);
            return new(__MR_MeshLoadSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshLoadSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshLoadSettings`/`Const_MeshLoadSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshLoadSettings
    {
        internal readonly Const_MeshLoadSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshLoadSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshLoadSettings(Const_MeshLoadSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshLoadSettings(Const_MeshLoadSettings arg) {return new(arg);}
        public _ByValue_MeshLoadSettings(MR.Misc._Moved<MeshLoadSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshLoadSettings(MR.Misc._Moved<MeshLoadSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshLoadSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshLoadSettings`/`Const_MeshLoadSettings` directly.
    public class _InOptMut_MeshLoadSettings
    {
        public MeshLoadSettings? Opt;

        public _InOptMut_MeshLoadSettings() {}
        public _InOptMut_MeshLoadSettings(MeshLoadSettings value) {Opt = value;}
        public static implicit operator _InOptMut_MeshLoadSettings(MeshLoadSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshLoadSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshLoadSettings`/`Const_MeshLoadSettings` to pass it to the function.
    public class _InOptConst_MeshLoadSettings
    {
        public Const_MeshLoadSettings? Opt;

        public _InOptConst_MeshLoadSettings() {}
        public _InOptConst_MeshLoadSettings(Const_MeshLoadSettings value) {Opt = value;}
        public static implicit operator _InOptConst_MeshLoadSettings(Const_MeshLoadSettings value) {return new(value);}
    }
}
