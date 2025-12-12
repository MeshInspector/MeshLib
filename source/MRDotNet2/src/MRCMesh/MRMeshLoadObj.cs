public static partial class MR
{
    public static partial class MeshLoad
    {
        /// Generated from class `MR::MeshLoad::ObjLoadSettings`.
        /// This is the const half of the class.
        public class Const_ObjLoadSettings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ObjLoadSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshLoad_ObjLoadSettings_Destroy(_Underlying *_this);
                __MR_MeshLoad_ObjLoadSettings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ObjLoadSettings() {Dispose(false);}

            /// if true then vertices will be returned relative to some transformation to avoid precision loss
            public unsafe bool CustomXf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_Get_customXf", ExactSpelling = true)]
                    extern static bool *__MR_MeshLoad_ObjLoadSettings_Get_customXf(_Underlying *_this);
                    return *__MR_MeshLoad_ObjLoadSettings_Get_customXf(_UnderlyingPtr);
                }
            }

            /// if true, the number of skipped faces (faces than can't be created) will be counted
            public unsafe bool CountSkippedFaces
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_Get_countSkippedFaces", ExactSpelling = true)]
                    extern static bool *__MR_MeshLoad_ObjLoadSettings_Get_countSkippedFaces(_Underlying *_this);
                    return *__MR_MeshLoad_ObjLoadSettings_Get_countSkippedFaces(_UnderlyingPtr);
                }
            }

            /// callback for set progress and stop process
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_Get_callback", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MeshLoad_ObjLoadSettings_Get_callback(_Underlying *_this);
                    return new(__MR_MeshLoad_ObjLoadSettings_Get_callback(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ObjLoadSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshLoad_ObjLoadSettings_DefaultConstruct();
            }

            /// Constructs `MR::MeshLoad::ObjLoadSettings` elementwise.
            public unsafe Const_ObjLoadSettings(bool customXf, bool countSkippedFaces, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_ConstructFrom(byte customXf, byte countSkippedFaces, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
                _UnderlyingPtr = __MR_MeshLoad_ObjLoadSettings_ConstructFrom(customXf ? (byte)1 : (byte)0, countSkippedFaces ? (byte)1 : (byte)0, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::MeshLoad::ObjLoadSettings::ObjLoadSettings`.
            public unsafe Const_ObjLoadSettings(MR.MeshLoad._ByValue_ObjLoadSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshLoad.ObjLoadSettings._Underlying *_other);
                _UnderlyingPtr = __MR_MeshLoad_ObjLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::MeshLoad::ObjLoadSettings`.
        /// This is the non-const half of the class.
        public class ObjLoadSettings : Const_ObjLoadSettings
        {
            internal unsafe ObjLoadSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// if true then vertices will be returned relative to some transformation to avoid precision loss
            public new unsafe ref bool CustomXf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_GetMutable_customXf", ExactSpelling = true)]
                    extern static bool *__MR_MeshLoad_ObjLoadSettings_GetMutable_customXf(_Underlying *_this);
                    return ref *__MR_MeshLoad_ObjLoadSettings_GetMutable_customXf(_UnderlyingPtr);
                }
            }

            /// if true, the number of skipped faces (faces than can't be created) will be counted
            public new unsafe ref bool CountSkippedFaces
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_GetMutable_countSkippedFaces", ExactSpelling = true)]
                    extern static bool *__MR_MeshLoad_ObjLoadSettings_GetMutable_countSkippedFaces(_Underlying *_this);
                    return ref *__MR_MeshLoad_ObjLoadSettings_GetMutable_countSkippedFaces(_UnderlyingPtr);
                }
            }

            /// callback for set progress and stop process
            public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_GetMutable_callback", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MeshLoad_ObjLoadSettings_GetMutable_callback(_Underlying *_this);
                    return new(__MR_MeshLoad_ObjLoadSettings_GetMutable_callback(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ObjLoadSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshLoad_ObjLoadSettings_DefaultConstruct();
            }

            /// Constructs `MR::MeshLoad::ObjLoadSettings` elementwise.
            public unsafe ObjLoadSettings(bool customXf, bool countSkippedFaces, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_ConstructFrom(byte customXf, byte countSkippedFaces, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
                _UnderlyingPtr = __MR_MeshLoad_ObjLoadSettings_ConstructFrom(customXf ? (byte)1 : (byte)0, countSkippedFaces ? (byte)1 : (byte)0, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::MeshLoad::ObjLoadSettings::ObjLoadSettings`.
            public unsafe ObjLoadSettings(MR.MeshLoad._ByValue_ObjLoadSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshLoad.ObjLoadSettings._Underlying *_other);
                _UnderlyingPtr = __MR_MeshLoad_ObjLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshLoad::ObjLoadSettings::operator=`.
            public unsafe MR.MeshLoad.ObjLoadSettings Assign(MR.MeshLoad._ByValue_ObjLoadSettings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_ObjLoadSettings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshLoad.ObjLoadSettings._Underlying *__MR_MeshLoad_ObjLoadSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshLoad.ObjLoadSettings._Underlying *_other);
                return new(__MR_MeshLoad_ObjLoadSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ObjLoadSettings` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ObjLoadSettings`/`Const_ObjLoadSettings` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ObjLoadSettings
        {
            internal readonly Const_ObjLoadSettings? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ObjLoadSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ObjLoadSettings(Const_ObjLoadSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ObjLoadSettings(Const_ObjLoadSettings arg) {return new(arg);}
            public _ByValue_ObjLoadSettings(MR.Misc._Moved<ObjLoadSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ObjLoadSettings(MR.Misc._Moved<ObjLoadSettings> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ObjLoadSettings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjLoadSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ObjLoadSettings`/`Const_ObjLoadSettings` directly.
        public class _InOptMut_ObjLoadSettings
        {
            public ObjLoadSettings? Opt;

            public _InOptMut_ObjLoadSettings() {}
            public _InOptMut_ObjLoadSettings(ObjLoadSettings value) {Opt = value;}
            public static implicit operator _InOptMut_ObjLoadSettings(ObjLoadSettings value) {return new(value);}
        }

        /// This is used for optional parameters of class `ObjLoadSettings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjLoadSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ObjLoadSettings`/`Const_ObjLoadSettings` to pass it to the function.
        public class _InOptConst_ObjLoadSettings
        {
            public Const_ObjLoadSettings? Opt;

            public _InOptConst_ObjLoadSettings() {}
            public _InOptConst_ObjLoadSettings(Const_ObjLoadSettings value) {Opt = value;}
            public static implicit operator _InOptConst_ObjLoadSettings(Const_ObjLoadSettings value) {return new(value);}
        }

        /// Generated from class `MR::MeshLoad::NamedMesh`.
        /// This is the const half of the class.
        public class Const_NamedMesh : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_NamedMesh(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshLoad_NamedMesh_Destroy(_Underlying *_this);
                __MR_MeshLoad_NamedMesh_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_NamedMesh() {Dispose(false);}

            public unsafe MR.Std.Const_String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_name", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_MeshLoad_NamedMesh_Get_name(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Mesh Mesh
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_mesh", ExactSpelling = true)]
                    extern static MR.Const_Mesh._Underlying *__MR_MeshLoad_NamedMesh_Get_mesh(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_mesh(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_VertCoords2 UvCoords
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_uvCoords", ExactSpelling = true)]
                    extern static MR.Const_VertCoords2._Underlying *__MR_MeshLoad_NamedMesh_Get_uvCoords(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_uvCoords(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_VertColors Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_colors", ExactSpelling = true)]
                    extern static MR.Const_VertColors._Underlying *__MR_MeshLoad_NamedMesh_Get_colors(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_colors(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Vector_StdFilesystemPath_MRTextureId TextureFiles
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_textureFiles", ExactSpelling = true)]
                    extern static MR.Const_Vector_StdFilesystemPath_MRTextureId._Underlying *__MR_MeshLoad_NamedMesh_Get_textureFiles(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_textureFiles(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_TexturePerFace TexturePerFace
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_texturePerFace", ExactSpelling = true)]
                    extern static MR.Const_TexturePerFace._Underlying *__MR_MeshLoad_NamedMesh_Get_texturePerFace(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_texturePerFace(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_Optional_MRColor DiffuseColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_diffuseColor", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_MeshLoad_NamedMesh_Get_diffuseColor(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_diffuseColor(_UnderlyingPtr), is_owning: false);
                }
            }

            /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
            public unsafe MR.Const_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_xf", ExactSpelling = true)]
                    extern static MR.Const_AffineXf3f._Underlying *__MR_MeshLoad_NamedMesh_Get_xf(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_Get_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
            public unsafe int SkippedFaceCount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_skippedFaceCount", ExactSpelling = true)]
                    extern static int *__MR_MeshLoad_NamedMesh_Get_skippedFaceCount(_Underlying *_this);
                    return *__MR_MeshLoad_NamedMesh_Get_skippedFaceCount(_UnderlyingPtr);
                }
            }

            /// counter of duplicated vertices (that created for resolve non-manifold geometry)
            public unsafe int DuplicatedVertexCount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_Get_duplicatedVertexCount", ExactSpelling = true)]
                    extern static int *__MR_MeshLoad_NamedMesh_Get_duplicatedVertexCount(_Underlying *_this);
                    return *__MR_MeshLoad_NamedMesh_Get_duplicatedVertexCount(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_NamedMesh() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshLoad_NamedMesh_DefaultConstruct();
            }

            /// Constructs `MR::MeshLoad::NamedMesh` elementwise.
            public unsafe Const_NamedMesh(ReadOnlySpan<char> name, MR._ByValue_Mesh mesh, MR._ByValue_VertCoords2 uvCoords, MR._ByValue_VertColors colors, MR._ByValue_Vector_StdFilesystemPath_MRTextureId textureFiles, MR._ByValue_TexturePerFace texturePerFace, MR._InOpt_Color diffuseColor, MR.AffineXf3f xf, int skippedFaceCount, int duplicatedVertexCount) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_ConstructFrom(byte *name, byte *name_end, MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Misc._PassBy uvCoords_pass_by, MR.VertCoords2._Underlying *uvCoords, MR.Misc._PassBy colors_pass_by, MR.VertColors._Underlying *colors, MR.Misc._PassBy textureFiles_pass_by, MR.Vector_StdFilesystemPath_MRTextureId._Underlying *textureFiles, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace, MR.Color *diffuseColor, MR.AffineXf3f xf, int skippedFaceCount, int duplicatedVertexCount);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_MeshLoad_NamedMesh_ConstructFrom(__ptr_name, __ptr_name + __len_name, mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, uvCoords.PassByMode, uvCoords.Value is not null ? uvCoords.Value._UnderlyingPtr : null, colors.PassByMode, colors.Value is not null ? colors.Value._UnderlyingPtr : null, textureFiles.PassByMode, textureFiles.Value is not null ? textureFiles.Value._UnderlyingPtr : null, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null, diffuseColor.HasValue ? &diffuseColor.Object : null, xf, skippedFaceCount, duplicatedVertexCount);
                }
            }

            /// Generated from constructor `MR::MeshLoad::NamedMesh::NamedMesh`.
            public unsafe Const_NamedMesh(MR.MeshLoad._ByValue_NamedMesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshLoad.NamedMesh._Underlying *_other);
                _UnderlyingPtr = __MR_MeshLoad_NamedMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::MeshLoad::NamedMesh`.
        /// This is the non-const half of the class.
        public class NamedMesh : Const_NamedMesh
        {
            internal unsafe NamedMesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_name(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mesh Mesh
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_mesh", ExactSpelling = true)]
                    extern static MR.Mesh._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_mesh(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_mesh(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.VertCoords2 UvCoords
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_uvCoords", ExactSpelling = true)]
                    extern static MR.VertCoords2._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_uvCoords(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_uvCoords(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.VertColors Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_colors", ExactSpelling = true)]
                    extern static MR.VertColors._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_colors(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_colors(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Vector_StdFilesystemPath_MRTextureId TextureFiles
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_textureFiles", ExactSpelling = true)]
                    extern static MR.Vector_StdFilesystemPath_MRTextureId._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_textureFiles(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_textureFiles(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.TexturePerFace TexturePerFace
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_texturePerFace", ExactSpelling = true)]
                    extern static MR.TexturePerFace._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_texturePerFace(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_texturePerFace(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.Optional_MRColor DiffuseColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_diffuseColor", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRColor._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_diffuseColor(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_diffuseColor(_UnderlyingPtr), is_owning: false);
                }
            }

            /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
            public new unsafe MR.Mut_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.Mut_AffineXf3f._Underlying *__MR_MeshLoad_NamedMesh_GetMutable_xf(_Underlying *_this);
                    return new(__MR_MeshLoad_NamedMesh_GetMutable_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
            public new unsafe ref int SkippedFaceCount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_skippedFaceCount", ExactSpelling = true)]
                    extern static int *__MR_MeshLoad_NamedMesh_GetMutable_skippedFaceCount(_Underlying *_this);
                    return ref *__MR_MeshLoad_NamedMesh_GetMutable_skippedFaceCount(_UnderlyingPtr);
                }
            }

            /// counter of duplicated vertices (that created for resolve non-manifold geometry)
            public new unsafe ref int DuplicatedVertexCount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_GetMutable_duplicatedVertexCount", ExactSpelling = true)]
                    extern static int *__MR_MeshLoad_NamedMesh_GetMutable_duplicatedVertexCount(_Underlying *_this);
                    return ref *__MR_MeshLoad_NamedMesh_GetMutable_duplicatedVertexCount(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe NamedMesh() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshLoad_NamedMesh_DefaultConstruct();
            }

            /// Constructs `MR::MeshLoad::NamedMesh` elementwise.
            public unsafe NamedMesh(ReadOnlySpan<char> name, MR._ByValue_Mesh mesh, MR._ByValue_VertCoords2 uvCoords, MR._ByValue_VertColors colors, MR._ByValue_Vector_StdFilesystemPath_MRTextureId textureFiles, MR._ByValue_TexturePerFace texturePerFace, MR._InOpt_Color diffuseColor, MR.AffineXf3f xf, int skippedFaceCount, int duplicatedVertexCount) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_ConstructFrom(byte *name, byte *name_end, MR.Misc._PassBy mesh_pass_by, MR.Mesh._Underlying *mesh, MR.Misc._PassBy uvCoords_pass_by, MR.VertCoords2._Underlying *uvCoords, MR.Misc._PassBy colors_pass_by, MR.VertColors._Underlying *colors, MR.Misc._PassBy textureFiles_pass_by, MR.Vector_StdFilesystemPath_MRTextureId._Underlying *textureFiles, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace, MR.Color *diffuseColor, MR.AffineXf3f xf, int skippedFaceCount, int duplicatedVertexCount);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_MeshLoad_NamedMesh_ConstructFrom(__ptr_name, __ptr_name + __len_name, mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingPtr : null, uvCoords.PassByMode, uvCoords.Value is not null ? uvCoords.Value._UnderlyingPtr : null, colors.PassByMode, colors.Value is not null ? colors.Value._UnderlyingPtr : null, textureFiles.PassByMode, textureFiles.Value is not null ? textureFiles.Value._UnderlyingPtr : null, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null, diffuseColor.HasValue ? &diffuseColor.Object : null, xf, skippedFaceCount, duplicatedVertexCount);
                }
            }

            /// Generated from constructor `MR::MeshLoad::NamedMesh::NamedMesh`.
            public unsafe NamedMesh(MR.MeshLoad._ByValue_NamedMesh _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshLoad.NamedMesh._Underlying *_other);
                _UnderlyingPtr = __MR_MeshLoad_NamedMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshLoad::NamedMesh::operator=`.
            public unsafe MR.MeshLoad.NamedMesh Assign(MR.MeshLoad._ByValue_NamedMesh _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_NamedMesh_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshLoad.NamedMesh._Underlying *__MR_MeshLoad_NamedMesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshLoad.NamedMesh._Underlying *_other);
                return new(__MR_MeshLoad_NamedMesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `NamedMesh` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `NamedMesh`/`Const_NamedMesh` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_NamedMesh
        {
            internal readonly Const_NamedMesh? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_NamedMesh() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_NamedMesh(Const_NamedMesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_NamedMesh(Const_NamedMesh arg) {return new(arg);}
            public _ByValue_NamedMesh(MR.Misc._Moved<NamedMesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_NamedMesh(MR.Misc._Moved<NamedMesh> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `NamedMesh` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_NamedMesh`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `NamedMesh`/`Const_NamedMesh` directly.
        public class _InOptMut_NamedMesh
        {
            public NamedMesh? Opt;

            public _InOptMut_NamedMesh() {}
            public _InOptMut_NamedMesh(NamedMesh value) {Opt = value;}
            public static implicit operator _InOptMut_NamedMesh(NamedMesh value) {return new(value);}
        }

        /// This is used for optional parameters of class `NamedMesh` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_NamedMesh`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `NamedMesh`/`Const_NamedMesh` to pass it to the function.
        public class _InOptConst_NamedMesh
        {
            public Const_NamedMesh? Opt;

            public _InOptConst_NamedMesh() {}
            public _InOptConst_NamedMesh(Const_NamedMesh value) {Opt = value;}
            public static implicit operator _InOptConst_NamedMesh(Const_NamedMesh value) {return new(value);}
        }

        /// loads meshes from .obj file
        /// Generated from function `MR::MeshLoad::fromSceneObjFile`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString> FromSceneObjFile(ReadOnlySpan<char> file, bool combineAllObjects, MR.MeshLoad.Const_ObjLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromSceneObjFile_3", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString._Underlying *__MR_MeshLoad_fromSceneObjFile_3(byte *file, byte *file_end, byte combineAllObjects, MR.MeshLoad.Const_ObjLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString(__MR_MeshLoad_fromSceneObjFile_3(__ptr_file, __ptr_file + __len_file, combineAllObjects ? (byte)1 : (byte)0, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads meshes from a stream with .obj file contents
        /// important on Windows: in stream must be open in binary mode
        /// \param dir working directory where materials and textures are located
        /// Generated from function `MR::MeshLoad::fromSceneObjFile`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString> FromSceneObjFile(MR.Std.Istream in_, bool combineAllObjects, ReadOnlySpan<char> dir, MR.MeshLoad.Const_ObjLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromSceneObjFile_4", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString._Underlying *__MR_MeshLoad_fromSceneObjFile_4(MR.Std.Istream._Underlying *in_, byte combineAllObjects, byte *dir, byte *dir_end, MR.MeshLoad.Const_ObjLoadSettings._Underlying *settings);
            byte[] __bytes_dir = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dir.Length)];
            int __len_dir = System.Text.Encoding.UTF8.GetBytes(dir, __bytes_dir);
            fixed (byte *__ptr_dir = __bytes_dir)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString(__MR_MeshLoad_fromSceneObjFile_4(in_._UnderlyingPtr, combineAllObjects ? (byte)1 : (byte)0, __ptr_dir, __ptr_dir + __len_dir, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads meshes from memory array with .obj file contents
        /// \param dir working directory where materials and textures are located
        /// Generated from function `MR::MeshLoad::fromSceneObjFile`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString> FromSceneObjFile(byte? data, ulong size, bool combineAllObjects, ReadOnlySpan<char> dir, MR.MeshLoad.Const_ObjLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromSceneObjFile_5", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString._Underlying *__MR_MeshLoad_fromSceneObjFile_5(byte *data, ulong size, byte combineAllObjects, byte *dir, byte *dir_end, MR.MeshLoad.Const_ObjLoadSettings._Underlying *settings);
            byte[] __bytes_dir = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dir.Length)];
            int __len_dir = System.Text.Encoding.UTF8.GetBytes(dir, __bytes_dir);
            fixed (byte *__ptr_dir = __bytes_dir)
            {
                byte __deref_data = data.GetValueOrDefault();
                return MR.Misc.Move(new MR.Expected_StdVectorMRMeshLoadNamedMesh_StdString(__MR_MeshLoad_fromSceneObjFile_5(data.HasValue ? &__deref_data : null, size, combineAllObjects ? (byte)1 : (byte)0, __ptr_dir, __ptr_dir + __len_dir, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// reads all objects from .OBJ file
        /// Generated from function `MR::MeshLoad::loadObjectFromObj`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjects_StdString> LoadObjectFromObj(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_loadObjectFromObj", ExactSpelling = true)]
            extern static MR.Expected_MRLoadedObjects_StdString._Underlying *__MR_MeshLoad_loadObjectFromObj(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRLoadedObjects_StdString(__MR_MeshLoad_loadObjectFromObj(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}
