public static partial class MR
{
    public static partial class MeshSave
    {
        /// Generated from class `MR::MeshSave::CtmSaveOptions`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::SaveSettings`
        /// This is the const half of the class.
        public class Const_CtmSaveOptions : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_CtmSaveOptions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshSave_CtmSaveOptions_Destroy(_Underlying *_this);
                __MR_MeshSave_CtmSaveOptions_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_CtmSaveOptions() {Dispose(false);}

            // Upcasts:
            public static unsafe implicit operator MR.Const_SaveSettings(Const_CtmSaveOptions self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_UpcastTo_MR_SaveSettings", ExactSpelling = true)]
                extern static MR.Const_SaveSettings._Underlying *__MR_MeshSave_CtmSaveOptions_UpcastTo_MR_SaveSettings(_Underlying *_this);
                MR.Const_SaveSettings ret = new(__MR_MeshSave_CtmSaveOptions_UpcastTo_MR_SaveSettings(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            public unsafe MR.MeshSave.CtmSaveOptions.MeshCompression MeshCompression_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_meshCompression", ExactSpelling = true)]
                    extern static MR.MeshSave.CtmSaveOptions.MeshCompression *__MR_MeshSave_CtmSaveOptions_Get_meshCompression(_Underlying *_this);
                    return *__MR_MeshSave_CtmSaveOptions_Get_meshCompression(_UnderlyingPtr);
                }
            }

            //~= 0.00098
            public unsafe float VertexPrecision
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_vertexPrecision", ExactSpelling = true)]
                    extern static float *__MR_MeshSave_CtmSaveOptions_Get_vertexPrecision(_Underlying *_this);
                    return *__MR_MeshSave_CtmSaveOptions_Get_vertexPrecision(_UnderlyingPtr);
                }
            }

            /// LZMA compression: 0 - minimal compression, but fast; 9 - maximal compression, but slow
            public unsafe int CompressionLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_compressionLevel", ExactSpelling = true)]
                    extern static int *__MR_MeshSave_CtmSaveOptions_Get_compressionLevel(_Underlying *_this);
                    return *__MR_MeshSave_CtmSaveOptions_Get_compressionLevel(_UnderlyingPtr);
                }
            }

            /// comment saved in the file
            public unsafe ref readonly byte * Comment
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_comment", ExactSpelling = true)]
                    extern static byte **__MR_MeshSave_CtmSaveOptions_Get_comment(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_Get_comment(_UnderlyingPtr);
                }
            }

            /// true - save valid points/vertices only (pack them);
            /// false - save all points/vertices preserving their indices
            public unsafe bool OnlyValidPoints
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_onlyValidPoints", ExactSpelling = true)]
                    extern static bool *__MR_MeshSave_CtmSaveOptions_Get_onlyValidPoints(_Underlying *_this);
                    return *__MR_MeshSave_CtmSaveOptions_Get_onlyValidPoints(_UnderlyingPtr);
                }
            }

            /// whether to allow packing or shuffling of primitives (triangles in meshes or edges in polylines);
            /// if packPrimitives=true, then ids of invalid primitives are reused by valid primitives
            /// and higher compression (in .ctm format) can be reached if the order of triangles is changed;
            /// if packPrimitives=false then all primitives maintain their ids, and invalid primitives are saved with all vertex ids equal to zero;
            /// currently this flag affects the saving in .ctm and .ply formats only
            public unsafe bool PackPrimitives
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_packPrimitives", ExactSpelling = true)]
                    extern static bool *__MR_MeshSave_CtmSaveOptions_Get_packPrimitives(_Underlying *_this);
                    return *__MR_MeshSave_CtmSaveOptions_Get_packPrimitives(_UnderlyingPtr);
                }
            }

            /// optional per-vertex color to save with the geometry
            public unsafe ref readonly void * Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_colors", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_Get_colors(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_Get_colors(_UnderlyingPtr);
                }
            }

            /// per-face colors for meshes, per-undirected-edge colors for polylines, unused for point clouds and other
            public unsafe ref readonly void * PrimitiveColors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_primitiveColors", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_Get_primitiveColors(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_Get_primitiveColors(_UnderlyingPtr);
                }
            }

            /// optional per-vertex uv coordinate to save with the geometry
            public unsafe ref readonly void * UvMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_uvMap", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_Get_uvMap(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_Get_uvMap(_UnderlyingPtr);
                }
            }

            /// optional texture to save with the geometry
            public unsafe ref readonly void * Texture
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_texture", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_Get_texture(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_Get_texture(_UnderlyingPtr);
                }
            }

            /// the name of file without extension to save texture in some formats (e.g. .OBJ, .PLY)
            public unsafe MR.Std.Const_String MaterialName
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_materialName", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_MeshSave_CtmSaveOptions_Get_materialName(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_Get_materialName(_UnderlyingPtr), is_owning: false);
                }
            }

            /// this transformation can optionally be applied to all vertices (points) of saved object
            public unsafe ref readonly MR.AffineXf3d * Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_xf", ExactSpelling = true)]
                    extern static MR.AffineXf3d **__MR_MeshSave_CtmSaveOptions_Get_xf(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_Get_xf(_UnderlyingPtr);
                }
            }

            /// units of input coordinates and transformation, to be serialized if the format supports it
            public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_lengthUnit", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_MeshSave_CtmSaveOptions_Get_lengthUnit(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// the color of whole object
            public unsafe MR.Std.Const_Optional_MRColor SolidColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_solidColor", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_MeshSave_CtmSaveOptions_Get_solidColor(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_Get_solidColor(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report save progress and cancel saving if user desires
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MeshSave_CtmSaveOptions_Get_progress(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_CtmSaveOptions() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshSave.CtmSaveOptions._Underlying *__MR_MeshSave_CtmSaveOptions_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshSave_CtmSaveOptions_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshSave::CtmSaveOptions::CtmSaveOptions`.
            public unsafe Const_CtmSaveOptions(MR.MeshSave._ByValue_CtmSaveOptions _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshSave.CtmSaveOptions._Underlying *__MR_MeshSave_CtmSaveOptions_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshSave.CtmSaveOptions._Underlying *_other);
                _UnderlyingPtr = __MR_MeshSave_CtmSaveOptions_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            public enum MeshCompression : int
            {
                ///< no compression at all, fast but not effective
                None = 0,
                ///< compression without any loss in vertex coordinates
                Lossless = 1,
                ///< compression with loss in vertex coordinates
                Lossy = 2,
            }
        }

        /// Generated from class `MR::MeshSave::CtmSaveOptions`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::SaveSettings`
        /// This is the non-const half of the class.
        public class CtmSaveOptions : Const_CtmSaveOptions
        {
            internal unsafe CtmSaveOptions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Upcasts:
            public static unsafe implicit operator MR.SaveSettings(CtmSaveOptions self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_UpcastTo_MR_SaveSettings", ExactSpelling = true)]
                extern static MR.SaveSettings._Underlying *__MR_MeshSave_CtmSaveOptions_UpcastTo_MR_SaveSettings(_Underlying *_this);
                MR.SaveSettings ret = new(__MR_MeshSave_CtmSaveOptions_UpcastTo_MR_SaveSettings(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            public new unsafe ref MR.MeshSave.CtmSaveOptions.MeshCompression MeshCompression_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_meshCompression", ExactSpelling = true)]
                    extern static MR.MeshSave.CtmSaveOptions.MeshCompression *__MR_MeshSave_CtmSaveOptions_GetMutable_meshCompression(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_meshCompression(_UnderlyingPtr);
                }
            }

            //~= 0.00098
            public new unsafe ref float VertexPrecision
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_vertexPrecision", ExactSpelling = true)]
                    extern static float *__MR_MeshSave_CtmSaveOptions_GetMutable_vertexPrecision(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_vertexPrecision(_UnderlyingPtr);
                }
            }

            /// LZMA compression: 0 - minimal compression, but fast; 9 - maximal compression, but slow
            public new unsafe ref int CompressionLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_compressionLevel", ExactSpelling = true)]
                    extern static int *__MR_MeshSave_CtmSaveOptions_GetMutable_compressionLevel(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_compressionLevel(_UnderlyingPtr);
                }
            }

            /// comment saved in the file
            public new unsafe ref readonly byte * Comment
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_comment", ExactSpelling = true)]
                    extern static byte **__MR_MeshSave_CtmSaveOptions_GetMutable_comment(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_comment(_UnderlyingPtr);
                }
            }

            /// true - save valid points/vertices only (pack them);
            /// false - save all points/vertices preserving their indices
            public new unsafe ref bool OnlyValidPoints
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_onlyValidPoints", ExactSpelling = true)]
                    extern static bool *__MR_MeshSave_CtmSaveOptions_GetMutable_onlyValidPoints(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_onlyValidPoints(_UnderlyingPtr);
                }
            }

            /// whether to allow packing or shuffling of primitives (triangles in meshes or edges in polylines);
            /// if packPrimitives=true, then ids of invalid primitives are reused by valid primitives
            /// and higher compression (in .ctm format) can be reached if the order of triangles is changed;
            /// if packPrimitives=false then all primitives maintain their ids, and invalid primitives are saved with all vertex ids equal to zero;
            /// currently this flag affects the saving in .ctm and .ply formats only
            public new unsafe ref bool PackPrimitives
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_packPrimitives", ExactSpelling = true)]
                    extern static bool *__MR_MeshSave_CtmSaveOptions_GetMutable_packPrimitives(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_packPrimitives(_UnderlyingPtr);
                }
            }

            /// optional per-vertex color to save with the geometry
            public new unsafe ref readonly void * Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_colors", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_GetMutable_colors(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_colors(_UnderlyingPtr);
                }
            }

            /// per-face colors for meshes, per-undirected-edge colors for polylines, unused for point clouds and other
            public new unsafe ref readonly void * PrimitiveColors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_primitiveColors", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_GetMutable_primitiveColors(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_primitiveColors(_UnderlyingPtr);
                }
            }

            /// optional per-vertex uv coordinate to save with the geometry
            public new unsafe ref readonly void * UvMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_uvMap", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_GetMutable_uvMap(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_uvMap(_UnderlyingPtr);
                }
            }

            /// optional texture to save with the geometry
            public new unsafe ref readonly void * Texture
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_texture", ExactSpelling = true)]
                    extern static void **__MR_MeshSave_CtmSaveOptions_GetMutable_texture(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_texture(_UnderlyingPtr);
                }
            }

            /// the name of file without extension to save texture in some formats (e.g. .OBJ, .PLY)
            public new unsafe MR.Std.String MaterialName
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_materialName", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_MeshSave_CtmSaveOptions_GetMutable_materialName(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_GetMutable_materialName(_UnderlyingPtr), is_owning: false);
                }
            }

            /// this transformation can optionally be applied to all vertices (points) of saved object
            public new unsafe ref readonly MR.AffineXf3d * Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.AffineXf3d **__MR_MeshSave_CtmSaveOptions_GetMutable_xf(_Underlying *_this);
                    return ref *__MR_MeshSave_CtmSaveOptions_GetMutable_xf(_UnderlyingPtr);
                }
            }

            /// units of input coordinates and transformation, to be serialized if the format supports it
            public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_lengthUnit", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_MeshSave_CtmSaveOptions_GetMutable_lengthUnit(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// the color of whole object
            public new unsafe MR.Std.Optional_MRColor SolidColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_solidColor", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRColor._Underlying *__MR_MeshSave_CtmSaveOptions_GetMutable_solidColor(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_GetMutable_solidColor(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report save progress and cancel saving if user desires
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MeshSave_CtmSaveOptions_GetMutable_progress(_Underlying *_this);
                    return new(__MR_MeshSave_CtmSaveOptions_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe CtmSaveOptions() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshSave.CtmSaveOptions._Underlying *__MR_MeshSave_CtmSaveOptions_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshSave_CtmSaveOptions_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshSave::CtmSaveOptions::CtmSaveOptions`.
            public unsafe CtmSaveOptions(MR.MeshSave._ByValue_CtmSaveOptions _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshSave.CtmSaveOptions._Underlying *__MR_MeshSave_CtmSaveOptions_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshSave.CtmSaveOptions._Underlying *_other);
                _UnderlyingPtr = __MR_MeshSave_CtmSaveOptions_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshSave::CtmSaveOptions::operator=`.
            public unsafe MR.MeshSave.CtmSaveOptions Assign(MR.MeshSave._ByValue_CtmSaveOptions _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_CtmSaveOptions_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshSave.CtmSaveOptions._Underlying *__MR_MeshSave_CtmSaveOptions_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshSave.CtmSaveOptions._Underlying *_other);
                return new(__MR_MeshSave_CtmSaveOptions_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `CtmSaveOptions` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `CtmSaveOptions`/`Const_CtmSaveOptions` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_CtmSaveOptions
        {
            internal readonly Const_CtmSaveOptions? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_CtmSaveOptions() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_CtmSaveOptions(Const_CtmSaveOptions new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_CtmSaveOptions(Const_CtmSaveOptions arg) {return new(arg);}
            public _ByValue_CtmSaveOptions(MR.Misc._Moved<CtmSaveOptions> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_CtmSaveOptions(MR.Misc._Moved<CtmSaveOptions> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `CtmSaveOptions` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_CtmSaveOptions`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CtmSaveOptions`/`Const_CtmSaveOptions` directly.
        public class _InOptMut_CtmSaveOptions
        {
            public CtmSaveOptions? Opt;

            public _InOptMut_CtmSaveOptions() {}
            public _InOptMut_CtmSaveOptions(CtmSaveOptions value) {Opt = value;}
            public static implicit operator _InOptMut_CtmSaveOptions(CtmSaveOptions value) {return new(value);}
        }

        /// This is used for optional parameters of class `CtmSaveOptions` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_CtmSaveOptions`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CtmSaveOptions`/`Const_CtmSaveOptions` to pass it to the function.
        public class _InOptConst_CtmSaveOptions
        {
            public Const_CtmSaveOptions? Opt;

            public _InOptConst_CtmSaveOptions() {}
            public _InOptConst_CtmSaveOptions(Const_CtmSaveOptions value) {Opt = value;}
            public static implicit operator _InOptConst_CtmSaveOptions(Const_CtmSaveOptions value) {return new(value);}
        }
    }

    public static partial class PointsSave
    {
        /// Generated from class `MR::PointsSave::CtmSavePointsOptions`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::SaveSettings`
        /// This is the const half of the class.
        public class Const_CtmSavePointsOptions : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_CtmSavePointsOptions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Destroy", ExactSpelling = true)]
                extern static void __MR_PointsSave_CtmSavePointsOptions_Destroy(_Underlying *_this);
                __MR_PointsSave_CtmSavePointsOptions_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_CtmSavePointsOptions() {Dispose(false);}

            // Upcasts:
            public static unsafe implicit operator MR.Const_SaveSettings(Const_CtmSavePointsOptions self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_UpcastTo_MR_SaveSettings", ExactSpelling = true)]
                extern static MR.Const_SaveSettings._Underlying *__MR_PointsSave_CtmSavePointsOptions_UpcastTo_MR_SaveSettings(_Underlying *_this);
                MR.Const_SaveSettings ret = new(__MR_PointsSave_CtmSavePointsOptions_UpcastTo_MR_SaveSettings(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            /// 0 - minimal compression, but fast; 9 - maximal compression, but slow
            public unsafe int CompressionLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_compressionLevel", ExactSpelling = true)]
                    extern static int *__MR_PointsSave_CtmSavePointsOptions_Get_compressionLevel(_Underlying *_this);
                    return *__MR_PointsSave_CtmSavePointsOptions_Get_compressionLevel(_UnderlyingPtr);
                }
            }

            /// comment saved in the file
            public unsafe ref readonly byte * Comment
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_comment", ExactSpelling = true)]
                    extern static byte **__MR_PointsSave_CtmSavePointsOptions_Get_comment(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_Get_comment(_UnderlyingPtr);
                }
            }

            /// true - save valid points/vertices only (pack them);
            /// false - save all points/vertices preserving their indices
            public unsafe bool OnlyValidPoints
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_onlyValidPoints", ExactSpelling = true)]
                    extern static bool *__MR_PointsSave_CtmSavePointsOptions_Get_onlyValidPoints(_Underlying *_this);
                    return *__MR_PointsSave_CtmSavePointsOptions_Get_onlyValidPoints(_UnderlyingPtr);
                }
            }

            /// whether to allow packing or shuffling of primitives (triangles in meshes or edges in polylines);
            /// if packPrimitives=true, then ids of invalid primitives are reused by valid primitives
            /// and higher compression (in .ctm format) can be reached if the order of triangles is changed;
            /// if packPrimitives=false then all primitives maintain their ids, and invalid primitives are saved with all vertex ids equal to zero;
            /// currently this flag affects the saving in .ctm and .ply formats only
            public unsafe bool PackPrimitives
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_packPrimitives", ExactSpelling = true)]
                    extern static bool *__MR_PointsSave_CtmSavePointsOptions_Get_packPrimitives(_Underlying *_this);
                    return *__MR_PointsSave_CtmSavePointsOptions_Get_packPrimitives(_UnderlyingPtr);
                }
            }

            /// optional per-vertex color to save with the geometry
            public unsafe ref readonly void * Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_colors", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_Get_colors(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_Get_colors(_UnderlyingPtr);
                }
            }

            /// per-face colors for meshes, per-undirected-edge colors for polylines, unused for point clouds and other
            public unsafe ref readonly void * PrimitiveColors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_primitiveColors", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_Get_primitiveColors(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_Get_primitiveColors(_UnderlyingPtr);
                }
            }

            /// optional per-vertex uv coordinate to save with the geometry
            public unsafe ref readonly void * UvMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_uvMap", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_Get_uvMap(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_Get_uvMap(_UnderlyingPtr);
                }
            }

            /// optional texture to save with the geometry
            public unsafe ref readonly void * Texture
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_texture", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_Get_texture(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_Get_texture(_UnderlyingPtr);
                }
            }

            /// the name of file without extension to save texture in some formats (e.g. .OBJ, .PLY)
            public unsafe MR.Std.Const_String MaterialName
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_materialName", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_PointsSave_CtmSavePointsOptions_Get_materialName(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_Get_materialName(_UnderlyingPtr), is_owning: false);
                }
            }

            /// this transformation can optionally be applied to all vertices (points) of saved object
            public unsafe ref readonly MR.AffineXf3d * Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_xf", ExactSpelling = true)]
                    extern static MR.AffineXf3d **__MR_PointsSave_CtmSavePointsOptions_Get_xf(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_Get_xf(_UnderlyingPtr);
                }
            }

            /// units of input coordinates and transformation, to be serialized if the format supports it
            public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_lengthUnit", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_PointsSave_CtmSavePointsOptions_Get_lengthUnit(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// the color of whole object
            public unsafe MR.Std.Const_Optional_MRColor SolidColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_solidColor", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_PointsSave_CtmSavePointsOptions_Get_solidColor(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_Get_solidColor(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report save progress and cancel saving if user desires
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PointsSave_CtmSavePointsOptions_Get_progress(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_CtmSavePointsOptions() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PointsSave.CtmSavePointsOptions._Underlying *__MR_PointsSave_CtmSavePointsOptions_DefaultConstruct();
                _UnderlyingPtr = __MR_PointsSave_CtmSavePointsOptions_DefaultConstruct();
            }

            /// Generated from constructor `MR::PointsSave::CtmSavePointsOptions::CtmSavePointsOptions`.
            public unsafe Const_CtmSavePointsOptions(MR.PointsSave._ByValue_CtmSavePointsOptions _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PointsSave.CtmSavePointsOptions._Underlying *__MR_PointsSave_CtmSavePointsOptions_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsSave.CtmSavePointsOptions._Underlying *_other);
                _UnderlyingPtr = __MR_PointsSave_CtmSavePointsOptions_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::PointsSave::CtmSavePointsOptions`.
        /// Base classes:
        ///   Direct: (non-virtual)
        ///     `MR::SaveSettings`
        /// This is the non-const half of the class.
        public class CtmSavePointsOptions : Const_CtmSavePointsOptions
        {
            internal unsafe CtmSavePointsOptions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Upcasts:
            public static unsafe implicit operator MR.SaveSettings(CtmSavePointsOptions self)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_UpcastTo_MR_SaveSettings", ExactSpelling = true)]
                extern static MR.SaveSettings._Underlying *__MR_PointsSave_CtmSavePointsOptions_UpcastTo_MR_SaveSettings(_Underlying *_this);
                MR.SaveSettings ret = new(__MR_PointsSave_CtmSavePointsOptions_UpcastTo_MR_SaveSettings(self._UnderlyingPtr), is_owning: false);
                ret._KeepAlive(self);
                return ret;
            }

            /// 0 - minimal compression, but fast; 9 - maximal compression, but slow
            public new unsafe ref int CompressionLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_compressionLevel", ExactSpelling = true)]
                    extern static int *__MR_PointsSave_CtmSavePointsOptions_GetMutable_compressionLevel(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_compressionLevel(_UnderlyingPtr);
                }
            }

            /// comment saved in the file
            public new unsafe ref readonly byte * Comment
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_comment", ExactSpelling = true)]
                    extern static byte **__MR_PointsSave_CtmSavePointsOptions_GetMutable_comment(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_comment(_UnderlyingPtr);
                }
            }

            /// true - save valid points/vertices only (pack them);
            /// false - save all points/vertices preserving their indices
            public new unsafe ref bool OnlyValidPoints
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_onlyValidPoints", ExactSpelling = true)]
                    extern static bool *__MR_PointsSave_CtmSavePointsOptions_GetMutable_onlyValidPoints(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_onlyValidPoints(_UnderlyingPtr);
                }
            }

            /// whether to allow packing or shuffling of primitives (triangles in meshes or edges in polylines);
            /// if packPrimitives=true, then ids of invalid primitives are reused by valid primitives
            /// and higher compression (in .ctm format) can be reached if the order of triangles is changed;
            /// if packPrimitives=false then all primitives maintain their ids, and invalid primitives are saved with all vertex ids equal to zero;
            /// currently this flag affects the saving in .ctm and .ply formats only
            public new unsafe ref bool PackPrimitives
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_packPrimitives", ExactSpelling = true)]
                    extern static bool *__MR_PointsSave_CtmSavePointsOptions_GetMutable_packPrimitives(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_packPrimitives(_UnderlyingPtr);
                }
            }

            /// optional per-vertex color to save with the geometry
            public new unsafe ref readonly void * Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_colors", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_GetMutable_colors(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_colors(_UnderlyingPtr);
                }
            }

            /// per-face colors for meshes, per-undirected-edge colors for polylines, unused for point clouds and other
            public new unsafe ref readonly void * PrimitiveColors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_primitiveColors", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_GetMutable_primitiveColors(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_primitiveColors(_UnderlyingPtr);
                }
            }

            /// optional per-vertex uv coordinate to save with the geometry
            public new unsafe ref readonly void * UvMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_uvMap", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_GetMutable_uvMap(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_uvMap(_UnderlyingPtr);
                }
            }

            /// optional texture to save with the geometry
            public new unsafe ref readonly void * Texture
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_texture", ExactSpelling = true)]
                    extern static void **__MR_PointsSave_CtmSavePointsOptions_GetMutable_texture(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_texture(_UnderlyingPtr);
                }
            }

            /// the name of file without extension to save texture in some formats (e.g. .OBJ, .PLY)
            public new unsafe MR.Std.String MaterialName
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_materialName", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_PointsSave_CtmSavePointsOptions_GetMutable_materialName(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_GetMutable_materialName(_UnderlyingPtr), is_owning: false);
                }
            }

            /// this transformation can optionally be applied to all vertices (points) of saved object
            public new unsafe ref readonly MR.AffineXf3d * Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.AffineXf3d **__MR_PointsSave_CtmSavePointsOptions_GetMutable_xf(_Underlying *_this);
                    return ref *__MR_PointsSave_CtmSavePointsOptions_GetMutable_xf(_UnderlyingPtr);
                }
            }

            /// units of input coordinates and transformation, to be serialized if the format supports it
            public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_lengthUnit", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_PointsSave_CtmSavePointsOptions_GetMutable_lengthUnit(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// the color of whole object
            public new unsafe MR.Std.Optional_MRColor SolidColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_solidColor", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRColor._Underlying *__MR_PointsSave_CtmSavePointsOptions_GetMutable_solidColor(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_GetMutable_solidColor(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report save progress and cancel saving if user desires
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PointsSave_CtmSavePointsOptions_GetMutable_progress(_Underlying *_this);
                    return new(__MR_PointsSave_CtmSavePointsOptions_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe CtmSavePointsOptions() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PointsSave.CtmSavePointsOptions._Underlying *__MR_PointsSave_CtmSavePointsOptions_DefaultConstruct();
                _UnderlyingPtr = __MR_PointsSave_CtmSavePointsOptions_DefaultConstruct();
            }

            /// Generated from constructor `MR::PointsSave::CtmSavePointsOptions::CtmSavePointsOptions`.
            public unsafe CtmSavePointsOptions(MR.PointsSave._ByValue_CtmSavePointsOptions _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PointsSave.CtmSavePointsOptions._Underlying *__MR_PointsSave_CtmSavePointsOptions_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsSave.CtmSavePointsOptions._Underlying *_other);
                _UnderlyingPtr = __MR_PointsSave_CtmSavePointsOptions_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::PointsSave::CtmSavePointsOptions::operator=`.
            public unsafe MR.PointsSave.CtmSavePointsOptions Assign(MR.PointsSave._ByValue_CtmSavePointsOptions _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_CtmSavePointsOptions_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PointsSave.CtmSavePointsOptions._Underlying *__MR_PointsSave_CtmSavePointsOptions_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsSave.CtmSavePointsOptions._Underlying *_other);
                return new(__MR_PointsSave_CtmSavePointsOptions_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `CtmSavePointsOptions` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `CtmSavePointsOptions`/`Const_CtmSavePointsOptions` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_CtmSavePointsOptions
        {
            internal readonly Const_CtmSavePointsOptions? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_CtmSavePointsOptions() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_CtmSavePointsOptions(Const_CtmSavePointsOptions new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_CtmSavePointsOptions(Const_CtmSavePointsOptions arg) {return new(arg);}
            public _ByValue_CtmSavePointsOptions(MR.Misc._Moved<CtmSavePointsOptions> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_CtmSavePointsOptions(MR.Misc._Moved<CtmSavePointsOptions> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `CtmSavePointsOptions` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_CtmSavePointsOptions`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CtmSavePointsOptions`/`Const_CtmSavePointsOptions` directly.
        public class _InOptMut_CtmSavePointsOptions
        {
            public CtmSavePointsOptions? Opt;

            public _InOptMut_CtmSavePointsOptions() {}
            public _InOptMut_CtmSavePointsOptions(CtmSavePointsOptions value) {Opt = value;}
            public static implicit operator _InOptMut_CtmSavePointsOptions(CtmSavePointsOptions value) {return new(value);}
        }

        /// This is used for optional parameters of class `CtmSavePointsOptions` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_CtmSavePointsOptions`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CtmSavePointsOptions`/`Const_CtmSavePointsOptions` to pass it to the function.
        public class _InOptConst_CtmSavePointsOptions
        {
            public Const_CtmSavePointsOptions? Opt;

            public _InOptConst_CtmSavePointsOptions() {}
            public _InOptConst_CtmSavePointsOptions(Const_CtmSavePointsOptions value) {Opt = value;}
            public static implicit operator _InOptConst_CtmSavePointsOptions(Const_CtmSavePointsOptions value) {return new(value);}
        }
    }

    public static partial class MeshLoad
    {
        /// loads from .ctm file
        /// Generated from function `MR::MeshLoad::fromCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromCtm(ReadOnlySpan<char> file, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromCtm_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromCtm_std_filesystem_path(byte *file, byte *file_end, MR.Const_MeshLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromCtm_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshLoad::fromCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> FromCtm(MR.Std.Istream in_, MR.Const_MeshLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshLoad_fromCtm_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_MeshLoad_fromCtm_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_MeshLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_MeshLoad_fromCtm_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    public static partial class MeshSave
    {
        /// saves in .ctm file
        /// Generated from function `MR::MeshSave::toCtm`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.MeshSave.Const_CtmSaveOptions options)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_MeshSave_CtmSaveOptions", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_MeshSave_CtmSaveOptions(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.MeshSave.Const_CtmSaveOptions._Underlying *options);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_MeshSave_CtmSaveOptions(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, options._UnderlyingPtr), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toCtm`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.MeshSave.Const_CtmSaveOptions options)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toCtm_std_ostream_ref_MR_MeshSave_CtmSaveOptions", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toCtm_std_ostream_ref_MR_MeshSave_CtmSaveOptions(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.MeshSave.Const_CtmSaveOptions._Underlying *options);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toCtm_std_ostream_ref_MR_MeshSave_CtmSaveOptions(mesh._UnderlyingPtr, out_._UnderlyingPtr, options._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::MeshSave::toCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_Mesh mesh, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings(MR.Const_Mesh._Underlying *mesh, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings(mesh._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::MeshSave::toCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_Mesh mesh, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshSave_toCtm_std_ostream_ref_MR_SaveSettings", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshSave_toCtm_std_ostream_ref_MR_SaveSettings(MR.Const_Mesh._Underlying *mesh, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshSave_toCtm_std_ostream_ref_MR_SaveSettings(mesh._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    public static partial class PointsLoad
    {
        /// loads from .ctm file
        /// Generated from function `MR::PointsLoad::fromCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromCtm(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromCtm_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromCtm_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromCtm_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromCtm(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromCtm_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromCtm_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromCtm_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }

    public static partial class PointsSave
    {
        /// saves in .ctm file
        /// Generated from function `MR::PointsSave::toCtm`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.PointsSave.Const_CtmSavePointsOptions options)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toCtm_const_std_filesystem_path_ref_MR_PointsSave_CtmSavePointsOptions", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toCtm_const_std_filesystem_path_ref_MR_PointsSave_CtmSavePointsOptions(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.PointsSave.Const_CtmSavePointsOptions._Underlying *options);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toCtm_const_std_filesystem_path_ref_MR_PointsSave_CtmSavePointsOptions(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, options._UnderlyingPtr), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsSave::toCtm`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_PointCloud points, MR.Std.Ostream out_, MR.PointsSave.Const_CtmSavePointsOptions options)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toCtm_std_ostream_ref_MR_PointsSave_CtmSavePointsOptions", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toCtm_std_ostream_ref_MR_PointsSave_CtmSavePointsOptions(MR.Const_PointCloud._Underlying *points, MR.Std.Ostream._Underlying *out_, MR.PointsSave.Const_CtmSavePointsOptions._Underlying *options);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toCtm_std_ostream_ref_MR_PointsSave_CtmSavePointsOptions(points._UnderlyingPtr, out_._UnderlyingPtr, options._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::PointsSave::toCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_PointCloud points, ReadOnlySpan<char> file, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings(MR.Const_PointCloud._Underlying *points, byte *file, byte *file_end, MR.Const_SaveSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings(points._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsSave::toCtm`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToCtm(MR.Const_PointCloud points, MR.Std.Ostream out_, MR.Const_SaveSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsSave_toCtm_std_ostream_ref_MR_SaveSettings", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsSave_toCtm_std_ostream_ref_MR_SaveSettings(MR.Const_PointCloud._Underlying *points, MR.Std.Ostream._Underlying *out_, MR.Const_SaveSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsSave_toCtm_std_ostream_ref_MR_SaveSettings(points._UnderlyingPtr, out_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }
    }
}
