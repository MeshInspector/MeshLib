public static partial class MR
{
    /// determines how to save points/lines/mesh
    /// Generated from class `MR::SaveSettings`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshSave::CtmSaveOptions`
    ///     `MR::PointsSave::CtmSavePointsOptions`
    /// This is the const half of the class.
    public class Const_SaveSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SaveSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_SaveSettings_Destroy(_Underlying *_this);
            __MR_SaveSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SaveSettings() {Dispose(false);}

        /// true - save valid points/vertices only (pack them);
        /// false - save all points/vertices preserving their indices
        public unsafe bool OnlyValidPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_onlyValidPoints", ExactSpelling = true)]
                extern static bool *__MR_SaveSettings_Get_onlyValidPoints(_Underlying *_this);
                return *__MR_SaveSettings_Get_onlyValidPoints(_UnderlyingPtr);
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
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_packPrimitives", ExactSpelling = true)]
                extern static bool *__MR_SaveSettings_Get_packPrimitives(_Underlying *_this);
                return *__MR_SaveSettings_Get_packPrimitives(_UnderlyingPtr);
            }
        }

        /// optional per-vertex color to save with the geometry
        public unsafe ref readonly void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_colors", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_Get_colors(_Underlying *_this);
                return ref *__MR_SaveSettings_Get_colors(_UnderlyingPtr);
            }
        }

        /// per-face colors for meshes, per-undirected-edge colors for polylines, unused for point clouds and other
        public unsafe ref readonly void * PrimitiveColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_primitiveColors", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_Get_primitiveColors(_Underlying *_this);
                return ref *__MR_SaveSettings_Get_primitiveColors(_UnderlyingPtr);
            }
        }

        /// optional per-vertex uv coordinate to save with the geometry
        public unsafe ref readonly void * UvMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_uvMap", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_Get_uvMap(_Underlying *_this);
                return ref *__MR_SaveSettings_Get_uvMap(_UnderlyingPtr);
            }
        }

        /// optional texture to save with the geometry
        public unsafe ref readonly void * Texture
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_texture", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_Get_texture(_Underlying *_this);
                return ref *__MR_SaveSettings_Get_texture(_UnderlyingPtr);
            }
        }

        /// the name of file without extension to save texture in some formats (e.g. .OBJ, .PLY)
        public unsafe MR.Std.Const_String MaterialName
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_materialName", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_SaveSettings_Get_materialName(_Underlying *_this);
                return new(__MR_SaveSettings_Get_materialName(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this transformation can optionally be applied to all vertices (points) of saved object
        public unsafe ref readonly MR.AffineXf3d * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3d **__MR_SaveSettings_Get_xf(_Underlying *_this);
                return ref *__MR_SaveSettings_Get_xf(_UnderlyingPtr);
            }
        }

        /// units of input coordinates and transformation, to be serialized if the format supports it
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_SaveSettings_Get_lengthUnit(_Underlying *_this);
                return new(__MR_SaveSettings_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the color of whole object
        public unsafe MR.Std.Const_Optional_MRColor SolidColor
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_solidColor", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_SaveSettings_Get_solidColor(_Underlying *_this);
                return new(__MR_SaveSettings_Get_solidColor(_UnderlyingPtr), is_owning: false);
            }
        }

        /// to report save progress and cancel saving if user desires
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_SaveSettings_Get_progress(_Underlying *_this);
                return new(__MR_SaveSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SaveSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SaveSettings_DefaultConstruct();
        }

        /// Constructs `MR::SaveSettings` elementwise.
        public unsafe Const_SaveSettings(bool onlyValidPoints, bool packPrimitives, MR.Const_VertColors? colors, MR.Std.Const_Vector_MRColor? primitiveColors, MR.Const_VertCoords2? uvMap, MR.Const_MeshTexture? texture, ReadOnlySpan<char> materialName, MR.Const_AffineXf3d? xf, MR.LengthUnit? lengthUnit, MR._InOpt_Color solidColor, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_ConstructFrom(byte onlyValidPoints, byte packPrimitives, MR.Const_VertColors._Underlying *colors, MR.Std.Const_Vector_MRColor._Underlying *primitiveColors, MR.Const_VertCoords2._Underlying *uvMap, MR.Const_MeshTexture._Underlying *texture, byte *materialName, byte *materialName_end, MR.Const_AffineXf3d._Underlying *xf, MR.LengthUnit *lengthUnit, MR.Color *solidColor, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            byte[] __bytes_materialName = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(materialName.Length)];
            int __len_materialName = System.Text.Encoding.UTF8.GetBytes(materialName, __bytes_materialName);
            fixed (byte *__ptr_materialName = __bytes_materialName)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_SaveSettings_ConstructFrom(onlyValidPoints ? (byte)1 : (byte)0, packPrimitives ? (byte)1 : (byte)0, colors is not null ? colors._UnderlyingPtr : null, primitiveColors is not null ? primitiveColors._UnderlyingPtr : null, uvMap is not null ? uvMap._UnderlyingPtr : null, texture is not null ? texture._UnderlyingPtr : null, __ptr_materialName, __ptr_materialName + __len_materialName, xf is not null ? xf._UnderlyingPtr : null, lengthUnit.HasValue ? &__deref_lengthUnit : null, solidColor.HasValue ? &solidColor.Object : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from constructor `MR::SaveSettings::SaveSettings`.
        public unsafe Const_SaveSettings(MR._ByValue_SaveSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SaveSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SaveSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// determines how to save points/lines/mesh
    /// Generated from class `MR::SaveSettings`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshSave::CtmSaveOptions`
    ///     `MR::PointsSave::CtmSavePointsOptions`
    /// This is the non-const half of the class.
    public class SaveSettings : Const_SaveSettings
    {
        internal unsafe SaveSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// true - save valid points/vertices only (pack them);
        /// false - save all points/vertices preserving their indices
        public new unsafe ref bool OnlyValidPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_onlyValidPoints", ExactSpelling = true)]
                extern static bool *__MR_SaveSettings_GetMutable_onlyValidPoints(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_onlyValidPoints(_UnderlyingPtr);
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
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_packPrimitives", ExactSpelling = true)]
                extern static bool *__MR_SaveSettings_GetMutable_packPrimitives(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_packPrimitives(_UnderlyingPtr);
            }
        }

        /// optional per-vertex color to save with the geometry
        public new unsafe ref readonly void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_colors", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_GetMutable_colors(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_colors(_UnderlyingPtr);
            }
        }

        /// per-face colors for meshes, per-undirected-edge colors for polylines, unused for point clouds and other
        public new unsafe ref readonly void * PrimitiveColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_primitiveColors", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_GetMutable_primitiveColors(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_primitiveColors(_UnderlyingPtr);
            }
        }

        /// optional per-vertex uv coordinate to save with the geometry
        public new unsafe ref readonly void * UvMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_uvMap", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_GetMutable_uvMap(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_uvMap(_UnderlyingPtr);
            }
        }

        /// optional texture to save with the geometry
        public new unsafe ref readonly void * Texture
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_texture", ExactSpelling = true)]
                extern static void **__MR_SaveSettings_GetMutable_texture(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_texture(_UnderlyingPtr);
            }
        }

        /// the name of file without extension to save texture in some formats (e.g. .OBJ, .PLY)
        public new unsafe MR.Std.String MaterialName
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_materialName", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_SaveSettings_GetMutable_materialName(_Underlying *_this);
                return new(__MR_SaveSettings_GetMutable_materialName(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this transformation can optionally be applied to all vertices (points) of saved object
        public new unsafe ref readonly MR.AffineXf3d * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3d **__MR_SaveSettings_GetMutable_xf(_Underlying *_this);
                return ref *__MR_SaveSettings_GetMutable_xf(_UnderlyingPtr);
            }
        }

        /// units of input coordinates and transformation, to be serialized if the format supports it
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_SaveSettings_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_SaveSettings_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the color of whole object
        public new unsafe MR.Std.Optional_MRColor SolidColor
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_solidColor", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_SaveSettings_GetMutable_solidColor(_Underlying *_this);
                return new(__MR_SaveSettings_GetMutable_solidColor(_UnderlyingPtr), is_owning: false);
            }
        }

        /// to report save progress and cancel saving if user desires
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_SaveSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_SaveSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SaveSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SaveSettings_DefaultConstruct();
        }

        /// Constructs `MR::SaveSettings` elementwise.
        public unsafe SaveSettings(bool onlyValidPoints, bool packPrimitives, MR.Const_VertColors? colors, MR.Std.Const_Vector_MRColor? primitiveColors, MR.Const_VertCoords2? uvMap, MR.Const_MeshTexture? texture, ReadOnlySpan<char> materialName, MR.Const_AffineXf3d? xf, MR.LengthUnit? lengthUnit, MR._InOpt_Color solidColor, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_ConstructFrom(byte onlyValidPoints, byte packPrimitives, MR.Const_VertColors._Underlying *colors, MR.Std.Const_Vector_MRColor._Underlying *primitiveColors, MR.Const_VertCoords2._Underlying *uvMap, MR.Const_MeshTexture._Underlying *texture, byte *materialName, byte *materialName_end, MR.Const_AffineXf3d._Underlying *xf, MR.LengthUnit *lengthUnit, MR.Color *solidColor, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            byte[] __bytes_materialName = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(materialName.Length)];
            int __len_materialName = System.Text.Encoding.UTF8.GetBytes(materialName, __bytes_materialName);
            fixed (byte *__ptr_materialName = __bytes_materialName)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_SaveSettings_ConstructFrom(onlyValidPoints ? (byte)1 : (byte)0, packPrimitives ? (byte)1 : (byte)0, colors is not null ? colors._UnderlyingPtr : null, primitiveColors is not null ? primitiveColors._UnderlyingPtr : null, uvMap is not null ? uvMap._UnderlyingPtr : null, texture is not null ? texture._UnderlyingPtr : null, __ptr_materialName, __ptr_materialName + __len_materialName, xf is not null ? xf._UnderlyingPtr : null, lengthUnit.HasValue ? &__deref_lengthUnit : null, solidColor.HasValue ? &solidColor.Object : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from constructor `MR::SaveSettings::SaveSettings`.
        public unsafe SaveSettings(MR._ByValue_SaveSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SaveSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SaveSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SaveSettings::operator=`.
        public unsafe MR.SaveSettings Assign(MR._ByValue_SaveSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SaveSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SaveSettings._Underlying *__MR_SaveSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SaveSettings._Underlying *_other);
            return new(__MR_SaveSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SaveSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SaveSettings`/`Const_SaveSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SaveSettings
    {
        internal readonly Const_SaveSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SaveSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SaveSettings(Const_SaveSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SaveSettings(Const_SaveSettings arg) {return new(arg);}
        public _ByValue_SaveSettings(MR.Misc._Moved<SaveSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SaveSettings(MR.Misc._Moved<SaveSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SaveSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SaveSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SaveSettings`/`Const_SaveSettings` directly.
    public class _InOptMut_SaveSettings
    {
        public SaveSettings? Opt;

        public _InOptMut_SaveSettings() {}
        public _InOptMut_SaveSettings(SaveSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SaveSettings(SaveSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SaveSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SaveSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SaveSettings`/`Const_SaveSettings` to pass it to the function.
    public class _InOptConst_SaveSettings
    {
        public Const_SaveSettings? Opt;

        public _InOptConst_SaveSettings() {}
        public _InOptConst_SaveSettings(Const_SaveSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SaveSettings(Const_SaveSettings value) {return new(value);}
    }

    /// maps valid points to packed sequential indices
    /// Generated from class `MR::VertRenumber`.
    /// This is the const half of the class.
    public class Const_VertRenumber : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertRenumber(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_Destroy", ExactSpelling = true)]
            extern static void __MR_VertRenumber_Destroy(_Underlying *_this);
            __MR_VertRenumber_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertRenumber() {Dispose(false);}

        /// Generated from constructor `MR::VertRenumber::VertRenumber`.
        public unsafe Const_VertRenumber(MR._ByValue_VertRenumber _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertRenumber._Underlying *__MR_VertRenumber_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertRenumber._Underlying *_other);
            _UnderlyingPtr = __MR_VertRenumber_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// prepares the mapping
        /// Generated from constructor `MR::VertRenumber::VertRenumber`.
        public unsafe Const_VertRenumber(MR.Const_VertBitSet validVerts, bool saveValidOnly) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_Construct", ExactSpelling = true)]
            extern static MR.VertRenumber._Underlying *__MR_VertRenumber_Construct(MR.Const_VertBitSet._Underlying *validVerts, byte saveValidOnly);
            _UnderlyingPtr = __MR_VertRenumber_Construct(validVerts._UnderlyingPtr, saveValidOnly ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::VertRenumber::saveValidOnly`.
        public unsafe bool SaveValidOnly()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_saveValidOnly", ExactSpelling = true)]
            extern static byte __MR_VertRenumber_saveValidOnly(_Underlying *_this);
            return __MR_VertRenumber_saveValidOnly(_UnderlyingPtr) != 0;
        }

        /// return the total number of vertices to be saved
        /// Generated from method `MR::VertRenumber::sizeVerts`.
        public unsafe int SizeVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_sizeVerts", ExactSpelling = true)]
            extern static int __MR_VertRenumber_sizeVerts(_Underlying *_this);
            return __MR_VertRenumber_sizeVerts(_UnderlyingPtr);
        }

        /// return packed index (if saveValidOnly = true) or same index (if saveValidOnly = false)
        /// Generated from method `MR::VertRenumber::operator()`.
        public unsafe int Call(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_call", ExactSpelling = true)]
            extern static int __MR_VertRenumber_call(_Underlying *_this, MR.VertId v);
            return __MR_VertRenumber_call(_UnderlyingPtr, v);
        }
    }

    /// maps valid points to packed sequential indices
    /// Generated from class `MR::VertRenumber`.
    /// This is the non-const half of the class.
    public class VertRenumber : Const_VertRenumber
    {
        internal unsafe VertRenumber(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::VertRenumber::VertRenumber`.
        public unsafe VertRenumber(MR._ByValue_VertRenumber _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertRenumber._Underlying *__MR_VertRenumber_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertRenumber._Underlying *_other);
            _UnderlyingPtr = __MR_VertRenumber_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// prepares the mapping
        /// Generated from constructor `MR::VertRenumber::VertRenumber`.
        public unsafe VertRenumber(MR.Const_VertBitSet validVerts, bool saveValidOnly) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_Construct", ExactSpelling = true)]
            extern static MR.VertRenumber._Underlying *__MR_VertRenumber_Construct(MR.Const_VertBitSet._Underlying *validVerts, byte saveValidOnly);
            _UnderlyingPtr = __MR_VertRenumber_Construct(validVerts._UnderlyingPtr, saveValidOnly ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::VertRenumber::operator=`.
        public unsafe MR.VertRenumber Assign(MR._ByValue_VertRenumber _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertRenumber_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertRenumber._Underlying *__MR_VertRenumber_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VertRenumber._Underlying *_other);
            return new(__MR_VertRenumber_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VertRenumber` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VertRenumber`/`Const_VertRenumber` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VertRenumber
    {
        internal readonly Const_VertRenumber? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VertRenumber(Const_VertRenumber new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VertRenumber(Const_VertRenumber arg) {return new(arg);}
        public _ByValue_VertRenumber(MR.Misc._Moved<VertRenumber> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VertRenumber(MR.Misc._Moved<VertRenumber> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VertRenumber` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertRenumber`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertRenumber`/`Const_VertRenumber` directly.
    public class _InOptMut_VertRenumber
    {
        public VertRenumber? Opt;

        public _InOptMut_VertRenumber() {}
        public _InOptMut_VertRenumber(VertRenumber value) {Opt = value;}
        public static implicit operator _InOptMut_VertRenumber(VertRenumber value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertRenumber` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertRenumber`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertRenumber`/`Const_VertRenumber` to pass it to the function.
    public class _InOptConst_VertRenumber
    {
        public Const_VertRenumber? Opt;

        public _InOptConst_VertRenumber() {}
        public _InOptConst_VertRenumber(Const_VertRenumber value) {Opt = value;}
        public static implicit operator _InOptConst_VertRenumber(Const_VertRenumber value) {return new(value);}
    }

    /// returns the point as is or after application of given transform to it in double precision
    /// Generated from function `MR::applyFloat`.
    public static unsafe MR.Vector3f ApplyFloat(MR.Const_AffineXf3d? xf, MR.Const_Vector3f p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_applyFloat_const_MR_AffineXf3d_ptr", ExactSpelling = true)]
        extern static MR.Vector3f __MR_applyFloat_const_MR_AffineXf3d_ptr(MR.Const_AffineXf3d._Underlying *xf, MR.Const_Vector3f._Underlying *p);
        return __MR_applyFloat_const_MR_AffineXf3d_ptr(xf is not null ? xf._UnderlyingPtr : null, p._UnderlyingPtr);
    }

    /// returns the normal as is or after application of given matrix to it in double precision
    /// Generated from function `MR::applyFloat`.
    public static unsafe MR.Vector3f ApplyFloat(MR.Const_Matrix3d? m, MR.Const_Vector3f n)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_applyFloat_const_MR_Matrix3d_ptr", ExactSpelling = true)]
        extern static MR.Vector3f __MR_applyFloat_const_MR_Matrix3d_ptr(MR.Const_Matrix3d._Underlying *m, MR.Const_Vector3f._Underlying *n);
        return __MR_applyFloat_const_MR_Matrix3d_ptr(m is not null ? m._UnderlyingPtr : null, n._UnderlyingPtr);
    }

    /// converts given point in double precision and applies given transformation to it
    /// Generated from function `MR::applyDouble`.
    public static unsafe MR.Vector3d ApplyDouble(MR.Const_AffineXf3d? xf, MR.Const_Vector3f p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_applyDouble_const_MR_AffineXf3d_ptr", ExactSpelling = true)]
        extern static MR.Vector3d __MR_applyDouble_const_MR_AffineXf3d_ptr(MR.Const_AffineXf3d._Underlying *xf, MR.Const_Vector3f._Underlying *p);
        return __MR_applyDouble_const_MR_AffineXf3d_ptr(xf is not null ? xf._UnderlyingPtr : null, p._UnderlyingPtr);
    }

    /// converts given normal in double precision and applies given matrix to it
    /// Generated from function `MR::applyDouble`.
    public static unsafe MR.Vector3d ApplyDouble(MR.Const_Matrix3d? m, MR.Const_Vector3f n)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_applyDouble_const_MR_Matrix3d_ptr", ExactSpelling = true)]
        extern static MR.Vector3d __MR_applyDouble_const_MR_Matrix3d_ptr(MR.Const_Matrix3d._Underlying *m, MR.Const_Vector3f._Underlying *n);
        return __MR_applyDouble_const_MR_Matrix3d_ptr(m is not null ? m._UnderlyingPtr : null, n._UnderlyingPtr);
    }

    /// if (xf) is null then just returns (verts);
    /// otherwise copies transformed points in (buf) and returns it
    /// Generated from function `MR::transformPoints`.
    public static unsafe MR.Const_VertCoords TransformPoints(MR.Const_VertCoords verts, MR.Const_VertBitSet validVerts, MR.Const_AffineXf3d? xf, MR.VertCoords buf, MR.Const_VertRenumber? vertRenumber = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_transformPoints", ExactSpelling = true)]
        extern static MR.Const_VertCoords._Underlying *__MR_transformPoints(MR.Const_VertCoords._Underlying *verts, MR.Const_VertBitSet._Underlying *validVerts, MR.Const_AffineXf3d._Underlying *xf, MR.VertCoords._Underlying *buf, MR.Const_VertRenumber._Underlying *vertRenumber);
        return new(__MR_transformPoints(verts._UnderlyingPtr, validVerts._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null, buf._UnderlyingPtr, vertRenumber is not null ? vertRenumber._UnderlyingPtr : null), is_owning: false);
    }

    /// if (m) is null then just returns (normals);
    /// otherwise copies transformed normals in (buf) and returns it
    /// Generated from function `MR::transformNormals`.
    public static unsafe MR.Const_VertCoords TransformNormals(MR.Const_VertCoords normals, MR.Const_VertBitSet validVerts, MR.Const_Matrix3d? m, MR.VertCoords buf)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_transformNormals", ExactSpelling = true)]
        extern static MR.Const_VertCoords._Underlying *__MR_transformNormals(MR.Const_VertCoords._Underlying *normals, MR.Const_VertBitSet._Underlying *validVerts, MR.Const_Matrix3d._Underlying *m, MR.VertCoords._Underlying *buf);
        return new(__MR_transformNormals(normals._UnderlyingPtr, validVerts._UnderlyingPtr, m is not null ? m._UnderlyingPtr : null, buf._UnderlyingPtr), is_owning: false);
    }
}
