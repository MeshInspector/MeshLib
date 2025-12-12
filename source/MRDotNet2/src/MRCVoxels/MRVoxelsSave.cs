public static partial class MR
{
    public static partial class VoxelsSave
    {
        // stores together all data for save voxel object as a group of images
        /// Generated from class `MR::VoxelsSave::SavingSettings`.
        /// This is the const half of the class.
        public class Const_SavingSettings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_SavingSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_Destroy", ExactSpelling = true)]
                extern static void __MR_VoxelsSave_SavingSettings_Destroy(_Underlying *_this);
                __MR_VoxelsSave_SavingSettings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_SavingSettings() {Dispose(false);}

            // path to directory where you want to save images
            public unsafe MR.Std.Filesystem.Const_Path Path
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_Get_path", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_VoxelsSave_SavingSettings_Get_path(_Underlying *_this);
                    return new(__MR_VoxelsSave_SavingSettings_Get_path(_UnderlyingPtr), is_owning: false);
                }
            }

            // format for file names, you should specify a placeholder for number and extension, eg "slice_{0:0{1}}.tif"
            public unsafe MR.Std.Const_String Format
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_Get_format", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_VoxelsSave_SavingSettings_Get_format(_Underlying *_this);
                    return new(__MR_VoxelsSave_SavingSettings_Get_format(_UnderlyingPtr), is_owning: false);
                }
            }

            // Plane which the object is sliced by. XY, XZ, or YZ
            public unsafe MR.SlicePlane SlicePlane
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_Get_slicePlane", ExactSpelling = true)]
                    extern static MR.SlicePlane *__MR_VoxelsSave_SavingSettings_Get_slicePlane(_Underlying *_this);
                    return *__MR_VoxelsSave_SavingSettings_Get_slicePlane(_UnderlyingPtr);
                }
            }

            // Callback reporting progress
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_Get_cb", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_VoxelsSave_SavingSettings_Get_cb(_Underlying *_this);
                    return new(__MR_VoxelsSave_SavingSettings_Get_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_SavingSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsSave_SavingSettings_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsSave::SavingSettings` elementwise.
            public unsafe Const_SavingSettings(ReadOnlySpan<char> path, ReadOnlySpan<char> format, MR.SlicePlane slicePlane, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_ConstructFrom(byte *path, byte *path_end, byte *format, byte *format_end, MR.SlicePlane slicePlane, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
                int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
                fixed (byte *__ptr_path = __bytes_path)
                {
                    byte[] __bytes_format = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(format.Length)];
                    int __len_format = System.Text.Encoding.UTF8.GetBytes(format, __bytes_format);
                    fixed (byte *__ptr_format = __bytes_format)
                    {
                        _UnderlyingPtr = __MR_VoxelsSave_SavingSettings_ConstructFrom(__ptr_path, __ptr_path + __len_path, __ptr_format, __ptr_format + __len_format, slicePlane, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
                    }
                }
            }

            /// Generated from constructor `MR::VoxelsSave::SavingSettings::SavingSettings`.
            public unsafe Const_SavingSettings(MR.VoxelsSave._ByValue_SavingSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsSave.SavingSettings._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsSave_SavingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        // stores together all data for save voxel object as a group of images
        /// Generated from class `MR::VoxelsSave::SavingSettings`.
        /// This is the non-const half of the class.
        public class SavingSettings : Const_SavingSettings
        {
            internal unsafe SavingSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // path to directory where you want to save images
            public new unsafe MR.Std.Filesystem.Path Path
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_GetMutable_path", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_VoxelsSave_SavingSettings_GetMutable_path(_Underlying *_this);
                    return new(__MR_VoxelsSave_SavingSettings_GetMutable_path(_UnderlyingPtr), is_owning: false);
                }
            }

            // format for file names, you should specify a placeholder for number and extension, eg "slice_{0:0{1}}.tif"
            public new unsafe MR.Std.String Format
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_GetMutable_format", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_VoxelsSave_SavingSettings_GetMutable_format(_Underlying *_this);
                    return new(__MR_VoxelsSave_SavingSettings_GetMutable_format(_UnderlyingPtr), is_owning: false);
                }
            }

            // Plane which the object is sliced by. XY, XZ, or YZ
            public new unsafe ref MR.SlicePlane SlicePlane
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_GetMutable_slicePlane", ExactSpelling = true)]
                    extern static MR.SlicePlane *__MR_VoxelsSave_SavingSettings_GetMutable_slicePlane(_Underlying *_this);
                    return ref *__MR_VoxelsSave_SavingSettings_GetMutable_slicePlane(_UnderlyingPtr);
                }
            }

            // Callback reporting progress
            public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_GetMutable_cb", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_VoxelsSave_SavingSettings_GetMutable_cb(_Underlying *_this);
                    return new(__MR_VoxelsSave_SavingSettings_GetMutable_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe SavingSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsSave_SavingSettings_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsSave::SavingSettings` elementwise.
            public unsafe SavingSettings(ReadOnlySpan<char> path, ReadOnlySpan<char> format, MR.SlicePlane slicePlane, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_ConstructFrom(byte *path, byte *path_end, byte *format, byte *format_end, MR.SlicePlane slicePlane, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
                int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
                fixed (byte *__ptr_path = __bytes_path)
                {
                    byte[] __bytes_format = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(format.Length)];
                    int __len_format = System.Text.Encoding.UTF8.GetBytes(format, __bytes_format);
                    fixed (byte *__ptr_format = __bytes_format)
                    {
                        _UnderlyingPtr = __MR_VoxelsSave_SavingSettings_ConstructFrom(__ptr_path, __ptr_path + __len_path, __ptr_format, __ptr_format + __len_format, slicePlane, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
                    }
                }
            }

            /// Generated from constructor `MR::VoxelsSave::SavingSettings::SavingSettings`.
            public unsafe SavingSettings(MR.VoxelsSave._ByValue_SavingSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsSave.SavingSettings._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsSave_SavingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::VoxelsSave::SavingSettings::operator=`.
            public unsafe MR.VoxelsSave.SavingSettings Assign(MR.VoxelsSave._ByValue_SavingSettings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_SavingSettings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsSave.SavingSettings._Underlying *__MR_VoxelsSave_SavingSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsSave.SavingSettings._Underlying *_other);
                return new(__MR_VoxelsSave_SavingSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `SavingSettings` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `SavingSettings`/`Const_SavingSettings` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_SavingSettings
        {
            internal readonly Const_SavingSettings? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_SavingSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_SavingSettings(Const_SavingSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_SavingSettings(Const_SavingSettings arg) {return new(arg);}
            public _ByValue_SavingSettings(MR.Misc._Moved<SavingSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_SavingSettings(MR.Misc._Moved<SavingSettings> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `SavingSettings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_SavingSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SavingSettings`/`Const_SavingSettings` directly.
        public class _InOptMut_SavingSettings
        {
            public SavingSettings? Opt;

            public _InOptMut_SavingSettings() {}
            public _InOptMut_SavingSettings(SavingSettings value) {Opt = value;}
            public static implicit operator _InOptMut_SavingSettings(SavingSettings value) {return new(value);}
        }

        /// This is used for optional parameters of class `SavingSettings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_SavingSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `SavingSettings`/`Const_SavingSettings` to pass it to the function.
        public class _InOptConst_SavingSettings
        {
            public Const_SavingSettings? Opt;

            public _InOptConst_SavingSettings() {}
            public _InOptConst_SavingSettings(Const_SavingSettings value) {Opt = value;}
            public static implicit operator _InOptConst_SavingSettings(Const_SavingSettings value) {return new(value);}
        }

        /// Save raw voxels file, writing parameters in file name
        /// Generated from function `MR::VoxelsSave::toRawAutoname`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToRawAutoname(MR.Const_VdbVolume vdbVolume, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toRawAutoname_MR_VdbVolume", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toRawAutoname_MR_VdbVolume(MR.Const_VdbVolume._Underlying *vdbVolume, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toRawAutoname_MR_VdbVolume(vdbVolume._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::toRawAutoname`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToRawAutoname(MR.Const_SimpleVolume simpleVolume, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toRawAutoname_MR_SimpleVolume", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toRawAutoname_MR_SimpleVolume(MR.Const_SimpleVolume._Underlying *simpleVolume, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toRawAutoname_MR_SimpleVolume(simpleVolume._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::gridToRawAutoname`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> GridToRawAutoname(MR.Const_FloatGrid grid, MR.Const_Vector3i dims, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_gridToRawAutoname", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_gridToRawAutoname(MR.Const_FloatGrid._Underlying *grid, MR.Const_Vector3i._Underlying *dims, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_gridToRawAutoname(grid._UnderlyingPtr, dims._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Save voxels in raw format with each value as 32-bit float in given binary stream
        /// Generated from function `MR::VoxelsSave::gridToRawFloat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> GridToRawFloat(MR.Const_FloatGrid grid, MR.Const_Vector3i dims, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_gridToRawFloat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_gridToRawFloat(MR.Const_FloatGrid._Underlying *grid, MR.Const_Vector3i._Underlying *dims, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_gridToRawFloat(grid._UnderlyingPtr, dims._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::VoxelsSave::toRawFloat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToRawFloat(MR.Const_VdbVolume vdbVolume, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toRawFloat_MR_VdbVolume", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toRawFloat_MR_VdbVolume(MR.Const_VdbVolume._Underlying *vdbVolume, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toRawFloat_MR_VdbVolume(vdbVolume._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::VoxelsSave::toRawFloat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToRawFloat(MR.Const_SimpleVolume simpleVolume, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toRawFloat_MR_SimpleVolume", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toRawFloat_MR_SimpleVolume(MR.Const_SimpleVolume._Underlying *simpleVolume, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toRawFloat_MR_SimpleVolume(simpleVolume._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Save voxels in Gav-format in given destination
        /// Generated from function `MR::VoxelsSave::toGav`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToGav(MR.Const_VdbVolume vdbVolume, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toGav_const_MR_VdbVolume_ref_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toGav_const_MR_VdbVolume_ref_std_filesystem_path(MR.Const_VdbVolume._Underlying *vdbVolume, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toGav_const_MR_VdbVolume_ref_std_filesystem_path(vdbVolume._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::toGav`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToGav(MR.Const_VdbVolume vdbVolume, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toGav_const_MR_VdbVolume_ref_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toGav_const_MR_VdbVolume_ref_std_ostream(MR.Const_VdbVolume._Underlying *vdbVolume, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toGav_const_MR_VdbVolume_ref_std_ostream(vdbVolume._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::VoxelsSave::toGav`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToGav(MR.Const_SimpleVolumeMinMax simpleVolumeMinMax, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toGav_const_MR_SimpleVolumeMinMax_ref_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toGav_const_MR_SimpleVolumeMinMax_ref_std_filesystem_path(MR.Const_SimpleVolumeMinMax._Underlying *simpleVolumeMinMax, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toGav_const_MR_SimpleVolumeMinMax_ref_std_filesystem_path(simpleVolumeMinMax._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::toGav`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToGav(MR.Const_SimpleVolumeMinMax simpleVolumeMinMax, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toGav_const_MR_SimpleVolumeMinMax_ref_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toGav_const_MR_SimpleVolumeMinMax_ref_std_ostream(MR.Const_SimpleVolumeMinMax._Underlying *simpleVolumeMinMax, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toGav_const_MR_SimpleVolumeMinMax_ref_std_ostream(simpleVolumeMinMax._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::VoxelsSave::toGav`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToGav(MR.Const_SimpleVolume simpleVolume, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toGav_const_MR_SimpleVolume_ref_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toGav_const_MR_SimpleVolume_ref_std_filesystem_path(MR.Const_SimpleVolume._Underlying *simpleVolume, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toGav_const_MR_SimpleVolume_ref_std_filesystem_path(simpleVolume._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::toGav`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToGav(MR.Const_SimpleVolume simpleVolume, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toGav_const_MR_SimpleVolume_ref_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toGav_const_MR_SimpleVolume_ref_std_ostream(MR.Const_SimpleVolume._Underlying *simpleVolume, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toGav_const_MR_SimpleVolume_ref_std_ostream(simpleVolume._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Save voxels file in OpenVDB format
        /// Generated from function `MR::VoxelsSave::gridToVdb`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> GridToVdb(MR.Const_FloatGrid grid, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_gridToVdb_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_gridToVdb_std_filesystem_path(MR.Const_FloatGrid._Underlying *grid, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_gridToVdb_std_filesystem_path(grid._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::gridToVdb`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> GridToVdb(MR.Const_FloatGrid grid, MR.Std.Ostream out_, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_gridToVdb_std_ostream", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_gridToVdb_std_ostream(MR.Const_FloatGrid._Underlying *grid, MR.Std.Ostream._Underlying *out_, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_gridToVdb_std_ostream(grid._UnderlyingPtr, out_._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::VoxelsSave::toVdb`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToVdb(MR.Const_VdbVolume vdbVolume, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toVdb", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toVdb(MR.Const_VdbVolume._Underlying *vdbVolume, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toVdb(vdbVolume._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Saves voxels in a file, detecting the format from file extension
        /// Generated from function `MR::VoxelsSave::gridToAnySupportedFormat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> GridToAnySupportedFormat(MR.Const_FloatGrid grid, MR.Const_Vector3i dims, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_gridToAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_gridToAnySupportedFormat(MR.Const_FloatGrid._Underlying *grid, MR.Const_Vector3i._Underlying *dims, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_gridToAnySupportedFormat(grid._UnderlyingPtr, dims._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsSave::toAnySupportedFormat`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToAnySupportedFormat(MR.Const_VdbVolume vdbVolume, ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toAnySupportedFormat(MR.Const_VdbVolume._Underlying *vdbVolume, byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toAnySupportedFormat(vdbVolume._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// save the slice by the active plane through the sliceNumber to an image file
        /// Generated from function `MR::VoxelsSave::saveSliceToImage`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SaveSliceToImage(ReadOnlySpan<char> path, MR.Const_VdbVolume vdbVolume, MR.SlicePlane slicePlain, int sliceNumber, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_saveSliceToImage", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_saveSliceToImage(byte *path, byte *path_end, MR.Const_VdbVolume._Underlying *vdbVolume, MR.SlicePlane *slicePlain, int sliceNumber, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_saveSliceToImage(__ptr_path, __ptr_path + __len_path, vdbVolume._UnderlyingPtr, &slicePlain, sliceNumber, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// save all slices by the active plane through all voxel planes along the active axis to an image file
        /// Generated from function `MR::VoxelsSave::saveAllSlicesToImage`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SaveAllSlicesToImage(MR.Const_VdbVolume vdbVolume, MR.VoxelsSave.Const_SavingSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_saveAllSlicesToImage", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_saveAllSlicesToImage(MR.Const_VdbVolume._Underlying *vdbVolume, MR.VoxelsSave.Const_SavingSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_saveAllSlicesToImage(vdbVolume._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
        }
    }

    /// Generated from function `MR::saveObjectVoxelsToFile`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SaveObjectVoxelsToFile(MR.Const_Object object_, ReadOnlySpan<char> path, MR.ObjectSave.Const_Settings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_saveObjectVoxelsToFile", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_saveObjectVoxelsToFile(MR.Const_Object._Underlying *object_, byte *path, byte *path_end, MR.ObjectSave.Const_Settings._Underlying *settings);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_saveObjectVoxelsToFile(object_._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, settings._UnderlyingPtr), is_owning: true));
        }
    }
}
