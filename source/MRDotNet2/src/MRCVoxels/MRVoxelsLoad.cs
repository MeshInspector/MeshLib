public static partial class MR
{
    public static partial class VoxelsLoad
    {
        /// Generated from class `MR::VoxelsLoad::RawParameters`.
        /// This is the const half of the class.
        public class Const_RawParameters : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_RawParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_Destroy", ExactSpelling = true)]
                extern static void __MR_VoxelsLoad_RawParameters_Destroy(_Underlying *_this);
                __MR_VoxelsLoad_RawParameters_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_RawParameters() {Dispose(false);}

            public unsafe MR.Const_Vector3i Dimensions
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_Get_dimensions", ExactSpelling = true)]
                    extern static MR.Const_Vector3i._Underlying *__MR_VoxelsLoad_RawParameters_Get_dimensions(_Underlying *_this);
                    return new(__MR_VoxelsLoad_RawParameters_Get_dimensions(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Vector3f VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_Get_voxelSize", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_VoxelsLoad_RawParameters_Get_voxelSize(_Underlying *_this);
                    return new(__MR_VoxelsLoad_RawParameters_Get_voxelSize(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< OpenVDB GridClass set as GRID_LEVEL_SET (need to set right surface normals direction)
            public unsafe bool GridLevelSet
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_Get_gridLevelSet", ExactSpelling = true)]
                    extern static bool *__MR_VoxelsLoad_RawParameters_Get_gridLevelSet(_Underlying *_this);
                    return *__MR_VoxelsLoad_RawParameters_Get_gridLevelSet(_UnderlyingPtr);
                }
            }

            public unsafe MR.ScalarType ScalarType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_Get_scalarType", ExactSpelling = true)]
                    extern static MR.ScalarType *__MR_VoxelsLoad_RawParameters_Get_scalarType(_Underlying *_this);
                    return *__MR_VoxelsLoad_RawParameters_Get_scalarType(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_RawParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_RawParameters_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::RawParameters` elementwise.
            public unsafe Const_RawParameters(MR.Vector3i dimensions, MR.Vector3f voxelSize, bool gridLevelSet, MR.ScalarType scalarType) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_ConstructFrom(MR.Vector3i dimensions, MR.Vector3f voxelSize, byte gridLevelSet, MR.ScalarType scalarType);
                _UnderlyingPtr = __MR_VoxelsLoad_RawParameters_ConstructFrom(dimensions, voxelSize, gridLevelSet ? (byte)1 : (byte)0, scalarType);
            }

            /// Generated from constructor `MR::VoxelsLoad::RawParameters::RawParameters`.
            public unsafe Const_RawParameters(MR.VoxelsLoad.Const_RawParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_ConstructFromAnother(MR.VoxelsLoad.RawParameters._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_RawParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::VoxelsLoad::RawParameters`.
        /// This is the non-const half of the class.
        public class RawParameters : Const_RawParameters
        {
            internal unsafe RawParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Vector3i Dimensions
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_GetMutable_dimensions", ExactSpelling = true)]
                    extern static MR.Mut_Vector3i._Underlying *__MR_VoxelsLoad_RawParameters_GetMutable_dimensions(_Underlying *_this);
                    return new(__MR_VoxelsLoad_RawParameters_GetMutable_dimensions(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Vector3f VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_GetMutable_voxelSize", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_VoxelsLoad_RawParameters_GetMutable_voxelSize(_Underlying *_this);
                    return new(__MR_VoxelsLoad_RawParameters_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< OpenVDB GridClass set as GRID_LEVEL_SET (need to set right surface normals direction)
            public new unsafe ref bool GridLevelSet
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_GetMutable_gridLevelSet", ExactSpelling = true)]
                    extern static bool *__MR_VoxelsLoad_RawParameters_GetMutable_gridLevelSet(_Underlying *_this);
                    return ref *__MR_VoxelsLoad_RawParameters_GetMutable_gridLevelSet(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.ScalarType ScalarType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_GetMutable_scalarType", ExactSpelling = true)]
                    extern static MR.ScalarType *__MR_VoxelsLoad_RawParameters_GetMutable_scalarType(_Underlying *_this);
                    return ref *__MR_VoxelsLoad_RawParameters_GetMutable_scalarType(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe RawParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_RawParameters_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::RawParameters` elementwise.
            public unsafe RawParameters(MR.Vector3i dimensions, MR.Vector3f voxelSize, bool gridLevelSet, MR.ScalarType scalarType) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_ConstructFrom(MR.Vector3i dimensions, MR.Vector3f voxelSize, byte gridLevelSet, MR.ScalarType scalarType);
                _UnderlyingPtr = __MR_VoxelsLoad_RawParameters_ConstructFrom(dimensions, voxelSize, gridLevelSet ? (byte)1 : (byte)0, scalarType);
            }

            /// Generated from constructor `MR::VoxelsLoad::RawParameters::RawParameters`.
            public unsafe RawParameters(MR.VoxelsLoad.Const_RawParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_ConstructFromAnother(MR.VoxelsLoad.RawParameters._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_RawParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::VoxelsLoad::RawParameters::operator=`.
            public unsafe MR.VoxelsLoad.RawParameters Assign(MR.VoxelsLoad.Const_RawParameters _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_RawParameters_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.RawParameters._Underlying *__MR_VoxelsLoad_RawParameters_AssignFromAnother(_Underlying *_this, MR.VoxelsLoad.RawParameters._Underlying *_other);
                return new(__MR_VoxelsLoad_RawParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `RawParameters` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_RawParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `RawParameters`/`Const_RawParameters` directly.
        public class _InOptMut_RawParameters
        {
            public RawParameters? Opt;

            public _InOptMut_RawParameters() {}
            public _InOptMut_RawParameters(RawParameters value) {Opt = value;}
            public static implicit operator _InOptMut_RawParameters(RawParameters value) {return new(value);}
        }

        /// This is used for optional parameters of class `RawParameters` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_RawParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `RawParameters`/`Const_RawParameters` to pass it to the function.
        public class _InOptConst_RawParameters
        {
            public Const_RawParameters? Opt;

            public _InOptConst_RawParameters() {}
            public _InOptConst_RawParameters(Const_RawParameters value) {Opt = value;}
            public static implicit operator _InOptConst_RawParameters(Const_RawParameters value) {return new(value);}
        }

        // Determines iso-surface orientation
        public enum GridType : int
        {
            // consider values less than iso as outer area
            DenseGrid = 0,
            // consider values less than iso as inner area
            LevelSet = 1,
        }

        /// Generated from class `MR::VoxelsLoad::LoadingTiffSettings`.
        /// This is the const half of the class.
        public class Const_LoadingTiffSettings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_LoadingTiffSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_Destroy", ExactSpelling = true)]
                extern static void __MR_VoxelsLoad_LoadingTiffSettings_Destroy(_Underlying *_this);
                __MR_VoxelsLoad_LoadingTiffSettings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_LoadingTiffSettings() {Dispose(false);}

            public unsafe MR.Std.Filesystem.Const_Path Dir
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_Get_dir", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_Get_dir(_Underlying *_this);
                    return new(__MR_VoxelsLoad_LoadingTiffSettings_Get_dir(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Vector3f VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_Get_voxelSize", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_Get_voxelSize(_Underlying *_this);
                    return new(__MR_VoxelsLoad_LoadingTiffSettings_Get_voxelSize(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.VoxelsLoad.GridType GridType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_Get_gridType", ExactSpelling = true)]
                    extern static MR.VoxelsLoad.GridType *__MR_VoxelsLoad_LoadingTiffSettings_Get_gridType(_Underlying *_this);
                    return *__MR_VoxelsLoad_LoadingTiffSettings_Get_gridType(_UnderlyingPtr);
                }
            }

            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_Get_cb", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_Get_cb(_Underlying *_this);
                    return new(__MR_VoxelsLoad_LoadingTiffSettings_Get_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_LoadingTiffSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_LoadingTiffSettings_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::LoadingTiffSettings` elementwise.
            public unsafe Const_LoadingTiffSettings(ReadOnlySpan<char> dir, MR.Vector3f voxelSize, MR.VoxelsLoad.GridType gridType, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_ConstructFrom(byte *dir, byte *dir_end, MR.Vector3f voxelSize, MR.VoxelsLoad.GridType gridType, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                byte[] __bytes_dir = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dir.Length)];
                int __len_dir = System.Text.Encoding.UTF8.GetBytes(dir, __bytes_dir);
                fixed (byte *__ptr_dir = __bytes_dir)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_LoadingTiffSettings_ConstructFrom(__ptr_dir, __ptr_dir + __len_dir, voxelSize, gridType, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::LoadingTiffSettings::LoadingTiffSettings`.
            public unsafe Const_LoadingTiffSettings(MR.VoxelsLoad._ByValue_LoadingTiffSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.LoadingTiffSettings._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_LoadingTiffSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::VoxelsLoad::LoadingTiffSettings`.
        /// This is the non-const half of the class.
        public class LoadingTiffSettings : Const_LoadingTiffSettings
        {
            internal unsafe LoadingTiffSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.Filesystem.Path Dir
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_GetMutable_dir", ExactSpelling = true)]
                    extern static MR.Std.Filesystem.Path._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_dir(_Underlying *_this);
                    return new(__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_dir(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Vector3f VoxelSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_GetMutable_voxelSize", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_voxelSize(_Underlying *_this);
                    return new(__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe ref MR.VoxelsLoad.GridType GridType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_GetMutable_gridType", ExactSpelling = true)]
                    extern static MR.VoxelsLoad.GridType *__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_gridType(_Underlying *_this);
                    return ref *__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_gridType(_UnderlyingPtr);
                }
            }

            public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_GetMutable_cb", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_cb(_Underlying *_this);
                    return new(__MR_VoxelsLoad_LoadingTiffSettings_GetMutable_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe LoadingTiffSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_LoadingTiffSettings_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::LoadingTiffSettings` elementwise.
            public unsafe LoadingTiffSettings(ReadOnlySpan<char> dir, MR.Vector3f voxelSize, MR.VoxelsLoad.GridType gridType, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_ConstructFrom(byte *dir, byte *dir_end, MR.Vector3f voxelSize, MR.VoxelsLoad.GridType gridType, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                byte[] __bytes_dir = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dir.Length)];
                int __len_dir = System.Text.Encoding.UTF8.GetBytes(dir, __bytes_dir);
                fixed (byte *__ptr_dir = __bytes_dir)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_LoadingTiffSettings_ConstructFrom(__ptr_dir, __ptr_dir + __len_dir, voxelSize, gridType, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::LoadingTiffSettings::LoadingTiffSettings`.
            public unsafe LoadingTiffSettings(MR.VoxelsLoad._ByValue_LoadingTiffSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.LoadingTiffSettings._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_LoadingTiffSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::VoxelsLoad::LoadingTiffSettings::operator=`.
            public unsafe MR.VoxelsLoad.LoadingTiffSettings Assign(MR.VoxelsLoad._ByValue_LoadingTiffSettings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_LoadingTiffSettings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.LoadingTiffSettings._Underlying *__MR_VoxelsLoad_LoadingTiffSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.LoadingTiffSettings._Underlying *_other);
                return new(__MR_VoxelsLoad_LoadingTiffSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `LoadingTiffSettings` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `LoadingTiffSettings`/`Const_LoadingTiffSettings` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_LoadingTiffSettings
        {
            internal readonly Const_LoadingTiffSettings? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_LoadingTiffSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_LoadingTiffSettings(Const_LoadingTiffSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_LoadingTiffSettings(Const_LoadingTiffSettings arg) {return new(arg);}
            public _ByValue_LoadingTiffSettings(MR.Misc._Moved<LoadingTiffSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_LoadingTiffSettings(MR.Misc._Moved<LoadingTiffSettings> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `LoadingTiffSettings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadingTiffSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `LoadingTiffSettings`/`Const_LoadingTiffSettings` directly.
        public class _InOptMut_LoadingTiffSettings
        {
            public LoadingTiffSettings? Opt;

            public _InOptMut_LoadingTiffSettings() {}
            public _InOptMut_LoadingTiffSettings(LoadingTiffSettings value) {Opt = value;}
            public static implicit operator _InOptMut_LoadingTiffSettings(LoadingTiffSettings value) {return new(value);}
        }

        /// This is used for optional parameters of class `LoadingTiffSettings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadingTiffSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `LoadingTiffSettings`/`Const_LoadingTiffSettings` to pass it to the function.
        public class _InOptConst_LoadingTiffSettings
        {
            public Const_LoadingTiffSettings? Opt;

            public _InOptConst_LoadingTiffSettings() {}
            public _InOptConst_LoadingTiffSettings(Const_LoadingTiffSettings value) {Opt = value;}
            public static implicit operator _InOptConst_LoadingTiffSettings(Const_LoadingTiffSettings value) {return new(value);}
        }

        /// Load raw voxels from file with provided parameters
        /// Generated from function `MR::VoxelsLoad::fromRaw`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> FromRaw(ReadOnlySpan<char> file, MR.VoxelsLoad.Const_RawParameters params_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromRaw_3_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromRaw_3_std_filesystem_path(byte *file, byte *file_end, MR.VoxelsLoad.Const_RawParameters._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_VoxelsLoad_fromRaw_3_std_filesystem_path(__ptr_file, __ptr_file + __len_file, params_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Load raw voxels from stream with provided parameters;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::VoxelsLoad::fromRaw`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> FromRaw(MR.Std.Istream in_, MR.VoxelsLoad.Const_RawParameters params_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromRaw_3_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromRaw_3_std_istream(MR.Std.Istream._Underlying *in_, MR.VoxelsLoad.Const_RawParameters._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_VoxelsLoad_fromRaw_3_std_istream(in_._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// Load raw voxels from file with provided parameters
        /// Generated from function `MR::VoxelsLoad::gridFromRaw`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRFloatGrid_StdString> GridFromRaw(ReadOnlySpan<char> file, MR.VoxelsLoad.Const_RawParameters params_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_gridFromRaw_3_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRFloatGrid_StdString._Underlying *__MR_VoxelsLoad_gridFromRaw_3_std_filesystem_path(byte *file, byte *file_end, MR.VoxelsLoad.Const_RawParameters._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRFloatGrid_StdString(__MR_VoxelsLoad_gridFromRaw_3_std_filesystem_path(__ptr_file, __ptr_file + __len_file, params_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Load raw voxels from stream with provided parameters;
        /// important on Windows: in stream must be open in binary mode
        /// Generated from function `MR::VoxelsLoad::gridFromRaw`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRFloatGrid_StdString> GridFromRaw(MR.Std.Istream in_, MR.VoxelsLoad.Const_RawParameters params_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_gridFromRaw_3_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRFloatGrid_StdString._Underlying *__MR_VoxelsLoad_gridFromRaw_3_std_istream(MR.Std.Istream._Underlying *in_, MR.VoxelsLoad.Const_RawParameters._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_MRFloatGrid_StdString(__MR_VoxelsLoad_gridFromRaw_3_std_istream(in_._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// finds raw voxels file and its encoding parameters
        /// \param file on input: file name probably without suffix with parameters
        ///             on output: if success existing file name
        /// Generated from function `MR::VoxelsLoad::findRawParameters`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVoxelsLoadRawParameters_StdString> FindRawParameters(MR.Std.Filesystem.Path file)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_findRawParameters", ExactSpelling = true)]
            extern static MR.Expected_MRVoxelsLoadRawParameters_StdString._Underlying *__MR_VoxelsLoad_findRawParameters(MR.Std.Filesystem.Path._Underlying *file);
            return MR.Misc.Move(new MR.Expected_MRVoxelsLoadRawParameters_StdString(__MR_VoxelsLoad_findRawParameters(file._UnderlyingPtr), is_owning: true));
        }

        /// Load raw voxels file, parsing parameters from name
        /// Generated from function `MR::VoxelsLoad::fromRaw`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> FromRaw(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromRaw_2", ExactSpelling = true)]
            extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromRaw_2(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_VoxelsLoad_fromRaw_2(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Load raw voxels file, parsing parameters from name
        /// Generated from function `MR::VoxelsLoad::gridFromRaw`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRFloatGrid_StdString> GridFromRaw(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_gridFromRaw_2", ExactSpelling = true)]
            extern static MR.Expected_MRFloatGrid_StdString._Underlying *__MR_VoxelsLoad_gridFromRaw_2(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRFloatGrid_StdString(__MR_VoxelsLoad_gridFromRaw_2(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Load all voxel volumes from OpenVDB file
        /// Generated from function `MR::VoxelsLoad::fromVdb`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRVdbVolume_StdString> FromVdb(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromVdb", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromVdb(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRVdbVolume_StdString(__MR_VoxelsLoad_fromVdb(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsLoad::gridsFromVdb`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRFloatGrid_StdString> GridsFromVdb(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_gridsFromVdb_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRFloatGrid_StdString._Underlying *__MR_VoxelsLoad_gridsFromVdb_std_filesystem_path(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRFloatGrid_StdString(__MR_VoxelsLoad_gridsFromVdb_std_filesystem_path(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::VoxelsLoad::gridsFromVdb`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRFloatGrid_StdString> GridsFromVdb(MR.Std.Istream in_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_gridsFromVdb_std_istream", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRFloatGrid_StdString._Underlying *__MR_VoxelsLoad_gridsFromVdb_std_istream(MR.Std.Istream._Underlying *in_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_StdVectorMRFloatGrid_StdString(__MR_VoxelsLoad_gridsFromVdb_std_istream(in_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// Load voxel from Gav-file with micro CT reconstruction
        /// Generated from function `MR::VoxelsLoad::fromGav`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> FromGav(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromGav_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromGav_std_filesystem_path(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_VoxelsLoad_fromGav_std_filesystem_path(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Load voxel from Gav-stream with micro CT reconstruction
        /// Generated from function `MR::VoxelsLoad::fromGav`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> FromGav(MR.Std.Istream in_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromGav_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromGav_std_istream(MR.Std.Istream._Underlying *in_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_VoxelsLoad_fromGav_std_istream(in_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// Detects the format from file extension and loads voxels from it
        /// Generated from function `MR::VoxelsLoad::gridsFromAnySupportedFormat`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRFloatGrid_StdString> GridsFromAnySupportedFormat(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_gridsFromAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRFloatGrid_StdString._Underlying *__MR_VoxelsLoad_gridsFromAnySupportedFormat(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRFloatGrid_StdString(__MR_VoxelsLoad_gridsFromAnySupportedFormat(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Detects the format from file extension and loads voxels from it
        /// Generated from function `MR::VoxelsLoad::fromAnySupportedFormat`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRVdbVolume_StdString> FromAnySupportedFormat(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_fromAnySupportedFormat", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_fromAnySupportedFormat(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRVdbVolume_StdString(__MR_VoxelsLoad_fromAnySupportedFormat(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Load voxels from a set of TIFF files
        /// Generated from function `MR::VoxelsLoad::loadTiffDir`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> LoadTiffDir(MR.VoxelsLoad.Const_LoadingTiffSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadTiffDir", ExactSpelling = true)]
            extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_VoxelsLoad_loadTiffDir(MR.VoxelsLoad.Const_LoadingTiffSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_VoxelsLoad_loadTiffDir(settings._UnderlyingPtr), is_owning: true));
        }
    }

    /// loads voxels from given file in new object
    /// Generated from function `MR::makeObjectVoxelsFromFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdSharedPtrMRObjectVoxels_StdString> MakeObjectVoxelsFromFile(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectVoxelsFromFile", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdSharedPtrMRObjectVoxels_StdString._Underlying *__MR_makeObjectVoxelsFromFile(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_StdVectorStdSharedPtrMRObjectVoxels_StdString(__MR_makeObjectVoxelsFromFile(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// Generated from function `MR::makeObjectFromVoxelsFile`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjects_StdString> MakeObjectFromVoxelsFile(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectFromVoxelsFile", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjects_StdString._Underlying *__MR_makeObjectFromVoxelsFile(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjects_StdString(__MR_makeObjectFromVoxelsFile(__ptr_file, __ptr_file + __len_file, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
        }
    }
}
