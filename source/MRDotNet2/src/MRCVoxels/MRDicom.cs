public static partial class MR
{
    public static partial class VoxelsLoad
    {
        /// Generated from class `MR::VoxelsLoad::DicomVolume`.
        /// This is the const half of the class.
        public class Const_DicomVolume : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_DicomVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_Destroy", ExactSpelling = true)]
                extern static void __MR_VoxelsLoad_DicomVolume_Destroy(_Underlying *_this);
                __MR_VoxelsLoad_DicomVolume_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_DicomVolume() {Dispose(false);}

            public unsafe MR.Const_SimpleVolumeMinMax Vol
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_Get_vol", ExactSpelling = true)]
                    extern static MR.Const_SimpleVolumeMinMax._Underlying *__MR_VoxelsLoad_DicomVolume_Get_vol(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolume_Get_vol(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_Get_name", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_VoxelsLoad_DicomVolume_Get_name(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolume_Get_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_Get_xf", ExactSpelling = true)]
                    extern static MR.Const_AffineXf3f._Underlying *__MR_VoxelsLoad_DicomVolume_Get_xf(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolume_Get_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_DicomVolume() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolume_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::DicomVolume` elementwise.
            public unsafe Const_DicomVolume(MR._ByValue_SimpleVolumeMinMax vol, ReadOnlySpan<char> name, MR.AffineXf3f xf) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.SimpleVolumeMinMax._Underlying *vol, byte *name, byte *name_end, MR.AffineXf3f xf);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_DicomVolume_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, __ptr_name, __ptr_name + __len_name, xf);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::DicomVolume::DicomVolume`.
            public unsafe Const_DicomVolume(MR.VoxelsLoad._ByValue_DicomVolume _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomVolume._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::VoxelsLoad::DicomVolume`.
        /// This is the non-const half of the class.
        public class DicomVolume : Const_DicomVolume
        {
            internal unsafe DicomVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.SimpleVolumeMinMax Vol
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_GetMutable_vol", ExactSpelling = true)]
                    extern static MR.SimpleVolumeMinMax._Underlying *__MR_VoxelsLoad_DicomVolume_GetMutable_vol(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolume_GetMutable_vol(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_GetMutable_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_VoxelsLoad_DicomVolume_GetMutable_name(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolume_GetMutable_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.Mut_AffineXf3f._Underlying *__MR_VoxelsLoad_DicomVolume_GetMutable_xf(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolume_GetMutable_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe DicomVolume() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolume_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::DicomVolume` elementwise.
            public unsafe DicomVolume(MR._ByValue_SimpleVolumeMinMax vol, ReadOnlySpan<char> name, MR.AffineXf3f xf) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.SimpleVolumeMinMax._Underlying *vol, byte *name, byte *name_end, MR.AffineXf3f xf);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_DicomVolume_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, __ptr_name, __ptr_name + __len_name, xf);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::DicomVolume::DicomVolume`.
            public unsafe DicomVolume(MR.VoxelsLoad._ByValue_DicomVolume _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomVolume._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::VoxelsLoad::DicomVolume::operator=`.
            public unsafe MR.VoxelsLoad.DicomVolume Assign(MR.VoxelsLoad._ByValue_DicomVolume _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolume_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolume._Underlying *__MR_VoxelsLoad_DicomVolume_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomVolume._Underlying *_other);
                return new(__MR_VoxelsLoad_DicomVolume_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `DicomVolume` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `DicomVolume`/`Const_DicomVolume` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_DicomVolume
        {
            internal readonly Const_DicomVolume? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_DicomVolume() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_DicomVolume(Const_DicomVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_DicomVolume(Const_DicomVolume arg) {return new(arg);}
            public _ByValue_DicomVolume(MR.Misc._Moved<DicomVolume> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_DicomVolume(MR.Misc._Moved<DicomVolume> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `DicomVolume` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_DicomVolume`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DicomVolume`/`Const_DicomVolume` directly.
        public class _InOptMut_DicomVolume
        {
            public DicomVolume? Opt;

            public _InOptMut_DicomVolume() {}
            public _InOptMut_DicomVolume(DicomVolume value) {Opt = value;}
            public static implicit operator _InOptMut_DicomVolume(DicomVolume value) {return new(value);}
        }

        /// This is used for optional parameters of class `DicomVolume` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_DicomVolume`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DicomVolume`/`Const_DicomVolume` to pass it to the function.
        public class _InOptConst_DicomVolume
        {
            public Const_DicomVolume? Opt;

            public _InOptConst_DicomVolume() {}
            public _InOptConst_DicomVolume(Const_DicomVolume value) {Opt = value;}
            public static implicit operator _InOptConst_DicomVolume(Const_DicomVolume value) {return new(value);}
        }

        /// Generated from class `MR::VoxelsLoad::DicomVolumeAsVdb`.
        /// This is the const half of the class.
        public class Const_DicomVolumeAsVdb : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_DicomVolumeAsVdb(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_Destroy", ExactSpelling = true)]
                extern static void __MR_VoxelsLoad_DicomVolumeAsVdb_Destroy(_Underlying *_this);
                __MR_VoxelsLoad_DicomVolumeAsVdb_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_DicomVolumeAsVdb() {Dispose(false);}

            public unsafe MR.Const_VdbVolume Vol
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_Get_vol", ExactSpelling = true)]
                    extern static MR.Const_VdbVolume._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_Get_vol(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolumeAsVdb_Get_vol(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_Get_name", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_Get_name(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolumeAsVdb_Get_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_Get_xf", ExactSpelling = true)]
                    extern static MR.Const_AffineXf3f._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_Get_xf(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolumeAsVdb_Get_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_DicomVolumeAsVdb() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolumeAsVdb_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::DicomVolumeAsVdb` elementwise.
            public unsafe Const_DicomVolumeAsVdb(MR._ByValue_VdbVolume vol, ReadOnlySpan<char> name, MR.AffineXf3f xf) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.VdbVolume._Underlying *vol, byte *name, byte *name_end, MR.AffineXf3f xf);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, __ptr_name, __ptr_name + __len_name, xf);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::DicomVolumeAsVdb::DicomVolumeAsVdb`.
            public unsafe Const_DicomVolumeAsVdb(MR.VoxelsLoad._ByValue_DicomVolumeAsVdb _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::VoxelsLoad::DicomVolumeAsVdb`.
        /// This is the non-const half of the class.
        public class DicomVolumeAsVdb : Const_DicomVolumeAsVdb
        {
            internal unsafe DicomVolumeAsVdb(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.VdbVolume Vol
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_vol", ExactSpelling = true)]
                    extern static MR.VdbVolume._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_vol(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_vol(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_name(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.Mut_AffineXf3f._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_xf(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomVolumeAsVdb_GetMutable_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe DicomVolumeAsVdb() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_DefaultConstruct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_DefaultConstruct();
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolumeAsVdb_DefaultConstruct();
            }

            /// Constructs `MR::VoxelsLoad::DicomVolumeAsVdb` elementwise.
            public unsafe DicomVolumeAsVdb(MR._ByValue_VdbVolume vol, ReadOnlySpan<char> name, MR.AffineXf3f xf) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFrom", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFrom(MR.Misc._PassBy vol_pass_by, MR.VdbVolume._Underlying *vol, byte *name, byte *name_end, MR.AffineXf3f xf);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFrom(vol.PassByMode, vol.Value is not null ? vol.Value._UnderlyingPtr : null, __ptr_name, __ptr_name + __len_name, xf);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::DicomVolumeAsVdb::DicomVolumeAsVdb`.
            public unsafe DicomVolumeAsVdb(MR.VoxelsLoad._ByValue_DicomVolumeAsVdb _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_DicomVolumeAsVdb_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::VoxelsLoad::DicomVolumeAsVdb::operator=`.
            public unsafe MR.VoxelsLoad.DicomVolumeAsVdb Assign(MR.VoxelsLoad._ByValue_DicomVolumeAsVdb _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomVolumeAsVdb_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *__MR_VoxelsLoad_DicomVolumeAsVdb_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomVolumeAsVdb._Underlying *_other);
                return new(__MR_VoxelsLoad_DicomVolumeAsVdb_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `DicomVolumeAsVdb` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `DicomVolumeAsVdb`/`Const_DicomVolumeAsVdb` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_DicomVolumeAsVdb
        {
            internal readonly Const_DicomVolumeAsVdb? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_DicomVolumeAsVdb() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_DicomVolumeAsVdb(Const_DicomVolumeAsVdb new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_DicomVolumeAsVdb(Const_DicomVolumeAsVdb arg) {return new(arg);}
            public _ByValue_DicomVolumeAsVdb(MR.Misc._Moved<DicomVolumeAsVdb> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_DicomVolumeAsVdb(MR.Misc._Moved<DicomVolumeAsVdb> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `DicomVolumeAsVdb` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_DicomVolumeAsVdb`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DicomVolumeAsVdb`/`Const_DicomVolumeAsVdb` directly.
        public class _InOptMut_DicomVolumeAsVdb
        {
            public DicomVolumeAsVdb? Opt;

            public _InOptMut_DicomVolumeAsVdb() {}
            public _InOptMut_DicomVolumeAsVdb(DicomVolumeAsVdb value) {Opt = value;}
            public static implicit operator _InOptMut_DicomVolumeAsVdb(DicomVolumeAsVdb value) {return new(value);}
        }

        /// This is used for optional parameters of class `DicomVolumeAsVdb` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_DicomVolumeAsVdb`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DicomVolumeAsVdb`/`Const_DicomVolumeAsVdb` to pass it to the function.
        public class _InOptConst_DicomVolumeAsVdb
        {
            public Const_DicomVolumeAsVdb? Opt;

            public _InOptConst_DicomVolumeAsVdb() {}
            public _InOptConst_DicomVolumeAsVdb(Const_DicomVolumeAsVdb value) {Opt = value;}
            public static implicit operator _InOptConst_DicomVolumeAsVdb(Const_DicomVolumeAsVdb value) {return new(value);}
        }

        public enum DicomStatusEnum : int
        {
            // valid DICOM and we can open it
            Ok = 0,
            // not a valid DICOM
            Invalid = 1,
            // a valid DICOM, but we do not support it (e.g. some MediaStorages)
            Unsupported = 2,
        }

        /// Generated from class `MR::VoxelsLoad::DicomStatus`.
        /// This is the const half of the class.
        public class Const_DicomStatus : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.VoxelsLoad.DicomStatusEnum>
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_DicomStatus(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_Destroy", ExactSpelling = true)]
                extern static void __MR_VoxelsLoad_DicomStatus_Destroy(_Underlying *_this);
                __MR_VoxelsLoad_DicomStatus_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_DicomStatus() {Dispose(false);}

            public unsafe MR.VoxelsLoad.DicomStatusEnum Status
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_Get_status", ExactSpelling = true)]
                    extern static MR.VoxelsLoad.DicomStatusEnum *__MR_VoxelsLoad_DicomStatus_Get_status(_Underlying *_this);
                    return *__MR_VoxelsLoad_DicomStatus_Get_status(_UnderlyingPtr);
                }
            }

            // if status is Unsupported, specify reason why
            public unsafe MR.Std.Const_String Reason
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_Get_reason", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_VoxelsLoad_DicomStatus_Get_reason(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomStatus_Get_reason(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::DicomStatus::DicomStatus`.
            public unsafe Const_DicomStatus(MR.VoxelsLoad._ByValue_DicomStatus _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomStatus._Underlying *__MR_VoxelsLoad_DicomStatus_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomStatus._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_DicomStatus_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            // implicit by design
            /// Generated from constructor `MR::VoxelsLoad::DicomStatus::DicomStatus`.
            /// Parameter `rs` defaults to `""`.
            public unsafe Const_DicomStatus(MR.VoxelsLoad.DicomStatusEnum st, MR.Misc.ReadOnlyCharSpanOpt rs = new()) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_Construct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomStatus._Underlying *__MR_VoxelsLoad_DicomStatus_Construct(MR.VoxelsLoad.DicomStatusEnum st, byte *rs, byte *rs_end);
                byte[] __bytes_rs;
                int __len_rs = 0;
                if (rs.HasValue)
                {
                    __bytes_rs = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(rs.Value.Length)];
                    __len_rs = System.Text.Encoding.UTF8.GetBytes(rs.Value, __bytes_rs);
                }
                fixed (byte *__ptr_rs = __bytes_rs)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_DicomStatus_Construct(st, rs.HasValue ? __ptr_rs : null, rs.HasValue ? __ptr_rs + __len_rs : null);
                }
            }

            /// Generated from conversion operator `MR::VoxelsLoad::DicomStatus::operator bool`.
            public static unsafe explicit operator bool(MR.VoxelsLoad.Const_DicomStatus _this)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_ConvertTo_bool", ExactSpelling = true)]
                extern static byte __MR_VoxelsLoad_DicomStatus_ConvertTo_bool(MR.VoxelsLoad.Const_DicomStatus._Underlying *_this);
                return __MR_VoxelsLoad_DicomStatus_ConvertTo_bool(_this._UnderlyingPtr) != 0;
            }

            /// Generated from method `MR::VoxelsLoad::DicomStatus::operator==`.
            public static unsafe bool operator==(MR.VoxelsLoad.Const_DicomStatus _this, MR.VoxelsLoad.DicomStatusEnum s)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VoxelsLoad_DicomStatus_MR_VoxelsLoad_DicomStatusEnum", ExactSpelling = true)]
                extern static byte __MR_equal_MR_VoxelsLoad_DicomStatus_MR_VoxelsLoad_DicomStatusEnum(MR.VoxelsLoad.Const_DicomStatus._Underlying *_this, MR.VoxelsLoad.DicomStatusEnum s);
                return __MR_equal_MR_VoxelsLoad_DicomStatus_MR_VoxelsLoad_DicomStatusEnum(_this._UnderlyingPtr, s) != 0;
            }

            public static unsafe bool operator!=(MR.VoxelsLoad.Const_DicomStatus _this, MR.VoxelsLoad.DicomStatusEnum s)
            {
                return !(_this == s);
            }

            // IEquatable:

            public bool Equals(MR.VoxelsLoad.DicomStatusEnum s)
            {
                return this == s;
            }

            public override bool Equals(object? other)
            {
                if (other is null)
                    return false;
                if (other is MR.VoxelsLoad.DicomStatusEnum)
                    return this == (MR.VoxelsLoad.DicomStatusEnum)other;
                return false;
            }
        }

        /// Generated from class `MR::VoxelsLoad::DicomStatus`.
        /// This is the non-const half of the class.
        public class DicomStatus : Const_DicomStatus
        {
            internal unsafe DicomStatus(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe ref MR.VoxelsLoad.DicomStatusEnum Status
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_GetMutable_status", ExactSpelling = true)]
                    extern static MR.VoxelsLoad.DicomStatusEnum *__MR_VoxelsLoad_DicomStatus_GetMutable_status(_Underlying *_this);
                    return ref *__MR_VoxelsLoad_DicomStatus_GetMutable_status(_UnderlyingPtr);
                }
            }

            // if status is Unsupported, specify reason why
            public new unsafe MR.Std.String Reason
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_GetMutable_reason", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_VoxelsLoad_DicomStatus_GetMutable_reason(_Underlying *_this);
                    return new(__MR_VoxelsLoad_DicomStatus_GetMutable_reason(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Generated from constructor `MR::VoxelsLoad::DicomStatus::DicomStatus`.
            public unsafe DicomStatus(MR.VoxelsLoad._ByValue_DicomStatus _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomStatus._Underlying *__MR_VoxelsLoad_DicomStatus_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomStatus._Underlying *_other);
                _UnderlyingPtr = __MR_VoxelsLoad_DicomStatus_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            // implicit by design
            /// Generated from constructor `MR::VoxelsLoad::DicomStatus::DicomStatus`.
            /// Parameter `rs` defaults to `""`.
            public unsafe DicomStatus(MR.VoxelsLoad.DicomStatusEnum st, MR.Misc.ReadOnlyCharSpanOpt rs = new()) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_Construct", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomStatus._Underlying *__MR_VoxelsLoad_DicomStatus_Construct(MR.VoxelsLoad.DicomStatusEnum st, byte *rs, byte *rs_end);
                byte[] __bytes_rs;
                int __len_rs = 0;
                if (rs.HasValue)
                {
                    __bytes_rs = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(rs.Value.Length)];
                    __len_rs = System.Text.Encoding.UTF8.GetBytes(rs.Value, __bytes_rs);
                }
                fixed (byte *__ptr_rs = __bytes_rs)
                {
                    _UnderlyingPtr = __MR_VoxelsLoad_DicomStatus_Construct(st, rs.HasValue ? __ptr_rs : null, rs.HasValue ? __ptr_rs + __len_rs : null);
                }
            }

            /// Generated from method `MR::VoxelsLoad::DicomStatus::operator=`.
            public unsafe MR.VoxelsLoad.DicomStatus Assign(MR.VoxelsLoad._ByValue_DicomStatus _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_DicomStatus_AssignFromAnother", ExactSpelling = true)]
                extern static MR.VoxelsLoad.DicomStatus._Underlying *__MR_VoxelsLoad_DicomStatus_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsLoad.DicomStatus._Underlying *_other);
                return new(__MR_VoxelsLoad_DicomStatus_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `DicomStatus` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `DicomStatus`/`Const_DicomStatus` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_DicomStatus
        {
            internal readonly Const_DicomStatus? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_DicomStatus(Const_DicomStatus new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_DicomStatus(Const_DicomStatus arg) {return new(arg);}
            public _ByValue_DicomStatus(MR.Misc._Moved<DicomStatus> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_DicomStatus(MR.Misc._Moved<DicomStatus> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `DicomStatus` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_DicomStatus`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DicomStatus`/`Const_DicomStatus` directly.
        public class _InOptMut_DicomStatus
        {
            public DicomStatus? Opt;

            public _InOptMut_DicomStatus() {}
            public _InOptMut_DicomStatus(DicomStatus value) {Opt = value;}
            public static implicit operator _InOptMut_DicomStatus(DicomStatus value) {return new(value);}
        }

        /// This is used for optional parameters of class `DicomStatus` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_DicomStatus`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `DicomStatus`/`Const_DicomStatus` to pass it to the function.
        public class _InOptConst_DicomStatus
        {
            public Const_DicomStatus? Opt;

            public _InOptConst_DicomStatus() {}
            public _InOptConst_DicomStatus(Const_DicomStatus value) {Opt = value;}
            public static implicit operator _InOptConst_DicomStatus(Const_DicomStatus value) {return new(value);}
        }

        /// check if file is a valid DICOM dataset file
        /// \param seriesUid - if set, the extracted series instance UID is copied to the variable
        /// Generated from function `MR::VoxelsLoad::isDicomFile`.
        public static unsafe MR.Misc._Moved<MR.VoxelsLoad.DicomStatus> IsDicomFile(ReadOnlySpan<char> path, MR.Std.String? seriesUid = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_isDicomFile", ExactSpelling = true)]
            extern static MR.VoxelsLoad.DicomStatus._Underlying *__MR_VoxelsLoad_isDicomFile(byte *path, byte *path_end, MR.Std.String._Underlying *seriesUid);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.VoxelsLoad.DicomStatus(__MR_VoxelsLoad_isDicomFile(__ptr_path, __ptr_path + __len_path, seriesUid is not null ? seriesUid._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// check if given folder contains at least one DICOM file
        /// Generated from function `MR::VoxelsLoad::isDicomFolder`.
        public static unsafe bool IsDicomFolder(ReadOnlySpan<char> dirPath)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_isDicomFolder", ExactSpelling = true)]
            extern static byte __MR_VoxelsLoad_isDicomFolder(byte *dirPath, byte *dirPath_end);
            byte[] __bytes_dirPath = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(dirPath.Length)];
            int __len_dirPath = System.Text.Encoding.UTF8.GetBytes(dirPath, __bytes_dirPath);
            fixed (byte *__ptr_dirPath = __bytes_dirPath)
            {
                return __MR_VoxelsLoad_isDicomFolder(__ptr_dirPath, __ptr_dirPath + __len_dirPath) != 0;
            }
        }

        /// returns all the dicom folders in \p path, searching recursively
        /// Generated from function `MR::VoxelsLoad::findDicomFoldersRecursively`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_StdFilesystemPath> FindDicomFoldersRecursively(ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_findDicomFoldersRecursively", ExactSpelling = true)]
            extern static MR.Std.Vector_StdFilesystemPath._Underlying *__MR_VoxelsLoad_findDicomFoldersRecursively(byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Std.Vector_StdFilesystemPath(__MR_VoxelsLoad_findDicomFoldersRecursively(__ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }

        /// Loads full volume from single DICOM file (not a slice file) as SimpleVolumeMinMax
        /// Generated from function `MR::VoxelsLoad::loadDicomFile`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVoxelsLoadDicomVolume_StdString> LoadDicomFile(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomFile", ExactSpelling = true)]
            extern static MR.Expected_MRVoxelsLoadDicomVolume_StdString._Underlying *__MR_VoxelsLoad_loadDicomFile(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRVoxelsLoadDicomVolume_StdString(__MR_VoxelsLoad_loadDicomFile(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Loads full volume from single DICOM file (not a slice file) as VdbVolume
        /// Generated from function `MR::VoxelsLoad::loadDicomFileAsVdb`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVoxelsLoadDicomVolumeAsVdb_StdString> LoadDicomFileAsVdb(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomFileAsVdb", ExactSpelling = true)]
            extern static MR.Expected_MRVoxelsLoadDicomVolumeAsVdb_StdString._Underlying *__MR_VoxelsLoad_loadDicomFileAsVdb(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRVoxelsLoadDicomVolumeAsVdb_StdString(__MR_VoxelsLoad_loadDicomFileAsVdb(__ptr_file, __ptr_file + __len_file, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Loads one volume from DICOM files located in given folder as SimpleVolumeMinMax
        /// Generated from function `MR::VoxelsLoad::loadDicomFolder`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVoxelsLoadDicomVolume_StdString> LoadDicomFolder(ReadOnlySpan<char> path, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomFolder", ExactSpelling = true)]
            extern static MR.Expected_MRVoxelsLoadDicomVolume_StdString._Underlying *__MR_VoxelsLoad_loadDicomFolder(byte *path, byte *path_end, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRVoxelsLoadDicomVolume_StdString(__MR_VoxelsLoad_loadDicomFolder(__ptr_path, __ptr_path + __len_path, maxNumThreads, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Loads one volume from DICOM files located in given folder as VdbVolume
        /// Generated from function `MR::VoxelsLoad::loadDicomFolderAsVdb`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRVoxelsLoadDicomVolumeAsVdb_StdString> LoadDicomFolderAsVdb(ReadOnlySpan<char> path, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomFolderAsVdb", ExactSpelling = true)]
            extern static MR.Expected_MRVoxelsLoadDicomVolumeAsVdb_StdString._Underlying *__MR_VoxelsLoad_loadDicomFolderAsVdb(byte *path, byte *path_end, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRVoxelsLoadDicomVolumeAsVdb_StdString(__MR_VoxelsLoad_loadDicomFolderAsVdb(__ptr_path, __ptr_path + __len_path, maxNumThreads, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Loads all volumes from DICOM files located in given folder as a number of SimpleVolumeMinMax
        /// Generated from function `MR::VoxelsLoad::loadDicomsFolder`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeStdString> LoadDicomsFolder(ReadOnlySpan<char> path, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomsFolder", ExactSpelling = true)]
            extern static MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeStdString._Underlying *__MR_VoxelsLoad_loadDicomsFolder(byte *path, byte *path_end, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeStdString(__MR_VoxelsLoad_loadDicomsFolder(__ptr_path, __ptr_path + __len_path, maxNumThreads, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Loads all volumes from DICOM files located in given folder as a number of VdbVolume
        /// Generated from function `MR::VoxelsLoad::loadDicomsFolderAsVdb`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeAsVdbStdString> LoadDicomsFolderAsVdb(ReadOnlySpan<char> path, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomsFolderAsVdb", ExactSpelling = true)]
            extern static MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeAsVdbStdString._Underlying *__MR_VoxelsLoad_loadDicomsFolderAsVdb(byte *path, byte *path_end, uint maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeAsVdbStdString(__MR_VoxelsLoad_loadDicomsFolderAsVdb(__ptr_path, __ptr_path + __len_path, maxNumThreads, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Loads every subfolder with DICOM volume as new object
        /// Generated from function `MR::VoxelsLoad::loadDicomsFolderTreeAsVdb`.
        /// Parameter `maxNumThreads` defaults to `4`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeAsVdbStdString> LoadDicomsFolderTreeAsVdb(ReadOnlySpan<char> path, uint? maxNumThreads = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_loadDicomsFolderTreeAsVdb", ExactSpelling = true)]
            extern static MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeAsVdbStdString._Underlying *__MR_VoxelsLoad_loadDicomsFolderTreeAsVdb(byte *path, byte *path_end, uint *maxNumThreads, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                uint __deref_maxNumThreads = maxNumThreads.GetValueOrDefault();
                return MR.Misc.Move(new MR.Std.Vector_ExpectedMRVoxelsLoadDicomVolumeAsVdbStdString(__MR_VoxelsLoad_loadDicomsFolderTreeAsVdb(__ptr_path, __ptr_path + __len_path, maxNumThreads.HasValue ? &__deref_maxNumThreads : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// converts DicomVolumeAsVdb in ObjectVoxels
        /// Generated from function `MR::VoxelsLoad::createObjectVoxels`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdSharedPtrMRObjectVoxels_StdString> CreateObjectVoxels(MR.VoxelsLoad.Const_DicomVolumeAsVdb dcm, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_createObjectVoxels", ExactSpelling = true)]
            extern static MR.Expected_StdSharedPtrMRObjectVoxels_StdString._Underlying *__MR_VoxelsLoad_createObjectVoxels(MR.VoxelsLoad.Const_DicomVolumeAsVdb._Underlying *dcm, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_StdSharedPtrMRObjectVoxels_StdString(__MR_VoxelsLoad_createObjectVoxels(dcm._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// Loads 3D volumetric data from dicom-files in given folder, and converts them into an ObjectVoxels
        /// Generated from function `MR::VoxelsLoad::makeObjectVoxelsFromDicomFolder`.
        /// Parameter `callback` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjects_StdString> MakeObjectVoxelsFromDicomFolder(ReadOnlySpan<char> folder, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsLoad_makeObjectVoxelsFromDicomFolder", ExactSpelling = true)]
            extern static MR.Expected_MRLoadedObjects_StdString._Underlying *__MR_VoxelsLoad_makeObjectVoxelsFromDicomFolder(byte *folder, byte *folder_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
            byte[] __bytes_folder = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(folder.Length)];
            int __len_folder = System.Text.Encoding.UTF8.GetBytes(folder, __bytes_folder);
            fixed (byte *__ptr_folder = __bytes_folder)
            {
                return MR.Misc.Move(new MR.Expected_MRLoadedObjects_StdString(__MR_VoxelsLoad_makeObjectVoxelsFromDicomFolder(__ptr_folder, __ptr_folder + __len_folder, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
            }
        }
    }

    public static partial class VoxelsSave
    {
        /// Save voxels object to a single 3d DICOM file
        /// Generated from function `MR::VoxelsSave::toDicom`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToDicom(MR.Const_VdbVolume vdbVolume, ReadOnlySpan<char> path, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toDicom", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toDicom(MR.Const_VdbVolume._Underlying *vdbVolume, byte *path, byte *path_end, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toDicom(vdbVolume._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Saves object to a single 3d DICOM file. \p sourceScale specifies the true scale of the voxel data
        /// which will be saved with "slope" and "intercept" parameters of the output dicom.
        /// Generated from function `MR::VoxelsSave::toDicom<unsigned short>`.
        /// Parameter `sourceScale` defaults to `{}`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ToDicom(MR.Const_SimpleVolumeU16 volume, ReadOnlySpan<char> path, MR.Std.Const_Optional_MRBox1f? sourceScale = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsSave_toDicom_unsigned_short", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_VoxelsSave_toDicom_unsigned_short(MR.Const_SimpleVolumeU16._Underlying *volume, byte *path, byte *path_end, MR.Std.Const_Optional_MRBox1f._Underlying *sourceScale, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_VoxelsSave_toDicom_unsigned_short(volume._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, sourceScale is not null ? sourceScale._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}
