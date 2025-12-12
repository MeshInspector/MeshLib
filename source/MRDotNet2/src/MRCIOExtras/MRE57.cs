public static partial class MR
{
    public static partial class PointsLoad
    {
        /// Generated from class `MR::PointsLoad::E57LoadSettings`.
        /// This is the const half of the class.
        public class Const_E57LoadSettings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_E57LoadSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_Destroy", ExactSpelling = true)]
                extern static void __MR_PointsLoad_E57LoadSettings_Destroy(_Underlying *_this);
                __MR_PointsLoad_E57LoadSettings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_E57LoadSettings() {Dispose(false);}

            /// true => if input file has more than one cloud, they all will be combined in one
            public unsafe bool CombineAllObjects
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_Get_combineAllObjects", ExactSpelling = true)]
                    extern static bool *__MR_PointsLoad_E57LoadSettings_Get_combineAllObjects(_Underlying *_this);
                    return *__MR_PointsLoad_E57LoadSettings_Get_combineAllObjects(_UnderlyingPtr);
                }
            }

            /// true => return only identity transforms, applying them to points
            public unsafe bool IdentityXf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_Get_identityXf", ExactSpelling = true)]
                    extern static bool *__MR_PointsLoad_E57LoadSettings_Get_identityXf(_Underlying *_this);
                    return *__MR_PointsLoad_E57LoadSettings_Get_identityXf(_UnderlyingPtr);
                }
            }

            /// progress report and cancellation
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PointsLoad_E57LoadSettings_Get_progress(_Underlying *_this);
                    return new(__MR_PointsLoad_E57LoadSettings_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_E57LoadSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_PointsLoad_E57LoadSettings_DefaultConstruct();
            }

            /// Constructs `MR::PointsLoad::E57LoadSettings` elementwise.
            public unsafe Const_E57LoadSettings(bool combineAllObjects, bool identityXf, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_ConstructFrom(byte combineAllObjects, byte identityXf, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                _UnderlyingPtr = __MR_PointsLoad_E57LoadSettings_ConstructFrom(combineAllObjects ? (byte)1 : (byte)0, identityXf ? (byte)1 : (byte)0, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::PointsLoad::E57LoadSettings::E57LoadSettings`.
            public unsafe Const_E57LoadSettings(MR.PointsLoad._ByValue_E57LoadSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsLoad.E57LoadSettings._Underlying *_other);
                _UnderlyingPtr = __MR_PointsLoad_E57LoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::PointsLoad::E57LoadSettings`.
        /// This is the non-const half of the class.
        public class E57LoadSettings : Const_E57LoadSettings
        {
            internal unsafe E57LoadSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// true => if input file has more than one cloud, they all will be combined in one
            public new unsafe ref bool CombineAllObjects
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_GetMutable_combineAllObjects", ExactSpelling = true)]
                    extern static bool *__MR_PointsLoad_E57LoadSettings_GetMutable_combineAllObjects(_Underlying *_this);
                    return ref *__MR_PointsLoad_E57LoadSettings_GetMutable_combineAllObjects(_UnderlyingPtr);
                }
            }

            /// true => return only identity transforms, applying them to points
            public new unsafe ref bool IdentityXf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_GetMutable_identityXf", ExactSpelling = true)]
                    extern static bool *__MR_PointsLoad_E57LoadSettings_GetMutable_identityXf(_Underlying *_this);
                    return ref *__MR_PointsLoad_E57LoadSettings_GetMutable_identityXf(_UnderlyingPtr);
                }
            }

            /// progress report and cancellation
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PointsLoad_E57LoadSettings_GetMutable_progress(_Underlying *_this);
                    return new(__MR_PointsLoad_E57LoadSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe E57LoadSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_PointsLoad_E57LoadSettings_DefaultConstruct();
            }

            /// Constructs `MR::PointsLoad::E57LoadSettings` elementwise.
            public unsafe E57LoadSettings(bool combineAllObjects, bool identityXf, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_ConstructFrom(byte combineAllObjects, byte identityXf, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                _UnderlyingPtr = __MR_PointsLoad_E57LoadSettings_ConstructFrom(combineAllObjects ? (byte)1 : (byte)0, identityXf ? (byte)1 : (byte)0, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::PointsLoad::E57LoadSettings::E57LoadSettings`.
            public unsafe E57LoadSettings(MR.PointsLoad._ByValue_E57LoadSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsLoad.E57LoadSettings._Underlying *_other);
                _UnderlyingPtr = __MR_PointsLoad_E57LoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::PointsLoad::E57LoadSettings::operator=`.
            public unsafe MR.PointsLoad.E57LoadSettings Assign(MR.PointsLoad._ByValue_E57LoadSettings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_E57LoadSettings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PointsLoad.E57LoadSettings._Underlying *__MR_PointsLoad_E57LoadSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsLoad.E57LoadSettings._Underlying *_other);
                return new(__MR_PointsLoad_E57LoadSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `E57LoadSettings` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `E57LoadSettings`/`Const_E57LoadSettings` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_E57LoadSettings
        {
            internal readonly Const_E57LoadSettings? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_E57LoadSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_E57LoadSettings(Const_E57LoadSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_E57LoadSettings(Const_E57LoadSettings arg) {return new(arg);}
            public _ByValue_E57LoadSettings(MR.Misc._Moved<E57LoadSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_E57LoadSettings(MR.Misc._Moved<E57LoadSettings> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `E57LoadSettings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_E57LoadSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `E57LoadSettings`/`Const_E57LoadSettings` directly.
        public class _InOptMut_E57LoadSettings
        {
            public E57LoadSettings? Opt;

            public _InOptMut_E57LoadSettings() {}
            public _InOptMut_E57LoadSettings(E57LoadSettings value) {Opt = value;}
            public static implicit operator _InOptMut_E57LoadSettings(E57LoadSettings value) {return new(value);}
        }

        /// This is used for optional parameters of class `E57LoadSettings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_E57LoadSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `E57LoadSettings`/`Const_E57LoadSettings` to pass it to the function.
        public class _InOptConst_E57LoadSettings
        {
            public Const_E57LoadSettings? Opt;

            public _InOptConst_E57LoadSettings() {}
            public _InOptConst_E57LoadSettings(Const_E57LoadSettings value) {Opt = value;}
            public static implicit operator _InOptConst_E57LoadSettings(Const_E57LoadSettings value) {return new(value);}
        }

        /// loads scene from e57 file
        /// Generated from class `MR::PointsLoad::NamedCloud`.
        /// This is the const half of the class.
        public class Const_NamedCloud : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_NamedCloud(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_Destroy", ExactSpelling = true)]
                extern static void __MR_PointsLoad_NamedCloud_Destroy(_Underlying *_this);
                __MR_PointsLoad_NamedCloud_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_NamedCloud() {Dispose(false);}

            public unsafe MR.Std.Const_String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_Get_name", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_PointsLoad_NamedCloud_Get_name(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_Get_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_PointCloud Cloud
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_Get_cloud", ExactSpelling = true)]
                    extern static MR.Const_PointCloud._Underlying *__MR_PointsLoad_NamedCloud_Get_cloud(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_Get_cloud(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_Get_xf", ExactSpelling = true)]
                    extern static MR.Const_AffineXf3f._Underlying *__MR_PointsLoad_NamedCloud_Get_xf(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_Get_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_VertColors Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_Get_colors", ExactSpelling = true)]
                    extern static MR.Const_VertColors._Underlying *__MR_PointsLoad_NamedCloud_Get_colors(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_Get_colors(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_NamedCloud() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_DefaultConstruct();
                _UnderlyingPtr = __MR_PointsLoad_NamedCloud_DefaultConstruct();
            }

            /// Constructs `MR::PointsLoad::NamedCloud` elementwise.
            public unsafe Const_NamedCloud(ReadOnlySpan<char> name, MR._ByValue_PointCloud cloud, MR.AffineXf3f xf, MR._ByValue_VertColors colors) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_ConstructFrom", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_ConstructFrom(byte *name, byte *name_end, MR.Misc._PassBy cloud_pass_by, MR.PointCloud._Underlying *cloud, MR.AffineXf3f xf, MR.Misc._PassBy colors_pass_by, MR.VertColors._Underlying *colors);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_PointsLoad_NamedCloud_ConstructFrom(__ptr_name, __ptr_name + __len_name, cloud.PassByMode, cloud.Value is not null ? cloud.Value._UnderlyingPtr : null, xf, colors.PassByMode, colors.Value is not null ? colors.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from constructor `MR::PointsLoad::NamedCloud::NamedCloud`.
            public unsafe Const_NamedCloud(MR.PointsLoad._ByValue_NamedCloud _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsLoad.NamedCloud._Underlying *_other);
                _UnderlyingPtr = __MR_PointsLoad_NamedCloud_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// loads scene from e57 file
        /// Generated from class `MR::PointsLoad::NamedCloud`.
        /// This is the non-const half of the class.
        public class NamedCloud : Const_NamedCloud
        {
            internal unsafe NamedCloud(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.String Name
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_GetMutable_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_PointsLoad_NamedCloud_GetMutable_name(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_GetMutable_name(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.PointCloud Cloud
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_GetMutable_cloud", ExactSpelling = true)]
                    extern static MR.PointCloud._Underlying *__MR_PointsLoad_NamedCloud_GetMutable_cloud(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_GetMutable_cloud(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.Mut_AffineXf3f._Underlying *__MR_PointsLoad_NamedCloud_GetMutable_xf(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_GetMutable_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.VertColors Colors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_GetMutable_colors", ExactSpelling = true)]
                    extern static MR.VertColors._Underlying *__MR_PointsLoad_NamedCloud_GetMutable_colors(_Underlying *_this);
                    return new(__MR_PointsLoad_NamedCloud_GetMutable_colors(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe NamedCloud() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_DefaultConstruct();
                _UnderlyingPtr = __MR_PointsLoad_NamedCloud_DefaultConstruct();
            }

            /// Constructs `MR::PointsLoad::NamedCloud` elementwise.
            public unsafe NamedCloud(ReadOnlySpan<char> name, MR._ByValue_PointCloud cloud, MR.AffineXf3f xf, MR._ByValue_VertColors colors) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_ConstructFrom", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_ConstructFrom(byte *name, byte *name_end, MR.Misc._PassBy cloud_pass_by, MR.PointCloud._Underlying *cloud, MR.AffineXf3f xf, MR.Misc._PassBy colors_pass_by, MR.VertColors._Underlying *colors);
                byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
                int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
                fixed (byte *__ptr_name = __bytes_name)
                {
                    _UnderlyingPtr = __MR_PointsLoad_NamedCloud_ConstructFrom(__ptr_name, __ptr_name + __len_name, cloud.PassByMode, cloud.Value is not null ? cloud.Value._UnderlyingPtr : null, xf, colors.PassByMode, colors.Value is not null ? colors.Value._UnderlyingPtr : null);
                }
            }

            /// Generated from constructor `MR::PointsLoad::NamedCloud::NamedCloud`.
            public unsafe NamedCloud(MR.PointsLoad._ByValue_NamedCloud _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsLoad.NamedCloud._Underlying *_other);
                _UnderlyingPtr = __MR_PointsLoad_NamedCloud_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::PointsLoad::NamedCloud::operator=`.
            public unsafe MR.PointsLoad.NamedCloud Assign(MR.PointsLoad._ByValue_NamedCloud _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_NamedCloud_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PointsLoad.NamedCloud._Underlying *__MR_PointsLoad_NamedCloud_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsLoad.NamedCloud._Underlying *_other);
                return new(__MR_PointsLoad_NamedCloud_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `NamedCloud` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `NamedCloud`/`Const_NamedCloud` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_NamedCloud
        {
            internal readonly Const_NamedCloud? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_NamedCloud() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_NamedCloud(Const_NamedCloud new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_NamedCloud(Const_NamedCloud arg) {return new(arg);}
            public _ByValue_NamedCloud(MR.Misc._Moved<NamedCloud> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_NamedCloud(MR.Misc._Moved<NamedCloud> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `NamedCloud` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_NamedCloud`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `NamedCloud`/`Const_NamedCloud` directly.
        public class _InOptMut_NamedCloud
        {
            public NamedCloud? Opt;

            public _InOptMut_NamedCloud() {}
            public _InOptMut_NamedCloud(NamedCloud value) {Opt = value;}
            public static implicit operator _InOptMut_NamedCloud(NamedCloud value) {return new(value);}
        }

        /// This is used for optional parameters of class `NamedCloud` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_NamedCloud`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `NamedCloud`/`Const_NamedCloud` to pass it to the function.
        public class _InOptConst_NamedCloud
        {
            public Const_NamedCloud? Opt;

            public _InOptConst_NamedCloud() {}
            public _InOptConst_NamedCloud(Const_NamedCloud value) {Opt = value;}
            public static implicit operator _InOptConst_NamedCloud(Const_NamedCloud value) {return new(value);}
        }

        /// Generated from function `MR::PointsLoad::fromSceneE57File`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRPointsLoadNamedCloud_StdString> FromSceneE57File(ReadOnlySpan<char> file, MR.PointsLoad.Const_E57LoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromSceneE57File", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMRPointsLoadNamedCloud_StdString._Underlying *__MR_PointsLoad_fromSceneE57File(byte *file, byte *file_end, MR.PointsLoad.Const_E57LoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_StdVectorMRPointsLoadNamedCloud_StdString(__MR_PointsLoad_fromSceneE57File(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// loads from .e57 file
        /// Generated from function `MR::PointsLoad::fromE57`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromE57(ReadOnlySpan<char> file, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromE57_std_filesystem_path", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromE57_std_filesystem_path(byte *file, byte *file_end, MR.Const_PointsLoadSettings._Underlying *settings);
            byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
            int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
            fixed (byte *__ptr_file = __bytes_file)
            {
                return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromE57_std_filesystem_path(__ptr_file, __ptr_file + __len_file, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
            }
        }

        /// Generated from function `MR::PointsLoad::fromE57`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRPointCloud_StdString> FromE57(MR.Std.Istream in_, MR.Const_PointsLoadSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_fromE57_std_istream", ExactSpelling = true)]
            extern static MR.Expected_MRPointCloud_StdString._Underlying *__MR_PointsLoad_fromE57_std_istream(MR.Std.Istream._Underlying *in_, MR.Const_PointsLoadSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_MRPointCloud_StdString(__MR_PointsLoad_fromE57_std_istream(in_._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::PointsLoad::loadObjectFromE57`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjects_StdString> LoadObjectFromE57(ReadOnlySpan<char> path, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoad_loadObjectFromE57", ExactSpelling = true)]
            extern static MR.Expected_MRLoadedObjects_StdString._Underlying *__MR_PointsLoad_loadObjectFromE57(byte *path, byte *path_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_MRLoadedObjects_StdString(__MR_PointsLoad_loadObjectFromE57(__ptr_path, __ptr_path + __len_path, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }
        }
    }
}
