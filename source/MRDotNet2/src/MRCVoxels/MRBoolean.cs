public static partial class MR
{
    // converter of meshes in or from signed distance volumetric representation
    /// Generated from class `MR::MeshVoxelsConverter`.
    /// This is the const half of the class.
    public class Const_MeshVoxelsConverter : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshVoxelsConverter(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshVoxelsConverter_Destroy(_Underlying *_this);
            __MR_MeshVoxelsConverter_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshVoxelsConverter() {Dispose(false);}

        // both in and from
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_Get_voxelSize(_Underlying *_this);
                return *__MR_MeshVoxelsConverter_Get_voxelSize(_UnderlyingPtr);
            }
        }

        // number voxels around surface to calculate distance in (should be positive)
        public unsafe float SurfaceOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_Get_surfaceOffset", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_Get_surfaceOffset(_Underlying *_this);
                return *__MR_MeshVoxelsConverter_Get_surfaceOffset(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_Get_callBack", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MeshVoxelsConverter_Get_callBack(_Underlying *_this);
                return new(__MR_MeshVoxelsConverter_Get_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        // the value is in voxels (not in meters!), 0 for no-offset
        public unsafe float OffsetVoxels
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_Get_offsetVoxels", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_Get_offsetVoxels(_Underlying *_this);
                return *__MR_MeshVoxelsConverter_Get_offsetVoxels(_UnderlyingPtr);
            }
        }

        // [0, 1] ratio of combining small triangles into bigger ones
        public unsafe float Adaptivity
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_Get_adaptivity", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_Get_adaptivity(_Underlying *_this);
                return *__MR_MeshVoxelsConverter_Get_adaptivity(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshVoxelsConverter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshVoxelsConverter_DefaultConstruct();
        }

        /// Constructs `MR::MeshVoxelsConverter` elementwise.
        public unsafe Const_MeshVoxelsConverter(float voxelSize, float surfaceOffset, MR.Std._ByValue_Function_BoolFuncFromFloat callBack, float offsetVoxels, float adaptivity) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_ConstructFrom(float voxelSize, float surfaceOffset, MR.Misc._PassBy callBack_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callBack, float offsetVoxels, float adaptivity);
            _UnderlyingPtr = __MR_MeshVoxelsConverter_ConstructFrom(voxelSize, surfaceOffset, callBack.PassByMode, callBack.Value is not null ? callBack.Value._UnderlyingPtr : null, offsetVoxels, adaptivity);
        }

        /// Generated from constructor `MR::MeshVoxelsConverter::MeshVoxelsConverter`.
        public unsafe Const_MeshVoxelsConverter(MR._ByValue_MeshVoxelsConverter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshVoxelsConverter._Underlying *_other);
            _UnderlyingPtr = __MR_MeshVoxelsConverter_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshVoxelsConverter::operator()`.
        /// Parameter `xf` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.FloatGrid> Call(MR.Const_MeshPart mp, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_call_2", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_MeshVoxelsConverter_call_2(_Underlying *_this, MR.Const_MeshPart._Underlying *mp, MR.Const_AffineXf3f._Underlying *xf);
            return MR.Misc.Move(new MR.FloatGrid(__MR_MeshVoxelsConverter_call_2(_UnderlyingPtr, mp._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from method `MR::MeshVoxelsConverter::operator()`.
        public unsafe MR.Misc._Moved<MR.FloatGrid> Call(MR.Const_ObjectMesh obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_call_1_MR_ObjectMesh", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_MeshVoxelsConverter_call_1_MR_ObjectMesh(_Underlying *_this, MR.Const_ObjectMesh._Underlying *obj);
            return MR.Misc.Move(new MR.FloatGrid(__MR_MeshVoxelsConverter_call_1_MR_ObjectMesh(_UnderlyingPtr, obj._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::MeshVoxelsConverter::operator()`.
        public unsafe MR.Misc._Moved<MR.Mesh> Call(MR.Const_FloatGrid grid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_call_1_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_MeshVoxelsConverter_call_1_MR_FloatGrid(_Underlying *_this, MR.Const_FloatGrid._Underlying *grid);
            return MR.Misc.Move(new MR.Mesh(__MR_MeshVoxelsConverter_call_1_MR_FloatGrid(_UnderlyingPtr, grid._UnderlyingPtr), is_owning: true));
        }
    }

    // converter of meshes in or from signed distance volumetric representation
    /// Generated from class `MR::MeshVoxelsConverter`.
    /// This is the non-const half of the class.
    public class MeshVoxelsConverter : Const_MeshVoxelsConverter
    {
        internal unsafe MeshVoxelsConverter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // both in and from
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_MeshVoxelsConverter_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        // number voxels around surface to calculate distance in (should be positive)
        public new unsafe ref float SurfaceOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_GetMutable_surfaceOffset", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_GetMutable_surfaceOffset(_Underlying *_this);
                return ref *__MR_MeshVoxelsConverter_GetMutable_surfaceOffset(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_GetMutable_callBack", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MeshVoxelsConverter_GetMutable_callBack(_Underlying *_this);
                return new(__MR_MeshVoxelsConverter_GetMutable_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        // the value is in voxels (not in meters!), 0 for no-offset
        public new unsafe ref float OffsetVoxels
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_GetMutable_offsetVoxels", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_GetMutable_offsetVoxels(_Underlying *_this);
                return ref *__MR_MeshVoxelsConverter_GetMutable_offsetVoxels(_UnderlyingPtr);
            }
        }

        // [0, 1] ratio of combining small triangles into bigger ones
        public new unsafe ref float Adaptivity
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_GetMutable_adaptivity", ExactSpelling = true)]
                extern static float *__MR_MeshVoxelsConverter_GetMutable_adaptivity(_Underlying *_this);
                return ref *__MR_MeshVoxelsConverter_GetMutable_adaptivity(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshVoxelsConverter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshVoxelsConverter_DefaultConstruct();
        }

        /// Constructs `MR::MeshVoxelsConverter` elementwise.
        public unsafe MeshVoxelsConverter(float voxelSize, float surfaceOffset, MR.Std._ByValue_Function_BoolFuncFromFloat callBack, float offsetVoxels, float adaptivity) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_ConstructFrom(float voxelSize, float surfaceOffset, MR.Misc._PassBy callBack_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callBack, float offsetVoxels, float adaptivity);
            _UnderlyingPtr = __MR_MeshVoxelsConverter_ConstructFrom(voxelSize, surfaceOffset, callBack.PassByMode, callBack.Value is not null ? callBack.Value._UnderlyingPtr : null, offsetVoxels, adaptivity);
        }

        /// Generated from constructor `MR::MeshVoxelsConverter::MeshVoxelsConverter`.
        public unsafe MeshVoxelsConverter(MR._ByValue_MeshVoxelsConverter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshVoxelsConverter._Underlying *_other);
            _UnderlyingPtr = __MR_MeshVoxelsConverter_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshVoxelsConverter::operator=`.
        public unsafe MR.MeshVoxelsConverter Assign(MR._ByValue_MeshVoxelsConverter _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVoxelsConverter_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshVoxelsConverter._Underlying *__MR_MeshVoxelsConverter_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshVoxelsConverter._Underlying *_other);
            return new(__MR_MeshVoxelsConverter_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshVoxelsConverter` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshVoxelsConverter`/`Const_MeshVoxelsConverter` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshVoxelsConverter
    {
        internal readonly Const_MeshVoxelsConverter? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshVoxelsConverter() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshVoxelsConverter(Const_MeshVoxelsConverter new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshVoxelsConverter(Const_MeshVoxelsConverter arg) {return new(arg);}
        public _ByValue_MeshVoxelsConverter(MR.Misc._Moved<MeshVoxelsConverter> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshVoxelsConverter(MR.Misc._Moved<MeshVoxelsConverter> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshVoxelsConverter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshVoxelsConverter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshVoxelsConverter`/`Const_MeshVoxelsConverter` directly.
    public class _InOptMut_MeshVoxelsConverter
    {
        public MeshVoxelsConverter? Opt;

        public _InOptMut_MeshVoxelsConverter() {}
        public _InOptMut_MeshVoxelsConverter(MeshVoxelsConverter value) {Opt = value;}
        public static implicit operator _InOptMut_MeshVoxelsConverter(MeshVoxelsConverter value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshVoxelsConverter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshVoxelsConverter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshVoxelsConverter`/`Const_MeshVoxelsConverter` to pass it to the function.
    public class _InOptConst_MeshVoxelsConverter
    {
        public Const_MeshVoxelsConverter? Opt;

        public _InOptConst_MeshVoxelsConverter() {}
        public _InOptConst_MeshVoxelsConverter(Const_MeshVoxelsConverter value) {Opt = value;}
        public static implicit operator _InOptConst_MeshVoxelsConverter(Const_MeshVoxelsConverter value) {return new(value);}
    }
}
