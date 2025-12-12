public static partial class MR
{
    /**
    * \brief Class for making mesh from regular distance map
    * \details distance for each lattice of map is defined as 
    * point on ray from surface point cloud to direction point cloud,
    * with length equal to 1/distance from distances file 
    * (if distance in file == 0 ray had no point) 
    *
    */
    /// Generated from class `MR::RegularMapMesher`.
    /// This is the const half of the class.
    public class Const_RegularMapMesher : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RegularMapMesher(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_Destroy", ExactSpelling = true)]
            extern static void __MR_RegularMapMesher_Destroy(_Underlying *_this);
            __MR_RegularMapMesher_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RegularMapMesher() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RegularMapMesher() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegularMapMesher._Underlying *__MR_RegularMapMesher_DefaultConstruct();
            _UnderlyingPtr = __MR_RegularMapMesher_DefaultConstruct();
        }

        /// Generated from constructor `MR::RegularMapMesher::RegularMapMesher`.
        public unsafe Const_RegularMapMesher(MR._ByValue_RegularMapMesher _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RegularMapMesher._Underlying *__MR_RegularMapMesher_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RegularMapMesher._Underlying *_other);
            _UnderlyingPtr = __MR_RegularMapMesher_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Creates mesh if all components were successfully loaded
        /// Generated from method `MR::RegularMapMesher::createMesh`.
        public unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> CreateMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_createMesh", ExactSpelling = true)]
            extern static MR.Expected_MRMesh_StdString._Underlying *__MR_RegularMapMesher_createMesh(_Underlying *_this);
            return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_RegularMapMesher_createMesh(_UnderlyingPtr), is_owning: true));
        }
    }

    /**
    * \brief Class for making mesh from regular distance map
    * \details distance for each lattice of map is defined as 
    * point on ray from surface point cloud to direction point cloud,
    * with length equal to 1/distance from distances file 
    * (if distance in file == 0 ray had no point) 
    *
    */
    /// Generated from class `MR::RegularMapMesher`.
    /// This is the non-const half of the class.
    public class RegularMapMesher : Const_RegularMapMesher
    {
        internal unsafe RegularMapMesher(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe RegularMapMesher() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegularMapMesher._Underlying *__MR_RegularMapMesher_DefaultConstruct();
            _UnderlyingPtr = __MR_RegularMapMesher_DefaultConstruct();
        }

        /// Generated from constructor `MR::RegularMapMesher::RegularMapMesher`.
        public unsafe RegularMapMesher(MR._ByValue_RegularMapMesher _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RegularMapMesher._Underlying *__MR_RegularMapMesher_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RegularMapMesher._Underlying *_other);
            _UnderlyingPtr = __MR_RegularMapMesher_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::RegularMapMesher::operator=`.
        public unsafe MR.RegularMapMesher Assign(MR._ByValue_RegularMapMesher _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RegularMapMesher._Underlying *__MR_RegularMapMesher_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RegularMapMesher._Underlying *_other);
            return new(__MR_RegularMapMesher_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Sets surface Point Cloud
        /// Generated from method `MR::RegularMapMesher::setSurfacePC`.
        public unsafe void SetSurfacePC(MR.Const_PointCloud surfacePC)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_setSurfacePC", ExactSpelling = true)]
            extern static void __MR_RegularMapMesher_setSurfacePC(_Underlying *_this, MR.Const_PointCloud._UnderlyingShared *surfacePC);
            __MR_RegularMapMesher_setSurfacePC(_UnderlyingPtr, surfacePC._UnderlyingSharedPtr);
        }

        /// Sets directions Point Cloud
        /// Generated from method `MR::RegularMapMesher::setDirectionsPC`.
        public unsafe void SetDirectionsPC(MR.Const_PointCloud directionsPC)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_setDirectionsPC", ExactSpelling = true)]
            extern static void __MR_RegularMapMesher_setDirectionsPC(_Underlying *_this, MR.Const_PointCloud._UnderlyingShared *directionsPC);
            __MR_RegularMapMesher_setDirectionsPC(_UnderlyingPtr, directionsPC._UnderlyingSharedPtr);
        }

        /// Loads distances form distances file (1/distance)
        /// Generated from method `MR::RegularMapMesher::loadDistances`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> LoadDistances(int width, int height, ReadOnlySpan<char> path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_loadDistances", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_RegularMapMesher_loadDistances(_Underlying *_this, int width, int height, byte *path, byte *path_end);
            byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
            int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
            fixed (byte *__ptr_path = __bytes_path)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_RegularMapMesher_loadDistances(_UnderlyingPtr, width, height, __ptr_path, __ptr_path + __len_path), is_owning: true));
            }
        }

        /// Sets distances
        /// Generated from method `MR::RegularMapMesher::setDistances`.
        public unsafe void SetDistances(int width, int height, MR.Std.Const_Vector_Float distances)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegularMapMesher_setDistances", ExactSpelling = true)]
            extern static void __MR_RegularMapMesher_setDistances(_Underlying *_this, int width, int height, MR.Std.Const_Vector_Float._Underlying *distances);
            __MR_RegularMapMesher_setDistances(_UnderlyingPtr, width, height, distances._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `RegularMapMesher` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RegularMapMesher`/`Const_RegularMapMesher` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RegularMapMesher
    {
        internal readonly Const_RegularMapMesher? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RegularMapMesher() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RegularMapMesher(Const_RegularMapMesher new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_RegularMapMesher(Const_RegularMapMesher arg) {return new(arg);}
        public _ByValue_RegularMapMesher(MR.Misc._Moved<RegularMapMesher> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RegularMapMesher(MR.Misc._Moved<RegularMapMesher> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `RegularMapMesher` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RegularMapMesher`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RegularMapMesher`/`Const_RegularMapMesher` directly.
    public class _InOptMut_RegularMapMesher
    {
        public RegularMapMesher? Opt;

        public _InOptMut_RegularMapMesher() {}
        public _InOptMut_RegularMapMesher(RegularMapMesher value) {Opt = value;}
        public static implicit operator _InOptMut_RegularMapMesher(RegularMapMesher value) {return new(value);}
    }

    /// This is used for optional parameters of class `RegularMapMesher` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RegularMapMesher`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RegularMapMesher`/`Const_RegularMapMesher` to pass it to the function.
    public class _InOptConst_RegularMapMesher
    {
        public Const_RegularMapMesher? Opt;

        public _InOptConst_RegularMapMesher() {}
        public _InOptConst_RegularMapMesher(Const_RegularMapMesher value) {Opt = value;}
        public static implicit operator _InOptConst_RegularMapMesher(Const_RegularMapMesher value) {return new(value);}
    }
}
