public static partial class MR
{
    /// represents a box in 3D space subdivided on voxels stored in T;
    /// and stores minimum and maximum values among all valid voxels
    /// Generated from class `MR::SimpleVolumeMinMax`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolume`
    ///     `MR::Box1f`
    /// This is the const half of the class.
    public class Const_SimpleVolumeMinMax : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box1f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Destroy", ExactSpelling = true)]
            extern static void __MR_SimpleVolumeMinMax_Destroy(_Underlying *_this);
            __MR_SimpleVolumeMinMax_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SimpleVolumeMinMax() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_SimpleVolume(Const_SimpleVolumeMinMax self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_UpcastTo_MR_SimpleVolume", ExactSpelling = true)]
            extern static MR.Const_SimpleVolume._Underlying *__MR_SimpleVolumeMinMax_UpcastTo_MR_SimpleVolume(_Underlying *_this);
            MR.Const_SimpleVolume ret = new(__MR_SimpleVolumeMinMax_UpcastTo_MR_SimpleVolume(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_Box1f(Const_SimpleVolumeMinMax self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_UpcastTo_MR_Box1f", ExactSpelling = true)]
            extern static MR.Const_Box1f._Underlying *__MR_SimpleVolumeMinMax_UpcastTo_MR_Box1f(_Underlying *_this);
            MR.Const_Box1f ret = new(__MR_SimpleVolumeMinMax_UpcastTo_MR_Box1f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.Const_Vector_Float_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Get_data", ExactSpelling = true)]
                extern static MR.Const_Vector_Float_MRVoxelId._Underlying *__MR_SimpleVolumeMinMax_Get_data(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMax_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_SimpleVolumeMinMax_Get_dims(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMax_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_SimpleVolumeMinMax_Get_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMax_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Get_elements", ExactSpelling = true)]
                extern static int *__MR_SimpleVolumeMinMax_Get_elements();
                return *__MR_SimpleVolumeMinMax_Get_elements();
            }
        }

        public unsafe float Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Get_min", ExactSpelling = true)]
                extern static float *__MR_SimpleVolumeMinMax_Get_min(_Underlying *_this);
                return *__MR_SimpleVolumeMinMax_Get_min(_UnderlyingPtr);
            }
        }

        public unsafe float Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_Get_max", ExactSpelling = true)]
                extern static float *__MR_SimpleVolumeMinMax_Get_max(_Underlying *_this);
                return *__MR_SimpleVolumeMinMax_Get_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SimpleVolumeMinMax() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMax._Underlying *__MR_SimpleVolumeMinMax_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolumeMinMax_DefaultConstruct();
        }

        /// Generated from constructor `MR::SimpleVolumeMinMax::SimpleVolumeMinMax`.
        public unsafe Const_SimpleVolumeMinMax(MR._ByValue_SimpleVolumeMinMax _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMax._Underlying *__MR_SimpleVolumeMinMax_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeMinMax._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolumeMinMax_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolumeMinMax::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_SimpleVolumeMinMax_heapBytes(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::SimpleVolumeMinMax::fromMinAndSize`.
        public static unsafe MR.Box1f FromMinAndSize(float min, float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1f __MR_SimpleVolumeMinMax_fromMinAndSize(float *min, float *size);
            return __MR_SimpleVolumeMinMax_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::SimpleVolumeMinMax::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_valid", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMax_valid(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::SimpleVolumeMinMax::center`.
        public unsafe float Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_center", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_center(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::SimpleVolumeMinMax::corner`.
        public unsafe float Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_corner", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_corner(_Underlying *_this, bool *c);
            return __MR_SimpleVolumeMinMax_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::SimpleVolumeMinMax::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMax_getMinBoxCorner(float *n);
            return __MR_SimpleVolumeMinMax_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::SimpleVolumeMinMax::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMax_getMaxBoxCorner(float *n);
            return __MR_SimpleVolumeMinMax_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::SimpleVolumeMinMax::size`.
        public unsafe float Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_size", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_size(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::SimpleVolumeMinMax::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_diagonal", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_diagonal(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::SimpleVolumeMinMax::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_volume", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_volume(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_volume(_UnderlyingPtr);
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::SimpleVolumeMinMax::getBoxClosestPointTo`.
        public unsafe float GetBoxClosestPointTo(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_getBoxClosestPointTo", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_getBoxClosestPointTo(_Underlying *_this, float *pt);
            return __MR_SimpleVolumeMinMax_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::SimpleVolumeMinMax::intersects`.
        public unsafe bool Intersects(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_intersects", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMax_intersects(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_SimpleVolumeMinMax_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::SimpleVolumeMinMax::intersection`.
        public unsafe MR.Box1f Intersection(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_intersection", ExactSpelling = true)]
            extern static MR.Box1f __MR_SimpleVolumeMinMax_intersection(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_SimpleVolumeMinMax_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::SimpleVolumeMinMax::getProjection`.
        public unsafe float GetProjection(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_getProjection", ExactSpelling = true)]
            extern static float __MR_SimpleVolumeMinMax_getProjection(_Underlying *_this, float *pt);
            return __MR_SimpleVolumeMinMax_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::SimpleVolumeMinMax::expanded`.
        public unsafe MR.Box1f Expanded(float expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_expanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_SimpleVolumeMinMax_expanded(_Underlying *_this, float *expansion);
            return __MR_SimpleVolumeMinMax_expanded(_UnderlyingPtr, &expansion);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::SimpleVolumeMinMax::insignificantlyExpanded`.
        public unsafe MR.Box1f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_SimpleVolumeMinMax_insignificantlyExpanded(_Underlying *_this);
            return __MR_SimpleVolumeMinMax_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::SimpleVolumeMinMax::operator==`.
        public static unsafe bool operator==(MR.Const_SimpleVolumeMinMax _this, MR.Const_Box1f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SimpleVolumeMinMax_MR_Box1f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SimpleVolumeMinMax_MR_Box1f(MR.Const_SimpleVolumeMinMax._Underlying *_this, MR.Const_Box1f._Underlying *a);
            return __MR_equal_MR_SimpleVolumeMinMax_MR_Box1f(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SimpleVolumeMinMax _this, MR.Const_Box1f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box1f? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box1f)
                return this == (MR.Const_Box1f)other;
            return false;
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T;
    /// and stores minimum and maximum values among all valid voxels
    /// Generated from class `MR::SimpleVolumeMinMax`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolume`
    ///     `MR::Box1f`
    /// This is the non-const half of the class.
    public class SimpleVolumeMinMax : Const_SimpleVolumeMinMax
    {
        internal unsafe SimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.SimpleVolume(SimpleVolumeMinMax self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_UpcastTo_MR_SimpleVolume", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolumeMinMax_UpcastTo_MR_SimpleVolume(_Underlying *_this);
            MR.SimpleVolume ret = new(__MR_SimpleVolumeMinMax_UpcastTo_MR_SimpleVolume(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Mut_Box1f(SimpleVolumeMinMax self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_UpcastTo_MR_Box1f", ExactSpelling = true)]
            extern static MR.Mut_Box1f._Underlying *__MR_SimpleVolumeMinMax_UpcastTo_MR_Box1f(_Underlying *_this);
            MR.Mut_Box1f ret = new(__MR_SimpleVolumeMinMax_UpcastTo_MR_Box1f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe MR.Vector_Float_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_GetMutable_data", ExactSpelling = true)]
                extern static MR.Vector_Float_MRVoxelId._Underlying *__MR_SimpleVolumeMinMax_GetMutable_data(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMax_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_SimpleVolumeMinMax_GetMutable_dims(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMax_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_SimpleVolumeMinMax_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMax_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_GetMutable_min", ExactSpelling = true)]
                extern static float *__MR_SimpleVolumeMinMax_GetMutable_min(_Underlying *_this);
                return ref *__MR_SimpleVolumeMinMax_GetMutable_min(_UnderlyingPtr);
            }
        }

        public new unsafe ref float Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_GetMutable_max", ExactSpelling = true)]
                extern static float *__MR_SimpleVolumeMinMax_GetMutable_max(_Underlying *_this);
                return ref *__MR_SimpleVolumeMinMax_GetMutable_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SimpleVolumeMinMax() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMax._Underlying *__MR_SimpleVolumeMinMax_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolumeMinMax_DefaultConstruct();
        }

        /// Generated from constructor `MR::SimpleVolumeMinMax::SimpleVolumeMinMax`.
        public unsafe SimpleVolumeMinMax(MR._ByValue_SimpleVolumeMinMax _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMax._Underlying *__MR_SimpleVolumeMinMax_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeMinMax._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolumeMinMax_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolumeMinMax::operator=`.
        public unsafe MR.SimpleVolumeMinMax Assign(MR._ByValue_SimpleVolumeMinMax _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMax._Underlying *__MR_SimpleVolumeMinMax_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeMinMax._Underlying *_other);
            return new(__MR_SimpleVolumeMinMax_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::SimpleVolumeMinMax::intersect`.
        public unsafe MR.Mut_Box1f Intersect(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMax_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1f._Underlying *__MR_SimpleVolumeMinMax_intersect(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return new(__MR_SimpleVolumeMinMax_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SimpleVolumeMinMax` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SimpleVolumeMinMax`/`Const_SimpleVolumeMinMax` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SimpleVolumeMinMax
    {
        internal readonly Const_SimpleVolumeMinMax? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SimpleVolumeMinMax() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SimpleVolumeMinMax(Const_SimpleVolumeMinMax new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SimpleVolumeMinMax(Const_SimpleVolumeMinMax arg) {return new(arg);}
        public _ByValue_SimpleVolumeMinMax(MR.Misc._Moved<SimpleVolumeMinMax> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SimpleVolumeMinMax(MR.Misc._Moved<SimpleVolumeMinMax> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SimpleVolumeMinMax` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolumeMinMax`/`Const_SimpleVolumeMinMax` directly.
    public class _InOptMut_SimpleVolumeMinMax
    {
        public SimpleVolumeMinMax? Opt;

        public _InOptMut_SimpleVolumeMinMax() {}
        public _InOptMut_SimpleVolumeMinMax(SimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptMut_SimpleVolumeMinMax(SimpleVolumeMinMax value) {return new(value);}
    }

    /// This is used for optional parameters of class `SimpleVolumeMinMax` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolumeMinMax`/`Const_SimpleVolumeMinMax` to pass it to the function.
    public class _InOptConst_SimpleVolumeMinMax
    {
        public Const_SimpleVolumeMinMax? Opt;

        public _InOptConst_SimpleVolumeMinMax() {}
        public _InOptConst_SimpleVolumeMinMax(Const_SimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptConst_SimpleVolumeMinMax(Const_SimpleVolumeMinMax value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T;
    /// and stores minimum and maximum values among all valid voxels
    /// Generated from class `MR::SimpleVolumeMinMaxU16`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeU16`
    ///     `MR::Box<unsigned short>`
    /// This is the const half of the class.
    public class Const_SimpleVolumeMinMaxU16 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box_UnsignedShort>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SimpleVolumeMinMaxU16(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Destroy", ExactSpelling = true)]
            extern static void __MR_SimpleVolumeMinMaxU16_Destroy(_Underlying *_this);
            __MR_SimpleVolumeMinMaxU16_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SimpleVolumeMinMaxU16() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_SimpleVolumeU16(Const_SimpleVolumeMinMaxU16 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_UpcastTo_MR_SimpleVolumeU16", ExactSpelling = true)]
            extern static MR.Const_SimpleVolumeU16._Underlying *__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_SimpleVolumeU16(_Underlying *_this);
            MR.Const_SimpleVolumeU16 ret = new(__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_SimpleVolumeU16(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_Box_UnsignedShort(Const_SimpleVolumeMinMaxU16 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_UpcastTo_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static MR.Const_Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_Box_unsigned_short(_Underlying *_this);
            MR.Const_Box_UnsignedShort ret = new(__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_Box_unsigned_short(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.Const_Vector_UnsignedShort_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Get_data", ExactSpelling = true)]
                extern static MR.Const_Vector_UnsignedShort_MRVoxelId._Underlying *__MR_SimpleVolumeMinMaxU16_Get_data(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMaxU16_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_SimpleVolumeMinMaxU16_Get_dims(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMaxU16_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_SimpleVolumeMinMaxU16_Get_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMaxU16_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Get_elements", ExactSpelling = true)]
                extern static int *__MR_SimpleVolumeMinMaxU16_Get_elements();
                return *__MR_SimpleVolumeMinMaxU16_Get_elements();
            }
        }

        public unsafe ushort Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Get_min", ExactSpelling = true)]
                extern static ushort *__MR_SimpleVolumeMinMaxU16_Get_min(_Underlying *_this);
                return *__MR_SimpleVolumeMinMaxU16_Get_min(_UnderlyingPtr);
            }
        }

        public unsafe ushort Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_Get_max", ExactSpelling = true)]
                extern static ushort *__MR_SimpleVolumeMinMaxU16_Get_max(_Underlying *_this);
                return *__MR_SimpleVolumeMinMaxU16_Get_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SimpleVolumeMinMaxU16() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMaxU16._Underlying *__MR_SimpleVolumeMinMaxU16_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolumeMinMaxU16_DefaultConstruct();
        }

        /// Generated from constructor `MR::SimpleVolumeMinMaxU16::SimpleVolumeMinMaxU16`.
        public unsafe Const_SimpleVolumeMinMaxU16(MR._ByValue_SimpleVolumeMinMaxU16 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMaxU16._Underlying *__MR_SimpleVolumeMinMaxU16_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeMinMaxU16._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolumeMinMaxU16_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolumeMinMaxU16::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_SimpleVolumeMinMaxU16_heapBytes(_Underlying *_this);
            return __MR_SimpleVolumeMinMaxU16_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::SimpleVolumeMinMaxU16::fromMinAndSize`.
        public static unsafe MR.Box_UnsignedShort FromMinAndSize(ushort min, ushort size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_fromMinAndSize(ushort *min, ushort *size);
            return new(__MR_SimpleVolumeMinMaxU16_fromMinAndSize(&min, &size), is_owning: true);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::SimpleVolumeMinMaxU16::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_valid", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMaxU16_valid(_Underlying *_this);
            return __MR_SimpleVolumeMinMaxU16_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::SimpleVolumeMinMaxU16::center`.
        public unsafe ushort Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_center", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_center(_Underlying *_this);
            return __MR_SimpleVolumeMinMaxU16_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::SimpleVolumeMinMaxU16::corner`.
        public unsafe ushort Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_corner", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_corner(_Underlying *_this, bool *c);
            return __MR_SimpleVolumeMinMaxU16_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::SimpleVolumeMinMaxU16::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(ushort n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMaxU16_getMinBoxCorner(ushort *n);
            return __MR_SimpleVolumeMinMaxU16_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::SimpleVolumeMinMaxU16::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(ushort n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMaxU16_getMaxBoxCorner(ushort *n);
            return __MR_SimpleVolumeMinMaxU16_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::SimpleVolumeMinMaxU16::size`.
        public unsafe ushort Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_size", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_size(_Underlying *_this);
            return __MR_SimpleVolumeMinMaxU16_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::SimpleVolumeMinMaxU16::diagonal`.
        public unsafe ushort Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_diagonal", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_diagonal(_Underlying *_this);
            return __MR_SimpleVolumeMinMaxU16_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::SimpleVolumeMinMaxU16::volume`.
        public unsafe ushort Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_volume", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_volume(_Underlying *_this);
            return __MR_SimpleVolumeMinMaxU16_volume(_UnderlyingPtr);
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::SimpleVolumeMinMaxU16::getBoxClosestPointTo`.
        public unsafe ushort GetBoxClosestPointTo(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_getBoxClosestPointTo", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_getBoxClosestPointTo(_Underlying *_this, ushort *pt);
            return __MR_SimpleVolumeMinMaxU16_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::SimpleVolumeMinMaxU16::intersects`.
        public unsafe bool Intersects(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_intersects", ExactSpelling = true)]
            extern static byte __MR_SimpleVolumeMinMaxU16_intersects(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return __MR_SimpleVolumeMinMaxU16_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::SimpleVolumeMinMaxU16::intersection`.
        public unsafe MR.Box_UnsignedShort Intersection(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_intersection", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_intersection(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return new(__MR_SimpleVolumeMinMaxU16_intersection(_UnderlyingPtr, b._UnderlyingPtr), is_owning: true);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::SimpleVolumeMinMaxU16::getProjection`.
        public unsafe ushort GetProjection(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_getProjection", ExactSpelling = true)]
            extern static ushort __MR_SimpleVolumeMinMaxU16_getProjection(_Underlying *_this, ushort *pt);
            return __MR_SimpleVolumeMinMaxU16_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::SimpleVolumeMinMaxU16::expanded`.
        public unsafe MR.Box_UnsignedShort Expanded(ushort expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_expanded", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_expanded(_Underlying *_this, ushort *expansion);
            return new(__MR_SimpleVolumeMinMaxU16_expanded(_UnderlyingPtr, &expansion), is_owning: true);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::SimpleVolumeMinMaxU16::insignificantlyExpanded`.
        public unsafe MR.Box_UnsignedShort InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_insignificantlyExpanded(_Underlying *_this);
            return new(__MR_SimpleVolumeMinMaxU16_insignificantlyExpanded(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::SimpleVolumeMinMaxU16::operator==`.
        public static unsafe bool operator==(MR.Const_SimpleVolumeMinMaxU16 _this, MR.Const_Box_UnsignedShort a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_SimpleVolumeMinMaxU16_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static byte __MR_equal_MR_SimpleVolumeMinMaxU16_MR_Box_unsigned_short(MR.Const_SimpleVolumeMinMaxU16._Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *a);
            return __MR_equal_MR_SimpleVolumeMinMaxU16_MR_Box_unsigned_short(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_SimpleVolumeMinMaxU16 _this, MR.Const_Box_UnsignedShort a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box_UnsignedShort? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box_UnsignedShort)
                return this == (MR.Const_Box_UnsignedShort)other;
            return false;
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T;
    /// and stores minimum and maximum values among all valid voxels
    /// Generated from class `MR::SimpleVolumeMinMaxU16`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeU16`
    ///     `MR::Box<unsigned short>`
    /// This is the non-const half of the class.
    public class SimpleVolumeMinMaxU16 : Const_SimpleVolumeMinMaxU16
    {
        internal unsafe SimpleVolumeMinMaxU16(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.SimpleVolumeU16(SimpleVolumeMinMaxU16 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_UpcastTo_MR_SimpleVolumeU16", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_SimpleVolumeU16(_Underlying *_this);
            MR.SimpleVolumeU16 ret = new(__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_SimpleVolumeU16(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Box_UnsignedShort(SimpleVolumeMinMaxU16 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_UpcastTo_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_Box_unsigned_short(_Underlying *_this);
            MR.Box_UnsignedShort ret = new(__MR_SimpleVolumeMinMaxU16_UpcastTo_MR_Box_unsigned_short(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe MR.Vector_UnsignedShort_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_GetMutable_data", ExactSpelling = true)]
                extern static MR.Vector_UnsignedShort_MRVoxelId._Underlying *__MR_SimpleVolumeMinMaxU16_GetMutable_data(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMaxU16_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_SimpleVolumeMinMaxU16_GetMutable_dims(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMaxU16_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_SimpleVolumeMinMaxU16_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolumeMinMaxU16_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref ushort Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_GetMutable_min", ExactSpelling = true)]
                extern static ushort *__MR_SimpleVolumeMinMaxU16_GetMutable_min(_Underlying *_this);
                return ref *__MR_SimpleVolumeMinMaxU16_GetMutable_min(_UnderlyingPtr);
            }
        }

        public new unsafe ref ushort Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_GetMutable_max", ExactSpelling = true)]
                extern static ushort *__MR_SimpleVolumeMinMaxU16_GetMutable_max(_Underlying *_this);
                return ref *__MR_SimpleVolumeMinMaxU16_GetMutable_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SimpleVolumeMinMaxU16() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMaxU16._Underlying *__MR_SimpleVolumeMinMaxU16_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolumeMinMaxU16_DefaultConstruct();
        }

        /// Generated from constructor `MR::SimpleVolumeMinMaxU16::SimpleVolumeMinMaxU16`.
        public unsafe SimpleVolumeMinMaxU16(MR._ByValue_SimpleVolumeMinMaxU16 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMaxU16._Underlying *__MR_SimpleVolumeMinMaxU16_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeMinMaxU16._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolumeMinMaxU16_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolumeMinMaxU16::operator=`.
        public unsafe MR.SimpleVolumeMinMaxU16 Assign(MR._ByValue_SimpleVolumeMinMaxU16 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeMinMaxU16._Underlying *__MR_SimpleVolumeMinMaxU16_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeMinMaxU16._Underlying *_other);
            return new(__MR_SimpleVolumeMinMaxU16_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::SimpleVolumeMinMaxU16::intersect`.
        public unsafe MR.Box_UnsignedShort Intersect(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeMinMaxU16_intersect", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_SimpleVolumeMinMaxU16_intersect(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return new(__MR_SimpleVolumeMinMaxU16_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SimpleVolumeMinMaxU16` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SimpleVolumeMinMaxU16`/`Const_SimpleVolumeMinMaxU16` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SimpleVolumeMinMaxU16
    {
        internal readonly Const_SimpleVolumeMinMaxU16? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SimpleVolumeMinMaxU16() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SimpleVolumeMinMaxU16(Const_SimpleVolumeMinMaxU16 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SimpleVolumeMinMaxU16(Const_SimpleVolumeMinMaxU16 arg) {return new(arg);}
        public _ByValue_SimpleVolumeMinMaxU16(MR.Misc._Moved<SimpleVolumeMinMaxU16> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SimpleVolumeMinMaxU16(MR.Misc._Moved<SimpleVolumeMinMaxU16> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SimpleVolumeMinMaxU16` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SimpleVolumeMinMaxU16`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolumeMinMaxU16`/`Const_SimpleVolumeMinMaxU16` directly.
    public class _InOptMut_SimpleVolumeMinMaxU16
    {
        public SimpleVolumeMinMaxU16? Opt;

        public _InOptMut_SimpleVolumeMinMaxU16() {}
        public _InOptMut_SimpleVolumeMinMaxU16(SimpleVolumeMinMaxU16 value) {Opt = value;}
        public static implicit operator _InOptMut_SimpleVolumeMinMaxU16(SimpleVolumeMinMaxU16 value) {return new(value);}
    }

    /// This is used for optional parameters of class `SimpleVolumeMinMaxU16` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SimpleVolumeMinMaxU16`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolumeMinMaxU16`/`Const_SimpleVolumeMinMaxU16` to pass it to the function.
    public class _InOptConst_SimpleVolumeMinMaxU16
    {
        public Const_SimpleVolumeMinMaxU16? Opt;

        public _InOptConst_SimpleVolumeMinMaxU16() {}
        public _InOptConst_SimpleVolumeMinMaxU16(Const_SimpleVolumeMinMaxU16 value) {Opt = value;}
        public static implicit operator _InOptConst_SimpleVolumeMinMaxU16(Const_SimpleVolumeMinMaxU16 value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T;
    /// and stores minimum and maximum values among all valid voxels
    /// Generated from class `MR::VdbVolume`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelsVolume<MR::FloatGrid>`
    ///     `MR::Box1f`
    /// This is the const half of the class.
    public class Const_VdbVolume : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box1f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VdbVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_VdbVolume_Destroy(_Underlying *_this);
            __MR_VdbVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VdbVolume() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_VoxelsVolume_MRFloatGrid(Const_VdbVolume self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.Const_VoxelsVolume_MRFloatGrid._Underlying *__MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid(_Underlying *_this);
            MR.Const_VoxelsVolume_MRFloatGrid ret = new(__MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_Box1f(Const_VdbVolume self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_UpcastTo_MR_Box1f", ExactSpelling = true)]
            extern static MR.Const_Box1f._Underlying *__MR_VdbVolume_UpcastTo_MR_Box1f(_Underlying *_this);
            MR.Const_Box1f ret = new(__MR_VdbVolume_UpcastTo_MR_Box1f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.Const_FloatGrid Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Get_data", ExactSpelling = true)]
                extern static MR.Const_FloatGrid._Underlying *__MR_VdbVolume_Get_data(_Underlying *_this);
                return new(__MR_VdbVolume_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_VdbVolume_Get_dims(_Underlying *_this);
                return new(__MR_VdbVolume_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_VdbVolume_Get_voxelSize(_Underlying *_this);
                return new(__MR_VdbVolume_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Get_elements", ExactSpelling = true)]
                extern static int *__MR_VdbVolume_Get_elements();
                return *__MR_VdbVolume_Get_elements();
            }
        }

        public unsafe float Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Get_min", ExactSpelling = true)]
                extern static float *__MR_VdbVolume_Get_min(_Underlying *_this);
                return *__MR_VdbVolume_Get_min(_UnderlyingPtr);
            }
        }

        public unsafe float Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_Get_max", ExactSpelling = true)]
                extern static float *__MR_VdbVolume_Get_max(_Underlying *_this);
                return *__MR_VdbVolume_Get_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VdbVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_VdbVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_VdbVolume_DefaultConstruct();
        }

        /// Generated from constructor `MR::VdbVolume::VdbVolume`.
        public unsafe Const_VdbVolume(MR._ByValue_VdbVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VdbVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VdbVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VdbVolume::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_VdbVolume_heapBytes(_Underlying *_this);
            return __MR_VdbVolume_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::VdbVolume::fromMinAndSize`.
        public static unsafe MR.Box1f FromMinAndSize(float min, float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1f __MR_VdbVolume_fromMinAndSize(float *min, float *size);
            return __MR_VdbVolume_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::VdbVolume::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_valid", ExactSpelling = true)]
            extern static byte __MR_VdbVolume_valid(_Underlying *_this);
            return __MR_VdbVolume_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::VdbVolume::center`.
        public unsafe float Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_center", ExactSpelling = true)]
            extern static float __MR_VdbVolume_center(_Underlying *_this);
            return __MR_VdbVolume_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::VdbVolume::corner`.
        public unsafe float Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_corner", ExactSpelling = true)]
            extern static float __MR_VdbVolume_corner(_Underlying *_this, bool *c);
            return __MR_VdbVolume_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::VdbVolume::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_VdbVolume_getMinBoxCorner(float *n);
            return __MR_VdbVolume_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::VdbVolume::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_VdbVolume_getMaxBoxCorner(float *n);
            return __MR_VdbVolume_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::VdbVolume::size`.
        public unsafe float Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_size", ExactSpelling = true)]
            extern static float __MR_VdbVolume_size(_Underlying *_this);
            return __MR_VdbVolume_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::VdbVolume::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_diagonal", ExactSpelling = true)]
            extern static float __MR_VdbVolume_diagonal(_Underlying *_this);
            return __MR_VdbVolume_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::VdbVolume::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_volume", ExactSpelling = true)]
            extern static float __MR_VdbVolume_volume(_Underlying *_this);
            return __MR_VdbVolume_volume(_UnderlyingPtr);
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::VdbVolume::getBoxClosestPointTo`.
        public unsafe float GetBoxClosestPointTo(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_getBoxClosestPointTo", ExactSpelling = true)]
            extern static float __MR_VdbVolume_getBoxClosestPointTo(_Underlying *_this, float *pt);
            return __MR_VdbVolume_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::VdbVolume::intersects`.
        public unsafe bool Intersects(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_intersects", ExactSpelling = true)]
            extern static byte __MR_VdbVolume_intersects(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_VdbVolume_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::VdbVolume::intersection`.
        public unsafe MR.Box1f Intersection(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_intersection", ExactSpelling = true)]
            extern static MR.Box1f __MR_VdbVolume_intersection(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_VdbVolume_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::VdbVolume::getProjection`.
        public unsafe float GetProjection(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_getProjection", ExactSpelling = true)]
            extern static float __MR_VdbVolume_getProjection(_Underlying *_this, float *pt);
            return __MR_VdbVolume_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::VdbVolume::expanded`.
        public unsafe MR.Box1f Expanded(float expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_expanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_VdbVolume_expanded(_Underlying *_this, float *expansion);
            return __MR_VdbVolume_expanded(_UnderlyingPtr, &expansion);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::VdbVolume::insignificantlyExpanded`.
        public unsafe MR.Box1f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_VdbVolume_insignificantlyExpanded(_Underlying *_this);
            return __MR_VdbVolume_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::VdbVolume::operator==`.
        public static unsafe bool operator==(MR.Const_VdbVolume _this, MR.Const_Box1f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VdbVolume_MR_Box1f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VdbVolume_MR_Box1f(MR.Const_VdbVolume._Underlying *_this, MR.Const_Box1f._Underlying *a);
            return __MR_equal_MR_VdbVolume_MR_Box1f(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_VdbVolume _this, MR.Const_Box1f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box1f? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box1f)
                return this == (MR.Const_Box1f)other;
            return false;
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T;
    /// and stores minimum and maximum values among all valid voxels
    /// Generated from class `MR::VdbVolume`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelsVolume<MR::FloatGrid>`
    ///     `MR::Box1f`
    /// This is the non-const half of the class.
    public class VdbVolume : Const_VdbVolume
    {
        internal unsafe VdbVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.VoxelsVolume_MRFloatGrid(VdbVolume self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid(_Underlying *_this);
            MR.VoxelsVolume_MRFloatGrid ret = new(__MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Mut_Box1f(VdbVolume self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_UpcastTo_MR_Box1f", ExactSpelling = true)]
            extern static MR.Mut_Box1f._Underlying *__MR_VdbVolume_UpcastTo_MR_Box1f(_Underlying *_this);
            MR.Mut_Box1f ret = new(__MR_VdbVolume_UpcastTo_MR_Box1f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe MR.FloatGrid Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_GetMutable_data", ExactSpelling = true)]
                extern static MR.FloatGrid._Underlying *__MR_VdbVolume_GetMutable_data(_Underlying *_this);
                return new(__MR_VdbVolume_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_VdbVolume_GetMutable_dims(_Underlying *_this);
                return new(__MR_VdbVolume_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_VdbVolume_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_VdbVolume_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_GetMutable_min", ExactSpelling = true)]
                extern static float *__MR_VdbVolume_GetMutable_min(_Underlying *_this);
                return ref *__MR_VdbVolume_GetMutable_min(_UnderlyingPtr);
            }
        }

        public new unsafe ref float Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_GetMutable_max", ExactSpelling = true)]
                extern static float *__MR_VdbVolume_GetMutable_max(_Underlying *_this);
                return ref *__MR_VdbVolume_GetMutable_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VdbVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_VdbVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_VdbVolume_DefaultConstruct();
        }

        /// Generated from constructor `MR::VdbVolume::VdbVolume`.
        public unsafe VdbVolume(MR._ByValue_VdbVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VdbVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VdbVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VdbVolume::operator=`.
        public unsafe MR.VdbVolume Assign(MR._ByValue_VdbVolume _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_VdbVolume_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VdbVolume._Underlying *_other);
            return new(__MR_VdbVolume_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::VdbVolume::intersect`.
        public unsafe MR.Mut_Box1f Intersect(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VdbVolume_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1f._Underlying *__MR_VdbVolume_intersect(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return new(__MR_VdbVolume_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VdbVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VdbVolume`/`Const_VdbVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VdbVolume
    {
        internal readonly Const_VdbVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VdbVolume() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VdbVolume(Const_VdbVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VdbVolume(Const_VdbVolume arg) {return new(arg);}
        public _ByValue_VdbVolume(MR.Misc._Moved<VdbVolume> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VdbVolume(MR.Misc._Moved<VdbVolume> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VdbVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VdbVolume`/`Const_VdbVolume` directly.
    public class _InOptMut_VdbVolume
    {
        public VdbVolume? Opt;

        public _InOptMut_VdbVolume() {}
        public _InOptMut_VdbVolume(VdbVolume value) {Opt = value;}
        public static implicit operator _InOptMut_VdbVolume(VdbVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `VdbVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VdbVolume`/`Const_VdbVolume` to pass it to the function.
    public class _InOptConst_VdbVolume
    {
        public Const_VdbVolume? Opt;

        public _InOptConst_VdbVolume() {}
        public _InOptConst_VdbVolume(Const_VdbVolume value) {Opt = value;}
        public static implicit operator _InOptConst_VdbVolume(Const_VdbVolume value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::FunctionVolume`.
    /// This is the const half of the class.
    public class Const_FunctionVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FunctionVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_FunctionVolume_Destroy(_Underlying *_this);
            __MR_FunctionVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FunctionVolume() {Dispose(false);}

        public unsafe MR.Std.Const_Function_FloatFuncFromConstMRVector3iRef Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_Get_data", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromConstMRVector3iRef._Underlying *__MR_FunctionVolume_Get_data(_Underlying *_this);
                return new(__MR_FunctionVolume_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_FunctionVolume_Get_dims(_Underlying *_this);
                return new(__MR_FunctionVolume_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_FunctionVolume_Get_voxelSize(_Underlying *_this);
                return new(__MR_FunctionVolume_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FunctionVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_FunctionVolume_DefaultConstruct();
        }

        /// Constructs `MR::FunctionVolume` elementwise.
        public unsafe Const_FunctionVolume(MR.Std._ByValue_Function_FloatFuncFromConstMRVector3iRef data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_ConstructFrom", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Std.Function_FloatFuncFromConstMRVector3iRef._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_FunctionVolume_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::FunctionVolume::FunctionVolume`.
        public unsafe Const_FunctionVolume(MR._ByValue_FunctionVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FunctionVolume._Underlying *_other);
            _UnderlyingPtr = __MR_FunctionVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FunctionVolume::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_FunctionVolume_heapBytes(_Underlying *_this);
            return __MR_FunctionVolume_heapBytes(_UnderlyingPtr);
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::FunctionVolume`.
    /// This is the non-const half of the class.
    public class FunctionVolume : Const_FunctionVolume
    {
        internal unsafe FunctionVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_FloatFuncFromConstMRVector3iRef Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_GetMutable_data", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromConstMRVector3iRef._Underlying *__MR_FunctionVolume_GetMutable_data(_Underlying *_this);
                return new(__MR_FunctionVolume_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_FunctionVolume_GetMutable_dims(_Underlying *_this);
                return new(__MR_FunctionVolume_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_FunctionVolume_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_FunctionVolume_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FunctionVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_FunctionVolume_DefaultConstruct();
        }

        /// Constructs `MR::FunctionVolume` elementwise.
        public unsafe FunctionVolume(MR.Std._ByValue_Function_FloatFuncFromConstMRVector3iRef data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_ConstructFrom", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Std.Function_FloatFuncFromConstMRVector3iRef._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_FunctionVolume_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::FunctionVolume::FunctionVolume`.
        public unsafe FunctionVolume(MR._ByValue_FunctionVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FunctionVolume._Underlying *_other);
            _UnderlyingPtr = __MR_FunctionVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FunctionVolume::operator=`.
        public unsafe MR.FunctionVolume Assign(MR._ByValue_FunctionVolume _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolume_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FunctionVolume._Underlying *__MR_FunctionVolume_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FunctionVolume._Underlying *_other);
            return new(__MR_FunctionVolume_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FunctionVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FunctionVolume`/`Const_FunctionVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FunctionVolume
    {
        internal readonly Const_FunctionVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FunctionVolume() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FunctionVolume(Const_FunctionVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FunctionVolume(Const_FunctionVolume arg) {return new(arg);}
        public _ByValue_FunctionVolume(MR.Misc._Moved<FunctionVolume> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FunctionVolume(MR.Misc._Moved<FunctionVolume> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FunctionVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FunctionVolume`/`Const_FunctionVolume` directly.
    public class _InOptMut_FunctionVolume
    {
        public FunctionVolume? Opt;

        public _InOptMut_FunctionVolume() {}
        public _InOptMut_FunctionVolume(FunctionVolume value) {Opt = value;}
        public static implicit operator _InOptMut_FunctionVolume(FunctionVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `FunctionVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FunctionVolume`/`Const_FunctionVolume` to pass it to the function.
    public class _InOptConst_FunctionVolume
    {
        public Const_FunctionVolume? Opt;

        public _InOptConst_FunctionVolume() {}
        public _InOptConst_FunctionVolume(Const_FunctionVolume value) {Opt = value;}
        public static implicit operator _InOptConst_FunctionVolume(Const_FunctionVolume value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::FunctionVolumeU8`.
    /// This is the const half of the class.
    public class Const_FunctionVolumeU8 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FunctionVolumeU8(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_Destroy", ExactSpelling = true)]
            extern static void __MR_FunctionVolumeU8_Destroy(_Underlying *_this);
            __MR_FunctionVolumeU8_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FunctionVolumeU8() {Dispose(false);}

        public unsafe MR.Std.Const_Function_UnsignedCharFuncFromConstMRVector3iRef Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_Get_data", ExactSpelling = true)]
                extern static MR.Std.Const_Function_UnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_FunctionVolumeU8_Get_data(_Underlying *_this);
                return new(__MR_FunctionVolumeU8_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_FunctionVolumeU8_Get_dims(_Underlying *_this);
                return new(__MR_FunctionVolumeU8_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_FunctionVolumeU8_Get_voxelSize(_Underlying *_this);
                return new(__MR_FunctionVolumeU8_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FunctionVolumeU8() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_DefaultConstruct();
            _UnderlyingPtr = __MR_FunctionVolumeU8_DefaultConstruct();
        }

        /// Constructs `MR::FunctionVolumeU8` elementwise.
        public unsafe Const_FunctionVolumeU8(MR.Std._ByValue_Function_UnsignedCharFuncFromConstMRVector3iRef data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_ConstructFrom", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Std.Function_UnsignedCharFuncFromConstMRVector3iRef._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_FunctionVolumeU8_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::FunctionVolumeU8::FunctionVolumeU8`.
        public unsafe Const_FunctionVolumeU8(MR._ByValue_FunctionVolumeU8 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FunctionVolumeU8._Underlying *_other);
            _UnderlyingPtr = __MR_FunctionVolumeU8_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FunctionVolumeU8::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_FunctionVolumeU8_heapBytes(_Underlying *_this);
            return __MR_FunctionVolumeU8_heapBytes(_UnderlyingPtr);
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::FunctionVolumeU8`.
    /// This is the non-const half of the class.
    public class FunctionVolumeU8 : Const_FunctionVolumeU8
    {
        internal unsafe FunctionVolumeU8(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Function_UnsignedCharFuncFromConstMRVector3iRef Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_GetMutable_data", ExactSpelling = true)]
                extern static MR.Std.Function_UnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_FunctionVolumeU8_GetMutable_data(_Underlying *_this);
                return new(__MR_FunctionVolumeU8_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_FunctionVolumeU8_GetMutable_dims(_Underlying *_this);
                return new(__MR_FunctionVolumeU8_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_FunctionVolumeU8_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_FunctionVolumeU8_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FunctionVolumeU8() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_DefaultConstruct();
            _UnderlyingPtr = __MR_FunctionVolumeU8_DefaultConstruct();
        }

        /// Constructs `MR::FunctionVolumeU8` elementwise.
        public unsafe FunctionVolumeU8(MR.Std._ByValue_Function_UnsignedCharFuncFromConstMRVector3iRef data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_ConstructFrom", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Std.Function_UnsignedCharFuncFromConstMRVector3iRef._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_FunctionVolumeU8_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::FunctionVolumeU8::FunctionVolumeU8`.
        public unsafe FunctionVolumeU8(MR._ByValue_FunctionVolumeU8 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FunctionVolumeU8._Underlying *_other);
            _UnderlyingPtr = __MR_FunctionVolumeU8_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FunctionVolumeU8::operator=`.
        public unsafe MR.FunctionVolumeU8 Assign(MR._ByValue_FunctionVolumeU8 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FunctionVolumeU8_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FunctionVolumeU8._Underlying *__MR_FunctionVolumeU8_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FunctionVolumeU8._Underlying *_other);
            return new(__MR_FunctionVolumeU8_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FunctionVolumeU8` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FunctionVolumeU8`/`Const_FunctionVolumeU8` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FunctionVolumeU8
    {
        internal readonly Const_FunctionVolumeU8? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FunctionVolumeU8() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FunctionVolumeU8(Const_FunctionVolumeU8 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FunctionVolumeU8(Const_FunctionVolumeU8 arg) {return new(arg);}
        public _ByValue_FunctionVolumeU8(MR.Misc._Moved<FunctionVolumeU8> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FunctionVolumeU8(MR.Misc._Moved<FunctionVolumeU8> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FunctionVolumeU8` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FunctionVolumeU8`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FunctionVolumeU8`/`Const_FunctionVolumeU8` directly.
    public class _InOptMut_FunctionVolumeU8
    {
        public FunctionVolumeU8? Opt;

        public _InOptMut_FunctionVolumeU8() {}
        public _InOptMut_FunctionVolumeU8(FunctionVolumeU8 value) {Opt = value;}
        public static implicit operator _InOptMut_FunctionVolumeU8(FunctionVolumeU8 value) {return new(value);}
    }

    /// This is used for optional parameters of class `FunctionVolumeU8` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FunctionVolumeU8`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FunctionVolumeU8`/`Const_FunctionVolumeU8` to pass it to the function.
    public class _InOptConst_FunctionVolumeU8
    {
        public Const_FunctionVolumeU8? Opt;

        public _InOptConst_FunctionVolumeU8() {}
        public _InOptConst_FunctionVolumeU8(Const_FunctionVolumeU8 value) {Opt = value;}
        public static implicit operator _InOptConst_FunctionVolumeU8(Const_FunctionVolumeU8 value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::SimpleVolume`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMax`
    /// This is the const half of the class.
    public class Const_SimpleVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SimpleVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_SimpleVolume_Destroy(_Underlying *_this);
            __MR_SimpleVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SimpleVolume() {Dispose(false);}

        public unsafe MR.Const_Vector_Float_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_Get_data", ExactSpelling = true)]
                extern static MR.Const_Vector_Float_MRVoxelId._Underlying *__MR_SimpleVolume_Get_data(_Underlying *_this);
                return new(__MR_SimpleVolume_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_SimpleVolume_Get_dims(_Underlying *_this);
                return new(__MR_SimpleVolume_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_SimpleVolume_Get_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolume_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SimpleVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolume_DefaultConstruct();
        }

        /// Constructs `MR::SimpleVolume` elementwise.
        public unsafe Const_SimpleVolume(MR._ByValue_Vector_Float_MRVoxelId data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_ConstructFrom", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Vector_Float_MRVoxelId._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_SimpleVolume_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::SimpleVolume::SimpleVolume`.
        public unsafe Const_SimpleVolume(MR._ByValue_SimpleVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolume._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolume::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_SimpleVolume_heapBytes(_Underlying *_this);
            return __MR_SimpleVolume_heapBytes(_UnderlyingPtr);
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::SimpleVolume`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMax`
    /// This is the non-const half of the class.
    public class SimpleVolume : Const_SimpleVolume
    {
        internal unsafe SimpleVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Vector_Float_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_GetMutable_data", ExactSpelling = true)]
                extern static MR.Vector_Float_MRVoxelId._Underlying *__MR_SimpleVolume_GetMutable_data(_Underlying *_this);
                return new(__MR_SimpleVolume_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_SimpleVolume_GetMutable_dims(_Underlying *_this);
                return new(__MR_SimpleVolume_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_SimpleVolume_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolume_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SimpleVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolume_DefaultConstruct();
        }

        /// Constructs `MR::SimpleVolume` elementwise.
        public unsafe SimpleVolume(MR._ByValue_Vector_Float_MRVoxelId data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_ConstructFrom", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Vector_Float_MRVoxelId._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_SimpleVolume_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::SimpleVolume::SimpleVolume`.
        public unsafe SimpleVolume(MR._ByValue_SimpleVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolume._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolume::operator=`.
        public unsafe MR.SimpleVolume Assign(MR._ByValue_SimpleVolume _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolume_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolume._Underlying *__MR_SimpleVolume_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SimpleVolume._Underlying *_other);
            return new(__MR_SimpleVolume_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SimpleVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SimpleVolume`/`Const_SimpleVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SimpleVolume
    {
        internal readonly Const_SimpleVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SimpleVolume() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SimpleVolume(Const_SimpleVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SimpleVolume(Const_SimpleVolume arg) {return new(arg);}
        public _ByValue_SimpleVolume(MR.Misc._Moved<SimpleVolume> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SimpleVolume(MR.Misc._Moved<SimpleVolume> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SimpleVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SimpleVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolume`/`Const_SimpleVolume` directly.
    public class _InOptMut_SimpleVolume
    {
        public SimpleVolume? Opt;

        public _InOptMut_SimpleVolume() {}
        public _InOptMut_SimpleVolume(SimpleVolume value) {Opt = value;}
        public static implicit operator _InOptMut_SimpleVolume(SimpleVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `SimpleVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SimpleVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolume`/`Const_SimpleVolume` to pass it to the function.
    public class _InOptConst_SimpleVolume
    {
        public Const_SimpleVolume? Opt;

        public _InOptConst_SimpleVolume() {}
        public _InOptConst_SimpleVolume(Const_SimpleVolume value) {Opt = value;}
        public static implicit operator _InOptConst_SimpleVolume(Const_SimpleVolume value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::SimpleVolumeU16`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMaxU16`
    /// This is the const half of the class.
    public class Const_SimpleVolumeU16 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SimpleVolumeU16(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_Destroy", ExactSpelling = true)]
            extern static void __MR_SimpleVolumeU16_Destroy(_Underlying *_this);
            __MR_SimpleVolumeU16_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SimpleVolumeU16() {Dispose(false);}

        public unsafe MR.Const_Vector_UnsignedShort_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_Get_data", ExactSpelling = true)]
                extern static MR.Const_Vector_UnsignedShort_MRVoxelId._Underlying *__MR_SimpleVolumeU16_Get_data(_Underlying *_this);
                return new(__MR_SimpleVolumeU16_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_SimpleVolumeU16_Get_dims(_Underlying *_this);
                return new(__MR_SimpleVolumeU16_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_SimpleVolumeU16_Get_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolumeU16_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SimpleVolumeU16() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolumeU16_DefaultConstruct();
        }

        /// Constructs `MR::SimpleVolumeU16` elementwise.
        public unsafe Const_SimpleVolumeU16(MR._ByValue_Vector_UnsignedShort_MRVoxelId data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_ConstructFrom", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Vector_UnsignedShort_MRVoxelId._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_SimpleVolumeU16_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::SimpleVolumeU16::SimpleVolumeU16`.
        public unsafe Const_SimpleVolumeU16(MR._ByValue_SimpleVolumeU16 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeU16._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolumeU16_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolumeU16::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_SimpleVolumeU16_heapBytes(_Underlying *_this);
            return __MR_SimpleVolumeU16_heapBytes(_UnderlyingPtr);
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::SimpleVolumeU16`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMaxU16`
    /// This is the non-const half of the class.
    public class SimpleVolumeU16 : Const_SimpleVolumeU16
    {
        internal unsafe SimpleVolumeU16(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Vector_UnsignedShort_MRVoxelId Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_GetMutable_data", ExactSpelling = true)]
                extern static MR.Vector_UnsignedShort_MRVoxelId._Underlying *__MR_SimpleVolumeU16_GetMutable_data(_Underlying *_this);
                return new(__MR_SimpleVolumeU16_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_SimpleVolumeU16_GetMutable_dims(_Underlying *_this);
                return new(__MR_SimpleVolumeU16_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_SimpleVolumeU16_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_SimpleVolumeU16_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SimpleVolumeU16() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleVolumeU16_DefaultConstruct();
        }

        /// Constructs `MR::SimpleVolumeU16` elementwise.
        public unsafe SimpleVolumeU16(MR._ByValue_Vector_UnsignedShort_MRVoxelId data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_ConstructFrom", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.Vector_UnsignedShort_MRVoxelId._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_SimpleVolumeU16_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::SimpleVolumeU16::SimpleVolumeU16`.
        public unsafe SimpleVolumeU16(MR._ByValue_SimpleVolumeU16 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeU16._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleVolumeU16_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleVolumeU16::operator=`.
        public unsafe MR.SimpleVolumeU16 Assign(MR._ByValue_SimpleVolumeU16 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleVolumeU16_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SimpleVolumeU16._Underlying *__MR_SimpleVolumeU16_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SimpleVolumeU16._Underlying *_other);
            return new(__MR_SimpleVolumeU16_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SimpleVolumeU16` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SimpleVolumeU16`/`Const_SimpleVolumeU16` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SimpleVolumeU16
    {
        internal readonly Const_SimpleVolumeU16? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SimpleVolumeU16() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SimpleVolumeU16(Const_SimpleVolumeU16 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SimpleVolumeU16(Const_SimpleVolumeU16 arg) {return new(arg);}
        public _ByValue_SimpleVolumeU16(MR.Misc._Moved<SimpleVolumeU16> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SimpleVolumeU16(MR.Misc._Moved<SimpleVolumeU16> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SimpleVolumeU16` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SimpleVolumeU16`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolumeU16`/`Const_SimpleVolumeU16` directly.
    public class _InOptMut_SimpleVolumeU16
    {
        public SimpleVolumeU16? Opt;

        public _InOptMut_SimpleVolumeU16() {}
        public _InOptMut_SimpleVolumeU16(SimpleVolumeU16 value) {Opt = value;}
        public static implicit operator _InOptMut_SimpleVolumeU16(SimpleVolumeU16 value) {return new(value);}
    }

    /// This is used for optional parameters of class `SimpleVolumeU16` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SimpleVolumeU16`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleVolumeU16`/`Const_SimpleVolumeU16` to pass it to the function.
    public class _InOptConst_SimpleVolumeU16
    {
        public Const_SimpleVolumeU16? Opt;

        public _InOptConst_SimpleVolumeU16() {}
        public _InOptConst_SimpleVolumeU16(Const_SimpleVolumeU16 value) {Opt = value;}
        public static implicit operator _InOptConst_SimpleVolumeU16(Const_SimpleVolumeU16 value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::SimpleBinaryVolume`.
    /// This is the const half of the class.
    public class Const_SimpleBinaryVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SimpleBinaryVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_SimpleBinaryVolume_Destroy(_Underlying *_this);
            __MR_SimpleBinaryVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SimpleBinaryVolume() {Dispose(false);}

        public unsafe MR.Const_VoxelBitSet Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_Get_data", ExactSpelling = true)]
                extern static MR.Const_VoxelBitSet._Underlying *__MR_SimpleBinaryVolume_Get_data(_Underlying *_this);
                return new(__MR_SimpleBinaryVolume_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_SimpleBinaryVolume_Get_dims(_Underlying *_this);
                return new(__MR_SimpleBinaryVolume_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_SimpleBinaryVolume_Get_voxelSize(_Underlying *_this);
                return new(__MR_SimpleBinaryVolume_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SimpleBinaryVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleBinaryVolume_DefaultConstruct();
        }

        /// Constructs `MR::SimpleBinaryVolume` elementwise.
        public unsafe Const_SimpleBinaryVolume(MR._ByValue_VoxelBitSet data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_ConstructFrom", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.VoxelBitSet._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_SimpleBinaryVolume_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::SimpleBinaryVolume::SimpleBinaryVolume`.
        public unsafe Const_SimpleBinaryVolume(MR._ByValue_SimpleBinaryVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleBinaryVolume._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleBinaryVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleBinaryVolume::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_SimpleBinaryVolume_heapBytes(_Underlying *_this);
            return __MR_SimpleBinaryVolume_heapBytes(_UnderlyingPtr);
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::SimpleBinaryVolume`.
    /// This is the non-const half of the class.
    public class SimpleBinaryVolume : Const_SimpleBinaryVolume
    {
        internal unsafe SimpleBinaryVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.VoxelBitSet Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_GetMutable_data", ExactSpelling = true)]
                extern static MR.VoxelBitSet._Underlying *__MR_SimpleBinaryVolume_GetMutable_data(_Underlying *_this);
                return new(__MR_SimpleBinaryVolume_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_SimpleBinaryVolume_GetMutable_dims(_Underlying *_this);
                return new(__MR_SimpleBinaryVolume_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_SimpleBinaryVolume_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_SimpleBinaryVolume_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SimpleBinaryVolume() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_DefaultConstruct();
            _UnderlyingPtr = __MR_SimpleBinaryVolume_DefaultConstruct();
        }

        /// Constructs `MR::SimpleBinaryVolume` elementwise.
        public unsafe SimpleBinaryVolume(MR._ByValue_VoxelBitSet data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_ConstructFrom", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.VoxelBitSet._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_SimpleBinaryVolume_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::SimpleBinaryVolume::SimpleBinaryVolume`.
        public unsafe SimpleBinaryVolume(MR._ByValue_SimpleBinaryVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SimpleBinaryVolume._Underlying *_other);
            _UnderlyingPtr = __MR_SimpleBinaryVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SimpleBinaryVolume::operator=`.
        public unsafe MR.SimpleBinaryVolume Assign(MR._ByValue_SimpleBinaryVolume _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SimpleBinaryVolume_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SimpleBinaryVolume._Underlying *__MR_SimpleBinaryVolume_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SimpleBinaryVolume._Underlying *_other);
            return new(__MR_SimpleBinaryVolume_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SimpleBinaryVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SimpleBinaryVolume`/`Const_SimpleBinaryVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SimpleBinaryVolume
    {
        internal readonly Const_SimpleBinaryVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SimpleBinaryVolume() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SimpleBinaryVolume(Const_SimpleBinaryVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SimpleBinaryVolume(Const_SimpleBinaryVolume arg) {return new(arg);}
        public _ByValue_SimpleBinaryVolume(MR.Misc._Moved<SimpleBinaryVolume> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SimpleBinaryVolume(MR.Misc._Moved<SimpleBinaryVolume> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SimpleBinaryVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SimpleBinaryVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleBinaryVolume`/`Const_SimpleBinaryVolume` directly.
    public class _InOptMut_SimpleBinaryVolume
    {
        public SimpleBinaryVolume? Opt;

        public _InOptMut_SimpleBinaryVolume() {}
        public _InOptMut_SimpleBinaryVolume(SimpleBinaryVolume value) {Opt = value;}
        public static implicit operator _InOptMut_SimpleBinaryVolume(SimpleBinaryVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `SimpleBinaryVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SimpleBinaryVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SimpleBinaryVolume`/`Const_SimpleBinaryVolume` to pass it to the function.
    public class _InOptConst_SimpleBinaryVolume
    {
        public Const_SimpleBinaryVolume? Opt;

        public _InOptConst_SimpleBinaryVolume() {}
        public _InOptConst_SimpleBinaryVolume(Const_SimpleBinaryVolume value) {Opt = value;}
        public static implicit operator _InOptConst_SimpleBinaryVolume(Const_SimpleBinaryVolume value) {return new(value);}
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::VoxelsVolume<MR::FloatGrid>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::VdbVolume`
    /// This is the const half of the class.
    public class Const_VoxelsVolume_MRFloatGrid : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolume_MRFloatGrid(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolume_MR_FloatGrid_Destroy(_Underlying *_this);
            __MR_VoxelsVolume_MR_FloatGrid_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolume_MRFloatGrid() {Dispose(false);}

        public unsafe MR.Const_FloatGrid Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_Get_data", ExactSpelling = true)]
                extern static MR.Const_FloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_Get_data(_Underlying *_this);
                return new(__MR_VoxelsVolume_MR_FloatGrid_Get_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_Get_dims", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_VoxelsVolume_MR_FloatGrid_Get_dims(_Underlying *_this);
                return new(__MR_VoxelsVolume_MR_FloatGrid_Get_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_VoxelsVolume_MR_FloatGrid_Get_voxelSize(_Underlying *_this);
                return new(__MR_VoxelsVolume_MR_FloatGrid_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelsVolume_MRFloatGrid() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelsVolume_MR_FloatGrid_DefaultConstruct();
        }

        /// Constructs `MR::VoxelsVolume<MR::FloatGrid>` elementwise.
        public unsafe Const_VoxelsVolume_MRFloatGrid(MR._ByValue_FloatGrid data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_ConstructFrom", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.FloatGrid._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_VoxelsVolume_MR_FloatGrid_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::VoxelsVolume<MR::FloatGrid>::VoxelsVolume`.
        public unsafe Const_VoxelsVolume_MRFloatGrid(MR._ByValue_VoxelsVolume_MRFloatGrid _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsVolume_MRFloatGrid._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolume_MR_FloatGrid_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VoxelsVolume<MR::FloatGrid>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_VoxelsVolume_MR_FloatGrid_heapBytes(_Underlying *_this);
            return __MR_VoxelsVolume_MR_FloatGrid_heapBytes(_UnderlyingPtr);
        }
    }

    /// represents a box in 3D space subdivided on voxels stored in T
    /// Generated from class `MR::VoxelsVolume<MR::FloatGrid>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::VdbVolume`
    /// This is the non-const half of the class.
    public class VoxelsVolume_MRFloatGrid : Const_VoxelsVolume_MRFloatGrid
    {
        internal unsafe VoxelsVolume_MRFloatGrid(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.FloatGrid Data
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_GetMutable_data", ExactSpelling = true)]
                extern static MR.FloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_GetMutable_data(_Underlying *_this);
                return new(__MR_VoxelsVolume_MR_FloatGrid_GetMutable_data(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Dims
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_GetMutable_dims", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_VoxelsVolume_MR_FloatGrid_GetMutable_dims(_Underlying *_this);
                return new(__MR_VoxelsVolume_MR_FloatGrid_GetMutable_dims(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_VoxelsVolume_MR_FloatGrid_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_VoxelsVolume_MR_FloatGrid_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelsVolume_MRFloatGrid() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelsVolume_MR_FloatGrid_DefaultConstruct();
        }

        /// Constructs `MR::VoxelsVolume<MR::FloatGrid>` elementwise.
        public unsafe VoxelsVolume_MRFloatGrid(MR._ByValue_FloatGrid data, MR.Vector3i dims, MR.Vector3f voxelSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_ConstructFrom", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_ConstructFrom(MR.Misc._PassBy data_pass_by, MR.FloatGrid._Underlying *data, MR.Vector3i dims, MR.Vector3f voxelSize);
            _UnderlyingPtr = __MR_VoxelsVolume_MR_FloatGrid_ConstructFrom(data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null, dims, voxelSize);
        }

        /// Generated from constructor `MR::VoxelsVolume<MR::FloatGrid>::VoxelsVolume`.
        public unsafe VoxelsVolume_MRFloatGrid(MR._ByValue_VoxelsVolume_MRFloatGrid _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsVolume_MRFloatGrid._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolume_MR_FloatGrid_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VoxelsVolume<MR::FloatGrid>::operator=`.
        public unsafe MR.VoxelsVolume_MRFloatGrid Assign(MR._ByValue_VoxelsVolume_MRFloatGrid _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolume_MR_FloatGrid_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolume_MRFloatGrid._Underlying *__MR_VoxelsVolume_MR_FloatGrid_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsVolume_MRFloatGrid._Underlying *_other);
            return new(__MR_VoxelsVolume_MR_FloatGrid_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VoxelsVolume_MRFloatGrid` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VoxelsVolume_MRFloatGrid`/`Const_VoxelsVolume_MRFloatGrid` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VoxelsVolume_MRFloatGrid
    {
        internal readonly Const_VoxelsVolume_MRFloatGrid? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VoxelsVolume_MRFloatGrid() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VoxelsVolume_MRFloatGrid(Const_VoxelsVolume_MRFloatGrid new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VoxelsVolume_MRFloatGrid(Const_VoxelsVolume_MRFloatGrid arg) {return new(arg);}
        public _ByValue_VoxelsVolume_MRFloatGrid(MR.Misc._Moved<VoxelsVolume_MRFloatGrid> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VoxelsVolume_MRFloatGrid(MR.Misc._Moved<VoxelsVolume_MRFloatGrid> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VoxelsVolume_MRFloatGrid` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolume_MRFloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolume_MRFloatGrid`/`Const_VoxelsVolume_MRFloatGrid` directly.
    public class _InOptMut_VoxelsVolume_MRFloatGrid
    {
        public VoxelsVolume_MRFloatGrid? Opt;

        public _InOptMut_VoxelsVolume_MRFloatGrid() {}
        public _InOptMut_VoxelsVolume_MRFloatGrid(VoxelsVolume_MRFloatGrid value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolume_MRFloatGrid(VoxelsVolume_MRFloatGrid value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolume_MRFloatGrid` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolume_MRFloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolume_MRFloatGrid`/`Const_VoxelsVolume_MRFloatGrid` to pass it to the function.
    public class _InOptConst_VoxelsVolume_MRFloatGrid
    {
        public Const_VoxelsVolume_MRFloatGrid? Opt;

        public _InOptConst_VoxelsVolume_MRFloatGrid() {}
        public _InOptConst_VoxelsVolume_MRFloatGrid(Const_VoxelsVolume_MRFloatGrid value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolume_MRFloatGrid(Const_VoxelsVolume_MRFloatGrid value) {return new(value);}
    }

    /// Generated from class `MR::VoxelTraits<MR::Vector<float, MR::VoxelId>>`.
    /// This is the const half of the class.
    public class Const_VoxelTraits_MRVectorFloatMRVoxelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelTraits_MRVectorFloatMRVoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_float_MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelTraits_MR_Vector_float_MR_VoxelId_Destroy(_Underlying *_this);
            __MR_VoxelTraits_MR_Vector_float_MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelTraits_MRVectorFloatMRVoxelId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelTraits_MRVectorFloatMRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_float_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_float_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_float_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::Vector<float, MR::VoxelId>>::VoxelTraits`.
        public unsafe Const_VoxelTraits_MRVectorFloatMRVoxelId(MR.Const_VoxelTraits_MRVectorFloatMRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_float_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_float_MR_VoxelId_ConstructFromAnother(MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_float_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::VoxelTraits<MR::Vector<float, MR::VoxelId>>`.
    /// This is the non-const half of the class.
    public class VoxelTraits_MRVectorFloatMRVoxelId : Const_VoxelTraits_MRVectorFloatMRVoxelId
    {
        internal unsafe VoxelTraits_MRVectorFloatMRVoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelTraits_MRVectorFloatMRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_float_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_float_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_float_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::Vector<float, MR::VoxelId>>::VoxelTraits`.
        public unsafe VoxelTraits_MRVectorFloatMRVoxelId(MR.Const_VoxelTraits_MRVectorFloatMRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_float_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_float_MR_VoxelId_ConstructFromAnother(MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_float_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelTraits<MR::Vector<float, MR::VoxelId>>::operator=`.
        public unsafe MR.VoxelTraits_MRVectorFloatMRVoxelId Assign(MR.Const_VoxelTraits_MRVectorFloatMRVoxelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_float_MR_VoxelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_float_MR_VoxelId_AssignFromAnother(_Underlying *_this, MR.VoxelTraits_MRVectorFloatMRVoxelId._Underlying *_other);
            return new(__MR_VoxelTraits_MR_Vector_float_MR_VoxelId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelTraits_MRVectorFloatMRVoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelTraits_MRVectorFloatMRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRVectorFloatMRVoxelId`/`Const_VoxelTraits_MRVectorFloatMRVoxelId` directly.
    public class _InOptMut_VoxelTraits_MRVectorFloatMRVoxelId
    {
        public VoxelTraits_MRVectorFloatMRVoxelId? Opt;

        public _InOptMut_VoxelTraits_MRVectorFloatMRVoxelId() {}
        public _InOptMut_VoxelTraits_MRVectorFloatMRVoxelId(VoxelTraits_MRVectorFloatMRVoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelTraits_MRVectorFloatMRVoxelId(VoxelTraits_MRVectorFloatMRVoxelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelTraits_MRVectorFloatMRVoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelTraits_MRVectorFloatMRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRVectorFloatMRVoxelId`/`Const_VoxelTraits_MRVectorFloatMRVoxelId` to pass it to the function.
    public class _InOptConst_VoxelTraits_MRVectorFloatMRVoxelId
    {
        public Const_VoxelTraits_MRVectorFloatMRVoxelId? Opt;

        public _InOptConst_VoxelTraits_MRVectorFloatMRVoxelId() {}
        public _InOptConst_VoxelTraits_MRVectorFloatMRVoxelId(Const_VoxelTraits_MRVectorFloatMRVoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelTraits_MRVectorFloatMRVoxelId(Const_VoxelTraits_MRVectorFloatMRVoxelId value) {return new(value);}
    }

    /// Generated from class `MR::VoxelTraits<MR::Vector<unsigned short, MR::VoxelId>>`.
    /// This is the const half of the class.
    public class Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_Destroy(_Underlying *_this);
            __MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::Vector<unsigned short, MR::VoxelId>>::VoxelTraits`.
        public unsafe Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId(MR.Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_ConstructFromAnother(MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::VoxelTraits<MR::Vector<unsigned short, MR::VoxelId>>`.
    /// This is the non-const half of the class.
    public class VoxelTraits_MRVectorUnsignedShortMRVoxelId : Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId
    {
        internal unsafe VoxelTraits_MRVectorUnsignedShortMRVoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelTraits_MRVectorUnsignedShortMRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::Vector<unsigned short, MR::VoxelId>>::VoxelTraits`.
        public unsafe VoxelTraits_MRVectorUnsignedShortMRVoxelId(MR.Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_ConstructFromAnother(MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelTraits<MR::Vector<unsigned short, MR::VoxelId>>::operator=`.
        public unsafe MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId Assign(MR.Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *__MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_AssignFromAnother(_Underlying *_this, MR.VoxelTraits_MRVectorUnsignedShortMRVoxelId._Underlying *_other);
            return new(__MR_VoxelTraits_MR_Vector_unsigned_short_MR_VoxelId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelTraits_MRVectorUnsignedShortMRVoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelTraits_MRVectorUnsignedShortMRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRVectorUnsignedShortMRVoxelId`/`Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId` directly.
    public class _InOptMut_VoxelTraits_MRVectorUnsignedShortMRVoxelId
    {
        public VoxelTraits_MRVectorUnsignedShortMRVoxelId? Opt;

        public _InOptMut_VoxelTraits_MRVectorUnsignedShortMRVoxelId() {}
        public _InOptMut_VoxelTraits_MRVectorUnsignedShortMRVoxelId(VoxelTraits_MRVectorUnsignedShortMRVoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelTraits_MRVectorUnsignedShortMRVoxelId(VoxelTraits_MRVectorUnsignedShortMRVoxelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelTraits_MRVectorUnsignedShortMRVoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelTraits_MRVectorUnsignedShortMRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRVectorUnsignedShortMRVoxelId`/`Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId` to pass it to the function.
    public class _InOptConst_VoxelTraits_MRVectorUnsignedShortMRVoxelId
    {
        public Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId? Opt;

        public _InOptConst_VoxelTraits_MRVectorUnsignedShortMRVoxelId() {}
        public _InOptConst_VoxelTraits_MRVectorUnsignedShortMRVoxelId(Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelTraits_MRVectorUnsignedShortMRVoxelId(Const_VoxelTraits_MRVectorUnsignedShortMRVoxelId value) {return new(value);}
    }

    /// Generated from class `MR::VoxelTraits<std::function<float(const MR::Vector3i &)>>`.
    /// This is the const half of the class.
    public class Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_Destroy(_Underlying *_this);
            __MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<std::function<float(const MR::Vector3i &)>>::VoxelTraits`.
        public unsafe Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(MR.Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_ConstructFromAnother(MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::VoxelTraits<std::function<float(const MR::Vector3i &)>>`.
    /// This is the non-const half of the class.
    public class VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef : Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef
    {
        internal unsafe VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<std::function<float(const MR::Vector3i &)>>::VoxelTraits`.
        public unsafe VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(MR.Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_ConstructFromAnother(MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelTraits<std::function<float(const MR::Vector3i &)>>::operator=`.
        public unsafe MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef Assign(MR.Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_AssignFromAnother(_Underlying *_this, MR.VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef._Underlying *_other);
            return new(__MR_VoxelTraits_std_function_float_func_from_const_MR_Vector3i_ref_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef`/`Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef` directly.
    public class _InOptMut_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef
    {
        public VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef? Opt;

        public _InOptMut_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef() {}
        public _InOptMut_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef`/`Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef` to pass it to the function.
    public class _InOptConst_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef
    {
        public Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef? Opt;

        public _InOptConst_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef() {}
        public _InOptConst_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef(Const_VoxelTraits_StdFunctionFloatFuncFromConstMRVector3iRef value) {return new(value);}
    }

    /// Generated from class `MR::VoxelTraits<std::function<unsigned char(const MR::Vector3i &)>>`.
    /// This is the const half of the class.
    public class Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_Destroy(_Underlying *_this);
            __MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<std::function<unsigned char(const MR::Vector3i &)>>::VoxelTraits`.
        public unsafe Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(MR.Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_ConstructFromAnother(MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::VoxelTraits<std::function<unsigned char(const MR::Vector3i &)>>`.
    /// This is the non-const half of the class.
    public class VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef : Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef
    {
        internal unsafe VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<std::function<unsigned char(const MR::Vector3i &)>>::VoxelTraits`.
        public unsafe VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(MR.Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_ConstructFromAnother(MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelTraits<std::function<unsigned char(const MR::Vector3i &)>>::operator=`.
        public unsafe MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef Assign(MR.Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *__MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_AssignFromAnother(_Underlying *_this, MR.VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef._Underlying *_other);
            return new(__MR_VoxelTraits_std_function_unsigned_char_func_from_const_MR_Vector3i_ref_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef`/`Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef` directly.
    public class _InOptMut_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef
    {
        public VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef? Opt;

        public _InOptMut_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef() {}
        public _InOptMut_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef`/`Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef` to pass it to the function.
    public class _InOptConst_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef
    {
        public Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef? Opt;

        public _InOptConst_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef() {}
        public _InOptConst_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef(Const_VoxelTraits_StdFunctionUnsignedCharFuncFromConstMRVector3iRef value) {return new(value);}
    }

    /// Generated from class `MR::VoxelTraits<MR::VoxelBitSet>`.
    /// This is the const half of the class.
    public class Const_VoxelTraits_MRVoxelBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelTraits_MRVoxelBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_VoxelBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelTraits_MR_VoxelBitSet_Destroy(_Underlying *_this);
            __MR_VoxelTraits_MR_VoxelBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelTraits_MRVoxelBitSet() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelTraits_MRVoxelBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_VoxelBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVoxelBitSet._Underlying *__MR_VoxelTraits_MR_VoxelBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_VoxelBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::VoxelBitSet>::VoxelTraits`.
        public unsafe Const_VoxelTraits_MRVoxelBitSet(MR.Const_VoxelTraits_MRVoxelBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_VoxelBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVoxelBitSet._Underlying *__MR_VoxelTraits_MR_VoxelBitSet_ConstructFromAnother(MR.VoxelTraits_MRVoxelBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_VoxelBitSet_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::VoxelTraits<MR::VoxelBitSet>`.
    /// This is the non-const half of the class.
    public class VoxelTraits_MRVoxelBitSet : Const_VoxelTraits_MRVoxelBitSet
    {
        internal unsafe VoxelTraits_MRVoxelBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelTraits_MRVoxelBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_VoxelBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVoxelBitSet._Underlying *__MR_VoxelTraits_MR_VoxelBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_VoxelBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::VoxelBitSet>::VoxelTraits`.
        public unsafe VoxelTraits_MRVoxelBitSet(MR.Const_VoxelTraits_MRVoxelBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_VoxelBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVoxelBitSet._Underlying *__MR_VoxelTraits_MR_VoxelBitSet_ConstructFromAnother(MR.VoxelTraits_MRVoxelBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_VoxelBitSet_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelTraits<MR::VoxelBitSet>::operator=`.
        public unsafe MR.VoxelTraits_MRVoxelBitSet Assign(MR.Const_VoxelTraits_MRVoxelBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_VoxelBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRVoxelBitSet._Underlying *__MR_VoxelTraits_MR_VoxelBitSet_AssignFromAnother(_Underlying *_this, MR.VoxelTraits_MRVoxelBitSet._Underlying *_other);
            return new(__MR_VoxelTraits_MR_VoxelBitSet_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelTraits_MRVoxelBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelTraits_MRVoxelBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRVoxelBitSet`/`Const_VoxelTraits_MRVoxelBitSet` directly.
    public class _InOptMut_VoxelTraits_MRVoxelBitSet
    {
        public VoxelTraits_MRVoxelBitSet? Opt;

        public _InOptMut_VoxelTraits_MRVoxelBitSet() {}
        public _InOptMut_VoxelTraits_MRVoxelBitSet(VoxelTraits_MRVoxelBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelTraits_MRVoxelBitSet(VoxelTraits_MRVoxelBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelTraits_MRVoxelBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelTraits_MRVoxelBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRVoxelBitSet`/`Const_VoxelTraits_MRVoxelBitSet` to pass it to the function.
    public class _InOptConst_VoxelTraits_MRVoxelBitSet
    {
        public Const_VoxelTraits_MRVoxelBitSet? Opt;

        public _InOptConst_VoxelTraits_MRVoxelBitSet() {}
        public _InOptConst_VoxelTraits_MRVoxelBitSet(Const_VoxelTraits_MRVoxelBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelTraits_MRVoxelBitSet(Const_VoxelTraits_MRVoxelBitSet value) {return new(value);}
    }

    /// Generated from class `MR::VoxelTraits<MR::FloatGrid>`.
    /// This is the const half of the class.
    public class Const_VoxelTraits_MRFloatGrid : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelTraits_MRFloatGrid(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_FloatGrid_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelTraits_MR_FloatGrid_Destroy(_Underlying *_this);
            __MR_VoxelTraits_MR_FloatGrid_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelTraits_MRFloatGrid() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelTraits_MRFloatGrid() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_FloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRFloatGrid._Underlying *__MR_VoxelTraits_MR_FloatGrid_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_FloatGrid_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::FloatGrid>::VoxelTraits`.
        public unsafe Const_VoxelTraits_MRFloatGrid(MR.Const_VoxelTraits_MRFloatGrid _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_FloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRFloatGrid._Underlying *__MR_VoxelTraits_MR_FloatGrid_ConstructFromAnother(MR.VoxelTraits_MRFloatGrid._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_FloatGrid_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::VoxelTraits<MR::FloatGrid>`.
    /// This is the non-const half of the class.
    public class VoxelTraits_MRFloatGrid : Const_VoxelTraits_MRFloatGrid
    {
        internal unsafe VoxelTraits_MRFloatGrid(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelTraits_MRFloatGrid() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_FloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRFloatGrid._Underlying *__MR_VoxelTraits_MR_FloatGrid_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelTraits_MR_FloatGrid_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelTraits<MR::FloatGrid>::VoxelTraits`.
        public unsafe VoxelTraits_MRFloatGrid(MR.Const_VoxelTraits_MRFloatGrid _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_FloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRFloatGrid._Underlying *__MR_VoxelTraits_MR_FloatGrid_ConstructFromAnother(MR.VoxelTraits_MRFloatGrid._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelTraits_MR_FloatGrid_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelTraits<MR::FloatGrid>::operator=`.
        public unsafe MR.VoxelTraits_MRFloatGrid Assign(MR.Const_VoxelTraits_MRFloatGrid _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelTraits_MR_FloatGrid_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelTraits_MRFloatGrid._Underlying *__MR_VoxelTraits_MR_FloatGrid_AssignFromAnother(_Underlying *_this, MR.VoxelTraits_MRFloatGrid._Underlying *_other);
            return new(__MR_VoxelTraits_MR_FloatGrid_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelTraits_MRFloatGrid` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelTraits_MRFloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRFloatGrid`/`Const_VoxelTraits_MRFloatGrid` directly.
    public class _InOptMut_VoxelTraits_MRFloatGrid
    {
        public VoxelTraits_MRFloatGrid? Opt;

        public _InOptMut_VoxelTraits_MRFloatGrid() {}
        public _InOptMut_VoxelTraits_MRFloatGrid(VoxelTraits_MRFloatGrid value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelTraits_MRFloatGrid(VoxelTraits_MRFloatGrid value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelTraits_MRFloatGrid` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelTraits_MRFloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelTraits_MRFloatGrid`/`Const_VoxelTraits_MRFloatGrid` to pass it to the function.
    public class _InOptConst_VoxelTraits_MRFloatGrid
    {
        public Const_VoxelTraits_MRFloatGrid? Opt;

        public _InOptConst_VoxelTraits_MRFloatGrid() {}
        public _InOptConst_VoxelTraits_MRFloatGrid(Const_VoxelTraits_MRFloatGrid value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelTraits_MRFloatGrid(Const_VoxelTraits_MRFloatGrid value) {return new(value);}
    }

    /// converts function volume into simple volume
    /// Generated from function `MR::functionVolumeToSimpleVolume`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolumeMinMax_StdString> FunctionVolumeToSimpleVolume(MR.Const_FunctionVolume volume, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_functionVolumeToSimpleVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolumeMinMax_StdString._Underlying *__MR_functionVolumeToSimpleVolume(MR.Const_FunctionVolume._Underlying *volume, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolumeMinMax_StdString(__MR_functionVolumeToSimpleVolume(volume._UnderlyingPtr, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
    }
}
