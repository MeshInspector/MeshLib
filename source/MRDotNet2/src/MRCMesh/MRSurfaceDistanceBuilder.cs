public static partial class MR
{
    /// Generated from class `MR::VertDistance`.
    /// This is the const half of the class.
    public class Const_VertDistance : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertDistance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_Destroy", ExactSpelling = true)]
            extern static void __MR_VertDistance_Destroy(_Underlying *_this);
            __MR_VertDistance_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertDistance() {Dispose(false);}

        /// vertex in question
        public unsafe MR.Const_VertId Vert
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_Get_vert", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_VertDistance_Get_vert(_Underlying *_this);
                return new(__MR_VertDistance_Get_vert(_UnderlyingPtr), is_owning: false);
            }
        }

        /// best known distance to reach this vertex
        public unsafe float Distance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_Get_distance", ExactSpelling = true)]
                extern static float *__MR_VertDistance_Get_distance(_Underlying *_this);
                return *__MR_VertDistance_Get_distance(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertDistance() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_DefaultConstruct();
            _UnderlyingPtr = __MR_VertDistance_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertDistance::VertDistance`.
        public unsafe Const_VertDistance(MR.Const_VertDistance _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_ConstructFromAnother(MR.VertDistance._Underlying *_other);
            _UnderlyingPtr = __MR_VertDistance_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VertDistance::VertDistance`.
        public unsafe Const_VertDistance(MR.VertId v, float d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_Construct", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_Construct(MR.VertId v, float d);
            _UnderlyingPtr = __MR_VertDistance_Construct(v, d);
        }

        /// smaller distance to be the first
        /// Generated from function `MR::operator<`.
        public static unsafe bool operator<(MR.Const_VertDistance a, MR.Const_VertDistance b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_VertDistance", ExactSpelling = true)]
            extern static byte __MR_less_MR_VertDistance(MR.Const_VertDistance._Underlying *a, MR.Const_VertDistance._Underlying *b);
            return __MR_less_MR_VertDistance(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator>(MR.Const_VertDistance a, MR.Const_VertDistance b)
        {
            return b < a;
        }

        public static unsafe bool operator<=(MR.Const_VertDistance a, MR.Const_VertDistance b)
        {
            return !(b < a);
        }

        public static unsafe bool operator>=(MR.Const_VertDistance a, MR.Const_VertDistance b)
        {
            return !(a < b);
        }
    }

    /// Generated from class `MR::VertDistance`.
    /// This is the non-const half of the class.
    public class VertDistance : Const_VertDistance
    {
        internal unsafe VertDistance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// vertex in question
        public new unsafe MR.Mut_VertId Vert
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_GetMutable_vert", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_VertDistance_GetMutable_vert(_Underlying *_this);
                return new(__MR_VertDistance_GetMutable_vert(_UnderlyingPtr), is_owning: false);
            }
        }

        /// best known distance to reach this vertex
        public new unsafe ref float Distance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_GetMutable_distance", ExactSpelling = true)]
                extern static float *__MR_VertDistance_GetMutable_distance(_Underlying *_this);
                return ref *__MR_VertDistance_GetMutable_distance(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertDistance() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_DefaultConstruct();
            _UnderlyingPtr = __MR_VertDistance_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertDistance::VertDistance`.
        public unsafe VertDistance(MR.Const_VertDistance _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_ConstructFromAnother(MR.VertDistance._Underlying *_other);
            _UnderlyingPtr = __MR_VertDistance_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VertDistance::VertDistance`.
        public unsafe VertDistance(MR.VertId v, float d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_Construct", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_Construct(MR.VertId v, float d);
            _UnderlyingPtr = __MR_VertDistance_Construct(v, d);
        }

        /// Generated from method `MR::VertDistance::operator=`.
        public unsafe MR.VertDistance Assign(MR.Const_VertDistance _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertDistance_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertDistance._Underlying *__MR_VertDistance_AssignFromAnother(_Underlying *_this, MR.VertDistance._Underlying *_other);
            return new(__MR_VertDistance_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VertDistance` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertDistance`/`Const_VertDistance` directly.
    public class _InOptMut_VertDistance
    {
        public VertDistance? Opt;

        public _InOptMut_VertDistance() {}
        public _InOptMut_VertDistance(VertDistance value) {Opt = value;}
        public static implicit operator _InOptMut_VertDistance(VertDistance value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertDistance` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertDistance`/`Const_VertDistance` to pass it to the function.
    public class _InOptConst_VertDistance
    {
        public Const_VertDistance? Opt;

        public _InOptConst_VertDistance() {}
        public _InOptConst_VertDistance(Const_VertDistance value) {Opt = value;}
        public static implicit operator _InOptConst_VertDistance(Const_VertDistance value) {return new(value);}
    }

    /// this class is responsible for iterative construction of distance map along the surface
    /// Generated from class `MR::SurfaceDistanceBuilder`.
    /// This is the const half of the class.
    public class Const_SurfaceDistanceBuilder : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SurfaceDistanceBuilder(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_Destroy", ExactSpelling = true)]
            extern static void __MR_SurfaceDistanceBuilder_Destroy(_Underlying *_this);
            __MR_SurfaceDistanceBuilder_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SurfaceDistanceBuilder() {Dispose(false);}

        /// Generated from constructor `MR::SurfaceDistanceBuilder::SurfaceDistanceBuilder`.
        public unsafe Const_SurfaceDistanceBuilder(MR._ByValue_SurfaceDistanceBuilder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SurfaceDistanceBuilder._Underlying *__MR_SurfaceDistanceBuilder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SurfaceDistanceBuilder._Underlying *_other);
            _UnderlyingPtr = __MR_SurfaceDistanceBuilder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SurfaceDistanceBuilder::SurfaceDistanceBuilder`.
        public unsafe Const_SurfaceDistanceBuilder(MR.Const_Mesh mesh, MR.Const_VertBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_Construct_2", ExactSpelling = true)]
            extern static MR.SurfaceDistanceBuilder._Underlying *__MR_SurfaceDistanceBuilder_Construct_2(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            _UnderlyingPtr = __MR_SurfaceDistanceBuilder_Construct_2(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SurfaceDistanceBuilder::SurfaceDistanceBuilder`.
        public unsafe Const_SurfaceDistanceBuilder(MR.Const_Mesh mesh, MR.Const_Vector3f target, MR.Const_VertBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_Construct_3", ExactSpelling = true)]
            extern static MR.SurfaceDistanceBuilder._Underlying *__MR_SurfaceDistanceBuilder_Construct_3(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector3f._Underlying *target, MR.Const_VertBitSet._Underlying *region);
            _UnderlyingPtr = __MR_SurfaceDistanceBuilder_Construct_3(mesh._UnderlyingPtr, target._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// returns true if further growth is impossible
        /// Generated from method `MR::SurfaceDistanceBuilder::done`.
        public unsafe bool Done()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_done", ExactSpelling = true)]
            extern static byte __MR_SurfaceDistanceBuilder_done(_Underlying *_this);
            return __MR_SurfaceDistanceBuilder_done(_UnderlyingPtr) != 0;
        }

        /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
        /// Generated from method `MR::SurfaceDistanceBuilder::doneDistance`.
        public unsafe float DoneDistance()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_doneDistance", ExactSpelling = true)]
            extern static float __MR_SurfaceDistanceBuilder_doneDistance(_Underlying *_this);
            return __MR_SurfaceDistanceBuilder_doneDistance(_UnderlyingPtr);
        }
    }

    /// this class is responsible for iterative construction of distance map along the surface
    /// Generated from class `MR::SurfaceDistanceBuilder`.
    /// This is the non-const half of the class.
    public class SurfaceDistanceBuilder : Const_SurfaceDistanceBuilder
    {
        internal unsafe SurfaceDistanceBuilder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::SurfaceDistanceBuilder::SurfaceDistanceBuilder`.
        public unsafe SurfaceDistanceBuilder(MR._ByValue_SurfaceDistanceBuilder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SurfaceDistanceBuilder._Underlying *__MR_SurfaceDistanceBuilder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SurfaceDistanceBuilder._Underlying *_other);
            _UnderlyingPtr = __MR_SurfaceDistanceBuilder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SurfaceDistanceBuilder::SurfaceDistanceBuilder`.
        public unsafe SurfaceDistanceBuilder(MR.Const_Mesh mesh, MR.Const_VertBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_Construct_2", ExactSpelling = true)]
            extern static MR.SurfaceDistanceBuilder._Underlying *__MR_SurfaceDistanceBuilder_Construct_2(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            _UnderlyingPtr = __MR_SurfaceDistanceBuilder_Construct_2(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SurfaceDistanceBuilder::SurfaceDistanceBuilder`.
        public unsafe SurfaceDistanceBuilder(MR.Const_Mesh mesh, MR.Const_Vector3f target, MR.Const_VertBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_Construct_3", ExactSpelling = true)]
            extern static MR.SurfaceDistanceBuilder._Underlying *__MR_SurfaceDistanceBuilder_Construct_3(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector3f._Underlying *target, MR.Const_VertBitSet._Underlying *region);
            _UnderlyingPtr = __MR_SurfaceDistanceBuilder_Construct_3(mesh._UnderlyingPtr, target._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// initiates distance construction from given vertices with known start distance in all of them (region vertices will NOT be returned by growOne)
        /// Generated from method `MR::SurfaceDistanceBuilder::addStartRegion`.
        public unsafe void AddStartRegion(MR.Const_VertBitSet region, float startDistance)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_addStartRegion", ExactSpelling = true)]
            extern static void __MR_SurfaceDistanceBuilder_addStartRegion(_Underlying *_this, MR.Const_VertBitSet._Underlying *region, float startDistance);
            __MR_SurfaceDistanceBuilder_addStartRegion(_UnderlyingPtr, region._UnderlyingPtr, startDistance);
        }

        /// initiates distance construction from given start vertices with values in them (these vertices will NOT be returned by growOne if values in them are not decreased)
        /// Generated from method `MR::SurfaceDistanceBuilder::addStartVertices`.
        public unsafe void AddStartVertices(MR.Phmap.Const_FlatHashMap_MRVertId_Float startVertices)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_addStartVertices", ExactSpelling = true)]
            extern static void __MR_SurfaceDistanceBuilder_addStartVertices(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRVertId_Float._Underlying *startVertices);
            __MR_SurfaceDistanceBuilder_addStartVertices(_UnderlyingPtr, startVertices._UnderlyingPtr);
        }

        /// initiates distance construction from triangle vertices surrounding given start point (they all will be returned by growOne)
        /// Generated from method `MR::SurfaceDistanceBuilder::addStart`.
        public unsafe void AddStart(MR.Const_MeshTriPoint start)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_addStart", ExactSpelling = true)]
            extern static void __MR_SurfaceDistanceBuilder_addStart(_Underlying *_this, MR.Const_MeshTriPoint._Underlying *start);
            __MR_SurfaceDistanceBuilder_addStart(_UnderlyingPtr, start._UnderlyingPtr);
        }

        /// the maximum amount of times vertex distance can be updated in [1,255], 3 by default;
        /// the more the better obtuse triangles are handled
        /// Generated from method `MR::SurfaceDistanceBuilder::setMaxVertUpdates`.
        public unsafe void SetMaxVertUpdates(int v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_setMaxVertUpdates", ExactSpelling = true)]
            extern static void __MR_SurfaceDistanceBuilder_setMaxVertUpdates(_Underlying *_this, int v);
            __MR_SurfaceDistanceBuilder_setMaxVertUpdates(_UnderlyingPtr, v);
        }

        /// processes one more candidate vertex, which is returned
        /// Generated from method `MR::SurfaceDistanceBuilder::growOne`.
        public unsafe MR.VertId GrowOne()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_growOne", ExactSpelling = true)]
            extern static MR.VertId __MR_SurfaceDistanceBuilder_growOne(_Underlying *_this);
            return __MR_SurfaceDistanceBuilder_growOne(_UnderlyingPtr);
        }

        /// takes ownership over constructed distance map
        /// Generated from method `MR::SurfaceDistanceBuilder::takeDistanceMap`.
        public unsafe MR.Misc._Moved<MR.VertScalars> TakeDistanceMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SurfaceDistanceBuilder_takeDistanceMap", ExactSpelling = true)]
            extern static MR.VertScalars._Underlying *__MR_SurfaceDistanceBuilder_takeDistanceMap(_Underlying *_this);
            return MR.Misc.Move(new MR.VertScalars(__MR_SurfaceDistanceBuilder_takeDistanceMap(_UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `SurfaceDistanceBuilder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SurfaceDistanceBuilder`/`Const_SurfaceDistanceBuilder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SurfaceDistanceBuilder
    {
        internal readonly Const_SurfaceDistanceBuilder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SurfaceDistanceBuilder(Const_SurfaceDistanceBuilder new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SurfaceDistanceBuilder(Const_SurfaceDistanceBuilder arg) {return new(arg);}
        public _ByValue_SurfaceDistanceBuilder(MR.Misc._Moved<SurfaceDistanceBuilder> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SurfaceDistanceBuilder(MR.Misc._Moved<SurfaceDistanceBuilder> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SurfaceDistanceBuilder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SurfaceDistanceBuilder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SurfaceDistanceBuilder`/`Const_SurfaceDistanceBuilder` directly.
    public class _InOptMut_SurfaceDistanceBuilder
    {
        public SurfaceDistanceBuilder? Opt;

        public _InOptMut_SurfaceDistanceBuilder() {}
        public _InOptMut_SurfaceDistanceBuilder(SurfaceDistanceBuilder value) {Opt = value;}
        public static implicit operator _InOptMut_SurfaceDistanceBuilder(SurfaceDistanceBuilder value) {return new(value);}
    }

    /// This is used for optional parameters of class `SurfaceDistanceBuilder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SurfaceDistanceBuilder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SurfaceDistanceBuilder`/`Const_SurfaceDistanceBuilder` to pass it to the function.
    public class _InOptConst_SurfaceDistanceBuilder
    {
        public Const_SurfaceDistanceBuilder? Opt;

        public _InOptConst_SurfaceDistanceBuilder() {}
        public _InOptConst_SurfaceDistanceBuilder(Const_SurfaceDistanceBuilder value) {Opt = value;}
        public static implicit operator _InOptConst_SurfaceDistanceBuilder(Const_SurfaceDistanceBuilder value) {return new(value);}
    }
}
