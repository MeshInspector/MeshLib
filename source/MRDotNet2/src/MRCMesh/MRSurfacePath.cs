public static partial class MR
{
    public enum PathError : int
    {
        ///< no path can be found from start to end, because they are not from the same connected component
        StartEndNotConnected = 0,
        ///< report to developers for investigation
        InternalError = 1,
    }

    /// Generated from class `MR::ComputeSteepestDescentPathSettings`.
    /// This is the const half of the class.
    public class Const_ComputeSteepestDescentPathSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ComputeSteepestDescentPathSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_ComputeSteepestDescentPathSettings_Destroy(_Underlying *_this);
            __MR_ComputeSteepestDescentPathSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ComputeSteepestDescentPathSettings() {Dispose(false);}

        /// if valid, then the descent is stopped as soon as same triangle with (end) is reached
        public unsafe MR.Const_MeshTriPoint End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_Get_end", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_ComputeSteepestDescentPathSettings_Get_end(_Underlying *_this);
                return new(__MR_ComputeSteepestDescentPathSettings_Get_end(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if not nullptr, then the descent is stopped as soon as any vertex is reached, which is written in *outVertexReached
        public unsafe ref MR.VertId * OutVertexReached
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_Get_outVertexReached", ExactSpelling = true)]
                extern static MR.VertId **__MR_ComputeSteepestDescentPathSettings_Get_outVertexReached(_Underlying *_this);
                return ref *__MR_ComputeSteepestDescentPathSettings_Get_outVertexReached(_UnderlyingPtr);
            }
        }

        /// if not nullptr, then the descent is stopped as soon as any boundary point is reached, which is written in *outBdReached
        public unsafe ref void * OutBdReached
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_Get_outBdReached", ExactSpelling = true)]
                extern static void **__MR_ComputeSteepestDescentPathSettings_Get_outBdReached(_Underlying *_this);
                return ref *__MR_ComputeSteepestDescentPathSettings_Get_outBdReached(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ComputeSteepestDescentPathSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_ComputeSteepestDescentPathSettings_DefaultConstruct();
        }

        /// Constructs `MR::ComputeSteepestDescentPathSettings` elementwise.
        public unsafe Const_ComputeSteepestDescentPathSettings(MR.Const_MeshTriPoint end, MR.Mut_VertId? outVertexReached, MR.EdgePoint? outBdReached) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_ConstructFrom(MR.MeshTriPoint._Underlying *end, MR.Mut_VertId._Underlying *outVertexReached, MR.EdgePoint._Underlying *outBdReached);
            _UnderlyingPtr = __MR_ComputeSteepestDescentPathSettings_ConstructFrom(end._UnderlyingPtr, outVertexReached is not null ? outVertexReached._UnderlyingPtr : null, outBdReached is not null ? outBdReached._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ComputeSteepestDescentPathSettings::ComputeSteepestDescentPathSettings`.
        public unsafe Const_ComputeSteepestDescentPathSettings(MR.Const_ComputeSteepestDescentPathSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_ConstructFromAnother(MR.ComputeSteepestDescentPathSettings._Underlying *_other);
            _UnderlyingPtr = __MR_ComputeSteepestDescentPathSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ComputeSteepestDescentPathSettings`.
    /// This is the non-const half of the class.
    public class ComputeSteepestDescentPathSettings : Const_ComputeSteepestDescentPathSettings
    {
        internal unsafe ComputeSteepestDescentPathSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// if valid, then the descent is stopped as soon as same triangle with (end) is reached
        public new unsafe MR.MeshTriPoint End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_GetMutable_end", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_ComputeSteepestDescentPathSettings_GetMutable_end(_Underlying *_this);
                return new(__MR_ComputeSteepestDescentPathSettings_GetMutable_end(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if not nullptr, then the descent is stopped as soon as any vertex is reached, which is written in *outVertexReached
        public new unsafe ref MR.VertId * OutVertexReached
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_GetMutable_outVertexReached", ExactSpelling = true)]
                extern static MR.VertId **__MR_ComputeSteepestDescentPathSettings_GetMutable_outVertexReached(_Underlying *_this);
                return ref *__MR_ComputeSteepestDescentPathSettings_GetMutable_outVertexReached(_UnderlyingPtr);
            }
        }

        /// if not nullptr, then the descent is stopped as soon as any boundary point is reached, which is written in *outBdReached
        public new unsafe ref void * OutBdReached
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_GetMutable_outBdReached", ExactSpelling = true)]
                extern static void **__MR_ComputeSteepestDescentPathSettings_GetMutable_outBdReached(_Underlying *_this);
                return ref *__MR_ComputeSteepestDescentPathSettings_GetMutable_outBdReached(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ComputeSteepestDescentPathSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_ComputeSteepestDescentPathSettings_DefaultConstruct();
        }

        /// Constructs `MR::ComputeSteepestDescentPathSettings` elementwise.
        public unsafe ComputeSteepestDescentPathSettings(MR.Const_MeshTriPoint end, MR.Mut_VertId? outVertexReached, MR.EdgePoint? outBdReached) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_ConstructFrom(MR.MeshTriPoint._Underlying *end, MR.Mut_VertId._Underlying *outVertexReached, MR.EdgePoint._Underlying *outBdReached);
            _UnderlyingPtr = __MR_ComputeSteepestDescentPathSettings_ConstructFrom(end._UnderlyingPtr, outVertexReached is not null ? outVertexReached._UnderlyingPtr : null, outBdReached is not null ? outBdReached._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ComputeSteepestDescentPathSettings::ComputeSteepestDescentPathSettings`.
        public unsafe ComputeSteepestDescentPathSettings(MR.Const_ComputeSteepestDescentPathSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_ConstructFromAnother(MR.ComputeSteepestDescentPathSettings._Underlying *_other);
            _UnderlyingPtr = __MR_ComputeSteepestDescentPathSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ComputeSteepestDescentPathSettings::operator=`.
        public unsafe MR.ComputeSteepestDescentPathSettings Assign(MR.Const_ComputeSteepestDescentPathSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSteepestDescentPathSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ComputeSteepestDescentPathSettings._Underlying *__MR_ComputeSteepestDescentPathSettings_AssignFromAnother(_Underlying *_this, MR.ComputeSteepestDescentPathSettings._Underlying *_other);
            return new(__MR_ComputeSteepestDescentPathSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ComputeSteepestDescentPathSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ComputeSteepestDescentPathSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ComputeSteepestDescentPathSettings`/`Const_ComputeSteepestDescentPathSettings` directly.
    public class _InOptMut_ComputeSteepestDescentPathSettings
    {
        public ComputeSteepestDescentPathSettings? Opt;

        public _InOptMut_ComputeSteepestDescentPathSettings() {}
        public _InOptMut_ComputeSteepestDescentPathSettings(ComputeSteepestDescentPathSettings value) {Opt = value;}
        public static implicit operator _InOptMut_ComputeSteepestDescentPathSettings(ComputeSteepestDescentPathSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `ComputeSteepestDescentPathSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ComputeSteepestDescentPathSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ComputeSteepestDescentPathSettings`/`Const_ComputeSteepestDescentPathSettings` to pass it to the function.
    public class _InOptConst_ComputeSteepestDescentPathSettings
    {
        public Const_ComputeSteepestDescentPathSettings? Opt;

        public _InOptConst_ComputeSteepestDescentPathSettings() {}
        public _InOptConst_ComputeSteepestDescentPathSettings(Const_ComputeSteepestDescentPathSettings value) {Opt = value;}
        public static implicit operator _InOptConst_ComputeSteepestDescentPathSettings(Const_ComputeSteepestDescentPathSettings value) {return new(value);}
    }

    public enum ExtremeEdgeType : int
    {
        // where the field not-increases both in left and right triangles
        Ridge = 0,
        // where the field not-decreases both in left and right triangles
        Gorge = 1,
    }

    /// Generated from function `MR::toString`.
    public static unsafe MR.Misc._Moved<MR.Std.String> ToString(MR.PathError error)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_toString_MR_PathError", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_toString_MR_PathError(MR.PathError error);
        return MR.Misc.Move(new MR.Std.String(__MR_toString_MR_PathError(error), is_owning: true));
    }

    /// returns intermediate points of the geodesic path from start to end, where it crosses mesh edges;
    /// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument.
    /// It is the same as calling computeFastMarchingPath() then reducePath()
    /// Generated from function `MR::computeSurfacePath`.
    /// Parameter `maxGeodesicIters` defaults to `5`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgePoint_MRPathError> ComputeSurfacePath(MR.Const_MeshPart mp, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint end, int? maxGeodesicIters = null, MR.Const_VertBitSet? vertRegion = null, MR.VertScalars? outSurfaceDistances = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSurfacePath", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_computeSurfacePath(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *end, int *maxGeodesicIters, MR.Const_VertBitSet._Underlying *vertRegion, MR.VertScalars._Underlying *outSurfaceDistances);
        int __deref_maxGeodesicIters = maxGeodesicIters.GetValueOrDefault();
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgePoint_MRPathError(__MR_computeSurfacePath(mp._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, maxGeodesicIters.HasValue ? &__deref_maxGeodesicIters : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null, outSurfaceDistances is not null ? outSurfaceDistances._UnderlyingPtr : null), is_owning: true));
    }

    /// returns intermediate points of the geodesic path from start to end, where it crosses mesh edges;
    /// It is the same as calling computeGeodesicPathApprox() then reducePath()
    /// Generated from function `MR::computeGeodesicPath`.
    /// Parameter `atype` defaults to `GeodesicPathApprox::FastMarching`.
    /// Parameter `maxGeodesicIters` defaults to `100`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgePoint_MRPathError> ComputeGeodesicPath(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint end, MR.GeodesicPathApprox? atype = null, int? maxGeodesicIters = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeGeodesicPath", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_computeGeodesicPath(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *end, MR.GeodesicPathApprox *atype, int *maxGeodesicIters);
        MR.GeodesicPathApprox __deref_atype = atype.GetValueOrDefault();
        int __deref_maxGeodesicIters = maxGeodesicIters.GetValueOrDefault();
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgePoint_MRPathError(__MR_computeGeodesicPath(mesh._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, atype.HasValue ? &__deref_atype : null, maxGeodesicIters.HasValue ? &__deref_maxGeodesicIters : null), is_owning: true));
    }

    /// computes by given method and returns intermediate points of approximately geodesic path from start to end,
    /// every next point is located in the same triangle with the previous point
    /// Generated from function `MR::computeGeodesicPathApprox`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgePoint_MRPathError> ComputeGeodesicPathApprox(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint end, MR.GeodesicPathApprox atype)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeGeodesicPathApprox", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_computeGeodesicPathApprox(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *end, MR.GeodesicPathApprox atype);
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgePoint_MRPathError(__MR_computeGeodesicPathApprox(mesh._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, atype), is_owning: true));
    }

    /// computes by Fast Marching method and returns intermediate points of approximately geodesic path from start to end, where it crosses mesh edges;
    /// the path can be limited to given region: in face-format inside mp, or in vert-format in vertRegion argument
    /// Generated from function `MR::computeFastMarchingPath`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMREdgePoint_MRPathError> ComputeFastMarchingPath(MR.Const_MeshPart mp, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint end, MR.Const_VertBitSet? vertRegion = null, MR.VertScalars? outSurfaceDistances = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeFastMarchingPath", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_computeFastMarchingPath(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *end, MR.Const_VertBitSet._Underlying *vertRegion, MR.VertScalars._Underlying *outSurfaceDistances);
        return MR.Misc.Move(new MR.Expected_StdVectorMREdgePoint_MRPathError(__MR_computeFastMarchingPath(mp._UnderlyingPtr, start._UnderlyingPtr, end._UnderlyingPtr, vertRegion is not null ? vertRegion._UnderlyingPtr : null, outSurfaceDistances is not null ? outSurfaceDistances._UnderlyingPtr : null), is_owning: true));
    }

    /// computes the path (edge points crossed by the path) staring in given point
    /// and moving in each triangle in minus gradient direction of given field;
    /// the path stops when it reaches a local minimum in the field or one of stop conditions in settings
    /// Generated from function `MR::computeSteepestDescentPath`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgePoint> ComputeSteepestDescentPath(MR.Const_MeshPart mp, MR.Const_VertScalars field, MR.Const_MeshTriPoint start, MR.Const_ComputeSteepestDescentPathSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSteepestDescentPath_4", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgePoint._Underlying *__MR_computeSteepestDescentPath_4(MR.Const_MeshPart._Underlying *mp, MR.Const_VertScalars._Underlying *field, MR.Const_MeshTriPoint._Underlying *start, MR.Const_ComputeSteepestDescentPathSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Std.Vector_MREdgePoint(__MR_computeSteepestDescentPath_4(mp._UnderlyingPtr, field._UnderlyingPtr, start._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// computes the path (edge points crossed by the path) staring in given point
    /// and moving in each triangle in minus gradient direction of given field,
    /// and outputs the path in \param outPath if requested;
    /// the path stops when it reaches a local minimum in the field or one of stop conditions in settings
    /// Generated from function `MR::computeSteepestDescentPath`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe void ComputeSteepestDescentPath(MR.Const_MeshPart mp, MR.Const_VertScalars field, MR.Const_MeshTriPoint start, MR.Std.Vector_MREdgePoint? outPath, MR.Const_ComputeSteepestDescentPathSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSteepestDescentPath_5", ExactSpelling = true)]
        extern static void __MR_computeSteepestDescentPath_5(MR.Const_MeshPart._Underlying *mp, MR.Const_VertScalars._Underlying *field, MR.Const_MeshTriPoint._Underlying *start, MR.Std.Vector_MREdgePoint._Underlying *outPath, MR.Const_ComputeSteepestDescentPathSettings._Underlying *settings);
        __MR_computeSteepestDescentPath_5(mp._UnderlyingPtr, field._UnderlyingPtr, start._UnderlyingPtr, outPath is not null ? outPath._UnderlyingPtr : null, settings is not null ? settings._UnderlyingPtr : null);
    }

    /// finds the point along minus maximal gradient on the boundary of first ring boundary around given vertex
    /// Generated from function `MR::findSteepestDescentPoint`.
    public static unsafe MR.EdgePoint FindSteepestDescentPoint(MR.Const_MeshPart mp, MR.Const_VertScalars field, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSteepestDescentPoint_MR_VertId", ExactSpelling = true)]
        extern static MR.EdgePoint._Underlying *__MR_findSteepestDescentPoint_MR_VertId(MR.Const_MeshPart._Underlying *mp, MR.Const_VertScalars._Underlying *field, MR.VertId v);
        return new(__MR_findSteepestDescentPoint_MR_VertId(mp._UnderlyingPtr, field._UnderlyingPtr, v), is_owning: true);
    }

    /// finds the point along minus maximal gradient on the boundary of triangles around given point (the boundary of left and right edge triangles' union in case (ep) is inner edge point)
    /// Generated from function `MR::findSteepestDescentPoint`.
    public static unsafe MR.EdgePoint FindSteepestDescentPoint(MR.Const_MeshPart mp, MR.Const_VertScalars field, MR.Const_EdgePoint ep)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSteepestDescentPoint_MR_EdgePoint", ExactSpelling = true)]
        extern static MR.EdgePoint._Underlying *__MR_findSteepestDescentPoint_MR_EdgePoint(MR.Const_MeshPart._Underlying *mp, MR.Const_VertScalars._Underlying *field, MR.Const_EdgePoint._Underlying *ep);
        return new(__MR_findSteepestDescentPoint_MR_EdgePoint(mp._UnderlyingPtr, field._UnderlyingPtr, ep._UnderlyingPtr), is_owning: true);
    }

    /// finds the point along minus maximal gradient on the boundary of triangles around given point (the boundary of the triangle itself in case (tp) is inner triangle point)
    /// Generated from function `MR::findSteepestDescentPoint`.
    public static unsafe MR.EdgePoint FindSteepestDescentPoint(MR.Const_MeshPart mp, MR.Const_VertScalars field, MR.Const_MeshTriPoint tp)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSteepestDescentPoint_MR_MeshTriPoint", ExactSpelling = true)]
        extern static MR.EdgePoint._Underlying *__MR_findSteepestDescentPoint_MR_MeshTriPoint(MR.Const_MeshPart._Underlying *mp, MR.Const_VertScalars._Underlying *field, MR.Const_MeshTriPoint._Underlying *tp);
        return new(__MR_findSteepestDescentPoint_MR_MeshTriPoint(mp._UnderlyingPtr, field._UnderlyingPtr, tp._UnderlyingPtr), is_owning: true);
    }

    /// computes all edges in the mesh, where the field not-increases both in left and right triangles
    /// Generated from function `MR::findExtremeEdges`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindExtremeEdges(MR.Const_Mesh mesh, MR.Const_VertScalars field, MR.ExtremeEdgeType type)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findExtremeEdges", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_findExtremeEdges(MR.Const_Mesh._Underlying *mesh, MR.Const_VertScalars._Underlying *field, MR.ExtremeEdgeType type);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_findExtremeEdges(mesh._UnderlyingPtr, field._UnderlyingPtr, type), is_owning: true));
    }

    /// for each vertex from (starts) finds the closest vertex from (ends) in geodesic sense
    /// \param vertRegion consider paths going in this region only
    /// Generated from function `MR::computeClosestSurfacePathTargets`.
    public static unsafe MR.Misc._Moved<MR.Phmap.FlatHashMap_MRVertId_MRVertId> ComputeClosestSurfacePathTargets(MR.Const_Mesh mesh, MR.Const_VertBitSet starts, MR.Const_VertBitSet ends, MR.Const_VertBitSet? vertRegion = null, MR.VertScalars? outSurfaceDistances = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeClosestSurfacePathTargets", ExactSpelling = true)]
        extern static MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *__MR_computeClosestSurfacePathTargets(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *starts, MR.Const_VertBitSet._Underlying *ends, MR.Const_VertBitSet._Underlying *vertRegion, MR.VertScalars._Underlying *outSurfaceDistances);
        return MR.Misc.Move(new MR.Phmap.FlatHashMap_MRVertId_MRVertId(__MR_computeClosestSurfacePathTargets(mesh._UnderlyingPtr, starts._UnderlyingPtr, ends._UnderlyingPtr, vertRegion is not null ? vertRegion._UnderlyingPtr : null, outSurfaceDistances is not null ? outSurfaceDistances._UnderlyingPtr : null), is_owning: true));
    }

    /// returns a set of mesh lines passing via most of given vertices in auto-selected order;
    /// the lines try to avoid sharp turns in the vertices
    /// Generated from function `MR::getSurfacePathsViaVertices`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgePoint> GetSurfacePathsViaVertices(MR.Const_Mesh mesh, MR.Const_VertBitSet vs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getSurfacePathsViaVertices", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgePoint._Underlying *__MR_getSurfacePathsViaVertices(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *vs);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgePoint(__MR_getSurfacePathsViaVertices(mesh._UnderlyingPtr, vs._UnderlyingPtr), is_owning: true));
    }

    /// computes the length of surface path
    /// Generated from function `MR::surfacePathLength`.
    public static unsafe float SurfacePathLength(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint surfacePath)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_surfacePathLength", ExactSpelling = true)]
        extern static float __MR_surfacePathLength(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *surfacePath);
        return __MR_surfacePathLength(mesh._UnderlyingPtr, surfacePath._UnderlyingPtr);
    }

    /// converts lines on mesh in 3D contours by computing coordinate of each point
    /// Generated from function `MR::surfacePathToContour3f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> SurfacePathToContour3f(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint line)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_surfacePathToContour3f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_surfacePathToContour3f(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *line);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_surfacePathToContour3f(mesh._UnderlyingPtr, line._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::surfacePathsToContours3f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> SurfacePathsToContours3f(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMREdgePoint lines)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_surfacePathsToContours3f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_surfacePathsToContours3f(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMREdgePoint._Underlying *lines);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_surfacePathsToContours3f(mesh._UnderlyingPtr, lines._UnderlyingPtr), is_owning: true));
    }
}
