public static partial class MR
{
    public enum FilterType : byte
    {
        Linear = 0,
        Discrete = 1,
    }

    public enum WrapType : byte
    {
        Repeat = 0,
        Mirror = 1,
        Clamp = 2,
    }

    /// determines how points to be ordered
    public enum Reorder : byte
    {
        ///< the order is not changed
        None = 0,
        ///< the order is determined by lexicographical sorting by coordinates (optimal for uniform sampling)
        Lexicographically = 1,
        ///< the order is determined so to put close in space points in close indices (optimal for compression)
        AABBTree = 2,
    }

    /// determines the weight or mass of each vertex in applications like Laplacian
    public enum VertexMass : int
    {
        /// all vertices have same mass=1
        Unit = 0,
        /// vertex mass depends on local geometry and proportional to the area of first-ring triangles
        NeiArea = 1,
    }

    /// determines the weight of each edge in applications like Laplacian
    public enum EdgeWeights : int
    {
        /// all edges have same weight=1
        Unit = 0,
        /// edge weight depends on local geometry and uses cotangent values
        Cotan = 1,
    }

    /// typically returned from callbacks to control the behavior of main algorithm
    /// This enum is intended to be boolean.
    public enum Processing : byte
    {
        Continue = 0,
        Stop = 1,
    }

    /// the method how to choose between two opposite normal orientations
    public enum OrientNormals : int
    {
        TowardOrigin = 0,
        AwayFromOrigin = 1,
        Smart = 2,
    }

    public enum OffsetMode : int
    {
        ///< create mesh using dual marching cubes from OpenVDB library
        Smooth = 0,
        ///< create mesh using standard marching cubes implemented in MeshLib
        Standard = 1,
        ///< create mesh using standard marching cubes with additional sharpening implemented in MeshLib
        Sharpening = 2,
    }

    /// Type of object coloring,
    /// \note that texture are applied over main coloring
    public enum ColoringType : int
    {
        ///< Use one color for whole object
        SolidColor = 0,
        ///< Use different color (taken from faces colormap) for each primitive
        PrimitivesColorMap = 1,
        ///< Use different color (taken from faces colormap) for each face (primitive for object mesh)
        FacesColorMap = 1,
        ///< Use different color (taken from faces colormap) for each line (primitive for object lines)
        LinesColorMap = 1,
        ///< Use different color (taken from verts colormap) for each vertex
        VertsColorMap = 2,
    }

    public enum UseAABBTree : byte
    {
        // AABB-tree of the mesh will not be used, even if it is available
        No = 0,
        // AABB-tree of the mesh will be used even if it has to be constructed
        Yes = 1,
        // AABB-tree of the mesh will be used if it was previously constructed and available, and will not be used otherwise
        YesIfAlreadyConstructed = 2,
    }

    /// the algorithm to compute approximately geodesic path
    public enum GeodesicPathApprox : byte
    {
        /// compute edge-only path by building it from start and end simultaneously
        DijkstraBiDir = 0,
        /// compute edge-only path using A*-search algorithm
        DijkstraAStar = 1,
        /// use Fast Marching algorithm
        FastMarching = 2,
    }

    /// returns string representation of enum values
    /// Generated from function `MR::asString`.
    public static unsafe byte? AsString(MR.ColoringType ct)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_asString_MR_ColoringType", ExactSpelling = true)]
        extern static byte *__MR_asString_MR_ColoringType(MR.ColoringType ct);
        var __ret = __MR_asString_MR_ColoringType(ct);
        return __ret is not null ? *__ret : null;
    }
}
