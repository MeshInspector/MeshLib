using System;
using System.Runtime.InteropServices;
using static MR.DotNet.AffineXf3f;
using static MR.DotNet.Vector3f;
//using static MR.DotNet.Vector3f;

namespace MR.DotNet
{

    using VertId = int;
    using EdgeId = int;
    using FaceId = int;

    using VertCoordsReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<Vector3f>;
    using VertCoords = System.Collections.Generic.List<Vector3f>;
    using TriangulationReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<ThreeVertIds>;
    using Triangulation = System.Collections.Generic.List<ThreeVertIds>;
    using EdgePathReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<int>;
    using EdgePath = System.Collections.Generic.List<int>;

    public struct ThreeVertIds
    {
        public VertId v0;
        public VertId v1;
        public VertId v2;

        public ThreeVertIds(VertId v0_, VertId v1_, VertId v2_)
        {
            v0 = v0_;
            v1 = v1_;
            v2 = v2_;
        }
    };

    #region C_STRUCTS
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRVertId
    {
        public int id;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MREdgeId
    {
        public int id;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRFaceId
    {
        public int id;
    }

    /// parameters for \ref mrMakeTorus
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMakeTorusParameters
    {
        public float primaryRadius;
        public float secondaryRadius;
        public int primaryResolution;
        public int secondaryResolution;
        // TODO: points
    };

    /// parameters for \ref mrMakeSphere
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRSphereParams
    {
        public float radius;
        public int numMeshVertices;
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRTriangulation
    {
        public IntPtr data;
        public ulong size;
        public IntPtr reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MREdgePath
    {
        public IntPtr data;
        public ulong size;
        public IntPtr reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRPointOnFace
    {
        public MRFaceId face;
        public MRVector3f point;
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRTriPointf
    {
        /// barycentric coordinates:
        /// a+b in [0,1], a+b=0 => point is in v0, a+b=1 => point is on [v1,v2] edge
        /// a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public float a;
        /// b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public float b;
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshTriPoint
    {
        /// left face of this edge is considered
        public MREdgeId e;
        /// barycentric coordinates
        /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
        /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
        /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
        public MRTriPointf bary;
    };

    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshProjectionResult
    {
        /// the closest point on mesh, transformed by xf if it is given
        public MRPointOnFace proj;
        /// its barycentric representation
        public MRMeshTriPoint mtp;
        /// squared distance from pt to proj
        public float distSq;
    };

    /// optional parameters for \ref mrFindProjection
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRFindProjectionParameters
    {
        /// upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
        public float upDistLimitSq;
        /// mesh-to-point transformation, if not specified then identity transformation is assumed
        public IntPtr xf;
        /// low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
        public float loDistLimitSq;
        // TODO: validFaces
        // TODO: validProjections
    };
    [StructLayout(LayoutKind.Sequential)]
    internal struct MRMeshPart
    {
        public IntPtr mesh;
        public IntPtr region;
    };


    #endregion

    public interface MeshOrPoints
    {
        public VertCoordsReadOnly Points { get; }
        public BitSetReadOnly ValidPoints { get; }
        public Box3f BoundingBox { get; }
    };

    /// holds together mesh/point cloud and its transformation
    public class MeshOrPointsXf
    {
        public MeshOrPoints obj;
        public AffineXf3f xf;

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshOrPointsXfFromMesh(IntPtr mesh, ref MRAffineXf3f xf );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshOrPointsXfFromPointCloud(IntPtr pc, ref MRAffineXf3f xf );

        /// destructs a MeshOrPointsXf object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshOrPointsXfFree(IntPtr mp);

        public MeshOrPointsXf(MeshOrPoints obj, AffineXf3f xf)
        {
            this.obj = obj;
            this.xf = xf;

            if ( obj is Mesh )
                mrMeshOrPointsXf_ = mrMeshOrPointsXfFromMesh((obj as Mesh).mesh_, ref xf.xf_);

            if ( obj is PointCloud )
                mrMeshOrPointsXf_ = mrMeshOrPointsXfFromPointCloud((obj as PointCloud).pc_, ref xf.xf_);
        }

        ~MeshOrPointsXf()
        {
            mrMeshOrPointsXfFree(mrMeshOrPointsXf_);
        }

        internal IntPtr mrMeshOrPointsXf_;
    }

    public class Mesh : MeshOrPoints
    {
        #region C_FUNCTIONS

        /// tightly packs all arrays eliminating lone edges and invalid faces and vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshTopologyPack(IntPtr top);

        /// returns cached set of all valid vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshTopologyGetValidVerts(IntPtr top);

        /// returns cached set of all valid faces
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshTopologyGetValidFaces(IntPtr top);


        /// returns three vertex ids for valid triangles (which can be accessed by FaceId),
        /// vertex ids for invalid triangles are undefined, and shall not be read
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ref MRTriangulation mrMeshTopologyGetTriangulation(IntPtr top);

        /// returns the number of face records including invalid ones
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrMeshTopologyFaceSize(IntPtr top);
        /// returns one edge with no valid left face for every boundary in the mesh
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ref MREdgePath mrMeshTopologyFindHoleRepresentiveEdges(IntPtr top);

        /// gets 3 vertices of given triangular face;
        /// the vertices are returned in counter-clockwise order if look from mesh outside
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshTopologyGetLeftTriVerts(IntPtr top, MREdgeId a, ref MRVertId v0, ref MRVertId v1, ref MRVertId v2);

        /// returns the number of hole loops in the mesh;
        /// \param holeRepresentativeEdges optional output of the smallest edge id with no valid left face in every hole
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern int mrMeshTopologyFindNumHoles(IntPtr top, IntPtr holeRepresentativeEdges);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshFromTriangles(IntPtr vertexCoordinates, ulong vertexCoordinatesNum, IntPtr t, ulong tNum);

        /// constructs a mesh from vertex coordinates and a set of triangles with given ids;
        /// unlike simple \ref mrMeshFromTriangles it tries to resolve non-manifold vertices by creating duplicate vertices
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshFromTrianglesDuplicatingNonManifoldVertices(IntPtr vertexCoordinates, ulong vertexCoordinatesNum, IntPtr t, ulong tNum);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshPoints(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrMeshPointsNum(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeCube(ref MRVector3f size, ref MRVector3f baseCoords);

        /// initializes a default instance
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRMakeTorusParameters mrMakeTorusParametersNew();

        /// creates a mesh representing a torus
        /// Z is symmetry axis of this torus
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeTorus(ref MRMakeTorusParameters parameters);

        /// initializes a default instance <summary>
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRSphereParams mrSphereParamsNew();

        /// creates a mesh of sphere with irregular triangulation
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMakeSphere(ref MRSphereParams parameters);

        /// passes through all valid vertices and finds the minimal bounding box containing all of them;
        /// if toWorld transformation is given then returns minimal bounding box in world space
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern Box3f.MRBox3f mrMeshComputeBoundingBox(IntPtr mesh, IntPtr toWorld);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshTransform(IntPtr mesh, ref MRAffineXf3f xf, IntPtr region);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern double mrMeshVolume(IntPtr mesh, IntPtr region);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshPackOptimally(IntPtr mesh, bool preserveAABBTree);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrMeshTopology(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrMeshFree(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern IntPtr mrMeshLoadFromAnySupportedFormat(string file, IntPtr* errorStr);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern void mrMeshSaveToAnySupportedFormat(IntPtr mesh, string file, IntPtr* errorStr);

        /// creates a default instance
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRFindProjectionParameters mrFindProjectionParametersNew();

        /// computes the closest point on mesh (or its region) to given point
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRMeshProjectionResult mrFindProjection(ref MRVector3f pt, ref MRMeshPart mp, ref MRFindProjectionParameters parameters);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// gets total length of the string
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrStringSize(IntPtr str);

        /// deallocates the string object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrStringFree(IntPtr str);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrLoadIOExtras();

        #endregion
        #region Constructors

        internal Mesh(IntPtr mesh)
        {
            mesh_ = mesh;
            meshTopology_ = mrMeshTopology(mesh);
        }

        ~Mesh()
        {
            mrMeshFree(mesh_);
        }
        #endregion

        #region Properties
        public VertCoordsReadOnly Points
        {
            get
            {
                if (points_ is null)
                {
                    int numPoints = (int)mrMeshPointsNum(mesh_);
                    points_ = new VertCoords(numPoints);
                    int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));

                    var pointsPtr = mrMeshPoints(mesh_);
                    for (int i = 0; i < numPoints; i++)
                    {
                        IntPtr currentPointPtr = IntPtr.Add(pointsPtr, i * sizeOfVector3f);
                        var point = Marshal.PtrToStructure<MRVector3f>(currentPointPtr);
                        points_.Add(new Vector3f(point));
                    }
                }

                return points_.AsReadOnly();
            }
        }

        public BitSetReadOnly ValidPoints
        {
            get
            {
                if (validPoints_ is null)
                {
                    validPoints_ = new BitSet(mrMeshTopologyGetValidVerts(meshTopology_));
                }
                return validPoints_;
            }
        }

        public Box3f BoundingBox
        {
            get
            {
                if (boundingBox_ is null)
                {
                    boundingBox_ = new Box3f(mrMeshComputeBoundingBox(meshTopology_, (IntPtr)null));
                }

                return boundingBox_;
            }
        }

        public BitSetReadOnly ValidFaces
        {
            get
            {
                if (validFaces_ is null)
                {
                    validFaces_ = new BitSet(mrMeshTopologyGetValidFaces(meshTopology_));
                }
                return validFaces_;
            }
        }

        public TriangulationReadOnly Triangulation
        {
            get
            {
                if (triangulation_ is null)
                {
                    var mrtriangulation = mrMeshTopologyGetTriangulation(meshTopology_);
                    triangulation_ = new Triangulation((int)mrtriangulation.size);
                    int sizeOfThreeVertIds = Marshal.SizeOf(typeof(ThreeVertIds));

                    var triangulationPtr = mrtriangulation.data;
                    for (int i = 0; i < triangulation_.Capacity; i++)
                    {
                        IntPtr currentTriangulationPtr = IntPtr.Add(triangulationPtr, i * sizeOfThreeVertIds);
                        triangulation_.Add(Marshal.PtrToStructure<ThreeVertIds>(currentTriangulationPtr));
                    }
                }
                return triangulation_.AsReadOnly();
            }
        }

        public EdgePathReadOnly HoleRepresentiveEdges
        {
            get
            {
                if (holeRepresentiveEdges_ is null)
                {
                    var mrEdges = mrMeshTopologyFindHoleRepresentiveEdges(meshTopology_);
                    holeRepresentiveEdges_ = new EdgePath((int)mrEdges.size);
                    int sizeOfEdgeId = Marshal.SizeOf(typeof(EdgeId));

                    var edgesPtr = mrEdges.data;
                    for (int i = 0; i < holeRepresentiveEdges_.Count; i++)
                    {
                        IntPtr currentEdgePtr = IntPtr.Add(edgesPtr, i * sizeOfEdgeId);
                        holeRepresentiveEdges_[i] = Marshal.PtrToStructure<EdgeId>(currentEdgePtr);
                    }
                }

                return holeRepresentiveEdges_.AsReadOnly();
            }
        }
        #endregion // Properties
        #region Methods

        public VertId[] GetLeftTriVerts(EdgeId edgeId)
        {
            VertId[] res = new VertId[3];
            MRVertId v0 = new MRVertId();
            MRVertId v1 = new MRVertId();
            MRVertId v2 = new MRVertId();

            MREdgeId mrEdgeId = new MREdgeId();
            mrEdgeId.id = edgeId;

            mrMeshTopologyGetLeftTriVerts(meshTopology_, mrEdgeId, ref v0, ref v1, ref v2);
            res[0] = (VertId)v0.id;
            res[1] = (VertId)v1.id;
            res[2] = (VertId)v2.id;

            return res;
        }

        public void Transform(AffineXf3f xf)
        {
            mrMeshTransform(mesh_, ref xf.xf_, (IntPtr)null);
            clearManagedResources();
        }

        public void Transform(AffineXf3f xf, BitSet region)
        {
            mrMeshTransform(mesh_, ref xf.xf_, region.bs_);
            clearManagedResources();
        }

        public void PackOptimally()
        {
            mrMeshPackOptimally(mesh_, true);
            clearManagedResources();
        }

        public double Volume()
        {
            return mrMeshVolume(mesh_, (IntPtr)null);
        }

        public double Volume(BitSet region)
        {
            return mrMeshVolume(mesh_, region.bs_);
        }

        #endregion

        #region Create

        public static Mesh FromTriangles(VertCoords points, Triangulation triangles)
        {
            int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));
            IntPtr nativePoints = Marshal.AllocHGlobal(points.Count * sizeOfVector3f);

            int sizeOfThreeVertIds = Marshal.SizeOf(typeof(ThreeVertIds));
            IntPtr nativeTriangles = Marshal.AllocHGlobal(triangles.Count * sizeOfThreeVertIds);

            try
            {
                for (int i = 0; i < points.Count; i++)
                {
                    Marshal.StructureToPtr(points[i].vec_, IntPtr.Add(nativePoints, i * sizeOfVector3f), false);
                }

                for (int i = 0; i < triangles.Count; i++)
                {
                    Marshal.StructureToPtr(triangles[i], IntPtr.Add(nativeTriangles, i * sizeOfThreeVertIds), false);
                }

                return new Mesh(mrMeshFromTriangles(nativePoints, (ulong)points.Count, nativeTriangles, (ulong)triangles.Count));
            }
            finally
            {
                Marshal.FreeHGlobal(nativePoints);
                Marshal.FreeHGlobal(nativeTriangles);
            }
        }

        public static Mesh FromTrianglesDuplicatingNonManifoldVertices(VertCoords points, Triangulation triangles)
        {
            int sizeOfVector3f = Marshal.SizeOf(typeof(Vector3f));
            IntPtr nativePoints = Marshal.AllocHGlobal(points.Count * sizeOfVector3f);

            int sizeOfThreeVertIds = Marshal.SizeOf(typeof(ThreeVertIds));
            IntPtr nativeTriangles = Marshal.AllocHGlobal(triangles.Count * sizeOfThreeVertIds);

            try
            {
                for (int i = 0; i < points.Count; i++)
                {
                    Marshal.StructureToPtr(points[i], IntPtr.Add(nativePoints, i * sizeOfVector3f), false);
                }

                for (int i = 0; i < triangles.Count; i++)
                {
                    Marshal.StructureToPtr(triangles[i], IntPtr.Add(nativeTriangles, i * sizeOfThreeVertIds), false);
                }

                return new Mesh(mrMeshFromTrianglesDuplicatingNonManifoldVertices(nativePoints, (ulong)points.Count, nativeTriangles, (ulong)triangles.Count));
            }
            finally
            {
                Marshal.FreeHGlobal(nativePoints);
                Marshal.FreeHGlobal(nativeTriangles);
            }
        }

        unsafe public static Mesh FromAnySupportedFormat(string path)
        {
            mrLoadIOExtras();

            IntPtr errString = new IntPtr();
            var mesh = mrMeshLoadFromAnySupportedFormat(path, &errString);

            if (errString != IntPtr.Zero)
            {
                var errData = mrStringData(errString);
                string errorMessage = Marshal.PtrToStringAnsi(errData);
                throw new SystemException(errorMessage);
            }

            return new Mesh(mesh);
        }

        unsafe public static void ToAnySupportedFormat(Mesh mesh, string path)
        {
            mrLoadIOExtras();

            IntPtr errString = new IntPtr();
            mrMeshSaveToAnySupportedFormat(mesh.mesh_, path, &errString);
            if (errString != IntPtr.Zero)
            {
                var errData = mrStringData(errString);
                string errorMessage = Marshal.PtrToStringAnsi(errData);
                throw new SystemException(errorMessage);
            }
        }

        public static Mesh MakeCube(Vector3f size, Vector3f baseCoords)
        {
            return new Mesh(mrMakeCube(ref size.vec_, ref baseCoords.vec_));
        }

        public static Mesh MakeSphere(float radius, int vertexCount)
        {
            MRSphereParams mrSphereParams = new MRSphereParams();
            mrSphereParams.radius = radius;
            mrSphereParams.numMeshVertices = vertexCount;
            return new Mesh(mrMakeSphere(ref mrSphereParams));
        }

        public static Mesh MakeTorus(float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution)
        {
            MRMakeTorusParameters mrMakeTorusParameters = new MRMakeTorusParameters();
            mrMakeTorusParameters.primaryRadius = primaryRadius;
            mrMakeTorusParameters.secondaryRadius = secondaryRadius;
            mrMakeTorusParameters.primaryResolution = primaryResolution;
            mrMakeTorusParameters.secondaryResolution = secondaryResolution;
            return new Mesh(mrMakeTorus(ref mrMakeTorusParameters));
        }

        #endregion
        #region Find projection
        static public MeshProjectionResult FindProjection(Vector3f point, MeshPart meshPart, float maxDistanceSquared = float.MaxValue, AffineXf3f? xf = null, float minDistanceSquared = 0.0f)
        {
            MRFindProjectionParameters mrParams = new MRFindProjectionParameters();
            mrParams.loDistLimitSq = minDistanceSquared;
            mrParams.upDistLimitSq = maxDistanceSquared;
            mrParams.xf = xf is not null ? xf.XfAddr() : (IntPtr)null;

            var mrRes = mrFindProjection(ref point.vec_, ref meshPart.mrMeshPart, ref mrParams);

            MeshProjectionResult result = new MeshProjectionResult();
            result.distanceSquared = mrRes.distSq;

            result.pointOnFace = new PointOnFace();
            result.pointOnFace.point = new Vector3f(mrRes.proj.point);
            result.pointOnFace.faceId = mrRes.proj.face.id;

            result.meshTriPoint = new MeshTriPoint();
            result.meshTriPoint.e = mrRes.mtp.e.id;
            result.meshTriPoint.bary.a = mrRes.mtp.bary.a;
            result.meshTriPoint.bary.b = mrRes.mtp.bary.b;

            return result;
        }
        #endregion


        #region Private fields

        void clearManagedResources()
        {
            points_ = null;
            validPoints_ = null;
            validFaces_ = null;
            triangulation_ = null;
            holeRepresentiveEdges_ = null;
            boundingBox_ = null;
        }


        internal IntPtr mesh_;
        private IntPtr meshTopology_;

        private VertCoords? points_;
        private BitSet? validPoints_;
        private BitSet? validFaces_;
        private Triangulation? triangulation_;
        private EdgePath? holeRepresentiveEdges_;
        private Box3f? boundingBox_;
        #endregion

    }
    public struct TriPoint
    {
        ///< a in [0,1], a=0 => point is on [v2,v0] edge, a=1 => point is in v1
        public float a;
        ///< b in [0,1], b=0 => point is on [v0,v1] edge, b=1 => point is in v2
        public float b;
        /// given three values in three vertices, computes interpolated value at this barycentric coordinates
        public Vector3f Interpolate(Vector3f p0, Vector3f p1, Vector3f p2)
        {
            return p0 * (1 - a - b) + a * p1 + b * p2;
        }
    };

    public struct MeshTriPoint
    {
        public EdgeId e;
        public TriPoint bary;
    }

    public struct PointOnFace
    {
        public FaceId faceId;
        public Vector3f point;
    };

    public struct MeshProjectionResult
    {
        public PointOnFace pointOnFace;
        public MeshTriPoint meshTriPoint;
        public float distanceSquared;
    };

    public class MeshPart
    {
        public Mesh mesh;
        public BitSet? region;

        internal MRMeshPart mrMeshPart;
        public MeshPart(Mesh mesh, BitSet? region = null)
        {
            this.mesh = mesh;
            this.region = region;

            mrMeshPart.mesh = mesh.mesh_;
            mrMeshPart.region = region is null ? (IntPtr)null : region.bs_;
        }
    };
}
