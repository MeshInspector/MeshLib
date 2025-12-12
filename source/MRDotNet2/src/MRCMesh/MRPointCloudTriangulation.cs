public static partial class MR
{
    /**
    * \brief Parameters of point cloud triangulation
    *
    *
    * \sa \ref triangulatePointCloud
    */
    /// Generated from class `MR::TriangulationParameters`.
    /// This is the const half of the class.
    public class Const_TriangulationParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TriangulationParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_TriangulationParameters_Destroy(_Underlying *_this);
            __MR_TriangulationParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TriangulationParameters() {Dispose(false);}

        /**
        * \brief The number of nearest neighbor points to use for building of local triangulation
        * \note Too small value can make not optimal triangulation and additional holes\n
        * Too big value increases difficulty of optimization and decreases performance
        *
        <table border=0>
        <caption id="TriangulationParameters::numNeighbours_examples"></caption>
        <tr>
        <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
        <td> \image html triangulate/triangulate_2.png "Too small value" width = 350cm </td>
        </tr>
        </table>
        */
        public unsafe int NumNeighbours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_numNeighbours", ExactSpelling = true)]
                extern static int *__MR_TriangulationParameters_Get_numNeighbours(_Underlying *_this);
                return *__MR_TriangulationParameters_Get_numNeighbours(_UnderlyingPtr);
            }
        }

        /**
        * Radius of neighborhood around each point to consider for building local triangulation.
        * This is an alternative to numNeighbours parameter.
        * Please set to positive value only one of them.
        */
        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_radius", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_Get_radius(_Underlying *_this);
                return *__MR_TriangulationParameters_Get_radius(_UnderlyingPtr);
            }
        }

        /**
        * \brief Critical angle of triangles in local triangulation (angle between triangles in fan should be less then this value)
        *
        <table border=0>
        <caption id="TriangulationParameters::critAngle_examples"></caption>
        <tr>
        <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
        <td> \image html triangulate/triangulate_4.png "Too small value" width = 350cm </td>
        </tr>
        </table>
        */
        public unsafe float CritAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_critAngle", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_Get_critAngle(_Underlying *_this);
                return *__MR_TriangulationParameters_Get_critAngle(_UnderlyingPtr);
            }
        }

        /// the vertex is considered as boundary if its neighbor ring has angle more than this value
        public unsafe float BoundaryAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_boundaryAngle", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_Get_boundaryAngle(_Underlying *_this);
                return *__MR_TriangulationParameters_Get_boundaryAngle(_UnderlyingPtr);
            }
        }

        /**
        * \brief Critical length of hole (all holes with length less then this value will be filled)
        * \details If value is subzero it is set automaticly to 0.7*bbox.diagonal()
        */
        public unsafe float CritHoleLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_critHoleLength", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_Get_critHoleLength(_Underlying *_this);
                return *__MR_TriangulationParameters_Get_critHoleLength(_UnderlyingPtr);
            }
        }

        /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
        public unsafe bool AutomaticRadiusIncrease
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_automaticRadiusIncrease", ExactSpelling = true)]
                extern static bool *__MR_TriangulationParameters_Get_automaticRadiusIncrease(_Underlying *_this);
                return *__MR_TriangulationParameters_Get_automaticRadiusIncrease(_UnderlyingPtr);
            }
        }

        /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
        public unsafe ref readonly void * SearchNeighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_Get_searchNeighbors", ExactSpelling = true)]
                extern static void **__MR_TriangulationParameters_Get_searchNeighbors(_Underlying *_this);
                return ref *__MR_TriangulationParameters_Get_searchNeighbors(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TriangulationParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_TriangulationParameters_DefaultConstruct();
        }

        /// Constructs `MR::TriangulationParameters` elementwise.
        public unsafe Const_TriangulationParameters(int numNeighbours, float radius, float critAngle, float boundaryAngle, float critHoleLength, bool automaticRadiusIncrease, MR.Const_PointCloud? searchNeighbors) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_ConstructFrom(int numNeighbours, float radius, float critAngle, float boundaryAngle, float critHoleLength, byte automaticRadiusIncrease, MR.Const_PointCloud._Underlying *searchNeighbors);
            _UnderlyingPtr = __MR_TriangulationParameters_ConstructFrom(numNeighbours, radius, critAngle, boundaryAngle, critHoleLength, automaticRadiusIncrease ? (byte)1 : (byte)0, searchNeighbors is not null ? searchNeighbors._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TriangulationParameters::TriangulationParameters`.
        public unsafe Const_TriangulationParameters(MR.Const_TriangulationParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_ConstructFromAnother(MR.TriangulationParameters._Underlying *_other);
            _UnderlyingPtr = __MR_TriangulationParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /**
    * \brief Parameters of point cloud triangulation
    *
    *
    * \sa \ref triangulatePointCloud
    */
    /// Generated from class `MR::TriangulationParameters`.
    /// This is the non-const half of the class.
    public class TriangulationParameters : Const_TriangulationParameters
    {
        internal unsafe TriangulationParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /**
        * \brief The number of nearest neighbor points to use for building of local triangulation
        * \note Too small value can make not optimal triangulation and additional holes\n
        * Too big value increases difficulty of optimization and decreases performance
        *
        <table border=0>
        <caption id="TriangulationParameters::numNeighbours_examples"></caption>
        <tr>
        <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
        <td> \image html triangulate/triangulate_2.png "Too small value" width = 350cm </td>
        </tr>
        </table>
        */
        public new unsafe ref int NumNeighbours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_numNeighbours", ExactSpelling = true)]
                extern static int *__MR_TriangulationParameters_GetMutable_numNeighbours(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_numNeighbours(_UnderlyingPtr);
            }
        }

        /**
        * Radius of neighborhood around each point to consider for building local triangulation.
        * This is an alternative to numNeighbours parameter.
        * Please set to positive value only one of them.
        */
        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_GetMutable_radius(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_radius(_UnderlyingPtr);
            }
        }

        /**
        * \brief Critical angle of triangles in local triangulation (angle between triangles in fan should be less then this value)
        *
        <table border=0>
        <caption id="TriangulationParameters::critAngle_examples"></caption>
        <tr>
        <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
        <td> \image html triangulate/triangulate_4.png "Too small value" width = 350cm </td>
        </tr>
        </table>
        */
        public new unsafe ref float CritAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_critAngle", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_GetMutable_critAngle(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_critAngle(_UnderlyingPtr);
            }
        }

        /// the vertex is considered as boundary if its neighbor ring has angle more than this value
        public new unsafe ref float BoundaryAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_boundaryAngle", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_GetMutable_boundaryAngle(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_boundaryAngle(_UnderlyingPtr);
            }
        }

        /**
        * \brief Critical length of hole (all holes with length less then this value will be filled)
        * \details If value is subzero it is set automaticly to 0.7*bbox.diagonal()
        */
        public new unsafe ref float CritHoleLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_critHoleLength", ExactSpelling = true)]
                extern static float *__MR_TriangulationParameters_GetMutable_critHoleLength(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_critHoleLength(_UnderlyingPtr);
            }
        }

        /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
        public new unsafe ref bool AutomaticRadiusIncrease
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_automaticRadiusIncrease", ExactSpelling = true)]
                extern static bool *__MR_TriangulationParameters_GetMutable_automaticRadiusIncrease(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_automaticRadiusIncrease(_UnderlyingPtr);
            }
        }

        /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
        public new unsafe ref readonly void * SearchNeighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_GetMutable_searchNeighbors", ExactSpelling = true)]
                extern static void **__MR_TriangulationParameters_GetMutable_searchNeighbors(_Underlying *_this);
                return ref *__MR_TriangulationParameters_GetMutable_searchNeighbors(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TriangulationParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_TriangulationParameters_DefaultConstruct();
        }

        /// Constructs `MR::TriangulationParameters` elementwise.
        public unsafe TriangulationParameters(int numNeighbours, float radius, float critAngle, float boundaryAngle, float critHoleLength, bool automaticRadiusIncrease, MR.Const_PointCloud? searchNeighbors) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_ConstructFrom(int numNeighbours, float radius, float critAngle, float boundaryAngle, float critHoleLength, byte automaticRadiusIncrease, MR.Const_PointCloud._Underlying *searchNeighbors);
            _UnderlyingPtr = __MR_TriangulationParameters_ConstructFrom(numNeighbours, radius, critAngle, boundaryAngle, critHoleLength, automaticRadiusIncrease ? (byte)1 : (byte)0, searchNeighbors is not null ? searchNeighbors._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TriangulationParameters::TriangulationParameters`.
        public unsafe TriangulationParameters(MR.Const_TriangulationParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_ConstructFromAnother(MR.TriangulationParameters._Underlying *_other);
            _UnderlyingPtr = __MR_TriangulationParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TriangulationParameters::operator=`.
        public unsafe MR.TriangulationParameters Assign(MR.Const_TriangulationParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TriangulationParameters._Underlying *__MR_TriangulationParameters_AssignFromAnother(_Underlying *_this, MR.TriangulationParameters._Underlying *_other);
            return new(__MR_TriangulationParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TriangulationParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TriangulationParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriangulationParameters`/`Const_TriangulationParameters` directly.
    public class _InOptMut_TriangulationParameters
    {
        public TriangulationParameters? Opt;

        public _InOptMut_TriangulationParameters() {}
        public _InOptMut_TriangulationParameters(TriangulationParameters value) {Opt = value;}
        public static implicit operator _InOptMut_TriangulationParameters(TriangulationParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `TriangulationParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TriangulationParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriangulationParameters`/`Const_TriangulationParameters` to pass it to the function.
    public class _InOptConst_TriangulationParameters
    {
        public Const_TriangulationParameters? Opt;

        public _InOptConst_TriangulationParameters() {}
        public _InOptConst_TriangulationParameters(Const_TriangulationParameters value) {Opt = value;}
        public static implicit operator _InOptConst_TriangulationParameters(Const_TriangulationParameters value) {return new(value);}
    }

    /// Generated from class `MR::FillHolesWithExtraPointsParams`.
    /// This is the const half of the class.
    public class Const_FillHolesWithExtraPointsParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FillHolesWithExtraPointsParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_Destroy", ExactSpelling = true)]
            extern static void __MR_FillHolesWithExtraPointsParams_Destroy(_Underlying *_this);
            __MR_FillHolesWithExtraPointsParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FillHolesWithExtraPointsParams() {Dispose(false);}

        public unsafe MR.Const_TriangulationParameters Triangulation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_Get_triangulation", ExactSpelling = true)]
                extern static MR.Const_TriangulationParameters._Underlying *__MR_FillHolesWithExtraPointsParams_Get_triangulation(_Underlying *_this);
                return new(__MR_FillHolesWithExtraPointsParams_Get_triangulation(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if nullptr, then any boundary vertex of input mesh can get new triangles;
        /// otherwise only vertices from modifyBdVertices can get new triangles
        public unsafe ref readonly void * ModifyBdVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_Get_modifyBdVertices", ExactSpelling = true)]
                extern static void **__MR_FillHolesWithExtraPointsParams_Get_modifyBdVertices(_Underlying *_this);
                return ref *__MR_FillHolesWithExtraPointsParams_Get_modifyBdVertices(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FillHolesWithExtraPointsParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHolesWithExtraPointsParams_DefaultConstruct();
        }

        /// Constructs `MR::FillHolesWithExtraPointsParams` elementwise.
        public unsafe Const_FillHolesWithExtraPointsParams(MR.Const_TriangulationParameters triangulation, MR.Const_VertBitSet? modifyBdVertices) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_ConstructFrom(MR.TriangulationParameters._Underlying *triangulation, MR.Const_VertBitSet._Underlying *modifyBdVertices);
            _UnderlyingPtr = __MR_FillHolesWithExtraPointsParams_ConstructFrom(triangulation._UnderlyingPtr, modifyBdVertices is not null ? modifyBdVertices._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FillHolesWithExtraPointsParams::FillHolesWithExtraPointsParams`.
        public unsafe Const_FillHolesWithExtraPointsParams(MR.Const_FillHolesWithExtraPointsParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_ConstructFromAnother(MR.FillHolesWithExtraPointsParams._Underlying *_other);
            _UnderlyingPtr = __MR_FillHolesWithExtraPointsParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::FillHolesWithExtraPointsParams`.
    /// This is the non-const half of the class.
    public class FillHolesWithExtraPointsParams : Const_FillHolesWithExtraPointsParams
    {
        internal unsafe FillHolesWithExtraPointsParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.TriangulationParameters Triangulation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_GetMutable_triangulation", ExactSpelling = true)]
                extern static MR.TriangulationParameters._Underlying *__MR_FillHolesWithExtraPointsParams_GetMutable_triangulation(_Underlying *_this);
                return new(__MR_FillHolesWithExtraPointsParams_GetMutable_triangulation(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if nullptr, then any boundary vertex of input mesh can get new triangles;
        /// otherwise only vertices from modifyBdVertices can get new triangles
        public new unsafe ref readonly void * ModifyBdVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_GetMutable_modifyBdVertices", ExactSpelling = true)]
                extern static void **__MR_FillHolesWithExtraPointsParams_GetMutable_modifyBdVertices(_Underlying *_this);
                return ref *__MR_FillHolesWithExtraPointsParams_GetMutable_modifyBdVertices(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FillHolesWithExtraPointsParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHolesWithExtraPointsParams_DefaultConstruct();
        }

        /// Constructs `MR::FillHolesWithExtraPointsParams` elementwise.
        public unsafe FillHolesWithExtraPointsParams(MR.Const_TriangulationParameters triangulation, MR.Const_VertBitSet? modifyBdVertices) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_ConstructFrom(MR.TriangulationParameters._Underlying *triangulation, MR.Const_VertBitSet._Underlying *modifyBdVertices);
            _UnderlyingPtr = __MR_FillHolesWithExtraPointsParams_ConstructFrom(triangulation._UnderlyingPtr, modifyBdVertices is not null ? modifyBdVertices._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FillHolesWithExtraPointsParams::FillHolesWithExtraPointsParams`.
        public unsafe FillHolesWithExtraPointsParams(MR.Const_FillHolesWithExtraPointsParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_ConstructFromAnother(MR.FillHolesWithExtraPointsParams._Underlying *_other);
            _UnderlyingPtr = __MR_FillHolesWithExtraPointsParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::FillHolesWithExtraPointsParams::operator=`.
        public unsafe MR.FillHolesWithExtraPointsParams Assign(MR.Const_FillHolesWithExtraPointsParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHolesWithExtraPointsParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FillHolesWithExtraPointsParams._Underlying *__MR_FillHolesWithExtraPointsParams_AssignFromAnother(_Underlying *_this, MR.FillHolesWithExtraPointsParams._Underlying *_other);
            return new(__MR_FillHolesWithExtraPointsParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FillHolesWithExtraPointsParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FillHolesWithExtraPointsParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHolesWithExtraPointsParams`/`Const_FillHolesWithExtraPointsParams` directly.
    public class _InOptMut_FillHolesWithExtraPointsParams
    {
        public FillHolesWithExtraPointsParams? Opt;

        public _InOptMut_FillHolesWithExtraPointsParams() {}
        public _InOptMut_FillHolesWithExtraPointsParams(FillHolesWithExtraPointsParams value) {Opt = value;}
        public static implicit operator _InOptMut_FillHolesWithExtraPointsParams(FillHolesWithExtraPointsParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `FillHolesWithExtraPointsParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FillHolesWithExtraPointsParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHolesWithExtraPointsParams`/`Const_FillHolesWithExtraPointsParams` to pass it to the function.
    public class _InOptConst_FillHolesWithExtraPointsParams
    {
        public Const_FillHolesWithExtraPointsParams? Opt;

        public _InOptConst_FillHolesWithExtraPointsParams() {}
        public _InOptConst_FillHolesWithExtraPointsParams(Const_FillHolesWithExtraPointsParams value) {Opt = value;}
        public static implicit operator _InOptConst_FillHolesWithExtraPointsParams(Const_FillHolesWithExtraPointsParams value) {return new(value);}
    }

    /**
    * \brief Creates mesh from given point cloud according params
    * Returns empty optional if was interrupted by progress bar
    *
    <table border=0>
    <caption id="triangulatePointCloud_examples"></caption>
    <tr>
    <td> \image html triangulate/triangulate_0.png "Before" width = 350cm </td>
    <td> \image html triangulate/triangulate_3.png "After" width = 350cm </td>
    </tr>
    </table>
    */
    /// Generated from function `MR::triangulatePointCloud`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `progressCb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRMesh> TriangulatePointCloud(MR.Const_PointCloud pointCloud, MR.Const_TriangulationParameters? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? progressCb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_triangulatePointCloud", ExactSpelling = true)]
        extern static MR.Std.Optional_MRMesh._Underlying *__MR_triangulatePointCloud(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_TriangulationParameters._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCb);
        return MR.Misc.Move(new MR.Std.Optional_MRMesh(__MR_triangulatePointCloud(pointCloud._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, progressCb is not null ? progressCb._UnderlyingPtr : null), is_owning: true));
    }

    /// fills the holes in the mesh by adding triangles to it with the vertices in existing boundary vertices or given extra points (in any combination)
    /// \param extraPoints must have either properly oriented normals or no normals, and it will be temporary modified during the call
    /// \return false if the operation was canceled or incorrect input
    /// Generated from function `MR::fillHolesWithExtraPoints`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `progressCb` defaults to `{}`.
    public static unsafe bool FillHolesWithExtraPoints(MR.Mesh mesh, MR.PointCloud extraPoints, MR.Const_FillHolesWithExtraPointsParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? progressCb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillHolesWithExtraPoints", ExactSpelling = true)]
        extern static byte __MR_fillHolesWithExtraPoints(MR.Mesh._Underlying *mesh, MR.PointCloud._Underlying *extraPoints, MR.Const_FillHolesWithExtraPointsParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCb);
        return __MR_fillHolesWithExtraPoints(mesh._UnderlyingPtr, extraPoints._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, progressCb is not null ? progressCb._UnderlyingPtr : null) != 0;
    }
}
