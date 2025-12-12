public static partial class MR
{
    /// Map structure to find primitives of old topology by edges introduced in cutMesh
    /// Generated from class `MR::NewEdgesMap`.
    /// This is the const half of the class.
    public class Const_NewEdgesMap : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NewEdgesMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_Destroy", ExactSpelling = true)]
            extern static void __MR_NewEdgesMap_Destroy(_Underlying *_this);
            __MR_NewEdgesMap_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NewEdgesMap() {Dispose(false);}

        /// true here means that a subdivided edge is a part of some original edge edge before mesh subdivision;
        /// false here is both for unmodified edges and for new edges introduced within original triangles
        public unsafe MR.Const_UndirectedEdgeBitSet SplitEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_Get_splitEdges", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_NewEdgesMap_Get_splitEdges(_Underlying *_this);
                return new(__MR_NewEdgesMap_Get_splitEdges(_UnderlyingPtr), is_owning: false);
            }
        }

        /// maps every edge appeared during subdivision to an original edge before mesh subdivision;
        /// for splitEdges[key]=true, the value is arbitrary oriented original edge, for which key-edge is its part;
        /// for splitEdges[key]=false, the value is an original triangle
        public unsafe MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_Int Map
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_Get_map", ExactSpelling = true)]
                extern static MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_Int._Underlying *__MR_NewEdgesMap_Get_map(_Underlying *_this);
                return new(__MR_NewEdgesMap_Get_map(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NewEdgesMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_DefaultConstruct();
            _UnderlyingPtr = __MR_NewEdgesMap_DefaultConstruct();
        }

        /// Constructs `MR::NewEdgesMap` elementwise.
        public unsafe Const_NewEdgesMap(MR._ByValue_UndirectedEdgeBitSet splitEdges, MR.Phmap._ByValue_FlatHashMap_MRUndirectedEdgeId_Int map) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_ConstructFrom(MR.Misc._PassBy splitEdges_pass_by, MR.UndirectedEdgeBitSet._Underlying *splitEdges, MR.Misc._PassBy map_pass_by, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_Int._Underlying *map);
            _UnderlyingPtr = __MR_NewEdgesMap_ConstructFrom(splitEdges.PassByMode, splitEdges.Value is not null ? splitEdges.Value._UnderlyingPtr : null, map.PassByMode, map.Value is not null ? map.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::NewEdgesMap::NewEdgesMap`.
        public unsafe Const_NewEdgesMap(MR._ByValue_NewEdgesMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NewEdgesMap._Underlying *_other);
            _UnderlyingPtr = __MR_NewEdgesMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Map structure to find primitives of old topology by edges introduced in cutMesh
    /// Generated from class `MR::NewEdgesMap`.
    /// This is the non-const half of the class.
    public class NewEdgesMap : Const_NewEdgesMap
    {
        internal unsafe NewEdgesMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// true here means that a subdivided edge is a part of some original edge edge before mesh subdivision;
        /// false here is both for unmodified edges and for new edges introduced within original triangles
        public new unsafe MR.UndirectedEdgeBitSet SplitEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_GetMutable_splitEdges", ExactSpelling = true)]
                extern static MR.UndirectedEdgeBitSet._Underlying *__MR_NewEdgesMap_GetMutable_splitEdges(_Underlying *_this);
                return new(__MR_NewEdgesMap_GetMutable_splitEdges(_UnderlyingPtr), is_owning: false);
            }
        }

        /// maps every edge appeared during subdivision to an original edge before mesh subdivision;
        /// for splitEdges[key]=true, the value is arbitrary oriented original edge, for which key-edge is its part;
        /// for splitEdges[key]=false, the value is an original triangle
        public new unsafe MR.Phmap.FlatHashMap_MRUndirectedEdgeId_Int Map
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_GetMutable_map", ExactSpelling = true)]
                extern static MR.Phmap.FlatHashMap_MRUndirectedEdgeId_Int._Underlying *__MR_NewEdgesMap_GetMutable_map(_Underlying *_this);
                return new(__MR_NewEdgesMap_GetMutable_map(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NewEdgesMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_DefaultConstruct();
            _UnderlyingPtr = __MR_NewEdgesMap_DefaultConstruct();
        }

        /// Constructs `MR::NewEdgesMap` elementwise.
        public unsafe NewEdgesMap(MR._ByValue_UndirectedEdgeBitSet splitEdges, MR.Phmap._ByValue_FlatHashMap_MRUndirectedEdgeId_Int map) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_ConstructFrom(MR.Misc._PassBy splitEdges_pass_by, MR.UndirectedEdgeBitSet._Underlying *splitEdges, MR.Misc._PassBy map_pass_by, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_Int._Underlying *map);
            _UnderlyingPtr = __MR_NewEdgesMap_ConstructFrom(splitEdges.PassByMode, splitEdges.Value is not null ? splitEdges.Value._UnderlyingPtr : null, map.PassByMode, map.Value is not null ? map.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::NewEdgesMap::NewEdgesMap`.
        public unsafe NewEdgesMap(MR._ByValue_NewEdgesMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NewEdgesMap._Underlying *_other);
            _UnderlyingPtr = __MR_NewEdgesMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::NewEdgesMap::operator=`.
        public unsafe MR.NewEdgesMap Assign(MR._ByValue_NewEdgesMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NewEdgesMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NewEdgesMap._Underlying *__MR_NewEdgesMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.NewEdgesMap._Underlying *_other);
            return new(__MR_NewEdgesMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `NewEdgesMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `NewEdgesMap`/`Const_NewEdgesMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_NewEdgesMap
    {
        internal readonly Const_NewEdgesMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_NewEdgesMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_NewEdgesMap(Const_NewEdgesMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_NewEdgesMap(Const_NewEdgesMap arg) {return new(arg);}
        public _ByValue_NewEdgesMap(MR.Misc._Moved<NewEdgesMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_NewEdgesMap(MR.Misc._Moved<NewEdgesMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `NewEdgesMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NewEdgesMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NewEdgesMap`/`Const_NewEdgesMap` directly.
    public class _InOptMut_NewEdgesMap
    {
        public NewEdgesMap? Opt;

        public _InOptMut_NewEdgesMap() {}
        public _InOptMut_NewEdgesMap(NewEdgesMap value) {Opt = value;}
        public static implicit operator _InOptMut_NewEdgesMap(NewEdgesMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `NewEdgesMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NewEdgesMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NewEdgesMap`/`Const_NewEdgesMap` to pass it to the function.
    public class _InOptConst_NewEdgesMap
    {
        public Const_NewEdgesMap? Opt;

        public _InOptConst_NewEdgesMap() {}
        public _InOptConst_NewEdgesMap(Const_NewEdgesMap value) {Opt = value;}
        public static implicit operator _InOptConst_NewEdgesMap(Const_NewEdgesMap value) {return new(value);}
    }

    /** \struct MR::CutMeshParameters
    *
    * \brief Parameters of MR::cutMesh
    * 
    * This structure contains some options and optional outputs of MR::cutMesh function
    * \sa \ref MR::CutMeshResult
    */
    /// Generated from class `MR::CutMeshParameters`.
    /// This is the const half of the class.
    public class Const_CutMeshParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CutMeshParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_CutMeshParameters_Destroy(_Underlying *_this);
            __MR_CutMeshParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CutMeshParameters() {Dispose(false);}

        /// This is optional input for better contours resolving\n
        /// it provides additional info from other mesh used in boolean operation, useful to solve some degeneration
        /// \note Most likely you don't need this in case you call MR::cutMesh manualy, use case of it is MR::boolean
        public unsafe ref readonly void * SortData
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_Get_sortData", ExactSpelling = true)]
                extern static void **__MR_CutMeshParameters_Get_sortData(_Underlying *_this);
                return ref *__MR_CutMeshParameters_Get_sortData(_UnderlyingPtr);
            }
        }

        /// This is optional output - map from newly generated faces to old faces (N-1)
        public unsafe ref void * New2OldMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_Get_new2OldMap", ExactSpelling = true)]
                extern static void **__MR_CutMeshParameters_Get_new2OldMap(_Underlying *_this);
                return ref *__MR_CutMeshParameters_Get_new2OldMap(_UnderlyingPtr);
            }
        }

        public unsafe MR.CutMeshParameters.ForceFill ForceFillMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_Get_forceFillMode", ExactSpelling = true)]
                extern static MR.CutMeshParameters.ForceFill *__MR_CutMeshParameters_Get_forceFillMode(_Underlying *_this);
                return *__MR_CutMeshParameters_Get_forceFillMode(_UnderlyingPtr);
            }
        }

        /// Optional output map for each new edge introduced after cut maps edge from old topology or old face
        public unsafe ref void * New2oldEdgesMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_Get_new2oldEdgesMap", ExactSpelling = true)]
                extern static void **__MR_CutMeshParameters_Get_new2oldEdgesMap(_Underlying *_this);
                return ref *__MR_CutMeshParameters_Get_new2oldEdgesMap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CutMeshParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_CutMeshParameters_DefaultConstruct();
        }

        /// Constructs `MR::CutMeshParameters` elementwise.
        public unsafe Const_CutMeshParameters(MR.Const_SortIntersectionsData? sortData, MR.FaceMap? new2OldMap, MR.CutMeshParameters.ForceFill forceFillMode, MR.NewEdgesMap? new2oldEdgesMap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_ConstructFrom(MR.Const_SortIntersectionsData._Underlying *sortData, MR.FaceMap._Underlying *new2OldMap, MR.CutMeshParameters.ForceFill forceFillMode, MR.NewEdgesMap._Underlying *new2oldEdgesMap);
            _UnderlyingPtr = __MR_CutMeshParameters_ConstructFrom(sortData is not null ? sortData._UnderlyingPtr : null, new2OldMap is not null ? new2OldMap._UnderlyingPtr : null, forceFillMode, new2oldEdgesMap is not null ? new2oldEdgesMap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CutMeshParameters::CutMeshParameters`.
        public unsafe Const_CutMeshParameters(MR.Const_CutMeshParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_ConstructFromAnother(MR.CutMeshParameters._Underlying *_other);
            _UnderlyingPtr = __MR_CutMeshParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// This enum defines the MR::cutMesh behaviour in case of bad faces acure
        /// basicaly MR::cutMesh removes all faces which contours pass through, adds new edges to topology and fills all removed parts
        /// 
        /// \note Bad faces here mean faces where contours have intersections and cannot be cut and filled in an good way
        public enum ForceFill : int
        {
            //< if bad faces occur does not fill anything
            None = 0,
            //< fills all faces except bad ones
            Good = 1,
            //< fills all faces with bad ones, but on bad faces triangulation can also be bad (may have self-intersections or tunnels)
            All = 2,
        }
    }

    /** \struct MR::CutMeshParameters
    *
    * \brief Parameters of MR::cutMesh
    * 
    * This structure contains some options and optional outputs of MR::cutMesh function
    * \sa \ref MR::CutMeshResult
    */
    /// Generated from class `MR::CutMeshParameters`.
    /// This is the non-const half of the class.
    public class CutMeshParameters : Const_CutMeshParameters
    {
        internal unsafe CutMeshParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// This is optional input for better contours resolving\n
        /// it provides additional info from other mesh used in boolean operation, useful to solve some degeneration
        /// \note Most likely you don't need this in case you call MR::cutMesh manualy, use case of it is MR::boolean
        public new unsafe ref readonly void * SortData
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_GetMutable_sortData", ExactSpelling = true)]
                extern static void **__MR_CutMeshParameters_GetMutable_sortData(_Underlying *_this);
                return ref *__MR_CutMeshParameters_GetMutable_sortData(_UnderlyingPtr);
            }
        }

        /// This is optional output - map from newly generated faces to old faces (N-1)
        public new unsafe ref void * New2OldMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_GetMutable_new2OldMap", ExactSpelling = true)]
                extern static void **__MR_CutMeshParameters_GetMutable_new2OldMap(_Underlying *_this);
                return ref *__MR_CutMeshParameters_GetMutable_new2OldMap(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.CutMeshParameters.ForceFill ForceFillMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_GetMutable_forceFillMode", ExactSpelling = true)]
                extern static MR.CutMeshParameters.ForceFill *__MR_CutMeshParameters_GetMutable_forceFillMode(_Underlying *_this);
                return ref *__MR_CutMeshParameters_GetMutable_forceFillMode(_UnderlyingPtr);
            }
        }

        /// Optional output map for each new edge introduced after cut maps edge from old topology or old face
        public new unsafe ref void * New2oldEdgesMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_GetMutable_new2oldEdgesMap", ExactSpelling = true)]
                extern static void **__MR_CutMeshParameters_GetMutable_new2oldEdgesMap(_Underlying *_this);
                return ref *__MR_CutMeshParameters_GetMutable_new2oldEdgesMap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CutMeshParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_CutMeshParameters_DefaultConstruct();
        }

        /// Constructs `MR::CutMeshParameters` elementwise.
        public unsafe CutMeshParameters(MR.Const_SortIntersectionsData? sortData, MR.FaceMap? new2OldMap, MR.CutMeshParameters.ForceFill forceFillMode, MR.NewEdgesMap? new2oldEdgesMap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_ConstructFrom(MR.Const_SortIntersectionsData._Underlying *sortData, MR.FaceMap._Underlying *new2OldMap, MR.CutMeshParameters.ForceFill forceFillMode, MR.NewEdgesMap._Underlying *new2oldEdgesMap);
            _UnderlyingPtr = __MR_CutMeshParameters_ConstructFrom(sortData is not null ? sortData._UnderlyingPtr : null, new2OldMap is not null ? new2OldMap._UnderlyingPtr : null, forceFillMode, new2oldEdgesMap is not null ? new2oldEdgesMap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CutMeshParameters::CutMeshParameters`.
        public unsafe CutMeshParameters(MR.Const_CutMeshParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_ConstructFromAnother(MR.CutMeshParameters._Underlying *_other);
            _UnderlyingPtr = __MR_CutMeshParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CutMeshParameters::operator=`.
        public unsafe MR.CutMeshParameters Assign(MR.Const_CutMeshParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CutMeshParameters._Underlying *__MR_CutMeshParameters_AssignFromAnother(_Underlying *_this, MR.CutMeshParameters._Underlying *_other);
            return new(__MR_CutMeshParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CutMeshParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CutMeshParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CutMeshParameters`/`Const_CutMeshParameters` directly.
    public class _InOptMut_CutMeshParameters
    {
        public CutMeshParameters? Opt;

        public _InOptMut_CutMeshParameters() {}
        public _InOptMut_CutMeshParameters(CutMeshParameters value) {Opt = value;}
        public static implicit operator _InOptMut_CutMeshParameters(CutMeshParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `CutMeshParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CutMeshParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CutMeshParameters`/`Const_CutMeshParameters` to pass it to the function.
    public class _InOptConst_CutMeshParameters
    {
        public Const_CutMeshParameters? Opt;

        public _InOptConst_CutMeshParameters() {}
        public _InOptConst_CutMeshParameters(Const_CutMeshParameters value) {Opt = value;}
        public static implicit operator _InOptConst_CutMeshParameters(Const_CutMeshParameters value) {return new(value);}
    }

    /** \struct MR::CutMeshResult
    *
    * This structure contains result of MR::cutMesh function
    */
    /// Generated from class `MR::CutMeshResult`.
    /// This is the const half of the class.
    public class Const_CutMeshResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CutMeshResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_Destroy", ExactSpelling = true)]
            extern static void __MR_CutMeshResult_Destroy(_Underlying *_this);
            __MR_CutMeshResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CutMeshResult() {Dispose(false);}

        /// Paths of new edges on mesh, they represent same contours as input, but already cut
        public unsafe MR.Std.Const_Vector_StdVectorMREdgeId ResultCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_Get_resultCut", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *__MR_CutMeshResult_Get_resultCut(_Underlying *_this);
                return new(__MR_CutMeshResult_Get_resultCut(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Bitset of bad triangles - triangles where input contours have intersections and cannot be cut and filled in a good way
        /// \sa \ref MR::CutMeshParameters
        public unsafe MR.Const_FaceBitSet FbsWithContourIntersections
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_Get_fbsWithContourIntersections", ExactSpelling = true)]
                extern static MR.Const_FaceBitSet._Underlying *__MR_CutMeshResult_Get_fbsWithContourIntersections(_Underlying *_this);
                return new(__MR_CutMeshResult_Get_fbsWithContourIntersections(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CutMeshResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_DefaultConstruct();
            _UnderlyingPtr = __MR_CutMeshResult_DefaultConstruct();
        }

        /// Constructs `MR::CutMeshResult` elementwise.
        public unsafe Const_CutMeshResult(MR.Std._ByValue_Vector_StdVectorMREdgeId resultCut, MR._ByValue_FaceBitSet fbsWithContourIntersections) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_ConstructFrom(MR.Misc._PassBy resultCut_pass_by, MR.Std.Vector_StdVectorMREdgeId._Underlying *resultCut, MR.Misc._PassBy fbsWithContourIntersections_pass_by, MR.FaceBitSet._Underlying *fbsWithContourIntersections);
            _UnderlyingPtr = __MR_CutMeshResult_ConstructFrom(resultCut.PassByMode, resultCut.Value is not null ? resultCut.Value._UnderlyingPtr : null, fbsWithContourIntersections.PassByMode, fbsWithContourIntersections.Value is not null ? fbsWithContourIntersections.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CutMeshResult::CutMeshResult`.
        public unsafe Const_CutMeshResult(MR._ByValue_CutMeshResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CutMeshResult._Underlying *_other);
            _UnderlyingPtr = __MR_CutMeshResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /** \struct MR::CutMeshResult
    *
    * This structure contains result of MR::cutMesh function
    */
    /// Generated from class `MR::CutMeshResult`.
    /// This is the non-const half of the class.
    public class CutMeshResult : Const_CutMeshResult
    {
        internal unsafe CutMeshResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Paths of new edges on mesh, they represent same contours as input, but already cut
        public new unsafe MR.Std.Vector_StdVectorMREdgeId ResultCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_GetMutable_resultCut", ExactSpelling = true)]
                extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_CutMeshResult_GetMutable_resultCut(_Underlying *_this);
                return new(__MR_CutMeshResult_GetMutable_resultCut(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Bitset of bad triangles - triangles where input contours have intersections and cannot be cut and filled in a good way
        /// \sa \ref MR::CutMeshParameters
        public new unsafe MR.FaceBitSet FbsWithContourIntersections
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_GetMutable_fbsWithContourIntersections", ExactSpelling = true)]
                extern static MR.FaceBitSet._Underlying *__MR_CutMeshResult_GetMutable_fbsWithContourIntersections(_Underlying *_this);
                return new(__MR_CutMeshResult_GetMutable_fbsWithContourIntersections(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CutMeshResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_DefaultConstruct();
            _UnderlyingPtr = __MR_CutMeshResult_DefaultConstruct();
        }

        /// Constructs `MR::CutMeshResult` elementwise.
        public unsafe CutMeshResult(MR.Std._ByValue_Vector_StdVectorMREdgeId resultCut, MR._ByValue_FaceBitSet fbsWithContourIntersections) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_ConstructFrom(MR.Misc._PassBy resultCut_pass_by, MR.Std.Vector_StdVectorMREdgeId._Underlying *resultCut, MR.Misc._PassBy fbsWithContourIntersections_pass_by, MR.FaceBitSet._Underlying *fbsWithContourIntersections);
            _UnderlyingPtr = __MR_CutMeshResult_ConstructFrom(resultCut.PassByMode, resultCut.Value is not null ? resultCut.Value._UnderlyingPtr : null, fbsWithContourIntersections.PassByMode, fbsWithContourIntersections.Value is not null ? fbsWithContourIntersections.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CutMeshResult::CutMeshResult`.
        public unsafe CutMeshResult(MR._ByValue_CutMeshResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CutMeshResult._Underlying *_other);
            _UnderlyingPtr = __MR_CutMeshResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CutMeshResult::operator=`.
        public unsafe MR.CutMeshResult Assign(MR._ByValue_CutMeshResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutMeshResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CutMeshResult._Underlying *__MR_CutMeshResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CutMeshResult._Underlying *_other);
            return new(__MR_CutMeshResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `CutMeshResult` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CutMeshResult`/`Const_CutMeshResult` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CutMeshResult
    {
        internal readonly Const_CutMeshResult? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CutMeshResult() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_CutMeshResult(Const_CutMeshResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CutMeshResult(Const_CutMeshResult arg) {return new(arg);}
        public _ByValue_CutMeshResult(MR.Misc._Moved<CutMeshResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CutMeshResult(MR.Misc._Moved<CutMeshResult> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CutMeshResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CutMeshResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CutMeshResult`/`Const_CutMeshResult` directly.
    public class _InOptMut_CutMeshResult
    {
        public CutMeshResult? Opt;

        public _InOptMut_CutMeshResult() {}
        public _InOptMut_CutMeshResult(CutMeshResult value) {Opt = value;}
        public static implicit operator _InOptMut_CutMeshResult(CutMeshResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `CutMeshResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CutMeshResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CutMeshResult`/`Const_CutMeshResult` to pass it to the function.
    public class _InOptConst_CutMeshResult
    {
        public Const_CutMeshResult? Opt;

        public _InOptConst_CutMeshResult() {}
        public _InOptConst_CutMeshResult(Const_CutMeshResult value) {Opt = value;}
        public static implicit operator _InOptConst_CutMeshResult(Const_CutMeshResult value) {return new(value);}
    }

    /// Settings structurer for cutMeshByProjection function
    /// Generated from class `MR::CutByProjectionSettings`.
    /// This is the const half of the class.
    public class Const_CutByProjectionSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CutByProjectionSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_CutByProjectionSettings_Destroy(_Underlying *_this);
            __MR_CutByProjectionSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CutByProjectionSettings() {Dispose(false);}

        /// direction of projection (in mesh space)
        public unsafe MR.Const_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_Get_direction", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_CutByProjectionSettings_Get_direction(_Underlying *_this);
                return new(__MR_CutByProjectionSettings_Get_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if set - used to transform contours form its local space to mesh local space
        public unsafe ref readonly MR.AffineXf3f * Cont2mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_Get_cont2mesh", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_CutByProjectionSettings_Get_cont2mesh(_Underlying *_this);
                return ref *__MR_CutByProjectionSettings_Get_cont2mesh(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CutByProjectionSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_CutByProjectionSettings_DefaultConstruct();
        }

        /// Constructs `MR::CutByProjectionSettings` elementwise.
        public unsafe Const_CutByProjectionSettings(MR.Vector3f direction, MR.Const_AffineXf3f? cont2mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_ConstructFrom(MR.Vector3f direction, MR.Const_AffineXf3f._Underlying *cont2mesh);
            _UnderlyingPtr = __MR_CutByProjectionSettings_ConstructFrom(direction, cont2mesh is not null ? cont2mesh._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CutByProjectionSettings::CutByProjectionSettings`.
        public unsafe Const_CutByProjectionSettings(MR.Const_CutByProjectionSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_ConstructFromAnother(MR.CutByProjectionSettings._Underlying *_other);
            _UnderlyingPtr = __MR_CutByProjectionSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Settings structurer for cutMeshByProjection function
    /// Generated from class `MR::CutByProjectionSettings`.
    /// This is the non-const half of the class.
    public class CutByProjectionSettings : Const_CutByProjectionSettings
    {
        internal unsafe CutByProjectionSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// direction of projection (in mesh space)
        public new unsafe MR.Mut_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_GetMutable_direction", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_CutByProjectionSettings_GetMutable_direction(_Underlying *_this);
                return new(__MR_CutByProjectionSettings_GetMutable_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if set - used to transform contours form its local space to mesh local space
        public new unsafe ref readonly MR.AffineXf3f * Cont2mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_GetMutable_cont2mesh", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_CutByProjectionSettings_GetMutable_cont2mesh(_Underlying *_this);
                return ref *__MR_CutByProjectionSettings_GetMutable_cont2mesh(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe CutByProjectionSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_CutByProjectionSettings_DefaultConstruct();
        }

        /// Constructs `MR::CutByProjectionSettings` elementwise.
        public unsafe CutByProjectionSettings(MR.Vector3f direction, MR.Const_AffineXf3f? cont2mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_ConstructFrom(MR.Vector3f direction, MR.Const_AffineXf3f._Underlying *cont2mesh);
            _UnderlyingPtr = __MR_CutByProjectionSettings_ConstructFrom(direction, cont2mesh is not null ? cont2mesh._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::CutByProjectionSettings::CutByProjectionSettings`.
        public unsafe CutByProjectionSettings(MR.Const_CutByProjectionSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_ConstructFromAnother(MR.CutByProjectionSettings._Underlying *_other);
            _UnderlyingPtr = __MR_CutByProjectionSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::CutByProjectionSettings::operator=`.
        public unsafe MR.CutByProjectionSettings Assign(MR.Const_CutByProjectionSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CutByProjectionSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CutByProjectionSettings._Underlying *__MR_CutByProjectionSettings_AssignFromAnother(_Underlying *_this, MR.CutByProjectionSettings._Underlying *_other);
            return new(__MR_CutByProjectionSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `CutByProjectionSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CutByProjectionSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CutByProjectionSettings`/`Const_CutByProjectionSettings` directly.
    public class _InOptMut_CutByProjectionSettings
    {
        public CutByProjectionSettings? Opt;

        public _InOptMut_CutByProjectionSettings() {}
        public _InOptMut_CutByProjectionSettings(CutByProjectionSettings value) {Opt = value;}
        public static implicit operator _InOptMut_CutByProjectionSettings(CutByProjectionSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `CutByProjectionSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CutByProjectionSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CutByProjectionSettings`/`Const_CutByProjectionSettings` to pass it to the function.
    public class _InOptConst_CutByProjectionSettings
    {
        public Const_CutByProjectionSettings? Opt;

        public _InOptConst_CutByProjectionSettings() {}
        public _InOptConst_CutByProjectionSettings(Const_CutByProjectionSettings value) {Opt = value;}
        public static implicit operator _InOptConst_CutByProjectionSettings(Const_CutByProjectionSettings value) {return new(value);}
    }

    /**
    * \brief Cuts mesh by given contours
    * 
    * This function cuts mesh making new edges paths on place of input contours
    * \param mesh Input mesh that will be cut
    * \param contours Input contours to cut mesh with, find more \ref MR::OneMeshContours
    * \param params Parameters describing some cut options, find more \ref MR::CutMeshParameters
    * \return New edges that correspond to given contours, find more \ref MR::CutMeshResult
    * \parblock
    * \warning Input contours should have no intersections, faces where contours intersects (`bad faces`) will not be allowed for fill
    * \endparblock
    * \parblock
    * \warning Input mesh will be changed in any case, if `bad faces` are in mesh, mesh will be spoiled, \n
    * so if you cannot guarantee contours without intersections better make copy of mesh, before using this function
    * \endparblock
    */
    /// Generated from function `MR::cutMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.CutMeshResult> CutMesh(MR.Mesh mesh, MR.Std.Const_Vector_MROneMeshContour contours, MR.Const_CutMeshParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cutMesh", ExactSpelling = true)]
        extern static MR.CutMeshResult._Underlying *__MR_cutMesh(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_MROneMeshContour._Underlying *contours, MR.Const_CutMeshParameters._Underlying *params_);
        return MR.Misc.Move(new MR.CutMeshResult(__MR_cutMesh(mesh._UnderlyingPtr, contours._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Cuts \p mesh by \p contour by projecting all the points
    /// \param xf transformation from the CSYS of \p contour to the CSYS of \p mesh
    /// \note \p mesh is modified, see \ref cutMesh for info
    /// \note it might be useful to subdivide mesh before cut, to avoid issues related to lone contours
    /// \return Faces to the left of the polyline
    /// Generated from function `MR::cutMeshByContour`.
    /// Parameter `xf` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> CutMeshByContour(MR.Mesh mesh, MR.Std.Const_Vector_MRVector3f contour, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cutMeshByContour", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_cutMeshByContour(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_MRVector3f._Underlying *contour, MR.Const_AffineXf3f._Underlying *xf);
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_cutMeshByContour(mesh._UnderlyingPtr, contour._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null), is_owning: true));
    }

    /// Cuts \p mesh by \p contours by projecting all the points
    /// \param xf transformation from the CSYS of \p contour to the CSYS of \p mesh
    /// \note \p mesh is modified, see \ref cutMesh for info
    /// \note it might be useful to subdivide mesh before cut, to avoid issues related to lone contours
    /// \return Faces to the left of the polyline
    /// Generated from function `MR::cutMeshByContours`.
    /// Parameter `xf` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> CutMeshByContours(MR.Mesh mesh, MR.Std.Const_Vector_StdVectorMRVector3f contours, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cutMeshByContours", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_cutMeshByContours(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours, MR.Const_AffineXf3f._Underlying *xf);
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_cutMeshByContours(mesh._UnderlyingPtr, contours._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null), is_owning: true));
    }

    /// <summary>
    /// Performs orthographic projection with of given contours to mesh and cut result lines, 
    /// fails if any point of contours has missed mesh on projection stage or cut contours contains self-intersections
    /// \note it might be useful to subdivide mesh before cut, to avoid issues related to lone contours
    /// </summary>
    /// <param name="mesh"> for cutting, it will be changed</param>
    /// <param name="contours"> for projection onto mesh</param>
    /// <param name="settings"> to specify direction and \p contours to \p mesh space transformation</param>
    /// <returns>newly appeared edges on the mesh after cut or error</returns>
    /// Generated from function `MR::cutMeshByProjection`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdVectorMREdgeId_StdString> CutMeshByProjection(MR.Mesh mesh, MR.Std.Const_Vector_StdVectorMRVector3f contours, MR.Const_CutByProjectionSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cutMeshByProjection", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdVectorMREdgeId_StdString._Underlying *__MR_cutMeshByProjection(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours, MR.Const_CutByProjectionSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_StdVectorStdVectorMREdgeId_StdString(__MR_cutMeshByProjection(mesh._UnderlyingPtr, contours._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }

    /**
    * \brief Makes continuous contour by iso-line from mesh tri points, if first and last meshTriPoint is the same, makes closed contour
    *
    * Finds shortest paths between neighbor \p surfaceLine and build offset contour on surface for MR::cutMesh input
    * \param offset amount of offset form given point, note that absolute value is used and isoline in both direction returned
    * \param searchSettings settings for search geodesic path
    */
    /// Generated from function `MR::convertMeshTriPointsSurfaceOffsetToMeshContours`.
    /// Parameter `searchSettings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMROneMeshContour_StdString> ConvertMeshTriPointsSurfaceOffsetToMeshContours(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint surfaceLine, float offset, MR.Const_SearchPathSettings? searchSettings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertMeshTriPointsSurfaceOffsetToMeshContours_float", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMROneMeshContour_StdString._Underlying *__MR_convertMeshTriPointsSurfaceOffsetToMeshContours_float(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *surfaceLine, float offset, MR.SearchPathSettings._Underlying *searchSettings);
        return MR.Misc.Move(new MR.Expected_StdVectorMROneMeshContour_StdString(__MR_convertMeshTriPointsSurfaceOffsetToMeshContours_float(mesh._UnderlyingPtr, surfaceLine._UnderlyingPtr, offset, searchSettings is not null ? searchSettings._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Makes continuous contour by iso-line from mesh tri points, if first and last meshTriPoint is the same, makes closed contour
    *
    * Finds shortest paths between neighbor \p surfaceLine and build offset contour on surface for MR::cutMesh input
    * \param offsetAtPoint functor that returns amount of offset form arg point, note that absolute value is used and isoline in both direction returned
    * \param searchSettings settings for search geodesic path
    */
    /// Generated from function `MR::convertMeshTriPointsSurfaceOffsetToMeshContours`.
    /// Parameter `searchSettings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMROneMeshContour_StdString> ConvertMeshTriPointsSurfaceOffsetToMeshContours(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint surfaceLine, MR.Std.Const_Function_FloatFuncFromInt offsetAtPoint, MR.Const_SearchPathSettings? searchSettings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertMeshTriPointsSurfaceOffsetToMeshContours_std_function_float_func_from_int", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMROneMeshContour_StdString._Underlying *__MR_convertMeshTriPointsSurfaceOffsetToMeshContours_std_function_float_func_from_int(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *surfaceLine, MR.Std.Const_Function_FloatFuncFromInt._Underlying *offsetAtPoint, MR.SearchPathSettings._Underlying *searchSettings);
        return MR.Misc.Move(new MR.Expected_StdVectorMROneMeshContour_StdString(__MR_convertMeshTriPointsSurfaceOffsetToMeshContours_std_function_float_func_from_int(mesh._UnderlyingPtr, surfaceLine._UnderlyingPtr, offsetAtPoint._UnderlyingPtr, searchSettings is not null ? searchSettings._UnderlyingPtr : null), is_owning: true));
    }
}
