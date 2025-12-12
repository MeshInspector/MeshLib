public static partial class MR
{
    /**
    * Enum class of available CSG operations
    * \image html boolean/no_bool.png "Two separate meshes" width = 300cm
    * \sa \ref MR::boolean
    */
    public enum BooleanOperation : int
    {
        /// Part of mesh `A` that is inside of mesh `B`
        /// \image html boolean/inside_a.png "Inside A" width = 300cm
        InsideA = 0,
        /// Part of mesh `B` that is inside of mesh `A`
        /// \image html boolean/inside_b.png "Inside B" width = 300cm
        InsideB = 1,
        /// Part of mesh `A` that is outside of mesh `B`
        /// \image html boolean/outside_a.png "Outside A" width = 300cm
        OutsideA = 2,
        /// Part of mesh `B` that is outside of mesh `A`
        /// \image html boolean/outside_b.png "Outside B" width = 300cm
        OutsideB = 3,
        /// Union surface of two meshes (outside parts)
        /// \image html boolean/union.png "Union" width = 300cm
        Union = 4,
        /// Intersection surface of two meshes (inside parts)
        /// \image html boolean/intersection.png "Intersection" width = 300cm
        Intersection = 5,
        /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
        /// \image html boolean/b-a.png "Difference B-A" width = 300cm
        DifferenceBA = 6,
        /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
        /// \image html boolean/a-b.png "Difference A-B" width = 300cm
        DifferenceAB = 7,
        ///< not a valid operation
        Count = 8,
    }

    /** \struct MR::BooleanResultMapper
    * \brief Structure to map old mesh BitSets to new
    * \details Structure to easily map topology of MR::boolean input meshes to result mesh
    *
    * This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MR::boolean to result mesh topology primitives
    * \sa \ref MR::boolean
    */
    /// Generated from class `MR::BooleanResultMapper`.
    /// This is the const half of the class.
    public class Const_BooleanResultMapper : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BooleanResultMapper(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Destroy", ExactSpelling = true)]
            extern static void __MR_BooleanResultMapper_Destroy(_Underlying *_this);
            __MR_BooleanResultMapper_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BooleanResultMapper() {Dispose(false);}

        public unsafe MR.Std.Const_Array_MRBooleanResultMapperMaps_2 Maps_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Get_maps", ExactSpelling = true)]
                extern static MR.Std.Const_Array_MRBooleanResultMapperMaps_2._Underlying *__MR_BooleanResultMapper_Get_maps(_Underlying *_this);
                return new(__MR_BooleanResultMapper_Get_maps(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BooleanResultMapper() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanResultMapper._Underlying *__MR_BooleanResultMapper_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanResultMapper_DefaultConstruct();
        }

        /// Generated from constructor `MR::BooleanResultMapper::BooleanResultMapper`.
        public unsafe Const_BooleanResultMapper(MR._ByValue_BooleanResultMapper _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResultMapper._Underlying *__MR_BooleanResultMapper_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResultMapper._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanResultMapper_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Returns faces bitset of result mesh corresponding input one
        /// Generated from method `MR::BooleanResultMapper::map`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> Map(MR.Const_FaceBitSet oldBS, MR.BooleanResultMapper.MapObject obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_map_MR_FaceBitSet", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_BooleanResultMapper_map_MR_FaceBitSet(_Underlying *_this, MR.Const_FaceBitSet._Underlying *oldBS, MR.BooleanResultMapper.MapObject obj);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_BooleanResultMapper_map_MR_FaceBitSet(_UnderlyingPtr, oldBS._UnderlyingPtr, obj), is_owning: true));
        }

        /// Returns vertices bitset of result mesh corresponding input one
        /// Generated from method `MR::BooleanResultMapper::map`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> Map(MR.Const_VertBitSet oldBS, MR.BooleanResultMapper.MapObject obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_map_MR_VertBitSet", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_BooleanResultMapper_map_MR_VertBitSet(_Underlying *_this, MR.Const_VertBitSet._Underlying *oldBS, MR.BooleanResultMapper.MapObject obj);
            return MR.Misc.Move(new MR.VertBitSet(__MR_BooleanResultMapper_map_MR_VertBitSet(_UnderlyingPtr, oldBS._UnderlyingPtr, obj), is_owning: true));
        }

        /// Returns edges bitset of result mesh corresponding input one
        /// Generated from method `MR::BooleanResultMapper::map`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> Map(MR.Const_EdgeBitSet oldBS, MR.BooleanResultMapper.MapObject obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_map_MR_EdgeBitSet", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_BooleanResultMapper_map_MR_EdgeBitSet(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *oldBS, MR.BooleanResultMapper.MapObject obj);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_BooleanResultMapper_map_MR_EdgeBitSet(_UnderlyingPtr, oldBS._UnderlyingPtr, obj), is_owning: true));
        }

        /// Returns undirected edges bitset of result mesh corresponding input one
        /// Generated from method `MR::BooleanResultMapper::map`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> Map(MR.Const_UndirectedEdgeBitSet oldBS, MR.BooleanResultMapper.MapObject obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_map_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_BooleanResultMapper_map_MR_UndirectedEdgeBitSet(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *oldBS, MR.BooleanResultMapper.MapObject obj);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_BooleanResultMapper_map_MR_UndirectedEdgeBitSet(_UnderlyingPtr, oldBS._UnderlyingPtr, obj), is_owning: true));
        }

        /// Returns only new faces that are created during boolean operation
        /// Generated from method `MR::BooleanResultMapper::newFaces`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> NewFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_newFaces", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_BooleanResultMapper_newFaces(_Underlying *_this);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_BooleanResultMapper_newFaces(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::BooleanResultMapper::getMaps`.
        public unsafe MR.BooleanResultMapper.Const_Maps GetMaps(MR.BooleanResultMapper.MapObject index)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_getMaps", ExactSpelling = true)]
            extern static MR.BooleanResultMapper.Const_Maps._Underlying *__MR_BooleanResultMapper_getMaps(_Underlying *_this, MR.BooleanResultMapper.MapObject index);
            return new(__MR_BooleanResultMapper_getMaps(_UnderlyingPtr, index), is_owning: false);
        }

        /// Input object index enum
        public enum MapObject : int
        {
            A = 0,
            B = 1,
            Count = 2,
        }

        /// Generated from class `MR::BooleanResultMapper::Maps`.
        /// This is the const half of the class.
        public class Const_Maps : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Maps(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_Destroy", ExactSpelling = true)]
                extern static void __MR_BooleanResultMapper_Maps_Destroy(_Underlying *_this);
                __MR_BooleanResultMapper_Maps_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Maps() {Dispose(false);}

            /// "after cut" faces to "origin" faces
            /// this map is not 1-1, but N-1
            public unsafe MR.Const_FaceMap Cut2origin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_Get_cut2origin", ExactSpelling = true)]
                    extern static MR.Const_FaceMap._Underlying *__MR_BooleanResultMapper_Maps_Get_cut2origin(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_Get_cut2origin(_UnderlyingPtr), is_owning: false);
                }
            }

            /// "after cut" faces to "after stitch" faces (1-1)
            public unsafe MR.Const_FaceMap Cut2newFaces
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_Get_cut2newFaces", ExactSpelling = true)]
                    extern static MR.Const_FaceMap._Underlying *__MR_BooleanResultMapper_Maps_Get_cut2newFaces(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_Get_cut2newFaces(_UnderlyingPtr), is_owning: false);
                }
            }

            /// "origin" edges to "after stitch" edges (1-1)
            public unsafe MR.Const_WholeEdgeMap Old2newEdges
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_Get_old2newEdges", ExactSpelling = true)]
                    extern static MR.Const_WholeEdgeMap._Underlying *__MR_BooleanResultMapper_Maps_Get_old2newEdges(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_Get_old2newEdges(_UnderlyingPtr), is_owning: false);
                }
            }

            /// "origin" vertices to "after stitch" vertices (1-1)
            public unsafe MR.Const_VertMap Old2newVerts
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_Get_old2newVerts", ExactSpelling = true)]
                    extern static MR.Const_VertMap._Underlying *__MR_BooleanResultMapper_Maps_Get_old2newVerts(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_Get_old2newVerts(_UnderlyingPtr), is_owning: false);
                }
            }

            /// old topology indexes are valid if true
            public unsafe bool Identity
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_Get_identity", ExactSpelling = true)]
                    extern static bool *__MR_BooleanResultMapper_Maps_Get_identity(_Underlying *_this);
                    return *__MR_BooleanResultMapper_Maps_Get_identity(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Maps() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_DefaultConstruct", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_DefaultConstruct();
                _UnderlyingPtr = __MR_BooleanResultMapper_Maps_DefaultConstruct();
            }

            /// Constructs `MR::BooleanResultMapper::Maps` elementwise.
            public unsafe Const_Maps(MR._ByValue_FaceMap cut2origin, MR._ByValue_FaceMap cut2newFaces, MR._ByValue_WholeEdgeMap old2newEdges, MR._ByValue_VertMap old2newVerts, bool identity) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_ConstructFrom", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_ConstructFrom(MR.Misc._PassBy cut2origin_pass_by, MR.FaceMap._Underlying *cut2origin, MR.Misc._PassBy cut2newFaces_pass_by, MR.FaceMap._Underlying *cut2newFaces, MR.Misc._PassBy old2newEdges_pass_by, MR.WholeEdgeMap._Underlying *old2newEdges, MR.Misc._PassBy old2newVerts_pass_by, MR.VertMap._Underlying *old2newVerts, byte identity);
                _UnderlyingPtr = __MR_BooleanResultMapper_Maps_ConstructFrom(cut2origin.PassByMode, cut2origin.Value is not null ? cut2origin.Value._UnderlyingPtr : null, cut2newFaces.PassByMode, cut2newFaces.Value is not null ? cut2newFaces.Value._UnderlyingPtr : null, old2newEdges.PassByMode, old2newEdges.Value is not null ? old2newEdges.Value._UnderlyingPtr : null, old2newVerts.PassByMode, old2newVerts.Value is not null ? old2newVerts.Value._UnderlyingPtr : null, identity ? (byte)1 : (byte)0);
            }

            /// Generated from constructor `MR::BooleanResultMapper::Maps::Maps`.
            public unsafe Const_Maps(MR.BooleanResultMapper._ByValue_Maps _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResultMapper.Maps._Underlying *_other);
                _UnderlyingPtr = __MR_BooleanResultMapper_Maps_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::BooleanResultMapper::Maps`.
        /// This is the non-const half of the class.
        public class Maps : Const_Maps
        {
            internal unsafe Maps(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// "after cut" faces to "origin" faces
            /// this map is not 1-1, but N-1
            public new unsafe MR.FaceMap Cut2origin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_GetMutable_cut2origin", ExactSpelling = true)]
                    extern static MR.FaceMap._Underlying *__MR_BooleanResultMapper_Maps_GetMutable_cut2origin(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_GetMutable_cut2origin(_UnderlyingPtr), is_owning: false);
                }
            }

            /// "after cut" faces to "after stitch" faces (1-1)
            public new unsafe MR.FaceMap Cut2newFaces
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_GetMutable_cut2newFaces", ExactSpelling = true)]
                    extern static MR.FaceMap._Underlying *__MR_BooleanResultMapper_Maps_GetMutable_cut2newFaces(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_GetMutable_cut2newFaces(_UnderlyingPtr), is_owning: false);
                }
            }

            /// "origin" edges to "after stitch" edges (1-1)
            public new unsafe MR.WholeEdgeMap Old2newEdges
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_GetMutable_old2newEdges", ExactSpelling = true)]
                    extern static MR.WholeEdgeMap._Underlying *__MR_BooleanResultMapper_Maps_GetMutable_old2newEdges(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_GetMutable_old2newEdges(_UnderlyingPtr), is_owning: false);
                }
            }

            /// "origin" vertices to "after stitch" vertices (1-1)
            public new unsafe MR.VertMap Old2newVerts
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_GetMutable_old2newVerts", ExactSpelling = true)]
                    extern static MR.VertMap._Underlying *__MR_BooleanResultMapper_Maps_GetMutable_old2newVerts(_Underlying *_this);
                    return new(__MR_BooleanResultMapper_Maps_GetMutable_old2newVerts(_UnderlyingPtr), is_owning: false);
                }
            }

            /// old topology indexes are valid if true
            public new unsafe ref bool Identity
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_GetMutable_identity", ExactSpelling = true)]
                    extern static bool *__MR_BooleanResultMapper_Maps_GetMutable_identity(_Underlying *_this);
                    return ref *__MR_BooleanResultMapper_Maps_GetMutable_identity(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Maps() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_DefaultConstruct", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_DefaultConstruct();
                _UnderlyingPtr = __MR_BooleanResultMapper_Maps_DefaultConstruct();
            }

            /// Constructs `MR::BooleanResultMapper::Maps` elementwise.
            public unsafe Maps(MR._ByValue_FaceMap cut2origin, MR._ByValue_FaceMap cut2newFaces, MR._ByValue_WholeEdgeMap old2newEdges, MR._ByValue_VertMap old2newVerts, bool identity) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_ConstructFrom", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_ConstructFrom(MR.Misc._PassBy cut2origin_pass_by, MR.FaceMap._Underlying *cut2origin, MR.Misc._PassBy cut2newFaces_pass_by, MR.FaceMap._Underlying *cut2newFaces, MR.Misc._PassBy old2newEdges_pass_by, MR.WholeEdgeMap._Underlying *old2newEdges, MR.Misc._PassBy old2newVerts_pass_by, MR.VertMap._Underlying *old2newVerts, byte identity);
                _UnderlyingPtr = __MR_BooleanResultMapper_Maps_ConstructFrom(cut2origin.PassByMode, cut2origin.Value is not null ? cut2origin.Value._UnderlyingPtr : null, cut2newFaces.PassByMode, cut2newFaces.Value is not null ? cut2newFaces.Value._UnderlyingPtr : null, old2newEdges.PassByMode, old2newEdges.Value is not null ? old2newEdges.Value._UnderlyingPtr : null, old2newVerts.PassByMode, old2newVerts.Value is not null ? old2newVerts.Value._UnderlyingPtr : null, identity ? (byte)1 : (byte)0);
            }

            /// Generated from constructor `MR::BooleanResultMapper::Maps::Maps`.
            public unsafe Maps(MR.BooleanResultMapper._ByValue_Maps _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResultMapper.Maps._Underlying *_other);
                _UnderlyingPtr = __MR_BooleanResultMapper_Maps_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::BooleanResultMapper::Maps::operator=`.
            public unsafe MR.BooleanResultMapper.Maps Assign(MR.BooleanResultMapper._ByValue_Maps _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_Maps_AssignFromAnother", ExactSpelling = true)]
                extern static MR.BooleanResultMapper.Maps._Underlying *__MR_BooleanResultMapper_Maps_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BooleanResultMapper.Maps._Underlying *_other);
                return new(__MR_BooleanResultMapper_Maps_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Maps` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Maps`/`Const_Maps` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Maps
        {
            internal readonly Const_Maps? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Maps() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Maps(Const_Maps new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Maps(Const_Maps arg) {return new(arg);}
            public _ByValue_Maps(MR.Misc._Moved<Maps> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Maps(MR.Misc._Moved<Maps> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Maps` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Maps`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Maps`/`Const_Maps` directly.
        public class _InOptMut_Maps
        {
            public Maps? Opt;

            public _InOptMut_Maps() {}
            public _InOptMut_Maps(Maps value) {Opt = value;}
            public static implicit operator _InOptMut_Maps(Maps value) {return new(value);}
        }

        /// This is used for optional parameters of class `Maps` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Maps`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Maps`/`Const_Maps` to pass it to the function.
        public class _InOptConst_Maps
        {
            public Const_Maps? Opt;

            public _InOptConst_Maps() {}
            public _InOptConst_Maps(Const_Maps value) {Opt = value;}
            public static implicit operator _InOptConst_Maps(Const_Maps value) {return new(value);}
        }
    }

    /** \struct MR::BooleanResultMapper
    * \brief Structure to map old mesh BitSets to new
    * \details Structure to easily map topology of MR::boolean input meshes to result mesh
    *
    * This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MR::boolean to result mesh topology primitives
    * \sa \ref MR::boolean
    */
    /// Generated from class `MR::BooleanResultMapper`.
    /// This is the non-const half of the class.
    public class BooleanResultMapper : Const_BooleanResultMapper
    {
        internal unsafe BooleanResultMapper(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Array_MRBooleanResultMapperMaps_2 Maps_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_GetMutable_maps", ExactSpelling = true)]
                extern static MR.Std.Array_MRBooleanResultMapperMaps_2._Underlying *__MR_BooleanResultMapper_GetMutable_maps(_Underlying *_this);
                return new(__MR_BooleanResultMapper_GetMutable_maps(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BooleanResultMapper() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanResultMapper._Underlying *__MR_BooleanResultMapper_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanResultMapper_DefaultConstruct();
        }

        /// Generated from constructor `MR::BooleanResultMapper::BooleanResultMapper`.
        public unsafe BooleanResultMapper(MR._ByValue_BooleanResultMapper _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResultMapper._Underlying *__MR_BooleanResultMapper_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BooleanResultMapper._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanResultMapper_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BooleanResultMapper::operator=`.
        public unsafe MR.BooleanResultMapper Assign(MR._ByValue_BooleanResultMapper _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BooleanResultMapper._Underlying *__MR_BooleanResultMapper_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BooleanResultMapper._Underlying *_other);
            return new(__MR_BooleanResultMapper_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// returns updated oldBS leaving only faces that has corresponding ones in result mesh
        /// Generated from method `MR::BooleanResultMapper::filteredOldFaceBitSet`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> FilteredOldFaceBitSet(MR.Const_FaceBitSet oldBS, MR.BooleanResultMapper.MapObject obj)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanResultMapper_filteredOldFaceBitSet", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_BooleanResultMapper_filteredOldFaceBitSet(_Underlying *_this, MR.Const_FaceBitSet._Underlying *oldBS, MR.BooleanResultMapper.MapObject obj);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_BooleanResultMapper_filteredOldFaceBitSet(_UnderlyingPtr, oldBS._UnderlyingPtr, obj), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `BooleanResultMapper` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BooleanResultMapper`/`Const_BooleanResultMapper` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BooleanResultMapper
    {
        internal readonly Const_BooleanResultMapper? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BooleanResultMapper() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BooleanResultMapper(Const_BooleanResultMapper new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BooleanResultMapper(Const_BooleanResultMapper arg) {return new(arg);}
        public _ByValue_BooleanResultMapper(MR.Misc._Moved<BooleanResultMapper> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BooleanResultMapper(MR.Misc._Moved<BooleanResultMapper> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BooleanResultMapper` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BooleanResultMapper`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanResultMapper`/`Const_BooleanResultMapper` directly.
    public class _InOptMut_BooleanResultMapper
    {
        public BooleanResultMapper? Opt;

        public _InOptMut_BooleanResultMapper() {}
        public _InOptMut_BooleanResultMapper(BooleanResultMapper value) {Opt = value;}
        public static implicit operator _InOptMut_BooleanResultMapper(BooleanResultMapper value) {return new(value);}
    }

    /// This is used for optional parameters of class `BooleanResultMapper` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BooleanResultMapper`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanResultMapper`/`Const_BooleanResultMapper` to pass it to the function.
    public class _InOptConst_BooleanResultMapper
    {
        public Const_BooleanResultMapper? Opt;

        public _InOptConst_BooleanResultMapper() {}
        public _InOptConst_BooleanResultMapper(Const_BooleanResultMapper value) {Opt = value;}
        public static implicit operator _InOptConst_BooleanResultMapper(Const_BooleanResultMapper value) {return new(value);}
    }

    /// Parameters will be useful if specified
    /// Generated from class `MR::BooleanInternalParameters`.
    /// This is the const half of the class.
    public class Const_BooleanInternalParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BooleanInternalParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_BooleanInternalParameters_Destroy(_Underlying *_this);
            __MR_BooleanInternalParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BooleanInternalParameters() {Dispose(false);}

        /// Instance of original mesh with tree for better speed
        public unsafe ref readonly void * OriginalMeshA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_Get_originalMeshA", ExactSpelling = true)]
                extern static void **__MR_BooleanInternalParameters_Get_originalMeshA(_Underlying *_this);
                return ref *__MR_BooleanInternalParameters_Get_originalMeshA(_UnderlyingPtr);
            }
        }

        /// Instance of original mesh with tree for better speed
        public unsafe ref readonly void * OriginalMeshB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_Get_originalMeshB", ExactSpelling = true)]
                extern static void **__MR_BooleanInternalParameters_Get_originalMeshB(_Underlying *_this);
                return ref *__MR_BooleanInternalParameters_Get_originalMeshB(_UnderlyingPtr);
            }
        }

        /// Optional output cut edges of booleaned meshes
        public unsafe ref void * OptionalOutCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_Get_optionalOutCut", ExactSpelling = true)]
                extern static void **__MR_BooleanInternalParameters_Get_optionalOutCut(_Underlying *_this);
                return ref *__MR_BooleanInternalParameters_Get_optionalOutCut(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BooleanInternalParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanInternalParameters_DefaultConstruct();
        }

        /// Constructs `MR::BooleanInternalParameters` elementwise.
        public unsafe Const_BooleanInternalParameters(MR.Const_Mesh? originalMeshA, MR.Const_Mesh? originalMeshB, MR.Std.Vector_StdVectorMREdgeId? optionalOutCut) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_ConstructFrom(MR.Const_Mesh._Underlying *originalMeshA, MR.Const_Mesh._Underlying *originalMeshB, MR.Std.Vector_StdVectorMREdgeId._Underlying *optionalOutCut);
            _UnderlyingPtr = __MR_BooleanInternalParameters_ConstructFrom(originalMeshA is not null ? originalMeshA._UnderlyingPtr : null, originalMeshB is not null ? originalMeshB._UnderlyingPtr : null, optionalOutCut is not null ? optionalOutCut._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanInternalParameters::BooleanInternalParameters`.
        public unsafe Const_BooleanInternalParameters(MR.Const_BooleanInternalParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_ConstructFromAnother(MR.BooleanInternalParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanInternalParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Parameters will be useful if specified
    /// Generated from class `MR::BooleanInternalParameters`.
    /// This is the non-const half of the class.
    public class BooleanInternalParameters : Const_BooleanInternalParameters
    {
        internal unsafe BooleanInternalParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Instance of original mesh with tree for better speed
        public new unsafe ref readonly void * OriginalMeshA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_GetMutable_originalMeshA", ExactSpelling = true)]
                extern static void **__MR_BooleanInternalParameters_GetMutable_originalMeshA(_Underlying *_this);
                return ref *__MR_BooleanInternalParameters_GetMutable_originalMeshA(_UnderlyingPtr);
            }
        }

        /// Instance of original mesh with tree for better speed
        public new unsafe ref readonly void * OriginalMeshB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_GetMutable_originalMeshB", ExactSpelling = true)]
                extern static void **__MR_BooleanInternalParameters_GetMutable_originalMeshB(_Underlying *_this);
                return ref *__MR_BooleanInternalParameters_GetMutable_originalMeshB(_UnderlyingPtr);
            }
        }

        /// Optional output cut edges of booleaned meshes
        public new unsafe ref void * OptionalOutCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_GetMutable_optionalOutCut", ExactSpelling = true)]
                extern static void **__MR_BooleanInternalParameters_GetMutable_optionalOutCut(_Underlying *_this);
                return ref *__MR_BooleanInternalParameters_GetMutable_optionalOutCut(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BooleanInternalParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BooleanInternalParameters_DefaultConstruct();
        }

        /// Constructs `MR::BooleanInternalParameters` elementwise.
        public unsafe BooleanInternalParameters(MR.Const_Mesh? originalMeshA, MR.Const_Mesh? originalMeshB, MR.Std.Vector_StdVectorMREdgeId? optionalOutCut) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_ConstructFrom(MR.Const_Mesh._Underlying *originalMeshA, MR.Const_Mesh._Underlying *originalMeshB, MR.Std.Vector_StdVectorMREdgeId._Underlying *optionalOutCut);
            _UnderlyingPtr = __MR_BooleanInternalParameters_ConstructFrom(originalMeshA is not null ? originalMeshA._UnderlyingPtr : null, originalMeshB is not null ? originalMeshB._UnderlyingPtr : null, optionalOutCut is not null ? optionalOutCut._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BooleanInternalParameters::BooleanInternalParameters`.
        public unsafe BooleanInternalParameters(MR.Const_BooleanInternalParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_ConstructFromAnother(MR.BooleanInternalParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BooleanInternalParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::BooleanInternalParameters::operator=`.
        public unsafe MR.BooleanInternalParameters Assign(MR.Const_BooleanInternalParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BooleanInternalParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BooleanInternalParameters._Underlying *__MR_BooleanInternalParameters_AssignFromAnother(_Underlying *_this, MR.BooleanInternalParameters._Underlying *_other);
            return new(__MR_BooleanInternalParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `BooleanInternalParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BooleanInternalParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanInternalParameters`/`Const_BooleanInternalParameters` directly.
    public class _InOptMut_BooleanInternalParameters
    {
        public BooleanInternalParameters? Opt;

        public _InOptMut_BooleanInternalParameters() {}
        public _InOptMut_BooleanInternalParameters(BooleanInternalParameters value) {Opt = value;}
        public static implicit operator _InOptMut_BooleanInternalParameters(BooleanInternalParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `BooleanInternalParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BooleanInternalParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BooleanInternalParameters`/`Const_BooleanInternalParameters` to pass it to the function.
    public class _InOptConst_BooleanInternalParameters
    {
        public Const_BooleanInternalParameters? Opt;

        public _InOptConst_BooleanInternalParameters() {}
        public _InOptConst_BooleanInternalParameters(Const_BooleanInternalParameters value) {Opt = value;}
        public static implicit operator _InOptConst_BooleanInternalParameters(Const_BooleanInternalParameters value) {return new(value);}
    }

    /// Perform boolean operation on cut meshes
    /// \return mesh in space of meshA or error.
    /// \note: actually this function is meant to be internal, use "boolean" instead
    /// Generated from function `MR::doBooleanOperation`.
    /// Parameter `mergeAllNonIntersectingComponents` defaults to `false`.
    /// Parameter `intParams` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> DoBooleanOperation(MR.Misc._Moved<MR.Mesh> meshACut, MR.Misc._Moved<MR.Mesh> meshBCut, MR.Std.Const_Vector_StdVectorMREdgeId cutEdgesA, MR.Std.Const_Vector_StdVectorMREdgeId cutEdgesB, MR.BooleanOperation operation, MR.Const_AffineXf3f? rigidB2A = null, MR.BooleanResultMapper? mapper = null, bool? mergeAllNonIntersectingComponents = null, MR.Const_BooleanInternalParameters? intParams = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_doBooleanOperation", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_doBooleanOperation(MR.Mesh._Underlying *meshACut, MR.Mesh._Underlying *meshBCut, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *cutEdgesA, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *cutEdgesB, MR.BooleanOperation operation, MR.Const_AffineXf3f._Underlying *rigidB2A, MR.BooleanResultMapper._Underlying *mapper, byte *mergeAllNonIntersectingComponents, MR.Const_BooleanInternalParameters._Underlying *intParams);
        byte __deref_mergeAllNonIntersectingComponents = mergeAllNonIntersectingComponents.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_doBooleanOperation(meshACut.Value._UnderlyingPtr, meshBCut.Value._UnderlyingPtr, cutEdgesA._UnderlyingPtr, cutEdgesB._UnderlyingPtr, operation, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, mapper is not null ? mapper._UnderlyingPtr : null, mergeAllNonIntersectingComponents.HasValue ? &__deref_mergeAllNonIntersectingComponents : null, intParams is not null ? intParams._UnderlyingPtr : null), is_owning: true));
    }
}
