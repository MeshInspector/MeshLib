public static partial class MR
{
    /// settings defining regular grid, where each quadrangular cell is split on two triangles in one of two ways
    /// Generated from class `MR::GridSettings`.
    /// This is the const half of the class.
    public class Const_GridSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GridSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_GridSettings_Destroy(_Underlying *_this);
            __MR_GridSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GridSettings() {Dispose(false);}

        /// the number of cells in X and Y dimensions;
        /// the number of vertices will be at most (X+1)*(Y+1)
        public unsafe MR.Const_Vector2i Dim
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_Get_dim", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_GridSettings_Get_dim(_Underlying *_this);
                return new(__MR_GridSettings_Get_dim(_UnderlyingPtr), is_owning: false);
            }
        }

        /// grid coordinates to vertex Id; invalid vertex Id means that this vertex is missing in grid;
        /// index is x + y * ( settings.dim.x + 1 )
        public unsafe MR.Const_BMap_MRVertId_MRUint64T VertIds
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_Get_vertIds", ExactSpelling = true)]
                extern static MR.Const_BMap_MRVertId_MRUint64T._Underlying *__MR_GridSettings_Get_vertIds(_Underlying *_this);
                return new(__MR_GridSettings_Get_vertIds(_UnderlyingPtr), is_owning: false);
            }
        }

        /// grid coordinates of lower-left vertex and edge-type to edgeId with the origin in this vertex;
        /// both vertices of valid edge must be valid as well;
        /// index is 4 * ( x + y * ( settings.dim.x + 1 ) ) + edgeType
        public unsafe MR.Const_BMap_MRUndirectedEdgeId_MRUint64T UedgeIds
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_Get_uedgeIds", ExactSpelling = true)]
                extern static MR.Const_BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_GridSettings_Get_uedgeIds(_Underlying *_this);
                return new(__MR_GridSettings_Get_uedgeIds(_UnderlyingPtr), is_owning: false);
            }
        }

        /// grid coordinates of lower-left vertex and triangle-type to faceId;
        /// all 3 vertices and all 3 edges of valid face must be valid as well;
        /// index is 2 * ( x + y * settings.dim.x ) + triType
        public unsafe MR.Const_BMap_MRFaceId_MRUint64T FaceIds
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_Get_faceIds", ExactSpelling = true)]
                extern static MR.Const_BMap_MRFaceId_MRUint64T._Underlying *__MR_GridSettings_Get_faceIds(_Underlying *_this);
                return new(__MR_GridSettings_Get_faceIds(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GridSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_GridSettings_DefaultConstruct();
        }

        /// Constructs `MR::GridSettings` elementwise.
        public unsafe Const_GridSettings(MR.Vector2i dim, MR._ByValue_BMap_MRVertId_MRUint64T vertIds, MR._ByValue_BMap_MRUndirectedEdgeId_MRUint64T uedgeIds, MR._ByValue_BMap_MRFaceId_MRUint64T faceIds) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_ConstructFrom(MR.Vector2i dim, MR.Misc._PassBy vertIds_pass_by, MR.BMap_MRVertId_MRUint64T._Underlying *vertIds, MR.Misc._PassBy uedgeIds_pass_by, MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *uedgeIds, MR.Misc._PassBy faceIds_pass_by, MR.BMap_MRFaceId_MRUint64T._Underlying *faceIds);
            _UnderlyingPtr = __MR_GridSettings_ConstructFrom(dim, vertIds.PassByMode, vertIds.Value is not null ? vertIds.Value._UnderlyingPtr : null, uedgeIds.PassByMode, uedgeIds.Value is not null ? uedgeIds.Value._UnderlyingPtr : null, faceIds.PassByMode, faceIds.Value is not null ? faceIds.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::GridSettings::GridSettings`.
        public unsafe Const_GridSettings(MR._ByValue_GridSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GridSettings._Underlying *_other);
            _UnderlyingPtr = __MR_GridSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        public enum EdgeType : int
        {
            // (x,y) - (x+1,y)
            Horizontal = 0,
            // (x,y) - (x,y+1)
            Vertical = 1,
            // (x,y) - (x+1,y+1)
            DiagonalA = 2,
            // (x+1,y) - (x,y+1)
            DiagonalB = 3,
        }

        public enum TriType : int
        {
            // (x,y), (x+1,y), (x+1,y+1) if DiagonalA or (x,y), (x+1,y), (x,y+1) if DiagonalB
            Lower = 0,
            // (x,y), (x+1,y+1), (x,y+1) if DiagonalA or (x+1,y), (x+1,y+1), (x,y+1) if DiagonalB
            Upper = 1,
        }
    }

    /// settings defining regular grid, where each quadrangular cell is split on two triangles in one of two ways
    /// Generated from class `MR::GridSettings`.
    /// This is the non-const half of the class.
    public class GridSettings : Const_GridSettings
    {
        internal unsafe GridSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the number of cells in X and Y dimensions;
        /// the number of vertices will be at most (X+1)*(Y+1)
        public new unsafe MR.Mut_Vector2i Dim
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_GetMutable_dim", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_GridSettings_GetMutable_dim(_Underlying *_this);
                return new(__MR_GridSettings_GetMutable_dim(_UnderlyingPtr), is_owning: false);
            }
        }

        /// grid coordinates to vertex Id; invalid vertex Id means that this vertex is missing in grid;
        /// index is x + y * ( settings.dim.x + 1 )
        public new unsafe MR.BMap_MRVertId_MRUint64T VertIds
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_GetMutable_vertIds", ExactSpelling = true)]
                extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_GridSettings_GetMutable_vertIds(_Underlying *_this);
                return new(__MR_GridSettings_GetMutable_vertIds(_UnderlyingPtr), is_owning: false);
            }
        }

        /// grid coordinates of lower-left vertex and edge-type to edgeId with the origin in this vertex;
        /// both vertices of valid edge must be valid as well;
        /// index is 4 * ( x + y * ( settings.dim.x + 1 ) ) + edgeType
        public new unsafe MR.BMap_MRUndirectedEdgeId_MRUint64T UedgeIds
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_GetMutable_uedgeIds", ExactSpelling = true)]
                extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_GridSettings_GetMutable_uedgeIds(_Underlying *_this);
                return new(__MR_GridSettings_GetMutable_uedgeIds(_UnderlyingPtr), is_owning: false);
            }
        }

        /// grid coordinates of lower-left vertex and triangle-type to faceId;
        /// all 3 vertices and all 3 edges of valid face must be valid as well;
        /// index is 2 * ( x + y * settings.dim.x ) + triType
        public new unsafe MR.BMap_MRFaceId_MRUint64T FaceIds
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_GetMutable_faceIds", ExactSpelling = true)]
                extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_GridSettings_GetMutable_faceIds(_Underlying *_this);
                return new(__MR_GridSettings_GetMutable_faceIds(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe GridSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_GridSettings_DefaultConstruct();
        }

        /// Constructs `MR::GridSettings` elementwise.
        public unsafe GridSettings(MR.Vector2i dim, MR._ByValue_BMap_MRVertId_MRUint64T vertIds, MR._ByValue_BMap_MRUndirectedEdgeId_MRUint64T uedgeIds, MR._ByValue_BMap_MRFaceId_MRUint64T faceIds) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_ConstructFrom(MR.Vector2i dim, MR.Misc._PassBy vertIds_pass_by, MR.BMap_MRVertId_MRUint64T._Underlying *vertIds, MR.Misc._PassBy uedgeIds_pass_by, MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *uedgeIds, MR.Misc._PassBy faceIds_pass_by, MR.BMap_MRFaceId_MRUint64T._Underlying *faceIds);
            _UnderlyingPtr = __MR_GridSettings_ConstructFrom(dim, vertIds.PassByMode, vertIds.Value is not null ? vertIds.Value._UnderlyingPtr : null, uedgeIds.PassByMode, uedgeIds.Value is not null ? uedgeIds.Value._UnderlyingPtr : null, faceIds.PassByMode, faceIds.Value is not null ? faceIds.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::GridSettings::GridSettings`.
        public unsafe GridSettings(MR._ByValue_GridSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GridSettings._Underlying *_other);
            _UnderlyingPtr = __MR_GridSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::GridSettings::operator=`.
        public unsafe MR.GridSettings Assign(MR._ByValue_GridSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GridSettings._Underlying *__MR_GridSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GridSettings._Underlying *_other);
            return new(__MR_GridSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `GridSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `GridSettings`/`Const_GridSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_GridSettings
    {
        internal readonly Const_GridSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_GridSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_GridSettings(MR.Misc._Moved<GridSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_GridSettings(MR.Misc._Moved<GridSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `GridSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GridSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GridSettings`/`Const_GridSettings` directly.
    public class _InOptMut_GridSettings
    {
        public GridSettings? Opt;

        public _InOptMut_GridSettings() {}
        public _InOptMut_GridSettings(GridSettings value) {Opt = value;}
        public static implicit operator _InOptMut_GridSettings(GridSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `GridSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GridSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GridSettings`/`Const_GridSettings` to pass it to the function.
    public class _InOptConst_GridSettings
    {
        public Const_GridSettings? Opt;

        public _InOptConst_GridSettings() {}
        public _InOptConst_GridSettings(Const_GridSettings value) {Opt = value;}
        public static implicit operator _InOptConst_GridSettings(Const_GridSettings value) {return new(value);}
    }
}
