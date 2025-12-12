public static partial class MR
{
    /// given a spanning tree of edges in the mesh (or forest in case of several connected components),
    /// prepares to build quickly a path along tree edges between any two vertices
    /// Generated from class `MR::InTreePathBuilder`.
    /// This is the const half of the class.
    public class Const_InTreePathBuilder : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_InTreePathBuilder(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InTreePathBuilder_Destroy", ExactSpelling = true)]
            extern static void __MR_InTreePathBuilder_Destroy(_Underlying *_this);
            __MR_InTreePathBuilder_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_InTreePathBuilder() {Dispose(false);}

        /// Generated from constructor `MR::InTreePathBuilder::InTreePathBuilder`.
        public unsafe Const_InTreePathBuilder(MR._ByValue_InTreePathBuilder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InTreePathBuilder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InTreePathBuilder._Underlying *__MR_InTreePathBuilder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.InTreePathBuilder._Underlying *_other);
            _UnderlyingPtr = __MR_InTreePathBuilder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::InTreePathBuilder::InTreePathBuilder`.
        public unsafe Const_InTreePathBuilder(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet treeEdges) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InTreePathBuilder_Construct", ExactSpelling = true)]
            extern static MR.InTreePathBuilder._Underlying *__MR_InTreePathBuilder_Construct(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *treeEdges);
            _UnderlyingPtr = __MR_InTreePathBuilder_Construct(topology._UnderlyingPtr, treeEdges._UnderlyingPtr);
        }

        /// finds the path in tree from start vertex to finish vertex
        /// Generated from method `MR::InTreePathBuilder::build`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> Build(MR.VertId start, MR.VertId finish)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InTreePathBuilder_build", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeId._Underlying *__MR_InTreePathBuilder_build(_Underlying *_this, MR.VertId start, MR.VertId finish);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_InTreePathBuilder_build(_UnderlyingPtr, start, finish), is_owning: true));
        }
    }

    /// given a spanning tree of edges in the mesh (or forest in case of several connected components),
    /// prepares to build quickly a path along tree edges between any two vertices
    /// Generated from class `MR::InTreePathBuilder`.
    /// This is the non-const half of the class.
    public class InTreePathBuilder : Const_InTreePathBuilder
    {
        internal unsafe InTreePathBuilder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::InTreePathBuilder::InTreePathBuilder`.
        public unsafe InTreePathBuilder(MR._ByValue_InTreePathBuilder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InTreePathBuilder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InTreePathBuilder._Underlying *__MR_InTreePathBuilder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.InTreePathBuilder._Underlying *_other);
            _UnderlyingPtr = __MR_InTreePathBuilder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::InTreePathBuilder::InTreePathBuilder`.
        public unsafe InTreePathBuilder(MR.Const_MeshTopology topology, MR.Const_UndirectedEdgeBitSet treeEdges) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InTreePathBuilder_Construct", ExactSpelling = true)]
            extern static MR.InTreePathBuilder._Underlying *__MR_InTreePathBuilder_Construct(MR.Const_MeshTopology._Underlying *topology, MR.Const_UndirectedEdgeBitSet._Underlying *treeEdges);
            _UnderlyingPtr = __MR_InTreePathBuilder_Construct(topology._UnderlyingPtr, treeEdges._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `InTreePathBuilder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `InTreePathBuilder`/`Const_InTreePathBuilder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_InTreePathBuilder
    {
        internal readonly Const_InTreePathBuilder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_InTreePathBuilder(Const_InTreePathBuilder new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_InTreePathBuilder(Const_InTreePathBuilder arg) {return new(arg);}
        public _ByValue_InTreePathBuilder(MR.Misc._Moved<InTreePathBuilder> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_InTreePathBuilder(MR.Misc._Moved<InTreePathBuilder> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `InTreePathBuilder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_InTreePathBuilder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InTreePathBuilder`/`Const_InTreePathBuilder` directly.
    public class _InOptMut_InTreePathBuilder
    {
        public InTreePathBuilder? Opt;

        public _InOptMut_InTreePathBuilder() {}
        public _InOptMut_InTreePathBuilder(InTreePathBuilder value) {Opt = value;}
        public static implicit operator _InOptMut_InTreePathBuilder(InTreePathBuilder value) {return new(value);}
    }

    /// This is used for optional parameters of class `InTreePathBuilder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_InTreePathBuilder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InTreePathBuilder`/`Const_InTreePathBuilder` to pass it to the function.
    public class _InOptConst_InTreePathBuilder
    {
        public Const_InTreePathBuilder? Opt;

        public _InOptConst_InTreePathBuilder() {}
        public _InOptConst_InTreePathBuilder(Const_InTreePathBuilder value) {Opt = value;}
        public static implicit operator _InOptConst_InTreePathBuilder(Const_InTreePathBuilder value) {return new(value);}
    }
}
