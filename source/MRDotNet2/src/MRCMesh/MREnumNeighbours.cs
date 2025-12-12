public static partial class MR
{
    /// Generated from class `MR::EnumNeihbourVertices`.
    /// This is the const half of the class.
    public class Const_EnumNeihbourVertices : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EnumNeihbourVertices(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_Destroy", ExactSpelling = true)]
            extern static void __MR_EnumNeihbourVertices_Destroy(_Underlying *_this);
            __MR_EnumNeihbourVertices_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EnumNeihbourVertices() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EnumNeihbourVertices() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EnumNeihbourVertices._Underlying *__MR_EnumNeihbourVertices_DefaultConstruct();
            _UnderlyingPtr = __MR_EnumNeihbourVertices_DefaultConstruct();
        }

        /// Generated from constructor `MR::EnumNeihbourVertices::EnumNeihbourVertices`.
        public unsafe Const_EnumNeihbourVertices(MR._ByValue_EnumNeihbourVertices _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EnumNeihbourVertices._Underlying *__MR_EnumNeihbourVertices_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EnumNeihbourVertices._Underlying *_other);
            _UnderlyingPtr = __MR_EnumNeihbourVertices_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::EnumNeihbourVertices`.
    /// This is the non-const half of the class.
    public class EnumNeihbourVertices : Const_EnumNeihbourVertices
    {
        internal unsafe EnumNeihbourVertices(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe EnumNeihbourVertices() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EnumNeihbourVertices._Underlying *__MR_EnumNeihbourVertices_DefaultConstruct();
            _UnderlyingPtr = __MR_EnumNeihbourVertices_DefaultConstruct();
        }

        /// Generated from constructor `MR::EnumNeihbourVertices::EnumNeihbourVertices`.
        public unsafe EnumNeihbourVertices(MR._ByValue_EnumNeihbourVertices _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EnumNeihbourVertices._Underlying *__MR_EnumNeihbourVertices_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EnumNeihbourVertices._Underlying *_other);
            _UnderlyingPtr = __MR_EnumNeihbourVertices_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::EnumNeihbourVertices::operator=`.
        public unsafe MR.EnumNeihbourVertices Assign(MR._ByValue_EnumNeihbourVertices _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EnumNeihbourVertices._Underlying *__MR_EnumNeihbourVertices_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.EnumNeihbourVertices._Underlying *_other);
            return new(__MR_EnumNeihbourVertices_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// invokes given predicate for vertices starting from \param start,
        /// and continuing to all immediate neighbours in depth-first order until the predicate returns false
        /// Generated from method `MR::EnumNeihbourVertices::run`.
        public unsafe void Run(MR.Const_MeshTopology topology, MR.VertId start, MR.Std.Const_Function_BoolFuncFromMRVertId pred)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_run_MR_VertId", ExactSpelling = true)]
            extern static void __MR_EnumNeihbourVertices_run_MR_VertId(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, MR.VertId start, MR.Std.Const_Function_BoolFuncFromMRVertId._Underlying *pred);
            __MR_EnumNeihbourVertices_run_MR_VertId(_UnderlyingPtr, topology._UnderlyingPtr, start, pred._UnderlyingPtr);
        }

        /// Generated from method `MR::EnumNeihbourVertices::run`.
        public unsafe void Run(MR.Const_MeshTopology topology, MR.Const_VertBitSet start, MR.Std.Const_Function_BoolFuncFromMRVertId pred)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourVertices_run_MR_VertBitSet", ExactSpelling = true)]
            extern static void __MR_EnumNeihbourVertices_run_MR_VertBitSet(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, MR.Const_VertBitSet._Underlying *start, MR.Std.Const_Function_BoolFuncFromMRVertId._Underlying *pred);
            __MR_EnumNeihbourVertices_run_MR_VertBitSet(_UnderlyingPtr, topology._UnderlyingPtr, start._UnderlyingPtr, pred._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EnumNeihbourVertices` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EnumNeihbourVertices`/`Const_EnumNeihbourVertices` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EnumNeihbourVertices
    {
        internal readonly Const_EnumNeihbourVertices? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EnumNeihbourVertices() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_EnumNeihbourVertices(Const_EnumNeihbourVertices new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EnumNeihbourVertices(Const_EnumNeihbourVertices arg) {return new(arg);}
        public _ByValue_EnumNeihbourVertices(MR.Misc._Moved<EnumNeihbourVertices> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EnumNeihbourVertices(MR.Misc._Moved<EnumNeihbourVertices> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EnumNeihbourVertices` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EnumNeihbourVertices`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EnumNeihbourVertices`/`Const_EnumNeihbourVertices` directly.
    public class _InOptMut_EnumNeihbourVertices
    {
        public EnumNeihbourVertices? Opt;

        public _InOptMut_EnumNeihbourVertices() {}
        public _InOptMut_EnumNeihbourVertices(EnumNeihbourVertices value) {Opt = value;}
        public static implicit operator _InOptMut_EnumNeihbourVertices(EnumNeihbourVertices value) {return new(value);}
    }

    /// This is used for optional parameters of class `EnumNeihbourVertices` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EnumNeihbourVertices`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EnumNeihbourVertices`/`Const_EnumNeihbourVertices` to pass it to the function.
    public class _InOptConst_EnumNeihbourVertices
    {
        public Const_EnumNeihbourVertices? Opt;

        public _InOptConst_EnumNeihbourVertices() {}
        public _InOptConst_EnumNeihbourVertices(Const_EnumNeihbourVertices value) {Opt = value;}
        public static implicit operator _InOptConst_EnumNeihbourVertices(Const_EnumNeihbourVertices value) {return new(value);}
    }

    /// Generated from class `MR::EnumNeihbourFaces`.
    /// This is the const half of the class.
    public class Const_EnumNeihbourFaces : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EnumNeihbourFaces(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_Destroy", ExactSpelling = true)]
            extern static void __MR_EnumNeihbourFaces_Destroy(_Underlying *_this);
            __MR_EnumNeihbourFaces_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EnumNeihbourFaces() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EnumNeihbourFaces() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EnumNeihbourFaces._Underlying *__MR_EnumNeihbourFaces_DefaultConstruct();
            _UnderlyingPtr = __MR_EnumNeihbourFaces_DefaultConstruct();
        }

        /// Generated from constructor `MR::EnumNeihbourFaces::EnumNeihbourFaces`.
        public unsafe Const_EnumNeihbourFaces(MR._ByValue_EnumNeihbourFaces _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EnumNeihbourFaces._Underlying *__MR_EnumNeihbourFaces_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EnumNeihbourFaces._Underlying *_other);
            _UnderlyingPtr = __MR_EnumNeihbourFaces_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::EnumNeihbourFaces`.
    /// This is the non-const half of the class.
    public class EnumNeihbourFaces : Const_EnumNeihbourFaces
    {
        internal unsafe EnumNeihbourFaces(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe EnumNeihbourFaces() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EnumNeihbourFaces._Underlying *__MR_EnumNeihbourFaces_DefaultConstruct();
            _UnderlyingPtr = __MR_EnumNeihbourFaces_DefaultConstruct();
        }

        /// Generated from constructor `MR::EnumNeihbourFaces::EnumNeihbourFaces`.
        public unsafe EnumNeihbourFaces(MR._ByValue_EnumNeihbourFaces _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EnumNeihbourFaces._Underlying *__MR_EnumNeihbourFaces_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EnumNeihbourFaces._Underlying *_other);
            _UnderlyingPtr = __MR_EnumNeihbourFaces_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::EnumNeihbourFaces::operator=`.
        public unsafe MR.EnumNeihbourFaces Assign(MR._ByValue_EnumNeihbourFaces _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EnumNeihbourFaces._Underlying *__MR_EnumNeihbourFaces_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.EnumNeihbourFaces._Underlying *_other);
            return new(__MR_EnumNeihbourFaces_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// invokes given predicate for faces starting from all incident to \param start,
        /// and continuing to all immediate neighbours in depth-first order until the predicate returns false
        /// Generated from method `MR::EnumNeihbourFaces::run`.
        public unsafe void Run(MR.Const_MeshTopology topology, MR.VertId start, MR.Std.Const_Function_BoolFuncFromMRFaceId pred)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EnumNeihbourFaces_run", ExactSpelling = true)]
            extern static void __MR_EnumNeihbourFaces_run(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, MR.VertId start, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *pred);
            __MR_EnumNeihbourFaces_run(_UnderlyingPtr, topology._UnderlyingPtr, start, pred._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EnumNeihbourFaces` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EnumNeihbourFaces`/`Const_EnumNeihbourFaces` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EnumNeihbourFaces
    {
        internal readonly Const_EnumNeihbourFaces? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EnumNeihbourFaces() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_EnumNeihbourFaces(Const_EnumNeihbourFaces new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EnumNeihbourFaces(Const_EnumNeihbourFaces arg) {return new(arg);}
        public _ByValue_EnumNeihbourFaces(MR.Misc._Moved<EnumNeihbourFaces> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EnumNeihbourFaces(MR.Misc._Moved<EnumNeihbourFaces> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EnumNeihbourFaces` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EnumNeihbourFaces`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EnumNeihbourFaces`/`Const_EnumNeihbourFaces` directly.
    public class _InOptMut_EnumNeihbourFaces
    {
        public EnumNeihbourFaces? Opt;

        public _InOptMut_EnumNeihbourFaces() {}
        public _InOptMut_EnumNeihbourFaces(EnumNeihbourFaces value) {Opt = value;}
        public static implicit operator _InOptMut_EnumNeihbourFaces(EnumNeihbourFaces value) {return new(value);}
    }

    /// This is used for optional parameters of class `EnumNeihbourFaces` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EnumNeihbourFaces`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EnumNeihbourFaces`/`Const_EnumNeihbourFaces` to pass it to the function.
    public class _InOptConst_EnumNeihbourFaces
    {
        public Const_EnumNeihbourFaces? Opt;

        public _InOptConst_EnumNeihbourFaces() {}
        public _InOptConst_EnumNeihbourFaces(Const_EnumNeihbourFaces value) {Opt = value;}
        public static implicit operator _InOptConst_EnumNeihbourFaces(Const_EnumNeihbourFaces value) {return new(value);}
    }

    /// computes Euclidean 3D distances from given start point to all neighbor vertices within given \param range
    /// and to first vertices with the distance more than range
    /// Generated from function `MR::computeSpaceDistances`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSpaceDistances(MR.Const_Mesh mesh, MR.Const_PointOnFace start, float range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSpaceDistances", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSpaceDistances(MR.Const_Mesh._Underlying *mesh, MR.Const_PointOnFace._Underlying *start, float range);
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSpaceDistances(mesh._UnderlyingPtr, start._UnderlyingPtr, range), is_owning: true));
    }

    /// calculates all neighbor vertices within a given \param range
    /// and to first vertices with the distance more than range
    /// \param rangeSq square of range
    /// Generated from function `MR::findNeighborVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> FindNeighborVerts(MR.Const_Mesh mesh, MR.Const_PointOnFace start, float rangeSq)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findNeighborVerts", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_findNeighborVerts(MR.Const_Mesh._Underlying *mesh, MR.Const_PointOnFace._Underlying *start, float rangeSq);
        return MR.Misc.Move(new MR.VertBitSet(__MR_findNeighborVerts(mesh._UnderlyingPtr, start._UnderlyingPtr, rangeSq), is_owning: true));
    }
}
