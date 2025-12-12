public static partial class MR
{
    /// Generated from class `MR::FlowOrigin`.
    /// This is the const half of the class.
    public class Const_FlowOrigin : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FlowOrigin(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_Destroy", ExactSpelling = true)]
            extern static void __MR_FlowOrigin_Destroy(_Underlying *_this);
            __MR_FlowOrigin_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FlowOrigin() {Dispose(false);}

        /// point on the mesh, where this flow starts
        public unsafe MR.Const_MeshTriPoint Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_Get_point", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_FlowOrigin_Get_point(_Underlying *_this);
                return new(__MR_FlowOrigin_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// amount of flow, e.g. can be proportional to the horizontal area associated with the start point
        public unsafe float Amount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_Get_amount", ExactSpelling = true)]
                extern static float *__MR_FlowOrigin_Get_amount(_Underlying *_this);
                return *__MR_FlowOrigin_Get_amount(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FlowOrigin() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_DefaultConstruct();
            _UnderlyingPtr = __MR_FlowOrigin_DefaultConstruct();
        }

        /// Constructs `MR::FlowOrigin` elementwise.
        public unsafe Const_FlowOrigin(MR.Const_MeshTriPoint point, float amount) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_ConstructFrom", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_ConstructFrom(MR.MeshTriPoint._Underlying *point, float amount);
            _UnderlyingPtr = __MR_FlowOrigin_ConstructFrom(point._UnderlyingPtr, amount);
        }

        /// Generated from constructor `MR::FlowOrigin::FlowOrigin`.
        public unsafe Const_FlowOrigin(MR.Const_FlowOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_ConstructFromAnother(MR.FlowOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_FlowOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::FlowOrigin`.
    /// This is the non-const half of the class.
    public class FlowOrigin : Const_FlowOrigin
    {
        internal unsafe FlowOrigin(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// point on the mesh, where this flow starts
        public new unsafe MR.MeshTriPoint Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_GetMutable_point", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_FlowOrigin_GetMutable_point(_Underlying *_this);
                return new(__MR_FlowOrigin_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        /// amount of flow, e.g. can be proportional to the horizontal area associated with the start point
        public new unsafe ref float Amount
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_GetMutable_amount", ExactSpelling = true)]
                extern static float *__MR_FlowOrigin_GetMutable_amount(_Underlying *_this);
                return ref *__MR_FlowOrigin_GetMutable_amount(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FlowOrigin() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_DefaultConstruct();
            _UnderlyingPtr = __MR_FlowOrigin_DefaultConstruct();
        }

        /// Constructs `MR::FlowOrigin` elementwise.
        public unsafe FlowOrigin(MR.Const_MeshTriPoint point, float amount) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_ConstructFrom", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_ConstructFrom(MR.MeshTriPoint._Underlying *point, float amount);
            _UnderlyingPtr = __MR_FlowOrigin_ConstructFrom(point._UnderlyingPtr, amount);
        }

        /// Generated from constructor `MR::FlowOrigin::FlowOrigin`.
        public unsafe FlowOrigin(MR.Const_FlowOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_ConstructFromAnother(MR.FlowOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_FlowOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::FlowOrigin::operator=`.
        public unsafe MR.FlowOrigin Assign(MR.Const_FlowOrigin _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowOrigin_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FlowOrigin._Underlying *__MR_FlowOrigin_AssignFromAnother(_Underlying *_this, MR.FlowOrigin._Underlying *_other);
            return new(__MR_FlowOrigin_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FlowOrigin` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FlowOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FlowOrigin`/`Const_FlowOrigin` directly.
    public class _InOptMut_FlowOrigin
    {
        public FlowOrigin? Opt;

        public _InOptMut_FlowOrigin() {}
        public _InOptMut_FlowOrigin(FlowOrigin value) {Opt = value;}
        public static implicit operator _InOptMut_FlowOrigin(FlowOrigin value) {return new(value);}
    }

    /// This is used for optional parameters of class `FlowOrigin` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FlowOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FlowOrigin`/`Const_FlowOrigin` to pass it to the function.
    public class _InOptConst_FlowOrigin
    {
        public Const_FlowOrigin? Opt;

        public _InOptConst_FlowOrigin() {}
        public _InOptConst_FlowOrigin(Const_FlowOrigin value) {Opt = value;}
        public static implicit operator _InOptConst_FlowOrigin(Const_FlowOrigin value) {return new(value);}
    }

    /// Generated from class `MR::OutputFlows`.
    /// This is the const half of the class.
    public class Const_OutputFlows : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OutputFlows(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_Destroy", ExactSpelling = true)]
            extern static void __MR_OutputFlows_Destroy(_Underlying *_this);
            __MR_OutputFlows_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OutputFlows() {Dispose(false);}

        /// optional output: lines of all flows
        public unsafe ref void * PPolyline
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_Get_pPolyline", ExactSpelling = true)]
                extern static void **__MR_OutputFlows_Get_pPolyline(_Underlying *_this);
                return ref *__MR_OutputFlows_Get_pPolyline(_UnderlyingPtr);
            }
        }

        /// optional output: flow in each line of outPolyline
        public unsafe ref void * PFlowPerEdge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_Get_pFlowPerEdge", ExactSpelling = true)]
                extern static void **__MR_OutputFlows_Get_pFlowPerEdge(_Underlying *_this);
                return ref *__MR_OutputFlows_Get_pFlowPerEdge(_UnderlyingPtr);
            }
        }

        /// output in outPolyline only the flows with the amount greater than
        public unsafe float AmountGreaterThan
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_Get_amountGreaterThan", ExactSpelling = true)]
                extern static float *__MR_OutputFlows_Get_amountGreaterThan(_Underlying *_this);
                return *__MR_OutputFlows_Get_amountGreaterThan(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OutputFlows() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_DefaultConstruct();
            _UnderlyingPtr = __MR_OutputFlows_DefaultConstruct();
        }

        /// Constructs `MR::OutputFlows` elementwise.
        public unsafe Const_OutputFlows(MR.Polyline3? pPolyline, MR.UndirectedEdgeScalars? pFlowPerEdge, float amountGreaterThan) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_ConstructFrom", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_ConstructFrom(MR.Polyline3._Underlying *pPolyline, MR.UndirectedEdgeScalars._Underlying *pFlowPerEdge, float amountGreaterThan);
            _UnderlyingPtr = __MR_OutputFlows_ConstructFrom(pPolyline is not null ? pPolyline._UnderlyingPtr : null, pFlowPerEdge is not null ? pFlowPerEdge._UnderlyingPtr : null, amountGreaterThan);
        }

        /// Generated from constructor `MR::OutputFlows::OutputFlows`.
        public unsafe Const_OutputFlows(MR.Const_OutputFlows _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_ConstructFromAnother(MR.OutputFlows._Underlying *_other);
            _UnderlyingPtr = __MR_OutputFlows_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::OutputFlows`.
    /// This is the non-const half of the class.
    public class OutputFlows : Const_OutputFlows
    {
        internal unsafe OutputFlows(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// optional output: lines of all flows
        public new unsafe ref void * PPolyline
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_GetMutable_pPolyline", ExactSpelling = true)]
                extern static void **__MR_OutputFlows_GetMutable_pPolyline(_Underlying *_this);
                return ref *__MR_OutputFlows_GetMutable_pPolyline(_UnderlyingPtr);
            }
        }

        /// optional output: flow in each line of outPolyline
        public new unsafe ref void * PFlowPerEdge
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_GetMutable_pFlowPerEdge", ExactSpelling = true)]
                extern static void **__MR_OutputFlows_GetMutable_pFlowPerEdge(_Underlying *_this);
                return ref *__MR_OutputFlows_GetMutable_pFlowPerEdge(_UnderlyingPtr);
            }
        }

        /// output in outPolyline only the flows with the amount greater than
        public new unsafe ref float AmountGreaterThan
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_GetMutable_amountGreaterThan", ExactSpelling = true)]
                extern static float *__MR_OutputFlows_GetMutable_amountGreaterThan(_Underlying *_this);
                return ref *__MR_OutputFlows_GetMutable_amountGreaterThan(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OutputFlows() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_DefaultConstruct();
            _UnderlyingPtr = __MR_OutputFlows_DefaultConstruct();
        }

        /// Constructs `MR::OutputFlows` elementwise.
        public unsafe OutputFlows(MR.Polyline3? pPolyline, MR.UndirectedEdgeScalars? pFlowPerEdge, float amountGreaterThan) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_ConstructFrom", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_ConstructFrom(MR.Polyline3._Underlying *pPolyline, MR.UndirectedEdgeScalars._Underlying *pFlowPerEdge, float amountGreaterThan);
            _UnderlyingPtr = __MR_OutputFlows_ConstructFrom(pPolyline is not null ? pPolyline._UnderlyingPtr : null, pFlowPerEdge is not null ? pFlowPerEdge._UnderlyingPtr : null, amountGreaterThan);
        }

        /// Generated from constructor `MR::OutputFlows::OutputFlows`.
        public unsafe OutputFlows(MR.Const_OutputFlows _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_ConstructFromAnother(MR.OutputFlows._Underlying *_other);
            _UnderlyingPtr = __MR_OutputFlows_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OutputFlows::operator=`.
        public unsafe MR.OutputFlows Assign(MR.Const_OutputFlows _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutputFlows_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OutputFlows._Underlying *__MR_OutputFlows_AssignFromAnother(_Underlying *_this, MR.OutputFlows._Underlying *_other);
            return new(__MR_OutputFlows_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OutputFlows` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OutputFlows`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OutputFlows`/`Const_OutputFlows` directly.
    public class _InOptMut_OutputFlows
    {
        public OutputFlows? Opt;

        public _InOptMut_OutputFlows() {}
        public _InOptMut_OutputFlows(OutputFlows value) {Opt = value;}
        public static implicit operator _InOptMut_OutputFlows(OutputFlows value) {return new(value);}
    }

    /// This is used for optional parameters of class `OutputFlows` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OutputFlows`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OutputFlows`/`Const_OutputFlows` to pass it to the function.
    public class _InOptConst_OutputFlows
    {
        public Const_OutputFlows? Opt;

        public _InOptConst_OutputFlows() {}
        public _InOptConst_OutputFlows(Const_OutputFlows value) {Opt = value;}
        public static implicit operator _InOptConst_OutputFlows(Const_OutputFlows value) {return new(value);}
    }

    /// this class can track multiple flows and find in each mesh vertex the amount of water reached it
    /// Generated from class `MR::FlowAggregator`.
    /// This is the const half of the class.
    public class Const_FlowAggregator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FlowAggregator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Destroy", ExactSpelling = true)]
            extern static void __MR_FlowAggregator_Destroy(_Underlying *_this);
            __MR_FlowAggregator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FlowAggregator() {Dispose(false);}

        /// Generated from constructor `MR::FlowAggregator::FlowAggregator`.
        public unsafe Const_FlowAggregator(MR._ByValue_FlowAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FlowAggregator._Underlying *__MR_FlowAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FlowAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_FlowAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// prepares the processing of given mesh with given height in each vertex
        /// Generated from constructor `MR::FlowAggregator::FlowAggregator`.
        public unsafe Const_FlowAggregator(MR.Const_Mesh mesh, MR.Const_VertScalars heights) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Construct", ExactSpelling = true)]
            extern static MR.FlowAggregator._Underlying *__MR_FlowAggregator_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_VertScalars._Underlying *heights);
            _UnderlyingPtr = __MR_FlowAggregator_Construct(mesh._UnderlyingPtr, heights._UnderlyingPtr);
        }

        /// tracks multiple flows
        /// \param starts the origin of each flow (should be uniformly sampled over the terrain)
        /// \return the flow reached each mesh vertex
        /// Generated from method `MR::FlowAggregator::computeFlow`.
        /// Parameter `out_` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.VertScalars> ComputeFlow(MR.Std.Const_Vector_MRFlowOrigin starts, MR.Const_OutputFlows? out_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeFlow_2_std_vector_MR_FlowOrigin", ExactSpelling = true)]
            extern static MR.VertScalars._Underlying *__MR_FlowAggregator_computeFlow_2_std_vector_MR_FlowOrigin(_Underlying *_this, MR.Std.Const_Vector_MRFlowOrigin._Underlying *starts, MR.Const_OutputFlows._Underlying *out_);
            return MR.Misc.Move(new MR.VertScalars(__MR_FlowAggregator_computeFlow_2_std_vector_MR_FlowOrigin(_UnderlyingPtr, starts._UnderlyingPtr, out_ is not null ? out_._UnderlyingPtr : null), is_owning: true));
        }

        // same with all amounts equal to 1
        /// Generated from method `MR::FlowAggregator::computeFlow`.
        /// Parameter `out_` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.VertScalars> ComputeFlow(MR.Std.Const_Vector_MRMeshTriPoint starts, MR.Const_OutputFlows? out_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeFlow_2_std_vector_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.VertScalars._Underlying *__MR_FlowAggregator_computeFlow_2_std_vector_MR_MeshTriPoint(_Underlying *_this, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *starts, MR.Const_OutputFlows._Underlying *out_);
            return MR.Misc.Move(new MR.VertScalars(__MR_FlowAggregator_computeFlow_2_std_vector_MR_MeshTriPoint(_UnderlyingPtr, starts._UnderlyingPtr, out_ is not null ? out_._UnderlyingPtr : null), is_owning: true));
        }

        // general version that supplies starts in a functional way
        /// Generated from method `MR::FlowAggregator::computeFlow`.
        /// Parameter `out_` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.VertScalars> ComputeFlow(ulong numStarts, MR.Std.Const_Function_MRMeshTriPointFuncFromMRUint64T startById, MR.Std.Const_Function_FloatFuncFromMRUint64T amountById, MR.Std.Const_Function_ConstMRFaceBitSetPtrFuncFromMRUint64T regionById, MR.Const_OutputFlows? out_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeFlow_5", ExactSpelling = true)]
            extern static MR.VertScalars._Underlying *__MR_FlowAggregator_computeFlow_5(_Underlying *_this, ulong numStarts, MR.Std.Const_Function_MRMeshTriPointFuncFromMRUint64T._Underlying *startById, MR.Std.Const_Function_FloatFuncFromMRUint64T._Underlying *amountById, MR.Std.Const_Function_ConstMRFaceBitSetPtrFuncFromMRUint64T._Underlying *regionById, MR.Const_OutputFlows._Underlying *out_);
            return MR.Misc.Move(new MR.VertScalars(__MR_FlowAggregator_computeFlow_5(_UnderlyingPtr, numStarts, startById._UnderlyingPtr, amountById._UnderlyingPtr, regionById._UnderlyingPtr, out_ is not null ? out_._UnderlyingPtr : null), is_owning: true));
        }

        /// tracks multiple flows
        /// \param starts the origin of each flow (should be uniformly sampled over the terrain)
        /// \return the flows grouped by the final destination vertex
        /// Generated from method `MR::FlowAggregator::computeFlowsPerBasin`.
        public unsafe MR.Misc._Moved<MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows> ComputeFlowsPerBasin(MR.Std.Const_Vector_MRFlowOrigin starts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeFlowsPerBasin_1_std_vector_MR_FlowOrigin", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows._Underlying *__MR_FlowAggregator_computeFlowsPerBasin_1_std_vector_MR_FlowOrigin(_Underlying *_this, MR.Std.Const_Vector_MRFlowOrigin._Underlying *starts);
            return MR.Misc.Move(new MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows(__MR_FlowAggregator_computeFlowsPerBasin_1_std_vector_MR_FlowOrigin(_UnderlyingPtr, starts._UnderlyingPtr), is_owning: true));
        }

        // same with all amounts equal to 1
        /// Generated from method `MR::FlowAggregator::computeFlowsPerBasin`.
        public unsafe MR.Misc._Moved<MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows> ComputeFlowsPerBasin(MR.Std.Const_Vector_MRMeshTriPoint starts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeFlowsPerBasin_1_std_vector_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows._Underlying *__MR_FlowAggregator_computeFlowsPerBasin_1_std_vector_MR_MeshTriPoint(_Underlying *_this, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *starts);
            return MR.Misc.Move(new MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows(__MR_FlowAggregator_computeFlowsPerBasin_1_std_vector_MR_MeshTriPoint(_UnderlyingPtr, starts._UnderlyingPtr), is_owning: true));
        }

        // general version that supplies starts in a functional way
        /// Generated from method `MR::FlowAggregator::computeFlowsPerBasin`.
        public unsafe MR.Misc._Moved<MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows> ComputeFlowsPerBasin(ulong numStarts, MR.Std.Const_Function_MRMeshTriPointFuncFromMRUint64T startById, MR.Std.Const_Function_FloatFuncFromMRUint64T amountById)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeFlowsPerBasin_3", ExactSpelling = true)]
            extern static MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows._Underlying *__MR_FlowAggregator_computeFlowsPerBasin_3(_Underlying *_this, ulong numStarts, MR.Std.Const_Function_MRMeshTriPointFuncFromMRUint64T._Underlying *startById, MR.Std.Const_Function_FloatFuncFromMRUint64T._Underlying *amountById);
            return MR.Misc.Move(new MR.Phmap.FlatHashMap_MRVertId_MRFlowAggregatorFlows(__MR_FlowAggregator_computeFlowsPerBasin_3(_UnderlyingPtr, numStarts, startById._UnderlyingPtr, amountById._UnderlyingPtr), is_owning: true));
        }

        /// finds the edges on the mesh that divides catchment basin
        /// (every triangle is attributed to the final destination point based on the path originated from its centroid)
        /// Generated from method `MR::FlowAggregator::computeCatchmentDelineation`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> ComputeCatchmentDelineation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_computeCatchmentDelineation", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_FlowAggregator_computeCatchmentDelineation(_Underlying *_this);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_FlowAggregator_computeCatchmentDelineation(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from class `MR::FlowAggregator::Flows`.
        /// This is the const half of the class.
        public class Const_Flows : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Flows(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_Destroy", ExactSpelling = true)]
                extern static void __MR_FlowAggregator_Flows_Destroy(_Underlying *_this);
                __MR_FlowAggregator_Flows_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Flows() {Dispose(false);}

            public unsafe MR.Const_Polyline3 Polyline
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_Get_polyline", ExactSpelling = true)]
                    extern static MR.Const_Polyline3._Underlying *__MR_FlowAggregator_Flows_Get_polyline(_Underlying *_this);
                    return new(__MR_FlowAggregator_Flows_Get_polyline(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_UndirectedEdgeScalars FlowPerEdge
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_Get_flowPerEdge", ExactSpelling = true)]
                    extern static MR.Const_UndirectedEdgeScalars._Underlying *__MR_FlowAggregator_Flows_Get_flowPerEdge(_Underlying *_this);
                    return new(__MR_FlowAggregator_Flows_Get_flowPerEdge(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Flows() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_DefaultConstruct();
                _UnderlyingPtr = __MR_FlowAggregator_Flows_DefaultConstruct();
            }

            /// Constructs `MR::FlowAggregator::Flows` elementwise.
            public unsafe Const_Flows(MR._ByValue_Polyline3 polyline, MR._ByValue_UndirectedEdgeScalars flowPerEdge) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_ConstructFrom", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_ConstructFrom(MR.Misc._PassBy polyline_pass_by, MR.Polyline3._Underlying *polyline, MR.Misc._PassBy flowPerEdge_pass_by, MR.UndirectedEdgeScalars._Underlying *flowPerEdge);
                _UnderlyingPtr = __MR_FlowAggregator_Flows_ConstructFrom(polyline.PassByMode, polyline.Value is not null ? polyline.Value._UnderlyingPtr : null, flowPerEdge.PassByMode, flowPerEdge.Value is not null ? flowPerEdge.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::FlowAggregator::Flows::Flows`.
            public unsafe Const_Flows(MR.FlowAggregator._ByValue_Flows _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FlowAggregator.Flows._Underlying *_other);
                _UnderlyingPtr = __MR_FlowAggregator_Flows_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::FlowAggregator::Flows`.
        /// This is the non-const half of the class.
        public class Flows : Const_Flows
        {
            internal unsafe Flows(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Polyline3 Polyline
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_GetMutable_polyline", ExactSpelling = true)]
                    extern static MR.Polyline3._Underlying *__MR_FlowAggregator_Flows_GetMutable_polyline(_Underlying *_this);
                    return new(__MR_FlowAggregator_Flows_GetMutable_polyline(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.UndirectedEdgeScalars FlowPerEdge
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_GetMutable_flowPerEdge", ExactSpelling = true)]
                    extern static MR.UndirectedEdgeScalars._Underlying *__MR_FlowAggregator_Flows_GetMutable_flowPerEdge(_Underlying *_this);
                    return new(__MR_FlowAggregator_Flows_GetMutable_flowPerEdge(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Flows() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_DefaultConstruct", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_DefaultConstruct();
                _UnderlyingPtr = __MR_FlowAggregator_Flows_DefaultConstruct();
            }

            /// Constructs `MR::FlowAggregator::Flows` elementwise.
            public unsafe Flows(MR._ByValue_Polyline3 polyline, MR._ByValue_UndirectedEdgeScalars flowPerEdge) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_ConstructFrom", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_ConstructFrom(MR.Misc._PassBy polyline_pass_by, MR.Polyline3._Underlying *polyline, MR.Misc._PassBy flowPerEdge_pass_by, MR.UndirectedEdgeScalars._Underlying *flowPerEdge);
                _UnderlyingPtr = __MR_FlowAggregator_Flows_ConstructFrom(polyline.PassByMode, polyline.Value is not null ? polyline.Value._UnderlyingPtr : null, flowPerEdge.PassByMode, flowPerEdge.Value is not null ? flowPerEdge.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::FlowAggregator::Flows::Flows`.
            public unsafe Flows(MR.FlowAggregator._ByValue_Flows _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FlowAggregator.Flows._Underlying *_other);
                _UnderlyingPtr = __MR_FlowAggregator_Flows_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::FlowAggregator::Flows::operator=`.
            public unsafe MR.FlowAggregator.Flows Assign(MR.FlowAggregator._ByValue_Flows _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Flows_AssignFromAnother", ExactSpelling = true)]
                extern static MR.FlowAggregator.Flows._Underlying *__MR_FlowAggregator_Flows_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FlowAggregator.Flows._Underlying *_other);
                return new(__MR_FlowAggregator_Flows_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Flows` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Flows`/`Const_Flows` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Flows
        {
            internal readonly Const_Flows? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Flows() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Flows(Const_Flows new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Flows(Const_Flows arg) {return new(arg);}
            public _ByValue_Flows(MR.Misc._Moved<Flows> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Flows(MR.Misc._Moved<Flows> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Flows` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Flows`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Flows`/`Const_Flows` directly.
        public class _InOptMut_Flows
        {
            public Flows? Opt;

            public _InOptMut_Flows() {}
            public _InOptMut_Flows(Flows value) {Opt = value;}
            public static implicit operator _InOptMut_Flows(Flows value) {return new(value);}
        }

        /// This is used for optional parameters of class `Flows` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Flows`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Flows`/`Const_Flows` to pass it to the function.
        public class _InOptConst_Flows
        {
            public Const_Flows? Opt;

            public _InOptConst_Flows() {}
            public _InOptConst_Flows(Const_Flows value) {Opt = value;}
            public static implicit operator _InOptConst_Flows(Const_Flows value) {return new(value);}
        }
    }

    /// this class can track multiple flows and find in each mesh vertex the amount of water reached it
    /// Generated from class `MR::FlowAggregator`.
    /// This is the non-const half of the class.
    public class FlowAggregator : Const_FlowAggregator
    {
        internal unsafe FlowAggregator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::FlowAggregator::FlowAggregator`.
        public unsafe FlowAggregator(MR._ByValue_FlowAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FlowAggregator._Underlying *__MR_FlowAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FlowAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_FlowAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// prepares the processing of given mesh with given height in each vertex
        /// Generated from constructor `MR::FlowAggregator::FlowAggregator`.
        public unsafe FlowAggregator(MR.Const_Mesh mesh, MR.Const_VertScalars heights) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FlowAggregator_Construct", ExactSpelling = true)]
            extern static MR.FlowAggregator._Underlying *__MR_FlowAggregator_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_VertScalars._Underlying *heights);
            _UnderlyingPtr = __MR_FlowAggregator_Construct(mesh._UnderlyingPtr, heights._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FlowAggregator` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FlowAggregator`/`Const_FlowAggregator` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FlowAggregator
    {
        internal readonly Const_FlowAggregator? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FlowAggregator(Const_FlowAggregator new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FlowAggregator(Const_FlowAggregator arg) {return new(arg);}
        public _ByValue_FlowAggregator(MR.Misc._Moved<FlowAggregator> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FlowAggregator(MR.Misc._Moved<FlowAggregator> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FlowAggregator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FlowAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FlowAggregator`/`Const_FlowAggregator` directly.
    public class _InOptMut_FlowAggregator
    {
        public FlowAggregator? Opt;

        public _InOptMut_FlowAggregator() {}
        public _InOptMut_FlowAggregator(FlowAggregator value) {Opt = value;}
        public static implicit operator _InOptMut_FlowAggregator(FlowAggregator value) {return new(value);}
    }

    /// This is used for optional parameters of class `FlowAggregator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FlowAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FlowAggregator`/`Const_FlowAggregator` to pass it to the function.
    public class _InOptConst_FlowAggregator
    {
        public Const_FlowAggregator? Opt;

        public _InOptConst_FlowAggregator() {}
        public _InOptConst_FlowAggregator(Const_FlowAggregator value) {Opt = value;}
        public static implicit operator _InOptConst_FlowAggregator(Const_FlowAggregator value) {return new(value);}
    }
}
