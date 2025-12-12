public static partial class MR
{
    // stores basic params for trimWithPlane function
    /// Generated from class `MR::TrimWithPlaneParams`.
    /// This is the const half of the class.
    public class Const_TrimWithPlaneParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TrimWithPlaneParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_Destroy", ExactSpelling = true)]
            extern static void __MR_TrimWithPlaneParams_Destroy(_Underlying *_this);
            __MR_TrimWithPlaneParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TrimWithPlaneParams() {Dispose(false);}

        //Input plane to cut mesh with
        public unsafe MR.Const_Plane3f Plane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_Get_plane", ExactSpelling = true)]
                extern static MR.Const_Plane3f._Underlying *__MR_TrimWithPlaneParams_Get_plane(_Underlying *_this);
                return new(__MR_TrimWithPlaneParams_Get_plane(_UnderlyingPtr), is_owning: false);
            }
        }

        // if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
        public unsafe float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_Get_eps", ExactSpelling = true)]
                extern static float *__MR_TrimWithPlaneParams_Get_eps(_Underlying *_this);
                return *__MR_TrimWithPlaneParams_Get_eps(_UnderlyingPtr);
            }
        }

        // is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeIdFloat OnEdgeSplitCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_Get_onEdgeSplitCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *__MR_TrimWithPlaneParams_Get_onEdgeSplitCallback(_Underlying *_this);
                return new(__MR_TrimWithPlaneParams_Get_onEdgeSplitCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TrimWithPlaneParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_DefaultConstruct();
            _UnderlyingPtr = __MR_TrimWithPlaneParams_DefaultConstruct();
        }

        /// Constructs `MR::TrimWithPlaneParams` elementwise.
        public unsafe Const_TrimWithPlaneParams(MR.Const_Plane3f plane, float eps, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeIdFloat onEdgeSplitCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_ConstructFrom(MR.Plane3f._Underlying *plane, float eps, MR.Misc._PassBy onEdgeSplitCallback_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *onEdgeSplitCallback);
            _UnderlyingPtr = __MR_TrimWithPlaneParams_ConstructFrom(plane._UnderlyingPtr, eps, onEdgeSplitCallback.PassByMode, onEdgeSplitCallback.Value is not null ? onEdgeSplitCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TrimWithPlaneParams::TrimWithPlaneParams`.
        public unsafe Const_TrimWithPlaneParams(MR._ByValue_TrimWithPlaneParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TrimWithPlaneParams._Underlying *_other);
            _UnderlyingPtr = __MR_TrimWithPlaneParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    // stores basic params for trimWithPlane function
    /// Generated from class `MR::TrimWithPlaneParams`.
    /// This is the non-const half of the class.
    public class TrimWithPlaneParams : Const_TrimWithPlaneParams
    {
        internal unsafe TrimWithPlaneParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        //Input plane to cut mesh with
        public new unsafe MR.Plane3f Plane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_GetMutable_plane", ExactSpelling = true)]
                extern static MR.Plane3f._Underlying *__MR_TrimWithPlaneParams_GetMutable_plane(_Underlying *_this);
                return new(__MR_TrimWithPlaneParams_GetMutable_plane(_UnderlyingPtr), is_owning: false);
            }
        }

        // if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
        public new unsafe ref float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_GetMutable_eps", ExactSpelling = true)]
                extern static float *__MR_TrimWithPlaneParams_GetMutable_eps(_Underlying *_this);
                return ref *__MR_TrimWithPlaneParams_GetMutable_eps(_UnderlyingPtr);
            }
        }

        // is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat OnEdgeSplitCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_GetMutable_onEdgeSplitCallback", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *__MR_TrimWithPlaneParams_GetMutable_onEdgeSplitCallback(_Underlying *_this);
                return new(__MR_TrimWithPlaneParams_GetMutable_onEdgeSplitCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TrimWithPlaneParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_DefaultConstruct();
            _UnderlyingPtr = __MR_TrimWithPlaneParams_DefaultConstruct();
        }

        /// Constructs `MR::TrimWithPlaneParams` elementwise.
        public unsafe TrimWithPlaneParams(MR.Const_Plane3f plane, float eps, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeIdFloat onEdgeSplitCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_ConstructFrom(MR.Plane3f._Underlying *plane, float eps, MR.Misc._PassBy onEdgeSplitCallback_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *onEdgeSplitCallback);
            _UnderlyingPtr = __MR_TrimWithPlaneParams_ConstructFrom(plane._UnderlyingPtr, eps, onEdgeSplitCallback.PassByMode, onEdgeSplitCallback.Value is not null ? onEdgeSplitCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TrimWithPlaneParams::TrimWithPlaneParams`.
        public unsafe TrimWithPlaneParams(MR._ByValue_TrimWithPlaneParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TrimWithPlaneParams._Underlying *_other);
            _UnderlyingPtr = __MR_TrimWithPlaneParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::TrimWithPlaneParams::operator=`.
        public unsafe MR.TrimWithPlaneParams Assign(MR._ByValue_TrimWithPlaneParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimWithPlaneParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TrimWithPlaneParams._Underlying *__MR_TrimWithPlaneParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TrimWithPlaneParams._Underlying *_other);
            return new(__MR_TrimWithPlaneParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TrimWithPlaneParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TrimWithPlaneParams`/`Const_TrimWithPlaneParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TrimWithPlaneParams
    {
        internal readonly Const_TrimWithPlaneParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TrimWithPlaneParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TrimWithPlaneParams(Const_TrimWithPlaneParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TrimWithPlaneParams(Const_TrimWithPlaneParams arg) {return new(arg);}
        public _ByValue_TrimWithPlaneParams(MR.Misc._Moved<TrimWithPlaneParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TrimWithPlaneParams(MR.Misc._Moved<TrimWithPlaneParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TrimWithPlaneParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TrimWithPlaneParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TrimWithPlaneParams`/`Const_TrimWithPlaneParams` directly.
    public class _InOptMut_TrimWithPlaneParams
    {
        public TrimWithPlaneParams? Opt;

        public _InOptMut_TrimWithPlaneParams() {}
        public _InOptMut_TrimWithPlaneParams(TrimWithPlaneParams value) {Opt = value;}
        public static implicit operator _InOptMut_TrimWithPlaneParams(TrimWithPlaneParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `TrimWithPlaneParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TrimWithPlaneParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TrimWithPlaneParams`/`Const_TrimWithPlaneParams` to pass it to the function.
    public class _InOptConst_TrimWithPlaneParams
    {
        public Const_TrimWithPlaneParams? Opt;

        public _InOptConst_TrimWithPlaneParams() {}
        public _InOptConst_TrimWithPlaneParams(Const_TrimWithPlaneParams value) {Opt = value;}
        public static implicit operator _InOptConst_TrimWithPlaneParams(Const_TrimWithPlaneParams value) {return new(value);}
    }

    // stores optional output params for trimWithPlane function
    /// Generated from class `MR::TrimOptionalOutput`.
    /// This is the const half of the class.
    public class Const_TrimOptionalOutput : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TrimOptionalOutput(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Destroy", ExactSpelling = true)]
            extern static void __MR_TrimOptionalOutput_Destroy(_Underlying *_this);
            __MR_TrimOptionalOutput_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TrimOptionalOutput() {Dispose(false);}

        // newly appeared hole boundary edges
        public unsafe ref void * OutCutEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Get_outCutEdges", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_Get_outCutEdges(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_Get_outCutEdges(_UnderlyingPtr);
            }
        }

        // newly appeared hole contours where each edge does not have right face
        public unsafe ref void * OutCutContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Get_outCutContours", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_Get_outCutContours(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_Get_outCutContours(_UnderlyingPtr);
            }
        }

        // mapping from newly appeared triangle to its original triangle (part to full)
        public unsafe ref void * New2Old
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Get_new2Old", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_Get_new2Old(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_Get_new2Old(_UnderlyingPtr);
            }
        }

        // left part of the trimmed mesh
        public unsafe ref void * OtherPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Get_otherPart", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_Get_otherPart(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_Get_otherPart(_UnderlyingPtr);
            }
        }

        // mapping from newly appeared triangle to its original triangle (part to full) in otherPart
        public unsafe ref void * OtherNew2Old
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Get_otherNew2Old", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_Get_otherNew2Old(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_Get_otherNew2Old(_UnderlyingPtr);
            }
        }

        // newly appeared hole contours where each edge does not have right face in otherPart
        public unsafe ref void * OtherOutCutContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_Get_otherOutCutContours", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_Get_otherOutCutContours(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_Get_otherOutCutContours(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TrimOptionalOutput() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_DefaultConstruct();
            _UnderlyingPtr = __MR_TrimOptionalOutput_DefaultConstruct();
        }

        /// Constructs `MR::TrimOptionalOutput` elementwise.
        public unsafe Const_TrimOptionalOutput(MR.UndirectedEdgeBitSet? outCutEdges, MR.Std.Vector_StdVectorMREdgeId? outCutContours, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old, MR.Mesh? otherPart, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? otherNew2Old, MR.Std.Vector_StdVectorMREdgeId? otherOutCutContours) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_ConstructFrom", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_ConstructFrom(MR.UndirectedEdgeBitSet._Underlying *outCutEdges, MR.Std.Vector_StdVectorMREdgeId._Underlying *outCutContours, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old, MR.Mesh._Underlying *otherPart, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *otherNew2Old, MR.Std.Vector_StdVectorMREdgeId._Underlying *otherOutCutContours);
            _UnderlyingPtr = __MR_TrimOptionalOutput_ConstructFrom(outCutEdges is not null ? outCutEdges._UnderlyingPtr : null, outCutContours is not null ? outCutContours._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null, otherPart is not null ? otherPart._UnderlyingPtr : null, otherNew2Old is not null ? otherNew2Old._UnderlyingPtr : null, otherOutCutContours is not null ? otherOutCutContours._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TrimOptionalOutput::TrimOptionalOutput`.
        public unsafe Const_TrimOptionalOutput(MR.Const_TrimOptionalOutput _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_ConstructFromAnother(MR.TrimOptionalOutput._Underlying *_other);
            _UnderlyingPtr = __MR_TrimOptionalOutput_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // stores optional output params for trimWithPlane function
    /// Generated from class `MR::TrimOptionalOutput`.
    /// This is the non-const half of the class.
    public class TrimOptionalOutput : Const_TrimOptionalOutput
    {
        internal unsafe TrimOptionalOutput(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // newly appeared hole boundary edges
        public new unsafe ref void * OutCutEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_GetMutable_outCutEdges", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_GetMutable_outCutEdges(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_GetMutable_outCutEdges(_UnderlyingPtr);
            }
        }

        // newly appeared hole contours where each edge does not have right face
        public new unsafe ref void * OutCutContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_GetMutable_outCutContours", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_GetMutable_outCutContours(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_GetMutable_outCutContours(_UnderlyingPtr);
            }
        }

        // mapping from newly appeared triangle to its original triangle (part to full)
        public new unsafe ref void * New2Old
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_GetMutable_new2Old", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_GetMutable_new2Old(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_GetMutable_new2Old(_UnderlyingPtr);
            }
        }

        // left part of the trimmed mesh
        public new unsafe ref void * OtherPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_GetMutable_otherPart", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_GetMutable_otherPart(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_GetMutable_otherPart(_UnderlyingPtr);
            }
        }

        // mapping from newly appeared triangle to its original triangle (part to full) in otherPart
        public new unsafe ref void * OtherNew2Old
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_GetMutable_otherNew2Old", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_GetMutable_otherNew2Old(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_GetMutable_otherNew2Old(_UnderlyingPtr);
            }
        }

        // newly appeared hole contours where each edge does not have right face in otherPart
        public new unsafe ref void * OtherOutCutContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_GetMutable_otherOutCutContours", ExactSpelling = true)]
                extern static void **__MR_TrimOptionalOutput_GetMutable_otherOutCutContours(_Underlying *_this);
                return ref *__MR_TrimOptionalOutput_GetMutable_otherOutCutContours(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TrimOptionalOutput() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_DefaultConstruct();
            _UnderlyingPtr = __MR_TrimOptionalOutput_DefaultConstruct();
        }

        /// Constructs `MR::TrimOptionalOutput` elementwise.
        public unsafe TrimOptionalOutput(MR.UndirectedEdgeBitSet? outCutEdges, MR.Std.Vector_StdVectorMREdgeId? outCutContours, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old, MR.Mesh? otherPart, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? otherNew2Old, MR.Std.Vector_StdVectorMREdgeId? otherOutCutContours) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_ConstructFrom", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_ConstructFrom(MR.UndirectedEdgeBitSet._Underlying *outCutEdges, MR.Std.Vector_StdVectorMREdgeId._Underlying *outCutContours, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old, MR.Mesh._Underlying *otherPart, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *otherNew2Old, MR.Std.Vector_StdVectorMREdgeId._Underlying *otherOutCutContours);
            _UnderlyingPtr = __MR_TrimOptionalOutput_ConstructFrom(outCutEdges is not null ? outCutEdges._UnderlyingPtr : null, outCutContours is not null ? outCutContours._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null, otherPart is not null ? otherPart._UnderlyingPtr : null, otherNew2Old is not null ? otherNew2Old._UnderlyingPtr : null, otherOutCutContours is not null ? otherOutCutContours._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::TrimOptionalOutput::TrimOptionalOutput`.
        public unsafe TrimOptionalOutput(MR.Const_TrimOptionalOutput _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_ConstructFromAnother(MR.TrimOptionalOutput._Underlying *_other);
            _UnderlyingPtr = __MR_TrimOptionalOutput_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TrimOptionalOutput::operator=`.
        public unsafe MR.TrimOptionalOutput Assign(MR.Const_TrimOptionalOutput _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrimOptionalOutput_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TrimOptionalOutput._Underlying *__MR_TrimOptionalOutput_AssignFromAnother(_Underlying *_this, MR.TrimOptionalOutput._Underlying *_other);
            return new(__MR_TrimOptionalOutput_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TrimOptionalOutput` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TrimOptionalOutput`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TrimOptionalOutput`/`Const_TrimOptionalOutput` directly.
    public class _InOptMut_TrimOptionalOutput
    {
        public TrimOptionalOutput? Opt;

        public _InOptMut_TrimOptionalOutput() {}
        public _InOptMut_TrimOptionalOutput(TrimOptionalOutput value) {Opt = value;}
        public static implicit operator _InOptMut_TrimOptionalOutput(TrimOptionalOutput value) {return new(value);}
    }

    /// This is used for optional parameters of class `TrimOptionalOutput` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TrimOptionalOutput`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TrimOptionalOutput`/`Const_TrimOptionalOutput` to pass it to the function.
    public class _InOptConst_TrimOptionalOutput
    {
        public Const_TrimOptionalOutput? Opt;

        public _InOptConst_TrimOptionalOutput() {}
        public _InOptConst_TrimOptionalOutput(Const_TrimOptionalOutput value) {Opt = value;}
        public static implicit operator _InOptConst_TrimOptionalOutput(Const_TrimOptionalOutput value) {return new(value);}
    }

    /// subdivides all triangles intersected by given plane, leaving smaller triangles that only touch the plane;
    /// \return all triangles on the positive side of the plane
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    /// \param eps if existing vertex is within eps distance from the plane, then move the vertex not introducing new ones
    /// \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
    /// Generated from function `MR::subdivideWithPlane`.
    /// Parameter `eps` defaults to `0`.
    /// Parameter `onEdgeSplitCallback` defaults to `nullptr`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> SubdivideWithPlane(MR.Mesh mesh, MR.Const_Plane3f plane, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null, float? eps = null, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeIdFloat? onEdgeSplitCallback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdivideWithPlane_5", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_subdivideWithPlane_5(MR.Mesh._Underlying *mesh, MR.Const_Plane3f._Underlying *plane, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old, float *eps, MR.Misc._PassBy onEdgeSplitCallback_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *onEdgeSplitCallback);
        float __deref_eps = eps.GetValueOrDefault();
        return MR.Misc.Move(new MR.FaceBitSet(__MR_subdivideWithPlane_5(mesh._UnderlyingPtr, plane._UnderlyingPtr, new2Old is not null ? new2Old._UnderlyingPtr : null, eps.HasValue ? &__deref_eps : null, onEdgeSplitCallback is not null ? onEdgeSplitCallback.PassByMode : MR.Misc._PassBy.default_arg, onEdgeSplitCallback is not null && onEdgeSplitCallback.Value is not null ? onEdgeSplitCallback.Value._UnderlyingPtr : null), is_owning: true));
    }

    /** \brief trim mesh by plane
    *
    * This function cuts mesh with plane, leaving only part of mesh that lay in positive direction of normal
    * \param mesh Input mesh that will be cut
    * \param params stores basic params for trimWithPlane function
    * \param optOut stores optional output params for trimWithPlane function
    */
    /// Generated from function `MR::trimWithPlane`.
    /// Parameter `optOut` defaults to `{}`.
    public static unsafe void TrimWithPlane(MR.Mesh mesh, MR.Const_TrimWithPlaneParams params_, MR.Const_TrimOptionalOutput? optOut = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trimWithPlane_MR_Mesh", ExactSpelling = true)]
        extern static void __MR_trimWithPlane_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_TrimWithPlaneParams._Underlying *params_, MR.Const_TrimOptionalOutput._Underlying *optOut);
        __MR_trimWithPlane_MR_Mesh(mesh._UnderlyingPtr, params_._UnderlyingPtr, optOut is not null ? optOut._UnderlyingPtr : null);
    }
}
