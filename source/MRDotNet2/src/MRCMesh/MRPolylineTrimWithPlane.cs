public static partial class MR
{
    /// Generated from class `MR::DividePolylineParameters`.
    /// This is the const half of the class.
    public class Const_DividePolylineParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DividePolylineParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_DividePolylineParameters_Destroy(_Underlying *_this);
            __MR_DividePolylineParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DividePolylineParameters() {Dispose(false);}

        /// onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeIdFloat OnEdgeSplitCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_onEdgeSplitCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *__MR_DividePolylineParameters_Get_onEdgeSplitCallback(_Underlying *_this);
                return new(__MR_DividePolylineParameters_Get_onEdgeSplitCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closeLineAfterCut if true, the ends of resulting polyline will be connected by new edges (can make a polyline closed, even if the original one was open)
        /// if close, only cut edges (no new edges will be created)
        public unsafe bool CloseLineAfterCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_closeLineAfterCut", ExactSpelling = true)]
                extern static bool *__MR_DividePolylineParameters_Get_closeLineAfterCut(_Underlying *_this);
                return *__MR_DividePolylineParameters_Get_closeLineAfterCut(_UnderlyingPtr);
            }
        }

        /// map from input polyline verts to output
        public unsafe ref void * OutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_outVmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_Get_outVmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_Get_outVmap(_UnderlyingPtr);
            }
        }

        /// map from input polyline edges to output
        public unsafe ref void * OutEmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_outEmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_Get_outEmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_Get_outEmap(_UnderlyingPtr);
            }
        }

        /// otherPart Optional return, polyline composed from edges on the negative side of the plane
        public unsafe ref void * OtherPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_otherPart", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_Get_otherPart(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_Get_otherPart(_UnderlyingPtr);
            }
        }

        ///  map from input polyline verts to other output
        public unsafe ref void * OtherOutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_otherOutVmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_Get_otherOutVmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_Get_otherOutVmap(_UnderlyingPtr);
            }
        }

        /// map from input polyline edges to other output
        public unsafe ref void * OtherOutEmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_Get_otherOutEmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_Get_otherOutEmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_Get_otherOutEmap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DividePolylineParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_DividePolylineParameters_DefaultConstruct();
        }

        /// Constructs `MR::DividePolylineParameters` elementwise.
        public unsafe Const_DividePolylineParameters(MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeIdFloat onEdgeSplitCallback, bool closeLineAfterCut, MR.VertMap? outVmap, MR.EdgeMap? outEmap, MR.Polyline3? otherPart, MR.VertMap? otherOutVmap, MR.EdgeMap? otherOutEmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_ConstructFrom(MR.Misc._PassBy onEdgeSplitCallback_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *onEdgeSplitCallback, byte closeLineAfterCut, MR.VertMap._Underlying *outVmap, MR.EdgeMap._Underlying *outEmap, MR.Polyline3._Underlying *otherPart, MR.VertMap._Underlying *otherOutVmap, MR.EdgeMap._Underlying *otherOutEmap);
            _UnderlyingPtr = __MR_DividePolylineParameters_ConstructFrom(onEdgeSplitCallback.PassByMode, onEdgeSplitCallback.Value is not null ? onEdgeSplitCallback.Value._UnderlyingPtr : null, closeLineAfterCut ? (byte)1 : (byte)0, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null, otherPart is not null ? otherPart._UnderlyingPtr : null, otherOutVmap is not null ? otherOutVmap._UnderlyingPtr : null, otherOutEmap is not null ? otherOutEmap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DividePolylineParameters::DividePolylineParameters`.
        public unsafe Const_DividePolylineParameters(MR._ByValue_DividePolylineParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DividePolylineParameters._Underlying *_other);
            _UnderlyingPtr = __MR_DividePolylineParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DividePolylineParameters`.
    /// This is the non-const half of the class.
    public class DividePolylineParameters : Const_DividePolylineParameters
    {
        internal unsafe DividePolylineParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat OnEdgeSplitCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_onEdgeSplitCallback", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *__MR_DividePolylineParameters_GetMutable_onEdgeSplitCallback(_Underlying *_this);
                return new(__MR_DividePolylineParameters_GetMutable_onEdgeSplitCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// closeLineAfterCut if true, the ends of resulting polyline will be connected by new edges (can make a polyline closed, even if the original one was open)
        /// if close, only cut edges (no new edges will be created)
        public new unsafe ref bool CloseLineAfterCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_closeLineAfterCut", ExactSpelling = true)]
                extern static bool *__MR_DividePolylineParameters_GetMutable_closeLineAfterCut(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_GetMutable_closeLineAfterCut(_UnderlyingPtr);
            }
        }

        /// map from input polyline verts to output
        public new unsafe ref void * OutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_outVmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_GetMutable_outVmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_GetMutable_outVmap(_UnderlyingPtr);
            }
        }

        /// map from input polyline edges to output
        public new unsafe ref void * OutEmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_outEmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_GetMutable_outEmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_GetMutable_outEmap(_UnderlyingPtr);
            }
        }

        /// otherPart Optional return, polyline composed from edges on the negative side of the plane
        public new unsafe ref void * OtherPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_otherPart", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_GetMutable_otherPart(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_GetMutable_otherPart(_UnderlyingPtr);
            }
        }

        ///  map from input polyline verts to other output
        public new unsafe ref void * OtherOutVmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_otherOutVmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_GetMutable_otherOutVmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_GetMutable_otherOutVmap(_UnderlyingPtr);
            }
        }

        /// map from input polyline edges to other output
        public new unsafe ref void * OtherOutEmap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_GetMutable_otherOutEmap", ExactSpelling = true)]
                extern static void **__MR_DividePolylineParameters_GetMutable_otherOutEmap(_Underlying *_this);
                return ref *__MR_DividePolylineParameters_GetMutable_otherOutEmap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DividePolylineParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_DividePolylineParameters_DefaultConstruct();
        }

        /// Constructs `MR::DividePolylineParameters` elementwise.
        public unsafe DividePolylineParameters(MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeIdFloat onEdgeSplitCallback, bool closeLineAfterCut, MR.VertMap? outVmap, MR.EdgeMap? outEmap, MR.Polyline3? otherPart, MR.VertMap? otherOutVmap, MR.EdgeMap? otherOutEmap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_ConstructFrom(MR.Misc._PassBy onEdgeSplitCallback_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *onEdgeSplitCallback, byte closeLineAfterCut, MR.VertMap._Underlying *outVmap, MR.EdgeMap._Underlying *outEmap, MR.Polyline3._Underlying *otherPart, MR.VertMap._Underlying *otherOutVmap, MR.EdgeMap._Underlying *otherOutEmap);
            _UnderlyingPtr = __MR_DividePolylineParameters_ConstructFrom(onEdgeSplitCallback.PassByMode, onEdgeSplitCallback.Value is not null ? onEdgeSplitCallback.Value._UnderlyingPtr : null, closeLineAfterCut ? (byte)1 : (byte)0, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null, otherPart is not null ? otherPart._UnderlyingPtr : null, otherOutVmap is not null ? otherOutVmap._UnderlyingPtr : null, otherOutEmap is not null ? otherOutEmap._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DividePolylineParameters::DividePolylineParameters`.
        public unsafe DividePolylineParameters(MR._ByValue_DividePolylineParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DividePolylineParameters._Underlying *_other);
            _UnderlyingPtr = __MR_DividePolylineParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DividePolylineParameters::operator=`.
        public unsafe MR.DividePolylineParameters Assign(MR._ByValue_DividePolylineParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DividePolylineParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DividePolylineParameters._Underlying *__MR_DividePolylineParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DividePolylineParameters._Underlying *_other);
            return new(__MR_DividePolylineParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DividePolylineParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DividePolylineParameters`/`Const_DividePolylineParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DividePolylineParameters
    {
        internal readonly Const_DividePolylineParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DividePolylineParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DividePolylineParameters(Const_DividePolylineParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DividePolylineParameters(Const_DividePolylineParameters arg) {return new(arg);}
        public _ByValue_DividePolylineParameters(MR.Misc._Moved<DividePolylineParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DividePolylineParameters(MR.Misc._Moved<DividePolylineParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DividePolylineParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DividePolylineParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DividePolylineParameters`/`Const_DividePolylineParameters` directly.
    public class _InOptMut_DividePolylineParameters
    {
        public DividePolylineParameters? Opt;

        public _InOptMut_DividePolylineParameters() {}
        public _InOptMut_DividePolylineParameters(DividePolylineParameters value) {Opt = value;}
        public static implicit operator _InOptMut_DividePolylineParameters(DividePolylineParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `DividePolylineParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DividePolylineParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DividePolylineParameters`/`Const_DividePolylineParameters` to pass it to the function.
    public class _InOptConst_DividePolylineParameters
    {
        public Const_DividePolylineParameters? Opt;

        public _InOptConst_DividePolylineParameters() {}
        public _InOptConst_DividePolylineParameters(Const_DividePolylineParameters value) {Opt = value;}
        public static implicit operator _InOptConst_DividePolylineParameters(Const_DividePolylineParameters value) {return new(value);}
    }

    /// This function splits edges intersected by the plane
    /// \return edges located above the plane (in direction of normal to plane)
    /// \param polyline Input polyline that will be cut by the plane
    /// \param plane Input plane to cut polyline with
    /// \param newPositiveEdges edges with origin on the plane and oriented to the positive direction (only adds bits to the existing ones)
    /// \param onEdgeSplitCallback is invoked each time when an edge is split. Receives edge ID before split, edge ID after split, and weight of the origin vertex
    /// Generated from function `MR::subdivideWithPlane`.
    /// Parameter `onEdgeSplitCallback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> SubdivideWithPlane(MR.Polyline3 polyline, MR.Const_Plane3f plane, MR.EdgeBitSet? newPositiveEdges = null, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeIdFloat? onEdgeSplitCallback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdivideWithPlane_4", ExactSpelling = true)]
        extern static MR.UndirectedEdgeBitSet._Underlying *__MR_subdivideWithPlane_4(MR.Polyline3._Underlying *polyline, MR.Const_Plane3f._Underlying *plane, MR.EdgeBitSet._Underlying *newPositiveEdges, MR.Misc._PassBy onEdgeSplitCallback_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeIdFloat._Underlying *onEdgeSplitCallback);
        return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_subdivideWithPlane_4(polyline._UnderlyingPtr, plane._UnderlyingPtr, newPositiveEdges is not null ? newPositiveEdges._UnderlyingPtr : null, onEdgeSplitCallback is not null ? onEdgeSplitCallback.PassByMode : MR.Misc._PassBy.default_arg, onEdgeSplitCallback is not null && onEdgeSplitCallback.Value is not null ? onEdgeSplitCallback.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// This function divides polyline with a plane, leaving only part of polyline that lies in positive direction of normal
    /// \param polyline Input polyline that will be cut by the plane
    /// \param plane Input plane to cut polyline with
    /// \param params Parameters of the function, containing optional output
    /// Generated from function `MR::trimWithPlane`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe void TrimWithPlane(MR.Polyline3 polyline, MR.Const_Plane3f plane, MR.Const_DividePolylineParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_trimWithPlane_MR_Polyline3", ExactSpelling = true)]
        extern static void __MR_trimWithPlane_MR_Polyline3(MR.Polyline3._Underlying *polyline, MR.Const_Plane3f._Underlying *plane, MR.Const_DividePolylineParameters._Underlying *params_);
        __MR_trimWithPlane_MR_Polyline3(polyline._UnderlyingPtr, plane._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null);
    }

    /// This function cuts polyline with a plane
    /// \details plane cuts an edge if one end of the edge is below the plane and the other is not
    /// \return Edge segments that are closer to the plane than \param eps. Segments are oriented according by plane normal ( segment.a <= segment.b)
    /// \param polyline Input polyline that will be cut by the plane
    /// \param plane Input plane to cut polyline with
    /// \param eps Maximal distance from the plane
    /// \param positiveEdges Edges in a positive half-space relative to the plane or on the plane itself (only adds bits to the existing ones)
    /// Generated from function `MR::extractSectionsFromPolyline`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeSegment> ExtractSectionsFromPolyline(MR.Const_Polyline3 polyline, MR.Const_Plane3f plane, float eps, MR.UndirectedEdgeBitSet? positiveEdges = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractSectionsFromPolyline", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeSegment._Underlying *__MR_extractSectionsFromPolyline(MR.Const_Polyline3._Underlying *polyline, MR.Const_Plane3f._Underlying *plane, float eps, MR.UndirectedEdgeBitSet._Underlying *positiveEdges);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeSegment(__MR_extractSectionsFromPolyline(polyline._UnderlyingPtr, plane._UnderlyingPtr, eps, positiveEdges is not null ? positiveEdges._UnderlyingPtr : null), is_owning: true));
    }
}
