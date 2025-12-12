public static partial class MR
{
    /** \struct MR::FillHoleMetric
    * \brief Holds metrics for fillHole and buildCylinderBetweenTwoHoles triangulation\n
    * 
    * This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions\n
    * 
    * \sa \ref getCircumscribedMetric
    * \sa \ref getPlaneFillMetric
    * \sa \ref getEdgeLengthFillMetric
    * \sa \ref getEdgeLengthStitchMetric
    * \sa \ref getComplexStitchMetric
    * \sa \ref fillHole
    * \sa \ref buildCylinderBetweenTwoHoles
    */
    /// Generated from class `MR::FillHoleMetric`.
    /// This is the const half of the class.
    public class Const_FillHoleMetric : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FillHoleMetric(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_Destroy", ExactSpelling = true)]
            extern static void __MR_FillHoleMetric_Destroy(_Underlying *_this);
            __MR_FillHoleMetric_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FillHoleMetric() {Dispose(false);}

        /// is called for each triangle, if it is set
        public unsafe MR.Std.Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId TriangleMetric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_Get_triangleMetric", ExactSpelling = true)]
                extern static MR.Std.Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *__MR_FillHoleMetric_Get_triangleMetric(_Underlying *_this);
                return new(__MR_FillHoleMetric_Get_triangleMetric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// is called for each edge, if it is set
        public unsafe MR.Std.Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId EdgeMetric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_Get_edgeMetric", ExactSpelling = true)]
                extern static MR.Std.Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId._Underlying *__MR_FillHoleMetric_Get_edgeMetric(_Underlying *_this);
                return new(__MR_FillHoleMetric_Get_edgeMetric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// is called to combine metrics from different candidates, if it is not set it just summarizes input
        public unsafe MR.Std.Const_Function_DoubleFuncFromDoubleDouble CombineMetric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_Get_combineMetric", ExactSpelling = true)]
                extern static MR.Std.Const_Function_DoubleFuncFromDoubleDouble._Underlying *__MR_FillHoleMetric_Get_combineMetric(_Underlying *_this);
                return new(__MR_FillHoleMetric_Get_combineMetric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FillHoleMetric() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleMetric_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleMetric` elementwise.
        public unsafe Const_FillHoleMetric(MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId triangleMetric, MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId edgeMetric, MR.Std._ByValue_Function_DoubleFuncFromDoubleDouble combineMetric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_ConstructFrom(MR.Misc._PassBy triangleMetric_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *triangleMetric, MR.Misc._PassBy edgeMetric_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId._Underlying *edgeMetric, MR.Misc._PassBy combineMetric_pass_by, MR.Std.Function_DoubleFuncFromDoubleDouble._Underlying *combineMetric);
            _UnderlyingPtr = __MR_FillHoleMetric_ConstructFrom(triangleMetric.PassByMode, triangleMetric.Value is not null ? triangleMetric.Value._UnderlyingPtr : null, edgeMetric.PassByMode, edgeMetric.Value is not null ? edgeMetric.Value._UnderlyingPtr : null, combineMetric.PassByMode, combineMetric.Value is not null ? combineMetric.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FillHoleMetric::FillHoleMetric`.
        public unsafe Const_FillHoleMetric(MR._ByValue_FillHoleMetric _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FillHoleMetric._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleMetric_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /** \struct MR::FillHoleMetric
    * \brief Holds metrics for fillHole and buildCylinderBetweenTwoHoles triangulation\n
    * 
    * This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions\n
    * 
    * \sa \ref getCircumscribedMetric
    * \sa \ref getPlaneFillMetric
    * \sa \ref getEdgeLengthFillMetric
    * \sa \ref getEdgeLengthStitchMetric
    * \sa \ref getComplexStitchMetric
    * \sa \ref fillHole
    * \sa \ref buildCylinderBetweenTwoHoles
    */
    /// Generated from class `MR::FillHoleMetric`.
    /// This is the non-const half of the class.
    public class FillHoleMetric : Const_FillHoleMetric
    {
        internal unsafe FillHoleMetric(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// is called for each triangle, if it is set
        public new unsafe MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId TriangleMetric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_GetMutable_triangleMetric", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *__MR_FillHoleMetric_GetMutable_triangleMetric(_Underlying *_this);
                return new(__MR_FillHoleMetric_GetMutable_triangleMetric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// is called for each edge, if it is set
        public new unsafe MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId EdgeMetric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_GetMutable_edgeMetric", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId._Underlying *__MR_FillHoleMetric_GetMutable_edgeMetric(_Underlying *_this);
                return new(__MR_FillHoleMetric_GetMutable_edgeMetric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// is called to combine metrics from different candidates, if it is not set it just summarizes input
        public new unsafe MR.Std.Function_DoubleFuncFromDoubleDouble CombineMetric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_GetMutable_combineMetric", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromDoubleDouble._Underlying *__MR_FillHoleMetric_GetMutable_combineMetric(_Underlying *_this);
                return new(__MR_FillHoleMetric_GetMutable_combineMetric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FillHoleMetric() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleMetric_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleMetric` elementwise.
        public unsafe FillHoleMetric(MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId triangleMetric, MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId edgeMetric, MR.Std._ByValue_Function_DoubleFuncFromDoubleDouble combineMetric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_ConstructFrom(MR.Misc._PassBy triangleMetric_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *triangleMetric, MR.Misc._PassBy edgeMetric_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertIdMRVertId._Underlying *edgeMetric, MR.Misc._PassBy combineMetric_pass_by, MR.Std.Function_DoubleFuncFromDoubleDouble._Underlying *combineMetric);
            _UnderlyingPtr = __MR_FillHoleMetric_ConstructFrom(triangleMetric.PassByMode, triangleMetric.Value is not null ? triangleMetric.Value._UnderlyingPtr : null, edgeMetric.PassByMode, edgeMetric.Value is not null ? edgeMetric.Value._UnderlyingPtr : null, combineMetric.PassByMode, combineMetric.Value is not null ? combineMetric.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FillHoleMetric::FillHoleMetric`.
        public unsafe FillHoleMetric(MR._ByValue_FillHoleMetric _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FillHoleMetric._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleMetric_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FillHoleMetric::operator=`.
        public unsafe MR.FillHoleMetric Assign(MR._ByValue_FillHoleMetric _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleMetric_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleMetric._Underlying *__MR_FillHoleMetric_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FillHoleMetric._Underlying *_other);
            return new(__MR_FillHoleMetric_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FillHoleMetric` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FillHoleMetric`/`Const_FillHoleMetric` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FillHoleMetric
    {
        internal readonly Const_FillHoleMetric? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FillHoleMetric() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FillHoleMetric(Const_FillHoleMetric new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FillHoleMetric(Const_FillHoleMetric arg) {return new(arg);}
        public _ByValue_FillHoleMetric(MR.Misc._Moved<FillHoleMetric> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FillHoleMetric(MR.Misc._Moved<FillHoleMetric> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FillHoleMetric` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FillHoleMetric`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleMetric`/`Const_FillHoleMetric` directly.
    public class _InOptMut_FillHoleMetric
    {
        public FillHoleMetric? Opt;

        public _InOptMut_FillHoleMetric() {}
        public _InOptMut_FillHoleMetric(FillHoleMetric value) {Opt = value;}
        public static implicit operator _InOptMut_FillHoleMetric(FillHoleMetric value) {return new(value);}
    }

    /// This is used for optional parameters of class `FillHoleMetric` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FillHoleMetric`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleMetric`/`Const_FillHoleMetric` to pass it to the function.
    public class _InOptConst_FillHoleMetric
    {
        public Const_FillHoleMetric? Opt;

        public _InOptConst_FillHoleMetric() {}
        public _InOptConst_FillHoleMetric(Const_FillHoleMetric value) {Opt = value;}
        public static implicit operator _InOptConst_FillHoleMetric(Const_FillHoleMetric value) {return new(value);}
    }

    /// Computes combined metric after filling a hole
    /// Generated from function `MR::calcCombinedFillMetric`.
    public static unsafe double CalcCombinedFillMetric(MR.Const_Mesh mesh, MR.Const_FaceBitSet filledRegion, MR.Const_FillHoleMetric metric)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcCombinedFillMetric", ExactSpelling = true)]
        extern static double __MR_calcCombinedFillMetric(MR.Const_Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *filledRegion, MR.Const_FillHoleMetric._Underlying *metric);
        return __MR_calcCombinedFillMetric(mesh._UnderlyingPtr, filledRegion._UnderlyingPtr, metric._UnderlyingPtr);
    }

    /// This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.
    /// It is rather fast to calculate, and it results in typically good triangulations.
    /// Generated from function `MR::getCircumscribedMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetCircumscribedMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getCircumscribedMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getCircumscribedMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getCircumscribedMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Same as getCircumscribedFillMetric, but with extra penalty for the triangles having
    /// normals looking in the opposite side of plane containing left of (e).
    /// Generated from function `MR::getPlaneFillMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetPlaneFillMetric(MR.Const_Mesh mesh, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPlaneFillMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getPlaneFillMetric(MR.Const_Mesh._Underlying *mesh, MR.EdgeId e);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getPlaneFillMetric(mesh._UnderlyingPtr, e), is_owning: true));
    }

    /// Similar to getPlaneFillMetric with extra penalty for the triangles having
    /// normals looking in the opposite side of plane containing left of (e),
    /// but the metric minimizes the sum of circumcircle radius times aspect ratio for all triangles in the triangulation.
    /// Generated from function `MR::getPlaneNormalizedFillMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetPlaneNormalizedFillMetric(MR.Const_Mesh mesh, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPlaneNormalizedFillMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getPlaneNormalizedFillMetric(MR.Const_Mesh._Underlying *mesh, MR.EdgeId e);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getPlaneNormalizedFillMetric(mesh._UnderlyingPtr, e), is_owning: true));
    }

    /// This metric minimizes the sum of triangleMetric for all triangles in the triangulation
    /// plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n
    /// Where\n
    /// triangleMetric is proportional to triangle aspect ratio\n
    /// edgeMetric is proportional to ( 1 - dihedralAngleCos )
    /// Generated from function `MR::getComplexStitchMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetComplexStitchMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getComplexStitchMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getComplexStitchMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getComplexStitchMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Simple metric minimizing the sum of all edge lengths
    /// Generated from function `MR::getEdgeLengthFillMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetEdgeLengthFillMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getEdgeLengthFillMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getEdgeLengthFillMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getEdgeLengthFillMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Forbids connecting vertices from the same hole \n
    /// Simple metric minimizing edge length
    /// Generated from function `MR::getEdgeLengthStitchMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetEdgeLengthStitchMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getEdgeLengthStitchMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getEdgeLengthStitchMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getEdgeLengthStitchMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Forbids connecting vertices from the same hole \n
    /// penalize for large area and face normal deviation from upDir \n
    /// All new faces should be parallel to given direction
    /// Generated from function `MR::getVerticalStitchMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetVerticalStitchMetric(MR.Const_Mesh mesh, MR.Const_Vector3f upDir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVerticalStitchMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getVerticalStitchMetric(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector3f._Underlying *upDir);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getVerticalStitchMetric(mesh._UnderlyingPtr, upDir._UnderlyingPtr), is_owning: true));
    }

    /// Forbids connecting vertices from the same hole \n
    /// penalize for long edges and its deviation from upDir \n
    /// All new faces should be parallel to given direction
    /// Generated from function `MR::getVerticalStitchMetricEdgeBased`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetVerticalStitchMetricEdgeBased(MR.Const_Mesh mesh, MR.Const_Vector3f upDir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVerticalStitchMetricEdgeBased", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getVerticalStitchMetricEdgeBased(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector3f._Underlying *upDir);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getVerticalStitchMetricEdgeBased(mesh._UnderlyingPtr, upDir._UnderlyingPtr), is_owning: true));
    }

    /// This metric minimizes the sum of triangleMetric for all triangles in the triangulation
    /// plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n
    /// Where\n
    /// triangleMetric is proportional to weighted triangle area and triangle aspect ratio\n
    /// edgeMetric grows with angle between triangles as ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4.
    /// Generated from function `MR::getComplexFillMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetComplexFillMetric(MR.Const_Mesh mesh, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getComplexFillMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getComplexFillMetric(MR.Const_Mesh._Underlying *mesh, MR.EdgeId e);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getComplexFillMetric(mesh._UnderlyingPtr, e), is_owning: true));
    }

    /// This metric minimizes summary projection of new edges to plane normal, (try do produce edges parallel to plane)
    /// Generated from function `MR::getParallelPlaneFillMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetParallelPlaneFillMetric(MR.Const_Mesh mesh, MR.EdgeId e, MR.Const_Plane3f? plane = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getParallelPlaneFillMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getParallelPlaneFillMetric(MR.Const_Mesh._Underlying *mesh, MR.EdgeId e, MR.Const_Plane3f._Underlying *plane);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getParallelPlaneFillMetric(mesh._UnderlyingPtr, e, plane is not null ? plane._UnderlyingPtr : null), is_owning: true));
    }

    /// This metric minimizes the maximal dihedral angle between the faces in the triangulation
    /// and on its boundary
    /// Generated from function `MR::getMaxDihedralAngleMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetMaxDihedralAngleMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMaxDihedralAngleMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getMaxDihedralAngleMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getMaxDihedralAngleMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// This metric consists of two parts
    /// 1) for each triangle: it is the circumcircle diameter,
    ///    this avoids the appearance of degenerate triangles;
    /// 2) for each edge: square root of double total area of triangles to its left and right
    ///    times the factor depending extensionally on absolute dihedral angle between left and right triangles,
    ///    this makes visually triangulated surface as smooth as possible.
    /// For planar holes it is the same as getCircumscribedMetric.
    /// Generated from function `MR::getUniversalMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetUniversalMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getUniversalMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getUniversalMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getUniversalMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// This metric maximizes the minimal angle among all faces in the triangulation
    /// Generated from function `MR::getMinTriAngleMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetMinTriAngleMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMinTriAngleMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getMinTriAngleMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getMinTriAngleMetric(mesh._UnderlyingPtr), is_owning: true));
    }

    /// This metric is for triangulation construction with minimal summed area of triangles.
    /// Warning: this metric can produce degenerated triangles
    /// Generated from function `MR::getMinAreaMetric`.
    public static unsafe MR.Misc._Moved<MR.FillHoleMetric> GetMinAreaMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getMinAreaMetric", ExactSpelling = true)]
        extern static MR.FillHoleMetric._Underlying *__MR_getMinAreaMetric(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FillHoleMetric(__MR_getMinAreaMetric(mesh._UnderlyingPtr), is_owning: true));
    }
}
