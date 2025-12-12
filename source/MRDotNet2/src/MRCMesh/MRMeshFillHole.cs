public static partial class MR
{
    /** \struct MR::FillHoleParams
    * \brief Parameters structure for MR::fillHole\n
    * Structure has some options to control MR::fillHole
    * 
    * \sa \ref fillHole
    * \sa \ref FillHoleMetric
    */
    /// Generated from class `MR::FillHoleParams`.
    /// This is the const half of the class.
    public class Const_FillHoleParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FillHoleParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Destroy", ExactSpelling = true)]
            extern static void __MR_FillHoleParams_Destroy(_Underlying *_this);
            __MR_FillHoleParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FillHoleParams() {Dispose(false);}

        /** Specifies triangulation metric\n
        * default for MR::fillHole: getCircumscribedFillMetric\n
        * \sa \ref FillHoleMetric
        */
        public unsafe MR.Const_FillHoleMetric Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_metric", ExactSpelling = true)]
                extern static MR.Const_FillHoleMetric._Underlying *__MR_FillHoleParams_Get_metric(_Underlying *_this);
                return new(__MR_FillHoleParams_Get_metric(_UnderlyingPtr), is_owning: false);
            }
        }

        /** If true, hole filling will minimize the sum of metrics including boundary edges,
        *   where one triangle was present before hole filling, and another is added during hole filling.
        *   This makes boundary edges same smooth as inner edges of the patch.
        *   If false, edge metric will not be applied to boundary edges, and the patch tends to make a sharper turn there.
        */
        public unsafe bool SmoothBd
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_smoothBd", ExactSpelling = true)]
                extern static bool *__MR_FillHoleParams_Get_smoothBd(_Underlying *_this);
                return *__MR_FillHoleParams_Get_smoothBd(_UnderlyingPtr);
            }
        }

        /// If not nullptr accumulate new faces
        public unsafe ref void * OutNewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_outNewFaces", ExactSpelling = true)]
                extern static void **__MR_FillHoleParams_Get_outNewFaces(_Underlying *_this);
                return ref *__MR_FillHoleParams_Get_outNewFaces(_UnderlyingPtr);
            }
        }

        public unsafe MR.FillHoleParams.MultipleEdgesResolveMode MultipleEdgesResolveMode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_multipleEdgesResolveMode", ExactSpelling = true)]
                extern static MR.FillHoleParams.MultipleEdgesResolveMode *__MR_FillHoleParams_Get_multipleEdgesResolveMode(_Underlying *_this);
                return *__MR_FillHoleParams_Get_multipleEdgesResolveMode(_UnderlyingPtr);
            }
        }

        /** If true creates degenerate faces band around hole to have sharp angle visualization
        * \warning This flag bad for result topology, most likely you do not need it
        */
        public unsafe bool MakeDegenerateBand
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_makeDegenerateBand", ExactSpelling = true)]
                extern static bool *__MR_FillHoleParams_Get_makeDegenerateBand(_Underlying *_this);
                return *__MR_FillHoleParams_Get_makeDegenerateBand(_UnderlyingPtr);
            }
        }

        /** The maximum number of polygon subdivisions on a triangle and two smaller polygons,
        * must be 2 or larger
        */
        public unsafe int MaxPolygonSubdivisions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_maxPolygonSubdivisions", ExactSpelling = true)]
                extern static int *__MR_FillHoleParams_Get_maxPolygonSubdivisions(_Underlying *_this);
                return *__MR_FillHoleParams_Get_maxPolygonSubdivisions(_UnderlyingPtr);
            }
        }

        /** Input/output value, if it is present: 
        * returns true if triangulation was bad and do not actually fill hole, 
        * if triangulation is ok returns false; 
        * if it is not present fill hole trivially in case of bad triangulation, (or leaves bad triangulation, depending on metric)
        */
        public unsafe ref byte * StopBeforeBadTriangulation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_Get_stopBeforeBadTriangulation", ExactSpelling = true)]
                extern static byte **__MR_FillHoleParams_Get_stopBeforeBadTriangulation(_Underlying *_this);
                return ref *__MR_FillHoleParams_Get_stopBeforeBadTriangulation(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FillHoleParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleParams_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleParams` elementwise.
        public unsafe Const_FillHoleParams(MR._ByValue_FillHoleMetric metric, bool smoothBd, MR.FaceBitSet? outNewFaces, MR.FillHoleParams.MultipleEdgesResolveMode multipleEdgesResolveMode, bool makeDegenerateBand, int maxPolygonSubdivisions, MR.Misc.InOut<bool>? stopBeforeBadTriangulation) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_ConstructFrom(MR.Misc._PassBy metric_pass_by, MR.FillHoleMetric._Underlying *metric, byte smoothBd, MR.FaceBitSet._Underlying *outNewFaces, MR.FillHoleParams.MultipleEdgesResolveMode multipleEdgesResolveMode, byte makeDegenerateBand, int maxPolygonSubdivisions, bool *stopBeforeBadTriangulation);
            bool __value_stopBeforeBadTriangulation = stopBeforeBadTriangulation is not null ? stopBeforeBadTriangulation.Value : default(bool);
            _UnderlyingPtr = __MR_FillHoleParams_ConstructFrom(metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null, smoothBd ? (byte)1 : (byte)0, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null, multipleEdgesResolveMode, makeDegenerateBand ? (byte)1 : (byte)0, maxPolygonSubdivisions, stopBeforeBadTriangulation is not null ? &__value_stopBeforeBadTriangulation : null);
            if (stopBeforeBadTriangulation is not null) stopBeforeBadTriangulation.Value = __value_stopBeforeBadTriangulation;
        }

        /// Generated from constructor `MR::FillHoleParams::FillHoleParams`.
        public unsafe Const_FillHoleParams(MR._ByValue_FillHoleParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FillHoleParams._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /** If Strong makes additional efforts to avoid creating multiple edges, 
        * in some rare cases it is not possible (cases with extremely bad topology), 
        * if you faced one try to use \ref MR::duplicateMultiHoleVertices before \ref MR::fillHole
        * 
        * If Simple avoid creating edges that already exist in topology (default)
        * 
        * If None do not avoid multiple edges
        */
        public enum MultipleEdgesResolveMode : int
        {
            None = 0,
            Simple = 1,
            Strong = 2,
        }
    }

    /** \struct MR::FillHoleParams
    * \brief Parameters structure for MR::fillHole\n
    * Structure has some options to control MR::fillHole
    * 
    * \sa \ref fillHole
    * \sa \ref FillHoleMetric
    */
    /// Generated from class `MR::FillHoleParams`.
    /// This is the non-const half of the class.
    public class FillHoleParams : Const_FillHoleParams
    {
        internal unsafe FillHoleParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /** Specifies triangulation metric\n
        * default for MR::fillHole: getCircumscribedFillMetric\n
        * \sa \ref FillHoleMetric
        */
        public new unsafe MR.FillHoleMetric Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_metric", ExactSpelling = true)]
                extern static MR.FillHoleMetric._Underlying *__MR_FillHoleParams_GetMutable_metric(_Underlying *_this);
                return new(__MR_FillHoleParams_GetMutable_metric(_UnderlyingPtr), is_owning: false);
            }
        }

        /** If true, hole filling will minimize the sum of metrics including boundary edges,
        *   where one triangle was present before hole filling, and another is added during hole filling.
        *   This makes boundary edges same smooth as inner edges of the patch.
        *   If false, edge metric will not be applied to boundary edges, and the patch tends to make a sharper turn there.
        */
        public new unsafe ref bool SmoothBd
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_smoothBd", ExactSpelling = true)]
                extern static bool *__MR_FillHoleParams_GetMutable_smoothBd(_Underlying *_this);
                return ref *__MR_FillHoleParams_GetMutable_smoothBd(_UnderlyingPtr);
            }
        }

        /// If not nullptr accumulate new faces
        public new unsafe ref void * OutNewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_outNewFaces", ExactSpelling = true)]
                extern static void **__MR_FillHoleParams_GetMutable_outNewFaces(_Underlying *_this);
                return ref *__MR_FillHoleParams_GetMutable_outNewFaces(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.FillHoleParams.MultipleEdgesResolveMode MultipleEdgesResolveMode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_multipleEdgesResolveMode", ExactSpelling = true)]
                extern static MR.FillHoleParams.MultipleEdgesResolveMode *__MR_FillHoleParams_GetMutable_multipleEdgesResolveMode(_Underlying *_this);
                return ref *__MR_FillHoleParams_GetMutable_multipleEdgesResolveMode(_UnderlyingPtr);
            }
        }

        /** If true creates degenerate faces band around hole to have sharp angle visualization
        * \warning This flag bad for result topology, most likely you do not need it
        */
        public new unsafe ref bool MakeDegenerateBand
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_makeDegenerateBand", ExactSpelling = true)]
                extern static bool *__MR_FillHoleParams_GetMutable_makeDegenerateBand(_Underlying *_this);
                return ref *__MR_FillHoleParams_GetMutable_makeDegenerateBand(_UnderlyingPtr);
            }
        }

        /** The maximum number of polygon subdivisions on a triangle and two smaller polygons,
        * must be 2 or larger
        */
        public new unsafe ref int MaxPolygonSubdivisions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_maxPolygonSubdivisions", ExactSpelling = true)]
                extern static int *__MR_FillHoleParams_GetMutable_maxPolygonSubdivisions(_Underlying *_this);
                return ref *__MR_FillHoleParams_GetMutable_maxPolygonSubdivisions(_UnderlyingPtr);
            }
        }

        /** Input/output value, if it is present: 
        * returns true if triangulation was bad and do not actually fill hole, 
        * if triangulation is ok returns false; 
        * if it is not present fill hole trivially in case of bad triangulation, (or leaves bad triangulation, depending on metric)
        */
        public new unsafe ref byte * StopBeforeBadTriangulation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_GetMutable_stopBeforeBadTriangulation", ExactSpelling = true)]
                extern static byte **__MR_FillHoleParams_GetMutable_stopBeforeBadTriangulation(_Underlying *_this);
                return ref *__MR_FillHoleParams_GetMutable_stopBeforeBadTriangulation(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FillHoleParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleParams_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleParams` elementwise.
        public unsafe FillHoleParams(MR._ByValue_FillHoleMetric metric, bool smoothBd, MR.FaceBitSet? outNewFaces, MR.FillHoleParams.MultipleEdgesResolveMode multipleEdgesResolveMode, bool makeDegenerateBand, int maxPolygonSubdivisions, MR.Misc.InOut<bool>? stopBeforeBadTriangulation) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_ConstructFrom(MR.Misc._PassBy metric_pass_by, MR.FillHoleMetric._Underlying *metric, byte smoothBd, MR.FaceBitSet._Underlying *outNewFaces, MR.FillHoleParams.MultipleEdgesResolveMode multipleEdgesResolveMode, byte makeDegenerateBand, int maxPolygonSubdivisions, bool *stopBeforeBadTriangulation);
            bool __value_stopBeforeBadTriangulation = stopBeforeBadTriangulation is not null ? stopBeforeBadTriangulation.Value : default(bool);
            _UnderlyingPtr = __MR_FillHoleParams_ConstructFrom(metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null, smoothBd ? (byte)1 : (byte)0, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null, multipleEdgesResolveMode, makeDegenerateBand ? (byte)1 : (byte)0, maxPolygonSubdivisions, stopBeforeBadTriangulation is not null ? &__value_stopBeforeBadTriangulation : null);
            if (stopBeforeBadTriangulation is not null) stopBeforeBadTriangulation.Value = __value_stopBeforeBadTriangulation;
        }

        /// Generated from constructor `MR::FillHoleParams::FillHoleParams`.
        public unsafe FillHoleParams(MR._ByValue_FillHoleParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FillHoleParams._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FillHoleParams::operator=`.
        public unsafe MR.FillHoleParams Assign(MR._ByValue_FillHoleParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleParams._Underlying *__MR_FillHoleParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FillHoleParams._Underlying *_other);
            return new(__MR_FillHoleParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FillHoleParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FillHoleParams`/`Const_FillHoleParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FillHoleParams
    {
        internal readonly Const_FillHoleParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FillHoleParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FillHoleParams(Const_FillHoleParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FillHoleParams(Const_FillHoleParams arg) {return new(arg);}
        public _ByValue_FillHoleParams(MR.Misc._Moved<FillHoleParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FillHoleParams(MR.Misc._Moved<FillHoleParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FillHoleParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FillHoleParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleParams`/`Const_FillHoleParams` directly.
    public class _InOptMut_FillHoleParams
    {
        public FillHoleParams? Opt;

        public _InOptMut_FillHoleParams() {}
        public _InOptMut_FillHoleParams(FillHoleParams value) {Opt = value;}
        public static implicit operator _InOptMut_FillHoleParams(FillHoleParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `FillHoleParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FillHoleParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleParams`/`Const_FillHoleParams` to pass it to the function.
    public class _InOptConst_FillHoleParams
    {
        public Const_FillHoleParams? Opt;

        public _InOptConst_FillHoleParams() {}
        public _InOptConst_FillHoleParams(Const_FillHoleParams value) {Opt = value;}
        public static implicit operator _InOptConst_FillHoleParams(Const_FillHoleParams value) {return new(value);}
    }

    /** \struct MR::StitchHolesParams
    * \brief Parameters structure for MR::buildCylinderBetweenTwoHoles\n
    * Structure has some options to control MR::buildCylinderBetweenTwoHoles
    *
    * \sa \ref buildCylinderBetweenTwoHoles
    * \sa \ref FillHoleMetric
    */
    /// Generated from class `MR::StitchHolesParams`.
    /// This is the const half of the class.
    public class Const_StitchHolesParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_StitchHolesParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_Destroy", ExactSpelling = true)]
            extern static void __MR_StitchHolesParams_Destroy(_Underlying *_this);
            __MR_StitchHolesParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_StitchHolesParams() {Dispose(false);}

        /** Specifies triangulation metric\n
        * default for MR::buildCylinderBetweenTwoHoles: getComplexStitchMetric
        * \sa \ref FillHoleMetric
        */
        public unsafe MR.Const_FillHoleMetric Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_Get_metric", ExactSpelling = true)]
                extern static MR.Const_FillHoleMetric._Underlying *__MR_StitchHolesParams_Get_metric(_Underlying *_this);
                return new(__MR_StitchHolesParams_Get_metric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If not nullptr accumulate new faces
        public unsafe ref void * OutNewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_Get_outNewFaces", ExactSpelling = true)]
                extern static void **__MR_StitchHolesParams_Get_outNewFaces(_Underlying *_this);
                return ref *__MR_StitchHolesParams_Get_outNewFaces(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_StitchHolesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_StitchHolesParams_DefaultConstruct();
        }

        /// Constructs `MR::StitchHolesParams` elementwise.
        public unsafe Const_StitchHolesParams(MR._ByValue_FillHoleMetric metric, MR.FaceBitSet? outNewFaces) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_ConstructFrom(MR.Misc._PassBy metric_pass_by, MR.FillHoleMetric._Underlying *metric, MR.FaceBitSet._Underlying *outNewFaces);
            _UnderlyingPtr = __MR_StitchHolesParams_ConstructFrom(metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::StitchHolesParams::StitchHolesParams`.
        public unsafe Const_StitchHolesParams(MR._ByValue_StitchHolesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.StitchHolesParams._Underlying *_other);
            _UnderlyingPtr = __MR_StitchHolesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /** \struct MR::StitchHolesParams
    * \brief Parameters structure for MR::buildCylinderBetweenTwoHoles\n
    * Structure has some options to control MR::buildCylinderBetweenTwoHoles
    *
    * \sa \ref buildCylinderBetweenTwoHoles
    * \sa \ref FillHoleMetric
    */
    /// Generated from class `MR::StitchHolesParams`.
    /// This is the non-const half of the class.
    public class StitchHolesParams : Const_StitchHolesParams
    {
        internal unsafe StitchHolesParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /** Specifies triangulation metric\n
        * default for MR::buildCylinderBetweenTwoHoles: getComplexStitchMetric
        * \sa \ref FillHoleMetric
        */
        public new unsafe MR.FillHoleMetric Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_GetMutable_metric", ExactSpelling = true)]
                extern static MR.FillHoleMetric._Underlying *__MR_StitchHolesParams_GetMutable_metric(_Underlying *_this);
                return new(__MR_StitchHolesParams_GetMutable_metric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If not nullptr accumulate new faces
        public new unsafe ref void * OutNewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_GetMutable_outNewFaces", ExactSpelling = true)]
                extern static void **__MR_StitchHolesParams_GetMutable_outNewFaces(_Underlying *_this);
                return ref *__MR_StitchHolesParams_GetMutable_outNewFaces(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe StitchHolesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_StitchHolesParams_DefaultConstruct();
        }

        /// Constructs `MR::StitchHolesParams` elementwise.
        public unsafe StitchHolesParams(MR._ByValue_FillHoleMetric metric, MR.FaceBitSet? outNewFaces) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_ConstructFrom(MR.Misc._PassBy metric_pass_by, MR.FillHoleMetric._Underlying *metric, MR.FaceBitSet._Underlying *outNewFaces);
            _UnderlyingPtr = __MR_StitchHolesParams_ConstructFrom(metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::StitchHolesParams::StitchHolesParams`.
        public unsafe StitchHolesParams(MR._ByValue_StitchHolesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.StitchHolesParams._Underlying *_other);
            _UnderlyingPtr = __MR_StitchHolesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::StitchHolesParams::operator=`.
        public unsafe MR.StitchHolesParams Assign(MR._ByValue_StitchHolesParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_StitchHolesParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.StitchHolesParams._Underlying *__MR_StitchHolesParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.StitchHolesParams._Underlying *_other);
            return new(__MR_StitchHolesParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `StitchHolesParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `StitchHolesParams`/`Const_StitchHolesParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_StitchHolesParams
    {
        internal readonly Const_StitchHolesParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_StitchHolesParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_StitchHolesParams(Const_StitchHolesParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_StitchHolesParams(Const_StitchHolesParams arg) {return new(arg);}
        public _ByValue_StitchHolesParams(MR.Misc._Moved<StitchHolesParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_StitchHolesParams(MR.Misc._Moved<StitchHolesParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `StitchHolesParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_StitchHolesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `StitchHolesParams`/`Const_StitchHolesParams` directly.
    public class _InOptMut_StitchHolesParams
    {
        public StitchHolesParams? Opt;

        public _InOptMut_StitchHolesParams() {}
        public _InOptMut_StitchHolesParams(StitchHolesParams value) {Opt = value;}
        public static implicit operator _InOptMut_StitchHolesParams(StitchHolesParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `StitchHolesParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_StitchHolesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `StitchHolesParams`/`Const_StitchHolesParams` to pass it to the function.
    public class _InOptConst_StitchHolesParams
    {
        public Const_StitchHolesParams? Opt;

        public _InOptConst_StitchHolesParams() {}
        public _InOptConst_StitchHolesParams(Const_StitchHolesParams value) {Opt = value;}
        public static implicit operator _InOptConst_StitchHolesParams(Const_StitchHolesParams value) {return new(value);}
    }

    /// Generated from class `MR::FillHoleItem`.
    /// This is the const half of the class.
    public class Const_FillHoleItem : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FillHoleItem(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_Destroy", ExactSpelling = true)]
            extern static void __MR_FillHoleItem_Destroy(_Underlying *_this);
            __MR_FillHoleItem_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FillHoleItem() {Dispose(false);}

        // if not-negative number then it is edgeid;
        // otherwise it refers to the edge created recently
        public unsafe int EdgeCode1
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_Get_edgeCode1", ExactSpelling = true)]
                extern static int *__MR_FillHoleItem_Get_edgeCode1(_Underlying *_this);
                return *__MR_FillHoleItem_Get_edgeCode1(_UnderlyingPtr);
            }
        }

        // if not-negative number then it is edgeid;
        // otherwise it refers to the edge created recently
        public unsafe int EdgeCode2
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_Get_edgeCode2", ExactSpelling = true)]
                extern static int *__MR_FillHoleItem_Get_edgeCode2(_Underlying *_this);
                return *__MR_FillHoleItem_Get_edgeCode2(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FillHoleItem() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleItem_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleItem` elementwise.
        public unsafe Const_FillHoleItem(int edgeCode1, int edgeCode2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_ConstructFrom(int edgeCode1, int edgeCode2);
            _UnderlyingPtr = __MR_FillHoleItem_ConstructFrom(edgeCode1, edgeCode2);
        }

        /// Generated from constructor `MR::FillHoleItem::FillHoleItem`.
        public unsafe Const_FillHoleItem(MR.Const_FillHoleItem _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_ConstructFromAnother(MR.FillHoleItem._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleItem_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::FillHoleItem`.
    /// This is the non-const half of the class.
    public class FillHoleItem : Const_FillHoleItem
    {
        internal unsafe FillHoleItem(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // if not-negative number then it is edgeid;
        // otherwise it refers to the edge created recently
        public new unsafe ref int EdgeCode1
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_GetMutable_edgeCode1", ExactSpelling = true)]
                extern static int *__MR_FillHoleItem_GetMutable_edgeCode1(_Underlying *_this);
                return ref *__MR_FillHoleItem_GetMutable_edgeCode1(_UnderlyingPtr);
            }
        }

        // if not-negative number then it is edgeid;
        // otherwise it refers to the edge created recently
        public new unsafe ref int EdgeCode2
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_GetMutable_edgeCode2", ExactSpelling = true)]
                extern static int *__MR_FillHoleItem_GetMutable_edgeCode2(_Underlying *_this);
                return ref *__MR_FillHoleItem_GetMutable_edgeCode2(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FillHoleItem() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleItem_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleItem` elementwise.
        public unsafe FillHoleItem(int edgeCode1, int edgeCode2) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_ConstructFrom(int edgeCode1, int edgeCode2);
            _UnderlyingPtr = __MR_FillHoleItem_ConstructFrom(edgeCode1, edgeCode2);
        }

        /// Generated from constructor `MR::FillHoleItem::FillHoleItem`.
        public unsafe FillHoleItem(MR.Const_FillHoleItem _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_ConstructFromAnother(MR.FillHoleItem._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleItem_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::FillHoleItem::operator=`.
        public unsafe MR.FillHoleItem Assign(MR.Const_FillHoleItem _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleItem_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleItem._Underlying *__MR_FillHoleItem_AssignFromAnother(_Underlying *_this, MR.FillHoleItem._Underlying *_other);
            return new(__MR_FillHoleItem_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FillHoleItem` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FillHoleItem`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleItem`/`Const_FillHoleItem` directly.
    public class _InOptMut_FillHoleItem
    {
        public FillHoleItem? Opt;

        public _InOptMut_FillHoleItem() {}
        public _InOptMut_FillHoleItem(FillHoleItem value) {Opt = value;}
        public static implicit operator _InOptMut_FillHoleItem(FillHoleItem value) {return new(value);}
    }

    /// This is used for optional parameters of class `FillHoleItem` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FillHoleItem`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleItem`/`Const_FillHoleItem` to pass it to the function.
    public class _InOptConst_FillHoleItem
    {
        public Const_FillHoleItem? Opt;

        public _InOptConst_FillHoleItem() {}
        public _InOptConst_FillHoleItem(Const_FillHoleItem value) {Opt = value;}
        public static implicit operator _InOptConst_FillHoleItem(Const_FillHoleItem value) {return new(value);}
    }

    /// concise representation of proposed hole triangulation
    /// Generated from class `MR::HoleFillPlan`.
    /// This is the const half of the class.
    public class Const_HoleFillPlan : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_HoleFillPlan(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_Destroy", ExactSpelling = true)]
            extern static void __MR_HoleFillPlan_Destroy(_Underlying *_this);
            __MR_HoleFillPlan_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_HoleFillPlan() {Dispose(false);}

        public unsafe MR.Std.Const_Vector_MRFillHoleItem Items
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_Get_items", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRFillHoleItem._Underlying *__MR_HoleFillPlan_Get_items(_Underlying *_this);
                return new(__MR_HoleFillPlan_Get_items(_UnderlyingPtr), is_owning: false);
            }
        }

        // the number of triangles in the filling
        public unsafe int NumTris
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_Get_numTris", ExactSpelling = true)]
                extern static int *__MR_HoleFillPlan_Get_numTris(_Underlying *_this);
                return *__MR_HoleFillPlan_Get_numTris(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_HoleFillPlan() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_DefaultConstruct", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_DefaultConstruct();
            _UnderlyingPtr = __MR_HoleFillPlan_DefaultConstruct();
        }

        /// Constructs `MR::HoleFillPlan` elementwise.
        public unsafe Const_HoleFillPlan(MR.Std._ByValue_Vector_MRFillHoleItem items, int numTris) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_ConstructFrom", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_ConstructFrom(MR.Misc._PassBy items_pass_by, MR.Std.Vector_MRFillHoleItem._Underlying *items, int numTris);
            _UnderlyingPtr = __MR_HoleFillPlan_ConstructFrom(items.PassByMode, items.Value is not null ? items.Value._UnderlyingPtr : null, numTris);
        }

        /// Generated from constructor `MR::HoleFillPlan::HoleFillPlan`.
        public unsafe Const_HoleFillPlan(MR._ByValue_HoleFillPlan _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.HoleFillPlan._Underlying *_other);
            _UnderlyingPtr = __MR_HoleFillPlan_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// concise representation of proposed hole triangulation
    /// Generated from class `MR::HoleFillPlan`.
    /// This is the non-const half of the class.
    public class HoleFillPlan : Const_HoleFillPlan
    {
        internal unsafe HoleFillPlan(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Vector_MRFillHoleItem Items
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_GetMutable_items", ExactSpelling = true)]
                extern static MR.Std.Vector_MRFillHoleItem._Underlying *__MR_HoleFillPlan_GetMutable_items(_Underlying *_this);
                return new(__MR_HoleFillPlan_GetMutable_items(_UnderlyingPtr), is_owning: false);
            }
        }

        // the number of triangles in the filling
        public new unsafe ref int NumTris
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_GetMutable_numTris", ExactSpelling = true)]
                extern static int *__MR_HoleFillPlan_GetMutable_numTris(_Underlying *_this);
                return ref *__MR_HoleFillPlan_GetMutable_numTris(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe HoleFillPlan() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_DefaultConstruct", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_DefaultConstruct();
            _UnderlyingPtr = __MR_HoleFillPlan_DefaultConstruct();
        }

        /// Constructs `MR::HoleFillPlan` elementwise.
        public unsafe HoleFillPlan(MR.Std._ByValue_Vector_MRFillHoleItem items, int numTris) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_ConstructFrom", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_ConstructFrom(MR.Misc._PassBy items_pass_by, MR.Std.Vector_MRFillHoleItem._Underlying *items, int numTris);
            _UnderlyingPtr = __MR_HoleFillPlan_ConstructFrom(items.PassByMode, items.Value is not null ? items.Value._UnderlyingPtr : null, numTris);
        }

        /// Generated from constructor `MR::HoleFillPlan::HoleFillPlan`.
        public unsafe HoleFillPlan(MR._ByValue_HoleFillPlan _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.HoleFillPlan._Underlying *_other);
            _UnderlyingPtr = __MR_HoleFillPlan_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::HoleFillPlan::operator=`.
        public unsafe MR.HoleFillPlan Assign(MR._ByValue_HoleFillPlan _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_HoleFillPlan_AssignFromAnother", ExactSpelling = true)]
            extern static MR.HoleFillPlan._Underlying *__MR_HoleFillPlan_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.HoleFillPlan._Underlying *_other);
            return new(__MR_HoleFillPlan_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `HoleFillPlan` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `HoleFillPlan`/`Const_HoleFillPlan` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_HoleFillPlan
    {
        internal readonly Const_HoleFillPlan? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_HoleFillPlan() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_HoleFillPlan(Const_HoleFillPlan new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_HoleFillPlan(Const_HoleFillPlan arg) {return new(arg);}
        public _ByValue_HoleFillPlan(MR.Misc._Moved<HoleFillPlan> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_HoleFillPlan(MR.Misc._Moved<HoleFillPlan> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `HoleFillPlan` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_HoleFillPlan`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `HoleFillPlan`/`Const_HoleFillPlan` directly.
    public class _InOptMut_HoleFillPlan
    {
        public HoleFillPlan? Opt;

        public _InOptMut_HoleFillPlan() {}
        public _InOptMut_HoleFillPlan(HoleFillPlan value) {Opt = value;}
        public static implicit operator _InOptMut_HoleFillPlan(HoleFillPlan value) {return new(value);}
    }

    /// This is used for optional parameters of class `HoleFillPlan` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_HoleFillPlan`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `HoleFillPlan`/`Const_HoleFillPlan` to pass it to the function.
    public class _InOptConst_HoleFillPlan
    {
        public Const_HoleFillPlan? Opt;

        public _InOptConst_HoleFillPlan() {}
        public _InOptConst_HoleFillPlan(Const_HoleFillPlan value) {Opt = value;}
        public static implicit operator _InOptConst_HoleFillPlan(Const_HoleFillPlan value) {return new(value);}
    }

    /// Generated from class `MR::MakeBridgeResult`.
    /// This is the const half of the class.
    public class Const_MakeBridgeResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MakeBridgeResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MakeBridgeResult_Destroy(_Underlying *_this);
            __MR_MakeBridgeResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MakeBridgeResult() {Dispose(false);}

        /// the number of faces added to the mesh
        public unsafe int NewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_Get_newFaces", ExactSpelling = true)]
                extern static int *__MR_MakeBridgeResult_Get_newFaces(_Underlying *_this);
                return *__MR_MakeBridgeResult_Get_newFaces(_UnderlyingPtr);
            }
        }

        /// the edge na (nb) if valid is a new boundary edge of the created bridge without left face,
        /// having the same origin as input edge a (b)
        public unsafe MR.Const_EdgeId Na
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_Get_na", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_MakeBridgeResult_Get_na(_Underlying *_this);
                return new(__MR_MakeBridgeResult_Get_na(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the edge na (nb) if valid is a new boundary edge of the created bridge without left face,
        /// having the same origin as input edge a (b)
        public unsafe MR.Const_EdgeId Nb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_Get_nb", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_MakeBridgeResult_Get_nb(_Underlying *_this);
                return new(__MR_MakeBridgeResult_Get_nb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MakeBridgeResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MakeBridgeResult_DefaultConstruct();
        }

        /// Constructs `MR::MakeBridgeResult` elementwise.
        public unsafe Const_MakeBridgeResult(int newFaces, MR.EdgeId na, MR.EdgeId nb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_ConstructFrom(int newFaces, MR.EdgeId na, MR.EdgeId nb);
            _UnderlyingPtr = __MR_MakeBridgeResult_ConstructFrom(newFaces, na, nb);
        }

        /// Generated from constructor `MR::MakeBridgeResult::MakeBridgeResult`.
        public unsafe Const_MakeBridgeResult(MR.Const_MakeBridgeResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_ConstructFromAnother(MR.MakeBridgeResult._Underlying *_other);
            _UnderlyingPtr = __MR_MakeBridgeResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// bridge construction is successful if at least one new face was created
        /// Generated from conversion operator `MR::MakeBridgeResult::operator bool`.
        public static unsafe explicit operator bool(MR.Const_MakeBridgeResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_MakeBridgeResult_ConvertTo_bool(MR.Const_MakeBridgeResult._Underlying *_this);
            return __MR_MakeBridgeResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::MakeBridgeResult`.
    /// This is the non-const half of the class.
    public class MakeBridgeResult : Const_MakeBridgeResult
    {
        internal unsafe MakeBridgeResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the number of faces added to the mesh
        public new unsafe ref int NewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_GetMutable_newFaces", ExactSpelling = true)]
                extern static int *__MR_MakeBridgeResult_GetMutable_newFaces(_Underlying *_this);
                return ref *__MR_MakeBridgeResult_GetMutable_newFaces(_UnderlyingPtr);
            }
        }

        /// the edge na (nb) if valid is a new boundary edge of the created bridge without left face,
        /// having the same origin as input edge a (b)
        public new unsafe MR.Mut_EdgeId Na
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_GetMutable_na", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_MakeBridgeResult_GetMutable_na(_Underlying *_this);
                return new(__MR_MakeBridgeResult_GetMutable_na(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the edge na (nb) if valid is a new boundary edge of the created bridge without left face,
        /// having the same origin as input edge a (b)
        public new unsafe MR.Mut_EdgeId Nb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_GetMutable_nb", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_MakeBridgeResult_GetMutable_nb(_Underlying *_this);
                return new(__MR_MakeBridgeResult_GetMutable_nb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MakeBridgeResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MakeBridgeResult_DefaultConstruct();
        }

        /// Constructs `MR::MakeBridgeResult` elementwise.
        public unsafe MakeBridgeResult(int newFaces, MR.EdgeId na, MR.EdgeId nb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_ConstructFrom(int newFaces, MR.EdgeId na, MR.EdgeId nb);
            _UnderlyingPtr = __MR_MakeBridgeResult_ConstructFrom(newFaces, na, nb);
        }

        /// Generated from constructor `MR::MakeBridgeResult::MakeBridgeResult`.
        public unsafe MakeBridgeResult(MR.Const_MakeBridgeResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_ConstructFromAnother(MR.MakeBridgeResult._Underlying *_other);
            _UnderlyingPtr = __MR_MakeBridgeResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MakeBridgeResult::operator=`.
        public unsafe MR.MakeBridgeResult Assign(MR.Const_MakeBridgeResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeBridgeResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MakeBridgeResult._Underlying *__MR_MakeBridgeResult_AssignFromAnother(_Underlying *_this, MR.MakeBridgeResult._Underlying *_other);
            return new(__MR_MakeBridgeResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MakeBridgeResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MakeBridgeResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MakeBridgeResult`/`Const_MakeBridgeResult` directly.
    public class _InOptMut_MakeBridgeResult
    {
        public MakeBridgeResult? Opt;

        public _InOptMut_MakeBridgeResult() {}
        public _InOptMut_MakeBridgeResult(MakeBridgeResult value) {Opt = value;}
        public static implicit operator _InOptMut_MakeBridgeResult(MakeBridgeResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MakeBridgeResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MakeBridgeResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MakeBridgeResult`/`Const_MakeBridgeResult` to pass it to the function.
    public class _InOptConst_MakeBridgeResult
    {
        public Const_MakeBridgeResult? Opt;

        public _InOptConst_MakeBridgeResult() {}
        public _InOptConst_MakeBridgeResult(Const_MakeBridgeResult value) {Opt = value;}
        public static implicit operator _InOptConst_MakeBridgeResult(Const_MakeBridgeResult value) {return new(value);}
    }

    /** \brief Stitches two holes in Mesh\n
    *
    * Build cylindrical patch to fill space between two holes represented by one of their edges each,\n
    * default metric: ComplexStitchMetric
    *
    * \image html fill/before_stitch.png "Before" width = 250cm
    * \image html fill/stitch.png "After" width = 250cm
    * 
    * Next picture show, how newly generated faces can be smoothed
    * \ref MR::positionVertsSmoothly
    * \ref MR::subdivideMesh
    * \image html fill/stitch_smooth.png "Stitch with smooth" width = 250cm
    * 
    * \snippet cpp-examples/MeshStitchHole.dox.cpp 0
    * 
    * \param mesh mesh with hole
    * \param a EdgeId which represents 1st hole (should not have valid left FaceId)
    * \param b EdgeId which represents 2nd hole (should not have valid left FaceId)
    * \param params parameters of holes stitching
    *
    * \sa \ref fillHole
    * \sa \ref StitchHolesParams
    */
    /// Generated from function `MR::buildCylinderBetweenTwoHoles`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe void BuildCylinderBetweenTwoHoles(MR.Mesh mesh, MR.EdgeId a, MR.EdgeId b, MR.Const_StitchHolesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildCylinderBetweenTwoHoles_4", ExactSpelling = true)]
        extern static void __MR_buildCylinderBetweenTwoHoles_4(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.EdgeId b, MR.Const_StitchHolesParams._Underlying *params_);
        __MR_buildCylinderBetweenTwoHoles_4(mesh._UnderlyingPtr, a, b, params_ is not null ? params_._UnderlyingPtr : null);
    }

    /// this version finds holes in the mesh by itself and returns false if they are not found
    /// Generated from function `MR::buildCylinderBetweenTwoHoles`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe bool BuildCylinderBetweenTwoHoles(MR.Mesh mesh, MR.Const_StitchHolesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildCylinderBetweenTwoHoles_2", ExactSpelling = true)]
        extern static byte __MR_buildCylinderBetweenTwoHoles_2(MR.Mesh._Underlying *mesh, MR.Const_StitchHolesParams._Underlying *params_);
        return __MR_buildCylinderBetweenTwoHoles_2(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null) != 0;
    }

    /** \brief Fills hole in mesh\n
    * 
    * Fills given hole represented by one of its edges (having no valid left face),\n
    * uses fillHoleTrivially if cannot fill hole without multiple edges,\n
    * default metric: CircumscribedFillMetric
    * 
    * \image html fill/before_fill.png "Before" width = 250cm
    * \image html fill/fill.png "After" width = 250cm
    *
    * Next picture show, how newly generated faces can be smoothed
    * \ref MR::positionVertsSmoothly
    * \ref MR::subdivideMesh
    * \image html fill/fill_smooth.png "Fill with smooth" width = 250cm
    * 
    * \param mesh mesh with hole
    * \param a EdgeId which represents hole (should not have valid left FaceId)
    * \param params parameters of hole filling
    * 
    * \sa \ref buildCylinderBetweenTwoHoles
    * \sa \ref fillHoleTrivially
    * \sa \ref FillHoleParams
    */
    /// Generated from function `MR::fillHole`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe void FillHole(MR.Mesh mesh, MR.EdgeId a, MR.Const_FillHoleParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillHole", ExactSpelling = true)]
        extern static void __MR_fillHole(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.Const_FillHoleParams._Underlying *params_);
        __MR_fillHole(mesh._UnderlyingPtr, a, params_ is not null ? params_._UnderlyingPtr : null);
    }

    /// fill all holes given by their representative edges in \param as
    /// Generated from function `MR::fillHoles`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe void FillHoles(MR.Mesh mesh, MR.Std.Const_Vector_MREdgeId as_, MR.Const_FillHoleParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillHoles", ExactSpelling = true)]
        extern static void __MR_fillHoles(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *as_, MR.Const_FillHoleParams._Underlying *params_);
        __MR_fillHoles(mesh._UnderlyingPtr, as_._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null);
    }

    /// returns true if given loop is a boundary of one hole in given mesh topology:
    /// * every edge in the loop does not have left face,
    /// * next/prev edges in the loop are related as follows: next = topology.prev( prev.sym() )
    /// if the function returns true, then any edge from the loop passed to \ref fillHole will fill the same hole
    /// Generated from function `MR::isHoleBd`.
    public static unsafe bool IsHoleBd(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId loop)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isHoleBd", ExactSpelling = true)]
        extern static byte __MR_isHoleBd(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *loop);
        return __MR_isHoleBd(topology._UnderlyingPtr, loop._UnderlyingPtr) != 0;
    }

    /// prepares the plan how to triangulate the face or hole to the left of (e) (not filling it immediately),
    /// several getHoleFillPlan can work in parallel
    /// Generated from function `MR::getHoleFillPlan`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.HoleFillPlan> GetHoleFillPlan(MR.Const_Mesh mesh, MR.EdgeId e, MR.Const_FillHoleParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getHoleFillPlan", ExactSpelling = true)]
        extern static MR.HoleFillPlan._Underlying *__MR_getHoleFillPlan(MR.Const_Mesh._Underlying *mesh, MR.EdgeId e, MR.Const_FillHoleParams._Underlying *params_);
        return MR.Misc.Move(new MR.HoleFillPlan(__MR_getHoleFillPlan(mesh._UnderlyingPtr, e, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// prepares the plans how to triangulate the faces or holes, each given by a boundary edge (with filling target to the left),
    /// the plans are prepared in parallel with minimal memory allocation compared to manual calling of several getHoleFillPlan(), but it can inefficient when some holes are very complex
    /// Generated from function `MR::getHoleFillPlans`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRHoleFillPlan> GetHoleFillPlans(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId holeRepresentativeEdges, MR.Const_FillHoleParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getHoleFillPlans", ExactSpelling = true)]
        extern static MR.Std.Vector_MRHoleFillPlan._Underlying *__MR_getHoleFillPlans(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *holeRepresentativeEdges, MR.Const_FillHoleParams._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Vector_MRHoleFillPlan(__MR_getHoleFillPlans(mesh._UnderlyingPtr, holeRepresentativeEdges._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// prepares the plan how to triangulate the planar face or planar hole to the left of (e) (not filling it immediately),
    /// several getPlanarHoleFillPlan can work in parallel
    /// Generated from function `MR::getPlanarHoleFillPlan`.
    public static unsafe MR.Misc._Moved<MR.HoleFillPlan> GetPlanarHoleFillPlan(MR.Const_Mesh mesh, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPlanarHoleFillPlan", ExactSpelling = true)]
        extern static MR.HoleFillPlan._Underlying *__MR_getPlanarHoleFillPlan(MR.Const_Mesh._Underlying *mesh, MR.EdgeId e);
        return MR.Misc.Move(new MR.HoleFillPlan(__MR_getPlanarHoleFillPlan(mesh._UnderlyingPtr, e), is_owning: true));
    }

    /// prepares the plans how to triangulate the planar faces or holes, each given by a boundary edge (with filling target to the left),
    /// the plans are prepared in parallel with minimal memory allocation compared to manual calling of several getPlanarHoleFillPlan(), but it can inefficient when some holes are very complex
    /// Generated from function `MR::getPlanarHoleFillPlans`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRHoleFillPlan> GetPlanarHoleFillPlans(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId holeRepresentativeEdges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPlanarHoleFillPlans", ExactSpelling = true)]
        extern static MR.Std.Vector_MRHoleFillPlan._Underlying *__MR_getPlanarHoleFillPlans(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *holeRepresentativeEdges);
        return MR.Misc.Move(new MR.Std.Vector_MRHoleFillPlan(__MR_getPlanarHoleFillPlans(mesh._UnderlyingPtr, holeRepresentativeEdges._UnderlyingPtr), is_owning: true));
    }

    /// quickly triangulates the face or hole to the left of (e) given the plan (quickly compared to fillHole function)
    /// Generated from function `MR::executeHoleFillPlan`.
    public static unsafe void ExecuteHoleFillPlan(MR.Mesh mesh, MR.EdgeId a0, MR.HoleFillPlan plan, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_executeHoleFillPlan", ExactSpelling = true)]
        extern static void __MR_executeHoleFillPlan(MR.Mesh._Underlying *mesh, MR.EdgeId a0, MR.HoleFillPlan._Underlying *plan, MR.FaceBitSet._Underlying *outNewFaces);
        __MR_executeHoleFillPlan(mesh._UnderlyingPtr, a0, plan._UnderlyingPtr, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }

    /** \brief Triangulates face of hole in mesh trivially\n
    *
    *
    * Fills given hole represented by one of its edges (having no valid left face)\n
    * by creating one new vertex in the centroid of boundary vertices and connecting new vertex with all boundary vertices.
    *
    * \image html fill/before_fill.png "Before" width = 250cm
    * \image html fill/fill_triv.png "After" width = 250cm
    *
    * Next picture show, how newly generated faces can be smoothed
    * \ref MR::positionVertsSmoothly
    * \ref MR::subdivideMesh
    * \image html fill/fill_triv_smooth.png "Trivial fill with smooth" width = 250cm
    *
    * \param mesh mesh with hole
    * \param a EdgeId points on the face or hole to the left that will be triangulated
    * \param outNewFaces optional output newly generated faces
    * \return new vertex
    * 
    * \sa \ref fillHole
    */
    /// Generated from function `MR::fillHoleTrivially`.
    public static unsafe MR.VertId FillHoleTrivially(MR.Mesh mesh, MR.EdgeId a, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillHoleTrivially", ExactSpelling = true)]
        extern static MR.VertId __MR_fillHoleTrivially(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.FaceBitSet._Underlying *outNewFaces);
        return __MR_fillHoleTrivially(mesh._UnderlyingPtr, a, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }

    /// adds cylindrical extension of given hole represented by one of its edges (having no valid left face)
    /// by adding new vertices located in given plane and 2 * number_of_hole_edge triangles;
    /// \return the edge of new hole opposite to input edge (a)
    /// Generated from function `MR::extendHole`.
    public static unsafe MR.EdgeId ExtendHole(MR.Mesh mesh, MR.EdgeId a, MR.Const_Plane3f plane, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extendHole_MR_Plane3f", ExactSpelling = true)]
        extern static MR.EdgeId __MR_extendHole_MR_Plane3f(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.Const_Plane3f._Underlying *plane, MR.FaceBitSet._Underlying *outNewFaces);
        return __MR_extendHole_MR_Plane3f(mesh._UnderlyingPtr, a, plane._UnderlyingPtr, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }

    /// adds cylindrical extension of too all holes of the mesh by calling extendHole(...);
    /// \return representative edges of one per every hole after extension
    /// Generated from function `MR::extendAllHoles`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> ExtendAllHoles(MR.Mesh mesh, MR.Const_Plane3f plane, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extendAllHoles", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_extendAllHoles(MR.Mesh._Underlying *mesh, MR.Const_Plane3f._Underlying *plane, MR.FaceBitSet._Underlying *outNewFaces);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_extendAllHoles(mesh._UnderlyingPtr, plane._UnderlyingPtr, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null), is_owning: true));
    }

    /// adds extension of given hole represented by one of its edges (having no valid left face)
    /// by adding new vertices located at getVertPos( existing vertex position );
    /// \return the edge of new hole opposite to input edge (a)
    /// Generated from function `MR::extendHole`.
    public static unsafe MR.EdgeId ExtendHole(MR.Mesh mesh, MR.EdgeId a, MR.Std._ByValue_Function_MRVector3fFuncFromConstMRVector3fRef getVertPos, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extendHole_std_function_MR_Vector3f_func_from_const_MR_Vector3f_ref", ExactSpelling = true)]
        extern static MR.EdgeId __MR_extendHole_std_function_MR_Vector3f_func_from_const_MR_Vector3f_ref(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.Misc._PassBy getVertPos_pass_by, MR.Std.Function_MRVector3fFuncFromConstMRVector3fRef._Underlying *getVertPos, MR.FaceBitSet._Underlying *outNewFaces);
        return __MR_extendHole_std_function_MR_Vector3f_func_from_const_MR_Vector3f_ref(mesh._UnderlyingPtr, a, getVertPos.PassByMode, getVertPos.Value is not null ? getVertPos.Value._UnderlyingPtr : null, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }

    /// adds cylindrical extension of given hole represented by one of its edges (having no valid left face)
    /// by adding new vertices located in lowest point of the hole -dir*holeExtension and 2 * number_of_hole_edge triangles;
    /// \return the edge of new hole opposite to input edge (a)
    /// Generated from function `MR::buildBottom`.
    public static unsafe MR.EdgeId BuildBottom(MR.Mesh mesh, MR.EdgeId a, MR.Vector3f dir, float holeExtension, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildBottom", ExactSpelling = true)]
        extern static MR.EdgeId __MR_buildBottom(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.Vector3f dir, float holeExtension, MR.FaceBitSet._Underlying *outNewFaces);
        return __MR_buildBottom(mesh._UnderlyingPtr, a, dir, holeExtension, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }

    /// creates a band of degenerate triangles around given hole;
    /// \return the edge of new hole opposite to input edge (a)
    /// Generated from function `MR::makeDegenerateBandAroundHole`.
    public static unsafe MR.EdgeId MakeDegenerateBandAroundHole(MR.Mesh mesh, MR.EdgeId a, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDegenerateBandAroundHole", ExactSpelling = true)]
        extern static MR.EdgeId __MR_makeDegenerateBandAroundHole(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.FaceBitSet._Underlying *outNewFaces);
        return __MR_makeDegenerateBandAroundHole(mesh._UnderlyingPtr, a, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }

    /// creates a bridge between two boundary edges a and b (both having no valid left face);
    /// bridge consists of one quadrangle in general (beware that it cannot be rendered) or of one triangle if a and b are neighboring edges on the boundary;
    /// \return false if bridge cannot be created because otherwise multiple edges appear
    /// Generated from function `MR::makeQuadBridge`.
    public static unsafe MR.MakeBridgeResult MakeQuadBridge(MR.MeshTopology topology, MR.EdgeId a, MR.EdgeId b, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeQuadBridge", ExactSpelling = true)]
        extern static MR.MakeBridgeResult._Underlying *__MR_makeQuadBridge(MR.MeshTopology._Underlying *topology, MR.EdgeId a, MR.EdgeId b, MR.FaceBitSet._Underlying *outNewFaces);
        return new(__MR_makeQuadBridge(topology._UnderlyingPtr, a, b, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null), is_owning: true);
    }

    /// creates a bridge between two boundary edges a and b (both having no valid left face);
    /// bridge consists of two triangles in general or of one triangle if a and b are neighboring edges on the boundary;
    /// \return MakeBridgeResult evaluating to false if bridge cannot be created because otherwise multiple edges appear
    /// Generated from function `MR::makeBridge`.
    public static unsafe MR.MakeBridgeResult MakeBridge(MR.MeshTopology topology, MR.EdgeId a, MR.EdgeId b, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeBridge", ExactSpelling = true)]
        extern static MR.MakeBridgeResult._Underlying *__MR_makeBridge(MR.MeshTopology._Underlying *topology, MR.EdgeId a, MR.EdgeId b, MR.FaceBitSet._Underlying *outNewFaces);
        return new(__MR_makeBridge(topology._UnderlyingPtr, a, b, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null), is_owning: true);
    }

    /// creates a bridge between two boundary edges a and b (both having no valid left face);
    /// bridge consists of strip of quadrangles (each consisting of two triangles) in general or of some triangles if a and b are neighboring edges on the boundary;
    /// the bridge is made as smooth as possible with small angles in between its links and on the boundary with existed triangles;
    /// \param samplingStep boundaries of the bridge will be subdivided until the distance between neighbor points becomes less than this distance
    /// \return MakeBridgeResult evaluating to false if bridge cannot be created because otherwise multiple edges appear
    /// Generated from function `MR::makeSmoothBridge`.
    public static unsafe MR.MakeBridgeResult MakeSmoothBridge(MR.Mesh mesh, MR.EdgeId a, MR.EdgeId b, float samplingStep, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSmoothBridge", ExactSpelling = true)]
        extern static MR.MakeBridgeResult._Underlying *__MR_makeSmoothBridge(MR.Mesh._Underlying *mesh, MR.EdgeId a, MR.EdgeId b, float samplingStep, MR.FaceBitSet._Underlying *outNewFaces);
        return new(__MR_makeSmoothBridge(mesh._UnderlyingPtr, a, b, samplingStep, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null), is_owning: true);
    }

    /// creates a new bridge edge between origins of two boundary edges a and b (both having no valid left face);
    /// \return invalid id if bridge cannot be created because otherwise multiple edges appear
    /// Generated from function `MR::makeBridgeEdge`.
    public static unsafe MR.EdgeId MakeBridgeEdge(MR.MeshTopology topology, MR.EdgeId a, MR.EdgeId b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeBridgeEdge", ExactSpelling = true)]
        extern static MR.EdgeId __MR_makeBridgeEdge(MR.MeshTopology._Underlying *topology, MR.EdgeId a, MR.EdgeId b);
        return __MR_makeBridgeEdge(topology._UnderlyingPtr, a, b);
    }

    /// given quadrangle face to the left of a, splits it in two triangles with new diagonal edge via dest(a)
    /// Generated from function `MR::splitQuad`.
    public static unsafe void SplitQuad(MR.MeshTopology topology, MR.EdgeId a, MR.FaceBitSet? outNewFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_splitQuad", ExactSpelling = true)]
        extern static void __MR_splitQuad(MR.MeshTopology._Underlying *topology, MR.EdgeId a, MR.FaceBitSet._Underlying *outNewFaces);
        __MR_splitQuad(topology._UnderlyingPtr, a, outNewFaces is not null ? outNewFaces._UnderlyingPtr : null);
    }
}
