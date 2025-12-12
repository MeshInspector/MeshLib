public static partial class MR
{
    /// Generated from class `MR::MarkedContour3f`.
    /// This is the const half of the class.
    public class Const_MarkedContour3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MarkedContour3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_Destroy", ExactSpelling = true)]
            extern static void __MR_MarkedContour3f_Destroy(_Underlying *_this);
            __MR_MarkedContour3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MarkedContour3f() {Dispose(false);}

        public unsafe MR.Std.Const_Vector_MRVector3f Contour
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_Get_contour", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRVector3f._Underlying *__MR_MarkedContour3f_Get_contour(_Underlying *_this);
                return new(__MR_MarkedContour3f_Get_contour(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< indices of control (marked) points
        public unsafe MR.Const_BitSet Marks
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_Get_marks", ExactSpelling = true)]
                extern static MR.Const_BitSet._Underlying *__MR_MarkedContour3f_Get_marks(_Underlying *_this);
                return new(__MR_MarkedContour3f_Get_marks(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MarkedContour3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_DefaultConstruct();
            _UnderlyingPtr = __MR_MarkedContour3f_DefaultConstruct();
        }

        /// Constructs `MR::MarkedContour3f` elementwise.
        public unsafe Const_MarkedContour3f(MR.Std._ByValue_Vector_MRVector3f contour, MR._ByValue_BitSet marks) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_ConstructFrom(MR.Misc._PassBy contour_pass_by, MR.Std.Vector_MRVector3f._Underlying *contour, MR.Misc._PassBy marks_pass_by, MR.BitSet._Underlying *marks);
            _UnderlyingPtr = __MR_MarkedContour3f_ConstructFrom(contour.PassByMode, contour.Value is not null ? contour.Value._UnderlyingPtr : null, marks.PassByMode, marks.Value is not null ? marks.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MarkedContour3f::MarkedContour3f`.
        public unsafe Const_MarkedContour3f(MR._ByValue_MarkedContour3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MarkedContour3f._Underlying *_other);
            _UnderlyingPtr = __MR_MarkedContour3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::MarkedContour3f`.
    /// This is the non-const half of the class.
    public class MarkedContour3f : Const_MarkedContour3f
    {
        internal unsafe MarkedContour3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Vector_MRVector3f Contour
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_GetMutable_contour", ExactSpelling = true)]
                extern static MR.Std.Vector_MRVector3f._Underlying *__MR_MarkedContour3f_GetMutable_contour(_Underlying *_this);
                return new(__MR_MarkedContour3f_GetMutable_contour(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< indices of control (marked) points
        public new unsafe MR.BitSet Marks
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_GetMutable_marks", ExactSpelling = true)]
                extern static MR.BitSet._Underlying *__MR_MarkedContour3f_GetMutable_marks(_Underlying *_this);
                return new(__MR_MarkedContour3f_GetMutable_marks(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MarkedContour3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_DefaultConstruct();
            _UnderlyingPtr = __MR_MarkedContour3f_DefaultConstruct();
        }

        /// Constructs `MR::MarkedContour3f` elementwise.
        public unsafe MarkedContour3f(MR.Std._ByValue_Vector_MRVector3f contour, MR._ByValue_BitSet marks) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_ConstructFrom(MR.Misc._PassBy contour_pass_by, MR.Std.Vector_MRVector3f._Underlying *contour, MR.Misc._PassBy marks_pass_by, MR.BitSet._Underlying *marks);
            _UnderlyingPtr = __MR_MarkedContour3f_ConstructFrom(contour.PassByMode, contour.Value is not null ? contour.Value._UnderlyingPtr : null, marks.PassByMode, marks.Value is not null ? marks.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MarkedContour3f::MarkedContour3f`.
        public unsafe MarkedContour3f(MR._ByValue_MarkedContour3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MarkedContour3f._Underlying *_other);
            _UnderlyingPtr = __MR_MarkedContour3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MarkedContour3f::operator=`.
        public unsafe MR.MarkedContour3f Assign(MR._ByValue_MarkedContour3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarkedContour3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MarkedContour3f._Underlying *__MR_MarkedContour3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MarkedContour3f._Underlying *_other);
            return new(__MR_MarkedContour3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MarkedContour3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MarkedContour3f`/`Const_MarkedContour3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MarkedContour3f
    {
        internal readonly Const_MarkedContour3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MarkedContour3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MarkedContour3f(Const_MarkedContour3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MarkedContour3f(Const_MarkedContour3f arg) {return new(arg);}
        public _ByValue_MarkedContour3f(MR.Misc._Moved<MarkedContour3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MarkedContour3f(MR.Misc._Moved<MarkedContour3f> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MarkedContour3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MarkedContour3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MarkedContour3f`/`Const_MarkedContour3f` directly.
    public class _InOptMut_MarkedContour3f
    {
        public MarkedContour3f? Opt;

        public _InOptMut_MarkedContour3f() {}
        public _InOptMut_MarkedContour3f(MarkedContour3f value) {Opt = value;}
        public static implicit operator _InOptMut_MarkedContour3f(MarkedContour3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `MarkedContour3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MarkedContour3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MarkedContour3f`/`Const_MarkedContour3f` to pass it to the function.
    public class _InOptConst_MarkedContour3f
    {
        public Const_MarkedContour3f? Opt;

        public _InOptConst_MarkedContour3f() {}
        public _InOptConst_MarkedContour3f(Const_MarkedContour3f value) {Opt = value;}
        public static implicit operator _InOptConst_MarkedContour3f(Const_MarkedContour3f value) {return new(value);}
    }

    /// Generated from class `MR::SplineSettings`.
    /// This is the const half of the class.
    public class Const_SplineSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SplineSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_SplineSettings_Destroy(_Underlying *_this);
            __MR_SplineSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SplineSettings() {Dispose(false);}

        /// additional points will be added between each pair of control points,
        /// until the distance between neighbor points becomes less than this distance
        public unsafe float SamplingStep
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_Get_samplingStep", ExactSpelling = true)]
                extern static float *__MR_SplineSettings_Get_samplingStep(_Underlying *_this);
                return *__MR_SplineSettings_Get_samplingStep(_UnderlyingPtr);
            }
        }

        /// a positive value, the more the value the closer resulting spline will be to given control points
        public unsafe float ControlStability
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_Get_controlStability", ExactSpelling = true)]
                extern static float *__MR_SplineSettings_Get_controlStability(_Underlying *_this);
                return *__MR_SplineSettings_Get_controlStability(_UnderlyingPtr);
            }
        }

        /// the shape of resulting spline depends on the total number of points in the contour,
        /// which in turn depends on the length of input contour being sampled;
        /// setting iterations greater than one allows you to pass a constructed spline as a better input contour to the next run of the algorithm
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_SplineSettings_Get_iterations(_Underlying *_this);
                return *__MR_SplineSettings_Get_iterations(_UnderlyingPtr);
            }
        }

        /// optional parameter with the normals of input points that will be resampled to become normals of output points
        public unsafe ref void * Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_Get_normals", ExactSpelling = true)]
                extern static void **__MR_SplineSettings_Get_normals(_Underlying *_this);
                return ref *__MR_SplineSettings_Get_normals(_UnderlyingPtr);
            }
        }

        /// if true and normals are provided, then the curve at marked points will try to be orthogonal to given normal there
        public unsafe bool NormalsAffectShape
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_Get_normalsAffectShape", ExactSpelling = true)]
                extern static bool *__MR_SplineSettings_Get_normalsAffectShape(_Underlying *_this);
                return *__MR_SplineSettings_Get_normalsAffectShape(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SplineSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SplineSettings_DefaultConstruct();
        }

        /// Constructs `MR::SplineSettings` elementwise.
        public unsafe Const_SplineSettings(float samplingStep, float controlStability, int iterations, MR.Std.Vector_MRVector3f? normals, bool normalsAffectShape) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_ConstructFrom(float samplingStep, float controlStability, int iterations, MR.Std.Vector_MRVector3f._Underlying *normals, byte normalsAffectShape);
            _UnderlyingPtr = __MR_SplineSettings_ConstructFrom(samplingStep, controlStability, iterations, normals is not null ? normals._UnderlyingPtr : null, normalsAffectShape ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::SplineSettings::SplineSettings`.
        public unsafe Const_SplineSettings(MR.Const_SplineSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_ConstructFromAnother(MR.SplineSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SplineSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::SplineSettings`.
    /// This is the non-const half of the class.
    public class SplineSettings : Const_SplineSettings
    {
        internal unsafe SplineSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// additional points will be added between each pair of control points,
        /// until the distance between neighbor points becomes less than this distance
        public new unsafe ref float SamplingStep
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_GetMutable_samplingStep", ExactSpelling = true)]
                extern static float *__MR_SplineSettings_GetMutable_samplingStep(_Underlying *_this);
                return ref *__MR_SplineSettings_GetMutable_samplingStep(_UnderlyingPtr);
            }
        }

        /// a positive value, the more the value the closer resulting spline will be to given control points
        public new unsafe ref float ControlStability
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_GetMutable_controlStability", ExactSpelling = true)]
                extern static float *__MR_SplineSettings_GetMutable_controlStability(_Underlying *_this);
                return ref *__MR_SplineSettings_GetMutable_controlStability(_UnderlyingPtr);
            }
        }

        /// the shape of resulting spline depends on the total number of points in the contour,
        /// which in turn depends on the length of input contour being sampled;
        /// setting iterations greater than one allows you to pass a constructed spline as a better input contour to the next run of the algorithm
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_SplineSettings_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_SplineSettings_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// optional parameter with the normals of input points that will be resampled to become normals of output points
        public new unsafe ref void * Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_GetMutable_normals", ExactSpelling = true)]
                extern static void **__MR_SplineSettings_GetMutable_normals(_Underlying *_this);
                return ref *__MR_SplineSettings_GetMutable_normals(_UnderlyingPtr);
            }
        }

        /// if true and normals are provided, then the curve at marked points will try to be orthogonal to given normal there
        public new unsafe ref bool NormalsAffectShape
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_GetMutable_normalsAffectShape", ExactSpelling = true)]
                extern static bool *__MR_SplineSettings_GetMutable_normalsAffectShape(_Underlying *_this);
                return ref *__MR_SplineSettings_GetMutable_normalsAffectShape(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SplineSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SplineSettings_DefaultConstruct();
        }

        /// Constructs `MR::SplineSettings` elementwise.
        public unsafe SplineSettings(float samplingStep, float controlStability, int iterations, MR.Std.Vector_MRVector3f? normals, bool normalsAffectShape) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_ConstructFrom(float samplingStep, float controlStability, int iterations, MR.Std.Vector_MRVector3f._Underlying *normals, byte normalsAffectShape);
            _UnderlyingPtr = __MR_SplineSettings_ConstructFrom(samplingStep, controlStability, iterations, normals is not null ? normals._UnderlyingPtr : null, normalsAffectShape ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::SplineSettings::SplineSettings`.
        public unsafe SplineSettings(MR.Const_SplineSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_ConstructFromAnother(MR.SplineSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SplineSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SplineSettings::operator=`.
        public unsafe MR.SplineSettings Assign(MR.Const_SplineSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SplineSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SplineSettings._Underlying *__MR_SplineSettings_AssignFromAnother(_Underlying *_this, MR.SplineSettings._Underlying *_other);
            return new(__MR_SplineSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SplineSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SplineSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SplineSettings`/`Const_SplineSettings` directly.
    public class _InOptMut_SplineSettings
    {
        public SplineSettings? Opt;

        public _InOptMut_SplineSettings() {}
        public _InOptMut_SplineSettings(SplineSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SplineSettings(SplineSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SplineSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SplineSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SplineSettings`/`Const_SplineSettings` to pass it to the function.
    public class _InOptConst_SplineSettings
    {
        public Const_SplineSettings? Opt;

        public _InOptConst_SplineSettings() {}
        public _InOptConst_SplineSettings(Const_SplineSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SplineSettings(Const_SplineSettings value) {return new(value);}
    }

    /// Generated from function `MR::isClosed`.
    public static unsafe bool IsClosed(MR.Std.Const_Vector_MRVector3f c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isClosed_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static byte __MR_isClosed_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *c);
        return __MR_isClosed_std_vector_MR_Vector3f(c._UnderlyingPtr) != 0;
    }

    /// \return marked contour with all points from (in) marked
    /// Generated from function `MR::markedContour`.
    public static unsafe MR.Misc._Moved<MR.MarkedContour3f> MarkedContour(MR.Std._ByValue_Vector_MRVector3f in_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_markedContour", ExactSpelling = true)]
        extern static MR.MarkedContour3f._Underlying *__MR_markedContour(MR.Misc._PassBy in__pass_by, MR.Std.Vector_MRVector3f._Underlying *in_);
        return MR.Misc.Move(new MR.MarkedContour3f(__MR_markedContour(in_.PassByMode, in_.Value is not null ? in_.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// \return marked contour with only first and last points from (in) marked
    /// Generated from function `MR::markedFirstLast`.
    public static unsafe MR.Misc._Moved<MR.MarkedContour3f> MarkedFirstLast(MR.Std._ByValue_Vector_MRVector3f in_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_markedFirstLast", ExactSpelling = true)]
        extern static MR.MarkedContour3f._Underlying *__MR_markedFirstLast(MR.Misc._PassBy in__pass_by, MR.Std.Vector_MRVector3f._Underlying *in_);
        return MR.Misc.Move(new MR.MarkedContour3f(__MR_markedFirstLast(in_.PassByMode, in_.Value is not null ? in_.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// keeps all marked points from input contour and adds/removes other points to have them as many as possible,
    /// but at the distance along the input line not shorter than (minStep) from their neighbor points
    /// \param normals optional parameter with the normals of input points that will be resampled to become normals of output points
    /// Generated from function `MR::resample`.
    public static unsafe MR.Misc._Moved<MR.MarkedContour3f> Resample(MR.Const_MarkedContour3f in_, float minStep, MR.Std.Vector_MRVector3f? normals = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_resample", ExactSpelling = true)]
        extern static MR.MarkedContour3f._Underlying *__MR_resample(MR.Const_MarkedContour3f._Underlying *in_, float minStep, MR.Std.Vector_MRVector3f._Underlying *normals);
        return MR.Misc.Move(new MR.MarkedContour3f(__MR_resample(in_._UnderlyingPtr, minStep, normals is not null ? normals._UnderlyingPtr : null), is_owning: true));
    }

    /// \param in input marked contour
    /// \param markStability a positive value, the more the value the closer marked points will be to their original positions
    /// \param normals if provided the curve at marked points will try to be orthogonal to given normal there
    /// \return contour with same number of points and same marked, where each return point tries to be on a smooth curve
    /// Generated from function `MR::makeSpline`.
    /// Parameter `markStability` defaults to `1`.
    public static unsafe MR.Misc._Moved<MR.MarkedContour3f> MakeSpline(MR._ByValue_MarkedContour3f in_, float? markStability = null, MR.Std.Const_Vector_MRVector3f? normals = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSpline_3_float", ExactSpelling = true)]
        extern static MR.MarkedContour3f._Underlying *__MR_makeSpline_3_float(MR.Misc._PassBy in__pass_by, MR.MarkedContour3f._Underlying *in_, float *markStability, MR.Std.Const_Vector_MRVector3f._Underlying *normals);
        float __deref_markStability = markStability.GetValueOrDefault();
        return MR.Misc.Move(new MR.MarkedContour3f(__MR_makeSpline_3_float(in_.PassByMode, in_.Value is not null ? in_.Value._UnderlyingPtr : null, markStability.HasValue ? &__deref_markStability : null, normals is not null ? normals._UnderlyingPtr : null), is_owning: true));
    }

    /// \param in input marked contour
    /// \param normals the curve at marked points will try to be orthogonal to given normal there
    /// \param markStability a positive value, the more the value the closer marked points will be to their original positions
    /// \return contour with same number of points and same marked, where each return point tries to be on a smooth curve
    /// Generated from function `MR::makeSpline`.
    /// Parameter `markStability` defaults to `1`.
    public static unsafe MR.Misc._Moved<MR.MarkedContour3f> MakeSpline(MR._ByValue_MarkedContour3f in_, MR.Std.Const_Vector_MRVector3f normals, float? markStability = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSpline_3_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.MarkedContour3f._Underlying *__MR_makeSpline_3_std_vector_MR_Vector3f(MR.Misc._PassBy in__pass_by, MR.MarkedContour3f._Underlying *in_, MR.Std.Const_Vector_MRVector3f._Underlying *normals, float *markStability);
        float __deref_markStability = markStability.GetValueOrDefault();
        return MR.Misc.Move(new MR.MarkedContour3f(__MR_makeSpline_3_std_vector_MR_Vector3f(in_.PassByMode, in_.Value is not null ? in_.Value._UnderlyingPtr : null, normals._UnderlyingPtr, markStability.HasValue ? &__deref_markStability : null), is_owning: true));
    }

    /// \param controlPoints ordered point the spline to interpolate
    /// \return spline contour with same or more points than initially given, marks field highlights the points corresponding to input ones
    /// Generated from function `MR::makeSpline`.
    public static unsafe MR.Misc._Moved<MR.MarkedContour3f> MakeSpline(MR.Std.Const_Vector_MRVector3f controlPoints, MR.Const_SplineSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSpline_2", ExactSpelling = true)]
        extern static MR.MarkedContour3f._Underlying *__MR_makeSpline_2(MR.Std.Const_Vector_MRVector3f._Underlying *controlPoints, MR.Const_SplineSettings._Underlying *settings);
        return MR.Misc.Move(new MR.MarkedContour3f(__MR_makeSpline_2(controlPoints._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
