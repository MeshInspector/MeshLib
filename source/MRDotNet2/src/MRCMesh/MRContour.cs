public static partial class MR
{
    /// >0 for clockwise loop, < 0 for CCW loop
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcOrientedArea<float, float>`.
    public static unsafe float CalcOrientedArea(MR.Std.Const_Vector_MRVector2f contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcOrientedArea_float_float_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static float __MR_calcOrientedArea_float_float_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *contour);
        return __MR_calcOrientedArea_float_float_std_vector_MR_Vector2f(contour._UnderlyingPtr);
    }

    /// >0 for clockwise loop, < 0 for CCW loop
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcOrientedArea<double, double>`.
    public static unsafe double CalcOrientedArea(MR.Std.Const_Vector_MRVector2d contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcOrientedArea_double_double_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static double __MR_calcOrientedArea_double_double_std_vector_MR_Vector2d(MR.Std.Const_Vector_MRVector2d._Underlying *contour);
        return __MR_calcOrientedArea_double_double_std_vector_MR_Vector2d(contour._UnderlyingPtr);
    }

    /// returns the vector with the magnitude equal to contour area, and directed to see the contour
    /// in ccw order from the vector tip
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcOrientedArea<float, float>`.
    public static unsafe MR.Vector3f CalcOrientedArea(MR.Std.Const_Vector_MRVector3f contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcOrientedArea_float_float_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Vector3f __MR_calcOrientedArea_float_float_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *contour);
        return __MR_calcOrientedArea_float_float_std_vector_MR_Vector3f(contour._UnderlyingPtr);
    }

    /// returns the vector with the magnitude equal to contour area, and directed to see the contour
    /// in ccw order from the vector tip
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcOrientedArea<double, double>`.
    public static unsafe MR.Vector3d CalcOrientedArea(MR.Std.Const_Vector_MRVector3d contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcOrientedArea_double_double_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Vector3d __MR_calcOrientedArea_double_double_std_vector_MR_Vector3d(MR.Std.Const_Vector_MRVector3d._Underlying *contour);
        return __MR_calcOrientedArea_double_double_std_vector_MR_Vector3d(contour._UnderlyingPtr);
    }

    /// returns sum length of the given contour
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcLength<MR::Vector2f, float>`.
    public static unsafe float CalcLength(MR.Std.Const_Vector_MRVector2f contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcLength_MR_Vector2f_float", ExactSpelling = true)]
        extern static float __MR_calcLength_MR_Vector2f_float(MR.Std.Const_Vector_MRVector2f._Underlying *contour);
        return __MR_calcLength_MR_Vector2f_float(contour._UnderlyingPtr);
    }

    /// returns sum length of the given contour
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcLength<MR::Vector2d, double>`.
    public static unsafe double CalcLength(MR.Std.Const_Vector_MRVector2d contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcLength_MR_Vector2d_double", ExactSpelling = true)]
        extern static double __MR_calcLength_MR_Vector2d_double(MR.Std.Const_Vector_MRVector2d._Underlying *contour);
        return __MR_calcLength_MR_Vector2d_double(contour._UnderlyingPtr);
    }

    /// returns sum length of the given contour
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcLength<MR::Vector3f, float>`.
    public static unsafe float CalcLength(MR.Std.Const_Vector_MRVector3f contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcLength_MR_Vector3f_float", ExactSpelling = true)]
        extern static float __MR_calcLength_MR_Vector3f_float(MR.Std.Const_Vector_MRVector3f._Underlying *contour);
        return __MR_calcLength_MR_Vector3f_float(contour._UnderlyingPtr);
    }

    /// returns sum length of the given contour
    /// \tparam R is the type for the accumulation and for result
    /// Generated from function `MR::calcLength<MR::Vector3d, double>`.
    public static unsafe double CalcLength(MR.Std.Const_Vector_MRVector3d contour)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcLength_MR_Vector3d_double", ExactSpelling = true)]
        extern static double __MR_calcLength_MR_Vector3d_double(MR.Std.Const_Vector_MRVector3d._Underlying *contour);
        return __MR_calcLength_MR_Vector3d_double(contour._UnderlyingPtr);
    }

    // Instantiate the templates when generating bindings.
    /// Generated from function `MR::convertContourTo2f<std::vector<MR::Vector2f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2f> ConvertContourTo2f(MR.Std.Const_Vector_MRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2f_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2f._Underlying *__MR_convertContourTo2f_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2f(__MR_convertContourTo2f_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    // Instantiate the templates when generating bindings.
    /// Generated from function `MR::convertContourTo2f<std::vector<MR::Vector2d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2f> ConvertContourTo2f(MR.Std.Const_Vector_MRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2f_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2f._Underlying *__MR_convertContourTo2f_std_vector_MR_Vector2d(MR.Std.Const_Vector_MRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2f(__MR_convertContourTo2f_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    // Instantiate the templates when generating bindings.
    /// Generated from function `MR::convertContourTo2f<std::vector<MR::Vector3f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2f> ConvertContourTo2f(MR.Std.Const_Vector_MRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2f_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2f._Underlying *__MR_convertContourTo2f_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2f(__MR_convertContourTo2f_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    // Instantiate the templates when generating bindings.
    /// Generated from function `MR::convertContourTo2f<std::vector<MR::Vector3d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2f> ConvertContourTo2f(MR.Std.Const_Vector_MRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2f_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2f._Underlying *__MR_convertContourTo2f_std_vector_MR_Vector3d(MR.Std.Const_Vector_MRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2f(__MR_convertContourTo2f_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3f<std::vector<MR::Vector2f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> ConvertContourTo3f(MR.Std.Const_Vector_MRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3f_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_convertContourTo3f_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_convertContourTo3f_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3f<std::vector<MR::Vector2d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> ConvertContourTo3f(MR.Std.Const_Vector_MRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3f_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_convertContourTo3f_std_vector_MR_Vector2d(MR.Std.Const_Vector_MRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_convertContourTo3f_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3f<std::vector<MR::Vector3f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> ConvertContourTo3f(MR.Std.Const_Vector_MRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3f_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_convertContourTo3f_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_convertContourTo3f_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3f<std::vector<MR::Vector3d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> ConvertContourTo3f(MR.Std.Const_Vector_MRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3f_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_convertContourTo3f_std_vector_MR_Vector3d(MR.Std.Const_Vector_MRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_convertContourTo3f_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo2d<std::vector<MR::Vector2f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2d> ConvertContourTo2d(MR.Std.Const_Vector_MRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2d_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2d._Underlying *__MR_convertContourTo2d_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2d(__MR_convertContourTo2d_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo2d<std::vector<MR::Vector2d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2d> ConvertContourTo2d(MR.Std.Const_Vector_MRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2d_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2d._Underlying *__MR_convertContourTo2d_std_vector_MR_Vector2d(MR.Std.Const_Vector_MRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2d(__MR_convertContourTo2d_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo2d<std::vector<MR::Vector3f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2d> ConvertContourTo2d(MR.Std.Const_Vector_MRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2d_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2d._Underlying *__MR_convertContourTo2d_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2d(__MR_convertContourTo2d_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo2d<std::vector<MR::Vector3d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2d> ConvertContourTo2d(MR.Std.Const_Vector_MRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo2d_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2d._Underlying *__MR_convertContourTo2d_std_vector_MR_Vector3d(MR.Std.Const_Vector_MRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2d(__MR_convertContourTo2d_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3d<std::vector<MR::Vector2f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3d> ConvertContourTo3d(MR.Std.Const_Vector_MRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3d_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3d._Underlying *__MR_convertContourTo3d_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3d(__MR_convertContourTo3d_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3d<std::vector<MR::Vector2d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3d> ConvertContourTo3d(MR.Std.Const_Vector_MRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3d_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3d._Underlying *__MR_convertContourTo3d_std_vector_MR_Vector2d(MR.Std.Const_Vector_MRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3d(__MR_convertContourTo3d_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3d<std::vector<MR::Vector3f>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3d> ConvertContourTo3d(MR.Std.Const_Vector_MRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3d_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3d._Underlying *__MR_convertContourTo3d_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3d(__MR_convertContourTo3d_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContourTo3d<std::vector<MR::Vector3d>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3d> ConvertContourTo3d(MR.Std.Const_Vector_MRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContourTo3d_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3d._Underlying *__MR_convertContourTo3d_std_vector_MR_Vector3d(MR.Std.Const_Vector_MRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3d(__MR_convertContourTo3d_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2f<std::vector<std::vector<MR::Vector2f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> ConvertContoursTo2f(MR.Std.Const_Vector_StdVectorMRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2f_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2f<std::vector<std::vector<MR::Vector2d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> ConvertContoursTo2f(MR.Std.Const_Vector_StdVectorMRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2f_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2f<std::vector<std::vector<MR::Vector3f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> ConvertContoursTo2f(MR.Std.Const_Vector_StdVectorMRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2f_std_vector_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector3f(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2f<std::vector<std::vector<MR::Vector3d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> ConvertContoursTo2f(MR.Std.Const_Vector_StdVectorMRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2f_std_vector_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector3d(MR.Std.Const_Vector_StdVectorMRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_convertContoursTo2f_std_vector_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3f<std::vector<std::vector<MR::Vector2f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> ConvertContoursTo3f(MR.Std.Const_Vector_StdVectorMRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3f_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3f<std::vector<std::vector<MR::Vector2d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> ConvertContoursTo3f(MR.Std.Const_Vector_StdVectorMRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3f_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3f<std::vector<std::vector<MR::Vector3f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> ConvertContoursTo3f(MR.Std.Const_Vector_StdVectorMRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3f_std_vector_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector3f(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3f<std::vector<std::vector<MR::Vector3d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> ConvertContoursTo3f(MR.Std.Const_Vector_StdVectorMRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3f_std_vector_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector3d(MR.Std.Const_Vector_StdVectorMRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_convertContoursTo3f_std_vector_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2d<std::vector<std::vector<MR::Vector2f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2d> ConvertContoursTo2d(MR.Std.Const_Vector_StdVectorMRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2d_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2d._Underlying *__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2d(__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2d<std::vector<std::vector<MR::Vector2d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2d> ConvertContoursTo2d(MR.Std.Const_Vector_StdVectorMRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2d_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2d._Underlying *__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2d(__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2d<std::vector<std::vector<MR::Vector3f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2d> ConvertContoursTo2d(MR.Std.Const_Vector_StdVectorMRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2d_std_vector_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2d._Underlying *__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector3f(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2d(__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo2d<std::vector<std::vector<MR::Vector3d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2d> ConvertContoursTo2d(MR.Std.Const_Vector_StdVectorMRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo2d_std_vector_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector2d._Underlying *__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector3d(MR.Std.Const_Vector_StdVectorMRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2d(__MR_convertContoursTo2d_std_vector_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3d<std::vector<std::vector<MR::Vector2f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3d> ConvertContoursTo3d(MR.Std.Const_Vector_StdVectorMRVector2f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3d_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3d._Underlying *__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3d(__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector2f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3d<std::vector<std::vector<MR::Vector2d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3d> ConvertContoursTo3d(MR.Std.Const_Vector_StdVectorMRVector2d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3d_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3d._Underlying *__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3d(__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector2d(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3d<std::vector<std::vector<MR::Vector3f>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3d> ConvertContoursTo3d(MR.Std.Const_Vector_StdVectorMRVector3f from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3d_std_vector_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3d._Underlying *__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector3f(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3d(__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector3f(from._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::convertContoursTo3d<std::vector<std::vector<MR::Vector3d>>>`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3d> ConvertContoursTo3d(MR.Std.Const_Vector_StdVectorMRVector3d from)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertContoursTo3d_std_vector_std_vector_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMRVector3d._Underlying *__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector3d(MR.Std.Const_Vector_StdVectorMRVector3d._Underlying *from);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3d(__MR_convertContoursTo3d_std_vector_std_vector_MR_Vector3d(from._UnderlyingPtr), is_owning: true));
    }
}
