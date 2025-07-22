#pragma once

#include "MRMeshFwd.h"

#include "MRPch/MRSuppressWarning.h"

MR_SUPPRESS_WARNING_PUSH
#if defined( _MSC_VER )
#pragma warning(disable:5054)  //operator '&': deprecated between enumerations of different types
#pragma warning(disable:4127)  //C4127. "Consider using 'if constexpr' statement instead"
#elif defined( __apple_build_version__ )
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined( __clang__ )
#elif defined( __GNUC__ )
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Eigenvalues>
MR_SUPPRESS_WARNING_POP

// https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf

namespace MR
{

template <typename T>
class Cylinder3Approximation
{
private:
    enum class CylinderFitterType
    {
        // The algorithm implimentation needs an initial approximation to refine the cylinder axis.
        // In this option, we sort through several possible options distributed over the hemisphere.
        HemisphereSearchFit,

        // In this case, we assume that there is an external estimate for the cylinder axis.
        // Therefore, we specify only the position that is given from the outside
        SpecificAxisFit

        // TODO for Meshes try to impliment specific algorithm from https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
        // TODO Also, an estimate of the cylinder axis can be obtained by the gravel component method or the like. But this requires additional. experiments.
        // TODO for future try eigen vector covariance   https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    };

    CylinderFitterType fitter_ = CylinderFitterType::HemisphereSearchFit;

    //  CylinderFitterType::SpecificAxisFit params
    Eigen::Vector<T, 3> baseCylinderAxis_;

    // CylinderFitterType::HemisphereSearchFit params
    size_t thetaResolution_ = 0;
    size_t phiResolution_ = 0;
    bool isMultithread_ = true;

    //Input data converted to Eigen format and normalized to the avgPoint position of all points in the cloud.
    std::vector<Eigen::Vector<T, 3>> normalizedPoints_ = {};

    // Precalculated values for speed up.
    // In https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf page 35-36:
    // Text below is a direct copy from pdf file:
    // The sample application that used equation (94) directly was really slow.
    // On an Intel CoreTM i7-6700 CPU at 3.40 GHz, the single - threaded version for 10765 points required 129 seconds
    // and the multithreaded version using 8 hyperthreads required 22 seconds.The evaluation of G using the precomputed summations is much faster.
    // The single - threaded version required 85 milliseconds and the multithreaded version using 8 hyperthreads required 22 milliseconds.

    Eigen::Vector <T, 6>  precomputedMu_ = {};
    Eigen::Matrix <T, 3, 3>  precomputedF0_ = {};
    Eigen::Matrix <T, 3, 6>  precomputedF1_ = {};
    Eigen::Matrix <T, 6, 6>  precomputedF2_ = {};

public:
    MRMESH_API Cylinder3Approximation();

    MRMESH_API void reset();

    // Solver for CylinderFitterType::HemisphereSearchFit type
    // thetaResolution_, phiResolution_  must be positive and as large as it posible. Price is CPU time. (~ 100 gives good results).
    MRMESH_API T solveGeneral( const std::vector<MR::Vector3<T>>& points, Cylinder3<T>& cylinder, size_t theta = 180, size_t phi = 90, bool isMultithread = true );

    // Solver for CylinderFitterType::SpecificAxisFit type
    // Simplet way in case of we already know clinder axis
    MRMESH_API T solveSpecificAxis( const std::vector<MR::Vector3<T>>& points, Cylinder3<T>& cylinder, MR::Vector3<T> const& cylinderAxis );

private:
    // main solver.
    T solve( const std::vector<MR::Vector3<T>>& points, Cylinder3<T>& cylinder );

    void updatePrecomputeParams( const std::vector <MR::Vector3<T>>& points, Vector3<T>& average );

    // Core minimization function.
    // Functional that needs to be minimized to obtain the optimal value of W (i.e. the cylinder axis)
    // General definition is formula 94, but for speed up we use formula 99.
    // https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
    T G( const Eigen::Vector<T, 3>& W, Eigen::Vector<T, 3>& PC, T& rsqr ) const;

    T fitCylindeHemisphereSingleThreaded( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare ) const;

    class BestHemisphereStoredData
    {
    public:
        T error = std::numeric_limits<T>::max();
        T rootSquare = std::numeric_limits<T>::max();
        Eigen::Vector<T, 3> W;
        Eigen::Vector<T, 3> PC;
    };

    T fitCylindeHemisphereMultiThreaded( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare ) const;

    T SpecificAxisFit( Eigen::Vector<T, 3>& PC, Eigen::Vector<T, 3>& W, T& resultedRootSquare );
};

}
