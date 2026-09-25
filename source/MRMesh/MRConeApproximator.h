#pragma once

#include "MRMeshFwd.h"
#include "MRCone3.h"
#include "MRPch/MREigenCore.h"

namespace MR
{

enum class ConeFitterType
{
    ApproximationPCM, // approximation of cone axis by principal component method
    HemisphereSearchFit,
    SpecificAxisFit
};

struct Cone3ApproximationParams {
    int levenbergMarquardtMaxIteration = 40;
    ConeFitterType coneFitterType = ConeFitterType::HemisphereSearchFit;
    int hemisphereSearchPhiResolution = 30;
    int hemisphereSearchThetaResolution = 30;
};

// Class for approximation cloud point by cone.
// We will calculate the initial approximation of the cone and then use a minimizer to refine the parameters.
// minimizer is LevenbergMarquardt now.
// TODO: Possible we could add GaussNewton in future.
template <typename T>
class Cone3Approximation
{
public:

    Cone3Approximation() = default;

    // returns RMS for original points
    MRMESH_API T solve( const std::vector<MR::Vector3<T>>& points,
        Cone3<T>& cone, const Cone3ApproximationParams& params = {} );

private:

    // cone fitter main params
    Cone3ApproximationParams params_;

    // solver for single axis case.
    T solveFixedAxis_( const std::vector<MR::Vector3<T>>& points,
        Cone3<T>& cone, bool useConeInputAsInitialGuess = false );

    T solveApproximationPCM_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone );

    T solveSpecificAxisFit_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone );

    // brute force solver across hole hemisphere for cone axis original extimation.
    T solveHemisphereSearchFit_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone );

    // Calculate and return a length of cone based on set of initil points and inifinite cone surface given by cone param.
    T calculateConeHeight_( const std::vector<MR::Vector3<T>>& points, Cone3<T>& cone );

    T getApproximationRMS_( const std::vector<MR::Vector3<T>>& points, const Cone3<T>& cone );

    MR::Vector3<T> computeCenter_( const std::vector<MR::Vector3<T>>& points );

    void computeCenterAndNormal_( const std::vector<MR::Vector3<T>>& points, MR::Vector3<T>& center, MR::Vector3<T>& U );

    // Calculates the initial parameters of the cone, which will later be used for minimization.
    Cone3<T> computeInitialCone_( const std::vector<MR::Vector3<T>>& points, const MR::Vector3<T>& center, const MR::Vector3<T>& axis );

    // Function for finding the best approximation of a straight line in general form y = a*x + b
    void findBestFitLine_( const std::vector<MR::Vector2<T>>& xyPairs, T& lineA, T& lineB, MR::Vector2<T>* avg = nullptr );

    // Convert data from Eigen minimizator representation to cone params.
    void fitParamsToCone_( Eigen::Vector<T, Eigen::Dynamic>& fittedParams, Cone3<T>& cone );

    // Convert data from cone params to Eigen minimizator representation.
    void coneToFitParams_( Cone3<T>& cone, Eigen::Vector<T, Eigen::Dynamic>& fittedParams );
};

}
