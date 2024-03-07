#include "MRRenderMeasurementsObject.h"

#include "MRPch/MRFmt.h"
#include "MRViewer/MRImGuiMeasurementIndicators.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"

namespace MR
{

constexpr int cCurveMaxSubdivisionDepth = 10;

// How many digits (after the decimal point) to print for distances.
static int cDistanceDigits = 3;

// How many digits (after the decimal point) to print for angles (in degrees).
static int cAngleDegreeDigits = 1;

[[nodiscard]] static ImVec2 toScreenCoords( const Viewport& viewport, const Vector3f& point )
{
    auto rect = viewport.getViewportRect();
    Vector3f result = viewport.projectToViewportSpace( point );
    return ImVec2( result.x, result.y ) + ImVec2( rect.min.x, ImGui::GetIO().DisplaySize.y - rect.max.y );
}

void RenderMeasurementsObject::RadiusTask::renderPass()
{
    Vector3f worldRadiusVec = params.radiusAsVector;
    float radiusValue = worldRadiusVec.length();

    // Rotate the original radius vector to look better from the current viewing angle.
    if ( params.vis.isSpherical )
    {
        worldRadiusVec -= dot( worldRadiusVec, dirTowardsCamera ) * dirTowardsCamera;
        worldRadiusVec = worldRadiusVec.normalized() * radiusValue;
    }
    else
    {
        // `getViewXf().A.z` is the direction towards the camera.
        Vector3f c = cross( dirTowardsCamera, params.normal );
        // I hoped this would fix the excessive spinning of the arrows, but this causes them to jump sometimes, so I'm keeping this disabled for now.
        // if ( dot( worldRadiusVec, c ) < 0 )
        //     c = -c;
        float t = std::asin( std::min( 1.f, c.length() ) ) / ( MR::PI_F / 2 );
        worldRadiusVec = ( worldRadiusVec * ( ( 1 - t ) / radiusValue ) + c.normalized() * t ).normalized() * radiusValue;
    }

    #if 0 // Alternative rendering by drawing a diameter line across the circle.
    ImVec2 a = toScreenCoords( radius->getWorldCenter() - worldRadiusVec );
    ImVec2 b = toScreenCoords( radius->getWorldCenter() + worldRadiusVec );

    ImGuiMeasurementIndicators::distance( ImGuiMeasurementIndicators::Element::both, menuScaling, {},
        a, b, {
            ImGuiMeasurementIndicators::StringIcon::diameter, std::size_t( params.vis.isSpherical ),
            fmt::format( "{}  {:.{}f}", params.vis.isSpherical ? "S" : "", radiusValue, cDistanceDigits ),
        },
        {
            .moveTextToLineEndIndex = true,
        }
    );
    #else
    ImVec2 center = toScreenCoords( *viewport, params.center );
    ImVec2 point = toScreenCoords( *viewport, params.center + worldRadiusVec );

    float lengthMultiplier = params.vis.visualLengthMultiplier;
    ImVec2 farPoint = toScreenCoords( *viewport, params.center + worldRadiusVec * ( 1 + lengthMultiplier ) );

    float minRadiusLen = 32 * menuScaling;

    if ( ImGuiMath::lengthSq( farPoint - point ) < minRadiusLen * minRadiusLen )
        farPoint = point + ImGuiMath::normalize( point - center ) * minRadiusLen;

    ImGuiMeasurementIndicators::StringWithIcon string = fmt::format(
        "{}{}  {:.{}f}",
        params.vis.isSpherical ? "S" : "",
        params.vis.drawAsDiameter ? "" : "R",
        radiusValue * ( params.vis.drawAsDiameter ? 2 : 1 ), cDistanceDigits
    );
    if ( params.vis.drawAsDiameter )
    {
        string.icon = ImGuiMeasurementIndicators::StringIcon::diameter;
        string.iconPos = params.vis.isSpherical ? 1 : 0;
    }

    ImGuiMeasurementIndicators::line( ImGuiMeasurementIndicators::Element::both, menuScaling, {},
        farPoint, point, {
            .capA = { .text = string },
            .capB = { .decoration = ImGuiMeasurementIndicators::LineCap::Decoration::arrow },
        }
    );
    #endif
}

void RenderMeasurementsObject::AngleTask::renderPass()
{
    // It would be nice to reuse this buffer between all the curves in the scene...
    std::vector<ImVec2> pointBuffer;

    Vector3f referenceVector = params.rays[0];
    Vector3f targetVector = params.rays[1];

    Vector3f rotationAxis = cross( referenceVector, targetVector );

    if ( rotationAxis != Vector3f{} )
    {
        const ImGuiMeasurementIndicators::Params indicatorParams;
        const ImGuiMeasurementIndicators::CurveParams curveParams{ .maxSubdivisionDepth = cCurveMaxSubdivisionDepth };

        float totalLenThreshold = indicatorParams.totalLenThreshold * menuScaling;
        float invertedOverhang = indicatorParams.invertedOverhang * menuScaling;
        float notchHalfLen = indicatorParams.notchHalfLen * menuScaling;

        // Tweak the rays for cones to make them face the camera.
        if ( params.vis.isConical )
        {
            Vector3f coneAxis = ( referenceVector.normalized() + targetVector.normalized() ).normalized();

            Vector3f vectorA = ( dirTowardsCamera - coneAxis * dot( dirTowardsCamera, coneAxis ) ).normalized();
            Vector3f vectorB = rotationAxis.normalized();

            Vector3f coneAxisSigned = cross( vectorA, vectorB );

            auto matrix = Matrix3f::rotation( coneAxisSigned, std::asin( std::min( 1.f, coneAxisSigned.length() ) ) * ( dot( vectorA, vectorB ) < 0 ? 1.f : -1.f ) );
            referenceVector = matrix * referenceVector;
            targetVector = matrix * targetVector;
            rotationAxis = matrix * rotationAxis;
        }

        Vector3f worldCenterPoint = params.center;

        ImVec2 screenCenterPoint = toScreenCoords( *viewport, worldCenterPoint );

        // Make the two vectors have the same length, the shorter of the two.
        float referenceVectorLength = 0;
        if ( referenceVector.lengthSq() > targetVector.lengthSq() )
            referenceVector = referenceVector.normalized() * ( referenceVectorLength = targetVector.length() );
        else
            targetVector = targetVector.normalized() * ( referenceVectorLength = referenceVector.length() );

        float angleValue = std::acos( std::clamp( dot( params.rays[0].normalized(), params.rays[1].normalized() ), -1.f, 1.f ) );

        // This data is used when subdividing the curve at a certain depth.
        struct DepthStepData
        {
            float angleStep = 0;
            Matrix3f matrix;
        };
        // Data per depth.
        DepthStepData depthStepData[cCurveMaxSubdivisionDepth];
        int numGeneratedDepthSteps = 0;

        // Pre-bake the first element, to set the first value of `angleStep`.
        depthStepData[0].angleStep = angleValue / 2;
        depthStepData[0].matrix = Matrix3f::rotation( rotationAxis, depthStepData[0].angleStep );
        numGeneratedDepthSteps++;

        // Each point in the curve can be represented as one of:
        // * `BaseVector` (one of the two base rays),
        // * `MidpointVector` (the ray before them), or
        // * `Vector3f` (any other ray).
        // This polymorphism happens at compile-time, and lets us snatch the midpoint from the curve subdivision algorithm, when it gets produced.
        // We need to know the midpoint and its neighbors in the curve to correctly render the text label.

        struct BaseVector
        {
            Vector3f vector;
            operator Vector3f() const {return vector;}
        };
        struct MidpointVector
        {
            Vector3f vector;
            operator Vector3f() const {return vector;}
        };

        auto getCurvePoint = [&]( const Vector3f& vector ) -> ImVec2
        {
            return toScreenCoords( *viewport, worldCenterPoint + vector );
        };

        auto bisectVectorLow = [&]( const Vector3f& a, const Vector3f& b, int depth ) -> Vector3f
        {
            (void)b;
            if ( numGeneratedDepthSteps <=/*sic*/ depth )
            {
                #if __GNUC__ == 11 || __GNUC__ == 12
                #pragma GCC diagnostic push
                // A false positive, it seems? `bisectState()` ensures `depths < curveMaxSubdivisionDepth`.
                // And `numGEneratedDepthSteps` starts from 1, it can't be zero either.
                #pragma GCC diagnostic ignored "-Warray-bounds"
                #endif
                depthStepData[numGeneratedDepthSteps].angleStep = depthStepData[numGeneratedDepthSteps - 1].angleStep / 2;
                #if __GNUC__ == 11 || __GNUC__ == 12
                #pragma GCC diagnostic pop
                #endif
                depthStepData[numGeneratedDepthSteps].matrix = Matrix3f::rotation( rotationAxis, depthStepData[numGeneratedDepthSteps].angleStep );
                numGeneratedDepthSteps++;
            }
            return depthStepData[depth].matrix * a;
        };
        auto bisectVector = overloaded{
            bisectVectorLow,
            [&]( const BaseVector& a, const BaseVector& b, int depth ) -> MidpointVector
            {
                return { bisectVectorLow( a, b, depth ) };
            },
        };

        std::size_t midpointIndex = std::size_t( -1 );
        auto onInsertPoint = [&]<typename T>( ImVec2 point, const T& state ) -> void
        {
            (void)point;
            (void)state;
            if constexpr ( std::is_same_v<T, MidpointVector> )
                midpointIndex = pointBuffer.size();
        };

        pointBuffer.clear();
        if ( pointBuffer.capacity() == 0 )
            pointBuffer.reserve( 512 ); // This should be large enough for most cases.

        auto curve = ImGuiMeasurementIndicators::prepareCurve( curveParams, pointBuffer, BaseVector{ referenceVector }, BaseVector{ targetVector },
            getCurvePoint, bisectVector, onInsertPoint
        );

        ImVec2 textPos;
        ImVec2 normal;
        bool useInvertedStyle = ImGuiMath::lengthSq( curve.b - curve.a ) < totalLenThreshold * totalLenThreshold;
        if ( midpointIndex >= pointBuffer.size() )
        {
            // Bad midpoint, fall back to calculating the text position from the two base points.
            textPos = ( curve.a + curve.b ) / 2;
            normal = ImGuiMath::rot90( ImGuiMath::normalize( curve.b - curve.a ) );
        }
        else
        {
            // Midpoint is good.
            textPos = pointBuffer[midpointIndex];

            // Check if the curve is long enough to not be drawn inverted...
            useInvertedStyle = useInvertedStyle &&
                ImGuiMath::lengthSq( curve.a - textPos ) < totalLenThreshold * totalLenThreshold &&
                ImGuiMath::lengthSq( curve.b - textPos ) < totalLenThreshold * totalLenThreshold;

            // Now try to probe adjacent points for the normal...

            bool haveNormal = false;
            if ( midpointIndex > 0 )
            {
                haveNormal = true;
                normal += ImGuiMath::rot90( textPos - pointBuffer[midpointIndex - 1] );
            }
            if ( midpointIndex + 1 < pointBuffer.size() )
            {
                haveNormal = true;
                normal += ImGuiMath::rot90( pointBuffer[midpointIndex + 1] - textPos );
            }
            if ( !haveNormal )
                normal = ImGuiMath::rot90( curve.b - curve.a );

            normal = ImGuiMath::normalize( normal );
        }
        if ( ImGuiMath::dot( normal, textPos - screenCenterPoint ) < 0 )
            normal = -normal;

        ImGuiMeasurementIndicators::LineParams lineParams;
        lineParams.capA.decoration = ImGuiMeasurementIndicators::LineCap::Decoration::arrow;
        lineParams.capB.decoration = ImGuiMeasurementIndicators::LineCap::Decoration::arrow;
        lineParams.midPoints = curve.midPoints;

        ImVec2 invertedStartA, invertedStartB;
        if ( useInvertedStyle )
        {
            lineParams.flags |= ImGuiMeasurementIndicators::LineFlags::narrow;
            lineParams.capA.decoration = {};
            lineParams.capB.decoration = {};

            ImVec2 dirA = ImGuiMath::normalize( curve.a - ( curve.midPoints.empty() ? curve.b : curve.midPoints.front() ) );
            ImVec2 dirB = ImGuiMath::normalize( curve.b - ( curve.midPoints.empty() ? curve.a : curve.midPoints.back() ) );

            invertedStartA = curve.a + dirA * invertedOverhang;
            invertedStartB = curve.b + dirB * invertedOverhang;
        }

        auto drawElem = [&]( ImGuiMeasurementIndicators::Element elem )
        {
            // The main curve.
            ImGuiMeasurementIndicators::line( elem, menuScaling, indicatorParams, curve.a, curve.b, lineParams );

            // Inverted arrows.
            if ( useInvertedStyle )
            {
                ImGuiMeasurementIndicators::LineParams invArrowParams{ .capB = { .decoration = ImGuiMeasurementIndicators::LineCap::Decoration::arrow } };
                ImGuiMeasurementIndicators::line( elem, menuScaling, indicatorParams, invertedStartA, curve.a, invArrowParams );
                ImGuiMeasurementIndicators::line( elem, menuScaling, indicatorParams, invertedStartB, curve.b, invArrowParams );
            }

            { // The notches at the arrow tips, optionally extended to the center point.
                ImVec2 offsets[2];

                for ( bool index : { false, true } )
                {
                    ImVec2& offset = offsets[index];
                    offset = curve.endPoint( index ) - screenCenterPoint;
                    float len = ImGuiMath::length( offset );
                    if ( len > 0 )
                        offset /= len;

                    if ( params.vis.shouldVisualizeRay[index] )
                        offset *= len / 3.f; // An arbitrary factor.
                    else
                        offset *= notchHalfLen;
                }

                if ( params.vis.shouldVisualizeRay[0] && params.vis.shouldVisualizeRay[1] )
                {
                    ImGuiMeasurementIndicators::line( elem, menuScaling, indicatorParams, curve.a + offsets[0], curve.b + offsets[1], { .midPoints = { &screenCenterPoint, 1 } } );
                }
                else
                {
                    for ( bool index : { false, true } )
                    {
                        ImGuiMeasurementIndicators::line( elem, menuScaling, indicatorParams,
                            curve.endPoint( index ) + offsets[index],
                            params.vis.shouldVisualizeRay[index] ? screenCenterPoint : curve.endPoint( index ) - offsets[index]
                        );
                    }
                }
            }

            // The text.
            ImGuiMeasurementIndicators::text( elem, menuScaling, indicatorParams,
                // This is `U+00B0 DEGREE SIGN`.
                textPos, fmt::format( "{:.{}f}\xC2\xB0", angleValue * 180 / MR::PI_F, cAngleDegreeDigits ),
                normal
            );
        };

        drawElem( ImGuiMeasurementIndicators::Element::outline );
        drawElem( ImGuiMeasurementIndicators::Element::main );
    }
}

void RenderMeasurementsObject::LengthTask::renderPass()
{
    ImVec2 a = toScreenCoords( *viewport, params.points[0] );
    ImVec2 b = toScreenCoords( *viewport, params.points[1] );
    float distanceValue = ( params.points[1] - params.points[0] ).length() * ( params.vis.drawAsNegative ? -1.f : 1.f );

    ImGuiMeasurementIndicators::distance( ImGuiMeasurementIndicators::Element::both, menuScaling, {},
        a, b, fmt::format( "{:.{}f}", distanceValue, cDistanceDigits )
    );
}

RenderMeasurementsObject::RenderMeasurementsObject( const VisualObject& object )
    : object( &object ), interface( dynamic_cast<const IObjectWithMeasurements *>( &object ) )
{}

void RenderMeasurementsObject::renderUi( const UiRenderParams& params )
{
    if ( !interface )
        return; // This object has no measurements.

    bool first = true;
    AffineXf3f worldXf;
    Vector3f dirTowardsCamera;

    auto lambda = [&]<MeasurementPropertyEnum Kind, typename TaskType>()
    {
        for ( std::size_t index = 0; auto ptr = object->getVisualizePropertyMaskOpt( Kind( index ) ); index++ )
        {
            if ( !ptr->contains( params.viewportId ) )
                continue; // Disabled in this viewport.


            auto newTask = std::make_shared<TaskType>();
            interface->getMeasurementParameters( index, newTask->params );

            newTask->viewport = &getViewerInstance().viewport( params.viewportId );

            if ( first )
            {
                first = false;
                worldXf = object->worldXf();
                dirTowardsCamera = newTask->viewport->getViewXf().A.z.normalized();
            }

            newTask->menuScaling = params.scale;
            newTask->dirTowardsCamera = dirTowardsCamera;


            // Convert everything to world coordinates, and compute depth.

            if constexpr ( std::is_same_v<Kind, RadiusVisualizePropertyType> )
            {
                newTask->params.center = worldXf( newTask->params.center );
                newTask->params.radiusAsVector = worldXf.A * newTask->params.radiusAsVector;
                newTask->params.normal = worldXf.A * newTask->params.normal;

                newTask->renderTaskDepth = getViewerInstance().viewport( params.viewportId ).projectToViewportSpace( newTask->params.center ).z;
            }
            else if constexpr ( std::is_same_v<Kind, AngleVisualizePropertyType> )
            {
                newTask->params.center = worldXf( newTask->params.center );

                float rayLengths[2]{};
                Vector3f depthRefPoint{};
                for ( bool i : { false, true } )
                {
                    newTask->params.rays[i] = worldXf.A * newTask->params.rays[i];
                    rayLengths[i] = newTask->params.rays[i].length();
                    if ( rayLengths[i] > 0 )
                        depthRefPoint += newTask->params.rays[i] / rayLengths[i];
                }

                depthRefPoint *= std::min( rayLengths[0], rayLengths[1] );
                depthRefPoint += newTask->params.center;

                newTask->renderTaskDepth = getViewerInstance().viewport( params.viewportId ).projectToViewportSpace( depthRefPoint ).z;
            }
            else if constexpr ( std::is_same_v<Kind, LengthVisualizePropertyType> )
            {
                for ( bool i : { false, true } )
                    newTask->params.points[i] = worldXf( newTask->params.points[i] );

                Vector3f depthRefPoint = newTask->params.points[0] + ( newTask->params.points[1] - newTask->params.points[0] ) / 2.f;
                newTask->renderTaskDepth = getViewerInstance().viewport( params.viewportId ).projectToViewportSpace( depthRefPoint ).z;
            }
            else
            {
                static_assert( dependent_false<Kind>, "Don't know how to visualize this measurement!" );
            }

            params.tasks->push_back( std::move( newTask ) );
        }
    };

    lambda.operator()<RadiusVisualizePropertyType, RadiusTask>();
    lambda.operator()<AngleVisualizePropertyType, AngleTask>();
    lambda.operator()<LengthVisualizePropertyType, LengthTask>();
}

}
