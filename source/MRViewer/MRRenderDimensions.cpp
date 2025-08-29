#include "MRRenderDimensions.h"

#include "MRMesh/MRSceneColors.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRImGuiMeasurementIndicators.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRRibbonFontManager.h"
#include "MRViewer/MRUnits.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"

namespace MR::RenderDimensions
{

constexpr int cCurveMaxSubdivisionDepth = 10;

static std::string lengthToString( float value )
{
    return valueToString<LengthUnit>( value, { .unitSuffix = false, .style = NumberStyle::normal, .stripTrailingZeroes = false } );
}
// `dir == 0` - symmetric, `dir > 0` - positive, `dir < 0` - negative.
static std::string lengthToleranceToString( float value, int dir )
{
    return valueToString<LengthUnit>( value, { .unitSuffix = false, .style = NumberStyle::normal, .plusSign = dir != 0, .zeroMode = dir >= 0 ? ZeroMode::alwaysPositive : ZeroMode::alwaysNegative, .stripTrailingZeroes = true } );
}
static std::string angleToString( float value )
{
    return valueToString<AngleUnit>( value, { .style = NumberStyle::normal, .stripTrailingZeroes = false } );
}

[[nodiscard]] static ImVec2 toScreenCoords( const Viewport& viewport, const Vector3f& point )
{
    auto rect = viewport.getViewportRect();
    Vector3f result = viewport.projectToViewportSpace( point );
    return ImVec2( result.x, result.y ) + ImVec2( rect.min.x, ImGui::GetIO().DisplaySize.y - rect.max.y );
}

RadiusTask::RadiusTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const RadiusParams& params )
    : menuScaling_( uiParams.scale ), viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ), params_( params )
{
    params_.center = xf( params_.center );
    params_.radiusAsVector = xf.A * params_.radiusAsVector;
    params_.normal = ( xf.A * params_.normal ).normalized();

    Vector3f depthRefPoint = params_.center;
    if ( !params_.drawAsDiameter )
        depthRefPoint += params_.radiusAsVector * ( 1 + params_.visualLengthMultiplier );
    renderTaskDepth = viewport_->projectToViewportSpace( depthRefPoint ).z;
}

void RadiusTask::renderPass()
{
    const Vector3f dirTowardsCamera = viewport_->getViewXf().A.z.normalized();

    Vector3f worldRadiusVec = params_.radiusAsVector;
    float radiusValue = worldRadiusVec.length();

    // Rotate the original radius vector to look better from the current viewing angle.
    if ( params_.isSpherical )
    {
        worldRadiusVec -= dot( worldRadiusVec, dirTowardsCamera ) * dirTowardsCamera;
        worldRadiusVec = worldRadiusVec.normalized() * radiusValue;
    }
    else
    {
        // `getViewXf().A.z` is the direction towards the camera.
        Vector3f c = cross( dirTowardsCamera, params_.normal );
        // I hoped this would fix the excessive spinning of the arrows, but this causes them to jump sometimes, so I'm keeping this disabled for now.
        // if ( dot( worldRadiusVec, c ) < 0 )
        //     c = -c;
        float t = std::asin( std::min( 1.f, c.length() ) ) / ( MR::PI_F / 2 );
        worldRadiusVec = ( worldRadiusVec * ( ( 1 - t ) / radiusValue ) + c.normalized() * t ).normalized() * radiusValue;
    }

    ImGuiMeasurementIndicators::Params indicatorParams;
    indicatorParams.colorMain = color_;

    #if 0 // Alternative rendering by drawing a diameter line across the circle. This is outdated.
    ImVec2 a = toScreenCoords( radius->getWorldCenter() - worldRadiusVec );
    ImVec2 b = toScreenCoords( radius->getWorldCenter() + worldRadiusVec );

    ImGuiMeasurementIndicators::distance( ImGuiMeasurementIndicators::Element::both, menuScaling, indicatorParams,
        a, b, {
            ImGuiMeasurementIndicators::TextIcon::diameter, std::size_t( params_.isSpherical ),
            fmt::format( "{}  {:.{}f}", params_.isSpherical ? "S" : "", radiusValue, cDistanceDigits ),
        },
        {
            .moveTextToLineEndIndex = true,
        }
    );
    #else
    ImVec2 center = toScreenCoords( *viewport_, params_.center );
    ImVec2 point = toScreenCoords( *viewport_, params_.center + worldRadiusVec );

    float lengthMultiplier = params_.visualLengthMultiplier;
    ImVec2 farPoint = toScreenCoords( *viewport_, params_.center + worldRadiusVec * ( 1 + lengthMultiplier ) );

    float minRadiusLen = 32 * menuScaling_;

    if ( ImGuiMath::lengthSq( farPoint - point ) < minRadiusLen * minRadiusLen )
        farPoint = point + ImGuiMath::normalize( point - center ) * minRadiusLen;

    ImGuiMeasurementIndicators::Text string;

    if ( params_.isSpherical )
        string.addText( "S" );

    if ( params_.drawAsDiameter )
        string.add( ImGuiMeasurementIndicators::TextIcon::diameter );
    else
        string.addText( "R" );

    string.addText( "  " );
    string.addText( lengthToString( radiusValue * ( params_.drawAsDiameter ? 2 : 1 ) ) );

    ImGuiMeasurementIndicators::line( ImGuiMeasurementIndicators::Element::both, menuScaling_, indicatorParams,
        farPoint, point, {
            .capA = { .text = string },
            .capB = { .decoration = ImGuiMeasurementIndicators::LineCap::Decoration::arrow },
        }
    );
    #endif
}

AngleTask::AngleTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const AngleParams& params )
    : menuScaling_( uiParams.scale ), viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ),params_( params )
{
    params_.center = xf( params_.center );
    for ( std::size_t i = 0; i < 2; i++ )
        params_.rays[i] = xf.A * params_.rays[i];

    // Estimate the depth.
    float rayLengths[2]{};
    Vector3f depthRefPoint{};
    for ( bool i : { false, true } )
    {
        rayLengths[i] = params_.rays[i].length();
        if ( rayLengths[i] > 0 )
            depthRefPoint += params_.rays[i] / rayLengths[i];
    }
    depthRefPoint *= std::min( rayLengths[0], rayLengths[1] );
    depthRefPoint += params_.center;
    renderTaskDepth = viewport_->projectToViewportSpace( depthRefPoint ).z;
}

void AngleTask::renderPass()
{
    // It would be nice to reuse this buffer between all the curves in the scene...
    std::vector<ImVec2> pointBuffer;

    Vector3f referenceVector = params_.rays[0];
    Vector3f targetVector = params_.rays[1];

    Vector3f rotationAxis = cross( referenceVector, targetVector );

    if ( rotationAxis != Vector3f{} )
    {
        const Vector3f dirTowardsCamera = viewport_->getViewXf().A.z.normalized();

        ImGuiMeasurementIndicators::Params indicatorParams;
        indicatorParams.colorMain = color_;
        const ImGuiMeasurementIndicators::CurveParams curveParams{ .maxSubdivisionDepth = cCurveMaxSubdivisionDepth };

        float totalLenThreshold = indicatorParams.totalLenThreshold * menuScaling_;
        float invertedOverhang = indicatorParams.invertedOverhang * menuScaling_;
        float notchHalfLen = indicatorParams.notchHalfLen * menuScaling_;

        // Tweak the rays for cones to make them face the camera.
        if ( params_.isConical )
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

        Vector3f worldCenterPoint = params_.center;

        ImVec2 screenCenterPoint = toScreenCoords( *viewport_, worldCenterPoint );

        // Make the two vectors have the same length, the shorter of the two.
        float referenceVectorLength = 0;
        if ( referenceVector.lengthSq() > targetVector.lengthSq() )
            referenceVector = referenceVector.normalized() * ( referenceVectorLength = targetVector.length() );
        else
            targetVector = targetVector.normalized() * ( referenceVectorLength = referenceVector.length() );

        float angleValue = std::acos( std::clamp( dot( params_.rays[0].normalized(), params_.rays[1].normalized() ), -1.f, 1.f ) );

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
            return toScreenCoords( *viewport_, worldCenterPoint + vector );
        };

        auto bisectVectorLow = [&]( const Vector3f& a, const Vector3f& b, int depth ) -> Vector3f
        {
            (void)b;
            if ( numGeneratedDepthSteps <=/*sic*/ depth )
            {
                #if __GNUC__ >= 11 && __GNUC__ <= 15
                #pragma GCC diagnostic push
                // A false positive, it seems? `bisectState()` ensures `depths < curveMaxSubdivisionDepth`.
                // And `numGEneratedDepthSteps` starts from 1, it can't be zero either.
                #pragma GCC diagnostic ignored "-Warray-bounds"
                #endif
                depthStepData[numGeneratedDepthSteps].angleStep = depthStepData[numGeneratedDepthSteps - 1].angleStep / 2;
                #if __GNUC__ >= 11 && __GNUC__ <= 15
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
            ImGuiMeasurementIndicators::line( elem, menuScaling_, indicatorParams, curve.a, curve.b, lineParams );

            // Inverted arrows.
            if ( useInvertedStyle )
            {
                ImGuiMeasurementIndicators::LineParams invArrowParams{ .capB = { .decoration = ImGuiMeasurementIndicators::LineCap::Decoration::arrow } };
                ImGuiMeasurementIndicators::line( elem, menuScaling_, indicatorParams, invertedStartA, curve.a, invArrowParams );
                ImGuiMeasurementIndicators::line( elem, menuScaling_, indicatorParams, invertedStartB, curve.b, invArrowParams );
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

                    if ( params_.shouldVisualizeRay[index] )
                        offset *= len / 3.f; // An arbitrary factor.
                    else
                        offset *= notchHalfLen;
                }

                if ( params_.shouldVisualizeRay[0] && params_.shouldVisualizeRay[1] )
                {
                    ImGuiMeasurementIndicators::line( elem, menuScaling_, indicatorParams, curve.a + offsets[0], curve.b + offsets[1], { .midPoints = { &screenCenterPoint, 1 } } );
                }
                else
                {
                    for ( bool index : { false, true } )
                    {
                        ImGuiMeasurementIndicators::line( elem, menuScaling_, indicatorParams,
                            curve.endPoint( index ) + offsets[index],
                            params_.shouldVisualizeRay[index] ? screenCenterPoint : curve.endPoint( index ) - offsets[index]
                        );
                    }
                }
            }

            // The text.
            ImGuiMeasurementIndicators::text( elem, menuScaling_, indicatorParams, textPos, angleToString( angleValue ), normal );
        };

        drawElem( ImGuiMeasurementIndicators::Element::outline );
        drawElem( ImGuiMeasurementIndicators::Element::main );
    }
}

LengthTask::LengthTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const LengthParams& params )
    : menuScaling_( uiParams.scale ), viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ), params_( params )
{
    for ( std::size_t i = 0; i < 2; i++ )
        params_.points[i] = xf( params_.points[i] );

    Vector3f depthRefPoint = params_.points[0] + ( params_.points[1] - params_.points[0] ) / 2.f;
    renderTaskDepth = viewport_->projectToViewportSpace( depthRefPoint ).z;
}

void LengthTask::renderPass()
{
    ImVec2 a = toScreenCoords( *viewport_, params_.points[0] );
    ImVec2 b = toScreenCoords( *viewport_, params_.points[1] );

    float distanceValue = 0;
    if ( params_.onlyOneAxis )
        distanceValue = std::abs( params_.points[1][*params_.onlyOneAxis] - params_.points[0][*params_.onlyOneAxis] );
    else
        distanceValue = ( params_.points[1] - params_.points[0] ).length();
    distanceValue *= ( params_.drawAsNegative ? -1.f : 1.f );

    ImGuiMeasurementIndicators::Params indicatorParams;
    indicatorParams.colorMain = color_;

    ImGuiMeasurementIndicators::Text text;

    std::string_view axisName;
    if ( params_.onlyOneAxis )
        axisName = std::array{" X", " Y", " Z"}[*params_.onlyOneAxis];

    // "Measured" prefix for value.
    if ( params_.referenceValue )
        text.addElem( { .var = fmt::format( "Measured{}: ", axisName ), .columnId = 0 } );

    const bool passOrFail = params_.referenceValue && params_.tolerance;
    const bool pass = passOrFail && distanceValue >= *params_.referenceValue + params_.tolerance->negative && distanceValue <= *params_.referenceValue + params_.tolerance->positive;

    // Style customization for value if we're in pass/fail mode.
    if ( passOrFail )
    {
        text.add( ImGuiMeasurementIndicators::TextColor( SceneColors::get( pass ? SceneColors::LabelsGood : SceneColors::LabelsBad ) ) );
        text.add( ImGuiMeasurementIndicators::TextFont{ RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::SemiBold ) } );
    }
    // The value itself.
    text.addElem( { .var = lengthToString( distanceValue ), .align = ImVec2( 1, 0 ), .columnId = 1 } );
    if ( passOrFail )
    {
        text.add( ImGuiMeasurementIndicators::TextColor{} );
        text.add( ImGuiMeasurementIndicators::TextFont{} );
    }

    // Nominal value.
    if ( params_.referenceValue )
    {
        text.addLine();
        text.addElem( { .var = fmt::format( "Nominal{}: ", axisName ), .columnId = 0 } );

        text.addElem( { .var = lengthToString( *params_.referenceValue ), .align = ImVec2( 1, 0 ), .columnId = 1 } ); // Not stripping zeroes here to align with the measured value.

        if ( params_.tolerance )
        {
            if ( params_.tolerance->positive == -params_.tolerance->negative )
                text.addText( fmt::format( " \xC2\xB1{}", lengthToleranceToString( params_.tolerance->positive, 0 ) ) ); // U+00B1 PLUS-MINUS SIGN
            else
                text.addText( fmt::format( " {}/{}", lengthToleranceToString( params_.tolerance->positive, 1 ), lengthToleranceToString( params_.tolerance->negative, -1 ) ) );
        }
    }

    ImGuiMeasurementIndicators::distance( ImGuiMeasurementIndicators::Element::both, menuScaling_, indicatorParams, a, b, text );
}

}
