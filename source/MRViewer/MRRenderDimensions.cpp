#include "MRRenderDimensions.h"

#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRImGuiMeasurementIndicators.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRRibbonFontManager.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRUIStyle.h"
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

static bool objectIsSelectable( const VisualObject* object )
{
    return object && !object->isGlobalAncillary();
}

static void selectObject( const VisualObject* object )
{
    if ( !objectIsSelectable( object ) )
        return;

    // Yes, a dumb cast. We could find the same object in the scene, but it's a waste of time.
    // Changing the `RenderObject` constructor parameter to accept a non-const reference requires changing a lot of stuff.
    RibbonMenu::instance()->simulateNameTagClickWithKeyboardModifiers( *const_cast<VisualObject*>( object ) );
}

static Color getBorderColorForObject( ViewportId viewportId, const VisualObject* object, float alpha )
{
    return object->getFrontColor( object->isSelected(), viewportId ).scaledAlpha( alpha );
}

static ImGuiMeasurementIndicators::TextParams makeTextParams( ViewportId viewportId, const VisualObject* object, const BasicClickableRectUiRenderTask& task )
{
    if ( !objectIsSelectable( object ) )
        return {};

    return {
        .borderColor = getBorderColorForObject( viewportId, object, 0.75f ), // Arbitrary alpha factor to make the borders less visible.
        .isHovered = task.isHovered,
        .isActive = task.isActive,
        .isSelected = object->isSelected(),
    };
}

static void beginPassFailTextStyle( ImGuiMeasurementIndicators::Text& text, bool pass )
{
    text.add( ImGuiMeasurementIndicators::TextColor( SceneColors::get( pass ? SceneColors::LabelsGood : SceneColors::LabelsBad ) ) );
}
static void endPassFailTextStyle( ImGuiMeasurementIndicators::Text& text )
{
    text.add( ImGuiMeasurementIndicators::TextColor{} );
}

[[nodiscard]] static ImVec2 toScreenCoords( const Viewport& viewport, const Vector3f& point )
{
    auto rect = viewport.getViewportRect();
    Vector3f result = viewport.projectToViewportSpace( point );
    return ImVec2( result.x, result.y ) + ImVec2( rect.min.x, ImGui::GetIO().DisplaySize.y - rect.max.y );
}

PointTask::PointTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const PointParams& params )
    : viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ), params_( params )
{
    params_.point = xf( params_.point );
    renderTaskDepth = viewport_->projectToViewportSpace( params_.point ).z;
}

void PointTask::renderPass()
{
    // We set those after we're done drawing.
    clickableCornerA = {};
    clickableCornerB = {};
    enabled = false;

    ImGuiMeasurementIndicators::Text text;
    if ( !params_.common.objectName.empty() )
    {
        text.addLine();
        text.addText( params_.common.objectName );
    }

    float deviation = 0;
    bool passOrFail = false;
    bool pass = false;

    if ( params_.referencePoint && params_.tolerance )
    {
        passOrFail = true;
        if ( params_.referenceNormal == Vector3f{} )
        {
            deviation = ( params_.point - *params_.referencePoint ).length();
            pass = deviation >= params_.tolerance->negative && deviation <= params_.tolerance->positive;
        }
        else
        {
            deviation = dot( params_.point - *params_.referencePoint, params_.referenceNormal ) / params_.referenceNormal.length();
            pass = deviation <= params_.tolerance->positive;
        }
    }

    int axis = -1;
    if ( params_.referencePoint && params_.referenceNormal != Vector3f{} )
    {
        if ( params_.referenceNormal.y == 0 && params_.referenceNormal.z == 0 )
            axis = 0;
        else if ( params_.referenceNormal.z == 0 && params_.referenceNormal.x == 0 )
            axis = 1;
        else if ( params_.referenceNormal.x == 0 && params_.referenceNormal.y == 0 )
            axis = 2;
    }
    int smallestAxis = axis == -1 ? 0 : axis;

    text.addLine();
    if ( params_.referencePoint && params_.tolerance )
    {
        text.addElem( {
            .var =
                axis == 0 ? "Measured X: " :
                axis == 1 ? "Measured Y: " :
                axis == 2 ? "Measured Z: " :
                "Measured: ",
            .columnId = 0,
        } );
    }

    if ( passOrFail )
        beginPassFailTextStyle( text, pass );

    for ( int i = 0; i < 3; i++ )
    {
        if ( axis == -1 || axis == i )
            text.addElem( { .var = ( i == smallestAxis ? "" : " " ) + lengthToString( params_.point[i] ), .align = ImVec2( 1, 0 ), .columnId = i + 1 } );
    }

    if ( passOrFail )
        endPassFailTextStyle( text );

    if ( params_.referencePoint && params_.tolerance )
    {
        text.addLine();
        text.addElem( {
            .var =
                axis == 0 ? "Nominal X: " :
                axis == 1 ? "Nominal Y: " :
                axis == 2 ? "Nominal Z: " :
                "Nominal: ",
            .columnId = 0
        } );

        for ( int i = 0; i < 3; i++ )
        {
            if ( axis == -1 || axis == i )
                text.addElem( { .var = ( i == smallestAxis ? "" : " " ) + lengthToString( ( *params_.referencePoint )[i] ), .align = ImVec2( 1, 0 ), .columnId = i + 1 } );
        }

        if ( params_.tolerance )
        {
            if ( params_.referenceNormal == Vector3f{} )
                text.addText( fmt::format( " {}", lengthToleranceToString( params_.tolerance->positive, 0 ) ) );
            else if ( params_.tolerance->positive == -params_.tolerance->negative )
                text.addText( fmt::format( " \xC2\xB1{}", lengthToleranceToString( params_.tolerance->positive, 0 ) ) ); // U+00B1 PLUS-MINUS SIGN
            else
                text.addText( fmt::format( " {}/{}", lengthToleranceToString( params_.tolerance->positive, 1 ), lengthToleranceToString( params_.tolerance->negative, -1 ) ) );
        }
    }

    ImGuiMeasurementIndicators::Params indicatorParams;
    indicatorParams.colorMain = color_;

    ImVec2 screenPoint = toScreenCoords( *viewport_, params_.point );
    text.update();
    ImVec2 screenTextPos = screenPoint + ( std::min( text.computedSize.x, text.computedSize.y ) + ImVec2( 24, 24 ) * UI::scale() ) * params_.align;

    auto textParams = makeTextParams( viewport_->id, params_.common.objectToSelect, *this );
    textParams.line = {
        .point = screenPoint,
        .capDecoration = ImGuiMeasurementIndicators::LineCapDecoration::point,
        .body = {
            .colorOverride = getBorderColorForObject( viewport_->id, params_.common.objectToSelect, 1.f ),
        },
    };

    auto textResult = ImGuiMeasurementIndicators::text( ImGuiMeasurementIndicators::Element::both, indicatorParams, screenTextPos, text, textParams );

    if ( textResult )
    {
        clickableCornerA = textResult->bgCornerA;
        clickableCornerB = textResult->bgCornerB;
    }

    enabled = objectIsSelectable( params_.common.objectToSelect );
}

void PointTask::onClick()
{
    selectObject( params_.common.objectToSelect );
}

RadiusTask::RadiusTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const RadiusParams& params )
    : viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ), params_( params )
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
    // We set those after we're done drawing.
    clickableCornerA = {};
    clickableCornerB = {};
    enabled = false;

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

    ImVec2 center = toScreenCoords( *viewport_, params_.center );
    ImVec2 point = toScreenCoords( *viewport_, params_.center + worldRadiusVec );

    float lengthMultiplier = params_.visualLengthMultiplier;
    ImVec2 farPoint = toScreenCoords( *viewport_, params_.center + worldRadiusVec * ( 1 + lengthMultiplier ) );

    float minRadiusLen = 32 * UI::scale();

    if ( ImGuiMath::lengthSq( farPoint - point ) < minRadiusLen * minRadiusLen )
        farPoint = point + ImGuiMath::normalize( point - center ) * minRadiusLen;

    ImGuiMeasurementIndicators::Text text;
    if ( !params_.common.objectName.empty() )
    {
        text.addText( params_.common.objectName );
        text.addLine();
    }

    if ( params_.isSpherical )
        text.addText( "S" );

    if ( params_.drawAsDiameter )
        text.add( ImGuiMeasurementIndicators::TextIcon::diameter );
    else
        text.addText( "R" );

    text.addText( " " );
    text.addText( lengthToString( radiusValue * ( params_.drawAsDiameter ? 2 : 1 ) ) );

    auto lineResult = ImGuiMeasurementIndicators::line( ImGuiMeasurementIndicators::Element::both, indicatorParams,
        farPoint, point, {
            .capA = { .text = text, .textParams = makeTextParams( viewport_->id, params_.common.objectToSelect, *this ) },
            .capB = { .decoration = ImGuiMeasurementIndicators::LineCapDecoration::arrow },
        }
    );

    if ( lineResult && lineResult->capA )
    {
        clickableCornerA = lineResult->capA->bgCornerA;
        clickableCornerB = lineResult->capA->bgCornerB;
    }

    enabled = objectIsSelectable( params_.common.objectToSelect );
}

void RadiusTask::onClick()
{
    selectObject( params_.common.objectToSelect );
}

AngleTask::AngleTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const AngleParams& params )
    : viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ),params_( params )
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
    // We set those after we're done drawing.
    clickableCornerA = {};
    clickableCornerB = {};
    enabled = false;

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

        float totalLenThreshold = indicatorParams.totalLenThreshold * UI::scale();
        float invertedOverhang = indicatorParams.invertedOverhang * UI::scale();
        float notchHalfLen = indicatorParams.notchHalfLen * UI::scale();

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
        lineParams.capA.decoration = ImGuiMeasurementIndicators::LineCapDecoration::arrow;
        lineParams.capB.decoration = ImGuiMeasurementIndicators::LineCapDecoration::arrow;
        lineParams.midPoints = curve.midPoints;

        ImVec2 invertedStartA, invertedStartB;
        if ( useInvertedStyle )
        {
            lineParams.body.flags |= ImGuiMeasurementIndicators::LineFlags::narrow;
            lineParams.capA.decoration = {};
            lineParams.capB.decoration = {};

            ImVec2 dirA = ImGuiMath::normalize( curve.a - ( curve.midPoints.empty() ? curve.b : curve.midPoints.front() ) );
            ImVec2 dirB = ImGuiMath::normalize( curve.b - ( curve.midPoints.empty() ? curve.a : curve.midPoints.back() ) );

            invertedStartA = curve.a + dirA * invertedOverhang;
            invertedStartB = curve.b + dirB * invertedOverhang;
        }

        ImGuiMeasurementIndicators::Text text;
        if ( !params_.common.objectName.empty() )
        {
            text.addText( params_.common.objectName );
            text.addLine();
        }
        text.addText( angleToString( angleValue ) );

        auto drawElem = [&]( ImGuiMeasurementIndicators::Element elem )
        {
            // The main curve.
            ImGuiMeasurementIndicators::line( elem, indicatorParams, curve.a, curve.b, lineParams );

            // Inverted arrows.
            if ( useInvertedStyle )
            {
                ImGuiMeasurementIndicators::LineParams invArrowParams{ .capB = { .decoration = ImGuiMeasurementIndicators::LineCapDecoration::arrow } };
                ImGuiMeasurementIndicators::line( elem, indicatorParams, invertedStartA, curve.a, invArrowParams );
                ImGuiMeasurementIndicators::line( elem, indicatorParams, invertedStartB, curve.b, invArrowParams );
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
                    ImGuiMeasurementIndicators::line( elem, indicatorParams, curve.a + offsets[0], curve.b + offsets[1], { .midPoints = { &screenCenterPoint, 1 } } );
                }
                else
                {
                    for ( bool index : { false, true } )
                    {
                        ImGuiMeasurementIndicators::line( elem, indicatorParams,
                            curve.endPoint( index ) + offsets[index],
                            params_.shouldVisualizeRay[index] ? screenCenterPoint : curve.endPoint( index ) - offsets[index]
                        );
                    }
                }
            }
        };

        drawElem( ImGuiMeasurementIndicators::Element::outline );
        drawElem( ImGuiMeasurementIndicators::Element::main );

        // The text.
        // This is intentionally outside of the `drawElem()` lambda, to be completely on top of the angle indicator.
        auto textResult = ImGuiMeasurementIndicators::text( ImGuiMeasurementIndicators::Element::both, indicatorParams, textPos, text, makeTextParams( viewport_->id, params_.common.objectToSelect, *this ), normal );
        if ( textResult )
        {
            clickableCornerA = textResult->bgCornerA;
            clickableCornerB = textResult->bgCornerB;
        }

        enabled = objectIsSelectable( params_.common.objectToSelect );
    }
}

void AngleTask::onClick()
{
    selectObject( params_.common.objectToSelect );
}

Vector3f LengthTask::computeCornerPoint()
{
    assert( params_.onlyOneAxis );
    if ( !params_.onlyOneAxis )
        return {};

    Vector3f ret = params_.points[0];
    ret[*params_.onlyOneAxis] = params_.points[1][*params_.onlyOneAxis];
    return ret;
}

LengthTask::LengthTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const LengthParams& params )
    : viewport_( &getViewerInstance().viewport( uiParams.viewportId ) ), color_( color ), params_( params )
{
    for ( std::size_t i = 0; i < 2; i++ )
        params_.points[i] = xf( params_.points[i] );

    Vector3f depthRefPoint = params_.points[0] + ( ( params_.onlyOneAxis ? computeCornerPoint() : params_.points[1] ) - params_.points[0] ) / 2.f;
    renderTaskDepth = viewport_->projectToViewportSpace( depthRefPoint ).z;
}

void LengthTask::renderPass()
{
    // We set those after we're done drawing.
    clickableCornerA = {};
    clickableCornerB = {};
    enabled = false;

    float distanceValue = 0;
    if ( params_.onlyOneAxis )
        distanceValue = std::abs( params_.points[1][*params_.onlyOneAxis] - params_.points[0][*params_.onlyOneAxis] );
    else
        distanceValue = ( params_.points[1] - params_.points[0] ).length();
    distanceValue *= ( params_.drawAsNegative ? -1.f : 1.f );

    ImGuiMeasurementIndicators::Params indicatorParams;
    indicatorParams.colorMain = color_;

    ImGuiMeasurementIndicators::Text text;
    if ( !params_.common.objectName.empty() )
    {
        text.addText( params_.common.objectName );
        text.addLine();
    }

    std::string_view axisName;
    if ( params_.onlyOneAxis )
        axisName = std::array{" X", " Y", " Z"}[*params_.onlyOneAxis];

    // "Measured" prefix for value.
    if ( params_.referenceValue )
        text.addElem( { .var = fmt::format( "Measured{}: ", axisName ), .columnId = 0 } );
    else if ( !axisName.empty() )
        text.addElem( { .var = fmt::format( "{}: ", axisName.substr( 1 ) ), .columnId = 0 } );

    const bool passOrFail = params_.referenceValue && params_.tolerance;
    const bool pass = passOrFail && distanceValue >= *params_.referenceValue + params_.tolerance->negative && distanceValue <= *params_.referenceValue + params_.tolerance->positive;

    // Style customization for value if we're in pass/fail mode.
    if ( passOrFail )
        beginPassFailTextStyle( text, pass );
    // The value itself.
    text.addElem( { .var = lengthToString( distanceValue ), .align = ImVec2( 1, 0 ), .columnId = 1 } );
    if ( passOrFail )
        endPassFailTextStyle( text );

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

    ImVec2 a = toScreenCoords( *viewport_, params_.points[0] );
    ImVec2 b = toScreenCoords( *viewport_, params_.points[1] );

    std::optional<ImGuiMeasurementIndicators::DistanceResult> distanceResult;

    if ( params_.onlyOneAxis )
    {
        Vector3f cornerPointWorld = computeCornerPoint();
        ImVec2 cornerPointScreen = toScreenCoords( *viewport_, cornerPointWorld );

        for ( auto elem : { ImGuiMeasurementIndicators::Element::outline, ImGuiMeasurementIndicators::Element::main } )
        {
            distanceResult = ImGuiMeasurementIndicators::distance( elem, indicatorParams, a, cornerPointScreen, text, { .textParams = makeTextParams( viewport_->id, params_.common.objectToSelect, *this ) } );
            ImGuiMeasurementIndicators::line( elem, indicatorParams, cornerPointScreen, b, {
                .body = { .flags = ImGuiMeasurementIndicators::LineFlags::narrow, .stipple = indicatorParams.stippleDashed },
                .capA = { .decoration = ImGuiMeasurementIndicators::LineCapDecoration::extend },
                .capB = { .decoration = ImGuiMeasurementIndicators::LineCapDecoration::point },
            } );
        }
    }
    else
    {
        distanceResult = ImGuiMeasurementIndicators::distance( ImGuiMeasurementIndicators::Element::both, indicatorParams, a, b, text, { .textParams = makeTextParams( viewport_->id, params_.common.objectToSelect, *this ) } );
    }

    if ( distanceResult && distanceResult->text )
    {
        clickableCornerA = distanceResult->text->bgCornerA;
        clickableCornerB = distanceResult->text->bgCornerB;
    }

    enabled = objectIsSelectable( params_.common.objectToSelect );
}

void LengthTask::onClick()
{
    selectObject( params_.common.objectToSelect );
}

}
