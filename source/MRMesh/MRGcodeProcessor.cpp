#include "MRGcodeProcessor.h"
#include "MRVector2.h"
#include "MRMatrix2.h"
#include "MRQuaternion.h"
#include "MRTimer.h"
#include <cassert>
#include <chrono>

namespace MR
{

constexpr float cInch = 25.4f;
constexpr int cPointInRotation = 21;

//////////////////////////////////////////////////////////////////////////
// GcodeExecutor

void GcodeProcessor::reset()
{
    workPlane_ = WorkPlane::xy;
    toWorkPlaneXf_ = Matrix3f();
    translationPos_ = cncSettings_.getHomePosition();
    rotationAngles_ = Vector3f();
    absoluteCoordinates_ = true;
    scaling_ = Vector3f::diagonal( 1.f );
    inches_ = false;
    gcodeSource_.clear();
    feedrateMax_ = 0.f;
}

void GcodeProcessor::setGcodeSource( const GcodeSource& gcodeSource )
{
    reset();
    gcodeSource_.resize( gcodeSource.size() );
    for ( int i = 0; i < gcodeSource.size(); ++i )
        gcodeSource_[i] = gcodeSource[i];
}

std::vector<MR::GcodeProcessor::MoveAction> GcodeProcessor::processSource()
{
    MR_TIMER;
    if ( gcodeSource_.empty() )
        return {};

    std::vector<MoveAction> res( gcodeSource_.size() );
    std::vector<Command> tmp;
    for ( int i = 0; i < gcodeSource_.size(); ++i )
        res[i] = processLine( gcodeSource_[i], tmp );

    for ( auto& action : res )
    {
        if ( action.idle && action.feedrate == 0.f )
            action.feedrate = feedrateMax_;
    }

    return res;
}

GcodeProcessor::MoveAction GcodeProcessor::processLine( const std::string_view& line, std::vector<Command> & commands )
{
    if ( line.empty() )
        return {};

    commands.clear();
    parseFrame_( line, commands );
    if ( commands.empty() )
        return {};

    resetTemporaryStates_();

    // TODO add check is valid command set

    for ( int i = 0; i < commands.size(); ++i )
        applyCommand_( commands[i] );

    MoveAction result;
    if ( coordType_ == CoordType::Movement )
        result = generateMoveAction_();
    else if ( coordType_ == CoordType::ReturnToHome )
        result = generateReturnToHomeAction_();
    else if ( coordType_ == CoordType::Scaling )
        updateScaling_();

    coordType_ = CoordType::Movement;
    return result;
}

void GcodeProcessor::setCNCMachineSettings( const CNCMachineSettings& settings )
{
    cncSettings_ = settings;
    const auto& axesOrder = cncSettings_.getRotationOrder();
    rotationAxesOrderMap_.resize( axesOrder.size() );
    for ( int i = 0; i < axesOrder.size(); ++i )
    {
        rotationAxesOrderMap_[i] = int( axesOrder[i] );
    }
}

void GcodeProcessor::parseFrame_( const std::string_view& frame, std::vector<Command> & outCommands )
{
    size_t it = 0;
    char* numEnd = nullptr;
    auto commentStartInd = frame.find( ';' );
    while ( std::isspace( frame[it] ) )
        ++it;
    while ( it < frame.size() )
    {
        if ( commentStartInd <= it )
            return;
        if ( frame[it] == '(' )
        {
            it = frame.find( ')', it + 1 );
            if ( it == std::string::npos )
                break;

            ++it;
            continue;
        }
        else if ( !std::isalpha( frame[it] ) )
        {
            ++it;
            continue;
        }
        Command cmd;
        cmd.key = char( std::tolower( frame[it] ) );
        ++it;
        cmd.value = std::strtof( frame.data() + it, &numEnd );
        size_t newIt = numEnd - frame.data();
        if ( newIt != it )
        {
            it = newIt;
            outCommands.push_back( cmd );
        }
        while ( it < frame.size() && std::isspace( frame[it] ) )
            ++it;
    }
}

void GcodeProcessor::applyCommand_( const Command& command )
{
    if ( command.key == 'g' )
    {
        applyCommandG_( command );
        return;
    }
    if ( command.key >= 'x' && command.key <= 'z' )
    {
        const int index = command.key - 'x';
        inputCoords_[index] = command.value;
        inputCoordsReaded_[index] = true;
    }
    if ( command.key >= 'a' && command.key <= 'c' )
    {
        const int index = command.key - 'a';
        inputRotation_[index] = command.value;
        inputRotationReaded_[index] = true;
    }
    if ( command.key == 'f' )
        feedrate_ = inches_ ? cInch * command.value : command.value;
    if ( command.key == 'r' )
        radius_ = command.value;
    if ( command.key >= 'i' && command.key <= 'k' )
    {
        const int index = command.key - 'i';
        if ( !arcCenter_ )
            arcCenter_ = Vector3f();
        ( *arcCenter_ )[index] = command.value;
    }
}

void GcodeProcessor::applyCommandG_( const Command& command )
{
    int gValue = int( command.value );
    switch ( gValue )
    {
    case 0:
    case 1:
    case 2:
    case 3:
        coordType_ = CoordType::Movement;
        moveMode_ = MoveMode( gValue );
        break;
    case 17:
    case 18:
    case 19:
        updateWorkPlane_( static_cast< WorkPlane >( gValue - 17 ) );
        break;
    case 20:
        inches_ = true;
        break;
    case 21:
        inches_ = false;
        break;
    case 28:
        coordType_ = CoordType::ReturnToHome;
        break;
    case 50:
        scaling_ = Vector3f::diagonal( 1.f );
        break;
    case 51:
        coordType_ = CoordType::Scaling;
        break;
    case 90:
        absoluteCoordinates_ = true;
        break;
    case 91:
        absoluteCoordinates_ = false;
        break;
    default:
        break;
    }
}

GcodeProcessor::MoveAction GcodeProcessor::generateMoveAction_()
{
    MoveAction res;

    Vector3f newTranslationPos = calcNewTranslationPos_();
    Vector3f newRotationAngles = calcNewRotationAngles_();

    const bool anyCoordReaded = inputCoordsReaded_[0] || inputCoordsReaded_[1] || inputCoordsReaded_[2];
    const bool anyRotationReaded = inputRotationReaded_[0] || inputRotationReaded_[1] || inputRotationReaded_[2];

    if ( ( moveMode_ == MoveMode::Idle || moveMode_ == MoveMode::Line ) && anyCoordReaded )
        res = moveLine_( newTranslationPos, newRotationAngles );
    else if ( ( moveMode_ == MoveMode::Clockwise || moveMode_ == MoveMode::Counterclockwise ) && (anyCoordReaded || arcCenter_) )
        res = moveArc_( newTranslationPos, newRotationAngles, moveMode_ == MoveMode::Clockwise );
    else if ( anyRotationReaded )
    {
        res = getToolRotationPoints_( newRotationAngles );
    }
    assert( res.action.path.size() == res.toolDirection.size() );
    res.idle = ( moveMode_ == MoveMode::Idle || !( anyCoordReaded || anyRotationReaded || arcCenter_ ) );

    if ( moveMode_ == MoveMode::Idle )
        res.feedrate = cncSettings_.getFeedrateIdle();
    else
    {
        res.feedrate = feedrate_;
        feedrateMax_ = std::max( feedrateMax_, feedrate_ );
    }

    translationPos_ = newTranslationPos;
    Vector3f startAngles = rotationAngles_;
    updateRotationAngleAndMatrix_( newRotationAngles );
    const auto& rotationOrder = cncSettings_.getRotationOrder();
    for ( int i = 0; i < rotationOrder.size(); ++i )
    {
        const auto& limits = cncSettings_.getRotationLimits( rotationOrder[i] );
        if ( !limits )
            continue;
        const float& angle = rotationAngles_[int( rotationOrder[i] )];
        const float& startAngle = startAngles[int( rotationOrder[i] )];
        if ( startAngle < limits->x || startAngle > limits->y || angle < limits->x || angle > limits->y )
        {
            res.action.warning += ( !res.action.warning.empty() ? "\n" : "" ) + std::string("Error input angle: Going beyond the limits.");
            break;
        }
    }

    return res;
}

GcodeProcessor::MoveAction GcodeProcessor::generateReturnToHomeAction_()
{
    MoveAction res;
    Vector3f newTranslationPos = calcNewTranslationPos_();

    if ( newTranslationPos != translationPos_ )
    {
        res = moveLine_( newTranslationPos, rotationAngles_ );
        translationPos_ = newTranslationPos;
    }
    MoveAction res2 = moveLine_( cncSettings_.getHomePosition(), rotationAngles_ );
    translationPos_ = cncSettings_.getHomePosition();
    if ( res.action.path.empty() )
    {
        res.action.path = res2.action.path;
        res.toolDirection = res2.toolDirection;
    }
    else if ( !res2.action.path.empty() )
    {
        res.action.path.insert( res.action.path.end(), res2.action.path.begin() + 1, res2.action.path.end() );
        res.toolDirection.insert( res.toolDirection.end(), res2.toolDirection.begin() + 1, res2.toolDirection.end() );
    }
    res.action.warning += ( res.action.warning.empty() ? "" : "\n" ) + res2.action.warning;

    res.idle = true;
    res.feedrate = cncSettings_.getFeedrateIdle();
    return res;
}

void GcodeProcessor::resetTemporaryStates_()
{
    inputCoords_ = {};
    inputCoordsReaded_ = Vector3b( false, false, false );
    radius_ = {};
    arcCenter_ = {};
    inputRotation_ = {};
    inputRotationReaded_ = Vector3b( false, false, false );
}

GcodeProcessor::MoveAction GcodeProcessor::moveLine_( const Vector3f& newPoint, const Vector3f& newAngles )
{
    MoveAction res;

    if ( newAngles == rotationAngles_ )
    {
        res.action.path = { calcRealCoordCached_( translationPos_ ), calcRealCoordCached_( newPoint ) };
        res.toolDirection = std::vector<Vector3f>( 2, calcRealCoordCached_( Vector3f::plusZ() ) );
        return res;
    }

    res.action.path.resize( cPointInRotation );
    res.toolDirection.resize( cPointInRotation );
    const Vector3f lineStep = ( newPoint - translationPos_ ) / ( cPointInRotation - 1.f );
    const Vector3f rotationAnglesStep_ = ( newAngles - rotationAngles_ ) / ( cPointInRotation - 1.f );
    for ( int i = 0; i < cPointInRotation; ++i )
    {
        const auto currentRotationAngles_ = rotationAnglesStep_ * float( i ) + rotationAngles_;
        res.action.path[i] = calcRealCoord_( translationPos_ + lineStep * float( i ), currentRotationAngles_ );
        res.toolDirection[i] = calcRealCoord_( Vector3f::plusZ(), currentRotationAngles_ );
    }

    return res;
}

GcodeProcessor::MoveAction GcodeProcessor::moveArc_( const Vector3f& newPoint, const Vector3f& newAngles, bool clockwise )
{
    MoveAction res;

    if ( radius_ )
        res.action = getArcPoints3_( *radius_, translationPos_, newPoint, clockwise );
    else if ( arcCenter_ )
        res.action = getArcPoints3_( translationPos_ + *arcCenter_, translationPos_, newPoint, clockwise );
    else
        res.action.warning = "Missing parameters.";

    if ( !res.action.path.empty() )
    {
        if ( newAngles == rotationAngles_ )
        {
            for ( auto& point : res.action.path )
                point = calcRealCoordCached_( point );
            res.toolDirection = std::vector<Vector3f>( res.action.path.size(), calcRealCoordCached_( Vector3f::plusZ() ) );
        }
        else
        {
            const int pointCount = int( res.action.path.size() );
            res.toolDirection.resize( pointCount );
            const Vector3f rotationAnglesStep_ = ( newAngles - rotationAngles_ ) / ( pointCount - 1.f ); // in this case, pointCount minimum value is 11 (as result getArcPoints2_ method)
            for ( int i = 0; i < pointCount; ++i )
            {
                auto& point = res.action.path[i];
                const auto currentRotationAngles_ = rotationAnglesStep_ * float( i ) + rotationAngles_;
                point = calcRealCoord_( point, currentRotationAngles_ );
                res.toolDirection[i] = calcRealCoord_( Vector3f::plusZ(), currentRotationAngles_ );
            }
        }
    }

    return res;
}

void GcodeProcessor::updateWorkPlane_( WorkPlane wp )
{
    workPlane_ = wp;
    if ( workPlane_ == WorkPlane::zx )
        toWorkPlaneXf_ = Matrix3f( { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } );
    else if ( workPlane_ == WorkPlane::yz )
        toWorkPlaneXf_ = Matrix3f( { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } );
    else
        toWorkPlaneXf_ = Matrix3f();
}

void GcodeProcessor::updateScaling_()
{
    for ( int i = 0; i < 3; ++i )
    {
        if ( inputCoordsReaded_[i] && inputCoords_[i] != 0 )
            scaling_[i] = inputCoords_[i];
    }
}

GcodeProcessor::MoveAction GcodeProcessor::getToolRotationPoints_( const Vector3f& newRotationAngles )
{
    if ( newRotationAngles == rotationAngles_ )
        return {};

    MoveAction res;

    Vector3f rotationAnglesStep_ = ( newRotationAngles - rotationAngles_ ) / ( cPointInRotation - 1.f );
    res.action.path.resize( cPointInRotation );
    res.toolDirection.resize( cPointInRotation );
    for ( int i = 0; i < cPointInRotation; ++i )
    {
        const auto currentRotationAngles_ = rotationAnglesStep_ * float(i) + rotationAngles_;
        res.action.path[i] = calcRealCoord_( translationPos_, currentRotationAngles_ );
        res.toolDirection[i] = calcRealCoord_( Vector3f::plusZ(), currentRotationAngles_ );
    }

    return res;
}

GcodeProcessor::BaseAction2f GcodeProcessor::getArcPoints2_( const Vector2f& beginPoint, const Vector2f& endPoint, bool clockwise )
{
    BaseAction2f res;
    const float beginLengthSq = beginPoint.lengthSq();
    const float endLengthSq = endPoint.lengthSq();
    const float maxLengthSq = std::max( beginLengthSq, endLengthSq );
    const float deltaLength2 = std::fabs( beginPoint.lengthSq() - endPoint.lengthSq() );
    if ( deltaLength2 >= ( 2.5f * accuracy_ * maxLengthSq ) )
        res.warning = "Begin and end radius are different: diff = " + std::to_string( std::sqrt( deltaLength2 ) );

    const Vector2f v1 = beginPoint / beginPoint.length();
    const Vector2f v2 = endPoint / endPoint.length();
    float beginAngle = std::atan2( v1.y, v1.x );
    float endAngle = std::atan2( v2.y, v2.x );
    if ( clockwise && ( beginAngle <= endAngle ) )
        beginAngle += PI_F * 2.f;
    else if ( !clockwise && ( endAngle <= beginAngle ) )
        endAngle += PI_F * 2.f;

    const int stepCount = std::clamp( int( std::ceil( std::fabs( endAngle - beginAngle ) / ( 6.f / 180.f * PI_F ) ) ), 10, 60 );
    const float angleStep = ( endAngle - beginAngle ) / stepCount;
    res.path.reserve( stepCount + 1 );
    res.path.push_back( beginPoint );
    for ( int i = 0; i < stepCount; ++i )
        res.path.emplace_back( Matrix2f::rotation( angleStep * ( i + 1 ) ) * beginPoint );

    return res;
}

GcodeProcessor::BaseAction3f GcodeProcessor::getArcPoints3_( const Vector3f& center, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise )
{
    const Vector3f c3 = toWorkPlaneXf_ * center;

    const Vector3f b3 = toWorkPlaneXf_ * beginPoint - c3;
    const Vector2f b2 = { b3.x, b3.y };

    const Vector3f e3 = toWorkPlaneXf_ * endPoint - c3;
    const Vector2f e2 = { e3.x, e3.y };

    const bool helical = std::fabs( b3.z - e3.z ) > accuracy_;

    const Matrix3f toWorldXf = toWorkPlaneXf_.inverse();
    auto res2 = getArcPoints2_( b2, e2, clockwise );

    BaseAction3f res3;
    res3.warning = std::move( res2.warning );
    res3.path.resize( res2.path.size() );
    const float zStep = res2.path.size() > 1 ? ( e3.z - b3.z ) / ( res2.path.size() - 1 ) : 0.f;
    for ( int i = 0; i < res2.path.size(); ++i )
        res3.path[i] = toWorldXf * ( Vector3f( res2.path[i].x, res2.path[i].y, helical ? b3.z + zStep * i : b3.z ) + c3 );

    return res3;
}

GcodeProcessor::BaseAction3f GcodeProcessor::getArcPoints3_( float r, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise )
{
    if ( r < accuracy_ )
        return { .path = { beginPoint, endPoint },.warning = "Wrong radius" };

    const Vector3f b3 = toWorkPlaneXf_ * beginPoint;
    const Vector2f b2 = { b3.x, b3.y };

    const Vector3f e3 = toWorkPlaneXf_ * endPoint;
    const Vector2f e2 = { e3.x, e3.y };

    const bool helical = std::fabs( b3.z - e3.z ) > accuracy_;

    const Vector2f middlePoint = ( b2 + e2 ) / 2.f;
    const Vector2f middleVec = middlePoint - b2;
    const Vector2f middleNormal = ( Matrix2f::rotation( -PI2_F ) * middleVec ).normalized();

    const float normalLenght = std::sqrt( r * r - middleVec.lengthSq() );
    const Vector2f c2 = middlePoint + middleNormal * normalLenght * ( clockwise == ( r > 0.f ) ? 1.f : -1.f );
    const Vector3f c3 = { c2.x, c2.y, 0.0f };


    const Matrix3f toWorldXf = toWorkPlaneXf_.inverse();
    auto res2 = getArcPoints2_( b2 - c2, e2 - c2, clockwise );
    BaseAction3f res3;
    res3.warning = std::move( res2.warning );
    res3.path.resize( res2.path.size() );
    const float zStep = res2.path.size() > 1 ? ( e3.z - b3.z ) / ( res2.path.size() - 1 ) : 0.f;
    for ( int i = 0; i < res2.path.size(); ++i )
        res3.path[i] = toWorldXf * ( Vector3f( res2.path[i].x, res2.path[i].y, helical ? b3.z + zStep * i : b3.z ) + c3 );

    return res3;
}

Vector3f GcodeProcessor::calcNewTranslationPos_()
{
    Vector3f res = mult( inputCoords_, scaling_ );
    if ( inches_ )
        res *= cInch;
    if ( absoluteCoordinates_ )
    {
        for ( int i = 0; i < 3; ++i )
        {
            if ( !inputCoordsReaded_[i] )
                res[i] = translationPos_[i];
        }
    }
    else
        res += translationPos_;

    return res;
}

Vector3f GcodeProcessor::calcNewRotationAngles_()
{
    Vector3f res = inputRotation_;
    if ( absoluteCoordinates_ )
    {
        for ( int i = 0; i < 3; ++i )
        {
            if ( !inputRotationReaded_[i] )
                res[i] = rotationAngles_[i];
        }
    }
    else
        res += rotationAngles_;

    return res;
}

Vector3f GcodeProcessor::calcRealCoord_( const Vector3f& translationPos, const Vector3f& rotationAngles )
{
    Vector3f res = translationPos;
    const auto& axesOrder = cncSettings_.getRotationOrder();
    for ( int i = 0; i < axesOrder.size(); ++i )
    {
        const int axisNumber = int( axesOrder[i] );
        const Matrix3f rotationMatrix = Matrix3f::rotation( cncSettings_.getRotationAxis( axesOrder[i] ), rotationAngles[axisNumber] / 180.f * PI_F );
        res = rotationMatrix * res;
    }
    return res;
}

void GcodeProcessor::updateRotationAngleAndMatrix_( const Vector3f& rotationAngles )
{
    for ( int i = 0; i < 3; ++i )
    {
        rotationAngles_[i] = rotationAngles[i];
        cacheRotationMatrix_[i] = Matrix3f::rotation( cncSettings_.getRotationAxis( CNCMachineSettings::RotationAxisName( i ) ), rotationAngles_[i] / 180.f * PI_F );
    }
}

MR::Vector3f GcodeProcessor::calcRealCoordCached_( const Vector3f& translationPos, const Vector3f& rotationAngles )
{
    updateRotationAngleAndMatrix_( rotationAngles );
    return calcRealCoordCached_( translationPos );
}

MR::Vector3f GcodeProcessor::calcRealCoordCached_( const Vector3f& translationPos )
{
    Vector3f res = translationPos;
    const auto& axesOrder = cncSettings_.getRotationOrder();
    for ( int i = 0; i < axesOrder.size(); ++i )
    {
        const int axisNumber = int( axesOrder[i] );
        res = cacheRotationMatrix_[axisNumber] * res;
    }
    return res;
}

}
