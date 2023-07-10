#include "MRGcodeProcessor.h"
#include "MRVector2.h"
#include "MRMatrix2.h"
#include "MRQuaternion.h"
#include <cassert>
#include <chrono>

namespace MR
{

constexpr float cInch = 25.4f;
constexpr int pointInRotation = 21;

//////////////////////////////////////////////////////////////////////////
// GcodeExecutor

GcodeProcessor::GcodeProcessor()
{
    rotationAxes_ = { Vector3f::minusX(), Vector3f::minusY(), Vector3f::plusZ() };
    setRotationOrder( { RotationParameterName::A, RotationParameterName::B, RotationParameterName::C } );
}

void GcodeProcessor::reset()
{
    workPlane_ = WorkPlane::xy;
    toWorkPlaneXf_ = Matrix3f();
    translationPos_ = Vector3f();
    rotationAngles_ = Vector3f();
    absoluteCoordinates_ = true;
    scaling_ = Vector3f::diagonal( 1.f );
    inches_ = false;
    gcodeSource_.clear();
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
    if ( gcodeSource_.empty() )
        return {};

    std::vector<MoveAction> res( gcodeSource_.size() );
    for ( int i = 0; i < gcodeSource_.size(); ++i )
        res[i] = processLine( gcodeSource_[i] );

    return res;
}

GcodeProcessor::MoveAction GcodeProcessor::processLine( const std::string_view& line )
{
    if ( line.empty() )
        return {};

    auto commands = parseFrame_( line );
    if ( commands.empty() )
        return {};

    resetTemporaryStates_();

    // TODO add check is valid command set

    for ( int i = 0; i < commands.size(); ++i )
        applyCommand_( commands[i] );

    if ( coordType_ == CoordType::Movement )
        return generateMoveAction_();

    if ( coordType_ == CoordType::Scaling )
        updateScaling_();

    coordType_ = CoordType::Movement;
    return {};
}

void GcodeProcessor::setRotationParams( RotationParameterName paramName, const Vector3f& rotationAxis )
{
    const int intParamName = int( paramName );
    const bool validParamName = intParamName >= int( RotationParameterName::A ) && intParamName < int( RotationParameterName::Count );
    assert( validParamName );
    if ( !validParamName )
        return;

    const int axisIndex = rotationAxesOrder_[intParamName];
    if ( axisIndex < 0 || axisIndex >= 3 )
        return;

    if ( rotationAxis.lengthSq() < 0.01f )
        return;

    rotationAxes_[axisIndex] = rotationAxis;
}

Vector3f GcodeProcessor::getRotationParams( RotationParameterName paramName )
{
    const int intParamName = int( paramName );
    const bool validParamName = intParamName >= int( RotationParameterName::A ) && intParamName < int( RotationParameterName::Count );
    assert( validParamName );
    if ( !validParamName )
        return {};

    const int axisIndex = rotationAxesOrder_[intParamName];
    if ( axisIndex < 0 || axisIndex >= 3 )
        return {};

    return rotationAxes_[axisIndex];
}

bool GcodeProcessor::setRotationOrder( std::array<RotationParameterName, 3> rotationAxisOrder )
{
    bool validInput = true;
    std::array<int, 3> newOrder = {-1, -1, -1};
    for ( int i = 0; i < 3; ++i )
    {
        const int parameterIndex = int( rotationAxisOrder[i] );
        if ( parameterIndex < int( RotationParameterName::None ) || parameterIndex >= int( RotationParameterName::Count ) )
        {
            validInput = false;
            break;
        }
        if ( parameterIndex != int( RotationParameterName::None ) )
            newOrder[i] = parameterIndex;
    }
    assert( validInput );
    if ( !validInput )
        return false;

    rotationAxesOrder_ = newOrder;
    return true;
}

std::array<GcodeProcessor::RotationParameterName, 3> GcodeProcessor::getRotationOrder()
{
    std::array<MR::GcodeProcessor::RotationParameterName, 3> res;
    for ( int i = 0; i < 3; ++i )
    {
        res[i] = RotationParameterName( rotationAxesOrder_[i] );
    }
    return res;
}

std::vector<GcodeProcessor::Command> GcodeProcessor::parseFrame_( const std::string_view& frame )
{
    std::vector<Command> commands;
    size_t it = 0;
    char* numEnd = nullptr;
    while ( std::isspace( frame[it] ) )
        ++it;
    while ( it < frame.size() )
    {
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
            commands.push_back( cmd );
        }
        while ( std::isspace( frame[it] ) )
            ++it;
    }

    return commands;
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
        if ( !inputRotation_ )
            inputRotation_ = rotationAngles_;
        ( *inputRotation_ )[index] = command.value;
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

    Vector3f newMotorsPos = calcCoordMotors_();

    const bool anyCoordReaded = inputCoordsReaded_[0] || inputCoordsReaded_[1] || inputCoordsReaded_[2];
    
    if ( ( moveMode_ == MoveMode::Idle || moveMode_ == MoveMode::Line ) && anyCoordReaded )
        res = moveLine_( newMotorsPos, moveMode_ == MoveMode::Idle );
    else if ( ( moveMode_ == MoveMode::Clockwise || moveMode_ == MoveMode::Counterclockwise ) && (anyCoordReaded || arcCenter_) )
        res = moveArc_( newMotorsPos, moveMode_ == MoveMode::Clockwise );
    else if ( inputRotation_ )
    {
        res = getToolRotationPoints_();
    }

    assert( res.action.path.size() == res.toolDirection.size() );

    translationPos_ = newMotorsPos;
    if ( inputRotation_ )
        updateRotationAngleAndMatrix_( *inputRotation_ );
    res.feedrate = feedrate_;

    return res;
}

void GcodeProcessor::resetTemporaryStates_()
{
    inputCoords_ = {};
    inputCoordsReaded_ = Vector3<bool>( false, false, false );
    radius_ = {};
    arcCenter_ = {};
    inputRotation_ = {};
}

GcodeProcessor::MoveAction GcodeProcessor::moveLine_( const Vector3f& newPoint, bool idle )
{
    // MoveAction res({ basePoint_, newPoint }, idle); //fatal error C1001: Internal compiler error.
    MoveAction res;
    res.idle = idle;

    if ( !inputRotation_ )
    {
        res.action.path = { calcRealCoordCached_( translationPos_ ), calcRealCoordCached_( newPoint ) };
        res.toolDirection = std::vector<Vector3f>( 2, calcRealCoordCached_( Vector3f::plusZ() ) );
        return res;
    }

    res.action.path.resize( pointInRotation );
    res.toolDirection.resize( pointInRotation );
    const Vector3f lineStep = ( newPoint - translationPos_ ) / ( pointInRotation - 1.f );
    const Vector3f rotationAnglesStep_ = ( *inputRotation_ - rotationAngles_ ) / ( pointInRotation - 1.f );
    for ( int i = 0; i < pointInRotation; ++i )
    {
        const auto currentRotationAngles_ = rotationAnglesStep_ * float( i ) + rotationAngles_;
        res.action.path[i] = calcRealCoord_( translationPos_ + lineStep * float( i ), currentRotationAngles_ );
        res.toolDirection[i] = calcRealCoord_( Vector3f::plusZ(), currentRotationAngles_ );
    }

    return res;
}

GcodeProcessor::MoveAction GcodeProcessor::moveArc_( const Vector3f& newPoint, bool clockwise )
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
        if ( !inputRotation_ )
        {
            for ( auto& point : res.action.path )
                point = calcRealCoordCached_( point );
            res.toolDirection = std::vector<Vector3f>( res.action.path.size(), calcRealCoordCached_( Vector3f::plusZ() ) );
        }
        else
        {
            const int pointCount = int( res.action.path.size() );
            res.toolDirection.resize( pointCount );
            const Vector3f rotationAnglesStep_ = ( *inputRotation_ - rotationAngles_ ) / ( pointCount - 1.f );
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

GcodeProcessor::MoveAction GcodeProcessor::getToolRotationPoints_()
{
    if ( !inputRotation_ )
        return {};

    MoveAction res;
    res.idle = moveMode_ == MoveMode::Idle;
    
    Vector3f rotationAnglesStep_ = ( *inputRotation_ - rotationAngles_ ) / ( pointInRotation - 1.f );
    res.action.path.resize( pointInRotation );
    res.toolDirection.resize( pointInRotation );
    for ( int i = 0; i < pointInRotation; ++i )
    {
        const auto currentRotationAngles_ = rotationAnglesStep_ * float(i) + rotationAngles_;
        res.action.path[i] = calcRealCoord_( translationPos_, currentRotationAngles_ );
        res.toolDirection[i] = calcRealCoord_( Vector3f::plusZ(), currentRotationAngles_ );
    }

    updateRotationAngleAndMatrix_( *inputRotation_ );
   
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

MR::Vector3f GcodeProcessor::calcCoordMotors_()
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

MR::Vector3f GcodeProcessor::calcRealCoord_( const Vector3f& translationPos, const Vector3f& rotationAngles )
{
    Vector3f res = translationPos;
    for ( int i = 0; i < 3; ++i )
    {
        const int axisIndex = rotationAxesOrder_[i];
        if ( axisIndex < 0 || axisIndex > 2 )
            continue;
        const Matrix3f rotationMatrix = Matrix3f::rotation( rotationAxes_[axisIndex], rotationAngles[axisIndex] / 180.f * PI_F );
        res = rotationMatrix * res;
    }
    return res;
}

void GcodeProcessor::updateRotationAngleAndMatrix_( const Vector3f& rotationAngles )
{
    for ( int i = 0; i < 3; ++i )
    {
        if ( rotationAngles[i] == rotationAngles_[i] )
            continue;
        rotationAngles_[i] = rotationAngles[i];
        cacheRotationMatrix_[i] = Matrix3f::rotation( rotationAxes_[i], rotationAngles_[i] / 180.f * PI_F );
    }
}

MR::Vector3f GcodeProcessor::calcRealCoordCached_( const Vector3f& translationPos, const Vector3f& rotationAngles )
{
    for ( int i = 0; i < 3; ++i )
    {
        if ( rotationAngles[i] == rotationAngles_[i] )
            continue;
        rotationAngles_[i] = rotationAngles[i];
        cacheRotationMatrix_[i] = Matrix3f::rotation( rotationAxes_[i], rotationAngles_[i] / 180.f * PI_F );
    }
    return calcRealCoordCached_( translationPos );
}

MR::Vector3f GcodeProcessor::calcRealCoordCached_( const Vector3f& translationPos )
{
    Vector3f res = translationPos;
    for ( int i = 0; i < 3; ++i )
    {
        const int axisIndex = rotationAxesOrder_[i];
        if ( axisIndex < 0 || axisIndex > 2 )
            continue;
        res = cacheRotationMatrix_[axisIndex] * res;
    }
    return res;
}

}
