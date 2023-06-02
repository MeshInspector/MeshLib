#include "MRGcodeProcessor.h"
#include "MRVector2.h"
#include "MRMatrix2.h"
#include <cassert>

namespace MR
{

constexpr float cInch = 2.54f;

//////////////////////////////////////////////////////////////////////////
// GcodeExecutor

void GcodeProcessor::reset()
{
    workPlane_ = WorkPlane::xy;
    toWorkPlaneXf_ = Matrix3f();
    basePoint_ = Vector3f();
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
    if ( command.key == 'f' )
        feedrate_ = command.value;
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

    Vector3f newPoint = calcRealNewCoord_();

    const bool anyCoordReaded = inputCoordsReaded_[0] || inputCoordsReaded_[1] || inputCoordsReaded_[2];
    
    if ( ( moveMode_ == MoveMode::Idle || moveMode_ == MoveMode::Line ) && anyCoordReaded )
        res = moveLine_( newPoint, moveMode_ == MoveMode::Idle );
    else if ( ( moveMode_ == MoveMode::Clockwise || moveMode_ == MoveMode::Counterclockwise ) && (anyCoordReaded || arcCenter_) )
        res = moveArc_( newPoint, moveMode_ == MoveMode::Clockwise );

    basePoint_ = newPoint;
    res.feedrate = feedrate_;

    return res;
}

void GcodeProcessor::resetTemporaryStates_()
{
    inputCoords_ = {};
    inputCoordsReaded_ = Vector3<bool>( false, false, false );
    radius_ = {};
    arcCenter_ = {};
}

GcodeProcessor::MoveAction GcodeProcessor::moveLine_( const Vector3f& newPoint, bool idle )
{
    // MoveAction res({ basePoint_, newPoint }, idle); //fatal error C1001: Internal compiler error.
    MoveAction res;
    res.idle = idle;
    if ( (newPoint - basePoint_).lengthSq() > (accuracy_ * accuracy_) )
        res.path = { basePoint_, newPoint };
    return res;
}

GcodeProcessor::MoveAction GcodeProcessor::moveArc_( const Vector3f& newPoint, bool clockwise )
{
    MoveAction res;
    if ( radius_ )
        res.path = getArcPoints3_( *radius_, basePoint_, newPoint, clockwise, res.errorText );
    else if ( arcCenter_ )
        res.path = getArcPoints3_( basePoint_ + *arcCenter_, basePoint_, newPoint, clockwise, res.errorText );
    else
        res.errorText = "Missing parameters.";

    if ( !res.errorText.empty() )
        res.valid = false;

    return res;
}

void GcodeProcessor::updateWorkPlane_( WorkPlane wp )
{
    workPlane_ = wp;
    constexpr float pi2 = PI2_F;
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

std::vector<Vector2f> GcodeProcessor::getArcPoints2_( const Vector2f& beginPoint, const Vector2f& endPoint, bool clockwise, std::string& errorText )
{
    const float beginLengthSq = beginPoint.lengthSq();
    const float endLengthSq = endPoint.lengthSq();
    const float maxLengthSq = std::max( beginLengthSq, endLengthSq );
    const float deltaLength2 = std::fabs( beginPoint.lengthSq() - endPoint.lengthSq() );
    if ( deltaLength2 >= ( 2.5f * accuracy_ * maxLengthSq ) )
        errorText = "Begin and end radius are different: diff = " + std::to_string( std::sqrt( deltaLength2 ) );

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
    std::vector<Vector2f> res;
    res.push_back( beginPoint );
    for ( int i = 0; i < stepCount; ++i )
    {
        const auto arcPoint = Matrix2f::rotation( angleStep * ( i + 1 ) ) * beginPoint;
        res.push_back( arcPoint );
    }

    return res;
}

std::vector<Vector3f> GcodeProcessor::getArcPoints3_( const Vector3f& center, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise, std::string& errorText )
{
    const Vector3f c3 = toWorkPlaneXf_ * center;

    const Vector3f b3 = toWorkPlaneXf_ * beginPoint - c3;
    const Vector2f b2 = { b3.x, b3.y };

    const Vector3f e3 = toWorkPlaneXf_ * endPoint - c3;
    const Vector2f e2 = { e3.x, e3.y };

    const bool helical = std::fabs( b3.z - e3.z ) > accuracy_;

    const Matrix3f toWorldXf = toWorkPlaneXf_.inverse();
    auto res2 = getArcPoints2_( b2, e2, clockwise, errorText );
    std::vector<Vector3f> res3( res2.size() );
    const float zStep = res2.size() > 1 ? ( e3.z - b3.z ) / ( res2.size() - 1 ) : 0.f;
    for ( int i = 0; i < res2.size(); ++i )
        res3[i] = toWorldXf * ( Vector3f( res2[i].x, res2[i].y, helical ? b3.z + zStep * i : b3.z ) + c3 );

    return res3;
}

std::vector<MR::Vector3f> GcodeProcessor::getArcPoints3_( float r, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise, std::string& errorText )
{
    if ( r < accuracy_ )
    {
        errorText = "Wrong radius";
        return { beginPoint, endPoint };
    }

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
    const Vector3f c3 = { c2.x, c2.y, b3.z };


    const Matrix3f toWorldXf = toWorkPlaneXf_.inverse();
    auto res2 = getArcPoints2_( b2 - c2, e2 - c2, clockwise, errorText );
    std::vector<Vector3f> res3( res2.size() );
    const float zStep = res2.size() > 1 ? ( e3.z - b3.z ) / ( res2.size() - 1 ) : 0.f;
    for ( int i = 0; i < res2.size(); ++i )
        res3[i] = toWorldXf * ( Vector3f( res2[i].x, res2[i].y, helical ? b3.z + zStep * i : b3.z ) + c3 );

    return res3;
}

MR::Vector3f GcodeProcessor::calcRealNewCoord_()
{
    Vector3f res = mult( inputCoords_, scaling_ );
    if ( inches_ )
        res *= cInch;
    if ( absoluteCoordinates_ )
    {
        for ( int i = 0; i < 3; ++i )
        {
            if ( !inputCoordsReaded_[i] )
                res[i] = basePoint_[i];
        }
    }
    else
        res += basePoint_;

    return res;
}

}
