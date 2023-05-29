#include "MRGcodeExecutor.h"
#include "MRVector2.h"
#include "MRMatrix2.h"

namespace MR
{

constexpr float cInch = 2.54f;

//////////////////////////////////////////////////////////////////////////
// GcodeExecutor

void GcodeExecutor::reset()
{
    workPlane_ = WorkPlane::xy;
    toWorkPlaneXf_ = Matrix3f();
    basePoint_ = Vector3f();
    absoluteCoordinates_ = true;
    scaling_ = Vector3f::diagonal( 1.f );
    inches_ = false;
    frameList_.clear();
}

void GcodeExecutor::setFrameList( const GcodeSource& frameList )
{
    reset();
    frameList_.resize( frameList.size() );
    for ( int i = 0; i < frameList.size(); ++i )
        frameList_[i] = frameList[i];
}

std::vector<MR::GcodeExecutor::MoveAction> GcodeExecutor::executeProgram()
{
    if ( frameList_.empty() )
        return {};

    std::vector<MoveAction> res( frameList_.size() );
    for ( int i = 0; i < frameList_.size(); ++i )
        res[i] = executeFrame( frameList_[i] );

    return res;
}

GcodeExecutor::MoveAction GcodeExecutor::executeFrame( const std::string_view& frame )
{
    if ( frame.empty() )
        return {};

    auto commands = parseFrame_( frame );
    if ( commands.empty() )
        return {};

    resetTemporaryStates_();

    // TODO add check is valid command set

    for ( int i = 0; i < commands.size(); ++i )
        applyCommand_( commands[i] );

    if ( coordType_ == CoordType::Movement )
        return applyMove_();

    if ( coordType_ == CoordType::Scaling )
        updateScaling_();

    coordType_ = CoordType::Movement;
    return {};
}

std::vector<GcodeExecutor::Command> GcodeExecutor::parseFrame_( const std::string_view& frame )
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

void GcodeExecutor::applyCommand_( const Command& command )
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

void GcodeExecutor::applyCommandG_( const Command& command )
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

GcodeExecutor::MoveAction GcodeExecutor::applyMove_()
{
    MoveAction res;
    res.idle = true;
    res.valid = false;

    Vector3f newPoint = calcRealNewCoord_();
    if ( newPoint == basePoint_ )
        return res;

    if ( moveMode_ == MoveMode::Idle || moveMode_ == MoveMode::Line )
        res = moveLine_( newPoint, moveMode_ == MoveMode::Idle );
    if ( moveMode_ == MoveMode::Clockwise || moveMode_ == MoveMode::Counterclockwise )
        res = moveArc_( newPoint, moveMode_ == MoveMode::Clockwise );

    basePoint_ = newPoint;
    res.feedrate = feedrate_;

    return res;
}

void GcodeExecutor::resetTemporaryStates_()
{
    inputCoords_ = {};
    inputCoordsReaded_ = Vector3<bool>( false, false, false );
    radius_ = {};
    arcCenter_ = {};
}

GcodeExecutor::MoveAction GcodeExecutor::moveLine_( const Vector3f& newPoint, bool idle )
{
    // MoveAction res({ basePoint_, newPoint }, idle); //fatal error C1001: Internal compiler error.
    MoveAction res;
    res.path = { basePoint_, newPoint };
    res.idle = idle;
    return res;
}

GcodeExecutor::MoveAction GcodeExecutor::moveArc_( const Vector3f& newPoint, bool clockwise )
{
    MoveAction res;
    if ( radius_ )
        res.path = getArcPoints3_( *radius_, basePoint_, newPoint, clockwise );
    else if ( arcCenter_ )
        res.path = getArcPoints3_( basePoint_ + *arcCenter_, basePoint_, newPoint, clockwise );

    if ( res.path.empty() )
    {
        res.path = { basePoint_, newPoint };
        res.idle = false;
        res.valid = false;
    }

    return res;
}

void GcodeExecutor::updateWorkPlane_( WorkPlane wp )
{
    workPlane_ = wp;
    constexpr float pi2 = PI2_F;
    if ( workPlane_ == WorkPlane::zx )
        toWorkPlaneXf_ = Matrix3f::rotation( Vector3f::plusZ(), pi2 ) * Matrix3f::rotation( Vector3f::plusX(), pi2 );
    else if ( workPlane_ == WorkPlane::yz )
        toWorkPlaneXf_ = Matrix3f::rotation( Vector3f::plusY(), -pi2 ) * Matrix3f::rotation( Vector3f::plusX(), -pi2 );
    else
        toWorkPlaneXf_ = Matrix3f();
}

void GcodeExecutor::updateScaling_()
{
    for ( int i = 0; i < 3; ++i )
    {
        if ( inputCoordsReaded_[i] && inputCoords_[i] != 0 )
            scaling_[i] = inputCoords_[i];
    }
}

std::vector<Vector2f> GcodeExecutor::getArcPoints2_( const Vector2f& beginPoint, const Vector2f& endPoint, bool clockwise )
{
    if ( std::fabs( beginPoint.lengthSq() - endPoint.lengthSq() ) >= 1.e-2f ) // equalityAccuracy_**2 ?
    {
        return {};
    }

    const Vector2f v1 = beginPoint / beginPoint.length();
    const Vector2f v2 = endPoint / endPoint.length();
    float beginAngle = std::atan2( v1.y, v1.x );
    float endAngle = std::atan2( v2.y, v2.x );
    if ( clockwise && ( beginAngle < endAngle ) )
        beginAngle += PI_F * 2.f;
    else if ( !clockwise && beginAngle > endAngle )
        endAngle += PI_F * 2.f;

    const int stepCount = std::clamp( int( std::ceil( ( endAngle - beginAngle ) / ( 6.f / 180.f * PI_F ) ) ), 10, 60 );
    const float angleStep = ( endAngle - beginAngle ) / stepCount;
    const auto rotM = Matrix2f::rotation( angleStep );
    std::vector<Vector2f> res;
    Vector2f arcPoint = beginPoint;
    res.push_back( arcPoint );
    for ( int i = 0; i < stepCount; ++i )
    {
        arcPoint = rotM * arcPoint;
        res.push_back( arcPoint );
    }

    return res;
}

std::vector<Vector3f> GcodeExecutor::getArcPoints3_( const Vector3f& center, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise )
{
    const Vector3f c3 = toWorkPlaneXf_ * center;

    const Vector3f b3 = toWorkPlaneXf_ * beginPoint - c3;
    if ( std::abs( b3.z ) > 1.e-3f )
        return {};
    const Vector2f b2 = { b3.x, b3.y };

    const Vector3f e3 = toWorkPlaneXf_ * endPoint - c3;
    if ( std::abs( e3.z ) > 1.e-3f )
        return {};
    const Vector2f e2 = { e3.x, e3.y };

    const Matrix3f toWorldXf = toWorkPlaneXf_.inverse();
    auto res2 = getArcPoints2_( b2, e2, clockwise );
    std::vector<Vector3f> res3( res2.size() );
    for ( int i = 0; i < res2.size(); ++i )
        res3[i] = toWorldXf * ( Vector3f( res2[i].x, res2[i].y, 0.f ) + c3 );

    return res3;
}

std::vector<MR::Vector3f> GcodeExecutor::getArcPoints3_( float r, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise )
{
    assert( r != 0 );
    if ( r == 0.f )
        return {};

    const Vector3f b3 = toWorkPlaneXf_ * beginPoint;
    const Vector2f b2 = { b3.x, b3.y };

    const Vector3f e3 = toWorkPlaneXf_ * endPoint;
    const Vector2f e2 = { e3.x, e3.y };

    if ( std::abs( e3.z - b3.z ) > 1.e-3f )
        return {};

    const Vector2f middlePoint = ( b2 + e2 ) / 2.f;
    const Vector2f middleVec = middlePoint - b2;
    const Vector2f middleNormal = ( Matrix2f::rotation( -PI2_F ) * middleVec ).normalized();

    const float normalLenght = std::sqrt( r * r - middleVec.lengthSq() );
    const Vector2f c2 = middlePoint + middleNormal * normalLenght * ( clockwise == ( r > 0.f ) ? 1.f : -1.f );
    const Vector3f c3 = { c2.x, c2.y, b3.z };


    const Matrix3f toWorldXf = toWorkPlaneXf_.inverse();
    auto res2 = getArcPoints2_( b2 - c2, e2 - c2, clockwise );
    std::vector<Vector3f> res3( res2.size() );
    for ( int i = 0; i < res2.size(); ++i )
        res3[i] = toWorldXf * ( Vector3f( res2[i].x, res2[i].y, 0.f ) + c3 );

    return res3;
}

MR::Vector3f GcodeExecutor::calcRealNewCoord_()
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
