#pragma once
#include "MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMatrix3.h"
#include <vector>
#include <string>
#include <optional>


namespace MR
{

// class to process g-code source and generate toolpath
class MRMESH_CLASS GcodeProcessor
{
public:
    // structure that stores information about the movement of the tool, specified by some string of commands
    struct MoveAction
    {
        std::vector<Vector3f> path;
        bool idle = false;
        bool valid = true;
        float feedrate = 100.f;
        std::string errorText;
    };

    // reset internal states
    MRMESH_API void reset();

    // set g-code source
    MRMESH_API void setGcodeSource( const GcodeSource& gcodeSource );
    // process all lines g-code source and generate corresponding move actions
    MRMESH_API std::vector<MoveAction> processSource();
    // process all commands from one line g-code source and generate corresponding move action
    MRMESH_API MoveAction processLine( const std::string_view& line );

private:

    struct Command
    {
        char key; // in lowercase
        float value;
    };
    enum class WorkPlane
    {
        xy,
        zx,
        yz
    };

    // parse program methods
    std::vector<Command> parseFrame_( const std::string_view& frame );
    void applyCommand_( const Command& command );
    void applyCommandG_( const Command& command );
    MoveAction generateMoveAction_();
    void resetTemporaryStates_();

    // g-command actions

    // g0, g1
    MoveAction moveLine_( const Vector3f& newPoint, bool idle );
    MoveAction moveArc_( const Vector3f& newPoint, bool clockwise );

    // g17, g18, g19
    void updateWorkPlane_( WorkPlane wp );

    // g51
    void updateScaling_();

    // g-command helper methods

    // sample arc points in 2d by begin point, end point and center in (0, 0)
    std::vector<MR::Vector2f> getArcPoints2_(const Vector2f& beginPoint, const Vector2f& endPoint, bool clockwise, std::string& errorText );
    // sample arc points in 3d by begin point, end point and center
    std::vector<MR::Vector3f> getArcPoints3_( const Vector3f& center, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise, std::string& errorText );
    // sample arc points in 3d by begin point, end point and radius
    // r > 0 : angle - [0, 180]
    // r < 0 : angle - (0, 360)
    std::vector<MR::Vector3f> getArcPoints3_( float r, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise, std::string& errorText );

    Vector3f calcRealNewCoord_();

    // mode of instrument movement
    enum class MoveMode
    {
        Idle, // fast idle movement (linear)
        Line, // working linear movement
        Clockwise, // working movement in an arc clockwise
        Counterclockwise // working movement in an arc counterclockwise
    };
    // type of coordinates to be entered.
    // movement - repeatable.
    // other - after execution, it will return to movement
    enum class CoordType
    {
        Movement,
        Scaling
    };
    CoordType coordType_ = CoordType::Movement;
    MoveMode moveMode_ = MoveMode::Idle;
    WorkPlane workPlane_ = WorkPlane::xy;
    Matrix3f toWorkPlaneXf_;
    Vector3f basePoint_;
    bool absoluteCoordinates_ = true; //absolute coordinates or relative coordinates
    Vector3f scaling_ = Vector3f::diagonal( 1.f );
    bool inches_ = false;
    float feedrate_ = 100.f;

    // temporary states
    Vector3f inputCoords_;
    Vector3<bool> inputCoordsReaded_;
    std::optional<float> radius_;
    std::optional<Vector3f> arcCenter_;

    std::vector<std::string_view> gcodeSource_; // string list with sets of command (frames)

    // internal settings
    float accuracy_ = 1.e-3f;
};

}
