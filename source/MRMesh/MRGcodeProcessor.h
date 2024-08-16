#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRMatrix3.h"
#include "MRCNCMachineSettings.h"
#include <vector>
#include <string>
#include <optional>
#include <functional>


namespace MR
{

// class to process g-code source and generate toolpath
class MRMESH_CLASS GcodeProcessor
{
public:

    template<typename Vec>
    struct BaseAction
    {
        // tool movement parsed from gcode
        std::vector<Vec> path;
        // parser warning
        std::string warning;
    };
    using BaseAction2f = BaseAction<Vector2f>;
    using BaseAction3f = BaseAction<Vector3f>;
    // structure that stores information about the movement of the tool, specified by some string of commands
    struct MoveAction
    {
        BaseAction3f action;
        std::vector<Vector3f> toolDirection; // tool direction for each point from action.path
        bool idle = true;
        float feedrate = 100.f;
        // return true if operation was parsed without warnings
        bool valid() const { return action.warning.empty(); }
        operator bool() const { return valid(); }
    };

    // reset internal states
    MRMESH_API void reset();

    // set g-code source
    MRMESH_API void setGcodeSource( const GcodeSource& gcodeSource );

    // process all lines g-code source and generate corresponding move actions
    MRMESH_API std::vector<MoveAction> processSource();

    struct Command
    {
        char key; // in lowercase
        float value;
    };

    // process all commands from one line g-code source and generate corresponding move action;
    // \param externalStorage to avoid memory allocation on each line
    MRMESH_API MoveAction processLine( const std::string_view& line, std::vector<Command> & externalStorage );

    // settings
    MRMESH_API void setCNCMachineSettings( const CNCMachineSettings& settings );
    const CNCMachineSettings& getCNCMachineSettings() { return cncSettings_; }

private:
    enum class WorkPlane
    {
        xy,
        zx,
        yz
    };

    // parse program methods
    static void parseFrame_( const std::string_view& frame, std::vector<Command> & outCommands );
    void applyCommand_( const Command& command );
    void applyCommandG_( const Command& command );
    MoveAction generateMoveAction_();
    MoveAction generateReturnToHomeAction_();
    void resetTemporaryStates_();

    // g-command actions

    // g0, g1
    MoveAction moveLine_( const Vector3f& newPoint, const Vector3f& newAngles );
    // g2, g3
    MoveAction moveArc_( const Vector3f& newPoint, const Vector3f& newAngles, bool clockwise );

    // g17, g18, g19
    void updateWorkPlane_( WorkPlane wp );

    // g51
    void updateScaling_();

    // g-command helper methods

    // sample arc points in 2d by begin point, end point and center in (0, 0)
    BaseAction2f getArcPoints2_( const Vector2f& beginPoint, const Vector2f& endPoint, bool clockwise );
    // sample arc points in 3d by begin point, end point and center
    BaseAction3f getArcPoints3_( const Vector3f& center, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise );
    // sample arc points in 3d by begin point, end point and radius
    // r > 0 : angle - [0, 180]
    // r < 0 : angle - (0, 360)
    BaseAction3f getArcPoints3_( float r, const Vector3f& beginPoint, const Vector3f& endPoint, bool clockwise );

    // sample arc points of tool movement during rotation
    MoveAction getToolRotationPoints_( const Vector3f& newRotationAngles );

    Vector3f calcNewTranslationPos_();
    Vector3f calcNewRotationAngles_();
    Vector3f calcRealCoord_( const Vector3f& translationPos, const Vector3f& rotationAngles );
    void updateRotationAngleAndMatrix_( const Vector3f& rotationAngles );
    Vector3f calcRealCoordCached_( const Vector3f& translationPos, const Vector3f& rotationAngles );
    Vector3f calcRealCoordCached_( const Vector3f& translationPos );

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
        ReturnToHome,
        Scaling
    };

    // internal states (such as current position and different work modes)
    CoordType coordType_ = CoordType::Movement;
    MoveMode moveMode_ = MoveMode::Idle;
    WorkPlane workPlane_ = WorkPlane::xy;
    Matrix3f toWorkPlaneXf_; // work plane for calculation arc movement
    Vector3f translationPos_; // last values of x, y, z (translation. positions of linear movement motors)
    Vector3f rotationAngles_; // last values of a, b, c (rotation. angles of circular movement motors)
    bool absoluteCoordinates_ = true; //absolute coordinates or relative coordinates
    Vector3f scaling_ = Vector3f::diagonal( 1.f );
    bool inches_ = false;
    float feedrate_ = 100.f;
    float feedrateMax_ = 0.f;

    // cached data
    std::array<Matrix3f, 3> cacheRotationMatrix_; // cached rotation matrices. to avoid calculating for each line (without rotation)

    // input data (data entered in last line)
    Vector3f inputCoords_; // x, y, z
    Vector3b inputCoordsReaded_; // x, y, z was read
    std::optional<float> radius_; // r
    std::optional<Vector3f> arcCenter_; // i, j, k
    Vector3f inputRotation_; // a, b, c
    Vector3b inputRotationReaded_; // a, b, c was read

    std::vector<std::string_view> gcodeSource_; // string list with sets of command (frames)

    // internal / machine settings
    float accuracy_ = 1.e-3f;
    CNCMachineSettings cncSettings_;
    std::vector<int> rotationAxesOrderMap_ = {0, 1, 2}; // mapping axis sequence number to axis number in storage

};

}
