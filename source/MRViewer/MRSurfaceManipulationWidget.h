#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MREnums.h"
#include "MRMesh/MRBitSet.h"
#include "MRViewer/MRViewport.h"
#include <chrono>

namespace MR
{

/// @brief widget for surface modifying
/// @detail available 3 modes:
/// add (move surface region in direction of normal)
/// remove (move surface region in opposite direction to normal)
/// relax (relax surface region)
class MRVIEWER_CLASS SurfaceManipulationWidget :
    public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener,
                         PostDrawListener>
{
public:
    MRVIEWER_API SurfaceManipulationWidget();
    MRVIEWER_API virtual ~SurfaceManipulationWidget();

    /// widget work modes
    enum class WorkMode
    {
        Add,
        Remove,
        Relax,
        Laplacian,
        Patch
    };

    /// Method for calculating mesh changes
    enum class DeviationCalculationMethod
    {
        PointToPoint, ///< distance between the start and end points
        PointToPlane, ///< distance between the initial plane (starting point and normal to it) and the end point
        ExactDistance ///< distance between the start and end meshes
    };


    /// Mesh change settings
    struct Settings
    {
        WorkMode workMode = WorkMode::Add;
        float radius = 1.f; ///< radius of editing region
        float relaxForce = 0.2f; ///< speed of relaxing, typical values (0 - 0.5]
        float editForce = 1.f; ///< the force of changing mesh
        float sharpness = 50.f; ///< effect of force on points far from center editing area. [0 - 100]
        float relaxForceAfterEdit = 0.25f; ///< force of relaxing modified area after editing (add / remove) is complete. [0 - 0.5], 0 - not relax
        EdgeWeights edgeWeights = EdgeWeights::Cotan; ///< edge weights for Laplacian and Patch
    };

    /// initialize widget according ObjectMesh
    MRVIEWER_API void init( const std::shared_ptr<ObjectMesh>& objectMesh );
    /// reset widget state
    MRVIEWER_API void reset();
    /// lock the mesh region (vertices in this region cannot be moved, added or deleted)
    /// @note boundary edges can be split to improve quality of the patch
    MRVIEWER_API void setFixedRegion( const FaceBitSet& region );

    /// set widget settings (mesh change settings)
    MRVIEWER_API void setSettings( const Settings& settings );
    /// get widget settings 
    MRVIEWER_API const Settings& getSettings() { return settings_; }

    /// mimum radius of editing area.
    MRVIEWER_API float getMinRadius() { return minRadius_; }

    /// get palette used for visualization point shifts
    Palette& palette() { return *palette_; }
    /// update texture used for colorize surface (use after change colorMap in palette)
    MRVIEWER_API void updateTexture();
    /// update texture uv coords used for colorize surface (use after change ranges in palette)
    MRVIEWER_API void updateUVs();
    /// enable visualization of mesh deviations
    MRVIEWER_API void enableDeviationVisualization( bool enable );
    /// set method for calculating mesh changes
    MRVIEWER_API void setDeviationCalculationMethod( DeviationCalculationMethod method );
    /// get method for calculating mesh changes
    MRVIEWER_API DeviationCalculationMethod deviationCalculationMethod() const { return deviationCalculationMethod_; }
    /// checks for a one-to-one correspondence between the vertices of the original grid and the modified one
    MRVIEWER_API bool sameValidVerticesAsInOriginMesh() const { return sameValidVerticesAsInOriginMesh_; }
    /// get min / max point shifts for (usefull for setup palette)
    MRVIEWER_API Vector2f getMinMax();

    /// allow the user to edit parts of object that are hidden in the current view by other objects
    MRVIEWER_API void setIgnoreOcclusion( bool ignore ) { ignoreOcclusion_ = ignore; }
    MRVIEWER_API bool ignoreOcclusion() const { return ignoreOcclusion_; }

    /// restricts editable area to vertices whose normals look into the same half-space as normal under cursor
    void setEditOnlyCodirectedSurface( bool edit ) { editOnlyCodirectedSurface_ = edit; }
    /// get state of an editable region restriction 
    bool isEditOnlyCodirectedSurface() const { return editOnlyCodirectedSurface_; }
protected:
    /// start modifying mesh surface
    MRVIEWER_API bool onMouseDown_( MouseButton button, int modifiers ) override;
    /// stop modifying mesh surface, generate history action
    MRVIEWER_API bool onMouseUp_( MouseButton button, int modifiers ) override;
    /// update
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;
    /// need to visualize bad region (draw grey circle)
    MRVIEWER_API void postDraw_() override;

    /// customize modifiers check on mouse down
    /// @return true if widget consumes event, false if modifiers do not satisfy widget requirements
    MRVIEWER_API virtual bool checkModifiers_( int modifiers ) const { return modifiers == 0; }

    /// called to change mesh with history record
    /// newFaces seems to be useful
    MRVIEWER_API virtual void appendMeshChangeHistory_( std::shared_ptr<Mesh> newMesh, const FaceBitSet& newFaces );

    void reallocData_( size_t size );
    void clearData_();

    void initConnections_();
    void resetConnections_();

    void changeSurface_();
    void updateUVmap_( bool set, bool wholeMesh = false );
    void updateRegion_( const Vector2f& mousePos );
    void abortEdit_();
    /// Laplacian
    void laplacianPickVert_( const PointOnFace& pick );
    void laplacianMoveVert_( const Vector2f& mousePos );

    void updateVizualizeSelection_();

    void updateRegionUVs_( const VertBitSet& region );
    void updateValueChanges_( const VertBitSet& region );
    void updateValueChangesPointToPoint_( const VertBitSet& region );
    void updateValueChangesPointToPlane_( const VertBitSet& region );
    void updateValueChangesExactDistance_( const VertBitSet& region );
    void createLastStableObjMesh_();
    void removeLastStableObjMesh_();

    /// this function is called after all modifications are finished;
    /// if we previously appended SmartChangeMeshPointsAction, then switch it from uncompressed to compressed format to occupy less amount of memory
    void compressChangePointsAction_();

    void updateDistancesAndRegion_( const Mesh& mesh, const VertBitSet& start, VertScalars& distances, VertBitSet& region, const VertBitSet* untouchable );

    Settings settings_;

    std::shared_ptr<ObjectMesh> obj_;
    VertBitSet unchangeableVerts_;
    float minRadius_ = 1.f;
    Vector2f mousePos_; ///< mouse position of last updateRegion_
    VertBitSet activePickedVertices_; ///< vertices that are considered under mouse in curernt frame (could be many in case of fast mouse mouvement)
    VertBitSet singleEditingRegion_;  ///< current (under the cursor) region of tool application
    VertBitSet visualizationRegion_;  ///< vertices of triangles partially or fully highlighted with red
    VertBitSet generalEditingRegion_; ///< united region of tool application since the last mouse down
    VertScalars pointsShift_;
    VertScalars editingDistanceMap_;
    VertScalars visualizationDistanceMap_;
    VertBitSet changedRegion_;
    VertScalars valueChanges_;
    VertScalars lastStableValueChanges_;
    std::shared_ptr<Mesh> originalMesh_; ///< original input mesh
    VertBitSet unknownSign_; ///< cached data to avoid reallocating memory
    std::shared_ptr<ObjectMesh> lastStableObjMesh_;
    bool firstInit_ = true; /// need to save settings in re-initial
    bool badRegion_ = false; /// in selected region less than 3 points

    bool mousePressed_ = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> timePoint_;
    boost::signals2::scoped_connection meshChangedConnection_;
    bool ownMeshChangedSignal_ = false;

    bool connectionsInitialized_ = false;

    /// Laplacian
    VertId touchVertId_; /// we fix this vertex in Laplacian and move it manually
    Vector3f touchVertIniPos_; /// initial position of fixed vertex
    Vector2i storedDown_;
    std::unique_ptr<Laplacian> laplacian_;

    /// prior to add/remove/smooth/deform modification, this action is created and current mesh coordinate are copied here
    class SmartChangeMeshPointsAction;
    std::shared_ptr<SmartChangeMeshPointsAction> historyAction_;

    /// true if historyAction_ is prepared but not yet appended to HistoryStore, which is done on first mouse move
    bool appendHistoryAction_ = false;

    std::shared_ptr<Palette> palette_;
    bool enableDeviationTexture_ = false;
    DeviationCalculationMethod deviationCalculationMethod_ = DeviationCalculationMethod::ExactDistance;
    bool sameValidVerticesAsInOriginMesh_ = true;

    /// allow the user to edit parts of object that are hidden in the current view by other objects
    bool ignoreOcclusion_ = false;
    bool editOnlyCodirectedSurface_ = true;
};

}

