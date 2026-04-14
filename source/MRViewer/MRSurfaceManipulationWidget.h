#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MREnums.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVector2.h"
#include <cfloat>

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
        float editForce = 1.f; ///< material thickness added or removed to the surface
        float sharpness = 50.f; ///< effect of force on points far from center editing area. [0 - 100]
        float relaxForceAfterEdit = 0.25f; ///< force of relaxing modified area after editing (add / remove) is complete. [0 - 0.5], 0 - not relax
        EdgeWeights edgeWeights = EdgeWeights::Cotan; ///< edge weights for Laplacian and Patch
        VertexMass vmass = VertexMass::NeiArea; ///< vertex weights for Laplacian and Patch
        bool laplacianBasedAddRemove = false; ///< if true in Add/Remove modes, the modification will be done using Laplacian solver, where the closest vertices will be attracted toward mouse cursor to form ideal ridges or grooves
        bool subdivideGrooves = false; ///< if true in Add/Remove modes, changed parts of mesh will be subdivided on mouse up
        bool mimicPatch = false; /// if true in Patch mode mixes `CloseSurfaceFillMetric` and disables smoothing
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

    /// minimum radius of editing area.
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

    /// returns true if the current object's mesh has the same topology as original input mesh (and vertices with same IDs can be compared)
    MRVIEWER_API bool sameOriginalMeshTopology() const { return sameOriginalMeshTopology_; }

    /// get min / max point shifts for (useful for setup palette)
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

    /// called to change mesh data with history record
    /// newFaces seems to be useful
    MRVIEWER_API virtual void appendMeshDataChangeHistory_( ObjectMeshData&& newMeshData, const FaceBitSet& newFaces );

    void reallocData_( size_t size );
    void clearData_();

    void initConnections_();
    void resetConnections_();

    void changeSurface_();
    void updateUVmap_( bool set, bool wholeMesh = false );
    void updateRegion_( const Vector2f& mousePos );
    void invalidateMetricsCache_();
    void abortEdit_();

    /// Laplacian
    void initLaplacian_( RememberShape rs ); // for singleEditingRegion_
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
    /// if we previously appended VersatileChangeMeshPointsAction, then switch it from uncompressed to compressed format to occupy less amount of memory
    void compressChangePointsAction_();

    void subdivideAfterAddRemove_();

    void updateDistancesAndRegion_( const Mesh& mesh, const std::vector<MeshTriPoint>& start, VertScalars& distances, VertBitSet& region, const VertBitSet* untouchable );

    Settings settings_;

    std::shared_ptr<ObjectMesh> obj_;
    VertBitSet unchangeableVerts_;
    float minRadius_ = 1.f;
    Vector2f mousePos_; ///< mouse position of last updateRegion_
    std::vector<MeshTriPoint> pointsUnderMouse_; ///< mesh points under mouse in the current frame (could be many in case of fast mouse movement)
    VertBitSet singleEditingRegion_;  ///< current (under the cursor) region of tool application
    VertBitSet visualizationRegion_;  ///< vertices of triangles partially or fully highlighted with red
    VertBitSet generalEditingRegion_; ///< united region of tool application since the last mouse down
    VertScalars pointsShift_;
    VertScalars editingDistanceMap_;
    VertScalars visualizationDistanceMap_;
    VertScalars valueChanges_;
    VertScalars lastStableValueChanges_;
    std::shared_ptr<Mesh> originalMesh_; ///< original input mesh
    VertBitSet unknownSign_; ///< cached data to avoid reallocating memory
    std::shared_ptr<ObjectMesh> lastStableObjMesh_;
    bool firstInit_ = true; /// need to save settings in re-initial
    bool badRegion_ = false; /// in selected region less than 3 points

    bool mousePressed_ = false;

    boost::signals2::scoped_connection meshChangedConnection_;
    bool ownMeshChangedSignal_ = false;

    bool connectionsInitialized_ = false;

    /// Laplacian
    VertId touchVertId_; /// we fix this vertex in Laplacian and move it manually
    Vector3f touchVertIniPos_; /// initial position of fixed vertex
    Vector2i storedDown_;
    std::unique_ptr<Laplacian> laplacian_;

    /// these are all vertices, which will are attracted to be under mouse considering material width since last mouse down
    VertBitSet pickedVerts_;

    struct PickedVertData
    {
        Vector3f target; // attraction point
        float minMouseDistSq = FLT_MAX; // minimal distance from a point under mouse to this vertex
    };

    /// same vertices as in pickedVerts_ mapped to PickedVertData
    HashMap<VertId, PickedVertData> pickedVertsToData_;

    /// prior to add/remove/smooth/deform modification, this action is created and 
    /// the current mesh coordinates are copied here
    std::shared_ptr<VersatileChangeMeshPointsAction> historyAction_;

    /// true if historyAction_ is prepared but not yet appended to HistoryStore, which is done on first mouse move
    bool appendHistoryAction_ = false;

    std::shared_ptr<Palette> palette_;
    bool enableDeviationTexture_ = false;
    DeviationCalculationMethod deviationCalculationMethod_ = DeviationCalculationMethod::ExactDistance;
    bool sameOriginalMeshTopology_ = true;

    /// allow the user to edit parts of object that are hidden in the current view by other objects
    bool ignoreOcclusion_ = false;
    bool editOnlyCodirectedSurface_ = true;
};

}

