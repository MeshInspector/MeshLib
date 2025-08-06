#pragma once

#include "MRMeshFwd.h"
#include "MRMeshMetrics.h"

MR_EXTERN_C_BEGIN

typedef enum MRFillHoleMetricMultipleEdgesResolveMode
{
    MRFillHoleMetricMultipleEdgesResolveModeNone = 0,
    MRFillHoleMetricMultipleEdgesResolveModeSimple,
    MRFillHoleMetricMultipleEdgesResolveModeStrong
} MRFillHoleParamsMultipleEdgesResolveMode;

/** \struct MRFillHoleParams
  * \brief Parameters structure for mrFillHole\n
  * Structure has some options to control mrFillHole
  *
  * \sa \ref mrFillHole
  * \sa \ref MRFillHoleMetric
  */
typedef struct MRFillHoleParams
{
    /** Specifies triangulation metric\n
      * default for mrFillHole: mrGetCircumscribedFillMetric\n
      * \sa \ref MRFillHoleMetric
      */
    const MRFillHoleMetric* metric;

    /** If true, hole filling will minimize the sum of metrics including boundary edges,
    *   where one triangle was present before hole filling, and another is added during hole filling.
    *   This makes boundary edges same smooth as inner edges of the patch.
    *   If false, edge metric will not be applied to boundary edges, and the patch tends to make a sharper turn there.
    */
    bool smoothBd;

    /// If not nullptr accumulate new faces
    MRFaceBitSet* outNewFaces;

    /** If Strong makes additional efforts to avoid creating multiple edges,
      * in some rare cases it is not possible (cases with extremely bad topology),
      * if you faced one try to use \ref MR::duplicateMultiHoleVertices before \ref MR::fillHole
      *
      * If Simple avoid creating edges that already exist in topology (default)
      *
      * If None do not avoid multiple edges
      */
    MRFillHoleParamsMultipleEdgesResolveMode multipleEdgesResolveMode;

    /** If true creates degenerate faces band around hole to have sharp angle visualization
      * \warning This flag bad for result topology, most likely you do not need it
      */
    bool makeDegenerateBand;

    /** The maximum number of polygon subdivisions on a triangle and two smaller polygons,
      * must be 2 or larger
      */
    int maxPolygonSubdivisions;

    /** Input/output value, if it is present:
      * returns true if triangulation was bad and do not actually fill hole,
      * if triangulation is ok returns false;
      * if it is not present fill hole trivially in case of bad triangulation, (or leaves bad triangulation, depending on metric)
      */
    bool* stopBeforeBadTriangulation;
} MRFillHoleParams;

MRMESHC_API MRFillHoleParams mrFillHoleParamsNew( void );

/** \brief Fills hole in mesh\n
  *
  * Fills given hole represented by one of its edges (having no valid left face),\n
  * uses fillHoleTrivially if cannot fill hole without multiple edges,\n
  * default metric: CircumscribedFillMetric
  *
  * \image html fill/before_fill.png "Before" width = 250cm
  * \image html fill/fill.png "After" width = 250cm
  *
  * Next picture show, how newly generated faces can be smoothed
  * \ref MR::positionVertsSmoothly
  * \ref MR::subdivideMesh
  * \image html fill/fill_smooth.png "Fill with smooth" width = 250cm
  *
  * \param mesh mesh with hole
  * \param a EdgeId which represents hole (should not have valid left FaceId)
  * \param params parameters of hole filling
  *
  * \sa \ref buildCylinderBetweenTwoHoles
  * \sa \ref fillHoleTrivially
  * \sa \ref FillHoleParams
  */
MRMESHC_API void mrFillHole( MRMesh* mesh, MREdgeId a, const MRFillHoleParams* params );

/// fill all holes given by their representative edges in \param as
MRMESHC_API void mrFillHoles( MRMesh* mesh, const MREdgeId* as, size_t asNum, const MRFillHoleParams* params );

/** \struct MRStitchHolesParams
  * \brief Parameters structure for mrBuildCylinderBetweenTwoHoles\n
  * Structure has some options to control mrBuildCylinderBetweenTwoHoles
  */
typedef struct MRStitchHolesParams
{
    /** Specifies triangulation metric\n
      * default for mrBuildCylinderBetweenTwoHoles: getComplexStitchMetric
      * \sa \ref MRFillHoleMetric
      */
    const MRFillHoleMetric* metric;
    /// If not NULL accumulate new faces
    MRFaceBitSet* outNewFaces;
} MRStitchHolesParams;

/** \brief Stitches two holes in Mesh\n
  *
  * Build cylindrical patch to fill space between two holes represented by one of their edges each,\n
  * default metric: ComplexStitchMetric
  *
  * \param mesh mesh with hole
  * \param a EdgeId which represents 1st hole (should not have valid left FaceId)
  * \param b EdgeId which represents 2nd hole (should not have valid left FaceId)
  * \param params parameters of holes stitching
  *
  * \sa \ref mrFillHole
  * \sa \ref MRStitchHolesParams
  */
MRMESHC_API void mrBuildCylinderBetweenTwoHoles( MRMesh* mesh, MREdgeId a, MREdgeId b, const MRStitchHolesParams* params );

MR_EXTERN_C_END
