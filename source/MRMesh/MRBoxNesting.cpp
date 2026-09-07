#include "MRBoxNesting.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRBestFit.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"
#include <queue>

namespace
{
constexpr float cSafeExpansion = 1.01f;
}

namespace MR
{

namespace Nesting
{

struct PlacedSocket
{
    int parentSocketId = -1; // -1 means root box
    Vector3i pos;
    Box3f box;
    ObjId objId;
    bool rotated{ false };
};

struct QueueNode
{
    int socketId = -1;
    double cost = -DBL_MAX;
    int numObjs{ 0 };

    friend bool operator <( const QueueNode& a, const QueueNode& b )
    {
        return std::tie( a.numObjs, a.cost ) < std::tie( b.numObjs, b.cost );
    }
};

class BoxNester
{
public:
    BoxNester( const Vector<Vector3f, ObjId>& sizes, const BoxNestingParams& params );

    /// suggest that all objs has box.min at zero
    Vector<NestingResult, ObjId> run();
private:
    BoxNestingParams params_;
    Vector<Vector3f, ObjId> sizes_;
    Vector<ObjId, ObjId> sortMapNew2Old_;

    std::vector<PlacedSocket> socketTree_;
    std::priority_queue<QueueNode> candidatesQ_;

    double calcCost_( const PlacedSocket& leafSocket ) const;
    bool isPlacementValid_( const PlacedSocket& socket ) const;

    bool testPreNested_( const Box3f& curBox ) const;
    bool addCustomToQ_( int parentSocketId, ObjId parentObjId, int& iterNumber );

    void init_();
    bool addCandidateCorners_( int parentSocketId, ObjId parentObjId, const Box3f& parentBox, int& iterNumber );
    bool addSingleCorner_( int parentSocketId, ObjId parentObjId, const Vector3f& pos, uint8_t cornerId, int& iterNumber );
    void addToQ_( PlacedSocket&& socket );

    // takenOrientationId: 8 possible orientations (4 corners * 2 flips, see comment in `calcBox_` function)
    PlacedSocket createSocket_( int parentSocketId, ObjId oId, const Vector3f& cornerPos, uint8_t applicationCornerId, uint8_t takenOrientationId );
    void addInterval_( Box3f& box, uint8_t thisCorner, uint8_t parentCorner );
    Box3f calcBox_( const Vector3f& pos, const Vector3f& size, uint8_t takenOrientation ) const;

    const PlacedSocket* processQ_();
};

Box3f BoxNester::calcBox_( const Vector3f& pos, const Vector3f& sizeP, uint8_t takenOrientation ) const
{
    auto size = sizeP;
    if ( takenOrientation % 2 == 1 )
        std::swap( size.x, size.y );

    Box3f box;
    auto applicationCornerId = takenOrientation / 2;
    for ( int i = 0; i < 3; ++i )
    {
        bool posMax = bool( applicationCornerId & ( 1 << i ) );
        box.min[i] = posMax ? pos[i] - size[i] : pos[i];
        box.max[i] = posMax ? pos[i] : pos[i] + size[i];
    }
    return box;
}

bool BoxNester::isPlacementValid_( const PlacedSocket& socket ) const
{
    if ( !params_.baseParams.nest.insignificantlyExpanded().contains( socket.box ) )
        return false;

    Vector3f expansion = Vector3f::diagonal( params_.baseParams.minInterval * 0.9f );
    auto curBox = socket.box.expanded( expansion );

    auto pSocketId = socket.parentSocketId;
    while ( pSocketId != -1 )
    {
        const auto& pSocket = socketTree_[pSocketId];
        if ( pSocket.box.intersects( curBox ) )
            return false;
        pSocketId = pSocket.parentSocketId;
    }
    return testPreNested_( curBox );
}

bool BoxNester::testPreNested_( const Box3f& curBox ) const
{
    if ( !params_.options.preNestedVolumes )
        return true;
    for ( const auto& box : *params_.options.preNestedVolumes )
        if ( box.intersects( curBox ) )
            return false;
    return true;
}

bool BoxNester::addCustomToQ_( int parentSocketId, ObjId parentObjId, int& iterNumber )
{
    if ( !params_.options.additinalSocketCorners )
        return true;
    for ( auto [pos, corner] : *params_.options.additinalSocketCorners )
    {
        if ( !addSingleCorner_( parentSocketId, parentObjId, pos, corner, iterNumber ) )
            return false;
    }
    return true;
}

void BoxNester::addToQ_( PlacedSocket&& socket )
{
    if ( !isPlacementValid_( socket ) )
        return;
    int id = int( socketTree_.size() );
    socketTree_.push_back( std::move( socket ) );
    candidatesQ_.push( { id, calcCost_( socketTree_.back() ),socketTree_.back().objId } );
}

Nesting::PlacedSocket BoxNester::createSocket_( int parentSocketId, ObjId oId, const Vector3f& cornerPos, uint8_t applicationCornerId, uint8_t takenOrientationId )
{
    PlacedSocket socket{
        .parentSocketId = parentSocketId,
        .box = calcBox_( cornerPos,sizes_[oId],takenOrientationId ),
        .objId = oId,
        .rotated = ( takenOrientationId % 2 == 1 )
    };
    addInterval_( socket.box, takenOrientationId / 2, applicationCornerId );
    return socket;
}

void BoxNester::addInterval_( Box3f& box, uint8_t thisCorner, uint8_t parentCorner )
{
    auto xorCorner = thisCorner ^ parentCorner;
    Vector3f addition;
    for ( int i = 0; i < 3; ++i )
    {
        uint8_t bit = 1 << i;
        if ( bool( xorCorner & bit ) )
            addition[i] += ( bool( parentCorner & bit ) ? params_.baseParams.minInterval : -params_.baseParams.minInterval );
    }
    box.min += addition;
    box.max += addition;
}

void BoxNester::init_()
{
    PlacedSocket rootSocket = createSocket_( -1, ObjId( 0 ), params_.baseParams.nest.min, 0, 0 );
    addToQ_( std::move( rootSocket ) );

    auto nestSize = params_.baseParams.nest.size();
    if ( std::abs( nestSize.x - nestSize.y ) > params_.baseParams.minInterval && params_.options.allowRotation ) // if nest's sides are not equal try both rotations of obj
    {
        rootSocket = createSocket_( -1, ObjId( 0 ), params_.baseParams.nest.min, 0, 1 );
        addToQ_( std::move( rootSocket ) );
    }

    int numIters = 0;
    addCustomToQ_( -1, ObjId(), numIters );
}

bool BoxNester::addCandidateCorners_( int parentSocketId, ObjId topSocketObjId, const Box3f& box, int& iter )
{
    uint8_t maxCornerId = params_.options.allow3dNesting ? 8 : 4;
    for ( uint8_t ac = 0; ac < maxCornerId; ++ac )
    {
        Vector3b cornerId;
        for ( uint8_t i = 0; i < 3; ++i )
            cornerId[i] = bool( ac & ( 1 << i ) );
        Vector3f corner = box.corner( cornerId );
        if ( !addSingleCorner_( parentSocketId, topSocketObjId, corner, ac, iter ) )
            return false;
    }
    return true;
}

bool BoxNester::addSingleCorner_( int parentSocketId, ObjId topSocketObjId, const Vector3f& corner, uint8_t ac, int& iter )
{
    uint8_t maxCornerId = params_.options.allow3dNesting ? 8 : 4;
    for ( uint8_t tc = 0; tc < maxCornerId; ++tc )
    {
        if ( params_.options.checkLessCombinations )
        {
            if ( tc != 0 )
                continue; // only allow 0b000 application
            if ( ac != 0b001 && ac != 0b010 && ac != 0b100 )
                continue; // only allow 0b001, 0b010, 0b100 corners
        }
        addToQ_( createSocket_( parentSocketId, topSocketObjId + 1, corner, ac, 2 * tc ) );
        if ( ++iter > params_.options.iterationLimit )
            return false;
        if ( params_.options.allowRotation )
        {
            addToQ_( createSocket_( parentSocketId, topSocketObjId + 1, corner, ac, 2 * tc + 1 ) );
            if ( ++iter > params_.options.iterationLimit )
                return false;
        }
    }
    return true;
}

BoxNester::BoxNester( const Vector<Vector3f, ObjId>& sizes, const BoxNestingParams& params ) :
    params_{ params }
{
    if ( !params_.options.priorityMetric )
        params_.options.priorityMetric = getNestPostionMinPriorityMetric( params_.baseParams.nest );

    std::vector<std::pair<Vector3f, ObjId>> forSort( sizes.size() );
    sizes_.resize( sizes.size() );
    sortMapNew2Old_.resize( sizes.size() );
    for ( int i = 0; i < sizes.size(); ++i )
    {
        forSort[i] = { sizes[ObjId( i )],ObjId( i ) };
    }
    if ( params_.options.volumeBasedOrder )
    {
        std::sort( forSort.begin(), forSort.end(), [] ( const auto& l, const auto& r )
        {
            return l.first.x * l.first.y * l.first.z > r.first.x * r.first.y * r.first.z;
        } );
    }
    else
    {
        // Use same size for all models if no sort
        /*
        auto it = std::min_element( forSort.begin(), forSort.end(), [] ( const auto& l, const auto& r )
        {
            return l.first.x * l.first.y * l.first.z > r.first.x * r.first.y * r.first.z;
        } );
        if ( it != forSort.end() )
            for ( auto& [s, _] : forSort )
                s = it->first;
        */
    }
    for ( int i = 0; i < sizes.size(); ++i )
    {
        sizes_[ObjId( i )] = forSort[i].first;
        sortMapNew2Old_[ObjId( i )] = forSort[i].second;
    }
}

Vector<NestingResult, ObjId> BoxNester::run()
{
    init_();
    auto lastSocket = processQ_();

    float expansionShiftCompensation = 1.0f;
    if ( params_.options.expansionFactor )
    {
        // we expand sizes by expansionFactor*1.01 so we need to compensate it back in rotation case
        expansionShiftCompensation = float( 1.0 / ( params_.options.expansionFactor->y * cSafeExpansion ) );
    }

    Vector<NestingResult, ObjId> result( sizes_.size() );
    while ( lastSocket )
    {
        auto& res = result[sortMapNew2Old_[lastSocket->objId]];
        res.nested = true;
        const auto& box = lastSocket->box;
        if ( !lastSocket->rotated )
        {
            res.xf = AffineXf3f::translation( box.min );
        }
        else
        {
            res.xf = AffineXf3f::translation( box.min ) *
                AffineXf3f::translation( Vector3f( sizes_[lastSocket->objId].y * expansionShiftCompensation, 0, 0 ) ) *
                AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusX(), Vector3f::plusY() ) );
        }

        if ( lastSocket->parentSocketId != -1 )
            lastSocket = &socketTree_[lastSocket->parentSocketId];
        else
            lastSocket = nullptr;
    }
    return result;
}

double BoxNester::calcCost_( const PlacedSocket& leafSocket ) const
{
    if ( !params_.options.priorityMetric )
    {
        assert( false );
        return 0.0;
    }

    params_.options.priorityMetric->init( leafSocket.box );
    const PlacedSocket* socket = &leafSocket;
    while ( socket )
    {
        params_.options.priorityMetric->addNested( socket->box );
        socket = socket->parentSocketId == -1 ? nullptr : &socketTree_[socket->parentSocketId];
    }
    if ( params_.options.preNestedVolumes )
    {
        for ( const auto& box : *params_.options.preNestedVolumes )
        {
            params_.options.priorityMetric->addNested( box );
        }
    }
    return params_.options.priorityMetric->complete();
}

/*
Vector3i BoxNester::calcPos_( const Vector3i& parentPos, uint8_t pc, uint8_t tc )
{
    Vector3i pos = parentPos;
    for ( uint8_t i = 0; i < 3; ++i )
    {
        uint8_t bit = 1 << i;
        bool bitPV = bool( pc & bit );
        bool bitTV = bool( tc & bit );
        if ( !bitPV && bitTV )
            --pos[i];
        else if ( bitPV && !bitTV )
            ++pos[i];
    }
    return pos;
}
*/

const PlacedSocket* BoxNester::processQ_()
{
    QueueNode minQueueNode;
    int iter = 0;
    bool stopIterating = false;
    while ( !candidatesQ_.empty() && !stopIterating )
    {
        auto top = candidatesQ_.top();
        candidatesQ_.pop();
        auto topSocketObjId = socketTree_[top.socketId].objId;
        if ( topSocketObjId == sizes_.backId() )
            return &socketTree_[top.socketId]; // found result
        if ( minQueueNode < top )
            minQueueNode = top;

        auto parentId = top.socketId;
        while ( parentId != -1 && !stopIterating )
        {
            auto box = socketTree_[parentId].box;
            stopIterating = !addCandidateCorners_( top.socketId, topSocketObjId, box, iter );
            parentId = socketTree_[parentId].parentSocketId;
        }
        if ( !stopIterating )
            stopIterating = !addCustomToQ_( top.socketId, topSocketObjId, iter );
    }
    if ( !candidatesQ_.empty() )
    {
        auto top = candidatesQ_.top();
        candidatesQ_.pop();
        if ( minQueueNode < top )
            minQueueNode = top;
    }
    if ( minQueueNode.socketId == -1 )
        return nullptr; // no result found
    return &socketTree_[minQueueNode.socketId]; // return best found
}

void calcToDenseXfAndDenseXYSize( const MeshXf& m, AffineXf3f& outXf, Vector3f& outSize, bool allowRotation )
{
    if ( !m.mesh )
    {
        assert( false );
        return;
    }
    AffineXf3f toDenseXf = m.xf;
    if ( allowRotation )
    {
        PointAccumulator pa;
        accumulateFaceCenters( pa, *m.mesh, &m.xf );
        Matrix3f resBasis;

        Vector3d centroid;
        Matrix3d eigenvectors;
        Vector3d eigenvalues;
        pa.getCenteredCovarianceEigen( centroid, eigenvectors, eigenvalues );

        auto zComp2dLengthSq = to2dim( eigenvectors.z * eigenvalues.z ).lengthSq();
        auto yComp2dLengthSq = to2dim( eigenvectors.y * eigenvalues.y ).lengthSq();

        resBasis.x = Vector3f( zComp2dLengthSq >= yComp2dLengthSq ? eigenvectors.z : eigenvectors.y );
        resBasis.x.z = 0.0f;
        resBasis.x = resBasis.x.normalized();
        resBasis.y = cross( resBasis.z, resBasis.x );
        resBasis = resBasis.transposed(); // fromPlaneXf

        auto toPlaneXf = AffineXf3f::linear( resBasis ).inverse();
        toDenseXf = toPlaneXf * toDenseXf;
    }
    auto denseBox = m.mesh->computeBoundingBox( &toDenseXf );
    outXf = AffineXf3f::translation( -denseBox.min ) * toDenseXf;
    outSize = denseBox.size();
}

std::shared_ptr<IBoxNestingPriority> getNestPostionMinPriorityMetric( const Box3f& nest )
{
    class NestPositionMinPriority : public IBoxNestingPriority
    {
    public:
        NestPositionMinPriority( float quantSize, int maxQuants, float maxVolume ) : quantSize_{ quantSize }, maxQuants_{ maxQuants }, maxVolume_{ maxVolume }
        {
        };
        virtual void init( const Box3f& thisBox ) override
        {
            pos_ = Vector3i( thisBox.min / quantSize_ );
            commonBox_ = thisBox;
        }
        virtual void addNested( const Box3f& box ) override
        {
            commonBox_.include( box );
        }
        virtual double complete() const override
        {
            size_t res =
                size_t( maxQuants_ - pos_.z ) * maxQuants_ * maxQuants_ +
                size_t( maxQuants_ - pos_.y ) * maxQuants_ +
                size_t( maxQuants_ - pos_.x );
            return double( res ) + double( maxVolume_ - commonBox_.volume() ) / maxVolume_;
        }
    private:
        float quantSize_ = 1.0f;
        int maxQuants_{ 20 };
        float maxVolume_{ 0.0f };
        Vector3i pos_;
        Box3f commonBox_;
    };
    auto nestSize = nest.size();
    return std::make_shared<NestPositionMinPriority>( std::max( { nestSize.x,nestSize.y,nestSize.z } ) / 20.0f, 20, nest.volume() );
}

std::shared_ptr<IBoxNestingPriority> getNeighborigDensityPriorityMetric( const Box3f& nest, float neighborhood )
{
    class NestNeighborDensityPriority : public IBoxNestingPriority
    {
    public:
        NestNeighborDensityPriority( const Box3f& nest, float neighborhood ) :nest_{ nest }, neighborhood_{ neighborhood }
        {
        }
        virtual void init( const Box3f& thisBox ) override
        {
            for ( int i = 0; i < 6; ++i )
            {
                int coord = i / 2;
                if ( i % 2 == 0 )
                {
                    outerBoxes_[i].min = thisBox.min;
                    outerBoxes_[i].min[coord] -= neighborhood_;
                    outerBoxes_[i].max = thisBox.max;
                    outerBoxes_[i].max[coord] = thisBox.min[coord];
                }
                else
                {
                    outerBoxes_[i].min = thisBox.min;
                    outerBoxes_[i].min[coord] = thisBox.max[coord];
                    outerBoxes_[i].max = thisBox.max;
                    outerBoxes_[i].max[coord] += neighborhood_;
                }
            }

            for ( int i = 0; i < 6; ++i )
            {
                outerBoxVolume_[i] = outerBoxes_[i].volume();
                auto nestInter = outerBoxes_[i].intersection( nest_ );
                sumNeigboringIntersectionVolume_[i] = nestInter.valid() ? ( outerBoxVolume_[i] - nestInter.volume() ) : 0.0;
            }
        }
        virtual void addNested( const Box3f& box ) override
        {
            for ( int i = 0; i < 6; ++i )
            {
                auto inter = outerBoxes_[i].intersection( box );
                if ( inter.valid() )
                    sumNeigboringIntersectionVolume_[i] += inter.volume();
            }
        }
        virtual double complete() const override
        {
            constexpr int cOrder[6] = { 4, 2, 0, 1, 3, 5 }; // ordered sum: -Z -Y -X +X +Y +Z
            double resCost = sumNeigboringIntersectionVolume_[cOrder[0]] / outerBoxVolume_[cOrder[0]];
            double nextWeightMultiplier = 1.0;
            for ( int i = 1; i < 6; ++i )
            {
                int prevCoord = cOrder[i - 1];
                double prevFillRatio = sumNeigboringIntersectionVolume_[prevCoord] / outerBoxVolume_[prevCoord];
                if ( prevFillRatio < 1.0 )
                    nextWeightMultiplier *= prevFillRatio;
                int thisCoord = cOrder[i];
                resCost += nextWeightMultiplier * sumNeigboringIntersectionVolume_[thisCoord] / outerBoxVolume_[thisCoord];
            }
            return resCost;
        }
    private:
        Box3f outerBoxes_[6];
        double sumNeigboringIntersectionVolume_[6] = { 0.0,0.0,0.0,0.0,0.0,0.0 };
        double outerBoxVolume_[6] = { 0.0,0.0,0.0,0.0,0.0,0.0 };

        Box3f nest_;
        float neighborhood_{ 0.0f };
    };
    return std::make_shared<NestNeighborDensityPriority>( nest, neighborhood );
}

Expected<void> fillNestingSocketCorneres( const std::vector<Box3f>& nestedBoxes, std::vector<BoxNestingCorner>& outCorners, const ProgressCallback& cb )
{
    MR_TIMER;
    struct BoxOrder
    {
        std::vector<BoxNestingCorner>* localStorage{ nullptr };
        int begin = 0;
        int end = 0;
    };

    tbb::enumerable_thread_specific<std::vector<BoxNestingCorner>> tls;
    std::vector<BoxOrder> order( nestedBoxes.size() ); // needed to have deterministic result vector
    auto keepGoing = ParallelFor( 0, int( nestedBoxes.size() ), tls, [&] ( int i, std::vector<BoxNestingCorner>& local )
    {
        BoxOrder thisOrder;
        thisOrder.localStorage = &local;
        thisOrder.begin = int( local.size() );
        const auto& thisBox = nestedBoxes[i];
        std::array<Vector3f, 8> corners
        {
            thisBox.corner( Vector3b( false,false,false ) ),
            thisBox.corner( Vector3b( true,false,false ) ),
            thisBox.corner( Vector3b( false,true,false ) ),
            thisBox.corner( Vector3b( true,true,false ) ),
            thisBox.corner( Vector3b( false,false,true ) ),
            thisBox.corner( Vector3b( true,false,true ) ),
            thisBox.corner( Vector3b( false,true,true ) ),
            thisBox.corner( Vector3b( true,true,true ) )
        };
        uint8_t cornerInsideBitMask = 0b00000000; // for each corner 0 means outside other box, 1 means inside
        // classify corners as "inside/outside"
        for ( uint8_t c = 0; c < 8; ++c )
        {
            for ( int j = 0; j < nestedBoxes.size(); ++j )
            {
                if ( i == j )
                    continue;
                if ( nestedBoxes[j].contains( corners[c] ) )
                {
                    cornerInsideBitMask |= ( 1 << c );
                    break;
                }
            }
        }
        // add simple outside corners
        for ( uint8_t c = 0; c < 8; ++c )
        {
            if ( bool( cornerInsideBitMask & ( 1 << c ) ) )
                continue; // do nothing if inside
            local.emplace_back( BoxNestingCorner{ .pos = corners[c],.bitMask = c } );
        }
        // add corners of intersections
        for ( uint8_t edgeI = 0; edgeI < 12; ++edgeI )
        {
            uint8_t mainAxisId = edgeI / 4; // 0-x, 1-y, 2-z
            uint8_t subAxes = edgeI % 4;
            uint8_t secAxisId = subAxes / 2;
            uint8_t lastAxisId = subAxes % 2;
            uint8_t minCornerId = 0;
            if ( mainAxisId == 2 )
                minCornerId |= ( secAxisId << 1 );
            else
                minCornerId |= ( secAxisId << 2 );

            if ( mainAxisId == 0 )
                minCornerId |= ( lastAxisId << 1 );
            else
                minCornerId |= ( lastAxisId << 0 );
            uint8_t maxCornerId = minCornerId + ( 1 << mainAxisId );
            bool minInside = bool( cornerInsideBitMask & ( 1 << minCornerId ) );
            bool maxInside = bool( cornerInsideBitMask & ( 1 << maxCornerId ) );
            if ( minInside == maxInside )
                continue; // nothing to add, both corners from one side

            // find furthest intersection and add to output
            BoxNestingCorner outCorner;
            outCorner.pos = minInside ? corners[minCornerId] : corners[maxCornerId];
            outCorner.bitMask = minInside ? minCornerId : maxCornerId;
            for ( int j = 0; j < nestedBoxes.size(); ++j )
            {
                if ( j == i )
                    continue;
                if ( minInside )
                    outCorner.pos[mainAxisId] = std::max( nestedBoxes[j].max[mainAxisId], outCorner.pos[mainAxisId] );
                else
                    outCorner.pos[mainAxisId] = std::min( nestedBoxes[j].min[mainAxisId], outCorner.pos[mainAxisId] );
            }
            local.emplace_back( std::move( outCorner ) );
        }
        thisOrder.end = int( local.size() );
        order[i] = thisOrder;
    }, subprogress( cb, 0.0f, 0.9f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    for ( const auto& o : order )
        outCorners.insert( outCorners.end(), o.localStorage->begin() + o.begin, o.localStorage->begin() + o.end );

    if ( !reportProgress( cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return {};
}

Expected<Vector<NestingResult, ObjId>> boxNesting( const Vector<MeshXf, ObjId>& meshes, const BoxNestingParams& params )
{
    MR_TIMER;

    // 1. Find XY dense boxes for all
    Vector<AffineXf3f, ObjId> toDenseXfs( meshes.size() );
    Vector<Vector3f, ObjId> sizes( meshes.size() );
    auto keepGoing = ParallelFor( toDenseXfs, [&] ( ObjId oId )
    {
        calcToDenseXfAndDenseXYSize( meshes[oId], toDenseXfs[oId], sizes[oId], params.options.allowRotation );
        if ( params.options.expansionFactor )
        {
            auto expandedSize = mult( sizes[oId], *params.options.expansionFactor * cSafeExpansion );
            sizes[oId] = expandedSize;
        }
    }, subprogress( params.options.cb, 0.0f, 0.5f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    // 2. Find best nesting
    Nesting::BoxNester nester( sizes, params );
    auto res = nester.run();
    if ( !reportProgress( params.options.cb, 0.75f ) )
        return unexpectedOperationCanceled();

    // 3. Compose result xf
    keepGoing = ParallelFor( res, [&] ( ObjId oId )
    {
        res[oId].xf = res[oId].xf * toDenseXfs[oId];
    }, subprogress( params.options.cb, 0.75f, 1.0f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();
    return res;
}

}

}
