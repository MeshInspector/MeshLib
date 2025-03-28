#include <MRMesh/MRProgressCallback.h>

#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, ProgressCallback )
{
    std::vector<float> progressValues;
    ProgressCallback cb = [&progressValues] ( float v )
    {
        progressValues.emplace_back( v );
        return true;
    };

    cb = subprogress( cb, 0.40f, 1.00f );
    cb = subprogress( cb, 0.80f, 1.00f );
    cb = subprogress( cb, 0.00f, 0.30f );

    cb( (float)10817394 / 10817408 );
    cb( (float)10817398 / 10817408 );
    cb( (float)10817401 / 10817408 );
    EXPECT_LE( progressValues[0], progressValues[1] );
    EXPECT_LE( progressValues[0], progressValues[2] );
    EXPECT_LE( progressValues[1], progressValues[2] );
}

} // namespace MR
