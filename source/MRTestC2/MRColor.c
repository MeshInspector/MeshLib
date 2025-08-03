#include "TestMacros.h"

#include "MRCMesh/MRColor.h"
#include "MRCMesh/MRVector.h"


void testMrColorNew( void )
{
    // Test if MR_Color_DefaultConstruct returns an opaque black color.
    MR_Color color = MR_Color_DefaultConstruct();
    TEST_ASSERT_INT_EQUAL(color.r, 0);
    TEST_ASSERT_INT_EQUAL(color.g, 0);
    TEST_ASSERT_INT_EQUAL(color.b, 0);
    TEST_ASSERT_INT_EQUAL(color.a, 255); // Opaque alpha
}

void testMrColorFromComponents( void )
{
    // Test if mrColorFromComponents correctly initializes the color.
    MR_Color color = MR_Color_Construct_4_int32_t(128, 64, 32, 200);
    TEST_ASSERT_INT_EQUAL(color.r, 128);
    TEST_ASSERT_INT_EQUAL(color.g, 64);
    TEST_ASSERT_INT_EQUAL(color.b, 32);
    TEST_ASSERT_INT_EQUAL(color.a, 200);
}

void testMrColorFromFloatComponents( void )
{
    // Test if mrColorFromFloatComponents correctly initializes the color
    // and rounds float values to [0..255].
    MR_Color color = MR_Color_Construct_4_float(0.5f, 0.25f, 0.125f, 1.0f);
    TEST_ASSERT_INT_EQUAL(color.r, 127); // 0.5 * 255 = 127
    TEST_ASSERT_INT_EQUAL(color.g, 63);  // 0.25 * 255 = 64
    TEST_ASSERT_INT_EQUAL(color.b, 31);  // 0.125 * 255 = 32
    TEST_ASSERT_INT_EQUAL(color.a, 255); // 1.0 * 255 = 255
}

void testMrColorGetUInt32( void )
{
    // Test if mrColorGetUInt32 correctly converts a color to a 32-bit integer.
    MR_Color color = MR_Color_Construct_4_int32_t(128, 64, 32, 200);
    unsigned int colorValue = MR_Color_getUInt32(&color);
    TEST_ASSERT_INT_EQUAL(colorValue, 0xC8204080);
}

void testMrVertColorsNewSized( void )
{
    // Test if mrVertColorsNewSized creates a new color map with the expected size.
    size_t size = 10;
    MR_VertColors* vertColors = MR_VertColors_Construct_1_uint64_t(size);
    TEST_ASSERT(vertColors != NULL);
    TEST_ASSERT(MR_VertColors_size(vertColors) == size);
    MR_VertColors_Destroy(vertColors);
}
