#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRExpected.h"
#include "MRViewer/MRImGui.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRVector.h"
#include <algorithm>
#include <filesystem>

namespace Json{ class Value; }

namespace MR
{

/**
 * @brief Class to hold one dimension texture with value to UV mapping
 *
 * Discrete mode will draw rectangle for each color
 * Continuous mode will draw interpolated color
 *
 */
class Palette
{
public:
    /// preset palette colors: from blue via green to red
    MRVIEWER_API static const std::vector<Color> DefaultColors;
    [[nodiscard]] static const std::vector<Color> & BlueGreenRedColors() { return DefaultColors; }

    /// simpler palette colors: from green to red
    [[nodiscard]] MRVIEWER_API static const std::vector<Color> & GreenRedColors();

    MRVIEWER_API Palette( const std::vector<Color>& colors );
    /**
     * @brief Set base palette colors
     * colors.size() should be more or equal 2
     * for discrete palette using vector of colors calculated by mixing the base colors
     * i.e. base {blue, red} -> discrete 3 {blue, 0.5*blue + 0.5*red, red}
     */
    MRVIEWER_API void setBaseColors( const std::vector<Color>& colors );
    /**
     * @brief set range limits for palette (need for find color by value)
     * all palette colors are evenly distributed between min and max
     */
    MRVIEWER_API void setRangeMinMax( float min, float max );
    /**
     * @brief set range limits for palette (need for find color by value)
     * two half palette colors are evenly distributed between MinNeg / MaxNeg and MinPos / MaxPos
     * for values between MaxNeg / MinPos return one color (from center palette)
     */
    MRVIEWER_API void setRangeMinMaxNegPos( float minNeg, float maxNeg, float minPos, float maxPos );

    // set number of different colors for discrete palette
    MRVIEWER_API void setDiscretizationNumber( int discretization );
    // set palette type (linear / discrete)
    MRVIEWER_API void setFilterType( FilterType type );

    /// Draws vertical legend with labels in ImGui window with given name
    /// Discrete: bar consists of single colored rectangles for each initial color
    /// Linear (default): color is changing from one to another during initial color list
    /// \param onlyTopHalf if true, draws only top half of the palette and labels stretched to whole window
    MRVIEWER_API void draw( const std::string& windowName, const ImVec2& pose, const ImVec2& size, bool onlyTopHalf = false );

    /// Draws vertical legend with labels in existing window or scene
    /// Discrete: bar consists of single colored rectangles for each initial color
    /// Linear (default): color is changing from one to another during initial color list
    /// \param onlyTopHalf if true, draws only top half of the palette and labels stretched to whole window
    /// \param labelBgColor label background color
    MRVIEWER_API void draw( ImDrawList* drawList, float scaling, const ImVec2& pos, const ImVec2& size, const Color& labelBgColor, bool onlyTopHalf = false ) const;
    MRVIEWER_API void draw( ImDrawList* drawList, float scaling, const ImVec2& pos, const ImVec2& size, bool onlyTopHalf = false ) const;

    // structure for label
    struct MRVIEWER_CLASS Label
    {
        float value = 0.f; // label position according normal scale (from Min to Max)
        std::string text; // label text

        // The special zero label.
        // The flag is there so we can hide it if we want.
        bool isZero = false;

        Label() = default;
        MRVIEWER_API Label( float val, std::string text );
    };
    // reset labels to standard view
    MRVIEWER_API void resetLabels();
    // set labels manually
    MRVIEWER_API void setCustomLabels( const std::vector<Label>& labels );
    // set labels visible
    MRVIEWER_API void setLabelsVisible( bool visible );


    // Setup palette data from JsonValue
    // @return return true if loading successful
    MRVIEWER_API bool loadFromJson( const Json::Value& root );
    // Serialize this palette data to JsonValue
    MRVIEWER_API void saveCurrentToJson( Json::Value& root ) const;


    /// return color according relative value and setting filter type
    /// \param relativeValue - value scaled from ragne [min, max] to [0; 1]
    /// Discrete: returns color for given value
    /// Linear: get interpolated color from the init vector
    MRVIEWER_API Color getColor( float relativeValue ) const;
    /// return invalid color
    Color getInvalidColor() const { return Color::gray(); }

    /// get colors for given vert value
    /// \param region only these vertices will be processed
    /// \param valids if given then defines subregion with valid values, and invalid values will get gray color
    MRVIEWER_API VertColors getVertColors( const VertScalars& values, const VertBitSet& region, const VertBitSet* valids, const VertBitSet* validsForStats );

    const MeshTexture& getTexture() const { return texture_; };

    // get relative position in [0,1], where 0 is for minimum and 1 is for maximum
    MRVIEWER_API float getRelativePos( float val ) const;

    /// get UV coordinate in palette for given value
    /// \param valid true - return coordinate of palette's color, false - return coordinate of gray
    UVCoord getUVcoord( float val, bool valid = true ) const
    {
        return {
            ( texEnd_ - texStart_ ) * getRelativePos( val ) + texStart_,
            valid ? 0.25f : 0.75f
        };
    }

    // If `bits` is non-zero, captures the pointer and returns a predicate that checks against this bitset.
    // Otherwise returns null.
    [[nodiscard]] MRVIEWER_API static VertPredicate predFromBitSet( const VertBitSet* bits );

    /// get UV coordinates in palette for given values
    /// \param region only these vertices will be processed
    /// \param valids if given then defines subregion with valid values, and invalid values will get gray color
    /// \param validsIfHistogram If specified, replaces \p valids for the purposes of creating the histogram and computing the percentages of vertices in different discretization steps.
    MRVIEWER_API VertUVCoords getUVcoords( const VertScalars & values, const VertBitSet & region, const VertPredicate & valids = {}, const VertPredicate & validsForStats = {} );

    VertUVCoords getUVcoords( const VertScalars & values, const VertBitSet & region, const VertBitSet * valids, const VertBitSet * validsForStats = nullptr )
    {
        return getUVcoords( values, region, predFromBitSet( valids ), predFromBitSet( validsForStats ) );
    }

    // base parameters of palette
    struct Parameters
    {
        std::vector<float> ranges = { 0.f, 1.f }; // range limits for palette
        std::vector<Color> baseColors; // palette base colors (need for calculate real colors according discretization)
        MinMaxf legendLimits; // if valid - limit legend range
        int discretization = 7; // number of different colors for discrete palette
    };

    [[nodiscard]] const Parameters& getParameters() const { return parameters_; }

    /// returns minimum value in the palette's range
    [[nodiscard]] float getRangeMin() const { return parameters_.ranges.front(); }

    /// returns maximum value in the palette's range
    [[nodiscard]] float getRangeMax() const { return parameters_.ranges.back(); }

    /// returns minimum squared value, not smaller than all squared values of palette's range
    [[nodiscard]] float getRangeSq() const { return std::max( sqr( getRangeMin() ), sqr( getRangeMax() ) ); }

    /// returns formated string for this value of palette
    MRVIEWER_API std::string getStringValue( float value ) const;
    /// returns maximal label count
    MRVIEWER_API int getMaxLabelCount();
    /// sets maximal label count
    MRVIEWER_API void setMaxLabelCount( int val );

    /// set legend limits. if min > max - limits are disabled
    MRVIEWER_API void setLegendLimits( const MinMaxf& limits );


    // Histogram:

    [[nodiscard]] bool isHistogramEnabled() const { return getNumHistogramBuckets() != 0; }
    [[nodiscard]] MRVIEWER_API int getNumHistogramBuckets() const;
    // Pass zero to disable the histogram. Pass `getDefaultNumHistogramBuckets()` or any other number to enable it.
    // This should probably be odd, to have a bucket for zero in the middle.
    MRVIEWER_API void setNumHistogramBuckets( int n );
    // Returns the recommended argument for `setNumHistogramBuckets()`.
    [[nodiscard]] MRVIEWER_API int getDefaultNumHistogramBuckets() const;


    // Should we maintain the percentages of distances in each discretization step?
    [[nodiscard]] bool isDiscretizationPercentagesEnabled() const { return enableHistogramDiscr_; }
    void enableDiscretizationPercentages( bool enable ) { histogramDiscr_.reset(); enableHistogramDiscr_ = enable; }


    // This is called automatically by `getValidVerts()` and `getUVcoords(), so usually you don't need to call this manually.
    // Call this after `setNumHistogramBuckets()`.
    MRVIEWER_API void updateStats( const VertScalars& values, const VertBitSet& region, const VertPredicate& vertPredicate );

    /// Create uniform labels
    /// \details creates labels for each color boundary (for a discrete palette)
    MRVIEWER_API std::vector<Label> createUniformLabels() const;

    struct Histogram
    {
        // If this is empty, the histogram is disabled.
        std::vector<int> buckets;
        // The buckets for out-of-range elements.
        int beforeBucket = 0;
        int afterBucket = 0;
        // The sum of all values in `buckets` and `{low,high}Bucket`.
        int numEntries = 0;
        // The max value in `buckets` (but ignoring `{low,high}Bucket`).
        int maxEntry = 0;

        // `reset()` sets this to true, and `finalize()` sets this to false.
        bool needsUpdate = true;

        MRVIEWER_API void reset();
        MRVIEWER_API void addValue( float value );
        // Call once after all `addValue()` calls.
        MRVIEWER_API void finalize();
    };

    // The normal histogram, if enabled (check with `isHistogramEnabled()`).
    [[nodiscard]] const Histogram &getHistogramValues() { return histogram_; }
    // This one has the size matching `getParameters().discretization`. Only has meaningful values if enabled, check with `isDiscretizationPercentagesEnabled()`.
    [[nodiscard]] const Histogram &getDiscrHistogramValues() const { return histogramDiscr_; }

private:
    void setRangeLimits_( const std::vector<float>& ranges );

    void updateDiscretizatedColors_();
    Color getBaseColor_( float val );

    // What color we assume the pallete is drawn on top of. Typically should be the viewport background color.
    const Color& getBackgroundColor_() const;


    // set labels with equal distance between
    void setUniformLabels_();
    // make labels with equal distance between
    void makeUniformLabels_( std::vector<Label>& labels ) const;
    // first label is equal to min value, last - to the max val
    // don't use with MinMaxNegPos mode
    void setZeroCentredLabels_();

    void updateCustomLabels_();

    void sortLabels_( std::vector<Label>& labels ) const;

    void updateLegendLimits_( const MinMaxf& limits );
    void updateLegendLimitIndexes_();

    std::vector<Label> customLabels_;
    std::vector<Label> labels_;
    bool showLabels_ = false;

    /// Returns the adjusted label, or null if it should be skipped.
    /// \param onlyTopHalf if true, draws only top half of the palette and labels stretched to whole window
    /// \param storage will sometimes be used as storage for the return value. Don't read `storage` directly after the call. You can pass any string, it'll be cleared.
    const char* getAdjustedLabelText_( std::size_t labelIndex, bool onlyTopHalf, std::string& storage ) const;

    /// Computes the max label pixel width.
    float getMaxLabelWidth_( bool onlyTopHalf = false ) const;

    struct StyleVariables
    {
        // Top-left window padding.
        ImVec2 windowPaddingA;
        // Bottom-right window padding.
        ImVec2 windowPaddingB;
        // Spacing between the labels and the colored rect.
        float labelToColoredRectSpacing {};
        // The min width of the colored rect.
        float minColoredRectWidth {};
    };
    StyleVariables getStyleVariables_( float scaling ) const;

    // stores OpenGL textures. Change useDiscrete_ to switch between them
    MeshTexture texture_;

    // texture positions of min and max values
    float texStart_ = 0, texEnd_ = 1;

    Parameters parameters_;

    bool isWindowOpen_ = false;

    bool useCustomLabels_ = false;

    int maxLabelCount_ = 0;

    float prevMaxLabelWidth_ = 0.0f;

    MinMaxi legendLimitIndexes_ = { 0, 7 };
    MinMaxf relativeLimits_ = { 0.f, 1.f };

    // This one is of a user-defined size.
    Histogram histogram_;

    // This one has size matching `parameters_.discretization`.
    Histogram histogramDiscr_;

    // Whether we should actually update `histogramDiscr_`.
    bool enableHistogramDiscr_ = false;

    static void resizeCallback_( ImGuiSizeCallbackData* data );
};

/// Class to save and load user palette presets
class PalettePresets
{
public:
    /// gets names of existing presets
    MRVIEWER_API static const std::vector<std::string>& getPresetNames();
    /// loads existing preset to given palette \n
    /// returns true if load was succeed
    MRVIEWER_API static bool loadPreset( const std::string& name, Palette& palette );
    /// saves given palette to preset with given name
    MRVIEWER_API static Expected<void> savePreset( const std::string& name, const Palette& palette );
    /// returns path to presets folder
    MRVIEWER_API static std::filesystem::path getPalettePresetsFolder();
private:
    PalettePresets();
    ~PalettePresets() = default;

    std::vector<std::string> names_;

    void update_();

    static PalettePresets& instance_();
};

}
