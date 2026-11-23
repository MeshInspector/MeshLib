#include <MRMesh/MRTranslationFormat.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <MRViewer/MRI18n.h>
#include <MRViewer/MRLocale.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <vector>

namespace MR
{

TEST( MRMesh, FormatSafeTranslationFmt )
{
    using Locale::isFormatSafeTranslation;
    EXPECT_TRUE( isFormatSafeTranslation( "Hello", "Hallo" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "Loaded {} of {} files", "Geladen {} von {}" ) );
    // arguments may be reordered by their numbers, and not all of them have to be used
    EXPECT_TRUE( isFormatSafeTranslation( "{0} of {1}", "{1} von {0}" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "{} of {}", "{1} von {0}" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "{} holes", "ein Loch" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "{name} is {}", "{} ist {name}" ) );
    // the format spec of an argument is kept or dropped
    EXPECT_TRUE( isFormatSafeTranslation( "Value {:.2f}", "Wert {:.2f}" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "Value {:.2f}", "Wert {}" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "Width {:>{}}", "Breite {:>{}}" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "at {:%H:%M}", "um {:%H:%M}" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "Set {{}} as {}", "Setze {{}} als {}" ) );
    // percent sign has no special meaning in fmt
    EXPECT_TRUE( isFormatSafeTranslation( "{}% of {}", "{} % sur {}" ) );

    EXPECT_FALSE( isFormatSafeTranslation( "Object {}", "Objekt {} {}" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "{0} of {1}", "{0} von {2}" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "Done", "Fertig {}" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "{name}", "{nom}" ) );
    // format specs moved to other arguments
    EXPECT_FALSE( isFormatSafeTranslation( "Value {:.2f} in {}", "In {} Wert {:.2f}" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "Angle {:.1f}", "Winkel {:d}" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "Width {:>{}}", "Breite {} {}" ) );
    // not a valid format string
    EXPECT_FALSE( isFormatSafeTranslation( "{} of {}", "{} von {1}" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "{} files", "{ Dateien" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "{} files", "{} Dateien }" ) );
}

TEST( MRMesh, FormatSafeTranslationPrintf )
{
    using Locale::isFormatSafeTranslation;
    EXPECT_TRUE( isFormatSafeTranslation( "%d holes", "%d Loecher" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "%s has %d points", "%s hat %d Punkte" ) );
    // the last arguments may be not used, flags, width and precision may differ
    EXPECT_TRUE( isFormatSafeTranslation( "%d of %s", "%d" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "%.2f mm, %5d", "%.3f mm, %-5u" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "%*d", "%*d" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "Progress %d%%", "Fortschritt %d %%" ) );
    // the original is not a format string: its percent sign is not doubled
    EXPECT_TRUE( isFormatSafeTranslation( "1% of tolerance", "1 % de la tolerance" ) );
    EXPECT_TRUE( isFormatSafeTranslation( "50% smaller", "50 % de moins" ) );

    EXPECT_FALSE( isFormatSafeTranslation( "%d holes", "%s Loecher" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%s has %d points", "%d Punkte in %s" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%d holes", "%d von %d" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%d", "%ld" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%d", "%f" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%5d", "%*d" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%s of %s", "%2$s von %1$s" ) );
    // percent sign not doubled in the translation
    EXPECT_FALSE( isFormatSafeTranslation( "Progress %d%%", "Fortschritt %d%" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "%d%% done", "%d% erledigt" ) );
    EXPECT_FALSE( isFormatSafeTranslation( "Sure", "100% sicher" ) );
}

#ifndef MRVIEWER_NO_LOCALE
/// makes the contents of .mo file with given entries, the first one must be the header with empty key
static std::string makeMoFile( const std::vector<std::pair<std::string, std::string>>& entries )
{
    const auto n = std::uint32_t( entries.size() );
    // hash table as msgfmt makes it; without it boost converts all strings, which Wasm build cannot do
    constexpr std::uint32_t cHashSize = 31;
    std::uint32_t hashTable[cHashSize] = {};
    for ( std::uint32_t i = 0; i < n; ++i )
    {
        std::uint32_t hash = 0;
        for ( const char* c = entries[i].first.c_str(); *c; ++c ) // plural original is not hashed
        {
            hash = ( hash << 4 ) + ( unsigned char )*c;
            if ( const auto high = hash & 0xf0000000 )
                hash ^= ( high >> 24 ) ^ high;
        }
        auto pos = hash % cHashSize;
        while ( hashTable[pos] != 0 )
            pos = ( pos + 1 + hash % ( cHashSize - 2 ) ) % cHashSize;
        hashTable[pos] = i + 1;
    }
    const auto hashOffset = 28 + 16 * n;
    const auto stringsOffset = hashOffset + std::uint32_t( sizeof( hashTable ) );
    // magic, revision, number of strings, offsets of two tables, size and offset of hash table
    const std::uint32_t header[7] = { 0x950412de, 0, n, 28, 28 + 8 * n, cHashSize, hashOffset };
    std::string res( ( const char* )header, sizeof( header ) ), strings;
    auto addTable = [&]( auto member )
    {
        for ( const auto& e : entries )
        {
            const std::string& str = e.*member;
            const std::uint32_t lengthAndOffset[2] = { std::uint32_t( str.size() ), stringsOffset + std::uint32_t( strings.size() ) };
            res.append( ( const char* )lengthAndOffset, sizeof( lengthAndOffset ) );
            strings.append( str );
            strings.push_back( '\0' );
        }
    };
    addTable( &std::pair<std::string, std::string>::first );
    addTable( &std::pair<std::string, std::string>::second );
    res.append( ( const char* )hashTable, sizeof( hashTable ) );
    res.append( strings );
    return res;
}

TEST( MRViewer, UnsafeTranslationsAreIgnored )
{
    using namespace std::string_literals;
    const char* cDomain = "MRTranslationFormatTest";
    const char* cFileName = "MRTranslationFormatTest.mo";
    UniqueTemporaryFolder folder;
    ASSERT_TRUE( bool( folder ) );
    const auto dir = folder / "xx" / "LC_MESSAGES";
    std::error_code ec;
    std::filesystem::create_directories( dir, ec );
    {
        std::ofstream out( dir / cFileName, std::ofstream::binary );
        out << makeMoFile( {
            { "", "Content-Type: text/plain; charset=UTF-8\nPlural-Forms: nplurals=2; plural=(n%100!=1);\n" },
            { "Object {}", "Objekt {}" },
            { "Loaded {} of {}", "Geladen {} von {} {}" },
            { "%d holes", "%s Loecher" },
            { "%d points", "%d Punkte" },
            // with context
            { "Menu\x04" "Open {}", "Oeffnen {}" },
            { "Menu\x04" "Close {}", "Schliessen {} {}" },
            // with plural forms
            { "{} file\0{} files"s, "eine Datei\0{} Dateien"s },
            { "{} hole\0{} holes"s, "{} Loch\0{} Loecher {}"s },
        } );
        ASSERT_TRUE( out.good() );
    }

    const auto prevLocale = Locale::getName();
    Locale::addCatalogPath( folder );
    const auto domain = Locale::addDomain( std::string( cDomain ) );
    ASSERT_TRUE( domain.valid() );
    Locale::set( "xx" );

    EXPECT_EQ( Locale::translate( "Object {}", domain ), "Objekt {}" );
    EXPECT_EQ( Locale::translate( "%d points", domain ), "%d Punkte" );
    EXPECT_EQ( Locale::translate( "Menu", "Open {}", domain ), "Oeffnen {}" );
    EXPECT_EQ( Locale::translate( "{} file", "{} files", 1, domain ), "eine Datei" );
    EXPECT_EQ( Locale::translate( "{} file", "{} files", 5, domain ), "{} Dateien" );
    // these translations would throw from fmt::format and crash in printf, so they are removed when the file is loaded
    EXPECT_EQ( Locale::translate( "Loaded {} of {}", domain ), "Loaded {} of {}" );
    EXPECT_EQ( Locale::translate( "%d holes", domain ), "%d holes" );
    EXPECT_EQ( Locale::translate( "Menu", "Close {}", domain ), "Close {}" );
    EXPECT_EQ( Locale::translate( "{} hole", "{} holes", 1, domain ), "{} hole" );
    EXPECT_EQ( Locale::translate( "{} hole", "{} holes", 5, domain ), "{} holes" );

    Locale::set( prevLocale );
}
#endif

} // namespace MR
