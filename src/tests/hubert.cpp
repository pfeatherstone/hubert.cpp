#include <filesystem>
#include <fstream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <hubert.h>

namespace fs = std::filesystem;

template<class T>
auto load_file(const fs::path& path)
{
    const auto     len{fs::file_size(path)};
    std::vector<T> data(len/sizeof(T));
    std::ifstream  file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + path.string());
    file.read((char*)data.data(), len);
    if (!file && file.gcount() != len) throw std::runtime_error("Failed to read file: " + path.string());
    return data;
}

void check_close(std::span<const float> actual, std::span<const float> expected, const float atol, const float rtol)
{
    REQUIRE(actual.size() == expected.size());

    for (size_t i{0} ; i < actual.size() ; ++i)
    {
        const float diff = std::abs(actual[i] - expected[i]);
        CAPTURE(i, actual[i], expected[i], diff, atol, rtol);
        CHECK(diff <= (atol + rtol*std::abs(expected[i])));
    }
}

TEST_CASE("test data", "[hubert]")
{
    std::vector<std::pair<fs::path,fs::path>> pairs;

    for (const auto& f : fs::directory_iterator(fs::path(HUBERT_TEST_DATA)))
    {
        constexpr std::string_view audio_suffix = "_audio";

        const fs::path    path = f.path();
        const std::string stem = path.stem().string();

        if (!f.is_regular_file() || path.extension() != ".dat" || !stem.ends_with(audio_suffix))
            continue;

        const std::string key       = stem.substr(0, stem.size() - audio_suffix.size());
        const fs::path feats_path   = path.parent_path() / (key + "_feats.dat");

        if (fs::exists(feats_path))
            pairs.emplace_back(path, feats_path);
    }

    hubert::model net;

    for (const auto& [audio_path, feats_path] : pairs)
    {
        CAPTURE(audio_path);
        CAPTURE(feats_path);
        auto audio      = load_file<float>(audio_path);
        auto feats_exp  = load_file<float>(feats_path);
        auto feats_cal  = net.encode(audio);
        CHECK(feats_exp.size() == feats_cal.size());
        check_close(feats_cal, feats_exp, 2e-5f, 1e-5f);
    }
}