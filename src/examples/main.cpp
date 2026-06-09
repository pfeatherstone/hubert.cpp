#include <chrono>
#include <cstring>
#include <vector>
#include <hubert.h>

using namespace std::chrono;

template<class T>
auto load_file(const char* file)
{
    std::vector<T> data;
    FILE* fp = fopen(file, "rb");
    if (!fp)
    {
        printf("Failed to open `%s`\n", file);
        return data;
    }

    fseek(fp, 0, SEEK_END);
    const auto size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    data.resize(size/sizeof(T));
    auto nread = fread((char*)&data[0], sizeof(T), data.size(), fp);
    if (nread != data.size())
        printf("Read %zu/%zu samples\n", nread, data.size());
    return data;
}

template<class C>
void save_file(const char* file, const C& data)
{
    using T = typename C::value_type;

    FILE* fp = fopen(file, "wb");
    if (!fp)
    {
        printf("Failed to open `%s`\n", file);
        return;
    }

    fwrite(data.data(), sizeof(T), data.size(), fp);
    fclose(fp);
}

void test_birch_canoe(hubert::model& net)
{
    const auto audio = load_file<float>("original.dat");
    const auto feats = net.encode(audio);
    save_file("feats.dat", feats);
}

int main()
{
    hubert::model net;

    printf("Testing on birch canoe...\n");
    test_birch_canoe(net);
    printf("Testing on birch canoe... Done\n");

    float audio[24000];
    memset(audio, 0, sizeof(audio));
    
    // Warmup
    auto feats = net.encode(audio);

    const int ntests = 100;
    const auto s0 = high_resolution_clock::now();
    for (size_t i{0} ; i < ntests ; ++i)
        feats = net.encode(audio);
    const auto s1 = high_resolution_clock::now();
    printf("Encoding rate %f encoded %zu samples into %zu feats\n", (std::size(audio)*ntests)/((s1-s0).count()*1e-9), std::size(audio), feats.size());

    return 0;
}