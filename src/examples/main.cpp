#include <chrono>
#include <cstring>
#include <vector>
#include <hubert.h>

using namespace std::chrono;

int main()
{
    hubert::model net;

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