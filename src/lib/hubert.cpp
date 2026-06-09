#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include "hubert.h"
#include "incbin.h"

//----------------------------------------------------------------------------------------------------------------

using MatrixXf      = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using MatrixXu16    = Eigen::Matrix<uint16_t, -1, -1, Eigen::RowMajor>;
using VectorXf      = Eigen::Vector<float, -1>;
using ArrayXf       = Eigen::Array<float, -1, 1>;
using Eigen::seq;
using Eigen::seqN;
using Eigen::placeholders::all;
using Eigen::placeholders::last;

//----------------------------------------------------------------------------------------------------------------

INCBIN(hubert_weights, HUBERT_DATA);

//----------------------------------------------------------------------------------------------------------------

namespace hubert
{

//----------------------------------------------------------------------------------------------------------------

    struct conv
    {
        size_t   nin{};
        size_t   nout{};
        size_t   k{};
        size_t   p{};
        size_t   s{};
        size_t   g{};
        bool     has_bias{};
        MatrixXf w;         // shape [nout, k*nin_g]
        VectorXf b;         // shape [nout]
        MatrixXf patches;   // shape [Tout, k*cin_g]
        MatrixXf out;       // shape [Tout, nout]

        size_t nin_g()  const {return nin / g;}
        size_t nout_g() const {return nout / g;}

        conv(size_t nin_, size_t nout_, size_t k_, size_t p_, size_t s_, size_t g_, bool has_bias_)
        : nin{nin_}, 
          nout{nout_}, 
          k{k_}, 
          p{p_},
          s{s_}, 
          g{g_},
          has_bias{has_bias_}
        {
            if (g == 0)         throw std::runtime_error("conv groups must be non-zero");
            if (nin % g != 0)   throw std::runtime_error("conv nin must be divisible by groups");
            if (nout % g != 0)  throw std::runtime_error("conv nout must be divisible by groups");
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            const size_t wsize = nout*k*nin_g();
            const size_t bsize = has_bias ? nout : 0;
            if (data.size() < (wsize+bsize)) 
                throw std::runtime_error("Not enough data in weights");

            size_t off{0};
            w    = Eigen::Map<const MatrixXf>(data.subspan(off, wsize).data(), nout, k*nin_g()); 
            off += wsize;

            if (has_bias)
            {
                b   = Eigen::Map<const VectorXf>(data.subspan(off, nout).data(), nout);
                off += nout;
            }
                
            return data.subspan(off);
        }

        std::span<float> operator()(std::span<const float> input)
        {
            const size_t Tin  = input.size() / nin;
            const size_t Tout = (Tin + 2*p - k) / s + 1;
            patches.resize(Tout, k*nin_g());
            out.resize(Tout, nout);

            // Grouped GEMM
            // For each group:
            //      Xg: [Tout,   k*cin_g]
            //      Wg: [cout_g, k*cin_g]
            //      Yg: [Tout,   cout_g]
            for (size_t gg{0} ; gg < g ; ++gg)
            {
                auto Yg = out(all, seqN(gg*nout_g(), nout_g()));
                auto Wg = w(seqN(gg*nout_g(), nout_g()), all);

                // im2col : patches[j, kk, c] = input[j*s - p + kk, g*cin_g + c]
                for (size_t j{0}; j < Tout; ++j)
                {
                    for (size_t kk{0} ; kk < k ; ++kk)
                    {
                        const int off = int(j*s) - int(p) + int(kk);
                        float*    out = patches.data() + (j*k + kk)*nin_g();

                        if (off < 0 || off >= (int)Tin) std::fill_n(out, nin_g(), 0.0);
                        else                            std::copy_n(&input[off*nin+gg*nin_g()], nin_g(), out);
                    }
                }

                Yg.noalias() = patches * Wg.transpose();
                if (has_bias)
                    Yg.rowwise() += b(seqN(gg*nout_g(), nout_g())).transpose();
            }

            return std::span<float>{out.data(), (size_t)out.size()};
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct model::impl
    {
        conv b0;

        impl()
        : b0(1,512,10,0,5,1,false)
        {
            auto weights = std::span{(const float*)ghubert_weightsData, ghubert_weightsSize/4};
            weights = b0.load_weights(weights);
            // if (!weights.empty()) throw std::runtime_error("Failed to load encoder weights");
        }

        std::span<const float> encode(std::span<const float> audio)
        {
            auto x = b0(audio);
            return x;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    model::model() : state{std::make_unique<impl>()} {}
    model::~model()                          = default;
    model::model(model&& other)            = default;
    model& model::operator=(model&& other) = default;

    std::span<const float> model::encode(std::span<const float> audio)
    {
        return state->encode(audio);
    }

//----------------------------------------------------------------------------------------------------------------

}