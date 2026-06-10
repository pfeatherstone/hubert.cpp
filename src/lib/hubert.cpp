#include <cassert>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>
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

    template <class T, size_t N, class... Args>
    auto make_array(Args&&... args)
    {
        return []<size_t... I>(std::index_sequence<I...>, Args&&... args) {
            return std::array<T, N>{((void)I, T(std::forward<Args>(args)...))...};
        }(std::make_index_sequence<N>{}, std::forward<Args>(args)...);
    }

//----------------------------------------------------------------------------------------------------------------

    constexpr void add(std::span<const float> a, std::span<const float> b, std::span<float> c)
    {
        for (size_t i{0} ; i < a.size() ; ++i)
            c[i] = a[i] + b[i];
    }

//----------------------------------------------------------------------------------------------------------------

    struct linear
    {
        MatrixXf w;
        VectorXf b;
        MatrixXf out;

        linear(size_t nin_, size_t nout_)
        : w(nout_, nin_),
          b(nout_)
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size() + b.size())) throw std::runtime_error("Not enough data in weights");
            size_t off{0};
            auto w_ = Eigen::Map<const MatrixXf>(data.subspan(off, w.size()).data(), nout(), nin()); off += w.size();
            auto b_ = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), nout());        off += b.size();
            w       = w_;
            b       = b_;
            return data.subspan(off);
        }

        size_t nin()  {return w.cols();}
        size_t nout() {return w.rows();}

        std::span<float> operator()(std::span<const float> input)
        {
            const size_t Tin = input.size() / nin();
            auto x = Eigen::Map<const MatrixXf>(input.data(), Tin, nin());
            out.noalias() = x * w.transpose() ;
            out.rowwise() += b.transpose();
            return std::span<float>{out.data(), static_cast<size_t>(out.size())};
        }
    };

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

    struct instance_norm
    {
        size_t   nin{};
        float    eps{};
        VectorXf w;
        VectorXf b;

        instance_norm(size_t nin_, float eps_ = 1e-5f)
        : nin{nin_},
          eps{eps_},
          w(nin),
          b(nin)
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size()+b.size())) throw std::runtime_error("Not enough data in groupnorm weights");
            size_t off{0};
            w = Eigen::Map<const VectorXf>(data.subspan(off, w.size()).data(), w.size()); off += w.size();
            b = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), b.size()); off += b.size();
            return data.subspan(off);
        }

        std::span<float> operator()(std::span<float> input)
        {
            const size_t T = input.size() / nin;
            auto X = Eigen::Map<MatrixXf>(input.data(), T, nin);

            for (size_t c{0}; c < nin; ++c)
            {
                auto x           = X.col(c).array();
                const float mean = x.mean();
                const float var  = (x - mean).square().mean();
                const float inv  = 1.0f / std::sqrt(var + eps);
                x = (x - mean) * inv * w(c) + b(c);
            }

            return input;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct layernorm
    {
        size_t   nin{};
        float    eps{};
        VectorXf w;
        VectorXf b;

        layernorm(size_t nin_, float eps_ = 1e-5f)
        : nin{nin_},
          eps{eps_},
          w(nin),
          b(nin)
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            if (data.size() < size_t(w.size()+b.size())) throw std::runtime_error("Not enough data in groupnorm weights");
            size_t off{0};
            w = Eigen::Map<const VectorXf>(data.subspan(off, w.size()).data(), w.size()); off += w.size();
            b = Eigen::Map<const VectorXf>(data.subspan(off, b.size()).data(), b.size()); off += b.size();
            return data.subspan(off);
        }

        std::span<float> operator()(std::span<float> input)
        {
            const size_t T = input.size() / nin;
            auto X = Eigen::Map<MatrixXf>(input.data(), T, nin);

            for (size_t t{0}; t < T; ++t)
            {
                auto x           = X.row(t).array();
                const float mean = x.mean();
                const float var  = (x - mean).square().mean();
                const float inv  = 1.0f / std::sqrt(var + eps);
                x                = ((x - mean) * inv) * w.transpose().array() + b.transpose().array();
            }

            return input;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    template <class Derived>
    auto gelu(const Eigen::ArrayBase<Derived>& x)
    {
        return 0.5f * x * (1.0f + (x * 0.70710678118654752440f).erf());
    }

    std::span<float> gelu(std::span<float> x)
    {
        auto X = Eigen::Map<ArrayXf>(x.data(), x.size()); 
        X = gelu(X);
        return x;
    }

//----------------------------------------------------------------------------------------------------------------

    struct pos_embedding
    {
        conv        c;
        layernorm   n;

        pos_embedding(size_t dim, size_t k, size_t p, size_t g)
        : c(dim,dim,k,p,1,g,true),
          n(dim)
        {
        }

        auto load_weights(std::span<const float> data) -> std::span<const float>
        {
            data = c.load_weights(data);
            data = n.load_weights(data);
            return data;
        }

        std::span<float> operator()(std::span<const float> input)
        {
            auto x = gelu(c(input));
            x = x.subspan(0, x.size()-c.nout);
            add(input, x, x);
            x = n(x);
            return x;
        }
    };

//----------------------------------------------------------------------------------------------------------------

    struct model::impl
    {
        conv               b0; 
        instance_norm      n0;
        std::array<conv,4> b1;
        std::array<conv,2> b2;
        linear             b3;
        pos_embedding      b4;

        impl()
        : b0(1,512,10,0,5,1,false), n0(512),
          b1(make_array<conv,4>(512,512,3,0,2,1,false)),
          b2(make_array<conv,2>(512,512,2,0,2,1,false)),
          b3(512, 768),
          b4(768,128,64,16)
        {
            auto weights = std::span{(const float*)ghubert_weightsData, ghubert_weightsSize/4};
            weights = b0.load_weights(weights);
            weights = n0.load_weights(weights);
            for (auto& b : b1) weights = b.load_weights(weights);
            for (auto& b : b2) weights = b.load_weights(weights);
            weights = b3.load_weights(weights);
            weights = b4.load_weights(weights);
            // if (!weights.empty()) throw std::runtime_error("Failed to load encoder weights");
        }

        std::span<const float> encode(std::span<const float> audio)
        {
            auto x = gelu(n0(b0(audio)));
            for (auto& b : b1) x = gelu(b(x));
            for (auto& b : b2) x = gelu(b(x));
            x = b3(x);
            x = b4(x);
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