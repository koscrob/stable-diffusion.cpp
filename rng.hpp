#ifndef __RNG_H__
#define __RNG_H__

#include <random>
#include <vector>

class RNG {
public:
    virtual void manual_seed(uint64_t seed)      = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;
};

template <typename Engine = std::default_random_engine>
class RNGEngine : public RNG {
private:
    Engine generator;

public:
    // Manually seed the generator
    void manual_seed(uint64_t seed) {
        generator.seed(static_cast<typename Engine::result_type>(seed));
    }

    // Generate 'n' random numbers from a normal distribution
    std::vector<float> randn(uint32_t n) {
        std::vector<float> result;
        std::normal_distribution<float> distribution(0.f, 1.f);
        for (uint32_t i = 0; i < n; ++i) {
            result.push_back(distribution(generator));
        }
        return result;
    }
};

using STDDefaultRNG = RNGEngine<std::default_random_engine>;

#endif  // __RNG_H__