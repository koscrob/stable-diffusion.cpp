#ifndef __RNG_H__
#define __RNG_H__

#include <random>
#include <vector>

class RNG {
public:
    virtual void manual_seed(uint64_t seed)      = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;
};

// Non-deterministic random number generator using hardware entropy source
class NonDeterministicRNG : public RNG {
private:
    std::random_device generator;

public:
    void manual_seed(uint64_t seed) {
        // NOTE: Can't set seed for a non-deterministic hardware entropy source.
        (void)seed;
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

using STDDefaultRNG    = RNGEngine<std::default_random_engine>; // Implementation dependent (usually Mersenne Twister)
using minstd_rand0RNG  = RNGEngine<std::minstd_rand0>;  // LCG - Lewis, Goodman and Miller (1969)
using minstd_randRNG   = RNGEngine<std::minstd_rand>;   // LCG - Park, Miller, and Stockmeyer (1993)
using mt19937RNG       = RNGEngine<std::mt19937>;       // 32-bit Mersenne Twister - Matsumoto and Nishimura (1998)
using ranlux24RNG      = RNGEngine<std::ranlux24>;      // RANLUX - Martin Lüscher and Fred James (1994)
using knuth_bRNG       = RNGEngine<std::knuth_b>;       // std::shuffle_order_engine<std::minstd_rand0, 256>
// TODO: Add and check if it can replace our custom implementation when available in C++26?
//using philox4x32RNG    = RNGEngine<std::philox4x32>;

#endif  // __RNG_H__