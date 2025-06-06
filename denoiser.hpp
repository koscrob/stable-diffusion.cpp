#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include "ggml_extend.hpp"
#include "gits_noise.inl"
#include <deque>

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

#define TIMESTEPS 1000
#define FLUX_TIMESTEPS 1000

struct SigmaSchedule {
    int version = 0;
    typedef std::function<float(float)> t_to_sigma_t;

    virtual std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) = 0;
};

struct DiscreteSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct ExponentialSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        std::vector<float> sigmas;

        // Calculate step size
        float log_sigma_min = std::log(sigma_min);
        float log_sigma_max = std::log(sigma_max);
        float step          = (log_sigma_max - log_sigma_min) / (n - 1);

        // Fill sigmas with exponential values
        for (uint32_t i = 0; i < n; ++i) {
            float sigma = std::exp(log_sigma_max - step * i);
            sigmas.push_back(sigma);
        }

        sigmas.push_back(0.0f);

        return sigmas;
    }
};

/* interp and linear_interp adapted from dpilger26's NumCpp library:
 * https://github.com/dpilger26/NumCpp/tree/5e40aab74d14e257d65d3dc385c9ff9e2120c60e */
constexpr double interp(double left, double right, double perc) noexcept {
    return (left * (1. - perc)) + (right * perc);
}

/* This will make the assumption that the reference x and y values are
 * already sorted in ascending order because they are being generated as
 * such in the calling function */
std::vector<double> linear_interp(std::vector<float> new_x,
                                  const std::vector<float> ref_x,
                                  const std::vector<float> ref_y) {
    const size_t len_x = new_x.size();
    size_t i           = 0;
    size_t j           = 0;
    std::vector<double> new_y(len_x);

    if (ref_x.size() != ref_y.size()) {
        LOG_ERROR("Linear Interpolation Failed: length mismatch");
        return new_y;
    }

    /* Adjusted bounds checking to ensure new_x is within ref_x range */
    if (new_x[0] < ref_x[0]) {
        new_x[0] = ref_x[0];
    }
    if (new_x.back() > ref_x.back()) {
        new_x.back() = ref_x.back();
    }

    while (i < len_x) {
        if ((ref_x[j] > new_x[i]) || (new_x[i] > ref_x[j + 1])) {
            j++;
            continue;
        }

        const double perc = static_cast<double>(new_x[i] - ref_x[j]) / static_cast<double>(ref_x[j + 1] - ref_x[j]);

        new_y[i] = interp(ref_y[j], ref_y[j + 1], perc);
        i++;
    }

    return new_y;
}

std::vector<float> linear_space(const float start, const float end, const size_t num_points) {
    std::vector<float> result(num_points);
    const float inc = (end - start) / (static_cast<float>(num_points - 1));

    if (num_points > 0) {
        result[0] = start;

        for (size_t i = 1; i < num_points; i++) {
            result[i] = result[i - 1] + inc;
        }
    }

    return result;
}

std::vector<float> log_linear_interpolation(std::vector<float> sigma_in,
                                            const size_t new_len) {
    const size_t s_len        = sigma_in.size();
    std::vector<float> x_vals = linear_space(0.f, 1.f, s_len);
    std::vector<float> y_vals(s_len);

    /* Reverses the input array to be ascending instead of descending,
     * also hits it with a log, it is log-linear interpolation after all */
    for (size_t i = 0; i < s_len; i++) {
        y_vals[i] = std::log(sigma_in[s_len - i - 1]);
    }

    std::vector<float> new_x_vals  = linear_space(0.f, 1.f, new_len);
    std::vector<double> new_y_vals = linear_interp(new_x_vals, x_vals, y_vals);
    std::vector<float> results(new_len);

    for (size_t i = 0; i < new_len; i++) {
        results[i] = static_cast<float>(std::exp(new_y_vals[new_len - i - 1]));
    }

    return results;
}

/*
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
*/
struct AYSSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        const std::vector<float> noise_levels[] = {
            /* SD1.5 */
            {14.6146412293f, 6.4745760956f, 3.8636745985f, 2.6946151520f,
             1.8841921177f, 1.3943805092f, 0.9642583904f, 0.6523686016f,
             0.3977456272f, 0.1515232662f, 0.0291671582f},
            /* SDXL */
            {14.6146412293f, 6.3184485287f, 3.7681790315f, 2.1811480769f,
             1.3405244945f, 0.8620721141f, 0.5550693289f, 0.3798540708f,
             0.2332364134f, 0.1114188177f, 0.0291671582f},
            /* SVD */
            {700.00f, 54.5f, 15.886f, 7.977f, 4.248f, 1.789f, 0.981f, 0.403f,
             0.173f, 0.034f, 0.002f},
        };

        std::vector<float> inputs;
        std::vector<float> results(n + 1);

        switch (version) {
            case VERSION_SD2: /* fallthrough */
                LOG_WARN("AYS not designed for SD2.X models");
            case VERSION_SD1:
                LOG_INFO("AYS using SD1.5 noise levels");
                inputs = noise_levels[0];
                break;
            case VERSION_SDXL:
                LOG_INFO("AYS using SDXL noise levels");
                inputs = noise_levels[1];
                break;
            case VERSION_SVD:
                LOG_INFO("AYS using SVD noise levels");
                inputs = noise_levels[2];
                break;
            default:
                LOG_ERROR("Version not compatable with AYS scheduler");
                return results;
        }

        /* Stretches those pre-calculated reference levels out to the desired
         * size using log-linear interpolation */
        if ((n + 1) != inputs.size()) {
            results = log_linear_interpolation(inputs, n + 1);
        } else {
            results = inputs;
        }

        /* Not sure if this is strictly neccessary */
        results[n] = 0.0f;

        return results;
    }
};

/*
 * GITS Scheduler: https://github.com/zju-pi/diff-sampler/tree/main/gits-main
 */
struct GITSSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        if (sigma_max <= 0.0f) {
            return std::vector<float>{};
        }

        std::vector<float> sigmas;

        // Assume coeff is provided (replace 1.20 with your dynamic coeff)
        float coeff = 1.20f;  // Default coefficient
        // Normalize coeff to the closest value in the array (0.80 to 1.50)
        coeff = std::round(coeff * 20.0f) / 20.0f;  // Round to the nearest 0.05
        // Calculate the index based on the coefficient
        int index = static_cast<int>((coeff - 0.80f) / 0.05f);
        // Ensure the index is within bounds
        index                                                 = std::max(0, std::min(index, static_cast<int>(GITS_NOISE.size() - 1)));
        const std::vector<std::vector<float>>& selected_noise = *GITS_NOISE[index];

        if (n <= 20) {
            sigmas = (selected_noise)[n - 2];
        } else {
            sigmas = log_linear_interpolation(selected_noise.back(), n + 1);
        }

        sigmas[n] = 0.0f;
        return sigmas;
    }
};

struct KarrasSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float rho = 7.f;

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

struct Denoiser {
    std::shared_ptr<SigmaSchedule> schedule                                                  = std::make_shared<DiscreteSchedule>();
    virtual float sigma_min()                                                                = 0;
    virtual float sigma_max()                                                                = 0;
    virtual float sigma_to_t(float sigma)                                                    = 0;
    virtual float t_to_sigma(float t)                                                        = 0;
    virtual std::vector<float> get_scalings(float sigma)                                     = 0;
    virtual ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) = 0;
    virtual ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent)             = 0;

    virtual std::vector<float> get_sigmas(uint32_t n) {
        auto bound_t_to_sigma = std::bind(&Denoiser::t_to_sigma, this, std::placeholders::_1);
        return schedule->get_sigmas(n, sigma_min(), sigma_max(), bound_t_to_sigma);
    }
};

struct CompVisDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    float sigma_data = 1.0f;

    float sigma_min() {
        return sigmas[0];
    }

    float sigma_max() {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }

    std::vector<float> get_scalings(float sigma) {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) {
        ggml_tensor_scale(noise, sigma);
        ggml_tensor_add(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) {
        return latent;
    }
};

struct CompVisVDenoiser : public CompVisDenoiser {
    std::vector<float> get_scalings(float sigma) {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

float time_snr_shift(float alpha, float t) {
    if (alpha == 1.0f) {
        return t;
    }
    return alpha * t / (1 + (alpha - 1) * t);
}

struct DiscreteFlowDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float shift = 3.0f;

    float sigma_data = 1.0f;

    DiscreteFlowDenoiser() {
        set_parameters();
    }

    void set_parameters() {
        for (int i = 1; i < TIMESTEPS + 1; i++) {
            sigmas[i - 1] = t_to_sigma(i);
        }
    }

    float sigma_min() {
        return sigmas[0];
    }

    float sigma_max() {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) {
        return sigma * 1000.f;
    }

    float t_to_sigma(float t) {
        t = t + 1;
        return time_snr_shift(shift, t / 1000.f);
    }

    std::vector<float> get_scalings(float sigma) {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) {
        ggml_tensor_scale(noise, sigma);
        ggml_tensor_scale(latent, 1.0f - sigma);
        ggml_tensor_add(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) {
        ggml_tensor_scale(latent, 1.0f / (1.0f - sigma));
        return latent;
    }
};

float flux_time_shift(float mu, float sigma, float t) {
    return std::exp(mu) / (std::exp(mu) + std::pow((1.0 / t - 1.0), sigma));
}

struct FluxFlowDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float shift = 1.15f;

    float sigma_data = 1.0f;

    FluxFlowDenoiser(float shift = 1.15f) {
        set_parameters(shift);
    }

    void set_parameters(float shift = 1.15f) {
        this->shift = shift;
        for (int i = 1; i < TIMESTEPS + 1; i++) {
            sigmas[i - 1] = t_to_sigma(i / TIMESTEPS * TIMESTEPS);
        }
    }

    float sigma_min() {
        return sigmas[0];
    }

    float sigma_max() {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) {
        return sigma;
    }

    float t_to_sigma(float t) {
        t = t + 1;
        return flux_time_shift(shift, 1.0f, t / TIMESTEPS);
    }

    std::vector<float> get_scalings(float sigma) {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) {
        ggml_tensor_scale(noise, sigma);
        ggml_tensor_scale(latent, 1.0f - sigma);
        ggml_tensor_add(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) {
        ggml_tensor_scale(latent, 1.0f / (1.0f - sigma));
        return latent;
    }
};

typedef std::function<ggml_tensor*(ggml_tensor*, float, int)> denoise_cb_t;

static inline float* array_view(const ggml_tensor* x) {
    return (float*)x->data;
}

// Converts a denoiser output to a Karras ODE derivative.
static inline void to_d(ggml_tensor* d, ggml_tensor* x, float sigma, ggml_tensor* denoised) {
    for (int i = 0; i < ggml_nelements(d); i++) {
        array_view(d)[i] = (array_view(x)[i] - array_view(denoised)[i]) / sigma;
    }
}

auto default_noise_sampler(ggml_tensor* x) {
    auto rng = std::make_shared<STDDefaultRNG>();
    return [x, rng](float sigma, float sigma_next) {
        return rng->randn(ggml_nelements(x));
    };
}

static std::pair<float, float> get_ancestral_step(float sigma_from, float sigma_to, float eta = 1.f) {
    if (eta == 0.f) {
        return {sigma_to, 0.f};
    }
    float sigma_from_2 = sigma_from * sigma_from;
    float sigma_to_2   = sigma_to * sigma_to;
    float sigma_up = std::min<float>(sigma_to, eta * std::sqrt(sigma_to_2 * (sigma_from_2 - sigma_to_2) / sigma_from_2));
    float sigma_down = std::sqrt(sigma_to_2 - sigma_up * sigma_up);
    return {sigma_down, sigma_up};
}

static inline void do_euler_step(ggml_tensor* dst, ggml_tensor* x, ggml_tensor* d, float dt){
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(x) && ggml_nelements(x) == ggml_nelements(d));
    for (int i = 0; i < ggml_nelements(dst); i++) {
        array_view(dst)[i] = array_view(x)[i] + array_view(d)[i] * dt;
    }
}

// Implements Algorithm 2 (Euler steps) from Karras et al. (2022).
static struct ggml_tensor* sample_euler(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas, 
    std::shared_ptr<RNG> rng,
    float s_churn = 0.f, 
    float s_tmin = 0.f, 
    float s_tmax = std::numeric_limits<float>::infinity(), 
    float s_noise = 1.f
) {
    auto d = ggml_dup_tensor(work_ctx, x);
    for (int i = 0; i < sigmas.size() - 1; i++) {
        float gamma = s_tmin <= sigmas[i] && sigmas[i] <= s_tmax
            ? std::min<float>(s_churn / (sigmas.size() - 1), std::sqrt(2.f) - 1.f)
            : 0.f;
        float sigma_hat = sigmas[i] * (gamma + 1);
        if (gamma > 0) {
            auto eps = rng->randn(ggml_nelements(x));
            for (int j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] += eps[j] * s_noise * std::sqrt(sigma_hat * sigma_hat - sigmas[i] * sigmas[i]);
            }
        }
        auto denoised = model(x, sigma_hat, i + 1);
        to_d(d, x, sigma_hat, denoised);
        // Euler method
        float dt = sigmas[i + 1] - sigma_hat;
        do_euler_step(x, x, d, dt);
    }
    return x;
}

// Ancestral sampling with Euler method steps.
static struct ggml_tensor* sample_euler_ancestral(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas,
    std::function<std::vector<float>(float,float)> noise_sampler = NULL,
    float eta = 1.f,
    float s_noise = 1.f
) {
    noise_sampler = noise_sampler ? noise_sampler : default_noise_sampler(x);
    auto d        = ggml_dup_tensor(work_ctx, x);
    for (int i = 0; i < sigmas.size() - 1; i++) {
        auto denoised = model(x, sigmas[i], i + 1);
        auto [sigma_down, sigma_up] = get_ancestral_step(sigmas[i], sigmas[i + 1], eta);
        to_d(d, x, sigmas[i], denoised);
        // Euler method
        float dt = sigma_down - sigmas[i];
        do_euler_step(x, x, d, dt);
        if (sigmas[i + 1] > 0) {
            auto noise   = noise_sampler(sigmas[i], sigmas[i + 1]);
            for (int j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] += noise[j] * s_noise * sigma_up;
            }
        }
    }
    return x;
}

// Implements Algorithm 1 (Heun steps) from Karras et al. (2022).
static struct ggml_tensor* sample_heun(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas,
    std::function<std::vector<float>(float,float)> noise_sampler = NULL,
    float s_churn = 0.f,
    float s_tmin = 0.f,
    float s_tmax = std::numeric_limits<float>::infinity(),
    float s_noise = 1.f
) {
    noise_sampler = noise_sampler ? noise_sampler : default_noise_sampler(x);
    auto d        = ggml_dup_tensor(work_ctx, x);
    auto d_2      = ggml_dup_tensor(work_ctx, x);
    auto x_2      = ggml_dup_tensor(work_ctx, x);
    for (int i = 0; i < sigmas.size() - 1; i++) {
        float gamma = s_tmin <= sigmas[i] && sigmas[i] <= s_tmax
            ? std::min<float>(s_churn / (sigmas.size() - 1), std::sqrt(2) - 1)
            : 0.f;
        float sigma_hat = sigmas[i] * (gamma + 1);
        if (gamma > 0.f) {
            auto eps = noise_sampler(sigmas[i], sigmas[i + 1]);
            for (int j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] += eps[j] * s_noise * std::sqrt(sigma_hat * sigma_hat - sigmas[i] * sigmas[i]);
            }
        }
        auto denoised = model(x, sigma_hat, i + 1);
        to_d(d, x, sigma_hat, denoised);
        float dt = sigmas[i + 1] - sigma_hat;
        if (sigmas[i + 1] == 0) {
            // Euler method
            do_euler_step(x, x, d, dt);
        } else {
            // Heun's method
            do_euler_step(x_2, x, d, dt);
            auto denoised_2 = model(x_2, sigmas[i + 1], i + 1);
            to_d(d_2, x_2, sigmas[i + 1], denoised_2);
            for (int j = 0; j < ggml_nelements(x); j++) {
                float d_avg = (array_view(d)[j] + array_view(d_2)[j]) * .5f;
                array_view(x)[j] += d_avg * dt;
            }
        }
    }
    return x;
}

// A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).
static struct ggml_tensor* sample_dpm_2(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas,
    std::shared_ptr<RNG> rng,
    float s_churn = 0.f,
    float s_tmin = 0.f,
    float s_tmax = std::numeric_limits<float>::infinity(),
    float s_noise = 1.f
) {
    auto d   = ggml_dup_tensor(work_ctx, x);
    auto d_2 = ggml_dup_tensor(work_ctx, x);
    auto x_2 = ggml_dup_tensor(work_ctx, x);
    for (int i = 0; i < sigmas.size() - 1; i++) {
        float gamma = s_tmin <= sigmas[i] && sigmas[i] <= s_tmax
            ? std::min<float>(s_churn / (sigmas.size() - 1), std::sqrt(2.f) - 1.f)
            : 0.f;
        float sigma_hat = sigmas[i] * (gamma + 1);
        if (gamma > 0) {
            auto eps = rng->randn(ggml_nelements(x));
            for (int j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] += eps[j] * s_noise * std::sqrt(sigma_hat * sigma_hat - sigmas[i] * sigmas[i]);
            }
        }
        auto denoised = model(x, sigma_hat, i + 1);
        to_d(d, x, sigma_hat, denoised);
        if (sigmas[i + 1] == 0) {
            // Euler method.
            float dt = sigmas[i + 1] - sigma_hat;
            do_euler_step(x, x, d, dt);
        } else {
            // DPM-Solver-2
            float sigma_mid = exp((log(sigma_hat) + log(sigmas[i + 1])) * .5f);
            float dt_1      = sigma_mid - sigma_hat;
            float dt_2      = sigmas[i + 1] - sigma_hat;
            do_euler_step(x_2, x, d, dt_1);
            auto denoised_2 = model(x_2, sigma_mid, i + 1);
            to_d(d_2, x_2, sigma_mid, denoised_2);
            do_euler_step(x, x, d_2, dt_2);
        }
    }
    return x;
}

// Ancestral sampling with DPM-Solver second-order steps.
static struct ggml_tensor* sample_dpm_2_ancestral(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas,
    std::function<std::vector<float>(float, float)> noise_sampler = NULL,
    float eta = 1.f,
    float s_noise = 1.f
) {
    noise_sampler = noise_sampler ? noise_sampler : default_noise_sampler(x);
    auto d   = ggml_dup_tensor(work_ctx, x);
    auto d_2 = ggml_dup_tensor(work_ctx, x);
    auto x_2 = ggml_dup_tensor(work_ctx, x);
    for (int i = 0; i < sigmas.size() - 1; i++) {
        auto denoised = model(x, sigmas[i], i + 1);
        auto [sigma_down, sigma_up] = get_ancestral_step(sigmas[i], sigmas[i + 1], eta);
        to_d(d, x, sigmas[i], denoised);
        if (sigma_down == 0.f) {
            // Euler method
            float dt = sigma_down - sigmas[i];
            do_euler_step(x, x, d, dt);
        } else {
            // DPM-Solver-2
            float sigma_mid = exp((log(sigmas[i]) + log(sigma_down)) * .5f);
            float dt_1      = sigma_mid - sigmas[i];
            float dt_2      = sigma_down - sigmas[i];
            do_euler_step(x_2, x, d, dt_1);
            auto denoised_2 = model(x_2, sigma_mid, i + 1);
            to_d(d_2, x_2, sigma_mid, denoised_2);
            do_euler_step(x, x, d_2, dt_2);
            auto noise = noise_sampler(sigmas[i], sigmas[i + 1]);
            for (int j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] += noise[j] * s_noise * sigma_up;
            }
        }
    }
    return x;
}

float integrate_quad(const std::function<float(float)>& f, float a, float b, float tol, int max_depth = 20) {
    auto simpsons_rule = [](const std::function<float(float)>& f, float a, float b) {
        float mid = (a + b) * .5f;
        return (b - a) / 6.f * (f(a) + 4.f * f(mid) + f(b));
    };
    float mid = (a + b) * .5f;
    float whole = simpsons_rule(f, a, b);
    float left = simpsons_rule(f, a, mid);
    float right = simpsons_rule(f, mid, b);
    float error = std::fabs(left + right - whole);
    if (error < 15.f * tol || max_depth <= 0) {
        return left + right + (left + right - whole) / 15.f;
    }
    return integrate_quad(f, a, mid, tol * .5f, max_depth - 1) +
           integrate_quad(f, mid, b, tol * .5f, max_depth - 1);
}

float linear_multistep_coeff(int order, std::vector<float> t, int i, int j) {
    GGML_ASSERT(order - 1 <= i);
    auto fn = [order,t, i, j](float tau) {
        float prod = 1.f;
        for (int k = 0; k < order; k++) {
            if (j == k) {
                continue;
            }
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k]);
        }
        return prod;
    };
    return integrate_quad(fn, t[i], t[i + 1], 1e-4f);
}

static struct ggml_tensor* sample_lms(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas,
    int order = 4
) {
    ggml_init_params params = {0};
    params.mem_size = (ggml_nbytes(x) + ggml_tensor_overhead()) * sigmas.size();
    params.mem_buffer = NULL;
    params.no_alloc   = false;
    ggml_context* temp_ctx  = ggml_init(params);
    GGML_ASSERT(temp_ctx != NULL);
    std::deque<ggml_tensor*> ds;
    for (int i = 0; i < sigmas.size() - 1; i++) {
        auto denoised = model(x, sigmas[i], i + 1);
        auto d = ggml_dup_tensor(temp_ctx, x);
        to_d(d, x, sigmas[i], denoised);
        ds.push_back(d);
        if (ds.size() > order) {
            ds.pop_front();
        }
        int cur_order = std::min<int>(i + 1, order);
        std::vector<float> coeffs;
        coeffs.reserve(cur_order);
        for (int j = 0; j < cur_order; j++) {
            coeffs.push_back(linear_multistep_coeff(cur_order, sigmas, i, j));
        }
        for (int j = 0; j < ggml_nelements(x); j++) {
            float sum = 0;
            for (int k = 0; k < cur_order; k++) {
                sum += coeffs[k] * array_view(ds[cur_order - 1 - k])[j];
            }
            array_view(x)[j] += sum;
        }
    }
    ggml_free(temp_ctx);
    return x;
}

// A PID controller for ODE adaptive step size control.
struct PIDStepSizeController {
    float h;
    float b1;
    float b2;
    float b3;
    float accept_safety;
    float eps;
    std::vector<float> errs;
    PIDStepSizeController(float h, float pcoeff, float icoeff, float dcoeff, float order = 1.f, float accept_safety = 0.81f, float eps = 1e-8f)
        : h(h), 
          b1(pcoeff + icoeff + dcoeff),
          b2(-(pcoeff + 2 * dcoeff) / order),
          b3(dcoeff / order),
          accept_safety(accept_safety), 
          eps(eps) {
    }
    float limiter(float x) {
        return 1 + std::atan(x - 1);
    }
    bool propose_step(float error) {
        float inv_error = 1.f / (error + eps);
        if (errs.empty()) {
            errs = {inv_error, inv_error, inv_error};
        }
        errs[0] = inv_error;
        float factor = pow(errs[0], b1) * pow(errs[1], b2) * pow(errs[2], b3);
        factor       = limiter(factor);
        bool accept  = factor >= accept_safety;
        if (accept) {
            errs[2] = errs[1];
            errs[1] = errs[0];
        }
        h *= factor;
        return accept;
    }
};

static inline std::vector<float> linspace(float start, float end, int steps) {
    assert(steps > 0);
    std::vector<float> result;
    result.reserve(steps);
    float step_size = (end - start) / (steps - 1);
    for (int i = 0; i < steps; i++) {
        result.push_back(start + i * step_size);
    }
    return result;
}

static float vector_norm(const std::vector<float>& vec, float p = 2.0) {
    assert(!vec.empty());
    if (p == 0) {
        return std::count_if(vec.begin(), vec.end(), [](float v) { return v != 0; });
    } else if (p == std::numeric_limits<float>::infinity()) {
        return *std::max_element(vec.begin(), vec.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
    } else {
        float sum = 0.0;
        for (float v : vec) {
            sum += std::pow(std::abs(v), p);
        }
        return std::pow(sum, 1.0 / p);
    }
}

// DPM-Solver. See https://arxiv.org/abs/2206.00927.
struct DPMSolver {
    ggml_context* work_ctx;
    denoise_cb_t model; 
    DPMSolver(ggml_context* work_ctx, denoise_cb_t model)
        : work_ctx(work_ctx), model(model) {
    }
    float to_t(float sigma) {
        return -log(sigma);
    }
    float to_sigma(float t) {
        return exp(-t);
    }
    ggml_tensor* calc_eps(ggml_tensor* x, float t, int cur_step=-1) {
        float sigma   = to_sigma(t);
        auto denoised = model(x, sigma, cur_step);
        auto eps      = ggml_dup_tensor(work_ctx, x);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(eps)[i] = (array_view(x)[i] - array_view(denoised)[i]) / sigma;
        }
        return eps;
    }
    ggml_tensor* dpm_solver_1_step(ggml_tensor* x, float t, float t_next, int cur_step = -1) {
        float h = t_next - t;
        auto eps = calc_eps(x, t, cur_step);
        auto x_1 = ggml_dup_tensor(work_ctx, x);
        float sigma = to_sigma(t_next);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(x_1)[i] = array_view(x)[i] - sigma * expm1(h) * array_view(eps)[i];
        }
        return x_1;
    }
    ggml_tensor* dpm_solver_2_step(ggml_tensor* x, float t, float t_next, int cur_step = -1, float r1 = .5f) {
        float h  = t_next - t;
        auto eps = calc_eps(x, t, cur_step);
        float s1 = t + r1 * h;
        auto u1  = ggml_dup_tensor(work_ctx, x);
        float sigma = to_sigma(s1);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(u1)[i] = array_view(x)[i] - sigma * expm1(r1 * h) * array_view(eps)[i];
        }
        auto eps_r1 = calc_eps(u1, s1, cur_step);
        auto x_2    = ggml_dup_tensor(work_ctx, x);
        sigma       = to_sigma(t_next);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(x_2)[i] = array_view(x)[i] - sigma * expm1(h) * array_view(eps)[i] -
                sigma / (2 * r1) * expm1(h) * (array_view(eps_r1)[i] - array_view(eps)[i]);
        }
        return x_2;
    }
    ggml_tensor* dpm_solver_3_step(ggml_tensor* x, float t, float t_next, int cur_step = -1, float r1 = 1.f / 3.f, float r2 = 2.f / 3.f) {
        float h = t_next - t;
        auto eps = calc_eps(x, t, cur_step);
        float s1 = t + r1 * h;
        float s2 = t + r2 * h;
        auto u1  = ggml_dup_tensor(work_ctx, x);
        float sigma = to_sigma(s1);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(u1)[i] = array_view(x)[i] - sigma * expm1(r1 * h) * array_view(eps)[i];
        }
        auto eps_r1 = calc_eps(u1, s1, cur_step);
        auto u2     = ggml_dup_tensor(work_ctx, x);
        sigma       = to_sigma(s2);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(u2)[i] = array_view(x)[i] - sigma * expm1(r2 * h) * array_view(eps)[i] -
                sigma * (r2 / r1) * (expm1(r2 * h) / (r2 * h) - 1) * (array_view(eps_r1)[i] - array_view(eps)[i]);
        }
        auto eps_r2 = calc_eps(u2, s2, cur_step);
        auto x_3    = ggml_dup_tensor(work_ctx, x);
        sigma       = to_sigma(t_next);
        for (size_t i = 0; i < ggml_nelements(x); i++) {
            array_view(x_3)[i] = array_view(x)[i] - sigma * expm1(h) * array_view(eps)[i] -
                sigma / r2 * (expm1(h) / h - 1) * (array_view(eps_r2)[i] - array_view(eps)[i]);
        }
        return x_3;
    }
    ggml_tensor* dpm_solver_fast(
        ggml_tensor* x,
        float t_start,
        float t_end,
        int nfe,
        float eta = 0.f,
        float s_noise = 1.f,
        std::function<std::vector<float>(float, float)> noise_sampler = NULL
    ) {
        if (t_start <= t_end && eta != 0.f) {
            LOG_ERROR("eta must be 0 for reverse sampling");
            return NULL;
        }
        noise_sampler = noise_sampler ? noise_sampler : default_noise_sampler(x);
        float m = floor(nfe / 3.f) + 1;
        std::vector<float> ts = linspace(t_start, t_end, m + 1);
        std::vector<int> orders (m, 3);
        if (nfe % 3 == 0) {
            orders[m - 2] = 2;
            orders[m - 1] = 1;
        } else {
            orders[m - 1] = nfe % 3;
        }
        for (int i = 0; i < orders.size(); i++) {
            float t = ts[i];
            float t_next = ts[i + 1];
            float t_next_, sd, su;
            if (eta != 0.f) {
                std::tie(sd, su) = get_ancestral_step(to_sigma(t), to_sigma(t_next), eta);
                t_next_ = std::min<float>(t_end, to_t(sd));
                su      = std::sqrt(to_sigma(t_next) * to_sigma(t_next) - to_sigma(t_next_) * to_sigma(t_next_));
            } else {
                t_next_ = t_next;
                su      = 0.f;
            }
            if (orders[i] == 1) {
                x = dpm_solver_1_step(x, t, t_next_, i * 3);
            } else if (orders[i] == 2) {
                x = dpm_solver_2_step(x, t, t_next_, i * 3);
            } else {
                x = dpm_solver_3_step(x, t, t_next_, i * 3);
            }
            if (su * s_noise != 0.f) {
                auto noise = noise_sampler(to_sigma(t), to_sigma(t_next));
                for (int j = 0; j < ggml_nelements(x); j++) {
                    array_view(x)[i] += su * s_noise * noise[i];
                }
            }
        }
        return x;
    }
    ggml_tensor* dpm_solver_adaptive(
        ggml_tensor* x,
        float t_start,
        float t_end,
        int order = 3,
        float rtol = 0.05f,
        float atol = 0.0078f,
        float h_init = 0.05f,
        float pcoeff = 0.f,
        float icoeff = 1.f,
        float dcoeff = 0.f,
        float accept_safety = 0.81f,
        float eta = 0.f,
        float s_noise = 1.f,
        std::function<std::vector<float>(float, float)> noise_sampler = NULL
    ) {
        if (order != 2 && order != 3) {
            LOG_ERROR("order should be 2 or 3");
            return NULL;
        }
        bool forward = t_end > t_start;
        if (!forward && eta != 0.f) {
            LOG_ERROR("eta must be 0 for reverse sampling");
            return NULL;
        }
        noise_sampler = noise_sampler ? noise_sampler : default_noise_sampler(x);
        h_init        = abs(h_init) * (forward ? 1 : - 1);
        float s       = t_start;
        ggml_tensor* x_prev = x;
        PIDStepSizeController pid(h_init, pcoeff, icoeff, dcoeff, eta == 0.f ? order : 1.5f, accept_safety);
        int cur_step = 0;
        auto x_low   = ggml_dup_tensor(work_ctx, x);
        auto x_high  = ggml_dup_tensor(work_ctx, x);
        while (forward ? s < t_end - 1e-5f : s > t_end + 1e-5f) {
            cur_step++;
            float t = forward ? std::min<float>(t_end, s + pid.h) : std::max<float>(t_end, s + pid.h);
            float t_, sd, su;
            if (eta != 0.f) {
                std::tie(sd, su) = get_ancestral_step(to_sigma(s), to_sigma(t), eta);
                t_               = std::min<float>(t_end, to_t(sd));
                su               = std::sqrt(to_sigma(t) * to_sigma(t) - to_sigma(t_) * to_sigma(t_));
            } else {
                t_ = t;
                su = 0.f;
            }
            auto eps = calc_eps(x, s, cur_step);
            if (order == 2) {
                x_low  = dpm_solver_1_step(x, s, t_, cur_step);
                x_high = dpm_solver_2_step(x, s, t_, cur_step);
            } else {
                x_low = dpm_solver_2_step(x, s, t_, cur_step, 1.f / 3.f);
                x_high = dpm_solver_3_step(x, s, t_, cur_step);
            }
            std::vector<float> delta;
            delta.reserve(ggml_nelements(x));
            for (size_t i = 0; i < ggml_nelements(x); i++) {
                float d_ = std::max<float>(atol, rtol * std::max<float>(abs(array_view(x_low)[i]), abs(array_view(x_prev)[i])));
                delta.push_back((array_view(x_low)[i] - array_view(x_high)[i]) / d_);
            }
            float error = vector_norm(delta) / std::sqrt(ggml_nelements(x));
            if (pid.propose_step(error)) {
                x_prev = x_low;
                auto noise = noise_sampler(to_sigma(s), to_sigma(t));
                for (size_t i = 0; i < ggml_nelements(x); i++) {
                    array_view(x)[i] = array_view(x_high)[i] + su * s_noise * noise[i];
                }
                s = t;
            }
        }
        return x;
    }
};

// DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927.
static struct ggml_tensor* sample_dpm_fast(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    float sigma_min,
    float sigma_max,
    int n,
    std::function<std::vector<float>(float, float)> noise_sampler = NULL,
    float eta = 0.f,
    float s_noise = 1.f
) {
    assert(sigma_min > 0 && sigma_max > 0);
    DPMSolver dpm_solver (work_ctx, model);
    auto result = dpm_solver.dpm_solver_fast(x, dpm_solver.to_t(sigma_max), dpm_solver.to_t(sigma_min), n, eta, s_noise, noise_sampler);
    for (size_t i = 0; i < ggml_nelements(x); i++) {
        array_view(x)[i] = array_view(result)[i];
    }
    return x;
}

// DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927.
static struct ggml_tensor* sample_dpm_adaptive(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    float sigma_min,
    float sigma_max,
    std::function<std::vector<float>(float, float)> noise_sampler = NULL,
    float eta = 0.,
    float s_noise = 1.f,
    int order = 3,
    float rtol = 0.05f,
    float atol = 0.0078f,
    float h_init = 0.05f,
    float pcoeff = 0.f,
    float icoeff = 1.f,
    float dcoeff = 0.f,
    float accept_safety = 0.81
) {
    assert(sigma_min > 0 && sigma_max > 0);
    DPMSolver dpm_solver(work_ctx, model);
    auto result = dpm_solver.dpm_solver_adaptive(
        x, dpm_solver.to_t(sigma_max), dpm_solver.to_t(sigma_min),
        order, rtol, atol, h_init, pcoeff, icoeff, dcoeff,
        accept_safety, eta, s_noise, noise_sampler
    );
    for (size_t i = 0; i < ggml_nelements(x); i++) {
        array_view(x)[i] = array_view(result)[i];
    }
    return x;
}

// Ancestral sampling with DPM-Solver++(2S) second-order steps.
static struct ggml_tensor* sample_dpmpp_2s_ancestral(
    ggml_context* work_ctx,
    denoise_cb_t model,
    ggml_tensor* x,
    std::vector<float> sigmas,
    std::function<std::vector<float>(float, float)> noise_sampler = NULL,
    float eta = 1.f,
    float s_noise = 1.f
) {
    noise_sampler = noise_sampler ? noise_sampler : default_noise_sampler(x);
    auto sigma_fn = [](float t) { return exp(-t); };
    auto t_fn     = [](float sigma) { return -log(sigma); };
    auto d        = ggml_dup_tensor(work_ctx, x);
    auto x_2      = ggml_dup_tensor(work_ctx, x);
    for (int i = 0; i < sigmas.size() - 1; i++) {
        auto denoised = model(x, sigmas[i], i + 1);
        auto [sigma_down, sigma_up] = get_ancestral_step(sigmas[i], sigmas[i + 1], eta);
        if (sigma_down == 0.f) {
            // Euler method
            to_d(d, x, sigmas[i], denoised);
            float dt = sigma_down - sigmas[i];
            do_euler_step(x, x, d, dt);
        } else {
            // DPM-Solver++(2S)
            float t      = t_fn(sigmas[i]);
            float t_next = t_fn(sigma_down);
            float r      = .5f;
            float h      = t_next - t;
            float s      = t + r * h;
            float s_t    = sigma_fn(s) / sigma_fn(t);
            for (size_t j = 0; j < ggml_nelements(x); j++) {
                array_view(x_2)[j] = s_t * array_view(x)[j] - expm1(-h * r) * array_view(denoised)[j];
            }
            auto denoised_2 = model(x_2, sigma_fn(s), i + 1);
            s_t             = sigma_fn(t_next) / sigma_fn(t);
            for (size_t j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] = s_t * array_view(x)[j] - expm1(-h) * array_view(denoised_2)[j];
            }
        }
        // Noise addition
        if (sigmas[i + 1] > 0.f && s_noise * sigma_up != 0.f) {
            auto noise = noise_sampler(sigmas[i], sigmas[i + 1]);
            for (size_t j = 0; j < ggml_nelements(x); j++) {
                array_view(x)[j] += noise[j] * s_noise * sigma_up;
            }
        }
    }
    return x;
}

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = t
static void sample_k_diffusion(sample_method_t method,
                               denoise_cb_t model,
                               ggml_context* work_ctx,
                               ggml_tensor* x,
                               std::vector<float> sigmas,
                               std::shared_ptr<RNG> rng,
                               float eta) {
    size_t steps = sigmas.size() - 1;

    auto noise_sampler = [x, rng](float sigma, float sigma_next) {
        return rng->randn(ggml_nelements(x));
    };

    switch (method) {
        case EULER: sample_euler(work_ctx, model, x, sigmas, rng); break;
        case EULER_A: sample_euler_ancestral(work_ctx, model, x, sigmas, noise_sampler); break;
        case HEUN: sample_heun(work_ctx, model, x, sigmas, noise_sampler); break;
        case DPM2: sample_dpm_2(work_ctx, model, x, sigmas, rng); break;
        case DPM2_A: sample_dpm_2_ancestral(work_ctx, model, x, sigmas, noise_sampler); break;
        case LMS: sample_lms(work_ctx, model, x, sigmas); break;
        case DPM_FAST: sample_dpm_fast(work_ctx, model, x, sigmas[sigmas.size() - (sigmas.back() == 0.f ? 2 : 1)], sigmas[0], sigmas.size() - 1, noise_sampler); break;
        case DPM_ADAPTIVE: sample_dpm_adaptive(work_ctx, model, x, sigmas[sigmas.size() - (sigmas.back() == 0.f ? 2 : 1)], sigmas[0], noise_sampler); break;
        case DPMPP2S_A: sample_dpmpp_2s_ancestral(work_ctx, model, x, sigmas, noise_sampler); break;
        case DPMPP2M:  // DPM++ (2M) from Karras et al (2022)
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float b                 = exp(-h) - 1.f;
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float r      = h_last / h;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case DPMPP2Mv2:  // Modified DPM++ (2M) from https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    float b = exp(-h) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float h_min  = std::min(h_last, h);
                    float h_max  = std::max(h_last, h);
                    float r      = h_max / h_min;
                    float h_d    = (h_max + h_min) / 2.f;
                    float b      = exp(-h_d) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case IPNDM:  // iPNDM sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
        {
            int max_order       = 4;
            ggml_tensor* x_next = x;
            std::vector<ggml_tensor*> buffer_model;

            for (int i = 0; i < steps; i++) {
                float sigma      = sigmas[i];
                float sigma_next = sigmas[i + 1];

                ggml_tensor* x_cur = x_next;
                float* vec_x_cur   = (float*)x_cur->data;
                float* vec_x_next  = (float*)x_next->data;

                // Denoising step
                ggml_tensor* denoised = model(x_cur, sigma, i + 1);
                float* vec_denoised   = (float*)denoised->data;
                // d_cur = (x_cur - denoised) / sigma
                struct ggml_tensor* d_cur = ggml_dup_tensor(work_ctx, x_cur);
                float* vec_d_cur          = (float*)d_cur->data;

                for (int j = 0; j < ggml_nelements(d_cur); j++) {
                    vec_d_cur[j] = (vec_x_cur[j] - vec_denoised[j]) / sigma;
                }

                int order = std::min(max_order, i + 1);

                // Calculate vec_x_next based on the order
                switch (order) {
                    case 1:  // First Euler step
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * vec_d_cur[j];
                        }
                        break;

                    case 2:  // Use one history point
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (3 * vec_d_cur[j] - vec_d_prev1[j]) / 2;
                        }
                    } break;

                    case 3:  // Use two history points
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (float*)buffer_model[buffer_model.size() - 2]->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (23 * vec_d_cur[j] - 16 * vec_d_prev1[j] + 5 * vec_d_prev2[j]) / 12;
                        }
                    } break;

                    case 4:  // Use three history points
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (float*)buffer_model[buffer_model.size() - 2]->data;
                        float* vec_d_prev3 = (float*)buffer_model[buffer_model.size() - 3]->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (55 * vec_d_cur[j] - 59 * vec_d_prev1[j] + 37 * vec_d_prev2[j] - 9 * vec_d_prev3[j]) / 24;
                        }
                    } break;
                }

                // Manage buffer_model
                if (buffer_model.size() == max_order - 1) {
                    // Shift elements to the left
                    for (int k = 0; k < max_order - 2; k++) {
                        buffer_model[k] = buffer_model[k + 1];
                    }
                    buffer_model.back() = d_cur;  // Replace the last element with d_cur
                } else {
                    buffer_model.push_back(d_cur);
                }
            }
        } break;
        case IPNDM_V:  // iPNDM_v sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
        {
            int max_order = 4;
            std::vector<ggml_tensor*> buffer_model;
            ggml_tensor* x_next = x;

            for (int i = 0; i < steps; i++) {
                float sigma  = sigmas[i];
                float t_next = sigmas[i + 1];

                // Denoising step
                ggml_tensor* denoised     = model(x, sigma, i + 1);
                float* vec_denoised       = (float*)denoised->data;
                struct ggml_tensor* d_cur = ggml_dup_tensor(work_ctx, x);
                float* vec_d_cur          = (float*)d_cur->data;
                float* vec_x              = (float*)x->data;

                // d_cur = (x - denoised) / sigma
                for (int j = 0; j < ggml_nelements(d_cur); j++) {
                    vec_d_cur[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                }

                int order   = std::min(max_order, i + 1);
                float h_n   = t_next - sigma;
                float h_n_1 = (i > 0) ? (sigma - sigmas[i - 1]) : h_n;

                switch (order) {
                    case 1:  // First Euler step
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += vec_d_cur[j] * h_n;
                        }
                        break;

                    case 2: {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((2 + (h_n / h_n_1)) * vec_d_cur[j] - (h_n / h_n_1) * vec_d_prev1[j]) / 2;
                        }
                        break;
                    }

                    case 3: {
                        float h_n_2        = (i > 1) ? (sigmas[i - 1] - sigmas[i - 2]) : h_n_1;
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (buffer_model.size() > 1) ? (float*)buffer_model[buffer_model.size() - 2]->data : vec_d_prev1;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((23 * vec_d_cur[j] - 16 * vec_d_prev1[j] + 5 * vec_d_prev2[j]) / 12);
                        }
                        break;
                    }

                    case 4: {
                        float h_n_2        = (i > 1) ? (sigmas[i - 1] - sigmas[i - 2]) : h_n_1;
                        float h_n_3        = (i > 2) ? (sigmas[i - 2] - sigmas[i - 3]) : h_n_2;
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (buffer_model.size() > 1) ? (float*)buffer_model[buffer_model.size() - 2]->data : vec_d_prev1;
                        float* vec_d_prev3 = (buffer_model.size() > 2) ? (float*)buffer_model[buffer_model.size() - 3]->data : vec_d_prev2;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((55 * vec_d_cur[j] - 59 * vec_d_prev1[j] + 37 * vec_d_prev2[j] - 9 * vec_d_prev3[j]) / 24);
                        }
                        break;
                    }
                }

                // Manage buffer_model
                if (buffer_model.size() == max_order - 1) {
                    buffer_model.erase(buffer_model.begin());
                }
                buffer_model.push_back(d_cur);

                // Prepare the next d tensor
                d_cur = ggml_dup_tensor(work_ctx, x_next);
            }
        } break;
        case LCM:  // Latent Consistency Models
        {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // x = denoised
                {
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_denoised[j];
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
                    ggml_tensor_set_f32_randn(noise, rng);
                    // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + sigmas[i + 1] * vec_noise[j];
                        }
                    }
                }
            }
        } break;
        case DDIM_TRAILING:  // Denoising Diffusion Implicit Models
                             // with the "trailing" timestep spacing
        {
            // See J. Song et al., "Denoising Diffusion Implicit
            // Models", arXiv:2010.02502 [cs.LG]
            //
            // DDIM itself needs alphas_cumprod (DDPM, J. Ho et al.,
            // arXiv:2006.11239 [cs.LG] with k-diffusion's start and
            // end beta) (which unfortunately k-diffusion's data
            // structure hides from the denoiser), and the sigmas are
            // also needed to invert the behavior of CompVisDenoiser
            // (k-diffusion's LMSDiscreteScheduler)
            float beta_start = 0.00085f;
            float beta_end = 0.0120f;
            std::vector<double> alphas_cumprod;
            std::vector<double> compvis_sigmas;

            alphas_cumprod.reserve(TIMESTEPS);
            compvis_sigmas.reserve(TIMESTEPS);
            for (int i = 0; i < TIMESTEPS; i++) {
                alphas_cumprod[i] =
                    (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
                    (1.0f -
                     std::pow(sqrtf(beta_start) +
                              (sqrtf(beta_end) - sqrtf(beta_start)) *
                              ((float)i / (TIMESTEPS - 1)), 2));
                compvis_sigmas[i] =
                    std::sqrt((1 - alphas_cumprod[i]) /
                              alphas_cumprod[i]);
            }

            struct ggml_tensor* pred_original_sample =
                ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* variance_noise =
                ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // The "trailing" DDIM timestep, see S. Lin et al.,
                // "Common Diffusion Noise Schedules and Sample Steps
                // are Flawed", arXiv:2305.08891 [cs], p. 4, Table
                // 2. Most variables below follow Diffusers naming
                //
                // Diffuser naming vs. Song et al. (2010), p. 5, (12)
                // and p. 16, (16) (<variable name> -> <name in
                // paper>):
                //
                // - pred_noise_t -> epsilon_theta^(t)(x_t)
                // - pred_original_sample -> f_theta^(t)(x_t) or x_0
                // - std_dev_t -> sigma_t (not the LMS sigma)
                // - eta -> eta (set to 0 at the moment)
                // - pred_sample_direction -> "direction pointing to
                //   x_t"
                // - pred_prev_sample -> "x_t-1"
                int timestep =
                    roundf(TIMESTEPS -
                           i * ((float)TIMESTEPS / steps)) - 1;
                // 1. get previous step value (=t-1)
                int prev_timestep = timestep - TIMESTEPS / steps;
                // The sigma here is chosen to cause the
                // CompVisDenoiser to produce t = timestep
                float sigma = compvis_sigmas[timestep];
                if (i == 0) {
                    // The function add_noise intializes x to
                    // Diffusers' latents * sigma (as in Diffusers'
                    // pipeline) or sample * sigma (Diffusers'
                    // scheduler), where this sigma = init_noise_sigma
                    // in Diffusers. For DDPM and DDIM however,
                    // init_noise_sigma = 1. But the k-diffusion
                    // model() also evaluates F_theta(c_in(sigma) x;
                    // ...) instead of the bare U-net F_theta, with
                    // c_in = 1 / sqrt(sigma^2 + 1), as defined in
                    // T. Karras et al., "Elucidating the Design Space
                    // of Diffusion-Based Generative Models",
                    // arXiv:2206.00364 [cs.CV], p. 3, Table 1. Hence
                    // the first call has to be prescaled as x <- x /
                    // (c_in * sigma) with the k-diffusion pipeline
                    // and CompVisDenoiser.
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1) /
                            sigma;
                    }
                }
                else {
                    // For the subsequent steps after the first one,
                    // at this point x = latents or x = sample, and
                    // needs to be prescaled with x <- sample / c_in
                    // to compensate for model() applying the scale
                    // c_in before the U-net F_theta
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1);
                    }
                }
                // Note (also noise_pred in Diffuser's pipeline)
                // model_output = model() is the D(x, sigma) as
                // defined in Karras et al. (2022), p. 3, Table 1 and
                // p. 8 (7), compare also p. 38 (226) therein.
                struct ggml_tensor* model_output =
                    model(x, sigma, i + 1);
                // Here model_output is still the k-diffusion denoiser
                // output, not the U-net output F_theta(c_in(sigma) x;
                // ...) in Karras et al. (2022), whereas Diffusers'
                // model_output is F_theta(...). Recover the actual
                // model_output, which is also referred to as the
                // "Karras ODE derivative" d or d_cur in several
                // samplers above.
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_model_output[j] =
                            (vec_x[j] - vec_model_output[j]) *
                            (1 / sigma);
                    }
                }
                // 2. compute alphas, betas
                float alpha_prod_t = alphas_cumprod[timestep];
                // Note final_alpha_cumprod = alphas_cumprod[0] due to
                // trailing timestep spacing
                float alpha_prod_t_prev = prev_timestep >= 0 ?
                    alphas_cumprod[prev_timestep] : alphas_cumprod[0];
                float beta_prod_t = 1 - alpha_prod_t;
                // 3. compute predicted original sample from predicted
                // noise also called "predicted x_0" of formula (12)
                // from https://arxiv.org/pdf/2010.02502.pdf
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    // Note the substitution of latents or sample = x
                    // * c_in = x / sqrt(sigma^2 + 1)
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_pred_original_sample[j] =
                            (vec_x[j] / std::sqrt(sigma * sigma + 1) -
                             std::sqrt(beta_prod_t) *
                             vec_model_output[j]) *
                            (1 / std::sqrt(alpha_prod_t));
                    }
                }
                // Assuming the "epsilon" prediction type, where below
                // pred_epsilon = model_output is inserted, and is not
                // defined/copied explicitly.
                //
                // 5. compute variance: "sigma_t(eta)" -> see formula
                // (16)
                //
                // sigma_t = sqrt((1 - alpha_t-1)/(1 - alpha_t)) *
                // sqrt(1 - alpha_t/alpha_t-1)
                float beta_prod_t_prev = 1 - alpha_prod_t_prev;
                float variance = (beta_prod_t_prev / beta_prod_t) *
                    (1 - alpha_prod_t / alpha_prod_t_prev);
                float std_dev_t = eta * std::sqrt(variance);
                // 6. compute "direction pointing to x_t" of formula
                // (12) from https://arxiv.org/pdf/2010.02502.pdf
                // 7. compute x_t without "random noise" of formula
                // (12) from https://arxiv.org/pdf/2010.02502.pdf
                {
                    float* vec_model_output = (float*)model_output->data;
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        // Two step inner loop without an explicit
                        // tensor
                        float pred_sample_direction =
                            std::sqrt(1 - alpha_prod_t_prev -
                                      std::pow(std_dev_t, 2)) *
                            vec_model_output[j];
                        vec_x[j] = std::sqrt(alpha_prod_t_prev) *
                            vec_pred_original_sample[j] +
                            pred_sample_direction;
                    }
                }
                if (eta > 0) {
                    ggml_tensor_set_f32_randn(variance_noise, rng);
                    float* vec_variance_noise =
                        (float*)variance_noise->data;
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] += std_dev_t * vec_variance_noise[j];
                    }
                }
                // See the note above: x = latents or sample here, and
                // is not scaled by the c_in. For the final output
                // this is correct, but for subsequent iterations, x
                // needs to be prescaled again, since k-diffusion's
                // model() differes from the bare U-net F_theta by the
                // factor c_in.
            }
        } break;
        case TCD:  // Strategic Stochastic Sampling (Algorithm 4) in
                   // Trajectory Consistency Distillation
        {
            // See J. Zheng et al., "Trajectory Consistency
            // Distillation: Improved Latent Consistency Distillation
            // by Semi-Linear Consistency Function with Trajectory
            // Mapping", arXiv:2402.19159 [cs.CV]
            float beta_start = 0.00085f;
            float beta_end = 0.0120f;
            std::vector<double> alphas_cumprod;
            std::vector<double> compvis_sigmas;

            alphas_cumprod.reserve(TIMESTEPS);
            compvis_sigmas.reserve(TIMESTEPS);
            for (int i = 0; i < TIMESTEPS; i++) {
                alphas_cumprod[i] =
                    (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
                    (1.0f -
                     std::pow(sqrtf(beta_start) +
                              (sqrtf(beta_end) - sqrtf(beta_start)) *
                              ((float)i / (TIMESTEPS - 1)), 2));
                compvis_sigmas[i] =
                    std::sqrt((1 - alphas_cumprod[i]) /
                              alphas_cumprod[i]);
            }
            int original_steps = 50;

            struct ggml_tensor* pred_original_sample =
                ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* noise =
                ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // Analytic form for TCD timesteps
                int timestep = TIMESTEPS - 1 -
                    (TIMESTEPS / original_steps) *
                    (int)floor(i * ((float)original_steps / steps));
                // 1. get previous step value
                int prev_timestep = i >= steps - 1 ? 0 :
                    TIMESTEPS - 1 - (TIMESTEPS / original_steps) *
                    (int)floor((i + 1) *
                               ((float)original_steps / steps));
                // Here timestep_s is tau_n' in Algorithm 4. The _s
                // notation appears to be that from C. Lu,
                // "DPM-Solver: A Fast ODE Solver for Diffusion
                // Probabilistic Model Sampling in Around 10 Steps",
                // arXiv:2206.00927 [cs.LG], but this notation is not
                // continued in Algorithm 4, where _n' is used.
                int timestep_s =
                    (int)floor((1 - eta) * prev_timestep);
                // Begin k-diffusion specific workaround for
                // evaluating F_theta(x; ...) from D(x, sigma), same
                // as in DDIM (and see there for detailed comments)
                float sigma = compvis_sigmas[timestep];
                if (i == 0) {
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1) /
                            sigma;
                    }
                }
                else {
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1);
                    }
                }
                struct ggml_tensor* model_output =
                    model(x, sigma, i + 1);
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_model_output[j] =
                            (vec_x[j] - vec_model_output[j]) *
                            (1 / sigma);
                    }
                }
                // 2. compute alphas, betas
                //
                // When comparing TCD with DDPM/DDIM note that Zheng
                // et al. (2024) follows the DPM-Solver notation for
                // alpha. One can find the following comment in the
                // original DPM-Solver code
                // (https://github.com/LuChengTHU/dpm-solver/):
                // "**Important**: Please pay special attention for
                // the args for `alphas_cumprod`: The `alphas_cumprod`
                // is the \hat{alpha_n} arrays in the notations of
                // DDPM. [...] Therefore, the notation \hat{alpha_n}
                // is different from the notation alpha_t in
                // DPM-Solver. In fact, we have alpha_{t_n} =
                // \sqrt{\hat{alpha_n}}, [...]"
                float alpha_prod_t = alphas_cumprod[timestep];
                float beta_prod_t = 1 - alpha_prod_t;
                // Note final_alpha_cumprod = alphas_cumprod[0] since
                // TCD is always "trailing"
                float alpha_prod_t_prev = prev_timestep >= 0 ?
                    alphas_cumprod[prev_timestep] : alphas_cumprod[0];
                // The subscript _s are the only portion in this
                // section (2) unique to TCD
                float alpha_prod_s = alphas_cumprod[timestep_s];
                float beta_prod_s = 1 - alpha_prod_s;
                // 3. Compute the predicted noised sample x_s based on
                // the model parameterization
                //
                // This section is also exactly the same as DDIM
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_pred_original_sample[j] =
                            (vec_x[j] / std::sqrt(sigma * sigma + 1) -
                             std::sqrt(beta_prod_t) *
                             vec_model_output[j]) *
                            (1 / std::sqrt(alpha_prod_t));
                    }
                }
                // This consistency function step can be difficult to
                // decipher from Algorithm 4, as it is simply stated
                // using a consistency function. This step is the
                // modified DDIM, i.e. p. 8 (32) in Zheng et
                // al. (2024), with eta set to 0 (see the paragraph
                // immediately thereafter that states this somewhat
                // obliquely).
                {
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        // Substituting x = pred_noised_sample and
                        // pred_epsilon = model_output
                        vec_x[j] =
                            std::sqrt(alpha_prod_s) *
                            vec_pred_original_sample[j] +
                            std::sqrt(beta_prod_s) *
                            vec_model_output[j];
                    }
                }
                // 4. Sample and inject noise z ~ N(0, I) for
                // MultiStep Inference Noise is not used on the final
                // timestep of the timestep schedule. This also means
                // that noise is not used for one-step sampling. Eta
                // (referred to as "gamma" in the paper) was
                // introduced to control the stochasticity in every
                // step. When eta = 0, it represents deterministic
                // sampling, whereas eta = 1 indicates full stochastic
                // sampling.
                if (eta > 0 && i != steps - 1) {
                    // In this case, x is still pred_noised_sample,
                    // continue in-place
                    ggml_tensor_set_f32_randn(noise, rng);
                    float* vec_x = (float*)x->data;
                    float* vec_noise = (float*)noise->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        // Corresponding to (35) in Zheng et
                        // al. (2024), substituting x =
                        // pred_noised_sample
                        vec_x[j] =
                            std::sqrt(alpha_prod_t_prev /
                                      alpha_prod_s) *
                            vec_x[j] +
                            std::sqrt(1 - alpha_prod_t_prev /
                                      alpha_prod_s) *
                            vec_noise[j];
                    }
                }
            }
        } break;

        default:
            LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
            abort();
    }
}

#endif  // __DENOISER_HPP__
