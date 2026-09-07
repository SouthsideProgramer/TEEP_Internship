#include "hls_filter.h"

#include <string.h>

#if defined(HLS_FILTER_BACKEND_CMSIS)
#include "arm_math.h"
#elif defined(HLS_FILTER_BACKEND_ESPDSP)
#include "esp_dsp.h"
#endif

/* Scale int16 mic samples to [-1, 1) -- the convention librosa/soundfile use
 * for the 16-bit PCM dataset, so firmware and Python see the same numbers. */
#define HLS_INT16_SCALE (1.0f / 32768.0f)

void hls_biquad_init(hls_biquad_t *f, const float *sos, uint8_t n_sections)
{
    if (n_sections > HLS_MAX_SECTIONS) {
        n_sections = HLS_MAX_SECTIONS;
    }
    f->n_sections = n_sections;
    for (uint8_t s = 0; s < n_sections; ++s) {
        const float *row = &sos[s * 5];
        float *dst = &f->coeffs[s * 5];
        dst[0] = row[0];
        dst[1] = row[1];
        dst[2] = row[2];
#if defined(HLS_FILTER_BACKEND_CMSIS)
        /* CMSIS writes the recursion as y[n] = ... + a1*y[n-1] + a2*y[n-2],
         * scipy and ESP-DSP as ... - a1*y[n-1] - a2*y[n-2]. */
        dst[3] = -row[3];
        dst[4] = -row[4];
#else
        dst[3] = row[3];
        dst[4] = row[4];
#endif
    }
    hls_biquad_reset(f);
}

void hls_biquad_reset(hls_biquad_t *f)
{
    memset(f->state, 0, sizeof(f->state));
}

void hls_biquad_process(hls_biquad_t *f, const float *in, float *out, uint32_t n)
{
    if (n == 0u) {
        return;
    }
#if defined(HLS_FILTER_BACKEND_CMSIS)
    /* The instance struct is three fields pointing at storage we already own,
     * so building it here is cheaper than caching a CMSIS type in our header. */
    arm_biquad_cascade_df2T_instance_f32 s;
    s.numStages = f->n_sections;
    s.pState = f->state;
    s.pCoeffs = f->coeffs;
    arm_biquad_cascade_df2T_f32(&s, in, out, n);
#elif defined(HLS_FILTER_BACKEND_ESPDSP)
    const float *src = in;
    for (uint8_t sec = 0; sec < f->n_sections; ++sec) {
        dsps_biquad_f32(src, out, (int)n, &f->coeffs[sec * 5], &f->state[sec * 2]);
        src = out; /* cascade the remaining sections in place */
    }
#else
    /* Transposed direct form II, the same arrangement scipy.signal.sosfilt uses. */
    for (uint32_t i = 0; i < n; ++i) {
        float x = in[i];
        for (uint8_t sec = 0; sec < f->n_sections; ++sec) {
            const float *c = &f->coeffs[sec * 5];
            float *z = &f->state[sec * 2];
            const float y = c[0] * x + z[0];
            z[0] = c[1] * x - c[3] * y + z[1];
            z[1] = c[2] * x - c[4] * y;
            x = y;
        }
        out[i] = x;
    }
#endif
}

float hls_biquad_process_sample(hls_biquad_t *f, float x)
{
    float y;
    hls_biquad_process(f, &x, &y, 1u);
    return y;
}

void hls_decimator_init(hls_decimator_t *d)
{
    hls_biquad_init(&d->lowpass, HLS_DECIM_SOS, (uint8_t)HLS_DECIM_SECTIONS);
    d->phase = 0u;
}

void hls_decimator_reset(hls_decimator_t *d)
{
    hls_biquad_reset(&d->lowpass);
    d->phase = 0u;
}

uint32_t hls_decimator_process(hls_decimator_t *d, const float *in, uint32_t n_in, float *out)
{
    uint32_t written = 0u;

    /* Chunked so the vendor kernels get a block, and so a long input needs no
     * buffer proportional to its length. */
    for (uint32_t off = 0u; off < n_in; off += HLS_BLOCK_MAX) {
        uint32_t chunk = n_in - off;
        if (chunk > HLS_BLOCK_MAX) {
            chunk = HLS_BLOCK_MAX;
        }
        hls_biquad_process(&d->lowpass, &in[off], d->scratch, chunk);
        for (uint32_t i = 0u; i < chunk; ++i) {
            if (d->phase == 0u) {
                out[written++] = d->scratch[i];
            }
            d->phase = (d->phase + 1u) % HLS_DECIM_FACTOR;
        }
    }
    return written;
}

void hls_pipeline_init(hls_pipeline_t *p)
{
    hls_decimator_init(&p->decim);
    hls_biquad_init(&p->heart, HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS);
    hls_biquad_init(&p->lung, HLS_LUNG_SOS, (uint8_t)HLS_LUNG_SECTIONS);
}

void hls_pipeline_reset(hls_pipeline_t *p)
{
    hls_decimator_reset(&p->decim);
    hls_biquad_reset(&p->heart);
    hls_biquad_reset(&p->lung);
}

uint32_t hls_pipeline_process(hls_pipeline_t *p, const int16_t *mic, uint32_t n_mic,
                              float *heart_out, float *lung_out, float *base_out)
{
    uint32_t written = 0u;

    for (uint32_t off = 0u; off < n_mic; off += HLS_BLOCK_MAX) {
        uint32_t chunk = n_mic - off;
        if (chunk > HLS_BLOCK_MAX) {
            chunk = HLS_BLOCK_MAX;
        }
        for (uint32_t i = 0u; i < chunk; ++i) {
            p->mic_block[i] = (float)mic[off + i] * HLS_INT16_SCALE;
        }

        const uint32_t n_base = hls_decimator_process(&p->decim, p->mic_block, chunk, p->base_block);
        if (n_base == 0u) {
            continue;
        }
        if (heart_out != NULL) {
            hls_biquad_process(&p->heart, p->base_block, &heart_out[written], n_base);
        }
        if (lung_out != NULL) {
            hls_biquad_process(&p->lung, p->base_block, &lung_out[written], n_base);
        }
        if (base_out != NULL) {
            memcpy(&base_out[written], p->base_block, n_base * sizeof(float));
        }
        written += n_base;
    }
    return written;
}
