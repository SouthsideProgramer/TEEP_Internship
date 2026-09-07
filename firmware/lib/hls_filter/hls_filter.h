/*
 * hls_filter -- causal streaming heart/lung band split for the MCU targets.
 *
 * This is the firmware counterpart of src/baseline/baseline1.py (Baseline 1,
 * fixed Butterworth bandpass). One deliberate difference from the Python
 * baseline: Python uses scipy.signal.sosfiltfilt (zero-phase, runs the filter
 * forwards then backwards, needs the whole 15 s clip in RAM). The firmware
 * runs scipy.signal.sosfilt's equivalent -- causal, single pass, sample- or
 * block-at-a-time, O(1) memory. That costs the filter's phase response: the
 * firmware output is phase-distorted relative to the Python baseline, and its
 * group delay is non-zero. See firmware/README.md for what that means for the
 * numbers in report/.
 *
 * Coefficients are designed off-line by firmware/tools/gen_filter_coeffs.py
 * and hardcoded in hls_filter_coeffs.h -- nothing is designed on the MCU.
 *
 * Three interchangeable backends, selected at compile time:
 *   HLS_FILTER_BACKEND_CMSIS   -- CMSIS-DSP arm_biquad_cascade_df2T_f32 (Nano 33 BLE Sense, Cortex-M4F)
 *   HLS_FILTER_BACKEND_ESPDSP  -- ESP-DSP dsps_biquad_f32 (ESP32-S3, Xtensa LX7)
 *   (neither defined)          -- portable C DF2T, used by the native host test
 * All three consume the same hls_filter_coeffs.h and must agree sample for
 * sample; the native test pins the portable path to scipy, and the board
 * builds swap in the vendor kernels behind the identical API.
 */
#ifndef HLS_FILTER_H
#define HLS_FILTER_H

#include <stdint.h>

#include "hls_filter_coeffs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Every cascade in hls_filter_coeffs.h is 4 sections; 8 leaves room to raise
 * the filter order without touching this header. */
#define HLS_MAX_SECTIONS 8u

/* Mic samples are decimated in chunks of this size so the vendor kernels are
 * called with a real block rather than one sample at a time. */
#define HLS_BLOCK_MAX 64u

/*
 * One biquad cascade. `coeffs` holds the backend's native layout, built from
 * the scipy {b0,b1,b2,a1,a2} rows at init (CMSIS wants a1/a2 negated), so the
 * generated header stays backend-independent.
 */
typedef struct {
    uint8_t n_sections;
    float coeffs[HLS_MAX_SECTIONS * 5];
    float state[HLS_MAX_SECTIONS * 2];
} hls_biquad_t;

/* `sos`: n_sections rows of {b0, b1, b2, a1, a2}, scipy sign convention. */
void hls_biquad_init(hls_biquad_t *f, const float *sos, uint8_t n_sections);
void hls_biquad_reset(hls_biquad_t *f);
/* in-place safe (out may alias in) */
void hls_biquad_process(hls_biquad_t *f, const float *in, float *out, uint32_t n);
float hls_biquad_process_sample(hls_biquad_t *f, float x);

/*
 * Mic rate -> baseline rate. Both boards capture at HLS_MIC_SR_HZ (16 kHz);
 * every Python number in this project was computed at HLS_BASELINE_SR_HZ
 * (4 kHz, the dataset's rate), so the firmware anti-alias filters and drops
 * HLS_DECIM_FACTOR-1 of every HLS_DECIM_FACTOR samples before the band split.
 * `phase` carries the decimation position across calls, so an input block
 * length that is not a multiple of the factor is handled correctly.
 */
typedef struct {
    hls_biquad_t lowpass;
    uint32_t phase;
    float scratch[HLS_BLOCK_MAX];
} hls_decimator_t;

void hls_decimator_init(hls_decimator_t *d);
void hls_decimator_reset(hls_decimator_t *d);
/* Writes at most n_in / HLS_DECIM_FACTOR + 1 samples to `out`; returns how many. */
uint32_t hls_decimator_process(hls_decimator_t *d, const float *in, uint32_t n_in, float *out);

/*
 * Full acquisition pipeline: int16 mic samples in, heart and lung estimates
 * out at the baseline rate.
 */
#define HLS_BLOCK_MAX_DECIMATED (HLS_BLOCK_MAX / HLS_DECIM_FACTOR + 1u)

typedef struct {
    hls_decimator_t decim;
    hls_biquad_t heart;
    hls_biquad_t lung;
    float mic_block[HLS_BLOCK_MAX];
    float base_block[HLS_BLOCK_MAX_DECIMATED];
} hls_pipeline_t;

void hls_pipeline_init(hls_pipeline_t *p);
void hls_pipeline_reset(hls_pipeline_t *p);

/*
 * `mic`      : n_mic int16 samples at HLS_MIC_SR_HZ, scaled by 1/32768 to match
 *              the [-1,1) convention librosa/soundfile hand the Python baseline.
 * `heart_out`,
 * `lung_out` : each must hold n_mic / HLS_DECIM_FACTOR + 1 floats. Either may
 *              be NULL to skip that band.
 * `base_out` : optional, receives the decimated (pre-band-split) signal; NULL to skip.
 * Returns the number of baseline-rate samples written.
 */
uint32_t hls_pipeline_process(hls_pipeline_t *p, const int16_t *mic, uint32_t n_mic,
                              float *heart_out, float *lung_out, float *base_out);

#ifdef __cplusplus
}
#endif

#endif /* HLS_FILTER_H */
