/*
 * Host-side verification of lib/hls_filter, built with PlatformIO's `native`
 * platform (x86, no board involved). It replays the golden vectors written by
 * firmware/tools/gen_filter_coeffs.py and diffs the C output against
 * scipy.signal.sosfilt, so a wrong coefficient, a wrong sign convention or a
 * broken state update is caught here rather than on hardware.
 *
 * Run:  make firmware-test      (or: pio run -e native && .pio/build/native/program golden)
 * Exit: 0 all checks passed, 1 otherwise.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hls_filter.h"

/* Well above the float32 quantisation floor reported in golden/manifest.txt
 * (~6e-8 bandpass, ~7e-7 decimation) but far below anything a real coefficient
 * or state bug could hide under. */
#define TOL_BANDPASS 1.0e-5f
#define TOL_DECIM    1.0e-5f

static int failures = 0;

static float *read_f32(const char *dir, const char *name, size_t *n_out)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "cannot open %s -- run `make firmware-coeffs` first\n", path);
        exit(2);
    }
    fseek(fp, 0, SEEK_END);
    const long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    const size_t n = (size_t)bytes / sizeof(float);
    float *buf = (float *)malloc(n * sizeof(float));
    if (buf == NULL || fread(buf, sizeof(float), n, fp) != n) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    fclose(fp);
    *n_out = n;
    return buf;
}

/* max |a-b| and the reference RMS, so the error can be read as a relative figure too */
static void compare(const char *label, const float *got, const float *want, size_t n, float tol)
{
    double max_abs = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    size_t worst = 0;

    for (size_t i = 0; i < n; ++i) {
        const double e = fabs((double)got[i] - (double)want[i]);
        if (e > max_abs) {
            max_abs = e;
            worst = i;
        }
        sum_sq_err += e * e;
        sum_sq_ref += (double)want[i] * (double)want[i];
    }
    const double rms_ref = sqrt(sum_sq_ref / (double)n);
    const double rms_err = sqrt(sum_sq_err / (double)n);
    const int ok = max_abs <= (double)tol;

    printf("  %-28s %s  n=%-6zu max|err|=%.3e (i=%zu)  rms err=%.3e  ref rms=%.3e  err/ref=%.2e\n",
           label, ok ? "PASS" : "FAIL", n, max_abs, worst, rms_err, rms_ref,
           rms_ref > 0.0 ? rms_err / rms_ref : 0.0);
    if (!ok) {
        printf("      tolerance %.3e exceeded: got %.9g want %.9g\n",
               (double)tol, (double)got[worst], (double)want[worst]);
        failures++;
    }
}

/* Same input, three block sizes: the streaming state must make them identical. */
static void check_block_invariance(const float *in, size_t n, const float *sos, uint8_t sections)
{
    static const uint32_t sizes[] = {1u, 7u, 64u, 333u};
    float *ref = (float *)malloc(n * sizeof(float));
    float *got = (float *)malloc(n * sizeof(float));
    hls_biquad_t f;

    hls_biquad_init(&f, sos, sections);
    hls_biquad_process(&f, in, ref, (uint32_t)n);

    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        const uint32_t bs = sizes[k];
        hls_biquad_init(&f, sos, sections);
        for (size_t off = 0; off < n; off += bs) {
            uint32_t chunk = (uint32_t)(n - off);
            if (chunk > bs) {
                chunk = bs;
            }
            hls_biquad_process(&f, &in[off], &got[off], chunk);
        }
        char label[64];
        snprintf(label, sizeof(label), "block size %u == one call", bs);
        /* bit-identical is the expectation here, not just within tolerance */
        compare(label, got, ref, n, 0.0f);
    }
    free(ref);
    free(got);
}

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "golden";
    size_t n_in, n_heart, n_lung, n_dec_in, n_dec_out;

    float *in = read_f32(dir, "bandpass_input", &n_in);
    float *heart_ref = read_f32(dir, "bandpass_heart", &n_heart);
    float *lung_ref = read_f32(dir, "bandpass_lung", &n_lung);
    float *dec_in = read_f32(dir, "decim_input", &n_dec_in);
    float *dec_ref = read_f32(dir, "decim_output", &n_dec_out);

    if (n_in != n_heart || n_in != n_lung) {
        fprintf(stderr, "golden bandpass vectors disagree on length\n");
        return 2;
    }

    printf("hls_filter native check  (backend: "
#if defined(HLS_FILTER_BACKEND_CMSIS)
           "CMSIS-DSP"
#elif defined(HLS_FILTER_BACKEND_ESPDSP)
           "ESP-DSP"
#else
           "portable C"
#endif
           ", golden dir: %s)\n", dir);

    printf("\n[1] bandpass vs scipy.signal.sosfilt\n");
    float *out = (float *)malloc(n_in * sizeof(float));
    hls_biquad_t f;

    hls_biquad_init(&f, HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS);
    hls_biquad_process(&f, in, out, (uint32_t)n_in);
    compare("heart 20-200 Hz", out, heart_ref, n_in, TOL_BANDPASS);

    hls_biquad_init(&f, HLS_LUNG_SOS, (uint8_t)HLS_LUNG_SECTIONS);
    hls_biquad_process(&f, in, out, (uint32_t)n_in);
    compare("lung 150-1000 Hz", out, lung_ref, n_in, TOL_BANDPASS);

    printf("\n[2] decimation 16 kHz -> 4 kHz vs scipy\n");
    float *dec_out = (float *)malloc((n_dec_in / HLS_DECIM_FACTOR + 1u) * sizeof(float));
    hls_decimator_t d;
    hls_decimator_init(&d);
    const uint32_t produced = hls_decimator_process(&d, dec_in, (uint32_t)n_dec_in, dec_out);
    if (produced != n_dec_out) {
        printf("  %-28s FAIL  produced %u samples, golden has %zu\n",
               "output length", produced, n_dec_out);
        failures++;
    } else {
        compare("lowpass + drop 3 of 4", dec_out, dec_ref, n_dec_out, TOL_DECIM);
    }

    printf("\n[3] streaming state: block size must not change the output\n");
    check_block_invariance(in, n_in, HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS);

    printf("\n[4] full pipeline (int16 mic -> heart/lung) == decimator + bandpass\n");
    int16_t *mic = (int16_t *)malloc(n_dec_in * sizeof(int16_t));
    for (size_t i = 0; i < n_dec_in; ++i) {
        float v = dec_in[i] * 32768.0f;
        if (v > 32767.0f) { v = 32767.0f; }
        if (v < -32768.0f) { v = -32768.0f; }
        mic[i] = (int16_t)lrintf(v);
    }
    const size_t cap = n_dec_in / HLS_DECIM_FACTOR + 1u;
    float *pipe_heart = (float *)malloc(cap * sizeof(float));
    float *pipe_base = (float *)malloc(cap * sizeof(float));
    float *expect_heart = (float *)malloc(cap * sizeof(float));
    float *expect_base = (float *)malloc(cap * sizeof(float));
    float *mic_f = (float *)malloc(n_dec_in * sizeof(float));

    hls_pipeline_t p;
    hls_pipeline_init(&p);
    const uint32_t n_pipe = hls_pipeline_process(&p, mic, (uint32_t)n_dec_in,
                                                 pipe_heart, NULL, pipe_base);

    for (size_t i = 0; i < n_dec_in; ++i) {
        mic_f[i] = (float)mic[i] / 32768.0f;
    }
    hls_decimator_init(&d);
    const uint32_t n_expect = hls_decimator_process(&d, mic_f, (uint32_t)n_dec_in, expect_base);
    hls_biquad_init(&f, HLS_HEART_SOS, (uint8_t)HLS_HEART_SECTIONS);
    hls_biquad_process(&f, expect_base, expect_heart, n_expect);

    if (n_pipe != n_expect) {
        printf("  %-28s FAIL  pipeline produced %u, expected %u\n", "output length", n_pipe, n_expect);
        failures++;
    } else {
        compare("decimated signal", pipe_base, expect_base, n_pipe, 0.0f);
        compare("heart band", pipe_heart, expect_heart, n_pipe, 0.0f);
    }

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
