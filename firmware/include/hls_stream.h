/*
 * Serial framing shared by both board firmwares.
 *
 * The band-split output is 4000 Hz x 2 channels x float32 = 32 kB/s, which no
 * text format survives at any sane baud rate, so frames go out as binary and
 * firmware/tools/capture_serial.py decodes them on the host. Both MCUs are
 * little-endian, so the structs go on the wire as-is.
 *
 * Frame: hls_frame_header_t (starting with HLS_STREAM_MAGIC), then `n_samples`
 * heart floats followed by `n_samples` lung floats. `dropped` is cumulative and
 * non-zero means the capture is no longer gap-free -- do not paper over it.
 */
#ifndef HLS_STREAM_H
#define HLS_STREAM_H

#include <stdint.h>

#if defined(__cplusplus)
#define HLS_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define HLS_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#define HLS_STATIC_ASSERT(cond, msg)
#endif

#define HLS_STREAM_MAGIC "HLS1"

typedef struct __attribute__((packed)) {
    char magic[4];
    uint16_t n_samples;   /* per channel, at HLS_BASELINE_SR_HZ */
    uint16_t seq;         /* wraps at 65536, for gap detection */
    uint32_t dropped;     /* cumulative mic samples lost to overrun */
} hls_frame_header_t;

/*
 * Dataset-replay records, used by the bench firmware (src/main_bench.cpp).
 *
 * A run is: one clip header, that clip's data frames, one clip footer -- per
 * clip -- then a single done record. Everything up to the done record is
 * binary; the human-readable summary is printed after it, so a host script can
 * stop parsing at a known point and a person watching the serial monitor still
 * gets something legible at the end.
 *
 * `compute_micros` is the filtering time alone: the int16 scaling and both
 * band cascades, with serial I/O excluded. That is the number worth putting in
 * a latency table.
 */
#define HLS_CLIP_MAGIC  "HLSC"
#define HLS_FOOTER_MAGIC "HLSE"
#define HLS_DONE_MAGIC  "HLSD"

typedef struct __attribute__((packed)) {
    char magic[4];
    char clip_id[12];     /* e.g. "M0001", NUL-padded */
    uint32_t n_samples;
    uint32_t sample_rate;
} hls_clip_header_t;

typedef struct __attribute__((packed)) {
    char magic[4];
    uint32_t n_samples;      /* echoed back, so a truncated run is detectable */
    uint32_t compute_micros; /* filtering only -- no serial I/O */
} hls_clip_footer_t;

typedef struct __attribute__((packed)) {
    char magic[4];
    uint32_t clip_count;
} hls_done_t;

/*
 * The host parsers in tools/ unpack these with struct formats that must match
 * byte for byte. A silent layout change would only show up as garbage on the
 * wire the first time a board is plugged in, so pin the sizes here; the
 * matching assertion on the Python side lives in tools/test_protocol.py.
 */
HLS_STATIC_ASSERT(sizeof(hls_frame_header_t) == 12, "frame header layout changed -- update tools/");
HLS_STATIC_ASSERT(sizeof(hls_clip_header_t) == 24, "clip header layout changed -- update tools/");
HLS_STATIC_ASSERT(sizeof(hls_clip_footer_t) == 12, "clip footer layout changed -- update tools/");
HLS_STATIC_ASSERT(sizeof(hls_done_t) == 8, "done record layout changed -- update tools/");

#endif /* HLS_STREAM_H */
