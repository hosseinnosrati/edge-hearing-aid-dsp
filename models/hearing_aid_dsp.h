/*
 * Edge Hearing Aid DSP - Embedded ANSI C99 Engine
 * High-performance, sample-by-sample real-time implementation
 * Zero dynamic memory allocation (no malloc/free).
 * Designed for ARM Cortex-M4, STM32, ESP32, and Texas Instruments DSPs.
 */

#ifndef HEARING_AID_DSP_H
#define HEARING_AID_DSP_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HA_FS                   16000
#define HA_AFC_TAPS             64
#define HA_NUM_BANDS            6

/* Biquad Direct-Form II Filter State */
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float d1, d2; /* delay states */
} ha_biquad_t;

/* Wide Dynamic Range Compression (WDRC) State */
typedef struct {
    float envelope;
    float alpha_attack;
    float alpha_release;
    float ct_db;       /* Compression Threshold (dB) */
    float cr;          /* Compression Ratio */
    float mpo_db;      /* Maximum Power Output (dB) */
} ha_wdrc_t;

/* Adaptive Acoustic Feedback Canceller (AFC) State */
typedef struct {
    float weights[HA_AFC_TAPS];
    float buffer[HA_AFC_TAPS];
    float mu;          /* Step size */
    float eps;         /* Regularization constant */
} ha_afc_t;

/* Initialize WDRC parameters */
static inline void ha_wdrc_init(ha_wdrc_t *comp, float ct_db, float cr, float mpo_db, float att_ms, float rel_ms, float fs) {
    comp->envelope = 1e-6f;
    comp->ct_db = ct_db;
    comp->cr = (cr < 1.0f) ? 1.0f : cr;
    comp->mpo_db = mpo_db;
    comp->alpha_attack = expf(-1.0f / (att_ms * 1e-3f * fs));
    comp->alpha_release = expf(-1.0f / (rel_ms * 1e-3f * fs));
}

/* Process single sample through WDRC compressor */
static inline float ha_wdrc_process_sample(ha_wdrc_t *comp, float sample) {
    float abs_s = fabsf(sample);
    if (abs_s > comp->envelope) {
        comp->envelope = comp->alpha_attack * comp->envelope + (1.0f - comp->alpha_attack) * abs_s;
    } else {
        comp->envelope = comp->alpha_release * comp->envelope + (1.0f - comp->alpha_release) * abs_s;
    }
    if (comp->envelope < 1e-6f) comp->envelope = 1e-6f;

    float env_db = 20.0f * log10f(comp->envelope);
    float gain_db = 0.0f;

    if (env_db > comp->ct_db) {
        gain_db = (1.0f / comp->cr - 1.0f) * (env_db - comp->ct_db);
    }
    if (env_db + gain_db > comp->mpo_db) {
        gain_db = comp->mpo_db - env_db;
    }

    float gain_lin = powf(10.0f, gain_db / 20.0f);
    float out = sample * gain_lin;
    if (out > 1.0f) out = 1.0f;
    if (out < -1.0f) out = -1.0f;
    return out;
}

/* Initialize AFC State */
static inline void ha_afc_init(ha_afc_t *afc, float step_size) {
    for (int i = 0; i < HA_AFC_TAPS; ++i) {
        afc->weights[i] = 0.0f;
        afc->buffer[i] = 0.0f;
    }
    afc->mu = step_size;
    afc->eps = 1e-4f;
}

/* Process single sample through NLMS Adaptive Feedback Cancellation */
static inline float ha_afc_process_sample(ha_afc_t *afc, float mic_sample, float speaker_sample) {
    /* Shift buffer */
    for (int i = HA_AFC_TAPS - 1; i > 0; --i) {
        afc->buffer[i] = afc->buffer[i - 1];
    }
    afc->buffer[0] = speaker_sample;

    /* Compute estimated feedback */
    float est_feedback = 0.0f;
    float norm = afc->eps;
    for (int i = 0; i < HA_AFC_TAPS; ++i) {
        est_feedback += afc->weights[i] * afc->buffer[i];
        norm += afc->buffer[i] * afc->buffer[i];
    }

    /* Error signal: clean speech */
    float error = mic_sample - est_feedback;

    /* NLMS weight update */
    float factor = afc->mu / norm * error;
    for (int i = 0; i < HA_AFC_TAPS; ++i) {
        afc->weights[i] += factor * afc->buffer[i];
    }

    return error;
}

/* Process single sample through Biquad direct form II */
static inline float ha_biquad_process(ha_biquad_t *f, float in) {
    float out = f->b0 * in + f->d1;
    f->d1 = f->b1 * in - f->a1 * out + f->d2;
    f->d2 = f->b2 * in - f->a2 * out;
    return out;
}

#ifdef __cplusplus
}
#endif

#endif /* HEARING_AID_DSP_H */
