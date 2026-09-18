---
type: ProjectLayout
title: STM32-Embedded Gesture Classifier
date: '2025-06-17'
client: ''
description: >-
  Lightweight CNN to classify various hand motions through an I2C-interfaced
  ADXL345 accelerometer, deployed onto an STM32 Nucleo using TensorFlow Lite
  Micro.
featuredImage:
  type: ImageBlock
  url: /images/IMG_5640.jpg
  altText: Project thumbnail image
  caption: ''
  elementId: ''
media:
  type: ImageBlock
  url: /images/IMG_5640.jpg
  altText: Project image
  caption: ''
  elementId: ''
addTitleSuffix: true
colors: colors-a
backgroundImage:
  type: BackgroundImage
  url: /images/bg2.jpg
  backgroundSize: cover
  backgroundPosition: center
  backgroundRepeat: no-repeat
  opacity: 100
---
This project is a TinyML pipeline that recognizes hand gestures from raw accelerometer motion and runs the classifier entirely on a microcontroller — no host, no network, no floating-point luxury. The target is an STM32F411RE Nucleo reading an ADXL345 three-axis accelerometer over I2C, with the goal of sub-50 ms inference for responsive human-robot interaction.

The interesting constraint here is not the model. A gesture classifier is a small 1D CNN. The interesting constraint is that every design choice — window size, filter order, quantization scheme — is bounded by what fits in 128 KB of SRAM and still answers in time.

### Part 1: Sensor and collection

The ADXL345 is configured over I2C into measurement mode at the ±2g range, which gives the resolution that matters for hand motion rather than for impact:

```
#define ADXL345_ADDR    (0x53 << 1)   // 7-bit address left-shifted for HAL
#define REG_POWER_CTL   0x2D
#define REG_DATA_FORMAT 0x31
#define REG_DATAX0      0x32

// set measurement mode
data = 0x08;
HAL_I2C_Mem_Write(&hi2c1, ADXL345_ADDR, REG_POWER_CTL, 1, &data, 1, HAL_MAX_DELAY);

// set data format (+/-2g)
data = 0x00;
HAL_I2C_Mem_Write(&hi2c1, ADXL345_ADDR, REG_DATA_FORMAT, 1, &data, 1, HAL_MAX_DELAY);
```

Labeled training windows are streamed off the board over UART and captured on the host with a `pyserial` collection script, which prompts for a gesture label, clears the serial buffer, counts the user in, and writes one labeled window per repetition. Collecting your own dataset is unglamorous and turns out to be most of the work.

### Part 2: Preprocessing

Raw accelerometer data carries two things the model should never have to learn around: a gravity component that depends on how the board is being held, and high-frequency noise that has nothing to do with intentional motion. Both are removed before training, with the identical transform intended for the device:

```
# design FIR low-pass filter
nyquist = 0.5 * sample_rate
cutoff_norm = filter_cutoff / nyquist
numtaps = 31
taps = firwin(numtaps, cutoff_norm)

for i in range(num_samples):
    w = data[i].reshape(window_size, 3)
    w = w - w.mean(axis=0)                            # DC removal
    for axis in range(3):
        w[:, axis] = filtfilt(taps, [1.0], w[:, axis])  # filtering
    max_abs = np.max(np.abs(w))
    if max_abs > 0:
        w = w / max_abs                               # per-window normalize
    processed[i] = w.flatten()
```

A 31-tap FIR low-pass at a 20 Hz cutoff (100 Hz sampling) keeps deliberate gesture dynamics and discards tremor. Per-window normalization makes the classifier care about the *shape* of a motion rather than how forcefully it was performed.

### Part 3: The model

The classifier is a small 1D CNN over the `(window_size, 3)` time series — two convolution/pooling stages for feature extraction, dropout, and a dense head:

```
def build_model(input_shape, num_classes):
    m = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    return m
```

Convolution over time is the right inductive bias for this problem: a gesture is a local pattern in the signal that should be recognized wherever in the window it happens to land.

### Part 4: Quantization for the target

Keras floats do not fit the deployment budget. The trained model is converted with full integer quantization, using a representative dataset drawn from real collected windows so that the converter can calibrate activation ranges rather than guess them:

```
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: repr_data_gen(repr_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
```

The resulting `.tflite` flatbuffer is embedded into the firmware as a C array and executed by a TensorFlow Lite Micro interpreter over a statically allocated tensor arena — a deliberate choice, since a bare-metal target has no business calling into a heap during real-time inference.

### Current status

The data collection, preprocessing, training, and quantization pipeline is complete and reproducible end to end. On-device integration — wiring the TFLM interpreter and tensor arena into the STM32 firmware and measuring true wall-clock inference latency against the sub-50 ms target — is the work in progress.

### For code:

<https://github.com/siddhshah/GestureClassification>
