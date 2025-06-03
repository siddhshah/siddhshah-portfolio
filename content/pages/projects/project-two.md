---
type: ProjectLayout
title: FPGA-Based Ultrasonic Radar
colors: colors-a
date: '2025-05-08'
client: ''
description: >-
  A real-time object mapping system using a Xilinx Spartan-7 FPGA and an
  ultrasonic sensor for better object navigation in autonomous robots.
featuredImage:
  type: ImageBlock
  url: /images/IMG_5278.jpg
  altText: Black-and-green radar
media:
  type: ImageBlock
  url: /images/IMG_5280.jpg
  altText: Project image
bottomSections:
  - type: TextSection
    title: Introduction
    subtitle: ''
    text: >
      This is a SystemVerilog-based 1D ultrasonic radar using the HC-SR04
      ultrasonic sensor and a Xilinx Spartan-7 FPGA that outputs real-time
      distance data for the closest object to the sensor.


      Distance is displayed through a live centimeter-based distance tracker,
      and a red on-screen 'X' will move along the screen based on the object's
      live position changes. Positional benchmarks are included for the user.
      Distance range is 2-400 cm.


      Moving within 15 centimeters of the sensor will play an aggressive audio
      alert on the connected audio device and rapidly flash the screen in red
      and black.
    colors: colors-f
    variant: variant-a
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: center
  - type: TextSection
    title: Accelerating Object Detection With Hardware
    text: >
      The goal of this project was to put a unique spin on standard ultrasonic
      sensor-based detection and examine the benefits of using an FPGA over a
      standard C/C++-based microcontroller e.g. Arduino, all while testing the
      hardware limitations of an FPGA for real-time data processing.


      Split into three main parts, this project exercises proficiency in
      breadboard circuit debugging, optimization of large-scale SystemVerilog
      projects in Xilinx Vivado, and ability to modify design considerations
      on-the-go.
    colors: colors-f
    variant: variant-a
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: left
  - type: TextSection
    title: 'Part 1: Sensor Circuit'
    subtitle: ''
    text: >
      The radar mapping system begins with a simple breadboard circuit
      containing the ultrasonic sensor, voltage dividers, and a debugging LED,
      where GND, TRIG, and ECHO are fed into the FPGA's GPIO pins.
    colors: colors-e
    variant: variant-b
    elementId: ''
    styles:
      self:
        height: auto
        width: wide
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: left
  - type: MediaGallerySection
    title: ''
    subtitle: ''
    images:
      - type: ImageBlock
        url: /images/IMG_5279.jpg
        altText: Image one
        caption: Image one caption
        elementId: ''
      - type: ImageBlock
        url: /images/IMG_5158.jpg
        altText: Image two
        caption: Image two caption
        elementId: ''
      - type: ImageBlock
        url: /images/Screenshot 2025-05-12 184126.png
        altText: Image three
        caption: Image three caption
        elementId: ''
      - type: ImageBlock
        url: /images/Screenshot 2025-05-11 160018.png
        altText: Image four
        caption: Image four caption
        elementId: ''
    colors: colors-e
    spacing: 16
    columns: 2
    aspectRatio: '1:1'
    showCaption: false
    enableHover: false
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-12
          - pb-12
          - pl-4
          - pr-4
        textAlign: center
  - type: TextSection
    title: ''
    text: >
      The HC-SR04 Ultrasonic Sensor has 4 pins: TRIG, ECHO, Vcc, GND. All pins
      (except GND) operate at a 5 volt input. I did not have a 5 volt power
      supply on hand (my RealDigital Urbana Board only outputs 3.3 V), so I
      created a voltage divider circuit to step down a 9 V load to 5 V. The ECHO
      also outputted 5 V, which had to be stepped down to 3.3 V (as the FPGA
      only expects 3.3 V GPIO inputs).


      The TRIG pin is the input: a short pulse is sent to the sensor (generated
      from a finite state machine), signaling it to record a measurement. The
      sensor does this by outputting an 8-cycle sonic burst.


      Upon reception of the reflected burst, the sensor outputs an ECHO signal
      that remains high for a duration proportional to the time it took for the
      8-cycle burst to come back to the sensor after it was sent. A longer time
      between sending and receiving indicates the burst traveled a longer
      distance before it was reflected back by the nearest object, or traveled
      for less time if the burst came back quicker. Therefore, the longer the
      ECHO, the farther the object.


      A blue LED was wired to the ECHO pin for simple inspection based
      debugging--LED pulse length should be proportional to object distance.
    colors: colors-f
    variant: variant-a
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: left
  - type: TextSection
    title: 'Part 2: Data Processing and DSP Filtering'
    subtitle: Xilinx Vivado
    text: >
      The next step is to receive the ECHO signal from the electronic circuit
      and process it into a usable distance value. This data processing was done
      in Xilinx Vivado using SystemVerilog. Due to the I/O structure of the
      breadboard-FPGA system, all debugging had to be done on hardware and not
      through behavioral simulations.
    colors: colors-c
    variant: variant-b
    elementId: ''
    styles:
      self:
        height: auto
        width: wide
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: left
  - type: MediaGallerySection
    title: ''
    subtitle: ''
    images:
      - type: ImageBlock
        url: /images/Screenshot 2025-06-02 200927.png
        altText: Image one
        caption: Image one caption
        elementId: ''
      - type: ImageBlock
        url: /images/Screenshot 2025-05-11 160732.png
        altText: Image two
        caption: Image two caption
        elementId: ''
      - type: ImageBlock
        url: /images/Screenshot 2025-05-11 161030.png
        altText: altText of the image
        caption: Caption of the image
        elementId: ''
      - type: ImageBlock
        url: /images/Screenshot 2025-06-02 201240.png
        altText: altText of the image
        caption: Caption of the image
        elementId: ''
    colors: colors-c
    spacing: 16
    columns: 2
    aspectRatio: '1:1'
    showCaption: false
    enableHover: false
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-12
          - pb-12
          - pl-4
          - pr-4
        textAlign: center
  - type: TextSection
    title: ''
    text: >+
      When the FPGA receives the ECHO signal, it is converted into a tick-based
      raw distance (22 bit-wide register), processed through clock-based (100
      MHz) counter logic and the aforementioned FSM. This data is then sent
      through a variety of DSP filters:


      *   A median-of-3 filter, which smooths data by taking the median of the
      first three incoming distance samples through use of a shift register,

          *   A median filter with a greater window e.g. median-of-5 or median-of-7 will provide greater noise removal at the cost of timing.

      *   a clamping filter, which removes arbitrary spikes caused by dips in
      between sensor measurements,


      *   and a deadband filter, which silences 1-2 pixel jitters in the
      on-screen's object position changes. These filters lead to a greatly
      effective stabilization of measurement data and smooth out a vast majority
      of any noise produced by the sensor.


      The final result was an impressively-stable distance measurement per
      measurement cycle, stored as clock-based (100 MHz) ticks.

    colors: colors-a
    variant: variant-a
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: left
  - type: TextSection
    title: 'Part 3: Data Visualization'
    subtitle: HDMI-to-VGA Display Monitor
    text: >
      To show off the results, I created object and pixel mapping modules to
      scale the distance down to centimeters, and use Vivado/RealDigital IPs to
      configure the data to be sent through an HDMI-to-VGA display adapter.
    colors: colors-d
    variant: variant-b
    elementId: ''
    styles:
      self:
        height: auto
        width: wide
        padding:
          - pt-28
          - pb-28
          - pl-4
          - pr-4
        textAlign: left
  - type: MediaGallerySection
    title: ''
    subtitle: ''
    images:
      - type: ImageBlock
        url: /images/gallery-1.jpg
        altText: Image one
        caption: Image one caption
        elementId: ''
      - type: ImageBlock
        url: /images/gallery-2.jpg
        altText: Image two
        caption: Image two caption
        elementId: ''
      - type: ImageBlock
        url: /images/gallery-3.jpg
        altText: Image three
        caption: Image three caption
        elementId: ''
      - type: ImageBlock
        url: /images/gallery-4.jpg
        altText: Image four
        caption: Image four caption
        elementId: ''
    colors: colors-d
    spacing: 16
    columns: 2
    aspectRatio: '1:1'
    showCaption: false
    enableHover: false
    elementId: ''
    styles:
      self:
        height: auto
        width: narrow
        padding:
          - pt-12
          - pb-12
          - pl-4
          - pr-4
        textAlign: center
---
