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
    title: Gallery
    subtitle: This is the subtitle
    images:
      - type: ImageBlock
        url: /images/IMG_5279.jpg
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
---
