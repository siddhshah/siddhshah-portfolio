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




      ![](/images/IMG_5279.jpg)![](/images/IMG_5158.jpg)![](/images/Screenshot%202025-05-12%20184126.png)![](/images/Screenshot%202025-05-11%20160018.png)Schematics
      were generated using Fritzing.
    colors: colors-e
    variant: variant-a
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
    variant: variant-a
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
        url: /images/IMG_5278.jpg
        altText: Image one
        caption: Image one caption
        elementId: ''
      - type: ImageBlock
        url: /images/Screenshot 2025-06-02 202038.png
        altText: Image two
        caption: Image two caption
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
  - type: TextSection
    title: ''
    text: >
      The distance is scaled to fit on a 640-pixel long VGA display, and an
      object mapper module places a red X on the screen based on the current
      pixel distance, converted from the filtered distance register. This object
      mapper module also assigns the left-hand side of the screen to 2 cm and
      the right-hand side to 400 cm, accurately representing the physical
      constraints of the sensor. Visual benchmarks in the style of a standard
      radar or measuring devices are generated for user benefit.


      A live counter displays the distance, in centimeters, of the object from
      the ultrasonic sensor. A font read-only-memory (ROM) is used to draw the
      dynamically-changing 16x8 pixel ASCII numbers based on a binary
      (background/foreground) coloring scheme.


      A proximity alert module drives an alert signal high whenever the object
      reaches within 15 centimeters of the sensor. This parameter, calculated
      via (THRESHOLD\_DISTANCE\_CM \* 2 \* 100 / 0.0343 \[decimal] -->
      THRESHOLD\_DISTANCE \[hex]), can be changed in the proximity alert module
      if a different threshold is desired. A tone generator then creates an
      oscillating square wave at a given frequency, wired to the auxiliary audio
      outputs of the FPGA (when the alert signal is high) to produce a sustained
      beep when the object is in proximity. 


      This alert signal is also used in a color mapper module, which creates the
      color scheme for the on-screen visuals, to flash the screen in red and
      black every 0.5 seconds. An RGB LED is wired to the alert signal for
      debugging purposes.
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
    title: More Design Considerations
    text: >+
      The initial design for this radar was 2-dimensional, accomplished through
      situating the HC-SR04 ultrasonic radar on a rotating HS-013 servo motor,
      where the VGA display would show the N closest objects to the sensor-motor
      system, constantly rotating through its active range and scanning for new
      objects, all with the same functionality (e.g. proximity alerts) as shown
      above. However, voltage load issues (due to the lack of a voltage
      regulator on-hand) and time constraints, the system was scaled down to a
      1-dimensional version, which could only capture a single object's
      position.


      Furthermore, data visualization was considered through Python, where a
      simple MicroBlaze SoC-based UART module would send the distance data to
      the serial monitor in Vitis HLS and captured by a Python script. Although
      filter design would have been more effective in a scripting language such
      as Python with its extensive SciPy signal libraries, the timing delay
      caused by data transmission would have rendered the "efficiency" portion
      of the radar system nearly useless.


      ### For code and design specifics/diagrams:


      <https://github.com/siddhshah/FPGA-UltrasonicRadar>

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
---
