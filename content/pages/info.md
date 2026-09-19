---
type: PageLayout
title: About
colors: colors-a
backgroundImage:
  type: BackgroundImage
  url: /images/bg4.jpg
  backgroundSize: cover
  backgroundPosition: center
  backgroundRepeat: no-repeat
  opacity: 75
sections:
  - elementId: ''
    colors: colors-f
    backgroundSize: full
    text: >+
      # Hey, I’m Siddh. I’m a computer engineering student at the University of
      Illinois Urbana-Champaign, graduating in December 2026. I like the layer
      where software stops being an abstraction and starts being a circuit —
      CUDA kernels, out-of-order RTL, and bare-metal firmware. Right now I’m
      writing test-platform firmware for satellite payload modules at Orion
      Space Solutions and studying lossless attention compression with
      OpenMachine.ai.

    styles:
      self:
        height: auto
        width: wide
        margin:
          - mt-0
          - mb-0
          - ml-0
          - mr-0
        padding:
          - pt-16
          - pb-12
          - pl-4
          - pr-4
        textAlign: left
    type: HeroSection
  - type: DividerSection
    styles:
      self:
        width: wide
        padding:
          - pt-8
          - pb-8
          - pl-4
          - pr-4
        borderWidth: 1
        borderStyle: solid
  - type: LabelsSection
    colors: colors-f
    subtitle: 'Languages:'
    items:
      - type: Label
        label: Python
      - type: Label
        label: C/C++
      - type: Label
        label: CUDA C++
      - type: Label
        label: SystemVerilog
      - type: Label
        label: Java
      - type: Label
        label: HTML/CSS
  - type: LabelsSection
    colors: colors-f
    subtitle: 'Software and frameworks:'
    items:
      - type: Label
        label: PyTorch
      - type: Label
        label: TensorFlow
      - type: Label
        label: JAX
      - type: Label
        label: Nsight Compute
      - type: Label
        label: Nsight Systems
      - type: Label
        label: Xilinx Vivado
      - type: Label
        label: Synopsys VCS/Verdi
      - type: Label
        label: Qiskit
  - type: LabelsSection
    colors: colors-f
    subtitle: 'Hardware and embedded:'
    items:
      - type: Label
        label: STM32
      - type: Label
        label: RP2040
      - type: Label
        label: I2C
      - type: Label
        label: SPI
      - type: Label
        label: UART
      - type: Label
        label: AXI4
      - type: Label
        label: KiCAD
  - type: DividerSection
    styles:
      self:
        width: wide
        padding:
          - pt-12
          - pb-12
          - pl-4
          - pr-4
        borderWidth: 1
        borderStyle: solid
  - type: FeaturedItemsSection
    colors: colors-f
    items:
      - type: FeaturedItem
        subtitle: 'Experience:'
        text: |-
          **2026 — Present**

          * electrical engineering intern @ Orion Space Solutions, Louisville CO

          * research collaborator @ OpenMachine.ai

          **2024 — 2025**

          * undergraduate researcher @ Human-Centered Autonomy Lab, UIUC

          **2024 — Present**

          * membership vice president @ IEEE UIUC
        styles:
          self:
            textAlign: left
      - type: FeaturedItem
        subtitle: 'Education:'
        text: |-
          **2023 — 2026**

          * bs computer engineering, minor in physics

          * university of illinois urbana-champaign
        styles:
          self:
            textAlign: left
    columns: 2
    spacingX: 60
    spacingY: 60
    styles:
      self:
        height: auto
        width: wide
        padding:
          - pt-8
          - pb-8
          - pl-4
          - pr-4
        textAlign: left
  - type: DividerSection
    styles:
      self:
        width: wide
        padding:
          - pt-12
          - pb-12
          - pl-4
          - pr-4
        borderWidth: 1
        borderStyle: solid
  - type: FeaturedItemsSection
    subtitle: 'You can find me here:'
    colors: colors-f
    items:
      - type: FeaturedItem
        actions:
          - type: Link
            label: GitHub
            url: 'https://github.com/siddhshah'
        styles:
          self:
            textAlign: left
      - type: FeaturedItem
        actions:
          - type: Link
            label: LinkedIn
            url: 'https://www.linkedin.com/in/siddhshah05'
        styles:
          self:
            textAlign: left
      - type: FeaturedItem
        actions:
          - type: Link
            label: Email
            url: 'mailto:siddh.shah90@gmail.com'
        styles:
          self:
            textAlign: left
    columns: 3
    spacingX: 120
    spacingY: 16
    styles:
      self:
        height: auto
        width: wide
        padding:
          - pt-8
          - pb-8
          - pl-4
          - pr-4
  - type: DividerSection
    styles:
      self:
        width: wide
        padding:
          - pt-8
          - pb-8
          - pl-4
          - pr-4
        borderWidth: 1
        borderStyle: solid
  - type: TextSection
    variant: variant-a
    subtitle: 'Contact:'
    colors: colors-f
    text: |
      [siddh.shah90@gmail.com](mailto:siddh.shah90@gmail.com) · [siddh2@illinois.edu](mailto:siddh2@illinois.edu)
  - type: DividerSection
    styles:
      self:
        width: wide
        padding:
          - pt-8
          - pb-8
          - pl-4
          - pr-4
        borderWidth: 1
        borderStyle: solid
  - type: ContactSection
    backgroundSize: full
    title: "Let’s talk... \U0001F4AC"
    colors: colors-f
    form:
      type: FormBlock
      elementId: contact
      fields:
        - name: firstName
          label: First Name
          hideLabel: true
          placeholder: First Name
          isRequired: true
          width: 1/2
          type: TextFormControl
        - name: lastName
          label: Last Name
          hideLabel: true
          placeholder: Last Name
          isRequired: false
          width: 1/2
          type: TextFormControl
        - name: email
          label: Email
          hideLabel: true
          placeholder: Email
          isRequired: true
          width: full
          type: EmailFormControl
        - name: message
          label: Message
          hideLabel: true
          placeholder: What would you like to talk about?
          isRequired: true
          width: full
          type: TextareaFormControl
      submitLabel: "Submit \U0001F680"
      styles:
        self:
          textAlign: center
    styles:
      self:
        height: auto
        width: narrow
        margin:
          - mt-0
          - mb-0
          - ml-4
          - mr-4
        padding:
          - pt-12
          - pb-12
          - pr-4
          - pl-4
        flexDirection: row
        textAlign: left
---
