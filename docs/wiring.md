# Wiring and power plan

> **Do not power two servos from the Arduino 5 V pin or USB port. Use the
> external regulated 5 V supply for servo power. Connect the external
> servo-supply ground to Arduino ground.**

## Intended connections

| Device | Connection | Notes |
| --- | --- | --- |
| InnoMaker U20CAM-1080P | Ubuntu laptop USB | Verify the discovered `/dev/video*` device before use. |
| Arduino Uno R3 | Ubuntu laptop USB | Record its stable `/dev/serial/by-id/...` identity after verification; do not hard-code or guess a `/dev/tty*` path. |
| Pan servo | Candidate Arduino digital pin 9 | Signal only; final pin and direction require bench confirmation. |
| Tilt servo | Candidate Arduino digital pin 10 | Signal only; final pin and direction require bench confirmation. |
| Both SG90s | External regulated 5 V / 3 A supply | Supply positive to servo V+, negative to servo ground. |
| Arduino ground | External supply ground | A common ground is required for servo signal reference. |

## Before applying power

1. Confirm supply polarity and voltage with a meter.
2. Disconnect external servo power while changing any wiring.
3. Mechanically center the pan/tilt assembly before attaching horns.
4. Start with conservative pan/tilt limits; prevent mechanical end-stop contact.
5. Keep the laptop and Arduino USB connected only after the external supply wiring has been checked.
6. Remove the camera from the pan/tilt assembly for the first powered motion test.
7. Test one servo at a time over a conservative motion range before attaching
   the second servo or enabling camera-driven commands.

USB supplies the Arduino and carries serial data; the external regulated supply
provides servo current. These power paths serve different roles even though
their grounds must be common. Verify the final supply current capacity against
the delivered servo specifications and measured stall behavior rather than
assuming the candidate 5 V / 3 A supply is sufficient.

`/dev/serial/by-id/<verified-uno>` is a documentation/configuration template,
not a device identity. Replace it with the exact discovered path before any
armed run.

Keep servo power wiring short and suitably sized. If the assembly behaves
erratically, remove power and investigate supply capacity, ground continuity,
polarity, voltage, and mechanical binding before changing software.
