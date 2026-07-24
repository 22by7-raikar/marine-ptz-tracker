# Wiring and power plan

> Do not power the SG90 servos from the Arduino 5 V pin or USB port.

## Intended connections

| Device | Connection | Notes |
| --- | --- | --- |
| InnoMaker U20CAM-1080P | Ubuntu laptop USB | Verify the discovered `/dev/video*` device before use. |
| Arduino Uno R3 | Ubuntu laptop USB | Serial device is expected to resemble `/dev/ttyACM0`; do not hard-code discovery assumptions. |
| Pan SG90 | Arduino digital signal pin (TBD) | Signal only; select and document a non-conflicting pin during firmware work. |
| Tilt SG90 | Arduino digital signal pin (TBD) | Signal only. |
| Both SG90s | External regulated 5 V / 3 A supply | Supply positive to servo V+, negative to servo ground. |
| Arduino ground | External supply ground | A common ground is required for servo signal reference. |

## Before applying power

1. Confirm supply polarity and voltage with a meter.
2. Mechanically center the pan/tilt assembly before attaching horns.
3. Start with conservative pan/tilt limits; prevent mechanical end-stop contact.
4. Keep the laptop and Arduino USB connected only after the external supply wiring has been checked.
5. Test one servo at a time before enabling camera-driven commands.

Keep servo power wiring short and suitably sized. If the assembly behaves erratically, remove power and investigate supply capacity, ground continuity, and mechanical binding before changing software.
