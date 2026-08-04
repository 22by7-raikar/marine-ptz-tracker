# Bill of materials and scope

| Item | Status | Purpose / notes |
| --- | --- | --- |
| Ubuntu laptop with NVIDIA GPU | Available | Capture, detection, control, telemetry, and serial host. |
| InnoMaker U20CAM-1080P USB camera | Delivered / observed | 1920×1080 MJPEG at 30 FPS was observed through the configured machine-specific by-id path. |
| Arduino Uno R3-compatible controller | Delivered / observed | Firmware upload, USB serial handshake, and bounded integrated commands were observed; recheck identity after USB changes. |
| XiaoR Geek two-servo pan/tilt assembly | Delivered / observed | Loaded controlled motion was observed within 75–105 degrees; exact servo current, long-duration wear, and broader travel remain unqualified. |
| Regulated 5 V, 3 A external supply | Delivered / used | Dedicated servo power with common ground is mandatory; polarity, voltage, and capacity still require operator checks before each powered setup. |
| Green Toys tugboat | Available | Initial physical marine target. |
| USB cable for Arduino | Verify | Data/power link from laptop. |
| Jumper wires / suitable servo extension | Verify | Servo signals, common ground, and power distribution. |
| Multimeter | Recommended | Required for polarity and supply-voltage checks. |

## Cost and scope separation

The following planning figures were supplied for this project. They exclude
tax, shipping, and any price not listed here.

| Category | Item | Planning price | Scope |
| --- | --- | ---: | --- |
| PTZ rig hardware | InnoMaker camera | about $57.00 | Deployed-rig component. |
| PTZ rig hardware | Pan/tilt platform | about $13.99 | Deployed-rig component. |
| PTZ rig hardware | Regulated supply | about $8.59 | Deployed-rig component. |
| Existing starter-kit parts | Arduino, servos, cable, wiring | Not priced here | Reuse/verify; not added to the planning subtotal. |
| Test fixture | Green Toys tugboat | Not priced here | Evaluation target, not part of the deployed rig. |
| Optional/reusable | Multimeter and suitable extensions | Not priced here | Reusable bench equipment. |

The three priced rig items total about **$79.58**, which is within the stated
rough $60–80 rig target before tax and shipping. This is a planning subtotal,
not a purchase record or a claim that every electrical, mechanical, or
long-duration characteristic has been qualified.

## Zoom scope

The current camera/rig has no motorized optical zoom. Detection and tracking
always use the full fixed-field-of-view image. The optional automatic digital
zoom is a downstream software crop for annotation and recording only; it does
not add optical detail or change detector/controller coordinates.

The software vision baseline is Ultralytics 8.4.104 with lightweight
`yolo11n.pt`, Torch 2.11.0+cu128, torchvision 0.26.0+cu128, and
opencv-python 5.0.0.93. Model weights are external artifacts and are not stored
in Git. Pyserial 3.x is the optional host serial dependency; it is not part of
base, development, vision, or CI installation.
