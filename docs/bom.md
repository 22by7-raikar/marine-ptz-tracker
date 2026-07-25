# Bill of materials

| Item | Status | Purpose / notes |
| --- | --- | --- |
| Ubuntu laptop with NVIDIA GPU | Available | Capture, detection, control, telemetry, and serial host. |
| InnoMaker U20CAM-1080P USB camera | Arriving | 1080p image source; verify V4L2 compatibility and device path. |
| Arduino Uno R3 | Available | Receives bounded PTZ commands and generates servo signals. |
| XiaoR Geek two-SG90 pan/tilt assembly | Available | Physical positioning mechanism. |
| Regulated 5 V, 3 A external supply | Available | Dedicated servo power; share ground with Arduino. |
| Green Toys tugboat | Available | Initial physical marine target. |
| USB cable for Arduino | Verify | Data/power link from laptop. |
| Jumper wires / suitable servo extension | Verify | Servo signals, common ground, and power distribution. |
| Multimeter | Recommended | Required for polarity and supply-voltage checks. |

The software vision baseline is Ultralytics 8.4.104 with lightweight
`yolo11n.pt`, Torch 2.11.0+cu128, torchvision 0.26.0+cu128, and
opencv-python 5.0.0.93. Model weights are external artifacts and are not stored
in Git. No serial library or camera-specific SDK is selected.
