# Guided pan/tilt calibration

Controlled operation verified the frozen 75–105 degree command envelope on
each logical and physical axis, with neutral at 90/90. Revalidate direction,
clearance, and all four endpoints after any mechanical, wiring, power, or USB
topology change. The calibration tool never expands beyond the configured
range.

Before arming, verify the external regulated 5 V servo supply, common ground,
mechanical clearance, and the exact Uno device identity. Keep direct power
removal available. Recheck `/dev/ttyACM0` after any USB topology change; the
software does not discover or select serial devices.

Run one axis at a time:

```bash
conda run --no-capture-output -n marine_ptz python tools/calibrate_ptz.py \
  --config configs/hardware.yaml \
  --port /dev/ttyACM0 \
  --axis pan \
  --arm-hardware
```

Use the same command with `--axis tilt` only after the pan session has ended
cleanly. The utility starts at 90/90, prints logical and physical targets, and
accepts these single-key controls:

| Key | Action |
| --- | --- |
| `+` / `-` | Move the selected logical axis by exactly one degree. |
| `c` | Send `CENTER`, then hold 90/90. |
| `t` | Request `STATUS`, then hold the reported position. |
| `q`, Escape, or `x` | Send bounded `DISABLE`, close serial, and exit. |

While waiting for an observation, the utility resends the unchanged bounded
pose at the configured 10 Hz rate so the firmware watchdog remains refreshed.
Ctrl+C and exceptions also enter bounded disable/close cleanup. It never moves
the unselected axis away from neutral and rejects attempts beyond the currently
configured limits.

## Observation order

1. At 90/90, confirm that `CENTER` and `STATUS` agree and that neither axis is
   buzzing, binding, pulling a cable, or approaching a collision.
2. For pan, command one degree in each logical direction. Confirm that logical
   increase maps to physical decrease and that the camera follows a target
   moving right in the image; logical decrease must do the converse.
3. Re-center, then repeat for tilt. Logical increase maps to physical decrease
   and must tilt the camera downward toward a target below frame center;
   logical decrease must tilt upward.
4. Starting near neutral, re-approach the frozen endpoints one degree at a
   time. Separately observe physical pan 75, pan 105, tilt 75, and tilt 105.
   Each must remain quiet, mechanically clear, cable-safe, and repeatable on
   the external supply before the current setup is approved.

Do not consider an expansion until all four frozen endpoints have been
re-observed and recorded for the current setup. Any future boundary investigation is a separate reviewed
configuration and firmware change. Expand only one boundary on one axis, at
most five degrees per trial. Stop immediately at buzzing, collision, cable
tension, mechanical binding, supply instability, or uncertain direction.
Record the first unsafe or questionable point and retain a safety margin inside
it; do not use that point as an operating limit.

After safe travel is reconfirmed, choose a neutral pitch that places the
typical target near the frame center without consuming the safety margin.
Record the measured direction, physical clearance, final operating limits,
neutral pitch, power behavior, and shutdown result. The repository makes no
claim that a wider range or different neutral is safe.
