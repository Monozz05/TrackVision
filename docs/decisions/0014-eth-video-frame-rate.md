# Decision 0014: ETH's video rate

- **Date:** 2026-11-02
- **Decided by:** the project owner.
- **Status:** **in force.** ETH's browser copies play at 15 fps, the rate it was filmed at, with every frame kept. This was changed
  on 2026-11-02, reverted the same day at the owner's request because many visible people had no box, and applied again the same day
  when the owner asked for ETH to be slowed down and for YOLO11 boxes (decision 0015). Nothing in `data/raw`, no trajectory, no metric
  and no stored result was ever affected.

## What happened

The owner reported people moving too fast at 1×. ETH's video file declares 25 fps, but its annotations are 6 frames apart at the
documented 0.4 s, which means it was filmed at 15 fps. The Phase 0B report recorded this mismatch. Played at 25 fps, people walk at
2.49 m/s on screen, while the same annotations give 1.50 m/s, so ETH plays 1.67 times too fast. HOTEL's annotations are 10 frames
apart at 25 fps, so HOTEL plays at its true speed.

ETH's copies were rebuilt at 15 fps, keeping every frame. The owner then asked for it to be reverted, because many visible people
have no box. That revert was done as asked, and the rate was applied again once the boxes were being dealt with.

## Why people had no box

The missing boxes do not come from the frame rate. The picture and the frame numbers stayed in step throughout, and the tracks were
identical at both rates. They come from the annotations, which leave people out. At frame 2473, for example, at least 8 people are
visible, and the annotations hold nobody between frames 2394 and 2862. Nobody is annotated on 25% of ETH's 0.4 s steps.

The annotations that exist are in step with the video: measured against the picture, annotated paths sit at or within a few frames
of zero shift in every stretch of ETH. So the boxes shown are right, and they are not everyone. Boxing everyone needs a real person
detector, which is decision 0015. Until those boxes exist, stretches with no boxes last 1.67 times longer on screen.

## Found while testing every frame

A sweep of all 166,228 frames of the 52 videos in Chrome found one mismatch: ZARA1 frame 0 reads as frame 1. ZARA1's browser copy
starts at 0.04 s instead of 0, because its source file does, so ZARA1's overlay runs one frame (0.04 s) ahead of the picture. It is
the only video that does this, its parts do not, and it was left unchanged at the owner's instruction not to alter the Workspace.

## How it is built

ETH's copies are built with `-r 15` before `-i`, which keeps every frame and changes only their timing.
