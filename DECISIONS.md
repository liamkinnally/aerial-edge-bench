# Decisions and takeaways

These are the choices I made while working through the aerial detection benchmark, what the results showed, and what I would test next. The measurements are from my Apple M4 Pro, not drone hardware.

## Model and training

I started with the nano model because compute and power are limited, both on my laptop and on smaller drones. I fine-tuned it on VisDrone because aerial images are closer to the view I wanted to test than ordinary ground-level photos. On the demo frame, the fine-tuned model found more objects than the stock COCO model. That comparison is visual only, since the models use different classes.

I used the first 10 epochs to make sure the training pipeline worked, then ran a separate 50-epoch training experiment. Validation mAP50 increased from 0.253 to 0.308. I stopped at 50 because it fit the time I had, not because I had established that training had converged. More training could help, but I haven't tested it.

## Measuring inference

I compared PyTorch FP32, ONNX FP32, and ONNX INT8 on the CPU I had available. A drone with an NVIDIA computer could behave differently, so I treated the Mac as a way to compare configurations, not as a stand-in for actual deployment performance.

I kept image loading outside the timed loop because a camera pipeline would already have frames in memory. I kept resizing and the rest of the prediction call inside the timer because that work still has to happen for each frame. I also disabled console logging. Mean latency stayed the same to one decimal place; p95 changed slightly, but that could just be noise.

One earlier PyTorch measurement was 16.7 ms with a 384 × 640 input. The later measurement was 22.9 ms with a 640 × 640 input. The checkpoint and measurement wrapper changed too, so I can't isolate how much of the difference came from image shape. For the final comparison, the configurations within each input-size group used matching actual input shapes.

The saved detection setup uses external non-maximum suppression (NMS). I kept that mode consistent across exports and measurements. I haven't tested the NMS-free mode yet, and it could change the speed results.

## What the results showed

**INT8 was smaller, but not faster on this CPU.** At 640, quantization reduced the ONNX file size by about 69%, from 9.861 MB to 3.045 MB. Validation mAP50–95 fell from 0.169 to 0.153, and mean latency rose from 10.9 ms to 12.2 ms. I expected a speed improvement, so this was the most surprising result. Smaller model files don't automatically mean faster inference; the runtime and hardware matter.

**The larger input improved detection accuracy, at a latency cost.** Moving ONNX FP32 inference from 640 to 960 raised mAP50–95 from 0.169 to 0.213. Pedestrian AP50–95 rose from 0.137 to 0.199. Mean latency increased from 10.9 ms to 22.2 ms. The interesting comparison was ONNX FP32 at 960 against PyTorch FP32 at 640: 22.2 ms versus 22.9 ms, with higher validation accuracy for ONNX at 960. That's an observed comparison on this machine, not proof the two runtimes have equivalent latency.

I trained at 640 and only changed the inference size for the 960 tests. Training at 960 is a separate experiment and might produce different results.

## Detection messages versus video

I compared the size of JSON detection messages against two ways of representing the same 353-frame clip: its compressed MP4 file and separately encoded JPEG frames. At the clip's recorded 24 frames/s, detection messages averaged 4.0 kB/s, compared with 63.3 kB/s for the MP4 file and 5,277.5 kB/s for JPEG frames.

Those are payload comparisons, not measured savings over a radio link. The ratios depend heavily on how the video is encoded, and I haven't tested a real camera or link. Also, 78 of the 353 frames produced empty detection messages. That might mean there were no detections, but it could also mean the model missed objects. Smaller messages aren't automatically better if useful information is lost.

## What I would test next

I'd first repeat the latency measurements across different images rather than timing one frame over and over. I'd also evaluate with the official VisDrone toolkit so the accuracy numbers are easier to compare with other work.

After that, I'd train at 960 rather than only increasing inference size, and try a model larger than nano to see whether the accuracy gain is worth the extra latency. I'd test NMS-free inference as well.

On target hardware, I'd compare the appropriate runtime and quantization options, including TensorRT on supported NVIDIA devices. I'd measure sustained latency, memory, heat, power, and actual camera-to-link behavior. I'd also want evaluation footage from the intended camera and environment, because accuracy on VisDrone doesn't tell me everything about how the model would perform there.

## Development

I worked out the experiment plan with Claude, wrote the timing loop from a proposed skeleton with review, and wrote the original bandwidth comparison math and this decisions log. Claude supplied the bandwidth frame loop and message format. Codex implemented the supporting training, export, validation, measurement, testing, and reporting code. The [README](README.md#development) has the contribution summary.

Some decisions started as suggestions I explored with Claude; others came from results I noticed while running the experiments. I kept the choices and interpretations here because they're the part of the work I want to be able to explain and revisit.
