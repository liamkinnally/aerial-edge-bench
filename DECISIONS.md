smallest model because hardware limitations both personal and for small drones. trained on drone photos because those are the types of photos it will see in use. stock model found next to nothing in drone clip while fine-tuned found more.

timed on CPU to compare PyTorch, ONNX, and INT8 on the hardware I have. some drones would use an NVIDIA computer but not the kind that I have access to. my Mac's CPU is a stand-in, and I haven't tested real drone hardware.

moved image loading out of timed loop because real drone's camera hands frames straight into memory. timing file read would be accounting for a step that isn't present in real system. would make the model look slower than it would be in use. resizing image stayed in loop because real drone would have to do that each frame.

turned off logging because isn't part of real use. the average was the same to one decimal place. the p95 dropped slightly, which could be noise.

first 10 epochs were cheap test for pipeline. separate 50 epochs after first 10 were for actually training. scores were about a fifth higher after training (mAP50 0.253 to 0.308) than the 10 epoch run. stopped at 50 because that's just what fit in an evening. didn't test whether more than 50 would be better.

PyTorch time changed from 16.7ms to 22.9ms on the Apple M4 Pro CPU. the first measurement used 384 x 640 pixels and the later measurement used 640 x 640 pixels. the input size was one difference. the weights and the measurement wrapper also changed, so I can't say how much each contributed. rows of the same size use identical input shapes, which makes the comparison between formats fair.

My results use the mode with the extra cleanup step, because that's what training defaulted to. I didn't test the NMS-free mode, which might be faster.

shrinking in INT8: file size 69% smaller than the ONNX FP32 file, however accuracy was about a 1/10th lower (0.169 to 0.153) and speed was about 12% slower (10.9ms to 12.2ms), which was surprising. INT8 speedups depend on the hardware, worth re-testing on target hardware.

bigger image size (640 to 960) overall increased accuracy about a quarter (0.169 to 0.213). pedestrian accuracy went up almost 50% (0.137 to 0.199). time was about double (10.9ms to 22.2ms). ONNX at 960 takes about the same time as original PyTorch model at 640 (22.2ms against 22.9ms) and is about a quarter more accurate, so the speed gained from exporting paid for the larger image. model was trained at 640 and only tested at 960, so training at 960 might do better, but I didn't test this.

I compared detection messages against two methods of sending video. detections were 4.0kB/s. MP4 file 63.3kB/s and JPEG per frame 5,278 kB/s. neither 16x nor 1321x are the exact savings to assume a real drone would get, as size depends on the video format. I didn't measure a radio link and I don't know what video format a real drone might use. these are comparisons between every frame as its own JPEG picture and the clip's own compressed MP4 file. 78 of the 353 frames produced an empty message because the model found nothing in them. a small message can mean the model missed things, so less data isn't necessarily good. has to be read alongside accuracy scoring.

I wrote the timing loop from a proposed skeleton with review and comparison math for the bandwidth script. Claude wrote the frame loop and message format for the bandwidth script. Codex wrote the wrapper, the result recording, and the MP4 calculation. everything else was written by Codex (training scripts, tests, and results files) following a plan I created with Claude. these measurement decisions were either made by myself or were things that I understood with help through Claude and agreed with.

what surprised me was that shrinking the model worked. the file got 69% smaller than the ONNX FP32 file and still kept 9/10ths of its accuracy. I was also surprised by the INT8 rows coming out slower, when the purpose for that shrinking was speed. so, shrinking cost less accuracy than I expected, and gave less speed than I expected.

next I would train at 960, not only test at it. on real device hardware I'd try the next size up from nano and see what the extra accuracy costs in speed. I'd perform much more training than 50 epochs, likely until the scores stop improving. I'd use TensorRT for shrinking for NVIDIA chips, and INT8 on the actual device. I would also measure a real link with the video format the drone actually uses, speed on the real device, heat and power over time, memory while running, and accuracy on the objects that matter in context. my dataset would also use footage from the real camera.
