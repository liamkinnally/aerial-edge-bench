"""The supplied CPU timing loop, with paths and run count supplied by the caller."""


def measure_latency(model_path, image_path, run_count):
    import time
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    image = cv2.imread(image_path)
    assert image is not None, "image failed to load"

    #warm up
    for _ in range(10):
        model(image, device = "cpu")

    times = []

    for _ in range(run_count):
        start = time.perf_counter()
        model(image, device = "cpu")
        elapsed_ms = (time.perf_counter()-start) * 1000
        times.append(elapsed_ms)

    mean = sum(times) / len(times)
    times.sort()
    p95 = times[int(0.95 * len(times))]
    fps = 1000 / mean

    print(f"mean {mean:.1f} ms | p95 {p95:.1f} ms | {fps:.1f} FPS")

    return mean, p95, fps, times, model, image
