# config.py

MATRIX_SIZES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (1536, 1536, 1536),
    (2048, 2048, 2048)
]

DTYPES = ["float32", "float16"]

MIN_RUNTIME = 0.3  # seconds for stable timing