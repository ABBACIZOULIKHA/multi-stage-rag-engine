import config
import time

device = config.get_device()
print(f"Device: {device}")
print(f"XPU available: {__import__('torch').xpu.is_available()}")

model = config.get_embed_model()
print(f"Embed model: {config.EMBED_MODEL}")
print(f"Model loaded successfully on {device}")

start = time.time()
emb = model.get_text_embedding("Hello world, testing BGE-M3 on GPU")
elapsed = time.time() - start
print(f"Embedding generated in {elapsed:.2f}s, dim={len(emb)}")
print(f"First 5 values: {emb[:5]}")
print("GPU embedding test PASSED!")
