import numpy as np
from multiprocessing import shared_memory
import time

def main():
    # Create a new shared memory block
    shm = shared_memory.SharedMemory(name="shm", create=True, size=48)  # 48 bytes for 2x3 float64 array

    # Create a NumPy array backed by shared memory
    shared_array = np.ndarray((2, 3), dtype=float, buffer=shm.buf)

    try:
        while True:
            # Generate a new random array
            new_array = np.random.rand(2, 3)
            shared_array[:] = new_array[:]  # Copy new data to shared memory
            print("Array published to shared memory:", shared_array)
            time.sleep(1.0)  # Wait for 0.5 seconds before updating again

    except KeyboardInterrupt:
        print("Exiting publisher...")

    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    main()
