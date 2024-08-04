import numpy as np
from multiprocessing import shared_memory
import time

def main():
    # Wait for the user to confirm that the publisher is running
    input("Press enter after the publisher script is running...")

    # Attach to the existing shared memory block
    existing_shm = shared_memory.SharedMemory(name="shm")

    # Assume the shape and dtype of the array are known
    array_shape = (2, 3)  # This needs to be predefined or communicated separately
    dtype = float

    # Create a NumPy array backed by the shared memory
    shared_array = np.ndarray(array_shape, dtype=dtype, buffer=existing_shm.buf)

    try:
        while True:
            print("Array read from shared memory:", shared_array.copy())
            time.sleep(0.1)  # Sync with the publishing rate

    except KeyboardInterrupt:
        print("Exiting listener...")

    finally:
        # Clean up
        existing_shm.close()

if __name__ == "__main__":
    main()
