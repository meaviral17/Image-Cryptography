import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
from math import log

########################################################################
# 1) UTILS: LOAD, SAVE, & DISPLAY
########################################################################

def load_image(image_path, mode='color'):
    """
    Loads an image from disk as a NumPy array (RGB or grayscale).
    """
    if mode == 'gray':
        pil_img = Image.open(image_path).convert('L')
        return np.array(pil_img)
    else:
        pil_img = Image.open(image_path).convert('RGB')
        return np.array(pil_img)

def save_image(img_array, out_path):
    """
    Saves a NumPy array as a PNG image.
    """
    if len(img_array.shape) == 2:  # grayscale
        out_img = Image.fromarray(img_array.astype('uint8'), mode='L')
    else:
        out_img = Image.fromarray(img_array.astype('uint8'), mode='RGB')
    out_img.save(out_path)
    print(f"Saved: {out_path}")

def show_image(img_array, title="Image"):
    """
    Displays an image array inline with matplotlib.
    """
    plt.figure(figsize=(5,5))
    # If RGB, show directly; if grayscale, specify cmap='gray'
    if len(img_array.shape) == 2:
        plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img_array)
    plt.title(title)
    plt.axis('off')
    plt.show()

def compute_histogram(img_array, title="Histogram"):
    """
    Plots a histogram of the given image array.
    """
    plt.figure()
    if len(img_array.shape) == 2:
        # Grayscale
        plt.hist(img_array.ravel(), bins=256, range=(0,255), color='gray')
    else:
        # Color
        colors = ('b','g','r')
        for i, c in enumerate(colors):
            hist = cv2.calcHist([img_array.astype('uint8')],[i],None,[256],[0,256])
            plt.plot(hist, color=c)
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()

def adjacent_pixel_correlation(img_array, sample_size=1024, title="Pixel Correlation"):
    """
    Plots correlation between horizontally adjacent pixels.
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    h, w = gray.shape
    samples_x = []
    samples_y = []
    for _ in range(sample_size):
        r = random.randint(0, h-1)
        c = random.randint(0, w-2)
        px1 = gray[r, c]
        px2 = gray[r, c+1]
        samples_x.append(px1)
        samples_y.append(px2)
    plt.figure()
    plt.scatter(samples_x, samples_y, s=2)
    plt.title(title)
    plt.xlabel("Pixel (r,c)")
    plt.ylabel("Pixel (r,c+1)")
    plt.show()

########################################################################
# 2) CHAOS MAP FUNCTIONS
#    (A) Arnold’s Cat
########################################################################

def arnold_cat_transform(img):
    """
    One iteration of Arnold's Cat:
        (x', y') = (x + y) mod n, (x + 2*y) mod n
    """
    n = img.shape[0]
    transformed = np.zeros_like(img)
    for x in range(n):
        for y in range(n):
            new_x = (x + y) % n
            new_y = (x + 2*y) % n
            transformed[new_x, new_y] = img[x, y]
    return transformed

def arnold_cat_encrypt(img_array, iterations):
    """
    Applies the Arnold transform 'iterations' times.
    """
    h, w = img_array.shape[:2]
    if h != w:
        raise ValueError("Arnold's Cat requires a square image!")
    result = img_array.copy()
    for _ in range(iterations):
        result = arnold_cat_transform(result)
    return result

def arnold_cat_period(n):
    """
    Estimate the period for an n x n image to return to original.
    """
    if (n % 2 == 0) and 5**int(round(log(n/2,5))) == int(n/2):
        return 3*n
    elif 5**int(round(log(n,5))) == int(n):
        return 2*n
    elif (n % 6 == 0) and 5**int(round(log(n/6,5))) == int(n/6):
        return 2*n
    else:
        return int(12*n/7)

def arnold_cat_decrypt(img_array, iterations):
    """
    Decrypt by continuing from 'iterations' up to period.
    """
    n = img_array.shape[0]
    per = arnold_cat_period(n)
    result = img_array.copy()
    for i in range(iterations, per):
        result = arnold_cat_transform(result)
    return result

########################################################################
#    (B) Henon Map (XOR-based)
########################################################################

def bits_to_bytes(bits):
    assert len(bits) % 8 == 0
    out = []
    for i in range(0, len(bits), 8):
        val = 0
        for b in bits[i:i+8]:
            val = (val << 1) | b
        out.append(val)
    return out

def henon_map_sequence(num_bits, x0=0.1, y0=0.1, a=1.4, b=0.3):
    x = x0
    y = y0
    bits = []
    for _ in range(num_bits):
        x_next = y + 1 - a*(x**2)
        y_next = b*x
        x, y = x_next, y_next
        bits.append(0 if x <= 0.4 else 1)
    return bits

def henon_encrypt(img_array, x0=0.1, y0=0.1):
    """
    XOR each pixel with a byte from Henon-based stream.
    """
    shape = img_array.shape
    if len(shape) == 2:
        h, w = shape
        c = 1
    else:
        h, w, c = shape

    total_pixels = h*w
    needed_bytes = total_pixels*c
    needed_bits = needed_bytes*8

    henon_bits = henon_map_sequence(needed_bits, x0, y0)
    henon_stream = bits_to_bytes(henon_bits)

    encrypted = np.zeros_like(img_array)
    idx = 0
    for r in range(h):
        for c_ in range(w):
            if c == 1:
                px = img_array[r, c_]
                encrypted[r, c_] = px ^ henon_stream[idx]
                idx += 1
            else:
                px_r, px_g, px_b = img_array[r, c_]
                encrypted[r, c_, 0] = px_r ^ henon_stream[idx]
                encrypted[r, c_, 1] = px_g ^ henon_stream[idx+1]
                encrypted[r, c_, 2] = px_b ^ henon_stream[idx+2]
                idx += 3
    return encrypted

def henon_decrypt(img_array, x0=0.1, y0=0.1):
    """
    Same function (XOR-based).
    """
    return henon_encrypt(img_array, x0, y0)

########################################################################
#    (C) Logistic Map (XOR-based)
########################################################################

def logistic_map_sequence(num_values, r=3.99, seed=0.12345):
    x = seed
    out_bytes = []
    for _ in range(num_values):
        x = r*x*(1 - x)
        out_bytes.append(int((x*1e6) % 256))
    return out_bytes

def logistic_encrypt(img_array, r=3.99, seed=0.12345):
    """
    XOR each pixel with logistic-based stream.
    """
    shape = img_array.shape
    if len(shape) == 2:
        h, w = shape
        c = 1
    else:
        h, w, c = shape

    total_pixels = h*w
    needed_bytes = total_pixels*c
    chaos_stream = logistic_map_sequence(needed_bytes, r, seed)

    encrypted = np.zeros_like(img_array)
    idx = 0
    for row in range(h):
        for col in range(w):
            if c == 1:
                px = img_array[row, col]
                encrypted[row, col] = px ^ chaos_stream[idx]
                idx += 1
            else:
                px_r, px_g, px_b = img_array[row, col]
                encrypted[row, col, 0] = px_r ^ chaos_stream[idx]
                encrypted[row, col, 1] = px_g ^ chaos_stream[idx+1]
                encrypted[row, col, 2] = px_b ^ chaos_stream[idx+2]
                idx += 3
    return encrypted

def logistic_decrypt(img_array, r=3.99, seed=0.12345):
    """
    XOR-based => same function.
    """
    return logistic_encrypt(img_array, r, seed)

########################################################################
# 3) TRIPLE-CHAOS ENCRYPT / DECRYPT (like Triple-DES)
########################################################################

def triple_chaos_encrypt(
    img_array,
    arnold_key=10,
    henon_key=(0.1,0.1),
    logistic_key=(3.99, 0.12345)
):
    """
    1) Arnold's Cat -> 2) Henon -> 3) Logistic
    """
    # Arnold
    step_arnold = arnold_cat_encrypt(img_array, arnold_key)
    # Henon
    x0, y0 = henon_key
    step_henon = henon_encrypt(step_arnold, x0, y0)
    # Logistic
    r, seed = logistic_key
    step_logistic = logistic_encrypt(step_henon, r, seed)
    return step_logistic

def triple_chaos_decrypt(
    cipher_array,
    arnold_key=10,
    henon_key=(0.1,0.1),
    logistic_key=(3.99, 0.12345)
):
    """
    Reverse in opposite order: Logistic -> Henon -> Arnold
    """
    # Undo Logistic
    r, seed = logistic_key
    step_logistic = logistic_decrypt(cipher_array, r, seed)
    # Undo Henon
    x0, y0 = henon_key
    step_henon = henon_decrypt(step_logistic, x0, y0)
    # Undo Arnold
    step_arnold = arnold_cat_decrypt(step_henon, arnold_key)
    return step_arnold

########################################################################
# 4) DEMO SCRIPT: APPLY & DISPLAY AT EACH STAGE
########################################################################

def run_demo():
    INPUT_IMAGE = "input.png"  # <--- Change to your actual image (square!)
    # Load your original image
    original = load_image(INPUT_IMAGE, mode='color')
    h, w = original.shape[:2]
    print(f"Loaded image '{INPUT_IMAGE}' with shape: {original.shape}")
    if h != w:
        print("WARNING: This is not a square image. Arnold's Cat won't work properly!")
        # Alternatively, you can remove the Arnold step or pad the image.

    # Show original
    show_image(original, title="(1) Original Image")
    compute_histogram(original, title="(1) Original Histogram")
    adjacent_pixel_correlation(original, title="(1) Original Correlation")

    #################################################
    # ENCRYPTION
    #################################################
    # A) Arnold
    arnold_key = 5  # or any integer
    step_arnold = arnold_cat_encrypt(original, arnold_key)
    show_image(step_arnold, title="(2) After Arnold's Cat")
    compute_histogram(step_arnold, "(2) Arnold's Cat Histogram")
    adjacent_pixel_correlation(step_arnold, title="(2) Arnold's Cat Correlation")

    # B) Henon
    henon_key = (0.1, 0.2)  # x0=0.1, y0=0.2
    step_henon = henon_encrypt(step_arnold, henon_key[0], henon_key[1])
    show_image(step_henon, title="(3) After Henon Encryption")
    compute_histogram(step_henon, "(3) Henon Histogram")
    adjacent_pixel_correlation(step_henon, title="(3) Henon Correlation")

    # C) Logistic
    logistic_key = (3.99, 0.11111)  # r=3.99, seed=0.11111
    final_cipher = logistic_encrypt(step_henon, logistic_key[0], logistic_key[1])
    show_image(final_cipher, title="(4) Final Cipher (Logistic)")
    compute_histogram(final_cipher, "(4) Final Cipher Histogram")
    adjacent_pixel_correlation(final_cipher, title="(4) Final Cipher Correlation")

    # Save final cipher
    save_image(final_cipher, "input_TripleChaosEnc.png")

    #################################################
    # DECRYPTION (Reverse the steps)
    #################################################
    # A) Undo Logistic
    dec_log = logistic_decrypt(final_cipher, logistic_key[0], logistic_key[1])
    show_image(dec_log, title="(5) Decrypt Step 1: Undo Logistic")

    # B) Undo Henon
    dec_henon = henon_decrypt(dec_log, henon_key[0], henon_key[1])
    show_image(dec_henon, title="(6) Decrypt Step 2: Undo Henon")

    # C) Undo Arnold
    dec_arnold = arnold_cat_decrypt(dec_henon, arnold_key)
    show_image(dec_arnold, title="(7) Final Decrypted")
    save_image(dec_arnold, "input_TripleChaosDec.png")

    # Optional: Compare final decrypted to original
    mse = np.mean((dec_arnold - original)**2)
    print(f"MSE between original and final decrypted = {mse:.6f}")

# If running as a script, call run_demo():
if __name__ == "__main__":
    run_demo()
