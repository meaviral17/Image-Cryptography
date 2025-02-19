"""
triple_chaos_demo.py

Demonstrates triple-chaos encryption (Arnold -> Henon -> Logistic) 
and saves all intermediate images and plots to an outputs/ folder.
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
from math import log
from scipy.integrate import solve_ivp

##############################################################################
# 1) Create an outputs/ folder if it doesn't exist
##############################################################################
os.makedirs("outputs", exist_ok=True)

##############################################################################
# 2) IMAGE IO & PLOTTING UTILS
##############################################################################

def load_image(image_path, mode='color'):
    """
    Loads an image from disk as a NumPy array.
      - mode='color' => shape (H, W, 3)
      - mode='gray'  => shape (H, W)
    """
    if mode == 'gray':
        pil_img = Image.open(image_path).convert('L')
        return np.array(pil_img)
    else:
        pil_img = Image.open(image_path).convert('RGB')
        return np.array(pil_img)

def save_image(img_array, out_path):
    """
    Saves a NumPy array to disk as a PNG.
    """
    if len(img_array.shape) == 2:
        out_img = Image.fromarray(img_array.astype('uint8'), 'L')
    else:
        out_img = Image.fromarray(img_array.astype('uint8'), 'RGB')
    out_img.save(out_path)
    print(f"Saved: {out_path}")

def show_and_save_image(img_array, title="Image", save_path=None):
    """
    Displays the image using matplotlib and optionally saves to 'save_path'.
    """
    plt.figure(figsize=(5,5))
    if len(img_array.shape) == 2:
        plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img_array)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def show_and_save_histogram(img_array, title="Histogram", save_path=None):
    """
    Plots and shows the histogram of an image, optionally saves to 'save_path'.
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
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def show_and_save_correlation(img_array, sample_size=1024, title="Correlation", save_path=None):
    """
    Plots horizontally adjacent pixel correlation, optionally saves to file.
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
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

##############################################################################
# 3) CHAOS MAP FUNCTIONS
#    A) Arnold's Cat
##############################################################################

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
    Apply Arnold's Cat transform 'iterations' times.
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
    Estimate period for n x n image returning to original under Arnold's Cat.
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
    Decrypt by continuing from 'iterations' to full period.
    """
    n = img_array.shape[0]
    per = arnold_cat_period(n)
    result = img_array.copy()
    for i in range(iterations, per):
        result = arnold_cat_transform(result)
    return result

##############################################################################
#    B) Henon Map (XOR-based)
##############################################################################

def bits_to_bytes(bits):
    """
    Convert a list of bits (0/1) into a list of bytes [0..255].
    """
    assert len(bits) % 8 == 0
    out = []
    for i in range(0, len(bits), 8):
        val = 0
        for b in bits[i:i+8]:
            val = (val << 1) | b
        out.append(val)
    return out

def henon_map_sequence(num_bits, x0=0.1, y0=0.1, a=1.4, b=0.3):
    """
    Generate 'num_bits' bits from Henon map.
    """
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
    XOR each pixel with Henon-based pseudorandom bytes.
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
        for cc in range(w):
            if c == 1:
                px = img_array[r, cc]
                encrypted[r, cc] = px ^ henon_stream[idx]
                idx += 1
            else:
                px_r, px_g, px_b = img_array[r, cc]
                encrypted[r, cc, 0] = px_r ^ henon_stream[idx]
                encrypted[r, cc, 1] = px_g ^ henon_stream[idx+1]
                encrypted[r, cc, 2] = px_b ^ henon_stream[idx+2]
                idx += 3
    return encrypted

def henon_decrypt(img_array, x0=0.1, y0=0.1):
    """
    Same as henon_encrypt (XOR-based).
    """
    return henon_encrypt(img_array, x0, y0)
        
##############################################################################
# 3) Rössler Chaotic Map (Continuous-time System)
##############################################################################

def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]

def generate_rossler_sequence(length, x0=0.1, y0=0.1, z0=0.1, a=0.2, b=0.2, c=5.7):
    t_span = [0, length * 0.1]  # Time interval
    t_eval = np.linspace(t_span[0], t_span[1], length)  # Sample points
    sol = solve_ivp(rossler_system, t_span, [x0, y0, z0], args=(a, b, c), t_eval=t_eval)
    seq = np.abs(sol.y[0] * 1e6) % 256  # Scale values to byte range
    return seq.astype(np.uint8)

def rossler_encrypt(img_array, x0=0.1, y0=0.1, z0=0.1):
    shape = img_array.shape
    h, w, c = shape if len(shape) == 3 else (*shape, 1)
    total_pixels = h * w * c
    chaos_stream = generate_rossler_sequence(total_pixels, x0, y0, z0)
    encrypted = img_array.flatten() ^ chaos_stream
    return encrypted.reshape(shape)

def rossler_decrypt(img_array, x0=0.1, y0=0.1, z0=0.1):
    return rossler_encrypt(img_array, x0, y0, z0)  # XOR-based is reversible

##############################################################################
# 4) TRIPLE-CHAOS (Arnold -> Henon -> Rossler)
##############################################################################

def triple_chaos_encrypt(
    img_array,
    arnold_key=10,
    henon_key=(0.1,0.1),
    rossler_key=(0.1, 0.1, 0.1)
):
    # 1) Arnold
    step1 = arnold_cat_encrypt(img_array, arnold_key)
    # 2) Henon
    x0, y0 = henon_key
    step2 = henon_encrypt(step1, x0, y0)
    # 3) Rossler
    x0, y0, z0 = rossler_key
    step3 = rossler_encrypt(step2, x0, y0, z0)
    return step3

def triple_chaos_decrypt(
    cipher_array,
    arnold_key=10,
    henon_key=(0.1,0.1),
    rossler_key=(0.1, 0.1, 0.1)
):
    # Reverse order: rossler -> henon -> arnold
    x0, y0, z0 = rossler_key
    step1 = rossler_decrypt(cipher_array, x0, y0, z0)

    x0, y0 = henon_key
    step2 = henon_decrypt(step1, x0, y0)

    step3 = arnold_cat_decrypt(step2, arnold_key)
    return step3

##############################################################################
# 5) DEMO FUNCTION: SHOW & SAVE AT EACH STAGE
##############################################################################

def run_demo():
    input_image = "input.png"  # change if needed
    print("Loading:", input_image)
    original = load_image(input_image, mode='color')
    h, w = original.shape[:2]
    print(f"Image shape: {original.shape}")
    if h != w:
        print("WARNING: Image is not square! Arnold's Cat requires square images.")

    # Show & Save Original
    show_and_save_image(
        original,
        title="(1) Original Image",
        save_path="outputs/1_original.png"
    )
    show_and_save_histogram(original, title="(1) Original Histogram", save_path="outputs/1_original_hist.png")
    show_and_save_correlation(original, title="(1) Original Correlation", save_path="outputs/1_original_corr.png")

    #################################################
    # Stage A: Arnold's Cat
    #################################################
    arnold_key = 5
    step_arnold = arnold_cat_encrypt(original, arnold_key)
    show_and_save_image(step_arnold, "(2) After Arnold's Cat", "outputs/2_arnold.png")
    show_and_save_histogram(step_arnold, "(2) Arnold Histogram", "outputs/2_arnold_hist.png")
    show_and_save_correlation(step_arnold, title="(2) Arnold Correlation", save_path="outputs/2_arnold_corr.png")

    #################################################
    # Stage B: Henon
    #################################################
    henon_key = (0.1, 0.2)
    x0, y0 = henon_key
    step_henon = henon_encrypt(step_arnold, x0, y0)
    show_and_save_image(step_henon, "(3) After Henon", "outputs/3_henon.png")
    show_and_save_histogram(step_henon, "(3) Henon Histogram", "outputs/3_henon_hist.png")
    show_and_save_correlation(step_henon, title="(3) Henon Correlation", save_path="outputs/3_henon_corr.png")

    #################################################
    # Stage C: Rossler
    #################################################
    rossler_key = (0.3, 0.2, 0.1)
    x0, y0, z0 = rossler_key
    final_cipher = rossler_encrypt(step_henon, x0, y0, z0)
    show_and_save_image(final_cipher, "(4) Final Cipher (Rossler)", "outputs/4_final_cipher.png")
    show_and_save_histogram(final_cipher, "(4) Cipher Histogram", "outputs/4_final_cipher_hist.png")
    show_and_save_correlation(final_cipher, title="(4) Cipher Correlation", save_path="outputs/4_final_cipher_corr.png")

    # Save final cipher
    save_image(final_cipher, "outputs/final_cipher.png")

    #################################################
    # DECRYPTION (Reverse Steps)
    #################################################
    # Step 1: Undo Rossler
    dec_log = rossler_decrypt(final_cipher, rossler_key[0], rossler_key[1])
    show_and_save_image(dec_log, "(5) Decrypt Step 1 (Undo Rossler)", "outputs/5_undo_rossler.png")

    # Step 2: Undo Henon
    dec_henon = henon_decrypt(dec_log, henon_key[0], henon_key[1])
    show_and_save_image(dec_henon, "(6) Decrypt Step 2 (Undo Henon)", "outputs/6_undo_henon.png")

    # Step 3: Undo Arnold
    dec_arnold = arnold_cat_decrypt(dec_henon, arnold_key)
    show_and_save_image(dec_arnold, "(7) Final Decrypted", "outputs/7_final_decrypted.png")
    save_image(dec_arnold, "outputs/final_decrypted.png")

    # Check MSE
    mse = np.mean((dec_arnold - original)**2)
    print(f"MSE between original and final decrypted = {mse:.6f}")

# If running as a script
if __name__ == "__main__":
    run_demo()
