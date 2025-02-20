# Triple Chaos Encryption

## Introduction

In the digital era, ensuring image security is a top priority. Traditional encryption methods like AES and RSA struggle with large image data, leading to inefficiencies in processing time and security. SecureVision presents an innovative approach using chaos-based encryption, leveraging chaotic system properties such as high sensitivity to initial conditions, ergodicity, and pseudo-random behavior. This project implements and compares chaos maps, including Arnold Cat Map, Henon Map, and Logistic Chaos Map, to deliver a high-security, computationally efficient image encryption system.

## Features

* Increased Security Complexity: The use of multiple chaos maps ensures a layered encryption approach that is more resilient to attacks than single-map encryption.

* Optimized Computational Efficiency: By strategically applying Arnold Cat, Henon, and Logistic Maps, our approach minimizes redundant computations and enhances processing speed.

* Hybrid Chaos-Based Encryption: Combines multiple chaotic maps into a unified framework to maximize encryption robustness.

* Ergodic and Pseudo-Random Properties: Utilizes chaos theory principles to make encryption unpredictable and highly resistant to cryptanalysis.

* Scalability and Adaptability: Suitable for various types of digital image encryption, ensuring broad applicability in security-critical fields.

## Henon Map Encryption Example

```python
def henon_encrypt(image, a=1.4, b=0.3):
    height, width = image.shape
    x, y = 0.1, 0.1  # Initial conditions

    for i in range(height):
        for j in range(width):
            x_new = 1 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new
            image[i, j] = (image[i, j] + int(255 * x)) % 256
    return image
```


## Installation

## Images and Workflow

Flowchart of Project:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/Pipeline.png?raw=true)

Input Image:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/input.png?raw=true)

Encoded Image:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/input_TripleChaosEnc.png?raw=true)

Decoded Image:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/input_TripleChaosDec.png?raw=true)