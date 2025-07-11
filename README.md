# Triple Chaos Encryption

## Introduction

In the digital era, ensuring image security is a top priority. Traditional encryption methods like AES and RSA struggle with large image data, leading to inefficiencies in processing time and security. SecureVision presents an innovative approach using chaos-based encryption, leveraging chaotic system properties such as high sensitivity to initial conditions, ergodicity, and pseudo-random behavior. This project implements and compares chaos maps, including Arnold Cat Map, Henon Map, and Rössler Map, to deliver a high-security, computationally efficient image encryption system.

## Features

* Increased Security Complexity: The use of multiple chaos maps ensures a layered encryption approach that is more resilient to attacks than single-map encryption.

* Optimized Computational Efficiency: By strategically applying Arnold Cat, Henon, and Rössler Maps, our approach minimizes redundant computations and enhances processing speed.

* Hybrid Chaos-Based Encryption: Combines multiple chaotic maps into a unified framework to maximize encryption robustness.

* Ergodic and Pseudo-Random Properties: Utilizes chaos theory principles to make encryption unpredictable and highly resistant to cryptanalysis.

* Scalability and Adaptability: Suitable for various types of digital image encryption, ensuring broad applicability in security-critical fields.


## Images and Workflow

Flowchart of Project:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/Pipeline.png?raw=true)

Input Image:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/input.png?raw=true)

Encoded Image:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/input_TripleChaosEnc.png?raw=true)

Decoded Image:

![image alt](https://github.com/meaviral17/Image-Cryptography/blob/main/input_TripleChaosDec.png?raw=true)


## Installation

### Step 1: Clone the Repository  
To begin, clone the project repository from GitHub using the following command:  
```sh
git clone https://github.com/meaviral17/Image-Cryptography.git
```
### Step 2: Navigate to the Project Directory
Once the repository is cloned, move into the project folder using:
```sh
cd IMAGE_CRYPTOGRAPHY
```
### Step 3: Install Dependencies
To ensure all required dependencies are installed, run the appropriate command based on your environment:
For Python-based installations:
```sh
pip install -r requirements.txt
```

### Step 4: Run the Application
Once all dependencies are installed, execute the application with:

```sh
python triple_chaos_demo.py
```
