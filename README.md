# Self-Supervised Point Tracking using Video Diffusion Models

This repository contains the code for self-supervised point tracking using video diffusion models.

## Installation

Follow these steps to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/TheGreatestEvers/ADL4CV.git
    cd ADL4CV
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Download and Prepare Dataset

1. Download the TAP-Vid dataset and place it in the main directory of the repository.

### 2. Extract and Combine Features

1. Extract and combine feature tensor from the Video Diffusion Model by executing the Jupyter notebook:
    ```bash
    jupyter notebook extract_and_combine_vdm_features.ipynb
    ```

### 3. Prepare Video Directory Structure

1. Create a directory with the following structure:

    ```plaintext
    a_video_dir
    ├── video_0
    │   └── mask
    └── video_1
        └── mask
    ```

    - Each `mask` directory must contain the foreground masks for the Davis videos.

### 4. Extract Optical Flow (OF) Point Pairs

1. Run the following script to extract OF point pairs:
    ```bash
    python setup_dino_of.py
    ```

### 5. Start Test-Time Training

1. Start the test-time training by running:
    ```bash
    python diffusion_tracker.py
    ```

    - Note: Ensure you select the video index by setting the macro `VIDEO_IDX`.

### 6. Plot and Evaluate Results

1. Run the following script to evaluate the tracking results:
    ```bash
    python visualize_and_evaluate.py
    ```