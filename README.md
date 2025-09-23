# MultiCapCLIP-AR-Architectural-Enhancements-and-Arabic-Fine-Tuning-for-Multilingual-Visual-Captioning

## Project Metadata
### Authors
- **Team:** Saliyah Alotaibi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
In recent years, vision-language models like CLIP have made significant progress in integrating text and images.  They excel particularly in zero-shot transfer and image-text retrieval. Their capacity to generalize across languages is, however, limited because they have received nearly all of their training on English captions. To close this gap, MultiCapCLIP was developed, which aligns photos with multilingually translated captions to create a common multilingual semantic space.
This project expands MultiCapCLIP to include Arabic, a morphologically rich language spoken by over 400 million people worldwide, which remains underrepresented in multimodal AI research. In Addition, this work proposes a modified architecture by adding an extra self-attention layer to the text encoder, enabling a more efficient capture of Arabic syntactic structures. This modification is designed to enhance the semantic alignment between Arabic captions and image embeddings, to reduce the performance gap with English.

## Problem Statement
The accuracy of CLIP and related multilingual models declines significantly when processing Arabic captions, despite strong results on English-centric datasets. This performance gap is primarily due to the scarcity of high-quality annotated Arabic image–text pairs, the morphological complexity of the Arabic language, and the architectural bias of most models toward English syntax. As a result, the reliability of Arabic retrieval tasks is reduced, which limits the practical deployment of these models in real-world applications across the MENA region.
To address these challenges, my project enhances MultiCapCLIP using Arabic COCO captions and incorporates an additional attention layer into the text encoder. This modification aims to reduce the performance gap between Arabic and English in multimodal tasks by enabling the model to better capture Arabic-specific semantic relationships.

## Application Area and Project Domain
With an emphasis on cross-lingual retrieval—the process of matching images with captions in several languages—this effort falls under the umbrella of multimodal AI and vision–language models. By focusing on Arabic, I aim to bridge a significant support gap for low-resource yet highly influential languages and contribute to the development of AI systems that are more inclusive and globally relevant.
Applications include Arabic-focused educational resources, multilingual search engines, and regional AI services for digital media, government, and healthcare in the MENA area. Enhancing image-caption retrieval in Arabic could provide millions of Arabic speakers with access to state-of-the-art AI, aligning AI innovation with local linguistic and cultural requirements.

## What is the paper trying to do, and what are you planning to do?
The original MultiCapCLIP paper extends CLIP to the multilingual setting by aligning images with multiple translated captions. This approach enables the model to generalize to languages beyond English and supports cross-lingual retrieval tasks without dependence on external machine translation during inference.

In this project, we reproduce the MultiCapCLIP approach with a focus on Arabic captions from the COCO dataset. We also introduce an additional attention layer to the text encoder to better capture linguistic nuances in Arabic. We compare three models: baseline CLIP (English only), MultiCapCLIP (with Arabic captions), and MultiCapCLIP with the added attention layer. Their performance is evaluated using retrieval and captioning metrics, including Recall@K, BLEU, METEOR, ROUGE, and CIDEr. Our main goal is to evaluate whether these architectural enhancements improve Arabic cross-lingual retrieval.
.


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [(ACL'2023) MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning](https://aclanthology.org/2023.acl-long.664/)

### Reference Dataset
- [MS COCO Dataset](https://cocodataset.org/#home)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
