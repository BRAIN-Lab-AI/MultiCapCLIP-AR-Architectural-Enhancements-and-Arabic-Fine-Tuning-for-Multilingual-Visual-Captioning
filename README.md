# MultiCapCLIP-SAB: A Supervised Attention Bridge for Enhanced Zero-Shot Visual Captioning

## Project Metadata
### Authors
- **Team:** Saliyah Alotaibi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Vision–language models have made significant strides in linking text and images, with CLIP becoming an important tool for learning new tasks without additional training and for finding matches between images and text. CLIP is trained on lots of data, so it can work well in new situations without special instructions. MultiCapCLIP improves on CLIP by allowing captions in multiple languages and generating text from idea prompts based on image features. But because the connection between vision and language is not explicitly taught, the model struggles to match image details to the right words.

To address this problem, this work introduces a better design that adds a Supervised Attention Bridge (SAB), a specialized module trained to match images and captions. Unlike the original MultiCapCLIP, which only trains on text and uses prompt auto-encoding, the SAB directly learns to connect CLIP’s image parts to the part of the model that creates text. This method better matches image details with text while maintaining a strong ability to handle new tasks without extra training.

The MultiCapCLIP-SAB model is trained on the MS COCO dataset and tested in a strict zero-shot setting on Flickr30k. By directly supervising the link between vision and language, this architecture is expected to improve both image-text retrieval and caption consistency. This change aims to address the limits of concept-prompt–based mappings and make multilingual visual captioning systems more robust, general, and interpretable.


## Problem Statement
The zero-shot captioning approach of MultiCapCLIP, which learns from text-only data, is powerful but relies on an indirect connection between vision and language mediated by "concept prompts." This can limit the model's ability to capture fine-grained visual details, as the mapping from image to concepts is not explicitly trained. The core challenge is to improve this visual-semantic alignment without sacrificing the model's zero-shot generalization capabilities.

To address this, our project introduces a key architectural enhancement: a supervised attention bridge. This multi-layer Transformer network is trained directly on image-caption pairs to learn an explicit mapping from CLIP's visual patch features to the representational space of the language decoder. The hypothesis is that by supervising this connection, the model can learn a much richer and more detailed visual grounding, which will translate to higher-quality captions, especially in a zero-shot setting on unseen datasets.

## Application Area and Project Domain
This work advances the state-of-the-art in multimodal AI by proposing a hybrid approach that combines the strengths of large-scale, pre-trained models with targeted, supervised training of a lightweight bridge component. The project focuses on enhancing zero-shot visual captioning, aiming to create a model that is both highly accurate and adaptable to new domains.

This improved architecture has significant practical applications. It can enable more precise automated content description for e-commerce and media, generate more descriptive and accurate captions for accessibility tools, and improve the visual grounding of conversational AI. By creating a more effective bridge between vision and language, this project paves the way for more reliable and capable multimodal systems that can be deployed with greater confidence in real-world scenarios.


## What is the paper trying to do, and what are you planning to do?
The original MultiCapCLIP paper introduces a framework for zero-shot captioning using text-only training. While innovative, its reliance on concept prompts creates an indirect link between the image and the generated text.

In this project, we enhance the MultiCapCLIP framework by integrating a Supervised Attention Bridge (SAB). Our plan is to create a more powerful captioning model through a two-stage process:

1.**Component Foundation:** We start with the core components of MultiCapCLIP: a frozen CLIP vision encoder and a pre-trained multilingual decoder (like mBART).

2.**Supervised Bridge Training:** We introduce and train our Supervised Attention Bridge. This bridge is a Transformer-based network that takes the visual patch features from CLIP and learns to transform them into a sequence of embeddings that the language decoder can directly use as its encoder context. This bridge is the only component trained, using supervised image-caption pairs from the COCO dataset.

Our primary goal is to evaluate if this architectural enhancement improves zero-shot captioning performance. We will compare two models:

1.**Baseline MultiCapCLIP:** The original model that uses concept prompts, trained on text-only data.

2.**MultiCapCLIP-SAB (Ours):** Our enhanced model featuring the supervised attention bridge.

Both models will be evaluated on the Flickr30k dataset in a strict zero-shot setting (i.e., without any fine-tuning on Flickr30k). Performance will be measured using standard captioning metrics (BLEU, METEOR, ROUGE, CIDEr) and retrieval metrics (Recall@K). This experiment will demonstrate whether the supervised training of the bridge component leads to better generalization on a completely unseen dataset.



# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/[presentation.pptx](https://github.com/BRAIN-Lab-AI/MultiCapCLIP-AR-Architectural-Enhancements-and-Arabic-Fine-Tuning-for-Multilingual-Visual-Captioning/blob/main/MultiCapCLIP-SAB-Enhanced-Visual-Language-Grounding.pptx))
- **Report:** [Project Report](https://github.com/BRAIN-Lab-AI/MultiCapCLIP-AR-Architectural-Enhancements-and-Arabic-Fine-Tuning-for-Multilingual-Visual-Captioning/blob/main/MultiCapCLIP_SAB__A_Supervised_Attention_Bridge_for_Enhanced_Visual_Language_Grounding__Copy_123456.pdf)

### Reference Paper
- [(ACL'2023) MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning](https://aclanthology.org/2023.acl-long.664/)

### Reference Dataset
- [MS COCO Dataset](https://cocodataset.org/#home)

### Dataset Setup
To reproduce our experiments, you need the **MS COCO 2014** dataset (val split + annotations).  
We provide a simple script to download and extract everything automatically.

 ```bash
mkdir -p /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/

wget -c http://images.cocodataset.org/zips/val2014.zip \
    -P /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/

wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip \
    -P /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/

unzip -o /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/val2014.zip \
    -d /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/

unzip -o /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/annotations_trainval2014.zip \
    -d /content/drive/MyDrive/MultiCapCLIP/data/MSCOCO/
 ```

## Project Technicalities

### Terminologies

-**Vision Encoder:** A frozen CLIP ViT-B/16 model that extracts patch-level visual features from an image.
-**Text Encoder:** The text encoder from a pre-trained multilingual model like mBART, used to create text embeddings.
-**Decoder:**  A frozen, pre-trained multilingual decoder (e.g., mBART) that generates captions from a sequence of embeddings.
-**Supervised Attention Bridge (SAB):** A multi-layer Transformer network that is trained to map the visual features from the Vision Encoder to the input space of the Decoder. This is the core architectural enhancement.
-**Zero-Shot Evaluation:** The process of evaluating a model on a dataset (e.g., Flickr30k) that it has never seen during its training phase (which was done on COCO).
-**Metrics:** BLEU(1,2,3,4), METEOR, ROUGE-L, CIDEr (for captioning evaluation).


### Research Gaps
-**Indirect Vision-Language Connection:** The original MultiCapCLIP relies on an indirect connection via "concept prompts," which can act as a bottleneck and prevent fine-grained understanding of visual details.

-**Lack of Explicit Visual Grounding:** The text-only training of the original model means it never explicitly learns to ground parts of a caption to specific regions of an image.

-**Performance Ceiling of Zero-Shot Methods:** Purely zero-shot models often lag behind supervised models in caption quality because they lack targeted training on paired image-text data.



### Problem Statements
-**Problem 1:** The quality of generated captions in zero-shot models is limited by the coarse and indirect mapping between visual input and textual output.

-**Problem 2:** Existing models struggle to generalize effectively to unseen datasets in a zero-shot manner, showing a significant performance drop compared to their performance on in-domain data.

-**Problem 3:** There is a need for an architecture that can leverage the strengths of large pre-trained models (like CLIP and mBART) while allowing for targeted, efficient, and supervised enhancement of the vision-language connection.



### Loopholes or Research Areas
Despite the progress achieved by vision–language models such as CLIP and MultiCapCLIP, several research gaps remain unresolved:

- **Evaluation Metrics:** Existing metrics (BLEU, CIDEr, ROUGE-L, METEOR) do not fully capture semantic grounding or visual relevance, limiting the ability to robustly judge caption quality.
- **Output Consistency:** Zero-shot captioning models often produce unstable or generic captions when evaluated on higher-resolution images or unseen datasets.
- **Computational Resources:** Full fine-tuning of large-scale models remains expensive and inaccessible for many research settings.

These limitations highlight the need for a lightweight, computationally efficient architecture that can improve grounding without retraining the entire model.


### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
### 1. **Hybrid Training Strategies**
Finding the optimal balance between frozen large-scale models and selectively trainable components. Our project focuses on updating only the attention bridge while keeping both the vision encoder (CLIP ViT-B/16) and text decoder (mBART) frozen.

### 2. **Bridge Architecture Exploration**
The configuration of the supervised attention bridge — including the number of layers, attention heads, hidden dimensions, and number of query tokens — directly impacts the model’s grounding capability. This architecture becomes a focused research area in our work.

### 3. **Cross-Dataset Generalization**
To evaluate true generalization, we adopt strict zero-shot evaluation settings. The model is trained exclusively on COCO and tested on unfamilar datasets like Flickr30k without any domain adaptation.




1.Introduce a Supervised Attention Bridge: Instead of relying on indirect concept prompts, we propose a dedicated Transformer-based network that directly learns to map CLIP visual features to the language decoder's input space.

2.Adopt a Hybrid Training Approach: We will keep the powerful, large-scale vision and language models frozen, and only train the lightweight bridge component. This is computationally efficient and reduces the risk of catastrophic forgetting.

3.Implement Strict Zero-Shot Evaluation: To prove the generalization power of our approach, we will train the model exclusively on the COCO dataset and evaluate its performance on the completely unseen Flickr30k dataset, ensuring a fair and challenging test of its capabilities.




### Proposed Solution: Code-Based Implementation
This repository contains a fully reproducible implementation of **MultiCapCLIP-SAB**, built on top of the official MultiCapCLIP architecture.

### ✔ Frozen Pretrained Models
- **CLIP ViT-B/16** for extracting high-quality visual features  
- **mBART** for multilingual caption generation  

Both remain frozen during training to reduce computation and preserve pretrained knowledge.

### ✔ Supervised Attention Bridge (SAB)
A trainable Transformer-based module that:
- Ingests CLIP embeddings  
- Injects learnable query tokens  
- Produces refined representations aligned with the language decoder  

### ✔ Targeted Training Pipeline
A PyTorch-based training script that:
- Loads image–caption pairs from COCO  
- Feeds CLIP embeddings into the SAB  
- Trains the SAB with cross-entropy loss  
- Keeps all other components frozen  

This ensures high efficiency and stable optimization.


### Key Components
-**'train_supervised_attention_bridge.py':** The main script for training the attention bridge on the COCO dataset. It handles data loading, model setup, and the training loop.

-**'model.py':** Contains the implementation of the AttentionBridge network, including the Transformer layers and cross-attention pooling mechanism.

-**'dataset.py':** Utility script for creating the COCO image-caption dataset, preparing images and tokenizing captions.

-**'flickr30k_all_in_one_eval.py':** Script for performing zero-shot evaluation on the Flickr30k dataset, calculating captioning and retrieval metrics.



## Model Workflow
The workflow of the MultiCapCLIP-SAB model is as follows:

1.**Input:**

•*Image:* An input image is fed into the frozen CLIP vision encoder.

•*Visual Features:* CLIP extracts a sequence of patch features (e.g., [batch, 50, 512]).



2.**Bridge Processing:**

•*Mapping:* The visual patch features are passed to the Supervised Attention Bridge.

•*Transformation:* The bridge, which has been trained on COCO, processes these features through its self-attention layers and cross-attention pooling to produce a fixed-length sequence of embeddings (e.g., [batch, 8, 1024]) that are now in the language model's representational space.



3.**Caption Generation:**

•*Decoding:* The embeddings from the bridge are fed as the encoder output to the frozen mBART decoder.

•*Generated Caption:* The decoder uses these conditioned embeddings to generate a descriptive caption for the image in a zero-shot manner.


How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/MultiCapCLIP-SAB.git
    cd MultiCapCLIP-SAB
    ```
!python /content/drive/MyDrive/MultiCapCLIP/train_supervised_attention_bridge.py
 2. **Set Up the Environment**
Create a virtual environment and install the required dependencies:

 Upgrade core Python tools
 ```bash
pip install -U pip setuptools wheel


Install Core Dependencies

 PyTorch (CUDA 11.7 build)
pip install torch==1.13.1+cu117 \
            torchvision==0.14.1+cu117 \
            torchaudio==0.13.1+cu117 \
            -f https://download.pytorch.org/whl/cu117/torch_stable.html

Transformers + Tokenizers (stable versions used in the project)
pip install "transformers==4.36.2" "tokenizers==0.15.2" --only-binary=:all:
pip install git+https://github.com/openai/CLIP.git
pip install "ruamel.yaml<0.18.0"
pip install ftfy regex tqdm sentencepiece
pip install opencv-python
pip install timm
pip install datasets accelerate
pip install nltk rouge-score
pip install decord
pip install wget
pip install stanfordcorenlp
pip install git+https://github.com/salaniz/pycocoevalcap@master
pip install matplotlib pillow
pip install -r requirements-2.txt
   ```
3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train_supervised_attention_bridge.py \
    --config configs/train_bridge.yaml \
    --output_dir output/supervised_attention_bridge/

    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate captions.
    ```bash
    python flickr30k_all_in_one_eval.py \
    --checkpoint checkpoints/best.pt \
    --image_path path/to/image.jp

    ```


## Acknowledgments
- **Open-Source Communities:** I acknowledge the open-source contributors behind PyTorch, Hugging Face Transformers, OpenAI CLIP, and the Python ecosystem (NumPy, PIL, torchvision), whose tools formed the foundation of this project.

- **Individuals:** Special thanks to Dr. Muzammil Behzad for his guidance and continuous support. I am also grateful to my family and friends for their encouragement throughout this work.

- **Resource Providers:** I appreciate the computational support provided by Google Colab and the storage services of Google Drive. I also acknowledge the creators of the MS COCO and Flickr30k datasets for making high-quality vision–language data available.
