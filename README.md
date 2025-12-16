# TREWA


This is the official implementation of the paper: Token Reduction in Vision Transformers via Discrete Wavelet Decomposition

## Abstract 

Vision Transformers (ViTs) have achieved remarkable performance across various computer vision tasks, yet their high computational cost remains a significant limitation in many real-world scenarios. As noted in prior studies, decreasing the number of tokens processed by the attention layers of a ViT directly reduces the required operations. Building on this idea, and drawing inspiration from signal processing, we reinterpret the token embeddings of a ViT layer as a signal, which allows us to apply the Discrete Wavelet Transform (DWT) to separate low- and high-frequency components. Guided by this insight, we present Token REduction via WAvelet decomposition (TREWA), a token-pruning strategy built upon DWT. For each image in a batch, TREWA selects a pruning level by comparing that image’s attention entropy with the batch one.  It then applies the DWT to the token embeddings and forwards only the low-frequency coefficients (i.e., those capturing the image’s main semantic structure) to the next attention layer, discarding 50–75\% of the tokens. We evaluate TREWA on four benchmark datasets in both pre-trained and training from scratch settings, comparing it against state-of-the-art pruning methods. Our results show a superior trade-off between accuracy and computational efficiency, validating the effectiveness of our frequency-domain token pruning strategy for accelerating ViTs.
