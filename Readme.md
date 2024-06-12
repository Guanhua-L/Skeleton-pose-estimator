# Skeleton Pose Estimator (SPE)

This project aims to use neural networks to recognize human skeletons from mmWave point cloud data, which protects privacy, and further predict human activities from the skeletons.

## Problem Statement
According to past experiments, predicting human activities from accurate skeletons (ground truth) achieves higher accuracy than using point clouds directly. However, the current SPE is not accurate in fine-grained skeleton prediction, which affects subsequent human activity prediction results.

## Solution
To address the inaccuracy, we use a Transformer to replace the original CNN+LSTM in SPE. The self-attention mechanism of the Transformer captures temporal features more effectively. Additionally, we consider uncertainty within the Transformer to handle the cold-start problem in machine learning, making the model more reliable.

## Conclusion
This project implemented the use of a Transformer to predict human skeletons from mmWave point clouds. Although the final results have not yet surpassed the original SPE method, the model converges more stably in the early epochs of training and requires less training time. If further optimization of the Transformer model can capture more spatial features, it should outperform the SPE results.

## Future Work
To extract more features, further research is needed on:

Embedding for point clouds to capture more spatial features.
Uncertainty in Transformers to further improve the accuracy of skeleton prediction and the reliability of the model.