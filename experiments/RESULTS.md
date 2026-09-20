# Master Benchmark Results

## 1. Validation Set (122 frames)

| Model | Macro Dice | Mean IoU | ASSD | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| TopoNet Baseline | 54.64% | 40.94% | 49.64 px | ~86.4 ms |
| TopoNet w/o L_per | 58.87% | 46.69% | 38.66 px | ~86.4 ms |
| TopoNet w/o L_cl | 58.63% | 46.37% | 32.01 px | ~86.4 ms |
| TopoNet w/o L_per & L_cl | 57.44% | 45.86% | 43.32 px | ~86.4 ms |
| TopoNet w/o BTF | 57.92% | 46.03% | 37.47 px | ~86.4 ms |
| TopoNet Full | 59.79% | 47.38% | 29.27 px | 11.6 FPS (86.4 ms) |
| Mask2Former Baseline | 66.56% | 53.63% | 25.69 px | 10.5 FPS (94.9 ms) |
| Mask2Former Full Attention | 66.17% | 53.18% | 23.45 px | 10.6 FPS (94.1 ms) |
| Mask2Former Single Scale | 65.35% | 52.22% | 35.77 px | 10.6 FPS (94.1 ms) |
| Mask2Former w/o Self Query | 66.34% | 53.12% | 22.54 px | 10.6 FPS (94.5 ms) |
| Mask2Former-Bezier | 59.20% | 44.97% | 30.98 px | 14.5 FPS (68.9 ms) |
| Mask2Former-Bezier-SingleScale | 58.59% | 44.56% | 29.89 px | 15.6 FPS (64.0 ms) |
| Mask2Former-Bezier-Landmark | 57.40% | 43.33% | 29.87 px | 15.1 FPS (66.1 ms) |
| Mask2Former Junction-MLP-Steered | **71.98%** | **59.67%** | 35.06 px | 6.5 FPS (154.0 ms) |
| Mask2Former Junction-Heatmap-Steered | 71.51% | 58.89% | 36.70 px | 6.5 FPS (154.1 ms) |

---

## 2. Test Set (109 frames)

| Model | Macro Dice | Mean IoU | ASSD | Inference Speed |
| :--- | :---: | :---: | :---: | :---: |
| TopoNet Full | 65.19% | 50.56% | 28.07 px | 11.6 FPS (86.4 ms) |
| BCRNet SOTA (Published) | 69.57% | - | - | - |
| Mask2Former Baseline | 65.73% | 53.05% | 28.43 px | 10.6 FPS (94.4 ms) |
| Mask2Former Full Attention | 65.68% | 52.94% | 26.86 px | 10.7 FPS (93.3 ms) |
| Mask2Former Single Scale | 64.58% | 51.54% | 36.65 px | 10.7 FPS (93.8 ms) |
| Mask2Former w/o Self Query | 64.12% | 51.05% | 24.02 px | 10.6 FPS (93.9 ms) |
| Mask2Former-Bezier | 55.74% | 41.64% | 38.46 px | 16.2 FPS (61.7 ms) |
| Mask2Former-Bezier-SingleScale | 56.12% | 42.08% | 35.24 px | 16.7 FPS (59.9 ms) |
| Mask2Former-Bezier-Landmark | 55.52% | 41.43% | 34.23 px | 16.7 FPS (60.0 ms) |
| Mask2Former Junction-MLP-Steered | **69.84%** | **57.64%** | 34.61 px | 6.5 FPS (153.1 ms) |
| Mask2Former Junction-Heatmap-Steered | **69.84%** | 57.42% | 37.49 px | 6.7 FPS (149.1 ms) |

Insight from analysing the result and the patient 40 difficult cases:
- TopoNet is way more computational heavier than the Mask2Former due to the loss functions: the clDice and the Betti Loss Matching due to they cannot be done parralelly
- The TopoNet reported from the paper and the Mask2Former has similar performance on both the test and the validation set.
- From the Mask2Former Ablation, we can see that the multiscalle and the pretrained weights of the model on the ADE20K (27000 images) play crucial roles in the performance while the mask attention makes no different. 
- Another interesting finding is that for the hardcases of the patient 40, which related to geometry deformation of the liver, the model with single scale actually perform quite well compare to other images. We could check images: 08730, 09000, 08940, 08790, 09330 to see the results.

September 18, 2026
- Checking if the Training dataset has the geometry deformation data enough for the model to understand the different when the liver contract or flipped
- We should consider about the scale of the model feature map before its going through the attention mechanism
- Using the pretrained model from ADE20K.

September 19, 2026
- From my observation, the single scale and the multiscale does not make much different
- We implemented a method to learn about the geometry deformation, the first one was the center of mass of each landmark which doesnt work well since they might be saturated
- We are moving on to the second method which is the joint point anchor, this method is the joint point of the landmarks and those point would be the guide (queries) to help the model changes its prediction based on the geometric discoveries of those new Point Queries.

September 20, 2026
- The result of using an addition query for diagnosing points anchor in the liver landmark is remarkable, pushing the validation score to 71.98 and the test score to 69.84 which is currently state of the art
- However using the MLP to predict the coordinate is not geometrically generalized, proof showed that the predicted coordinate is unrelated to the actual anchor points, but the queries does contain information about the deformation of the liver so it helped the model perform better
- We move on by using the spatial heatmap prediction rather than the MLP because we see that there are many cases that not all the 4 anchor points existed, therefore we create a heatmap for each of the anchor point for the model to learn
