# **Monoaural Speech Enhancement Using a Nested U-Net with Two-Level Skip Connections**   
   
This is a repo of the paper "Monoaural Speech Enhancement Using a Nested U-Net with Two-Level Skip Connections", which is submitted to INTERSPEECH2021.   

Abstract：Capturing the contextual information in multi-scale is known to be beneficial for improving the performance of DNN-based speech enhancement (SE) models. This paper proposes a new SE model, called NUNet-TLS, having two-level skip connections between the residual U-Blocks nested in each layer of a large U-Net structure. The proposed model also has a causal time-frequency attention (CTFA) at the output of the residual U-Block to boost dynamic representation of the speech context in multi-scale. Even having the two-level skip connections, the proposed model slightly increases the network parameters, but the performance improvement is significant. Experimental results show that the proposed NUNet-TLS has superior performance in various objective evaluation metrics to other state-of-the-art models. The code of our model is available at https://github.com/seorim0/NUNet-TLS.   


## Requirements 
This repo is tested on Ubuntu 20.04.   
```
# for train
python == 3.7.9   
pytorch == 1.9.0_cu111   
scipy == 1.6.0      
soundfile == 0.10.3  
# for evaluation
tensorboard == 2.7.0   
pesq == 0.0.2       
pystoi == 0.3.3       
matplotlib == 3.3.3      
```   


## Getting started    
1. Install the necessary libraries.   
2. Set directory paths for your dataset. ([config.py](https://github.com/seorim0/NUNet-TLS/blob/main/config.py)) 
```   
# dataset path
noisy_dirs_for_train = '../Dataset/train/noisy/'   
clean_dirs_for_train = '../Dataset/train/clean/'   
noisy_dirs_for_valid = '../Dataset/valid/noisy/'   
clean_dirs_for_valid = '../Dataset/valid/clean/'   
```   
* You need to modify the `find_pair` function in [tools.py](https://github.com/seorim0/NUNet-TLS/blob/main/tools.py) according to the data file name you have.        
* And if you need to adjust any parameter settings, you can simply change them.   
3. Run [train_interface.py](https://github.com/seorim0/NUNet-TLS/blob/main/train_interface.py)   


## Results   
<p align="center"><img src="https://user-images.githubusercontent.com/55497506/161367052-fa063e0b-9be5-4492-b85c-3ff28e76f6ec.PNG"  width="750" height="800"/></p>   
<p align="center"><img src="https://user-images.githubusercontent.com/55497506/161367285-d8c93e53-bef4-43ea-9cfd-734e7c0988aa.PNG"  width="750" height="450"/></p>   

## References   
**U2-Net: Going deeper with nested u-structure for salient object detection**   
X. Qin, Z. Zhang, C. Huang, M. Dehghan, O. R. Zaiane, and M. Jagersand   
[[paper]](https://www.sciencedirect.com/science/article/pii/S0031320320302077)  [[code]](https://github.com/xuebinqin/U-2-Net)   
**A nested u-net with self-attention and dense connectivity for monaural speech enhancement**   
X. Xiang, X. Zhang, and H. Chen      
[[paper]](https://ieeexplore.ieee.org/abstract/document/9616439)  
**Time-frequency attention for monaural speech enhancement**   
Q. Zhang, Q. Song, Z. Ni, A. Nicolson, and H. Li  
[[paper]](https://arxiv.org/abs/2111.07518)  
