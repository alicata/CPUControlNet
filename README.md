# CPUControlNet - Model Preserving Fine-Tuning with Adaptations
A modifed fork of ControlNet that works in CPU with no NVIDIA GPU dependency. Useful for hacking, testing, experimenting on laptop with tools complementary to a GPU implementation. ControlNet is a Domain Adaptation method applicable to Model-Preserving Fine-Tuning. 

# Motivation
Why? Sometimes experiments are more convenient executed on a laptop, but for many laptops the available GPU VRAM is only 4GB or less and the model might require 8GB GPU VRAM. 
This CPU port allows using host memory instead, and load the weights fully in a laptop with modest GPU.

# Setup & Weights
Follow ControlNet install instruction.

Download the control_sd15_canny.pth checkpoint and move checkopint file in the ./model folder. 
```
https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_canny.pth
```

# Test Usage
Test harcoded to load an image from /data/test.jpg,
```bash
python ./cpu_canny2image.py
```

# Model-Preserving Domain Adaptation

![img](github_page/he.png)

## Conditional Control: Preserve Primary Model through Locked Layer Versions and Trainable Layer Copies 
ControlNet adds conditional control to existing models without compromising the integrity of the original model. By utilizing a dual structure of "locked" (original) and "trainable" copies, it ensures that the primary model remains unaffected while allowing for domain-specific or conditional **adaptations** via the trainable copy.

## Zero Convolution: Neutral State Start
The "zero convolution" is a 1×1 convolution with both weight and bias initialized as zeros
* serves as a mechanism to start from a neutral state
* initially ControlNet doesn't introduce any modifications to the outputs
* As training progresses, these zero convolutions adapt and learn to introduce the desired changes

## Approach advantages:

* Preservation of Original Model: The "locked" version ensures that the pre-trained, production-ready model is not tampered with.
* Adaptability: The "trainable" copy allows for domain-specific or conditional modifications.
* Efficiency: Since no layer is trained from scratch and only fine-tuning is performed, training can be faster and feasible even on small-scale devices.
* Flexibility: The architecture supports merging, replacing, or offsetting of models, weights, or layers, providing flexibility in model management and deployment.

## Applications
* Domain adaptation through conditional modeling
* Applications where maintaining the integrity of the original model is crucial.
* Fine-tuning and adaptation without the risk of **forgetting** or deteriorating the original model's capabilities.


# Citation

    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2023},
      eprint={2302.05543},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

[Arxiv Link](https://arxiv.org/abs/2302.05543)
