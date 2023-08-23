![PentaMind Logo](https://github.com/stasaki/PentaMind/raw/main/pentamind-logo.png)

--------------------------------------------------------------------------------

PentaMind is a deep learning model for predicting cognitive performance from an image of intersecting pentagons. The model is trained on data collected at Rush Alzheimer's Disease Center (RADC) and is implemented using PyTorch.

## 📜 Link to the Paper

If you're interested in the details of our research, you can read our full paper:

[**Explainable deep learning approach for extracting cognitive features from hand-drawn images of intersecting pentagons**](https://www.nature.com/articles/s41746-023-00904-w)

By _Tasaki, S., Kim, N., Truty, T. et al._

Published in: _npj Digital Medicine volume 6, Article number: 157 (2023)_


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing.

### Prerequisites
Before running the code in this repository, you must have the following prerequisites installed on your system:

- NVIDIA GPU with CUDA support

This repository requires an NVIDIA GPU with CUDA support to run the code. If you do not have an NVIDIA GPU, the code may still run, but it will be significantly slower. Please refer to the NVIDIA website for more information on supported GPUs and CUDA support.

- Conda package manager

This repository uses Conda as the package manager for managing dependencies. Conda is a cross-platform, open-source package manager used for scientific computing, data science, and machine learning.
To install Conda on your system, please follow the instructions on the [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) website.


### Installing
1. Clone the repository:
```bash
git clone https://github.com/stasaki/PentaMind.git
```

2. Install the required packages:
```bash
conda env create -f environments/environment.yaml -n pentamind
conda activate pentamind
```

3. Download pre-trained models:
```bash
mkdir models
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-1_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-2_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-3_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-4_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-5_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-6_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-7_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-8_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-9_model.pth -P models/
wget https://huggingface.co/stasaking/pentamind/resolve/main/models/vgg19_bn-10_model.pth -P models/
```

### Running the model
- Train PentaMind model
```bash
python functions/PentaMind.py --model_name vgg19_bn --use_pretrained True --pheno_file_dir data/phenotype/ --image_file_dir data/images/ --output_dir results/PentaMind/ --optim sgd --batch-size 32 --epochs 90 --lr 0.001 --momentum 0.9 --weight-decay 0.0001
```

- Use PentaMind model to make a prediction
```bash
python functions/PentaMind-pred_ave.py --model_name vgg19_bn --model_loc models/vgg19_bn-1_model.pth --pheno_file data/phenotype/test.txt --image_file_dir data/images/ --output_dir results/PentaMind-pred_ave/ --batch-size 32
```

- To run the Intersecting Pentagons Simulator, open the `simulate_pentagon.ipynb` file on [Jupyter](https://jupyter.org/). You can then visualize the results by running the `simulate_pentagon.Rmd` file on [Rstudio](https://posit.co/download/rstudio-desktop/).


## Dataset
All data in the 'data' directory is purely synthetic. It is primarily for indicating the data format. The actual intersecting pentagon images and associated labels used in this study can be requested at the [RADC Resource Sharing Hub](https://www.radc.rush.edu/).

## License
The code of this project is licensed under the [BSD 3-Clause License](LICENSE). The pre-trained weights based on ROS/MAP/MARS studies are licensed under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). The pre-trained models provided in this code may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case.

## Acknowledgments
We would like to express our sincere thanks to all the participants who took part in ROS, MAP, and MARS studies. Your contributions have been invaluable to our research, and we could not have achieved our results without your participation.

We would also like to thank the machine learning community for providing easy access to pre-trained models. The availability of high-quality models has enabled us to build on previous work and make progress on our research goals.

Once again, thank you to all the individuals and organizations who have contributed to this project. Your efforts are greatly appreciated and have made a real difference in our work.

