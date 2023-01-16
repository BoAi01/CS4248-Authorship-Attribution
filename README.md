# CS4248-Authorship-Attribution

This is the course project of CS4248 Natural Language Processing at the National University of Singapore, instructed by Prof Hwee Tou Ng (AY2021/22). Our project was ranked the second highest in the cohort, but the repo is no longer under maintanance.

If you are intersted in the authorship attribution, you are welcome to check out our publication out of this project. See [this repo](https://github.com/BoAi01/Contra-X) for more details. 

## Env setup
PyTorch needs to be compiled with the correct CUDA version. Example for CUDA 11: </p>
```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch ``` </p>
For more info, see  [here](https://pytorch.org/).  </p>

Manually install APEX: 
```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

Install the rest of dependencies via </p>
```pip install -r requirements.txt```

## Training
One-time preparation of dataset </p>
``` python prepare_dataset.py ```

Start training with</p>
``` python main.py --dataset <dataset name in ['imdb62', 'enron', 'imdb', 'blog']> ```
