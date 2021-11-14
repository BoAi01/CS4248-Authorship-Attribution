# CS4248-Authorship-Attribution

## Env setup
PyTorch needs to be compiled with the correct CUDA version. Example for CUDA 11: </p>
```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch ``` </p>
For more info, see  [here](https://pytorch.org/).  </p>

Manually install APEX: 
```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```
For the other depdencies, manual installation is required. 

## Training
One-time preparation of dataset </p>
``` python prepare_dataset.py ```

Start training with</p>
``` python main.py --dataset <dataset name in ['imdb62', 'enron', 'imdb', 'blog']>  --gpu <gpu indices, optional> --samples-per-author <# of samples per author> ```
