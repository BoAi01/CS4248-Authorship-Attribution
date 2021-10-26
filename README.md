# CS4248-Authorship-Attribution

Most dependencies can be installed by 

<code>python -m pip install -r requirements.txt</code>


Except some packages that might need manual installation: 

In particular, pytorch needs to be compiled with the correct CUDA version. See [here](https://pytorch.org/). Example for CUDA 11

```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch ```
```
git clone https://github.com/NVIDIA/apex
  
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
  
pip install --upgrade tqdm

pip install transformers
  
pip install tensorboardX
  
pip install simpletransformers
```
