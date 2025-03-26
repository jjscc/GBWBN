# README
This is the official code for our CVPR 2025 publication: Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry.

This code will be released soon.


In case you have any problem, do not hesitate to contact me 2932723775@qq.com.

## Experiments

The implementation is based on the official code of 
    
- *Riemannian batch normalization for SPD neural networks* [[Neurips 2019](https://papers.nips.cc/paper_files/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html)] [[code](https://papers.nips.cc/paper_files/paper/2019/file/6e69ebbfad976d4637bb4b39de261bf7-Supplemental.zip)].
- *Riemannian Residual Neural Networks* [[Neurips 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c868aa7437dc9b29e674cd2e25689021-Abstract-Conference.html)] [[code](https://github.com/CUAI/Riemannian-Residual-Neural-Networks)].

### Dataset
We further release our preprocessed HDM05 and MAMEM-SSVEP-II datasets in the folder './GBWBN_SPDNet/data' and './GBWBN_SPDNet/data'. 

### Running experiments
The code of experiments on SPDNet in the folder './GBWBN_SPDNet', while the experiments on RResNet in the folder './GBWBN_RResNet'.
To train and test the experiments on the HDM05 and MAMEM-SSVEP-II datasets, run this command:

```train and test
python Hdm05.py
python mamem.py
```
## Reference
```bash
@inproceedings{RResNet,
  title={Riemannian residual neural networks},
  author={Katsman, Isay and Chen, Eric and Holalkere, Sidhanth and Asch, Anna and Lou, Aaron and Lim, Ser Nam and De Sa, Christopher M},
  booktitle={NeurIPS},
  year={2024},
}
```

```bash
@inproceedings{spdnetbn,
  title={Riemannian batch normalization for {SPD} neural networks},
  author={Brooks, Daniel and Schwander, Olivier and Barbaresco, Fr{\'e}d{\'e}ric and Schneider, Jean-Yves and Cord, Matthieu},
  booktitle={NeurIPS},
  year={2019}
}
```


