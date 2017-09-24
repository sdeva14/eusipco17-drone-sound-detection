# Empirical Study of Drone Sound Detection in Real-Life Environment with Deep Neural Networks, EUSIPCO17
This project contains Python experimental implementation used for research paper published in EUSIPCO17.

Original paper: https://arxiv.org/abs/1701.05779

Please use the following citation:

```
@inproceedings{jeon2017empirical,
  author       = {Jeon, Sungho and Shin, Jong-Woo and Lee, Young-Jun and Kim, Woong-Hee and Kwon, YoungHyoun and Yang, Hae-Yong},
  title	       = {Empirical Study of Drone Sound Detection in Real-Life Environment with Deep Neural Networks},
  year	       = 2017,
  booktitle    = {Signal Processing Conference (EUSIPCO), 2017 25th European},
  organization={IEEE}
}
```

Contact person: Sungho Jeon, sdeva14@gmail.com

## Project structure
  * `dd_rnn.py` -- Experiments using RNN (Tensorflow 0.12) 
  * `dd_cnn.py` -- Experiments using CNN (Tensorflow 0.12)
  * `dd_gmm.py` -- Experiments using GMM (Scikit-learn 0.18)


## Requirements
* Software dependencies
  * Python 2.7
  * Tensorflow 0.12
  * Scikit-learn 0.18
  * Librosa 0.4.3
  
Please note that our experimental audio data cannot be shared due to security contact of research project. However, I described how to augment audio dataset in the paper.
