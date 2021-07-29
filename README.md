# nqs-tensorflow2
This repository implements the neural-network quantum states [1] with Python 3 and Tensorflow 2 library to speed-up the process with graphics processing units (GPU).
The implementation is greatly inspired by [NetKet](https://www.netket.org/) library [2].
For similar library with Tensorflow 1 library with Python 2 (which is not maintained anymore), please see the following [repository](https://github.com/remmyzen/nqs-tensorflow).

This code is used in [3,4] where we propose several transfer learning protocols to improve the scalability, efficiency, and effectiveness of neural-network quantum states.

## Requirements
This project is based on Python. The code is tested on Python 3.8.5.
These are the main library requirements for the project:
* `tensorflow==2.3.0` to run on CPUs or `tensorflow-gpu==2.3.0` to run on GPUs
* `scipy`
* `sklearn`
* `jupyter` to run the notebooks.
* `matplotlib` for plotting purposes.

It is also available as requirements.txt in the project and do
``pip install -r requirements.txt``
to install the necessary libraries.

## Usage and Examples
The different examples to run the code is available in the `notebooks` directory.

## References
[1] G. Carleo and M. Troyer, Science 355, 602 (2017)

[2] G.  Carleo,   K.  Choo,   D.  Hofmann,   J.  E.  T.  Smith,T.  Westerhout,  F.  Alet,  E.  J.  Davis,  S.  Efthymiou,I. Glasser, S.-H. Lin, M. Mauri, G. Mazzola, C. B. Mendl,E. van Nieuwenburg, O. O’Reilly, H. Th ́eveniaut, G. Tor-lai,  F.  Vicentini,  and  A.  Wietek,  SoftwareX ,  100311(2019).

[3]  Zen, R., My, L., Tan, R., Hébert, F., Gattobigio, M., Miniatura, C., Poletti, D., Bressan, S.: Finding quantum critical points with neural-network quantum states. In: ECAI 2020 - 24th European Conference on Artificial Intelligence. Frontiers in Artificial Intelligence and Applications, vol. 325, pp. 1962–1969. IOS Press (2020)

[4] Zen, R., My, L., Tan, R., Hébert, F., Gattobigio, M., Miniatura, C., Poletti, D., Bressan, S.: Transfer learning for scalability of neural-network quantum states. Physical Review E 101(5), 053301 (2020)
