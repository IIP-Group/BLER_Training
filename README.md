# Bit Error and Block Error Rate Training for ML-Assisted Communication

You are using simple simulation scripts that implement the two experiments in the paper
*R. Wiesmayr, G. Marti, C. Dick, H. Song, and C. Studer
“Bit Error and Block Error Rate Training for ML-Assisted
Communication,” arXiv:2210.14103, Mar. 2023*, available at https://arxiv.org/abs/2210.14103

The simulations are implemented with [NVIDIA Sionna](https://github.com/NVlabs/sionna) Release v0.12.1 and own extensions.

Parts of the code are also based on
- *R. Wiesmayr, C. Dick, J. Hoydis, and C. Studer,
“DUIDD: Deep-unfolded interleaved detection and decoding
for MIMO wireless systems,” in Asilomar Conf.
Signals, Syst., Comput., Oct. 2022.*
- *C. Studer, S. Fateh, and D. Seethaler, “ASIC Implementation of Soft-Input Soft-Output MIMO Detection Using MMSE Parallel Interference
Cancellation,” IEEE Journal of Solid-State Circuits, vol. 46, no. 7, pp. 1754–1765, July 2011.*

If you are using this simulator (or parts of it) for a publication, please consider citing the above-mentioned references and clearly mention this in your paper.

## Running simulations
Please have your Python environment ready with NVIDIA Sionna v0.11, as the code was developed and tested for this version.

The simulation scripts are located in `./scr`. The upper parts contain multiple simulation parameters, which can be modified at will.
After the model definition part, the scripts first train the corresponding signal processing models for all loss functions under test,
and then run a performance benchmark. At the end, the bit error rate and block error rate curves are plotted and saved to files.

Before running the simulations, please create the following directories:
- `./data/weights` where the trained model weights are saved
- `./results` for the simulation results (BER and BLER curves), which are saved as `.csv` and `.pickle` files

## Version history

- Version 0.1: [wiesmayr@iis.ee.ethz.ch](wiesmayr@iis.ee.ethz.ch) - initial version for GitHub release
- Version 0.2: [wiesmayr@iis.ee.ethz.ch](wiesmayr@iis.ee.ethz.ch) - updated simulations:
  - Training on information bits
  - Changed MIMO BS ULA antenna yaw angle to point to sector center 
- Version 0.3: [wiesmayr@iis.ee.ethz.ch](wiesmayr@iis.ee.ethz.ch) - updated SNR deweighting algorithm:
  - Sampling the SNR range only on a low number of points to avoid empty accumulated loss bins
