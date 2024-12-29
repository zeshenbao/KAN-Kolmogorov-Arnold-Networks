# KAN-Kolmogorov-Arnold-Networks
Signals often suffer from noise due to the environment over which they are being
transmitted. To ensure reliable data transmission in communication systems, it is
therefore of importance to be able to denoise the signal and recover the true one.
Neural networks have previously been shown successful on this task, but are often
computationally expensive and well-suited for specific hardware. One promising
alternative for more CPU-based hardware could be Kolmogrov-Arnold networks. The aim of this study was to investigate Kolmogorov-Arnold Networks (KANs)
in their ability to denoise signals, and compare these to Multi-Layer Perceptrons’
ability to do so. This was done by generating discrete signals and channel matrices,
adding noise and then training the networks to denoise the data. The networks’
performance were evaluated on their test losses. From the experiments, it was concluded that both KANs and MLPs work well in denoising simple signals, even for large amounts of added noise. KANs generally performed better in denoising more complex data, as the channel matrices.

![test](results/Read_me.png)
