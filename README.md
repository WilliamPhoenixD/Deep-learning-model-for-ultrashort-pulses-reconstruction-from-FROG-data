# Deep-learning-model-for-ultrashort-pulses-reconstruction-from-FROG-data

## Problem description

The phase-retrieval problem appears when measuring ultrashort laser pulses. These pulses last less than a picosecond (~10^{-12} s), so no electronic measurement can directly capture the electric field of the pulse in the time or frequency domain. In recent years, optical techniques have advanced to the point of measuring on the attosecond scale (~10^{-18} s).

An optical measurement that determines the electric field of an ultrashort pulse is the **SHG-FROG** trace (Second Harmonic Generation – Frequency Resolved Optical Gating).

![Experimental setup](Figure1(a).png)

*Figure 1: Experimental setup for measuring SHG-FROG traces. Image credit: Rick Trebino.*

We will not go into experimental minutiae. The important part is that the pulse is autocorrelated with a delayed copy of itself and passed through a second-harmonic generation crystal, which produces a signal proportional to the product of the two incident pulses. Variants of this setup exist, but the objective is always to send the crystal’s signal through a spectrometer and obtain the **SHG-FROG trace** of the incident pulse, which is

$$
\tilde{T}(\omega,\tau)
= \left| \int_{-\infty}^{\infty} E(t)\,E(t-\tau)\,e^{i\omega t}\,dt \right|^{2}.
$$

This expression contains information in both time and frequency. For each delay \(\tau\) it tells us the spectrum obtained in the frequency domain (similar, though not identical, to a spectrogram in acoustics).

If we translate the pulse in time, conjugate it, or apply a global phase shift, the SHG-FROG trace remains the same. A standard manipulation converts the expression above into a two-dimensional Fourier transform:

$$
\tilde{T}(\omega,\tau)
= \left| \iint_{-\infty}^{\infty}
\bar{E}_{\text{sig}}(t,\Omega)\,e^{-i\omega t - i\Omega \tau}\,dt\,d\Omega \right|^{2},
$$

where \(\bar{E}_{\text{sig}}(t,\Omega)\) is the Fourier transform (with respect to \(\tau\)) of
\(E_{\text{sig}}(t,\tau) = E(t)\,E(t-\tau)\), commonly referred to as the **signal operator**.

The key takeaway is that \(\tilde{T}\) is the squared modulus of a 2-D Fourier transform of \(E_{\text{sig}}\). That means we know the **magnitudes** of the Fourier coefficients but not their **phases**—the classic **phase-retrieval problem**. For 1-D Fourier transforms this problem is ill-posed in general, but in higher dimensions it can be solved under practical assumptions.

![Equations page](sandbox:/mnt/data/c13a4f8d-9180-4c35-9b2f-978d6676ff89.png)

---

## Summary

In summary, theory tells us that from a SHG-FROG trace we can recover the time-domain electric field that generated it. The electric field can be written as intensity and phase:

$$
E(t) = \sqrt{I(t)}\,e^{i\phi(t)}.
$$

The experiment produces an \(N \times N\) real-valued array \(\tilde{T}_{mn}\) with \(m,n = 0,\ldots,N-1\). The objective is to retrieve the \(2N\) unknowns that describe \(E(t)\): either its real and imaginary parts, or equivalently amplitude and phase.

![Summary figures](sandbox:/mnt/data/41d033ee-19a7-4508-8480-7914381293d1.png)

*Figure 2: Goal—transform the \(N\times N\) SHG-FROG trace into the \(2N\) values that define the time-domain electric field.*

This problem can be addressed using dedicated phase-retrieval algorithms such as **GPA**, **PIE**, or **COPRA**. See ultrafast pulse-retrieval resources for implementations and further details.

---

## Problem from NN perspective

The aim of this project is to use **deep neural networks** to solve the phase-retrieval problem. The network’s task is to **invert** the mapping from the \(N\times N\) SHG-FROG trace (real numbers) to the \(2N\) real numbers representing the electric field’s real and imaginary parts.

![NN perspective](sandbox:/mnt/data/774e8550-833c-4995-af65-0416a7c1851b.png)

*Figure 3: Schematic of the input and output layers of the deep neural network that performs pulse retrieval.*

For a deeper discussion of the methodology and experiments, see the `reports/` folder.

---

## Author

This code was developed by **Xiangfeng(William) Deng** as part of a summer research program at the **Rice University**.

For questions or comments, contact: phoenix.william.d@gmail.com

---

## Reference

Tom Zahavy, Alex Dikopoltsev, Daniel Moss, Gil Ilan Haham, Oren Cohen, Shie Mannor, and Mordechai Segev, "Deep learning reconstruction of ultrashort pulses," Optica 5, 666-673 (2018)

No reference code provided by paper author

## License
Will be upload soon

