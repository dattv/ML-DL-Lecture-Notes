# Definition and Terminology in traditional Image processing methods
## Gabor filter
A Gabor filter (in image processing), named after Dennis Gabor, is a linear filter used for texture analysis, which means that it basically analyses whether there are any specific frequency content in the image in specific directions in a localized region around the point or region of analysis.
Frequency and orientation representations of Gabor filters are claimed by many contemporary vision scientists to be similar to those of the human visual system, though there is no empirical evidence and no functional rationale to support the idea.
They have been found to be particularly appropriate for texture representation and discrimination. In the spatial domain, a 2D Gabor filter is a Gaussian kernel function modulated by a sinusoidal plane wave.
Some authors claim that simple cells in the visual cortex of mammalian brains can be modeled by Gabor functions. Thus, image analysis with Gabor filters is thought by some to be similar to perception in the human visual system.

1. For documents image processing, Gabor features are idiel for identifying the scipt of a word in a multilingual document.

2. Gabor filters with different frequencies and with oridentations in different directions have been used to localize an extract text-oly regions forom complex document images (both gray and color), since text is rich in high frequency components. whereas pictures are relatively smooth in nature.

3. It has also been applied for facial expression recognition.

4. Gabor filters have also been widely used in pattern analysis application. For example, it has bene used to sutdy the directionslity distribution inside the posous spongy trabecular bon in the spine.

## Gradient 
```$\nabla_{\mathbf{v}}f\left(x\right) = \mathbf{x}\cdot\nabla f\left(\mathbf{x}\right) = v_x\frac{\partial f}{\partial x}\left(\mathbf{x}\right) + v_y\frac{\partial f}{\partial y}\left(\mathbf{x}\right)$```

``$f_x = h_x*f$``
``$f_y = h_y*f$``

``$h_x = \frac{1}{2}[1 0 -1]$``
``$h_y = \frac{1}{2}[1 0 -1]^T$``

## point set registration
[point set registration](https://en.wikipedia.org/wiki/Point_set_registration) In computer vision, point set registration also known as point matching, is the process of finding a spatial transformation that aligns two point set. The purpose of finding such a transformation includes mergin multiple datasets into a globally consistent model. and mapping a new measurement to a known dataset to idientify features or to estimate tis pose. A oint set may be raw data from 3D scanning or an array of rangefinders. For use in image processing an feature-based image resgistration, a point set may be a set of features obtained by feature extraction from an image, for eample corner detection. Point set registration is used in optical character recognition. augmented reality and aligning data from magnetic resonance imagegin with computer aided tomography scans.

1. Rigid Registration

2. Non-Rigid Registration




