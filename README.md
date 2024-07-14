# DPCN 2024 Report

## EWNETs and Signal Decomposition

**Authors**: Amey Choudhary, Aryan Gupta, Keshav Gupta (Team Baby Sharks)  
**Course**: Dynamical Processes in Complex Networks, Spring 2024  
**Instructor**: Dr. Chittaranjan Hens  
**Date**: April 21, 2024

---

## Abstract
In our project, we aim to understand the novel approach for forecasting epidemics, EWNETs, as proposed in "Epicasting: An Ensemble Wavelet Neural Network for forecasting epidemics" (2023). We reproduce the same and analyze the results for it. We also experiment with other methods for signal decomposition and provide a comparative report for them.

[Link to Paper](https://www.sciencedirect.com/science/article/pii/S0893608023002939)

---

## Table of Contents
- [Understanding of the Paper](#understanding-of-the-paper)
  - [Current Methods for Epidemic Forecasting](#current-methods-for-epidemic-forecasting)
  - [Explaining EWNETs](#explaining-ewnets)
    - [MODWT Approach](#modwt-approach)
    - [EWNET Model Architecture](#ewnet-model-architecture)
    - [EWNET Model Working](#ewnet-model-working)
- [Experimentation Results of EWNETs on Datasets](#experimentation-results-of-ewnets-on-datasets)
- [Using Other Signal Decomposition Methods](#using-other-signal-decomposition-methods)
  - [Seasonal and Trend Decomposition Using Loess (STL)](#seasonal-and-trend-decomposition-using-loess-stl)
  - [Singular Spectral Analysis (SSA)](#singular-spectral-analysis-ssa)
  - [Empirical Mode Decomposition (EMD)](#empirical-mode-decomposition-emd)
  - [Hilbert Huang Transform (HHT)](#hilbert-huang-transform-hht)
  - [Empirical Wavelet Transform (EWT)](#empirical-wavelet-transform-ewt)
- [Discussion and Conclusion](#discussion-and-conclusion)

---

## Understanding of the Paper

Epidemics are widespread occurrences of infectious illnesses. As they are among the top contributors of illnesses and death worldwide, their forecasting is an essential field of research. By predicting them accurately, it can assist stakeholders in developing countermeasures to contain and minimize their spread. This paper explains why epidemic forecasting is essential, how the current methods fall short, proposes a new method called “EWNETs,” provides a mathematical base for their properties, and predicts various epidemics using EWNETs.

### Current Methods for Epidemic Forecasting

Epidemiological modeling is a centuries-old field of research. Various methods have been developed over time for accurate forecasting, including mechanistic (or deterministic) and phenomenological methods. Compartmental models, such as SIR, use simple differential equations to model epidemic growth but tend to explain disease dynamics better than predicting the spread.

To overcome this, statistical and machine learning methods have been introduced. Examples include:

**Statistical Methods**
- Random Walk
- Autoregressive Integrated Moving Average (ARIMA) & its variations
- Bayesian Structural Time Series
- Exponential Smoothing State Space

**Machine Learning Methods**
- Artificial Neural Networks
- Support Vector Regression
- Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)
- Transformers
- Hybrid ARIMA-ANN

These models take time series data as input and output the predicted number. However, real-world epidemic datasets are complex, noisy, non-stationary, and non-linear, requiring preprocessing to obtain accurate results. Common transformations include log transformations and Fourier transformations, but these have limitations. Wavelet transformations, such as Discrete Wavelet Transform (DWT) and Maximum Overlap DWT (MODWT), are used to decompose data into simpler high and low-frequency components.

### Explaining EWNETs

#### MODWT Approach

To understand the Maximal Overlap Discrete Wavelet Transform (MODWT), it's helpful to contrast it with the more traditional Discrete Wavelet Transform (DWT).

**Discrete Wavelet Transform (DWT)**
- **Boundary Effects**: Discontinuities at the edges of the signal can distort the wavelet coefficients near the boundaries.
- **Lack of Shift Invariance**: Small shifts in the signal can result in significant changes in the wavelet coefficients.

**Maximal Overlap Discrete Wavelet Transform (MODWT)**
- **Circular Convolution**: Treats the signal as periodic, eliminating boundary effects.
- **Maximal Overlap**: Maximizes overlap between adjacent wavelet coefficients by extending the signal and applying the wavelet transform with a shift.
- **Iterative Decomposition**: Decomposes the signal into different frequency bands or scales with multiple levels of decomposition.
- **Update Formulas**: Combines approximation and detail coefficients after each level of decomposition to preserve important signal characteristics while minimizing boundary effects.

#### EWNET Model Architecture

The paper proposes a novel model, EWNETs, which utilizes MODWT signal decomposition as a preprocessing step. MODWT decomposes the data into J detail levels and 1 smooth level, where the sum of corresponding points in details and smooth yields back the original data point.

#### EWNET Model Working

- **Training**: MODWT is applied to the time series data, which is then inputted to an ensemble of autoregressive neural networks for prediction.
- **Testing**: The trained model is used to forecast future values by applying MODWT to the input data and using the neural networks for prediction.

---

## Experimentation Results of EWNETs on Datasets

The report includes detailed experimentation results of EWNETs on various datasets, demonstrating its effectiveness in short, medium, and long-term forecasts compared to other methods.

---

## Using Other Signal Decomposition Methods

In addition to MODWT, various other signal decomposition methods were experimented with:

### Seasonal and Trend Decomposition Using Loess (STL)

STL decomposes the signal into seasonal, trend, and residue components using locally estimated scatterplot smoothing (LOESS). It is robust to outliers, can model any seasonality, and can perform automatic parameter selection for smoothing.

### Singular Spectral Analysis (SSA)

SSA is used for additive decomposition and derives its name from the eigenvalues of the singular value decomposition of a covariance matrix. It is applicable to the study of classical time series, multivariate statistics, multivariate geometry, dynamical systems, and signal processing.

### Empirical Mode Decomposition (EMD)

EMD decomposes a signal into intrinsic mode functions (IMFs), which represent simple oscillatory modes. It is effective for non-linear and non-stationary time series analysis.

### Hilbert Huang Transform (HHT)

HHT combines EMD and Hilbert spectral analysis to provide a time-frequency representation of the signal. It is useful for analyzing non-linear and non-stationary data.

### Empirical Wavelet Transform (EWT)

EWT adaptively decomposes a signal based on its content, providing a flexible and data-driven approach to signal decomposition.

---

## Experimentation Results

The report includes comparative experimentation results of the various signal decomposition methods, apart from EWNET, on different datasets:

- **STL**
- **SSA**
- **EMD**
- **HHT**
- **EWT**

---

## Discussion and Conclusion

The report concludes that EWNETs, leveraging MODWT for signal decomposition, provides more accurate results for epidemic forecasting compared to other methods. The robustness of EWNETs in handling non-stationary and non-linear data makes it effective for real-world epidemic forecasting applications. The comparative analysis with other signal decomposition methods further highlights the potential of EWNETs in advancing epidemic forecasting techniques.

---

## Implementation

The implementation of EWNETs and other signal decomposition methods, along with the code and datasets used, can be found on our GitHub repository: [GitHub](https://github.com/AmeyChoudhary/DPCN_Baby_Sharks/tree/main/Amey).
