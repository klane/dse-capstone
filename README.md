# Deep Learning Based Traffic Forecasting

From the onset of the COVID-19 pandemic, a dramatic change in traffic patterns was observed across the country due to travel and other restrictions imposed by government agencies and health experts. The causes for these abrupt changes can be at least partially attributed to the severity of the pandemic, the widespread increase in remote work and online learning, business closures, etc. Traffic forecasting models should adapt to the current environment in order to learn long-term changes in traffic patterns.

This work is part of a capstone project for the [Master of Advanced Study (MAS)](https://jacobsschool.ucsd.edu/mas) program in [Data Science & Engineering (DSE)](https://jacobsschool.ucsd.edu/mas/dse) at the University of California, San Diego. We apply various deep learning methods to the problem of traffic forecasting, including a graph convolutional recurrent neural network that captures both the inherent spatial and temporal complexities present in traffic data. The algorithms implemented here are integrated into a larger [deep learning library for time series modeling](https://github.com/Rose-STL-Lab/torchTS) being developed for the open-source community.

**Team**:

- Aparna Gupta (aparna.gupta123@gmail.com)
- Kevin Lane (lane.kevin.a@gmail.com)
- Raul Martinez (gio.mtz3@gmail.com)
- Daniel Roten (droten76@gmail.com)
- Akash Shah (akashshah59@gmail.com)

**Advisor**:

- Dr. Rose Yu
- Dr. Ilkay Altintas

## Table of Contents

- [Dashboard](https://github.com/klane/dse-capstone/tree/main/dashboard)
- [Data](https://github.com/klane/dse-capstone/tree/main/data)
  - [Caltrans Performance Measurement System (PeMS)](https://pems.dot.ca.gov/)
  - [COVID-19](https://github.com/CSSEGISandData/COVID-19)
- [Database](https://github.com/klane/dse-capstone/tree/main/db)
- [Documents](https://github.com/klane/dse-capstone/tree/main/docs)
  - [Presentation](https://github.com/klane/dse-capstone/tree/main/docs/presentation)
  - [Report](https://github.com/klane/dse-capstone/tree/main/docs/report)
- [Exploratory Data Analysis (EDA)](https://github.com/klane/dse-capstone/tree/main/eda)
  - Traffic
  - COVID-19
- [Models](https://github.com/klane/dse-capstone/tree/main/models)
  - [Autoregressive Neural Network](https://github.com/klane/dse-capstone/tree/main/models/autoregressive)
  - [Diffusion Convolutional Recurrent Neural Network (DCRNN)](https://github.com/klane/dse-capstone/tree/main/models/dcrnn)
  - [Deep Seq2Seq](https://github.com/klane/dse-capstone/tree/main/models/seq2seq)
  - [Physics-Informed Neural Network](https://github.com/klane/dse-capstone/tree/main/models/differential_equations)
