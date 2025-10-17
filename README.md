# Magic Telescope Gamma & Hadron Event Classifier

![Magic Telescope Logo](https://upload.wikimedia.org/wikipedia/commons/0/05/MAGIC_Telescope_-_La_Palma.JPG) <!-- Optional, replace with your logo -->

** Deployed App: 
https://magic-telescope.streamlit.app/

## Overview
The **Magic Telescope Gamma & Hadron Event Classifier** is a machine learning-based application designed to classify events detected by the MAGIC (Major Atmospheric Gamma Imaging Cherenkov) telescope into **gamma-ray events** and **hadron events**. This project helps astronomers and researchers analyze telescope data more efficiently by automating event classification.

The classifier leverages advanced machine learning algorithms to provide accurate predictions based on event features extracted from the telescope dataset.

---

## Features
- Classifies telescope events as **Gamma** or **Hadron**.
- Uses a trained ML model for high-accuracy predictions.
- Interactive web interface for easy input and visualization.
- Provides **probability scores** for classification confidence.
- Lightweight and easy to deploy using **Streamlit** or similar frameworks.

---

## Dataset
The model is trained on the **MAGIC Gamma Telescope dataset**, which includes features such as:
- `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`, etc.

The dataset is publicly available and can be found [here](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope).



