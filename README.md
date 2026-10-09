<div align="center">

# OS-MASK

### Generating Fine-Grained Semantic Segmentation Labels for Remote Sensing Imagery from OpenStreetMap Priors

**Research implementation · OpenStreetMap-guided annotation · Remote sensing semantic segmentation**

[![Project](https://img.shields.io/badge/Project-OS--MASK-275D8C?style=flat-square)](https://github.com/Joinzm/OS-MASK)
[![Framework](https://img.shields.io/badge/Backbone-Meta%20SAM%203-555555?style=flat-square)](https://github.com/facebookresearch/sam3)
[![Code Release](https://img.shields.io/badge/Code%20Release-In%20Progress-orange?style=flat-square)](#code-availability)

</div>

---

## 📖 Overview

**OS-MASK** is a research framework for generating **fine-grained semantic segmentation labels** for remote sensing imagery using **OpenStreetMap (OSM) priors**. The project explores how geospatial prior information can support pixel-level annotation of complex remote sensing scenes.

This repository hosts the evolving implementation of OS-MASK. **The current release is partial**; the complete source code, configuration details, and reproducible usage instructions will be provided in a future update.

## ⚙️ Environment

OS-MASK relies on the **official Meta Segment Anything Model 3 (SAM 3)** implementation and its software environment.

1. Set up SAM 3 by following the installation instructions in the official repository: **[facebookresearch/sam3](https://github.com/facebookresearch/sam3)**.
2. Obtain the required SAM 3 model checkpoint(s) through the access and download procedures described by Meta.
3. Use the configured SAM 3 environment when running OS-MASK components.

> [!IMPORTANT]
> SAM 3 is an external dependency. Please follow Meta's official documentation for up-to-date installation requirements, model access, and usage terms. Model weights are **not distributed** through this repository.

## 📦 Code Availability

> [!NOTE]
> **Partial release.** This repository currently contains selected OS-MASK implementation files. The **complete implementation**, together with configuration and execution instructions, **will be made publicly available in a future update**.

Please note that the currently released files should **not** be interpreted as a complete, independently reproducible pipeline.

## 📝 Citation

If you use this repository in your research, please acknowledge **OS-MASK**. A formal paper citation and BibTeX entry will be added when the publication details are available.

## 🙏 Acknowledgments

This project builds on **[Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)** developed by Meta and uses **[OpenStreetMap](https://www.openstreetmap.org/)** as a source of geospatial priors. We acknowledge the contributions of the respective research and open-data communities.

---

<div align="center">

**OS-MASK** · Research code repository

</div>
