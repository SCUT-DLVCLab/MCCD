# <div align=center>MCCD: A Multi-Attribute Chinese Calligraphy Character Dataset Annotated with Script Styles, Dynasties, and Calligraphers</div>
<div align="center">

[![SCUT DLVC Lab](https://img.shields.io/badge/SCUT-DLVC_Lab-327FE6?logo=Academia&logoColor=white&style=plastic)](http://dlvc-lab.net/lianwen/)     ![ICDAR2025](https://img.shields.io/badge/2025-ICDAR?style=plastic&label=ICDAR&labelColor=grey&color=%2300BFFF) [![arxiv preprint](https://img.shields.io/badge/2507.06948-arxiv?style=plastic&label=arxiv&labelColor=grey&color=%23DC143C&link=https%3A%2F%2Farxiv.org%2Fabs%2F2507.06948)](https://arxiv.org/abs/2507.06948) [![Code](https://img.shields.io/badge/MCCD-Code?style=plastic&label=Code&labelColor=grey&color=%23DB7093)](https://github.com/SCUT-DLVCLab/MCCD)

</div>

![overview](https://github.com/SCUT-DLVCLab/MCCD/blob/main/picture/overview.png)

## ✨ introduction
- We introduce **Multi-Attribute Chinese Calligraphy Character Dataset (MCCD)**, an isolated Chinese character dataset with rich annotations including character, script style, dynasty, and calligrapher.
- **Extensive Multi-Attribute Collection:** MCCD dataset presents a meticulously curated collection of nearly 330,000 calligraphic character images, ensuring a comprehensive diversity of annotation categories for all characters and their attributes (style, dynasty, and calligrapher).
- **Multi-Attribute Subset Construction:** MCCD contains labels for 7,765 categories of characters, in addition to which three additional subsets are extracted from the dataset according to the attribute annotations for each character, including 10 styles of calligraphy, 15 major historical dynasties and 142 famous calligraphers, with the aim of optimizing task-specific utilization of the attribute information.
- **Benchmark Establishment:** We established benchmark performance metrics for single-task recognition (character and each attribute independently) and multi-task recognition (character combined with other attributes simultaneously) experiments using MCCD and all its subsets.

## 🔗 Download
✅ **Status:** Released

✅ **Dataset link:** [Baiduyun:8x7d]( https://pan.baidu.com/s/1qkM_1gizNRBONEY47nw50g?pwd=8x7d) / [OneDrive](https://1drv.ms/u/c/d3b0ec8fe3491f94/EXtpGohdU55Io8CUtbfjShkBocjgdcDl4y0j-_gJRN-LNQ?e=ShhRhh)

✅ **Data format:** PNG / lmdb

## 🛠️ Usage
- Clone this repo:
```bash
git clone https://github.com/SCUT-DLVCLab/MCCD.git
```
- The **data_loader** folder contains **read files** for single-attribute labeled lmdb as well as 2-attribute labeled and 4-attribute labeled lmdb data.

|Read File                 |Corresponding Dataset                                     
|--------------------------|---------------------------------------------|
|lmdb_dataset.py           |`MCCD-Character/ Style/ Dynasty/Calligrapher`            
|2task_MTL_lmdb_dataset.py |`dual_task`                                  |
|4task_MTL_lmdb_dataset.py |`four_task`|

❗**Note:**
- The MCCD dataset can only be used for non-commercial research purposes. For scholar or organization who wants to use the MCCD dataset, please first fill in this [Application Form](https://github.com/SCUT-DLVCLab/MCCD/blob/main/application-form/Application-Form-for-Using-MCCD.docx)  and sign the [Legal Commitment](https://github.com/SCUT-DLVCLab/MCCD/blob/main/application-form/Legal-Commitment.docx) and email them to us (eelwjin@scut.edu.cn, cc: yixin_zhao01@126.com). When submitting the application form to us, please list or attached 1-2 of your publications in the recent 6 years to indicate that you (or your team) do research in the related research fields of OCR, handwriting verification, handwriting analysis and recognition, document image processing, and so on.
- We will give you the decompression password after your application has been received and approved.
- All users must follow all use conditions; otherwise, the authorization will be revoked.

## 📧 Contact
☺️ If you have any questions, please feel free to contact [Yixin Zhao](https://github.com/Christinazyx) at yixin_zhao01@126.com. 

## 🔐License
MCCD should be used and distributed under [Creative Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## ©️ Copyright
- This repository can only be used for non-commercial research purposes.
- Copyright 2025, [Deep Learning and Vision Computing Lab (DLVC-Lab)](http://www.dlvc-lab.net/), South China University of Technology.

