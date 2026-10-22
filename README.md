## LifespanFix: Semi-supervised Cortical Correction Across the Human Lifespan

**[DBC Lab, UNC-Chapel Hill](https://liwang.web.unc.edu)**

[Tong Wang](https://scholar.google.com/citations?user=EnnmFqIAAAAJ&hl=zh-CN), [Yue Sun](https://scholar.google.com/citations?user=QMa2E4gAAAAJ&hl=zh-CN), [Li Wang](https://scholar.google.com/citations?user=xjzMf0cAAAAJ&hl=zh-CN)

> **Note:** This repository is under active development. Questions and suggestions are welcome via [GitHub Issues](https://github.com/ittong/LifespanFix/issues).

**LifespanFix** is an end-to-end model for topological and anatomical correction of cortical surfaces. It operates on white matter (WM) segmentations, corrects defects iteratively while preserving anatomically correct regions, and was trained on 3,189 subjects from 9 public cohorts spanning the lifespan.
<p align="center">
  <img src="imgs/beforeafter.jpg" width="700">
</p>

LifespanFix is trained with a **semi-supervised framework**: a generation network produces template-constrained pseudo labels that supervise the correction network (LifespanFix). At inference, only LifespanFix is needed; no registration is required.
<p align="center">
  <img src="imgs/framework.jpg" width="600">
</p>

## Installation
Install [PyTorch](https://pytorch.org/get-started/locally/) (CUDA recommended), then:
```
git clone https://github.com/ittong/LifespanFix.git
cd LifespanFix
pip install -r requirements.txt
```

## Inference
Inputs are binary WM masks (0 = background, 1 = WM) in `.nii.gz` format. LifespanFix was trained at 0.8 mm isotropic resolution.

Download a [model checkpoint](https://drive.google.com/drive/folders/1ENl2m2Es15bvlHNB7hV6UKN5m9aSv28_?usp=share_link) into `model/`, then run:
```
python infer.py --checkpoint model/correction_net_4error_free_wm.pth --input_dir test_example --output_dir results --device cuda:0
```
Each input `<name>.nii.gz` is saved as `<name>_iter3_label_<checkpoint tag>.nii.gz`. Useful options: `--iters` (default 3; three iterations are sufficient for most defects), `--threshold` (default 0.5; higher values allow fewer modifications), `--save_all_iters`, and `--device cpu`.

The number in a checkpoint name is the percentage of defect-free subjects used in training. A smaller number gives more aggressive correction; a larger number preserves more of the original segmentation.
<p align="center">
  <img src="imgs/iteration.jpg" width="600">
</p>

## Genus-0 topology (optional)
LifespanFix corrects patch-wise and does not explicitly guarantee strict genus-0 topology for the whole cortical surface. LifespanFix outputs can be used directly for voxel-based analyses such as WM volume. If a downstream step requires genus 0 (e.g., spherical mapping for surface registration), run:
```
python genus0.py --input_dir results --output_dir results_genus0 --save_defects
```
This saves `<name>_genus0.nii.gz` (and `<name>_defect.nii.gz`, the changed voxels) and prints the topology before and after. WM components disconnected from the main body are removed unless `--keep_islands` is given. Build surfaces from the output with classic marching cubes (e.g., `skimage.measure.marching_cubes(..., method='lorensen')`).

`genus0.py` implements the fast-marching topology correction of Bazin and Pham [1, 2]. **Our implementation (`utils/topology.py`) is written with reference to the Python/Numba version released with CortexODE [3]: [`util/tca.py`](https://github.com/m-qiang/CortexODE/blob/main/util/tca.py) in https://github.com/m-qiang/CortexODE.** As in that code, residual handles and holes are resolved by filling. Instead of the look-up table used there, simple points are tested with the topological numbers of Bertrand and Malandain [4]. Because genus-0 enforcement only fills, it may not match the anatomy at residual handles, so we recommend running LifespanFix first. In the paper, genus-0 enforcement was evaluated with the topology-preserving level-set method [5].

## Training
```
python train.py --data_dir data_dir --save_dir checkpoint --device cuda:0
```
`data_dir` contains one folder per subject with `wm.nii.gz` (WM segmentation at 0.8 mm), `templates/*.nii.gz` (30 WM templates deformably registered to `wm.nii.gz`) and, optionally, `defect.nii.gz` (detected topological defects; patches overlapping them are skipped). Holes and handles are simulated from the WM segmentation on the fly (`utils/simulation.py`). Default arguments follow the paper (patch size 48, 1,000 patches per subject, 10% defect-free subjects, 5 epochs); see `python train.py -h`.

## References
[1] P.-L. Bazin and D. L. Pham, "Topology correction using fast marching methods and its application to brain segmentation," *MICCAI*, 2005. doi: [10.1007/11566489_60](https://doi.org/10.1007/11566489_60)

[2] P.-L. Bazin and D. L. Pham, "Topology correction of segmented medical images using a fast marching algorithm," *Comput. Methods Programs Biomed.*, vol. 88, no. 2, pp. 182–190, 2007. doi: [10.1016/j.cmpb.2007.08.006](https://doi.org/10.1016/j.cmpb.2007.08.006)

[3] Q. Ma, L. Li, E. C. Robinson, B. Kainz, D. Rueckert, and A. Alansary, "CortexODE: Learning cortical surface reconstruction by neural ODEs," *IEEE Trans. Med. Imaging*, vol. 42, no. 2, pp. 430–443, 2023. doi: [10.1109/TMI.2022.3206221](https://doi.org/10.1109/TMI.2022.3206221). Code: https://github.com/m-qiang/CortexODE

[4] G. Bertrand and G. Malandain, "A new characterization of three-dimensional simple points," *Pattern Recognit. Lett.*, vol. 15, no. 2, pp. 169–175, 1994. doi: [10.1016/0167-8655(94)90046-9](https://doi.org/10.1016/0167-8655(94)90046-9)

[5] X. Han, C. Xu, and J. L. Prince, "A topology preserving level set method for geometric deformable models," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 25, no. 6, pp. 755–768, 2003. doi: [10.1109/TPAMI.2003.1201824](https://doi.org/10.1109/TPAMI.2003.1201824)

## Citation
If you find LifespanFix useful, please cite our paper. The BibTeX entry will be added once the paper is published.

## License
This project is released under the [Apache License 2.0](LICENSE).
